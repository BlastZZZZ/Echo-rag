import unittest

import numpy as np

from role_aware_graph_entry import (
    build_role_fact_rankings,
    fact_indices_and_tuples,
    merge_role_fact_seeds,
    role_aware_graph_entry_retrieve,
    role_graph_entry_query,
    top_fact_indices_for_scores,
    valid_roles,
)


class DummyEmbeddingStore:
    def __init__(self, rows):
        self.rows = rows

    def get_rows(self, keys):
        return {key: self.rows[key] for key in keys}

    def get_row(self, key):
        return self.rows[key]


class DummyConfig:
    linking_top_k = 2
    passage_node_weight = 0.05


class DummySystem:
    def __init__(self):
        self.ready_to_retrieve = False
        self.global_config = DummyConfig()
        self.fact_node_keys = ["f0", "f1", "f2"]
        self.passage_node_keys = ["p0", "p1"]
        self.fact_embedding_store = DummyEmbeddingStore(
            {
                "f0": {"content": "('a', 'rel', 'b')"},
                "f1": {"content": "('c', 'rel', 'd')"},
                "f2": {"content": "('e', 'rel', 'f')"},
            }
        )
        self.chunk_embedding_store = DummyEmbeddingStore(
            {
                "p0": {"content": "doc zero"},
                "p1": {"content": "doc one"},
            }
        )
        self.encoded_queries = []
        self.graph_calls = []

    def prepare_retrieval_objects(self):
        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries):
        self.encoded_queries.extend(queries)

    def get_fact_scores(self, query):
        if "first" in query:
            return np.asarray([0.9, 0.1, 0.2])
        return np.asarray([0.1, 0.8, 0.7])

    def graph_search_with_fact_entities(
        self,
        query,
        link_top_k,
        query_fact_scores,
        top_k_facts,
        top_k_fact_indices,
        passage_node_weight,
    ):
        self.graph_calls.append(
            {
                "query": query,
                "link_top_k": link_top_k,
                "top_k_facts": top_k_facts,
                "top_k_fact_indices": top_k_fact_indices,
                "passage_node_weight": passage_node_weight,
                "nonzero_scores": np.nonzero(query_fact_scores)[0].astype(int).tolist(),
            }
        )
        return np.asarray([1, 0]), np.asarray([0.7, 0.3])


class RoleAwareGraphEntryTest(unittest.TestCase):
    def test_role_query_uses_question_and_role_only(self) -> None:
        query = role_graph_entry_query(
            question="Who wrote X?",
            role={"description": "identify the writer"},
        )
        self.assertEqual(query, "Question: Who wrote X?\nEvidence role: identify the writer")

    def test_role_query_prefers_retrieval_text(self) -> None:
        query = role_graph_entry_query(
            question="Who wrote X?",
            role={
                "description": "bridge support for writer evidence",
                "retrieval_text": "Who wrote X?",
            },
        )
        self.assertEqual(query, "Question: Who wrote X?\nEvidence role: Who wrote X?")

    def test_valid_roles_drops_empty_and_duplicate_ids(self) -> None:
        roles = valid_roles(
            [
                {"role_id": "r0", "description": " first role "},
                {"role_id": "r0", "description": "duplicate"},
                {"role_id": "r1", "description": "   "},
                {"description": "second role"},
            ]
        )
        self.assertEqual([role["role_id"] for role in roles], ["r0", "r3"])

    def test_valid_roles_keeps_retrieval_and_provenance_texts_separate(self) -> None:
        roles = valid_roles(
            [
                {
                    "role_id": "r0",
                    "role_type": "bridge",
                    "retrieval_text": "Who wrote X?",
                    "provenance_text": "identify the writer as bridge evidence",
                }
            ]
        )
        self.assertEqual(roles[0]["description"], "identify the writer as bridge evidence")
        self.assertEqual(roles[0]["retrieval_text"], "Who wrote X?")
        self.assertEqual(roles[0]["provenance_text"], "identify the writer as bridge evidence")
        self.assertEqual(roles[0]["support_function"], "bridge")

    def test_top_fact_indices_for_scores_descending(self) -> None:
        self.assertEqual(top_fact_indices_for_scores(np.asarray([0.2, 0.9, 0.3]), 2), [1, 2])
        self.assertEqual(top_fact_indices_for_scores(np.asarray([0.2, 0.9]), 0), [])

    def test_merge_role_fact_seeds_is_role_balanced(self) -> None:
        rankings = build_role_fact_rankings(
            question="q",
            roles=[
                {"role_id": "r0", "description": "first"},
                {"role_id": "r1", "description": "second"},
            ],
            role_fact_scores={
                "r0": np.asarray([0.9, 0.8, 0.1]),
                "r1": np.asarray([0.2, 0.1, 0.7]),
            },
            role_fact_top_k=2,
        )
        seeds = merge_role_fact_seeds(rankings=rankings, num_facts=3, max_fact_seeds=3)
        self.assertEqual(seeds.fact_indices, [0, 2, 1])
        self.assertEqual(seeds.selected_by_role["r0"], [0, 1])
        self.assertEqual(seeds.selected_by_role["r1"], [2])
        self.assertAlmostEqual(float(seeds.merged_fact_scores[0]), 0.9)
        self.assertAlmostEqual(float(seeds.merged_fact_scores[2]), 0.7)

    def test_fact_indices_and_tuples_filters_unparseable_rows(self) -> None:
        class Store:
            def get_rows(self, keys):
                return {
                    "f0": {"content": "('s', 'p', 'o')"},
                    "f1": {"content": "bad"},
                }

        class System:
            fact_node_keys = ["f0", "f1"]
            fact_embedding_store = Store()

        indices, facts = fact_indices_and_tuples(System(), [0, 1])
        self.assertEqual(indices, [0])
        self.assertEqual(facts, [("s", "p", "o")])

    def test_role_aware_retrieve_runs_one_graph_search_per_question(self) -> None:
        system = DummySystem()
        results = role_aware_graph_entry_retrieve(
            system=system,
            queries=["question"],
            query_indices=[7],
            roles_by_query={
                7: [
                    {"role_id": "r0", "description": "first"},
                    {"role_id": "r1", "description": "second"},
                ]
            },
            num_to_retrieve=2,
            role_fact_top_k=2,
            max_fact_seeds=2,
            include_role_passage_entries=False,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].solution.docs, ["doc one", "doc zero"])
        self.assertEqual(len(system.graph_calls), 1)
        self.assertEqual(system.graph_calls[0]["top_k_fact_indices"], [0, 1])
        self.assertEqual(results[0].trace.role_count, 2)
        self.assertEqual(results[0].trace.graph_search_count, 1)

    def test_total_channel_budget_preserves_role_seed_budget_in_one_graph_search(self) -> None:
        system = DummySystem()
        results = role_aware_graph_entry_retrieve(
            system=system,
            queries=["question"],
            query_indices=[7],
            roles_by_query={
                7: [
                    {"role_id": "r0", "description": "first"},
                    {"role_id": "r1", "description": "second"},
                ]
            },
            num_to_retrieve=2,
            role_fact_top_k=2,
            max_fact_seeds=1,
            include_role_passage_entries=False,
            max_fact_seeds_mode="total_channel_budget",
            link_top_k_mode="total_channel_budget",
        )

        self.assertEqual(len(system.graph_calls), 1)
        self.assertEqual(system.graph_calls[0]["link_top_k"], 4)
        self.assertEqual(system.graph_calls[0]["top_k_fact_indices"], [0, 1, 2])
        self.assertEqual(results[0].trace.fact_seed_budget, 4)
        self.assertEqual(results[0].trace.link_top_k_effective, 4)
        self.assertEqual(results[0].trace.seed_union_mode, "role_balanced_round_robin")

    def test_role_aware_retrieve_does_not_fallback_without_roles(self) -> None:
        system = DummySystem()
        results = role_aware_graph_entry_retrieve(
            system=system,
            queries=["question"],
            roles_by_query={},
            num_to_retrieve=2,
            include_role_passage_entries=False,
        )
        self.assertEqual(results[0].solution.docs, [])
        self.assertEqual(results[0].trace.empty_reason, "no_valid_roles")
        self.assertEqual(system.graph_calls, [])


if __name__ == "__main__":
    unittest.main()
