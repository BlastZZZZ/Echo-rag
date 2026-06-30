import unittest
from unittest.mock import patch

import numpy as np

from role_channel_graph_retrieval import (
    best_channel_first_fusion,
    fuse_channel_rankings,
    pointwise_role_score_fusion,
    reciprocal_rank_fusion,
    role_channel_graph_retrieve,
    role_passage_seed_scores,
    round_robin_fusion,
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
        self.passage_node_keys = ["p0", "p1", "p2", "p3"]
        self.fact_embedding_store = DummyEmbeddingStore(
            {
                "f0": {"content": "('a', 'rel', 'b')"},
                "f1": {"content": "('c', 'rel', 'd')"},
                "f2": {"content": "('e', 'rel', 'f')"},
            }
        )
        self.chunk_embedding_store = DummyEmbeddingStore(
            {
                "p0": {"content": "doc0"},
                "p1": {"content": "doc1"},
                "p2": {"content": "doc2"},
                "p3": {"content": "doc3"},
            }
        )
        self.encoded_queries = []
        self.fact_score_queries = []
        self.graph_calls = []

    def prepare_retrieval_objects(self):
        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries):
        self.encoded_queries.extend(queries)

    def get_fact_scores(self, query):
        self.fact_score_queries.append(query)
        if "first" in query:
            return np.asarray([0.9, 0.2, 0.1])
        return np.asarray([0.1, 0.8, 0.7])

    def dense_passage_retrieval(self, query):
        if "first" in query:
            return np.asarray([0, 2, 1]), np.asarray([0.9, 0.4, 0.1])
        return np.asarray([1, 3, 0]), np.asarray([0.8, 0.3, 0.2])

    def rerank_facts(self, query, scores):
        if "first" in query:
            return [0], [("a", "rel", "b")], {}
        return [1], [("c", "rel", "d")], {}

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
                "query_fact_scores": np.asarray(query_fact_scores, dtype=float),
                "top_k_fact_indices": list(top_k_fact_indices),
            }
        )
        if "first" in query:
            return np.asarray([0, 2, 1]), np.asarray([0.7, 0.2, 0.1])
        return np.asarray([1, 3, 0]), np.asarray([0.6, 0.3, 0.2])


class RoleChannelGraphRetrievalTest(unittest.TestCase):
    def test_rrf_is_rank_based_and_deduplicated(self) -> None:
        docs, scores = reciprocal_rank_fusion([[1, 2, 2], [3, 1]], rrf_k=0.0, top_k=3)
        self.assertEqual(docs, [1, 3, 2])
        self.assertAlmostEqual(scores[0], 1.0 + 0.5)
        self.assertAlmostEqual(scores[1], 1.0)
        self.assertAlmostEqual(scores[2], 0.5)

    def test_round_robin_fusion_interleaves_channels(self) -> None:
        docs, _ = round_robin_fusion([[1, 2], [3, 1, 4]], top_k=4)
        self.assertEqual(docs, [1, 3, 2, 4])

    def test_best_channel_first_uses_channel_top_score(self) -> None:
        docs, _ = best_channel_first_fusion(
            [[1, 2], [3, 1]],
            [[0.5, 0.2], [0.9, 0.1]],
            top_k=3,
        )
        self.assertEqual(docs, [3, 1, 2])

    def test_pointwise_role_score_sums_normalized_channel_scores(self) -> None:
        docs, scores = pointwise_role_score_fusion(
            [[1, 2], [2, 3]],
            [[0.9, 0.1], [0.8, 0.4]],
            top_k=3,
        )
        self.assertEqual(docs, [1, 2, 3])
        self.assertEqual(scores[:2], [1.0, 1.0])

    def test_role_passage_seed_scores_uses_top_k(self) -> None:
        docs, scores = role_passage_seed_scores(
            question="q",
            role={"role_id": "r0", "description": "first"},
            sorted_doc_ids=np.asarray([9, 8, 7]),
            sorted_doc_scores=np.asarray([0.9, 0.8, 0.7]),
            role_passage_top_k=2,
        )
        self.assertEqual(docs, [9, 8])
        self.assertEqual(scores, {9: 0.9, 8: 0.8})

    def test_role_channel_retrieval_runs_one_graph_channel_per_role(self) -> None:
        def fake_graph_search(**kwargs):
            if "first" in kwargs["query"]:
                return np.asarray([0, 2, 1]), np.asarray([0.7, 0.2, 0.1])
            return np.asarray([1, 3, 0]), np.asarray([0.6, 0.3, 0.2])

        system = DummySystem()
        with patch("role_channel_graph_retrieval.graph_search_with_role_entries", side_effect=fake_graph_search) as mock_search:
            results = role_channel_graph_retrieve(
                system=system,
                queries=["question"],
                query_indices=[5],
                roles_by_query={
                    5: [
                        {"role_id": "r0", "description": "first"},
                        {"role_id": "r1", "description": "second"},
                    ]
                },
                num_to_retrieve=3,
                role_fact_top_k=2,
                role_passage_top_k=2,
                channel_output_top_k=3,
                rrf_k=60.0,
                channel_backend="seeded_entry",
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(mock_search.call_count, 2)
        self.assertEqual(results[0].solution.docs, ["doc0", "doc1", "doc2"])
        self.assertEqual(results[0].trace.channel_count, 2)
        self.assertEqual([channel.role_id for channel in results[0].trace.channels], ["r0", "r1"])

    def test_role_channel_retrieval_can_use_hipporag_graph_backend(self) -> None:
        def fake_graph_search(**kwargs):
            if "first" in kwargs["query"]:
                return np.asarray([0, 2, 1]), np.asarray([0.7, 0.2, 0.1])
            return np.asarray([1, 3, 0]), np.asarray([0.6, 0.3, 0.2])

        system = DummySystem()
        with patch("role_channel_graph_retrieval.graph_search_with_role_entries", side_effect=fake_graph_search) as mock_search:
            results = role_channel_graph_retrieve(
                system=system,
                queries=["question"],
                query_indices=[5],
                roles_by_query={
                    5: [
                        {"role_id": "r0", "description": "first"},
                        {"role_id": "r1", "description": "second"},
                    ]
                },
                num_to_retrieve=3,
                channel_output_top_k=3,
                rrf_k=60.0,
                channel_backend="hipporag_graph",
            )

        self.assertEqual(mock_search.call_count, 2)
        self.assertEqual(results[0].solution.docs, ["doc0", "doc1", "doc2"])
        self.assertEqual(results[0].trace.channels[0].selected_fact_indices, [0])
        self.assertEqual(results[0].trace.channels[1].selected_fact_indices, [1])
        self.assertEqual(results[0].trace.channels[0].selected_passage_indices, [0, 2, 1])
        self.assertEqual(results[0].trace.channels[1].selected_passage_indices, [1, 3, 0])
        self.assertEqual(mock_search.call_args_list[0].kwargs["passage_seed_scores"], {0: 0.9, 2: 0.4, 1: 0.1})
        self.assertEqual(mock_search.call_args_list[1].kwargs["passage_seed_scores"], {1: 0.8, 3: 0.3, 0: 0.2})

    def test_shared_question_by_demand_reuses_fact_scores_per_role(self) -> None:
        graph_calls = []

        def fake_graph_search(**kwargs):
            graph_calls.append(kwargs)
            if "first" in kwargs["query"]:
                return np.asarray([0, 2, 1]), np.asarray([0.7, 0.2, 0.1])
            return np.asarray([1, 3, 0]), np.asarray([0.6, 0.3, 0.2])

        system = DummySystem()
        with patch("role_channel_graph_retrieval.graph_search_with_role_entries", side_effect=fake_graph_search):
            results = role_channel_graph_retrieve(
                system=system,
                queries=["question"],
                query_indices=[5],
                roles_by_query={
                    5: [
                        {"role_id": "r0", "description": "first"},
                        {"role_id": "r1", "description": "second"},
                    ]
                },
                num_to_retrieve=3,
                channel_output_top_k=3,
                rrf_k=60.0,
                channel_backend="hipporag_graph",
                fact_filter_strategy="shared_question_by_demand",
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(system.fact_score_queries), 2)
        self.assertEqual(len(graph_calls), 2)
        self.assertEqual([channel.fact_score_source for channel in results[0].trace.channels], ["precomputed", "precomputed"])
        self.assertTrue(all(channel.fact_score_reuse_enabled for channel in results[0].trace.channels))
        self.assertEqual(results[0].trace.channels[0].selected_passage_indices, [0, 2, 1])
        self.assertEqual(results[0].trace.channels[1].selected_passage_indices, [1, 3, 0])
        np.testing.assert_allclose(graph_calls[0]["query_fact_scores"], np.asarray([0.9, 0.2, 0.1]))
        np.testing.assert_allclose(graph_calls[1]["query_fact_scores"], np.asarray([0.1, 0.8, 0.7]))

    def test_role_channel_retrieval_separates_retrieval_and_provenance_text(self) -> None:
        system = DummySystem()
        results = role_channel_graph_retrieve(
            system=system,
            queries=["question"],
            query_indices=[5],
            roles_by_query={
                5: [
                    {
                        "role_id": "r0",
                        "role_type": "bridge",
                        "description": "legacy description",
                        "retrieval_text": "first retrieval question",
                        "provenance_text": "identify bridge evidence",
                    }
                ]
            },
            num_to_retrieve=2,
            channel_output_top_k=2,
            channel_backend="hipporag_graph",
            role_passage_top_k=0,
        )
        channel = results[0].trace.channels[0]
        self.assertIn("first retrieval question", channel.retrieval_query)
        self.assertNotIn("identify bridge evidence", channel.retrieval_query)
        self.assertEqual(channel.role_description, "identify bridge evidence")
        self.assertEqual(channel.retrieval_text, "first retrieval question")
        self.assertEqual(channel.provenance_text, "identify bridge evidence")
        self.assertEqual(channel.support_function, "bridge")

    def test_role_channel_retrieval_does_not_fallback_without_roles(self) -> None:
        system = DummySystem()
        results = role_channel_graph_retrieve(
            system=system,
            queries=["question"],
            roles_by_query={},
            num_to_retrieve=3,
        )
        self.assertEqual(results[0].solution.docs, [])
        self.assertEqual(results[0].trace.empty_reason, "no_valid_roles")


if __name__ == "__main__":
    unittest.main()
