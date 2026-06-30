import unittest

from build_rolewise_retrieval_tasks import build_rolewise_retrieval_tasks, role_retrieval_query
from evaluate_rolewise_rcr import evaluate_rolewise_rcr
from materialize_rolewise_rank_cache import materialize_rolewise_rank_cache
from role_compatibility_scorer import passage_sentences, passage_text, role_compatibility_prompt
from rolewise_rcr import (
    build_rolewise_candidate_pool,
    build_rolewise_role_scores,
    normalize_rank_scores,
    parse_role_compatibility_cache,
    parse_rolewise_rank_cache,
    select_pointwise_role_merge_topk,
)


class RolewiseRcrTest(unittest.TestCase):
    def test_normalize_rank_scores_uses_numeric_scores_when_available(self) -> None:
        scores = normalize_rank_scores([10, 11, 12], [0.2, 0.6, 1.0])
        self.assertEqual(scores[10], 0.0)
        self.assertAlmostEqual(scores[11], 0.5)
        self.assertEqual(scores[12], 1.0)

    def test_normalize_rank_scores_falls_back_to_rank_prior(self) -> None:
        scores = normalize_rank_scores([10, 11, 12])
        self.assertEqual(scores[10], 1.0)
        self.assertEqual(scores[11], 0.5)
        self.assertEqual(scores[12], 0.0)

    def test_build_rolewise_candidate_pool_is_role_only_when_base_topk_is_zero(self) -> None:
        pool = build_rolewise_candidate_pool(
            roles=[{"role_id": "r0"}, {"role_id": "r1"}],
            role_rankings={
                "r0": {"doc_indices": [4, 1]},
                "r1": {"doc_indices": [5, 4]},
            },
            role_top_k=2,
            max_candidates=5,
        )
        self.assertEqual(pool, [4, 1, 5])

    def test_build_rolewise_role_scores_uses_role_specific_rankings(self) -> None:
        role_scores = build_rolewise_role_scores(
            candidate_docs=[0, 1, 2],
            roles=[{"role_id": "r0"}, {"role_id": "r1"}],
            role_rankings={
                "r0": {"doc_indices": [1, 0], "doc_scores": [0.8, 0.2]},
                "r1": {"doc_indices": [2]},
            },
        )
        self.assertEqual(role_scores[1]["r0"], 1.0)
        self.assertEqual(role_scores[0]["r0"], 0.0)
        self.assertEqual(role_scores[2]["r1"], 1.0)
        self.assertEqual(role_scores[1]["r1"], 0.0)

    def test_build_rolewise_role_scores_can_use_compatibility_cache(self) -> None:
        role_scores = build_rolewise_role_scores(
            candidate_docs=[0, 1, 2],
            roles=[{"role_id": "r0"}, {"role_id": "r1"}],
            role_rankings={
                "r0": {"doc_indices": [0]},
                "r1": {"doc_indices": [1]},
            },
            role_compatibility_scores={
                "r0": {2: 0.9},
                "r1": {0: 0.8},
            },
        )
        self.assertEqual(role_scores[2]["r0"], 0.9)
        self.assertEqual(role_scores[0]["r1"], 0.8)
        self.assertEqual(role_scores[0]["r0"], 0.0)

    def test_pointwise_role_merge_is_not_marginal_coverage(self) -> None:
        selected = select_pointwise_role_merge_topk(
            candidate_passages=[0, 1, 2],
            base_scores={0: 0.0, 1: 0.0, 2: 0.0},
            role_scores={
                0: {"r0": 1.0, "r1": 1.0},
                1: {"r0": 1.0, "r1": 0.0},
                2: {"r0": 0.0, "r1": 1.0},
            },
            role_ids=["r0", "r1"],
            k=2,
        )
        self.assertEqual(selected, [0, 1])

    def test_parse_rolewise_rank_cache_accepts_row_list(self) -> None:
        parsed = parse_rolewise_rank_cache(
            {
                "rankings": [
                    {"query_index": 0, "role_id": "r0", "doc_indices": [1]},
                    {"query_index": 0, "role_id": "r1", "doc_indices": [2]},
                ]
            }
        )
        self.assertEqual(sorted(parsed[0]), ["r0", "r1"])

    def test_parse_role_compatibility_cache_accepts_canonical_format(self) -> None:
        parsed = parse_role_compatibility_cache(
            {"scores_by_query": {"0": {"r0": {"11": 0.75}}}}
        )
        self.assertEqual(parsed[0]["r0"][11], 0.75)

    def test_role_compatibility_prompt_and_passage_text_are_question_only(self) -> None:
        prompt = role_compatibility_prompt(
            question="Who composed Film A?",
            role={"role_type": "bridge", "description": "Identify the composer."},
        )
        self.assertIn("Question: Who composed Film A?", prompt)
        self.assertIn("Evidence role (bridge): Identify the composer.", prompt)
        text = passage_text({"title": "Title", "text": "Body"}, max_chars=100)
        self.assertEqual(text, "Title\nBody")
        sentences = passage_sentences(
            {"title": "Title", "text": "First sentence. Second sentence."},
            max_sentences=1,
            max_sentence_chars=100,
        )
        self.assertEqual(sentences, ["Title\nFirst sentence."])

    def test_build_rolewise_retrieval_tasks_uses_question_only_inputs(self) -> None:
        payload = build_rolewise_retrieval_tasks(
            rows_by_qid={0: {"question": "Who composed Film A?"}},
            roles_by_query={0: [{"role_id": "r0", "role_type": "bridge", "description": "identify the composer"}]},
            query_mode="question_plus_role",
        )
        self.assertEqual(payload["num_tasks"], 1)
        self.assertEqual(payload["tasks"][0]["query_index"], 0)
        self.assertIn("Evidence role: identify the composer", payload["tasks"][0]["retrieval_query"])
        self.assertEqual(
            role_retrieval_query(
                question="q",
                role={"description": "find bridge"},
                mode="role_only",
            ),
            "find bridge",
        )

    def test_materialize_rolewise_rank_cache_maps_flat_outputs(self) -> None:
        payload = materialize_rolewise_rank_cache(
            task_payload={
                "query_mode": "question_plus_role",
                "tasks": [{"task_index": 0, "query_index": 7, "role_id": "r0", "retrieval_query": "role"}],
            },
            query_solutions=[{"doc_indices": [3, 4], "doc_scores": [0.9, 0.1]}],
            max_docs=1,
        )
        ranking = payload["role_rankings_by_query"]["7"]["r0"]
        self.assertEqual(ranking["doc_indices"], [3])
        self.assertEqual(ranking["base_scores"]["3"], 0.9)

    def test_evaluate_rolewise_rcr_reports_coverage_gain(self) -> None:
        corpus = [
            {"title": "Head", "text": "generic"},
            {"title": "Bridge", "text": "composer"},
            {"title": "Gold", "text": "birthplace"},
        ]
        payload = evaluate_rolewise_rcr(
            dataset="unit",
            corpus_records=corpus,
            rows_by_qid={0: {"question": "q", "gold_answers": ["a"], "gold_doc_indices": [1, 2]}},
            base_rank_cache={0: {"doc_indices": [0, 1], "base_scores": {0: 1.0, 1: 0.5}}},
            roles_by_query={0: [{"role_id": "r0", "description": "composer"}, {"role_id": "r1", "description": "birthplace"}]},
            rolewise_rank_cache={
                0: {
                    "r0": {"doc_indices": [1], "doc_scores": [1.0]},
                    "r1": {"doc_indices": [2], "doc_scores": [1.0]},
                }
            },
            role_compatibility_cache=None,
            role_top_k=1,
            max_candidates=3,
            reader_top_k=2,
            lambda_values=[0.0],
        )
        summary = payload["results"][1]["summary"]
        self.assertEqual(payload["variants"]["hipporag_v2"][0]["reader_doc_indices_topk"], [0, 1])
        self.assertEqual(payload["variants"]["rolewise_coverage_lambda_0_0"][0]["reader_doc_indices_topk"], [1, 2])
        self.assertGreater(summary["selected_r5"], summary["baseline_r5"])

    def test_evaluate_rolewise_rcr_can_use_compatibility_cache(self) -> None:
        corpus = [
            {"title": "Base", "text": "generic"},
            {"title": "Wrong", "text": "distractor"},
            {"title": "Gold", "text": "birthplace"},
        ]
        payload = evaluate_rolewise_rcr(
            dataset="unit",
            corpus_records=corpus,
            rows_by_qid={0: {"question": "q", "gold_answers": ["a"], "gold_doc_indices": [2]}},
            base_rank_cache={0: {"doc_indices": [0, 1], "base_scores": {0: 1.0, 1: 0.5}}},
            roles_by_query={0: [{"role_id": "r0", "description": "birthplace"}]},
            rolewise_rank_cache={0: {"r0": {"doc_indices": [1, 2], "doc_scores": [1.0, 0.1]}}},
            role_compatibility_cache={0: {"r0": {1: 0.0, 2: 1.0}}},
            role_top_k=2,
            max_candidates=3,
            reader_top_k=1,
            lambda_values=[0.0],
        )
        self.assertEqual(payload["config"]["compatibility_source"], "cache")
        self.assertEqual(payload["variants"]["rolewise_coverage_lambda_0_0"][0]["reader_doc_indices_topk"], [2])

    def test_evaluate_rolewise_rcr_does_not_fallback_to_baseline_without_roles(self) -> None:
        corpus = [
            {"title": "Base", "text": "baseline"},
            {"title": "Gold", "text": "answer"},
        ]
        payload = evaluate_rolewise_rcr(
            dataset="unit",
            corpus_records=corpus,
            rows_by_qid={0: {"question": "q", "gold_answers": ["a"], "gold_doc_indices": [1]}},
            base_rank_cache={0: {"doc_indices": [0, 1], "base_scores": {0: 1.0, 1: 0.5}}},
            roles_by_query={0: []},
            rolewise_rank_cache={0: {}},
            role_compatibility_cache=None,
            role_top_k=1,
            max_candidates=3,
            reader_top_k=2,
            lambda_values=[0.0],
        )
        self.assertEqual(payload["variants"]["hipporag_v2"][0]["reader_doc_indices_topk"], [0, 1])
        self.assertEqual(payload["variants"]["rolewise_pointwise"][0]["reader_doc_indices_topk"], [])
        self.assertEqual(payload["variants"]["rolewise_coverage_lambda_0_0"][0]["reader_doc_indices_topk"], [])
        self.assertEqual(payload["candidate_pool_summary"]["role_conditioning_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
