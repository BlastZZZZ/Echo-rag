import unittest

from role_coverage_reranker import (
    infer_role_ids,
    role_coverage_value,
    select_role_coverage_topk,
    unique_candidates,
)


class RoleCoverageRerankerTest(unittest.TestCase):
    def test_unique_candidates_preserves_first_seen_order(self) -> None:
        self.assertEqual(unique_candidates([3, "2", 3, None, "bad", 1]), [3, 2, 1])

    def test_infer_role_ids_uses_candidate_order(self) -> None:
        role_scores = {
            0: {"answer": 0.8},
            1: {"grounding": 1.0, "answer": 0.2},
        }
        self.assertEqual(infer_role_ids(role_scores, candidate_passages=[1, 0]), ["grounding", "answer"])

    def test_value_uses_soft_max_per_role_not_duplicate_sum(self) -> None:
        value = role_coverage_value(
            selected_passages=[0, 1],
            base_scores={0: 0.0, 1: 0.0},
            role_scores={0: {"grounding": 0.6}, 1: {"grounding": 0.9}},
            role_ids=["grounding"],
            base_relevance_weight=0.0,
        )
        self.assertAlmostEqual(value, 0.9)

    def test_greedy_prefers_new_role_coverage_over_redundant_rank(self) -> None:
        selected = select_role_coverage_topk(
            candidate_passages=[0, 1, 2],
            base_scores={0: 1.0, 1: 0.95, 2: 0.9},
            role_scores={
                0: {"grounding": 1.0},
                1: {"grounding": 0.95},
                2: {"answer": 1.0},
            },
            role_ids=["grounding", "answer"],
            base_relevance_weight=0.1,
            k=2,
        )
        self.assertEqual(selected, [0, 2])

    def test_large_base_relevance_weight_can_keep_high_ranked_redundant_passage(self) -> None:
        selected = select_role_coverage_topk(
            candidate_passages=[0, 1, 2],
            base_scores={0: 1.0, 1: 0.95, 2: 0.2},
            role_scores={
                0: {"grounding": 1.0},
                1: {"grounding": 0.95},
                2: {"answer": 1.0},
            },
            role_ids=["grounding", "answer"],
            base_relevance_weight=10.0,
            k=2,
        )
        self.assertEqual(selected, [0, 1])

    def test_role_ids_filter_unrequested_compatibility(self) -> None:
        selected = select_role_coverage_topk(
            candidate_passages=[0, 1],
            base_scores={0: 0.0, 1: 0.0},
            role_scores={0: {"noise": 1.0}, 1: {"answer": 0.8}},
            role_ids=["answer"],
            base_relevance_weight=0.0,
            k=1,
        )
        self.assertEqual(selected, [1])

    def test_selector_respects_k_and_uniqueness(self) -> None:
        selected = select_role_coverage_topk(
            candidate_passages=[0, 0, 1, 2],
            base_scores={0: 0.3, 1: 0.2, 2: 0.1},
            role_scores={0: {"r0": 1.0}, 1: {"r1": 1.0}, 2: {"r2": 1.0}},
            role_ids=["r0", "r1", "r2"],
            base_relevance_weight=0.0,
            k=2,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(set(selected)), 2)


if __name__ == "__main__":
    unittest.main()
