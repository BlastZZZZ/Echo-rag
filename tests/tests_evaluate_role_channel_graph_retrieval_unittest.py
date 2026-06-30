import unittest
from types import SimpleNamespace

from evaluate_role_channel_graph_retrieval import _fingerprint_payload, filter_query_indices


class EvaluateRoleChannelGraphRetrievalTest(unittest.TestCase):
    def test_query_index_filter_preserves_sorted_common_order(self) -> None:
        qids = filter_query_indices([1, 2, 3, 4], query_indices=[4, 2], limit_queries=-1)
        self.assertEqual(qids, [2, 4])

    def test_limit_applies_after_query_index_filter(self) -> None:
        qids = filter_query_indices([1, 2, 3, 4], query_indices=[2, 3, 4], limit_queries=2)
        self.assertEqual(qids, [2, 3])

    def test_no_filter_keeps_existing_limit_behavior(self) -> None:
        qids = filter_query_indices([1, 2, 3, 4], limit_queries=3)
        self.assertEqual(qids, [1, 2, 3])

    def test_fingerprint_tracks_fact_filter_mode(self) -> None:
        args = SimpleNamespace(
            dataset="musique",
            corpus_json="corpus.json",
            per_query_report="report.json",
            per_query_variant=None,
            baseline_top200_cache="top200.json",
            roles_json_path="roles.json",
            save_dir="save",
            reader_top_k=5,
            retrieval_top_k=200,
            num_to_retrieve=50,
            role_fact_top_k=5,
            role_passage_top_k=5,
            channel_output_top_k=50,
            rrf_k=60.0,
            fusion_method="rrf",
            channel_backend="hipporag_graph",
            entry_passage_node_weight=None,
            chunk_size=25,
            llm_name="qwen3-32b",
            llm_base_url="http://localhost:8039/v1",
            rerank_filter_mode="dspy",
            fact_filter_strategy="per_channel",
            rerank_filter_exception_policy="fallback",
            embedding_name="NV-Embed-v2",
            embedding_base_url="http://localhost:8018/v1/embeddings",
            embedding_batch_size=4,
            max_new_tokens=2048,
            openie_mode="online",
            qwen_disable_thinking=True,
        )

        dspy_payload = _fingerprint_payload(args, [0, 1])
        args.rerank_filter_mode = "passthrough"
        passthrough_payload = _fingerprint_payload(args, [0, 1])

        self.assertEqual(dspy_payload["rerank_filter_mode"], "dspy")
        self.assertEqual(passthrough_payload["rerank_filter_mode"], "passthrough")
        self.assertNotEqual(dspy_payload, passthrough_payload)


if __name__ == "__main__":
    unittest.main()
