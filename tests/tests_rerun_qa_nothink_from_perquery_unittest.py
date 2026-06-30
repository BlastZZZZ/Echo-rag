import tempfile
import unittest
from pathlib import Path

from rerun_qa_nothink_from_perquery import (
    NOTHINK_EXTRA_BODY,
    attach_metrics_by_variant,
    build_prompt_messages,
    cache_key,
    contains_think,
    load_corpus_passages,
    merge_gold_answers,
    make_reader_tasks,
    parse_answer,
    passage_from_corpus_row,
    query_focused_excerpt,
    query_focused_multi_excerpt,
    reader_passage_excerpt,
    recall_at_5,
    row_reader_doc_indices,
    write_json_atomic,
)


class RerunQaNoThinkFromPerqueryTest(unittest.TestCase):
    def test_passage_from_corpus_row_uses_title_newline_text(self):
        row = {"title": "Doc A", "text": "Body text."}
        self.assertEqual(passage_from_corpus_row(row), "Doc A\nBody text.")

    def test_load_corpus_passages_maps_position_and_idx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corpus.json"
            write_json_atomic(path, [{"idx": 7, "title": "T", "text": "X"}])

            passages = load_corpus_passages(path)

        self.assertEqual(passages[0], "T\nX")
        self.assertEqual(passages[7], "T\nX")

    def test_row_reader_doc_indices_prefers_reader_docs(self):
        row = {"reader_doc_indices_topk": [3, 2, 1], "retrieved_doc_indices_top5": [9, 8, 7]}
        self.assertEqual(row_reader_doc_indices(row), [3, 2, 1])

    def test_recall_at_5_uses_fixed_reader_context(self):
        row = {"gold_doc_indices": [1, 2], "reader_doc_indices_topk": [2, 9, 8, 7, 6]}
        self.assertEqual(recall_at_5(row), 0.5)

    def test_build_prompt_messages_falls_back_to_musique_template(self):
        messages = build_prompt_messages("What?", ["Title\nText"], "2wikimultihopqa")

        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("Wikipedia Title: Title\nText", messages[-1]["content"])
        self.assertIn("Question: What?", messages[-1]["content"])
        self.assertIn("Thought: ", messages[-1]["content"])

    def test_build_prompt_messages_short_span_asks_for_concise_context_phrase(self):
        messages = build_prompt_messages(
            "What did the Bohners start with?",
            ["Farm\nThe Bohners eventually decided to start with U-pick berries."],
            "hgrag_agriculture",
            reader_prompt_mode="short_span",
        )

        self.assertIn("extractive QA reader", messages[0]["content"])
        self.assertIn("shortest exact phrase", messages[1]["content"])
        self.assertIn("Prefer copying the answer span verbatim", messages[1]["content"])
        self.assertIn("Answer: <short phrase>", messages[1]["content"])
        self.assertNotIn("Thought:", messages[1]["content"])

    def test_query_focused_excerpt_selects_question_relevant_late_window(self):
        text = "Intro " * 80 + "The Bohners eventually decided to start with U-pick berries."

        excerpt = query_focused_excerpt(text, "What did the Bohners decide to start with?", max_chars=90)

        self.assertIn("Bohners", excerpt)
        self.assertIn("U-pick berries", excerpt)
        self.assertTrue(excerpt.startswith("... "))

    def test_reader_passage_excerpt_preserves_title_and_focuses_body(self):
        passage = "Farm profile\n" + ("Intro " * 80) + "The Bohners started with U-pick berries."

        excerpt = reader_passage_excerpt(
            passage,
            "What did the Bohners start with?",
            mode="query_focus",
            max_chars=90,
        )

        self.assertTrue(excerpt.startswith("Farm profile\n"))
        self.assertIn("U-pick berries", excerpt)

    def test_query_focused_multi_excerpt_keeps_distinct_relevant_windows(self):
        text = (
            "Growing Power works with youth. "
            + ("filler " * 80)
            + "Jones Valley Urban Farm sells crops through farmers' markets. "
            + ("tail " * 80)
        )

        excerpt = query_focused_multi_excerpt(
            text,
            "Where are crops from Growing Power and Jones Valley Urban Farm sold?",
            max_chars=260,
        )

        self.assertIn("Growing Power", excerpt)
        self.assertIn("farmers' markets", excerpt)
        self.assertIn("\n...\n", excerpt)

    def test_parse_answer_matches_hipporag_fallback(self):
        parsed, ok = parse_answer("Thought...\nAnswer: Paris")
        self.assertEqual(parsed, "Paris")
        self.assertTrue(ok)

        parsed, ok = parse_answer("<think>hidden</think>")
        self.assertEqual(parsed, "<think>hidden</think>")
        self.assertFalse(ok)

    def test_contains_think_is_case_insensitive(self):
        self.assertTrue(contains_think("<THINK>x</THINK>"))
        self.assertFalse(contains_think("Answer: x"))

    def test_cache_key_includes_no_think_extra_body(self):
        messages = [{"role": "user", "content": "Q"}]
        no_think = cache_key(messages, "m", 0.0, 400, NOTHINK_EXTRA_BODY)
        default = cache_key(messages, "m", 0.0, 400, {})

        self.assertNotEqual(no_think, default)

    def test_make_reader_tasks_from_perquery(self):
        data = {
            "variants": {
                "base": [
                    {
                        "query_index": 0,
                        "question": "Q?",
                        "gold_answers": ["A"],
                        "reader_doc_indices_topk": [0],
                    }
                ]
            }
        }

        tasks = make_reader_tasks(data, {0: "Doc\nText"}, dataset="2wikimultihopqa")

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].variant, "base")
        self.assertEqual(tasks[0].doc_indices, [0])
        self.assertEqual(tasks[0].gold_answers, ["A"])

    def test_make_reader_tasks_accepts_nested_variant_rows(self):
        data = {
            "variants": {
                "proprag": {
                    "retrieval": {"R@5": 1.0},
                    "rows": [
                        {
                            "query_index": 0,
                            "question": "Q?",
                            "gold_answers": ["A"],
                            "reader_doc_indices_topk": [0],
                        }
                    ],
                }
            }
        }

        tasks = make_reader_tasks(data, {0: "Doc\nText"}, dataset="musique", variants=["proprag"])

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].variant, "proprag")
        self.assertEqual(tasks[0].doc_indices, [0])

    def test_make_reader_tasks_can_use_query_focused_reader_passages(self):
        data = {
            "variants": {
                "base": [
                    {
                        "query_index": 0,
                        "question": "What did the Bohners start with?",
                        "gold_answers": ["U-pick berries"],
                        "reader_doc_indices_topk": [0],
                    }
                ]
            }
        }
        passage = "Farm profile\n" + ("Intro " * 80) + "The Bohners started with U-pick berries."

        tasks = make_reader_tasks(
            data,
            {0: passage},
            dataset="hgrag_agriculture",
            reader_prompt_mode="short_span",
            reader_passage_mode="query_focus",
            reader_max_passage_chars=90,
        )

        user_message = tasks[0].messages[1]["content"]
        self.assertIn("Farm profile", user_message)
        self.assertIn("U-pick berries", user_message)

    def test_make_reader_tasks_enriches_answer_aliases_by_query_index(self):
        data = {
            "variants": {
                "base": [
                    {
                        "query_index": 5,
                        "question": "Q?",
                        "gold_answers": ["La Goulette"],
                        "reader_doc_indices_topk": [0],
                    }
                ]
            }
        }

        tasks = make_reader_tasks(
            data,
            {0: "Doc\nText"},
            dataset="musique",
            answer_aliases_by_qid={5: ["La Goulette", "Tunis"]},
        )

        self.assertEqual(tasks[0].gold_answers, ["La Goulette", "Tunis"])

    def test_merge_gold_answers_deduplicates_aliases(self):
        self.assertEqual(merge_gold_answers(["A", "B"], ["a", "C"]), ["A", "B", "C"])

    def test_attach_metrics_by_variant_computes_exact_match_and_f1(self):
        data = {
            "variants": {
                "base": [
                    {
                        "query_index": 0,
                        "gold_answers": ["Paris"],
                        "gold_doc_indices": [1],
                        "reader_doc_indices_topk": [1],
                        "predicted_answer": "<think>bad</think>",
                        "ExactMatch": 0.0,
                        "F1": 0.0,
                    }
                ]
            }
        }
        raw_results = {
            ("base", 0): {
                "query_index": 0,
                "gold_answers": ["Paris"],
                "gold_doc_indices": [1],
                "reader_doc_indices_topk": [1],
                "original_predicted_answer": "<think>bad</think>",
                "original_ExactMatch": 0.0,
                "original_F1": 0.0,
                "response_content": "Answer: Paris",
                "predicted_answer": "Paris",
                "parsed_with_answer_marker": True,
                "think_leak": False,
            }
        }

        report = attach_metrics_by_variant(data, raw_results)

        metrics = report["base"]["metrics"]
        self.assertEqual(metrics["R@5"], 1.0)
        self.assertEqual(metrics["nothink_ExactMatch"], 1.0)
        self.assertEqual(metrics["nothink_F1"], 1.0)
        self.assertEqual(metrics["original_think_leak_count"], 1)

    def test_attach_metrics_by_variant_accepts_nested_variant_rows(self):
        data = {
            "variants": {
                "proprag": {
                    "rows": [
                        {
                            "query_index": 0,
                            "gold_answers": ["Paris"],
                            "gold_doc_indices": [1],
                            "reader_doc_indices_topk": [1],
                            "predicted_answer": "old",
                            "ExactMatch": 0.0,
                            "F1": 0.0,
                        }
                    ]
                }
            }
        }
        raw_results = {
            ("proprag", 0): {
                "query_index": 0,
                "gold_answers": ["Paris"],
                "gold_doc_indices": [1],
                "reader_doc_indices_topk": [1],
                "original_predicted_answer": "old",
                "original_ExactMatch": 0.0,
                "original_F1": 0.0,
                "response_content": "Answer: Paris",
                "predicted_answer": "Paris",
                "parsed_with_answer_marker": True,
                "think_leak": False,
            }
        }

        report = attach_metrics_by_variant(data, raw_results, variants=["proprag"])

        self.assertEqual(report["proprag"]["metrics"]["nothink_ExactMatch"], 1.0)


if __name__ == "__main__":
    unittest.main()
