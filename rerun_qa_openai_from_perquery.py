#!/usr/bin/env python3
"""Rerun QA from a saved per-query report with a plain OpenAI-compatible reader.

This is the GPT/OpenAI-compatible counterpart of
``rerun_qa_nothink_from_perquery.py``.  It reuses the same fixed top-5 context
construction and scoring helpers, but does not send Qwen-specific
``chat_template_kwargs`` in ``extra_body``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import rerun_qa_nothink_from_perquery as qa_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--dataset", default="musique")
    parser.add_argument(
        "--answer-alias-dataset",
        type=Path,
        default=None,
        help="Optional dataset JSON used to enrich gold_answers with answer_aliases.",
    )
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default="https://yunwu.ai/v1")
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI-compatible API key. If omitted, OPENAI_API_KEY is used.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument(
        "--reader-prompt-mode",
        choices=sorted(qa_base.READER_PROMPT_MODES),
        default="hipporag",
        help="Reader prompt style. 'hipporag' preserves the original template; 'short_span' asks for concise copied spans.",
    )
    parser.add_argument(
        "--reader-passage-mode",
        choices=sorted(qa_base.READER_PASSAGE_MODES),
        default="full",
        help="Reader context construction. 'query_focus' extracts a question-relevant window from each fixed Top-5 passage.",
    )
    parser.add_argument(
        "--reader-max-passage-chars",
        type=int,
        default=1200,
        help="Per-passage character budget used by --reader-passage-mode query_focus.",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-wait-seconds", type=float, default=2.0)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=300.0,
        help="HTTP request timeout for each reader call. Timeouts retry; they are not scored as empty answers.",
    )
    parser.add_argument("--focus-query-index", type=int, action="append", default=[43, 66])
    return parser.parse_args()


def cache_key(
    messages: Sequence[Mapping[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    return qa_base.cache_key(messages, model, temperature, max_tokens, {})


def call_reader_plain(
    client: Any,
    messages: Sequence[Mapping[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_wait_seconds: float,
) -> Tuple[str, Dict[str, Any]]:
    last_error: Optional[BaseException] = None
    for attempt in range(max(1, retries)):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            usage = getattr(response, "usage", None)
            return str(choice.message.content or ""), {
                "finish_reason": getattr(choice, "finish_reason", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            }
        except Exception as exc:  # pragma: no cover - live API path
            last_error = exc
            if attempt + 1 >= max(1, retries):
                break
            time.sleep(float(retry_wait_seconds) * (2**attempt))
    raise RuntimeError(f"Reader call failed after {retries} attempts: {last_error}") from last_error


def make_openai_client_with_timeout(base_url: str, api_key: str, timeout_seconds: float) -> Any:
    from openai import OpenAI

    timeout = max(1.0, float(timeout_seconds))
    try:
        import httpx

        http_client = httpx.Client(timeout=httpx.Timeout(timeout, read=timeout), trust_env=False)
        return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0)
    except Exception:  # pragma: no cover - dependency-specific fallback
        return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)


def run_reevaluation(
    tasks: Sequence[qa_base.ReaderTask],
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    cache_path: Path,
    concurrency: int,
    retries: int,
    retry_wait_seconds: float,
    request_timeout_seconds: float,
    reader_prompt_mode: str,
    reader_passage_mode: str,
    reader_max_passage_chars: int,
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    if cache_path.exists():
        cache = qa_base.load_json(cache_path)
        if not isinstance(cache, MutableMapping):
            raise ValueError(f"Cache file must contain a JSON object: {cache_path}")
    else:
        cache = {}

    cache_lock = threading.Lock()
    client = make_openai_client_with_timeout(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=request_timeout_seconds,
    )

    def run_one(task: qa_base.ReaderTask) -> Tuple[Tuple[str, int], Dict[str, Any], bool]:
        key = cache_key(task.messages, model, temperature, max_tokens)
        with cache_lock:
            cached = cache.get(key)
        cache_hit = cached is not None
        if cached is None:
            response_content, metadata = call_reader_plain(
                client=client,
                messages=task.messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                retries=retries,
                retry_wait_seconds=retry_wait_seconds,
            )
            cached = {"response_content": response_content, "metadata": metadata}
            with cache_lock:
                cache[key] = cached
                qa_base.write_json_atomic(cache_path, cache)

        response_content = str(cached.get("response_content") or "")
        predicted_answer, parsed = qa_base.parse_answer(response_content)
        row_payload = {
            "query_index": task.row.get("query_index", task.row_index),
            "row_index": task.row_index,
            "question": task.row.get("question"),
            "gold_answers": task.gold_answers,
            "gold_doc_indices": qa_base.as_int_list(task.row.get("gold_doc_indices")),
            "reader_doc_indices_topk": task.doc_indices,
            "original_predicted_answer": task.row.get("predicted_answer"),
            "original_ExactMatch": float(task.row.get("ExactMatch", 0.0) or 0.0),
            "original_F1": float(task.row.get("F1", 0.0) or 0.0),
            "response_content": response_content,
            "predicted_answer": predicted_answer,
            "parsed_with_answer_marker": parsed,
            "think_leak": qa_base.contains_think(response_content),
            "cache_hit": cache_hit,
            "metadata": cached.get("metadata") or {},
        }
        return (task.variant, task.row_index), row_payload, cache_hit

    results: Dict[Tuple[str, int], Dict[str, Any]] = {}
    completed = 0
    hits = 0
    total = len(tasks)
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        future_to_task = {executor.submit(run_one, task): task for task in tasks}
        for future in as_completed(future_to_task):
            key, row_payload, cache_hit = future.result()
            results[key] = row_payload
            completed += 1
            hits += int(cache_hit)
            if completed == total or completed % 10 == 0:
                print(f"[openai-qa] completed {completed}/{total} cache_hits={hits}", flush=True)
    return results


def summarize_variant(
    original_rows: Sequence[Mapping[str, Any]],
    reevaluated_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    base = qa_base.summarize_variant(original_rows, reevaluated_rows)
    return {
        "R@5": base["R@5"],
        "original_ExactMatch": base["original_ExactMatch"],
        "reader_ExactMatch": base["nothink_ExactMatch"],
        "delta_ExactMatch": base["delta_ExactMatch"],
        "original_F1": base["original_F1"],
        "reader_F1": base["nothink_F1"],
        "delta_F1": base["delta_F1"],
        "original_think_leak_count": base["original_think_leak_count"],
        "reader_think_leak_count": base["nothink_think_leak_count"],
        "reader_parse_failure_count": base["nothink_parse_failure_count"],
    }


def attach_metrics_by_variant(
    per_query_data: Mapping[str, Any],
    raw_results: Mapping[Tuple[str, int], Mapping[str, Any]],
    variants: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    raw_variants = per_query_data.get("variants")
    if not isinstance(raw_variants, Mapping):
        raise ValueError("Per-query report must contain a 'variants' object")
    selected = list(variants) if variants else list(raw_variants.keys())

    out: Dict[str, Dict[str, Any]] = {}
    for variant in selected:
        original_rows = raw_variants[variant]
        ordered_rows = [dict(raw_results[(variant, idx)]) for idx in range(len(original_rows))]
        gold_answers = [qa_base.normalize_gold_answers(row.get("gold_answers")) for row in ordered_rows]
        predictions = [str(row.get("predicted_answer") or "") for row in ordered_rows]
        em_scores, f1_scores = qa_base.evaluate_predictions(gold_answers, predictions)
        for row, em, f1 in zip(ordered_rows, em_scores, f1_scores):
            row["ExactMatch"] = em
            row["F1"] = f1
            row["delta_ExactMatch"] = em - float(row.get("original_ExactMatch", 0.0) or 0.0)
            row["delta_F1"] = f1 - float(row.get("original_F1", 0.0) or 0.0)
        out[variant] = {
            "metrics": summarize_variant(original_rows, ordered_rows),
            "rows": ordered_rows,
        }
    return out


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# OpenAI-Compatible Reader-Only Re-Evaluation",
        "",
        "This report fixes the saved reader-facing Top-5 contexts and reruns only the QA reader.",
        "No Qwen-specific `chat_template_kwargs` or other extra body is sent.",
        "",
        "## Config",
        "",
    ]
    meta = report.get("meta", {})
    for key in (
        "per_query_path",
        "corpus_path",
        "answer_alias_dataset_path",
        "answer_alias_queries",
        "dataset",
        "model",
        "base_url",
        "temperature",
        "max_tokens",
        "reader_prompt_mode",
        "reader_passage_mode",
        "reader_max_passage_chars",
        "concurrency",
    ):
        lines.append(f"- `{key}`: `{meta.get(key)}`")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Variant | R@5 fixed | Original EM | Reader EM | dEM | Original F1 | Reader F1 | dF1 | Orig `<think>` | Reader `<think>` | Parse misses |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant, payload in report.get("variants", {}).items():
        metrics = payload.get("metrics", {})
        lines.append(
            "| {variant} | {r5} | {oem} | {rem} | {dem} | {of1} | {rf1} | {df1} | {otl} | {rtl} | {pm} |".format(
                variant=variant,
                r5=qa_base.fmt(metrics.get("R@5")),
                oem=qa_base.fmt(metrics.get("original_ExactMatch")),
                rem=qa_base.fmt(metrics.get("reader_ExactMatch")),
                dem=qa_base.fmt(metrics.get("delta_ExactMatch"), digits=4),
                of1=qa_base.fmt(metrics.get("original_F1")),
                rf1=qa_base.fmt(metrics.get("reader_F1")),
                df1=qa_base.fmt(metrics.get("delta_F1"), digits=4),
                otl=metrics.get("original_think_leak_count"),
                rtl=metrics.get("reader_think_leak_count"),
                pm=metrics.get("reader_parse_failure_count"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    per_query_path = args.per_query.resolve()
    corpus_path = args.corpus.resolve()
    per_query_data = qa_base.load_json(per_query_path)
    corpus_passages = qa_base.load_corpus_passages(corpus_path)
    alias_dataset_path = (
        args.answer_alias_dataset.resolve()
        if args.answer_alias_dataset
        else qa_base.default_dataset_path(args.dataset)
    )
    answer_aliases = qa_base.load_answer_aliases(alias_dataset_path)
    tasks = qa_base.make_reader_tasks(
        per_query_data=per_query_data,
        corpus_passages=corpus_passages,
        dataset=args.dataset,
        variants=args.variant,
        answer_aliases_by_qid=answer_aliases,
        reader_prompt_mode=args.reader_prompt_mode,
        reader_passage_mode=args.reader_passage_mode,
        reader_max_passage_chars=args.reader_max_passage_chars,
    )
    print(f"[openai-qa] tasks={len(tasks)} variants={args.variant or 'all'}", flush=True)
    raw_results = run_reevaluation(
        tasks=tasks,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cache_path=args.cache,
        concurrency=args.concurrency,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        reader_prompt_mode=args.reader_prompt_mode,
        reader_passage_mode=args.reader_passage_mode,
        reader_max_passage_chars=args.reader_max_passage_chars,
    )
    variants = attach_metrics_by_variant(per_query_data, raw_results, variants=args.variant)
    report = {
        "meta": {
            "per_query_path": str(per_query_path),
            "corpus_path": str(corpus_path),
            "answer_alias_dataset_path": str(alias_dataset_path) if alias_dataset_path.exists() else None,
            "answer_alias_queries": len(answer_aliases),
            "dataset": args.dataset,
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "reader_prompt_mode": args.reader_prompt_mode,
            "reader_passage_mode": args.reader_passage_mode,
            "reader_max_passage_chars": args.reader_max_passage_chars,
            "concurrency": args.concurrency,
            "request_timeout_seconds": args.request_timeout_seconds,
            "extra_body": {},
            "cache_path": str(args.cache.resolve()),
            "focus_query_indices": args.focus_query_index,
            "num_tasks": len(tasks),
        },
        "variants": variants,
    }
    qa_base.write_json_atomic(args.output_json, report)
    args.output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"[openai-qa] wrote {args.output_json}", flush=True)
    print(f"[openai-qa] wrote {args.output_md}", flush=True)
    for variant, payload in variants.items():
        metrics = payload["metrics"]
        print(
            "[openai-qa] {variant} R@5={r5:.4f} EM {oem:.4f}->{rem:.4f} "
            "F1 {of1:.4f}->{rf1:.4f}".format(
                variant=variant,
                r5=metrics["R@5"],
                oem=metrics["original_ExactMatch"],
                rem=metrics["reader_ExactMatch"],
                of1=metrics["original_F1"],
                rf1=metrics["reader_F1"],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
