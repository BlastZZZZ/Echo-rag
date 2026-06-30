#!/usr/bin/env python3
"""Run and evaluate role-channel evidence exposure.

This evaluator preserves role channels through graph/PPR retrieval and emits
both channel traces and a diagnostic raw rank-fusion readout.  It is
intentionally separate from ``evaluate_role_aware_graph_entry.py`` because the
methodological question is different:

* one-shot graph entry asks whether roles can be collapsed before PPR;
* role-channel exposure asks whether roles must be preserved through PPR.

The raw fused top-k is not the ECHO-RAG v1 final evidence-set construction
stage.  ECHO selectors consume ``traces[].channels`` and construct a set from
the channel-exposed candidate pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from evaluate_role_coverage_reranker import (
    average,
    corpus_title_to_index,
    gold_doc_indices_for_row,
    load_baseline_rank_cache,
    load_json,
    load_per_query_rows,
    load_roles,
    recall_from_docs,
    save_json,
    unique_preserve_order,
)
from evaluate_role_aware_graph_entry import attach_r5, summarize_r5
from evaluate_rolewise_rcr import make_row, normalize_gold_answers
from role_channel_graph_retrieval import role_channel_graph_retrieve
from run_rolewise_hippov2_retrieval import (
    build_config,
    build_doc_to_index,
    corpus_docs,
    serialize_query_solution,
)


DEFAULT_EMBEDDING_NAME = "/mnt/nvme/Qwen3-Embedding-8B"
DEFAULT_EMBEDDING_BASE_URL = "http://localhost:8018/v1/embeddings"
DEFAULT_LLM_NAME = "qwen3-8b-train"
DEFAULT_LLM_BASE_URL = "http://localhost:8039/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus_json", required=True)
    parser.add_argument("--per_query_report", required=True)
    parser.add_argument("--baseline_top200_cache", required=True)
    parser.add_argument("--roles_json_path", required=True)
    parser.add_argument("--save_dir", required=True, help="Existing HippoRAG v2 dataset save_dir")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--save_md_path", required=True)
    parser.add_argument("--per_query_variant", default=None)
    parser.add_argument("--limit_queries", type=int, default=100)
    parser.add_argument(
        "--query_index",
        type=int,
        action="append",
        default=None,
        help="Optional query_index filter. Can be supplied multiple times; applied before limit_queries.",
    )
    parser.add_argument("--reader_top_k", type=int, default=5)
    parser.add_argument("--retrieval_top_k", type=int, default=200)
    parser.add_argument("--num_to_retrieve", type=int, default=50)
    parser.add_argument("--role_fact_top_k", type=int, default=5)
    parser.add_argument("--role_passage_top_k", type=int, default=5)
    parser.add_argument("--channel_output_top_k", type=int, default=50)
    parser.add_argument("--rrf_k", type=float, default=60.0)
    parser.add_argument(
        "--fusion_method",
        choices=("rrf", "round_robin", "best_channel_first", "pointwise_role_score"),
        default="rrf",
    )
    parser.add_argument("--channel_backend", choices=("hipporag_graph", "seeded_entry"), default="hipporag_graph")
    parser.add_argument("--entry_passage_node_weight", type=float, default=None)
    parser.add_argument("--chunk_size", type=int, default=25)
    parser.add_argument("--llm_name", default=DEFAULT_LLM_NAME)
    parser.add_argument("--llm_base_url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--embedding_name", default=DEFAULT_EMBEDDING_NAME)
    parser.add_argument("--embedding_base_url", default=DEFAULT_EMBEDDING_BASE_URL)
    parser.add_argument("--embedding_batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--llm_timeout_seconds",
        type=float,
        default=None,
        help="Optional OpenAI-compatible LLM connect/write/pool timeout for role-channel retrieval.",
    )
    parser.add_argument(
        "--llm_read_timeout_seconds",
        type=float,
        default=None,
        help="Optional OpenAI-compatible LLM read timeout for role-channel retrieval.",
    )
    parser.add_argument(
        "--llm_max_retries",
        type=int,
        default=None,
        help="Optional max retry attempts for role-channel retrieval LLM calls.",
    )
    parser.add_argument(
        "--rerank_filter_exception_policy",
        choices=("fallback", "raise"),
        default="fallback",
        help="Whether fact-filter LLM exceptions fall back to empty facts or abort the current run.",
    )
    parser.add_argument(
        "--rerank_filter_mode",
        choices=("dspy", "passthrough"),
        default="dspy",
        help="Use passthrough to keep embedding top facts instead of calling the DSPy fact filter.",
    )
    parser.add_argument(
        "--fact_filter_strategy",
        choices=("per_channel", "shared_question", "shared_question_by_demand"),
        default="per_channel",
        help=(
            "Use shared_question to run one global fact-filter call over the union of role-channel fact candidates. "
            "Use shared_question_by_demand to keep one call per question while assigning fact seeds per demand."
        ),
    )
    parser.add_argument("--openie_mode", default="online")
    parser.add_argument("--force_index_from_scratch", action="store_true")
    parser.add_argument("--force_openie_from_scratch", action="store_true")
    parser.add_argument("--qwen_disable_thinking", action="store_true")
    parser.add_argument(
        "--chunk_checkpoint_dir",
        default=None,
        help="Optional directory for chunk-level checkpoints. Disabled when omitted.",
    )
    parser.add_argument(
        "--resume_chunk_checkpoints",
        action="store_true",
        help="Reuse valid chunk checkpoints from --chunk_checkpoint_dir.",
    )
    parser.add_argument(
        "--fact_filter_cache_json",
        default=None,
        help="Optional exact-input cache for HippoRAG fact-filter calls.",
    )
    return parser.parse_args()


def _channel_counts(traces: Sequence[Mapping[str, Any]], key: str) -> List[int]:
    values: List[int] = []
    for trace in traces:
        channels = trace.get("channels", [])
        if not isinstance(channels, Sequence) or isinstance(channels, (str, bytes)):
            continue
        for channel in channels:
            if not isinstance(channel, Mapping):
                continue
            raw_values = channel.get(key, [])
            if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)):
                values.append(len(raw_values))
    return values


def filter_query_indices(
    common_qids: Sequence[int],
    *,
    query_indices: Sequence[int] | None = None,
    limit_queries: int | None = None,
) -> List[int]:
    """Apply an optional explicit qid filter and then the existing limit."""
    qids = [int(qid) for qid in common_qids]
    if query_indices:
        requested = {int(qid) for qid in query_indices}
        qids = [qid for qid in qids if qid in requested]
    if limit_queries is not None and int(limit_queries) >= 0:
        qids = qids[: int(limit_queries)]
    return qids


def _fingerprint_payload(args: argparse.Namespace, common_qids: Sequence[int]) -> Dict[str, Any]:
    """Return method-affecting settings used to validate chunk checkpoints."""
    return {
        "checkpoint_version": 2,
        "dataset": args.dataset,
        "corpus_json": args.corpus_json,
        "per_query_report": args.per_query_report,
        "per_query_variant": args.per_query_variant,
        "baseline_top200_cache": args.baseline_top200_cache,
        "roles_json_path": args.roles_json_path,
        "save_dir": args.save_dir,
        "common_qids": [int(qid) for qid in common_qids],
        "reader_top_k": int(args.reader_top_k),
        "retrieval_top_k": int(args.retrieval_top_k),
        "num_to_retrieve": int(args.num_to_retrieve),
        "role_fact_top_k": int(args.role_fact_top_k),
        "role_passage_top_k": int(args.role_passage_top_k),
        "channel_output_top_k": int(args.channel_output_top_k),
        "rrf_k": float(args.rrf_k),
        "fusion_method": args.fusion_method,
        "channel_backend": args.channel_backend,
        "entry_passage_node_weight": (
            float(args.entry_passage_node_weight)
            if args.entry_passage_node_weight is not None
            else None
        ),
        "chunk_size": int(args.chunk_size),
        "llm_name": args.llm_name,
        "llm_base_url": args.llm_base_url,
        "rerank_filter_mode": args.rerank_filter_mode,
        "fact_filter_strategy": args.fact_filter_strategy,
        "rerank_filter_exception_policy": args.rerank_filter_exception_policy,
        "embedding_name": args.embedding_name,
        "embedding_base_url": args.embedding_base_url,
        "embedding_batch_size": int(args.embedding_batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "openie_mode": args.openie_mode,
        "qwen_disable_thinking": bool(args.qwen_disable_thinking),
    }


def _fingerprint(args: argparse.Namespace, common_qids: Sequence[int]) -> str:
    raw = json.dumps(_fingerprint_payload(args, common_qids), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _chunk_checkpoint_path(checkpoint_dir: Path, chunk_qids: Sequence[int], chunk_ordinal: int) -> Path:
    start_qid = int(chunk_qids[0]) if chunk_qids else -1
    end_qid = int(chunk_qids[-1]) if chunk_qids else -1
    return checkpoint_dir / f"chunk_{chunk_ordinal:04d}_q{start_qid:06d}_q{end_qid:06d}.json"


def _load_valid_chunk_checkpoint(
    path: Path,
    *,
    expected_fingerprint: str,
    expected_qids: Sequence[int],
) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        return None
    if str(payload.get("config_fingerprint") or "") != expected_fingerprint:
        return None
    if [int(qid) for qid in payload.get("chunk_qids", [])] != [int(qid) for qid in expected_qids]:
        return None
    return dict(payload)


def _load_fact_filter_cache(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    payload = load_json(cache_path)
    if isinstance(payload, Mapping) and isinstance(payload.get("entries"), Mapping):
        return dict(payload.get("entries", {}))
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _save_fact_filter_cache(path: str | None, cache: Mapping[str, Any]) -> None:
    if not path:
        return
    save_json(path, {"cache_version": 1, "entries": dict(cache)})


def _rows_from_chunk_results(
    *,
    chunk_qids: Sequence[int],
    chunk_results: Sequence[Any],
    rows_by_qid: Mapping[int, Mapping[str, Any]],
    title_to_index: Mapping[str, int],
    base_rank_cache: Mapping[int, Mapping[str, Any]],
    corpus_records: Sequence[Mapping[str, Any]],
    doc_to_idx: Mapping[str, int],
    reader_top_k: int,
) -> Dict[str, Any]:
    baseline_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    missing_doc_count = 0

    for qid, result in zip(chunk_qids, chunk_results):
        row = rows_by_qid[int(qid)]
        question = str(row.get("question") or "")
        gold_docs = gold_doc_indices_for_row(row, title_to_index)
        gold_answers = normalize_gold_answers(row.get("gold_answers"))

        base_docs = unique_preserve_order(base_rank_cache[int(qid)].get("doc_indices", []))[: int(reader_top_k)]
        base_row = make_row(
            query_index=int(qid),
            question=question,
            gold_answers=gold_answers,
            gold_doc_indices=gold_docs,
            reader_doc_indices=base_docs,
            corpus_records=corpus_records,
        )
        baseline_rows.append(attach_r5(base_row, gold_docs, reader_top_k))

        serialized, missing = serialize_query_solution(result.solution, doc_to_idx)
        missing_doc_count += missing
        selected_docs = unique_preserve_order(serialized.get("doc_indices", []))[: int(reader_top_k)]
        selected_row = make_row(
            query_index=int(qid),
            question=question,
            gold_answers=gold_answers,
            gold_doc_indices=gold_docs,
            reader_doc_indices=selected_docs,
            corpus_records=corpus_records,
        )
        selected_row["channel_count"] = int(result.trace.channel_count)
        selected_row["empty_reason"] = result.trace.empty_reason
        selected_rows.append(attach_r5(selected_row, gold_docs, reader_top_k))
        trace = asdict(result.trace)
        trace["query_index"] = int(qid)
        traces.append(trace)

    return {
        "baseline_rows": baseline_rows,
        "selected_rows": selected_rows,
        "traces": traces,
        "missing_doc_count": int(missing_doc_count),
    }


def _append_chunk_payload(
    *,
    payload: Mapping[str, Any],
    baseline_rows: List[Dict[str, Any]],
    selected_rows: List[Dict[str, Any]],
    traces: List[Dict[str, Any]],
) -> int:
    baseline_rows.extend(list(payload.get("baseline_rows", []) or []))
    selected_rows.extend(list(payload.get("selected_rows", []) or []))
    traces.extend(list(payload.get("traces", []) or []))
    return int(payload.get("missing_doc_count", 0) or 0)


def make_md(payload: Mapping[str, Any]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), Mapping) else {}
    diagnostics = payload.get("diagnostics", {}) if isinstance(payload.get("diagnostics"), Mapping) else {}
    config = payload.get("config", {}) if isinstance(payload.get("config"), Mapping) else {}
    lines = [
        "# Role-channel Evidence Exposure",
        "",
        "## Method",
        "",
        "```text",
        "question",
        "-> question-only evidence roles",
        "-> one role-specific graph/PPR channel per role",
        "-> channel trace artifact for candidate-pool construction",
        "-> diagnostic raw rank fusion only",
        "```",
        "",
        "This run does not rebuild the graph, change OpenIE, or use gold labels during retrieval.",
        "The raw fusion row below is diagnostic; it is not the final ECHO-RAG v1 output.",
        "",
        "## Configuration",
        "",
        "```text",
        f"dataset: {payload.get('dataset')}",
        f"role_fact_top_k: {config.get('role_fact_top_k')}",
        f"role_passage_top_k: {config.get('role_passage_top_k')}",
        f"channel_output_top_k: {config.get('channel_output_top_k')}",
        f"rrf_k: {config.get('rrf_k')}",
        f"fusion_method: {config.get('fusion_method')}",
        f"channel_backend: {config.get('channel_backend')}",
        f"entry_passage_node_weight: {config.get('entry_passage_node_weight')}",
        f"reader_top_k: {config.get('reader_top_k')}",
        f"limit_queries: {config.get('limit_queries')}",
        "```",
        "",
        "## Retrieval",
        "",
        "| Method | R@5 | dR@5 | Changed | Improved | Worsened |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    base_r5 = float(summary.get("baseline_r5", 0.0))
    selected_r5 = float(summary.get("selected_r5", 0.0))
    lines.append(f"| HippoRAG v2 | {base_r5:.4f} | - | - | - | - |")
    lines.append(
        "| Diagnostic raw role-channel fusion | "
        f"{selected_r5:.4f} | {float(summary.get('delta_r5', 0.0)):+.4f} | "
        f"{int(summary.get('changed_query_count', 0))} | "
        f"{int(summary.get('improved_query_count', 0))} | "
        f"{int(summary.get('worsened_query_count', 0))} |"
    )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "```text",
            f"empty_role_count: {diagnostics.get('empty_role_count')}",
            f"empty_channel_count: {diagnostics.get('empty_channel_count')}",
            f"missing_doc_mappings: {diagnostics.get('missing_doc_mappings')}",
            f"avg_active_channels: {float(diagnostics.get('avg_active_channels', 0.0)):.2f}",
            f"avg_channel_fact_seeds: {float(diagnostics.get('avg_channel_fact_seeds', 0.0)):.2f}",
            f"avg_channel_passage_seeds: {float(diagnostics.get('avg_channel_passage_seeds', 0.0)):.2f}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from hipporag.HippoRAG import HippoRAG

    corpus_records = load_json(args.corpus_json)
    if not isinstance(corpus_records, list):
        raise ValueError(f"Corpus must be a list: {args.corpus_json}")
    args.corpus_len = len(corpus_records)
    docs = corpus_docs(corpus_records)
    doc_to_idx = build_doc_to_index(docs)

    per_query_variant, rows_by_qid = load_per_query_rows(args.per_query_report, args.per_query_variant)
    roles_by_query = load_roles(args.roles_json_path)
    base_rank_cache = load_baseline_rank_cache(args.baseline_top200_cache, candidate_top_k=args.num_to_retrieve)
    title_to_index = corpus_title_to_index(corpus_records)

    common_qids = filter_query_indices(
        sorted(set(rows_by_qid) & set(roles_by_query) & set(base_rank_cache)),
        query_indices=args.query_index,
        limit_queries=args.limit_queries,
    )

    config = build_config(args)
    if args.llm_timeout_seconds is not None:
        setattr(config, "llm_timeout_seconds", float(args.llm_timeout_seconds))
    if args.llm_read_timeout_seconds is not None:
        setattr(config, "llm_read_timeout_seconds", float(args.llm_read_timeout_seconds))
    if args.llm_max_retries is not None:
        config.max_retry_attempts = max(1, int(args.llm_max_retries))
    setattr(config, "rerank_filter_fail_on_exception", args.rerank_filter_exception_policy == "raise")
    setattr(config, "rerank_filter_mode", args.rerank_filter_mode)
    system = HippoRAG(global_config=config)
    fact_filter_cache = _load_fact_filter_cache(args.fact_filter_cache_json)
    fact_filter_cache_entries_before = len(fact_filter_cache)

    baseline_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    missing_doc_count = 0

    chunk_size = max(int(args.chunk_size), 1)
    checkpoint_dir = Path(args.chunk_checkpoint_dir) if args.chunk_checkpoint_dir else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_fingerprint = _fingerprint(args, common_qids)
    chunk_starts = list(range(0, len(common_qids), chunk_size))
    checkpoint_hits = 0
    checkpoint_writes = 0

    for chunk_ordinal, start in enumerate(chunk_starts):
        chunk_qids = common_qids[start : start + chunk_size]
        checkpoint_payload: Dict[str, Any] | None = None
        checkpoint_path = (
            _chunk_checkpoint_path(checkpoint_dir, chunk_qids, chunk_ordinal)
            if checkpoint_dir is not None
            else None
        )
        if checkpoint_path is not None and bool(args.resume_chunk_checkpoints):
            checkpoint_payload = _load_valid_chunk_checkpoint(
                checkpoint_path,
                expected_fingerprint=config_fingerprint,
                expected_qids=chunk_qids,
            )
            if checkpoint_payload is not None:
                checkpoint_hits += 1
                print(
                    f"[role-channel] checkpoint hit chunk {chunk_ordinal + 1}/{len(chunk_starts)} "
                    f"qids={int(chunk_qids[0])}-{int(chunk_qids[-1])}: {checkpoint_path}",
                    flush=True,
                )

        if checkpoint_payload is None:
            print(
                f"[role-channel] start chunk {chunk_ordinal + 1}/{len(chunk_starts)} "
                f"qids={int(chunk_qids[0])}-{int(chunk_qids[-1])}",
                flush=True,
            )
            chunk_queries = [str(rows_by_qid[qid].get("question") or "") for qid in chunk_qids]
            chunk_results = role_channel_graph_retrieve(
                system=system,
                queries=chunk_queries,
                query_indices=chunk_qids,
                roles_by_query=roles_by_query,
                num_to_retrieve=int(args.num_to_retrieve),
                role_fact_top_k=int(args.role_fact_top_k),
                role_passage_top_k=int(args.role_passage_top_k),
                channel_output_top_k=int(args.channel_output_top_k),
                rrf_k=float(args.rrf_k),
                fusion_method=args.fusion_method,
                channel_backend=args.channel_backend,
                entry_passage_node_weight=args.entry_passage_node_weight,
                fact_filter_cache=fact_filter_cache if args.fact_filter_cache_json else None,
                fact_filter_strategy=args.fact_filter_strategy,
            )
            rows_payload = _rows_from_chunk_results(
                chunk_qids=chunk_qids,
                chunk_results=chunk_results,
                rows_by_qid=rows_by_qid,
                title_to_index=title_to_index,
                base_rank_cache=base_rank_cache,
                corpus_records=corpus_records,
                doc_to_idx=doc_to_idx,
                reader_top_k=int(args.reader_top_k),
            )
            checkpoint_payload = {
                "checkpoint_version": 2,
                "config_fingerprint": config_fingerprint,
                "chunk_ordinal": int(chunk_ordinal),
                "chunk_start": int(start),
                "chunk_qids": [int(qid) for qid in chunk_qids],
                **rows_payload,
            }
            if checkpoint_path is not None:
                save_json(checkpoint_path, checkpoint_payload)
                checkpoint_writes += 1
                print(
                    f"[role-channel] saved checkpoint chunk {chunk_ordinal + 1}/{len(chunk_starts)}: "
                    f"{checkpoint_path}",
                    flush=True,
                )
            _save_fact_filter_cache(args.fact_filter_cache_json, fact_filter_cache)

        missing_doc_count += _append_chunk_payload(
            payload=checkpoint_payload,
            baseline_rows=baseline_rows,
            selected_rows=selected_rows,
            traces=traces,
        )
        print(
            f"[role-channel] done chunk {chunk_ordinal + 1}/{len(chunk_starts)} "
            f"total_queries={len(selected_rows)}",
            flush=True,
        )

    summary = summarize_r5(selected_rows, baseline_rows)
    empty_role_count = sum(1 for trace in traces if trace.get("empty_reason") == "no_valid_roles")
    empty_channel_count = sum(1 for trace in traces if trace.get("empty_reason") == "no_active_channels")
    active_channel_counts = [int(trace.get("channel_count", 0)) for trace in traces]
    channel_fact_counts = _channel_counts(traces, "selected_fact_indices")
    channel_passage_counts = _channel_counts(traces, "selected_passage_indices")

    payload: Dict[str, Any] = {
        "mode": "role_channel_graph_retrieval",
        "method_role": "evidence_exposure_with_diagnostic_raw_fusion",
        "dataset": args.dataset,
        "per_query_variant": per_query_variant,
        "summary": summary,
        "variants": {
            "hipporag_v2": baseline_rows,
            "role_channel_graph_retrieval": selected_rows,
        },
        "query_solutions": [
            {
                "question": row.get("question", ""),
                "doc_indices": row.get("reader_doc_indices_topk", []),
                "doc_scores": None,
            }
            for row in selected_rows
        ],
        "traces": traces,
        "config": {
            "dataset": args.dataset,
            "save_dir": args.save_dir,
            "llm_name": args.llm_name,
            "llm_base_url": args.llm_base_url,
            "embedding_name": args.embedding_name,
            "embedding_base_url": args.embedding_base_url,
            "llm_timeout_seconds": (
                float(args.llm_timeout_seconds)
                if args.llm_timeout_seconds is not None
                else None
            ),
            "llm_read_timeout_seconds": (
                float(args.llm_read_timeout_seconds)
                if args.llm_read_timeout_seconds is not None
                else None
            ),
            "llm_max_retries": int(args.llm_max_retries) if args.llm_max_retries is not None else None,
            "rerank_filter_mode": args.rerank_filter_mode,
            "fact_filter_strategy": args.fact_filter_strategy,
            "rerank_filter_exception_policy": args.rerank_filter_exception_policy,
            "retrieval_top_k": int(args.retrieval_top_k),
            "num_to_retrieve": int(args.num_to_retrieve),
            "reader_top_k": int(args.reader_top_k),
            "role_fact_top_k": int(args.role_fact_top_k),
            "role_passage_top_k": int(args.role_passage_top_k),
            "channel_output_top_k": int(args.channel_output_top_k),
            "rrf_k": float(args.rrf_k),
            "fusion_method": args.fusion_method,
            "channel_backend": args.channel_backend,
            "entry_passage_node_weight": (
                float(args.entry_passage_node_weight)
                if args.entry_passage_node_weight is not None
                else None
            ),
            "limit_queries": int(args.limit_queries) if args.limit_queries is not None else None,
            "query_index_filter": [int(qid) for qid in args.query_index] if args.query_index else None,
            "chunk_size": int(args.chunk_size),
            "openie_mode": args.openie_mode,
            "qwen_disable_thinking": bool(args.qwen_disable_thinking),
            "chunk_checkpoint_dir": args.chunk_checkpoint_dir,
            "resume_chunk_checkpoints": bool(args.resume_chunk_checkpoints),
            "fact_filter_cache_json": args.fact_filter_cache_json,
        },
        "diagnostics": {
            "num_queries": len(common_qids),
            "raw_fusion_is_final_echo_method": False,
            "empty_role_count": int(empty_role_count),
            "empty_channel_count": int(empty_channel_count),
            "missing_doc_mappings": int(missing_doc_count),
            "avg_active_channels": average(active_channel_counts),
            "avg_channel_fact_seeds": average(channel_fact_counts),
            "avg_channel_passage_seeds": average(channel_passage_counts),
            "chunk_checkpoint_hits": int(checkpoint_hits),
            "chunk_checkpoint_writes": int(checkpoint_writes),
            "fact_filter_cache_entries_before": int(fact_filter_cache_entries_before),
            "fact_filter_cache_entries_after": int(len(fact_filter_cache)),
            "fact_filter_cache_entries_added": int(len(fact_filter_cache) - fact_filter_cache_entries_before),
        },
        "source": {
            "corpus_json": args.corpus_json,
            "per_query_report": args.per_query_report,
            "baseline_top200_cache": args.baseline_top200_cache,
            "roles_json_path": args.roles_json_path,
        },
    }
    return payload


def main() -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    payload = run(args)
    save_json(args.output_json, payload)
    Path(args.save_md_path).write_text(make_md(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
