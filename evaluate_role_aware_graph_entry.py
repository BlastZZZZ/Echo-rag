#!/usr/bin/env python3
"""Run and evaluate Role-aware Graph Entry retrieval.

This is the first v2 experiment after Role-wise RCR v1.  It uses question-only
evidence roles to choose HippoRAG fact seeds, then runs the existing
HippoRAG graph search once per question.
"""

from __future__ import annotations

import argparse
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
    title_for_doc,
    unique_preserve_order,
)
from evaluate_rolewise_rcr import make_row, normalize_gold_answers
from role_aware_graph_entry import role_aware_graph_entry_retrieve
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
    parser.add_argument("--reader_top_k", type=int, default=5)
    parser.add_argument("--retrieval_top_k", type=int, default=200)
    parser.add_argument("--num_to_retrieve", type=int, default=50)
    parser.add_argument("--role_fact_top_k", type=int, default=5)
    parser.add_argument("--max_fact_seeds", type=int, default=5)
    parser.add_argument(
        "--max_fact_seeds_mode",
        choices=("fixed", "total_channel_budget"),
        default="fixed",
        help="Use total_channel_budget for one-shot v6 retrieval with role_count * linking_top_k fact seeds.",
    )
    parser.add_argument(
        "--link_top_k_mode",
        choices=("config", "total_channel_budget"),
        default="config",
        help="Use total_channel_budget to preserve the total channel linking budget inside one graph search.",
    )
    parser.add_argument("--role_passage_top_k", type=int, default=5)
    parser.add_argument("--max_passage_seeds", type=int, default=15)
    parser.add_argument("--entry_passage_node_weight", type=float, default=None)
    parser.add_argument("--fact_only", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=25)
    parser.add_argument("--llm_name", default=DEFAULT_LLM_NAME)
    parser.add_argument("--llm_base_url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--embedding_name", default=DEFAULT_EMBEDDING_NAME)
    parser.add_argument("--embedding_base_url", default=DEFAULT_EMBEDDING_BASE_URL)
    parser.add_argument("--embedding_batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--openie_mode", default="online")
    parser.add_argument("--force_index_from_scratch", action="store_true")
    parser.add_argument("--force_openie_from_scratch", action="store_true")
    parser.add_argument("--qwen_disable_thinking", action="store_true")
    return parser.parse_args()


def summarize_r5(rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    values = [float(row.get("r5", 0.0)) for row in rows]
    baseline_values = [float(row.get("r5", 0.0)) for row in baseline_rows]
    improved = 0
    worsened = 0
    changed = 0
    for row, base_row in zip(rows, baseline_rows):
        selected_docs = unique_preserve_order(row.get("reader_doc_indices_topk", []))
        base_docs = unique_preserve_order(base_row.get("reader_doc_indices_topk", []))
        if selected_docs != base_docs:
            changed += 1
        if float(row.get("r5", 0.0)) > float(base_row.get("r5", 0.0)):
            improved += 1
        elif float(row.get("r5", 0.0)) < float(base_row.get("r5", 0.0)):
            worsened += 1
    return {
        "num_queries": len(values),
        "baseline_r5": average(baseline_values),
        "selected_r5": average(values),
        "delta_r5": average(values) - average(baseline_values),
        "changed_query_count": changed,
        "improved_query_count": improved,
        "worsened_query_count": worsened,
    }


def attach_r5(row: Dict[str, Any], gold_docs: Sequence[int], reader_top_k: int) -> Dict[str, Any]:
    docs = unique_preserve_order(row.get("reader_doc_indices_topk", []))[: int(reader_top_k)]
    row["r5"] = recall_from_docs(gold_docs, docs, k=int(reader_top_k))
    return row


def make_md(payload: Mapping[str, Any]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), Mapping) else {}
    diagnostics = payload.get("diagnostics", {}) if isinstance(payload.get("diagnostics"), Mapping) else {}
    config = payload.get("config", {}) if isinstance(payload.get("config"), Mapping) else {}
    lines = [
        "# Role-aware Graph Entry Retrieval",
        "",
        "## Method",
        "",
        "```text",
        "question",
        "-> question-only evidence roles",
        "-> role-conditioned fact and passage entry selection",
        "-> one HippoRAG graph/PPR search",
        "-> reader-visible top-5 evidence set",
        "```",
        "",
        "This v2 run does not rebuild the graph, change OpenIE, or use gold labels during retrieval.",
        "",
        "## Configuration",
        "",
        "```text",
        f"dataset: {payload.get('dataset')}",
        f"role_fact_top_k: {config.get('role_fact_top_k')}",
        f"max_fact_seeds: {config.get('max_fact_seeds')}",
        f"max_fact_seeds_mode: {config.get('max_fact_seeds_mode')}",
        f"link_top_k_mode: {config.get('link_top_k_mode')}",
        f"role_passage_top_k: {config.get('role_passage_top_k')}",
        f"max_passage_seeds: {config.get('max_passage_seeds')}",
        f"entry_passage_node_weight: {config.get('entry_passage_node_weight')}",
        f"entry_components: {config.get('entry_components')}",
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
        "| Role-aware Graph Entry | "
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
            f"empty_fact_seed_count: {diagnostics.get('empty_fact_seed_count')}",
            f"missing_doc_mappings: {diagnostics.get('missing_doc_mappings')}",
            f"avg_selected_fact_seeds: {float(diagnostics.get('avg_selected_fact_seeds', 0.0)):.2f}",
            f"avg_selected_passage_seeds: {float(diagnostics.get('avg_selected_passage_seeds', 0.0)):.2f}",
            f"avg_graph_search_count: {float(diagnostics.get('avg_graph_search_count', 0.0)):.2f}",
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

    common_qids = sorted(set(rows_by_qid) & set(roles_by_query) & set(base_rank_cache))
    if args.limit_queries is not None and int(args.limit_queries) >= 0:
        common_qids = common_qids[: int(args.limit_queries)]

    config = build_config(args)
    system = HippoRAG(global_config=config)

    baseline_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    missing_doc_count = 0

    chunk_size = max(int(args.chunk_size), 1)
    for start in range(0, len(common_qids), chunk_size):
        chunk_qids = common_qids[start : start + chunk_size]
        chunk_queries = [str(rows_by_qid[qid].get("question") or "") for qid in chunk_qids]
        chunk_results = role_aware_graph_entry_retrieve(
            system=system,
            queries=chunk_queries,
            query_indices=chunk_qids,
            roles_by_query=roles_by_query,
            num_to_retrieve=int(args.num_to_retrieve),
            role_fact_top_k=int(args.role_fact_top_k),
            max_fact_seeds=int(args.max_fact_seeds),
            role_passage_top_k=int(args.role_passage_top_k),
            max_passage_seeds=int(args.max_passage_seeds),
            include_role_passage_entries=not bool(args.fact_only),
            entry_passage_node_weight=args.entry_passage_node_weight,
            max_fact_seeds_mode=args.max_fact_seeds_mode,
            link_top_k_mode=args.link_top_k_mode,
        )

        for qid, result in zip(chunk_qids, chunk_results):
            row = rows_by_qid[int(qid)]
            question = str(row.get("question") or "")
            gold_docs = gold_doc_indices_for_row(row, title_to_index)
            gold_answers = normalize_gold_answers(row.get("gold_answers"))

            base_docs = unique_preserve_order(base_rank_cache[int(qid)].get("doc_indices", []))[: int(args.reader_top_k)]
            base_row = make_row(
                query_index=int(qid),
                question=question,
                gold_answers=gold_answers,
                gold_doc_indices=gold_docs,
                reader_doc_indices=base_docs,
                corpus_records=corpus_records,
            )
            baseline_rows.append(attach_r5(base_row, gold_docs, args.reader_top_k))

            serialized, missing = serialize_query_solution(result.solution, doc_to_idx)
            missing_doc_count += missing
            selected_docs = unique_preserve_order(serialized.get("doc_indices", []))[: int(args.reader_top_k)]
            selected_row = make_row(
                query_index=int(qid),
                question=question,
                gold_answers=gold_answers,
                gold_doc_indices=gold_docs,
                reader_doc_indices=selected_docs,
                corpus_records=corpus_records,
            )
            selected_row["selected_fact_indices"] = list(result.trace.selected_fact_indices)
            selected_row["selected_passage_indices"] = list(result.trace.selected_passage_indices)
            selected_row["selected_fact_count"] = len(result.trace.selected_fact_indices)
            selected_row["selected_passage_count"] = len(result.trace.selected_passage_indices)
            selected_row["graph_search_count"] = int(result.trace.graph_search_count)
            selected_row["fact_seed_budget"] = int(result.trace.fact_seed_budget)
            selected_row["link_top_k_effective"] = int(result.trace.link_top_k_effective)
            selected_row["empty_reason"] = result.trace.empty_reason
            selected_rows.append(attach_r5(selected_row, gold_docs, args.reader_top_k))
            trace = asdict(result.trace)
            trace["query_index"] = int(qid)
            traces.append(trace)

    summary = summarize_r5(selected_rows, baseline_rows)
    empty_role_count = sum(1 for trace in traces if trace.get("empty_reason") == "no_valid_roles")
    empty_fact_seed_count = sum(1 for trace in traces if trace.get("empty_reason") == "no_fact_seeds")
    selected_fact_counts = [int(trace.get("selected_fact_indices") and len(trace.get("selected_fact_indices")) or 0) for trace in traces]
    selected_passage_counts = [int(trace.get("selected_passage_indices") and len(trace.get("selected_passage_indices")) or 0) for trace in traces]
    graph_search_counts = [int(trace.get("graph_search_count", 0) or 0) for trace in traces]

    payload: Dict[str, Any] = {
        "mode": "role_aware_graph_entry_retrieval",
        "dataset": args.dataset,
        "per_query_variant": per_query_variant,
        "summary": summary,
        "variants": {
            "hipporag_v2": baseline_rows,
            "role_aware_graph_entry": selected_rows,
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
            "retrieval_top_k": int(args.retrieval_top_k),
            "num_to_retrieve": int(args.num_to_retrieve),
            "reader_top_k": int(args.reader_top_k),
            "role_fact_top_k": int(args.role_fact_top_k),
            "max_fact_seeds": int(args.max_fact_seeds),
            "max_fact_seeds_mode": args.max_fact_seeds_mode,
            "link_top_k_mode": args.link_top_k_mode,
            "role_passage_top_k": int(args.role_passage_top_k),
            "max_passage_seeds": int(args.max_passage_seeds),
            "entry_passage_node_weight": (
                float(args.entry_passage_node_weight)
                if args.entry_passage_node_weight is not None
                else None
            ),
            "entry_components": "fact_only" if args.fact_only else "fact_and_passage",
            "limit_queries": int(args.limit_queries) if args.limit_queries is not None else None,
            "chunk_size": int(args.chunk_size),
            "openie_mode": args.openie_mode,
            "qwen_disable_thinking": bool(args.qwen_disable_thinking),
        },
        "diagnostics": {
            "num_queries": len(common_qids),
            "empty_role_count": int(empty_role_count),
            "empty_fact_seed_count": int(empty_fact_seed_count),
            "missing_doc_mappings": int(missing_doc_count),
            "avg_selected_fact_seeds": average(selected_fact_counts),
            "avg_selected_passage_seeds": average(selected_passage_counts),
            "avg_graph_search_count": average(graph_search_counts),
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
