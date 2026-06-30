#!/usr/bin/env python3
"""Evaluate role-conditioned candidate generation for expanded RCR.

This is the retrieval-side pilot for moving RCR upstream from post-PPR top-k
reranking.  The main Role-wise RCR path is role-only: HippoRAG v2 query-level
retrieval is used only as an evaluation baseline. Its top documents and scores
are not part of the role-conditioned method. The script compares:

* ``hipporag_v2``: original query-level HippoRAG v2 top-k;
* ``rolewise_pointwise``: role-conditioned candidates, then pointwise role score
  merge;
* ``rolewise_coverage_lambda_*``: the same candidate pool selected by marginal
  role coverage.

The script emits a per-query report that can be passed to
``rerun_qa_nothink_from_perquery.py`` for EM/F1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from evaluate_role_coverage_reranker import (
    average,
    corpus_title_to_index,
    gold_doc_indices_for_row,
    load_baseline_rank_cache,
    load_json,
    load_per_query_rows,
    load_roles,
    parse_lambda_values,
    recall_from_docs,
    save_json,
    title_for_doc,
    unique_preserve_order,
)
from role_coverage_reranker import role_coverage_value, select_role_coverage_topk
from rolewise_rcr import (
    build_rolewise_candidate_pool,
    build_rolewise_role_scores,
    parse_role_compatibility_cache,
    parse_rolewise_rank_cache,
    role_ids_from_roles,
    select_pointwise_role_merge_topk,
)


def normalize_gold_answers(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                out.extend(str(inner) for inner in item if inner is not None)
            else:
                out.append(str(item))
        return out
    return [str(value)]


def make_row(
    *,
    query_index: int,
    question: str,
    gold_answers: Sequence[str],
    gold_doc_indices: Sequence[int],
    reader_doc_indices: Sequence[int],
    corpus_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    docs = unique_preserve_order(reader_doc_indices)[:5]
    gold = unique_preserve_order(gold_doc_indices)
    return {
        "query_index": int(query_index),
        "question": str(question or ""),
        "gold_answers": list(gold_answers),
        "gold_doc_indices": gold,
        "gold_doc_titles": [title_for_doc(corpus_records, doc_idx) for doc_idx in gold],
        "reader_doc_indices_topk": docs,
        "retrieved_doc_indices_top5": docs,
        "reader_titles_topk": [title_for_doc(corpus_records, doc_idx) for doc_idx in docs],
        "predicted_answer": "",
        "ExactMatch": 0.0,
        "F1": 0.0,
    }


def _empty_stats() -> Dict[str, Any]:
    return {
        "r5_values": [],
        "changed": 0,
        "improved": 0,
        "worsened": 0,
        "gold_replacement": 0,
        "objective_values": [],
    }


def _update_stats(
    *,
    stats: Dict[str, Any],
    gold_docs: Sequence[int],
    baseline_docs: Sequence[int],
    selected_docs: Sequence[int],
    baseline_r5: float,
    selected_r5: float,
    objective_value: float,
) -> None:
    stats["r5_values"].append(float(selected_r5))
    stats["objective_values"].append(float(objective_value))
    if list(selected_docs)[: len(baseline_docs)] != list(baseline_docs):
        stats["changed"] += 1
    if selected_r5 > baseline_r5:
        stats["improved"] += 1
    elif selected_r5 < baseline_r5:
        stats["worsened"] += 1
    lost_gold = (set(gold_docs) & set(baseline_docs)) - (set(gold_docs) & set(selected_docs))
    if lost_gold:
        stats["gold_replacement"] += 1


def _summary(name: str, stats: Mapping[str, Any], baseline_values: Sequence[float]) -> Dict[str, Any]:
    selected_r5 = average(stats.get("r5_values", []) or [])
    baseline_r5 = average(baseline_values)
    n = len(baseline_values)
    return {
        "variant": name,
        "summary": {
            "num_queries": n,
            "baseline_r5": baseline_r5,
            "selected_r5": selected_r5,
            "delta_r5": selected_r5 - baseline_r5,
            "changed_query_count": int(stats.get("changed", 0)),
            "improved_query_count": int(stats.get("improved", 0)),
            "worsened_query_count": int(stats.get("worsened", 0)),
            "GoldReplacement@5": int(stats.get("gold_replacement", 0)) / float(n) if n else 0.0,
            "objective_top5": average(stats.get("objective_values", []) or []),
        },
    }


def evaluate_rolewise_rcr(
    *,
    dataset: str,
    corpus_records: Sequence[Mapping[str, Any]],
    rows_by_qid: Mapping[int, Mapping[str, Any]],
    base_rank_cache: Mapping[int, Mapping[str, Any]],
    roles_by_query: Mapping[int, Sequence[Mapping[str, Any]]],
    rolewise_rank_cache: Mapping[int, Mapping[str, Mapping[str, Any]]],
    role_compatibility_cache: Mapping[int, Mapping[str, Mapping[int, float]]] | None,
    role_top_k: int,
    max_candidates: int,
    reader_top_k: int,
    lambda_values: Sequence[float],
) -> Dict[str, Any]:
    title_to_index = corpus_title_to_index(corpus_records)
    baseline_values: List[float] = []
    candidate_pool_values: List[float] = []
    candidate_pool_sizes: List[int] = []
    role_ranking_counts: List[int] = []
    role_conditioning_failure_count = 0

    variants: Dict[str, List[Dict[str, Any]]] = {
        "hipporag_v2": [],
        "rolewise_pointwise": [],
    }
    stats: Dict[str, Dict[str, Any]] = {"rolewise_pointwise": _empty_stats()}
    for lambda_value in lambda_values:
        name = f"rolewise_coverage_lambda_{str(float(lambda_value)).replace('.', '_')}"
        variants[name] = []
        stats[name] = _empty_stats()

    for query_index in sorted(set(rows_by_qid) & set(base_rank_cache)):
        row = rows_by_qid[int(query_index)]
        question = str(row.get("question") or "")
        gold_docs = gold_doc_indices_for_row(row, title_to_index)
        gold_answers = normalize_gold_answers(row.get("gold_answers"))
        base_docs_all = unique_preserve_order(base_rank_cache[int(query_index)].get("doc_indices", []))
        if not base_docs_all:
            continue
        baseline_docs = base_docs_all[: int(reader_top_k)]
        variants["hipporag_v2"].append(
            make_row(
                query_index=int(query_index),
                question=question,
                gold_answers=gold_answers,
                gold_doc_indices=gold_docs,
                reader_doc_indices=baseline_docs,
                corpus_records=corpus_records,
            )
        )

        roles = list(roles_by_query.get(int(query_index), []) or [])
        role_rankings = rolewise_rank_cache.get(int(query_index), {}) or {}
        candidate_docs = build_rolewise_candidate_pool(
            roles=roles,
            role_rankings=role_rankings,
            role_top_k=role_top_k,
            max_candidates=max_candidates,
        )
        role_scores = build_rolewise_role_scores(
            candidate_docs=candidate_docs,
            roles=roles,
            role_rankings=role_rankings,
            role_compatibility_scores=(
                role_compatibility_cache.get(int(query_index), {})
                if role_compatibility_cache is not None
                else None
            ),
        )
        role_ids = role_ids_from_roles(roles)
        base_scores = {int(doc_idx): 0.0 for doc_idx in unique_preserve_order(candidate_docs)}

        role_conditioned_available = bool(role_ids and candidate_docs)
        if not role_conditioned_available:
            role_conditioning_failure_count += 1
        pointwise_docs = (
            select_pointwise_role_merge_topk(
                candidate_passages=candidate_docs,
                base_scores=base_scores,
                role_scores=role_scores,
                role_ids=role_ids,
                k=reader_top_k,
                base_relevance_weight=0.0,
            )
            if role_conditioned_available
            else []
        )
        variants["rolewise_pointwise"].append(
            make_row(
                query_index=int(query_index),
                question=question,
                gold_answers=gold_answers,
                gold_doc_indices=gold_docs,
                reader_doc_indices=pointwise_docs,
                corpus_records=corpus_records,
            )
        )

        if gold_docs:
            baseline_r5 = recall_from_docs(gold_docs, baseline_docs, reader_top_k)
            baseline_values.append(baseline_r5)
            candidate_pool_values.append(recall_from_docs(gold_docs, candidate_docs, max(len(candidate_docs), 1)))
            candidate_pool_sizes.append(len(candidate_docs))
            role_ranking_counts.append(len(role_rankings))
            pointwise_r5 = recall_from_docs(gold_docs, pointwise_docs, reader_top_k)
            _update_stats(
                stats=stats["rolewise_pointwise"],
                gold_docs=gold_docs,
                baseline_docs=baseline_docs,
                selected_docs=pointwise_docs,
                baseline_r5=baseline_r5,
                selected_r5=pointwise_r5,
                objective_value=role_coverage_value(
                    selected_passages=pointwise_docs,
                    base_scores=base_scores,
                    role_scores=role_scores,
                    role_ids=role_ids,
                    base_relevance_weight=0.0,
                ),
            )

        for lambda_value in lambda_values:
            name = f"rolewise_coverage_lambda_{str(float(lambda_value)).replace('.', '_')}"
            selected_docs = (
                select_role_coverage_topk(
                    candidate_passages=candidate_docs,
                    base_scores=base_scores,
                    role_scores=role_scores,
                    role_ids=role_ids,
                    k=reader_top_k,
                    base_relevance_weight=float(lambda_value),
                )
                if role_conditioned_available
                else []
            )
            variants[name].append(
                make_row(
                    query_index=int(query_index),
                    question=question,
                    gold_answers=gold_answers,
                    gold_doc_indices=gold_docs,
                    reader_doc_indices=selected_docs,
                    corpus_records=corpus_records,
                )
            )
            if gold_docs:
                selected_r5 = recall_from_docs(gold_docs, selected_docs, reader_top_k)
                _update_stats(
                    stats=stats[name],
                    gold_docs=gold_docs,
                    baseline_docs=baseline_docs,
                    selected_docs=selected_docs,
                    baseline_r5=baseline_r5,
                    selected_r5=selected_r5,
                    objective_value=role_coverage_value(
                        selected_passages=selected_docs,
                        base_scores=base_scores,
                        role_scores=role_scores,
                        role_ids=role_ids,
                        base_relevance_weight=float(lambda_value),
                    ),
                )

    results = [_summary(name, stats[name], baseline_values) for name in stats]
    return {
        "dataset": dataset,
        "mode": "rolewise_rcr_retrieval",
        "num_queries": len(variants["hipporag_v2"]),
        "config": {
            "role_top_k": int(role_top_k),
            "max_candidates": int(max_candidates),
            "reader_top_k": int(reader_top_k),
            "lambda_values": [float(value) for value in lambda_values],
            "compatibility_source": "cache" if role_compatibility_cache is not None else "rank",
        },
        "candidate_pool_summary": {
            "candidate_pool_recall": average(candidate_pool_values),
            "avg_candidate_pool_size": average(candidate_pool_sizes),
            "avg_role_rankings_per_query": average(role_ranking_counts),
            "role_conditioning_failure_count": int(role_conditioning_failure_count),
        },
        "results": results,
        "variants": variants,
    }


def write_markdown(path: str | Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Role-wise RCR Retrieval Pilot",
        "",
        "## Setup",
        "",
        "```text",
        f"dataset: {payload.get('dataset')}",
        f"mode: {payload.get('mode')}",
        f"config: {payload.get('config')}",
        "```",
        "",
        "## Retrieval Results",
        "",
        "| Variant | n | Base R@5 | Selected R@5 | dR@5 | Changed | Improved | Worsened | GoldReplacement@5 | Objective |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload.get("results", []) or []:
        summary = result.get("summary", {}) if isinstance(result, Mapping) else {}
        lines.append(
            "| {variant} | {n} | {base:.4f} | {selected:.4f} | {delta:+.4f} | {changed} | {improved} | {worsened} | {replace:.4f} | {objective:.4f} |".format(
                variant=str(result.get("variant") or ""),
                n=int(summary.get("num_queries", 0) or 0),
                base=float(summary.get("baseline_r5", 0.0) or 0.0),
                selected=float(summary.get("selected_r5", 0.0) or 0.0),
                delta=float(summary.get("delta_r5", 0.0) or 0.0),
                changed=int(summary.get("changed_query_count", 0) or 0),
                improved=int(summary.get("improved_query_count", 0) or 0),
                worsened=int(summary.get("worsened_query_count", 0) or 0),
                replace=float(summary.get("GoldReplacement@5", 0.0) or 0.0),
                objective=float(summary.get("objective_top5", 0.0) or 0.0),
            )
        )
    pool = payload.get("candidate_pool_summary", {}) or {}
    lines.extend(
        [
            "",
            "## Candidate Pool",
            "",
            "```text",
            f"candidate_pool_recall: {float(pool.get('candidate_pool_recall', 0.0) or 0.0):.4f}",
            f"avg_candidate_pool_size: {float(pool.get('avg_candidate_pool_size', 0.0) or 0.0):.2f}",
            f"avg_role_rankings_per_query: {float(pool.get('avg_role_rankings_per_query', 0.0) or 0.0):.2f}",
            f"role_conditioning_failure_count: {int(pool.get('role_conditioning_failure_count', 0) or 0)}",
            "```",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus_json", required=True)
    parser.add_argument("--per_query_report", required=True)
    parser.add_argument("--baseline_top200_cache", required=True)
    parser.add_argument("--roles_json_path", required=True)
    parser.add_argument("--rolewise_rank_cache_json", required=True)
    parser.add_argument("--role_compatibility_cache_json", default=None)
    parser.add_argument("--per_query_variant", default=None)
    parser.add_argument("--limit_queries", type=int, default=None)
    parser.add_argument("--base_score_top_k", type=int, default=200)
    parser.add_argument("--role_top_k", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=30)
    parser.add_argument("--reader_top_k", type=int, default=5)
    parser.add_argument("--lambda_values", default="0.0")
    parser.add_argument("--save_json_path", required=True)
    parser.add_argument("--save_md_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_records = load_json(args.corpus_json)
    if not isinstance(corpus_records, list):
        raise ValueError(f"Corpus must be a list: {args.corpus_json}")
    per_query_variant, rows_by_qid = load_per_query_rows(args.per_query_report, args.per_query_variant)
    base_rank_cache = load_baseline_rank_cache(args.baseline_top200_cache, args.base_score_top_k)
    roles_by_query = load_roles(args.roles_json_path)
    rolewise_rank_cache = parse_rolewise_rank_cache(load_json(args.rolewise_rank_cache_json))
    role_compatibility_cache = (
        parse_role_compatibility_cache(load_json(args.role_compatibility_cache_json))
        if args.role_compatibility_cache_json
        else None
    )

    common_qids = sorted(set(rows_by_qid) & set(base_rank_cache))
    if args.limit_queries is not None and int(args.limit_queries) >= 0:
        common_qids = common_qids[: int(args.limit_queries)]
    keep = set(common_qids)
    payload = evaluate_rolewise_rcr(
        dataset=args.dataset,
        corpus_records=corpus_records,
        rows_by_qid={qid: rows_by_qid[qid] for qid in keep},
        base_rank_cache={qid: base_rank_cache[qid] for qid in keep},
        roles_by_query={qid: roles_by_query.get(qid, []) for qid in keep},
        rolewise_rank_cache={qid: rolewise_rank_cache.get(qid, {}) for qid in keep},
        role_compatibility_cache=(
            {qid: role_compatibility_cache.get(qid, {}) for qid in keep}
            if role_compatibility_cache is not None
            else None
        ),
        role_top_k=args.role_top_k,
        max_candidates=args.max_candidates,
        reader_top_k=args.reader_top_k,
        lambda_values=parse_lambda_values(args.lambda_values),
    )
    payload["config"]["per_query_variant"] = per_query_variant
    payload["source"] = {
        "corpus_json": str(Path(args.corpus_json)),
        "per_query_report": str(Path(args.per_query_report)),
        "baseline_top200_cache": str(Path(args.baseline_top200_cache)),
        "roles_json_path": str(Path(args.roles_json_path)),
        "rolewise_rank_cache_json": str(Path(args.rolewise_rank_cache_json)),
        "role_compatibility_cache_json": str(Path(args.role_compatibility_cache_json)) if args.role_compatibility_cache_json else None,
    }
    save_json(args.save_json_path, payload)
    write_markdown(args.save_md_path, payload)


if __name__ == "__main__":
    main()
