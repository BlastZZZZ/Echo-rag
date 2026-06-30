#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def corpus_docs(corpus_records: Sequence[Mapping[str, Any]]) -> List[str]:
    docs: List[str] = []
    for record in corpus_records:
        title = str(record.get("title") or "")
        text = str(record.get("text") or "")
        docs.append(f"{title}\n{text}")
    return docs


def build_doc_to_index(docs: Sequence[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    duplicates: Dict[str, List[int]] = {}
    for idx, doc in enumerate(docs):
        if doc in mapping:
            duplicates.setdefault(doc, [mapping[doc]]).append(idx)
            continue
        mapping[doc] = int(idx)
    if duplicates:
        sample = next(iter(duplicates.values()))
        raise ValueError(f"Corpus contains duplicate passage text; sample indices={sample[:5]}")
    return mapping


def load_gold_rows(per_query_report: str | Path) -> List[Mapping[str, Any]]:
    payload = load_json(per_query_report)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping payload: {per_query_report}")
    variants = payload.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError(f"Missing variants in payload: {per_query_report}")
    rows = variants.get("dataset_gold_rows")
    if not isinstance(rows, list):
        raise ValueError(f"Missing dataset_gold_rows list: {per_query_report}")
    normalized = [row for row in rows if isinstance(row, Mapping)]
    normalized.sort(key=lambda row: int(row.get("query_index", 0)))
    return normalized


def working_paths(save_dir: Path, llm_name: str, embedding_name: str) -> Tuple[Path, Path, Path]:
    llm_label = str(llm_name).replace("/", "_")
    embedding_label = str(embedding_name).replace("/", "_")
    working_dir = save_dir / f"{llm_label}_{embedding_label}"
    return (
        working_dir,
        working_dir / "chunk_embeddings" / "vdb_chunk.parquet",
        working_dir / "graph.pickle",
    )


def serialize_query_solution(solution: Any, doc_to_idx: Mapping[str, int]) -> Tuple[Dict[str, Any], int]:
    doc_indices: List[int] = []
    doc_scores: List[float] = []
    missing_docs = 0

    docs = list(getattr(solution, "docs", []) or [])
    raw_scores = getattr(solution, "doc_scores", None)
    score_list: List[float] = []
    if raw_scores is not None:
        score_list = np.asarray(raw_scores, dtype=np.float32).tolist()

    for position, doc in enumerate(docs):
        doc_idx = doc_to_idx.get(str(doc))
        if doc_idx is None:
            missing_docs += 1
            continue
        doc_indices.append(int(doc_idx))
        if position < len(score_list):
            doc_scores.append(float(score_list[position]))

    return (
        {
            "question": str(getattr(solution, "question", "") or ""),
            "doc_indices": doc_indices,
            "doc_scores": doc_scores if doc_scores else None,
        },
        missing_docs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus-json", required=True)
    parser.add_argument("--per-query-report", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--baseline-output-json", required=True)
    parser.add_argument("--llm-name", default="qwen3-8b")
    parser.add_argument("--llm-base-url", default="http://localhost:8002/v1")
    parser.add_argument("--embedding-name", default="Transformers//root/models/bge-m3")
    parser.add_argument("--embedding-base-url", default="")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--retrieval-top-k", type=int, default=200)
    parser.add_argument("--limit-queries", type=int, default=100)
    parser.add_argument("--max-index-docs", type=int, default=0)
    parser.add_argument("--force-index-from-scratch", action="store_true")
    parser.add_argument("--force-openie-from-scratch", action="store_true")
    parser.add_argument("--qwen-disable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    code_root = Path(args.code_root).resolve()
    src_root = code_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    from hipporag.HippoRAG import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    corpus_records = load_json(args.corpus_json)
    if not isinstance(corpus_records, list):
        raise ValueError(f"Corpus must be a list: {args.corpus_json}")
    docs = corpus_docs(corpus_records)
    if int(args.max_index_docs) > 0:
        docs = docs[: int(args.max_index_docs)]
    doc_to_idx = build_doc_to_index(docs)

    rows = load_gold_rows(args.per_query_report)
    if int(args.limit_queries) >= 0:
        rows = rows[: int(args.limit_queries)]
    queries = [str(row.get("question") or "") for row in rows]
    if not all(queries):
        raise ValueError("Empty question found in per-query report")

    save_dir = Path(args.save_dir).resolve()
    working_dir, chunk_parquet, graph_pickle = working_paths(
        save_dir=save_dir,
        llm_name=args.llm_name,
        embedding_name=args.embedding_name,
    )
    openie_path = save_dir / f"openie_results_ner_{str(args.llm_name).replace('/', '_')}.json"

    config_field_names = {field.name for field in fields(BaseConfig)}
    config_kwargs = {
        "save_dir": str(save_dir),
        "llm_name": args.llm_name,
        "llm_base_url": args.llm_base_url,
        "embedding_model_name": args.embedding_name,
        "embedding_base_url": args.embedding_base_url,
        "embedding_batch_size": int(args.embedding_batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "openie_mode": "online",
        "retrieval_top_k": int(args.retrieval_top_k),
        "corpus_len": len(docs),
        "force_index_from_scratch": bool(args.force_index_from_scratch),
        "force_openie_from_scratch": bool(args.force_openie_from_scratch),
        "qwen_disable_thinking": bool(args.qwen_disable_thinking),
    }
    config = BaseConfig(**{key: value for key, value in config_kwargs.items() if key in config_field_names})

    print(f"[fresh-index] dataset={args.dataset} docs={len(docs)} queries={len(queries)}", flush=True)
    print(f"[fresh-index] save_dir={save_dir}", flush=True)
    print(f"[fresh-index] working_dir={working_dir}", flush=True)

    rag = HippoRAG(
        global_config=config,
        save_dir=str(save_dir),
        llm_model_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_name,
        embedding_base_url=args.embedding_base_url,
    )

    need_index = bool(args.force_index_from_scratch) or bool(args.force_openie_from_scratch)
    need_index = need_index or not chunk_parquet.exists() or not graph_pickle.exists() or not openie_path.exists()
    if need_index:
        print("[fresh-index] indexing start", flush=True)
        rag.index(docs=docs)
        print("[fresh-index] indexing done", flush=True)
    else:
        print("[fresh-index] reusing existing index artifacts", flush=True)

    print("[fresh-index] baseline retrieval start", flush=True)
    solutions = rag.retrieve(queries=queries, num_to_retrieve=int(args.retrieval_top_k))
    if isinstance(solutions, tuple):
        solutions = solutions[0]

    query_solutions: List[Mapping[str, Any]] = []
    missing_doc_count = 0
    for row, solution in zip(rows, solutions):
        serialized, missing = serialize_query_solution(solution, doc_to_idx)
        query_solutions.append(
            {
                "query_index": int(row.get("query_index", len(query_solutions))),
                **serialized,
            }
        )
        missing_doc_count += missing

    payload = {
        "mode": "fresh_hipporag_v2_baseline_cache",
        "dataset": args.dataset,
        "query_solutions": query_solutions,
        "config": {
            "save_dir": str(save_dir),
            "llm_name": args.llm_name,
            "llm_base_url": args.llm_base_url,
            "embedding_name": args.embedding_name,
            "embedding_base_url": args.embedding_base_url,
            "retrieval_top_k": int(args.retrieval_top_k),
            "limit_queries": int(args.limit_queries),
            "max_index_docs": int(args.max_index_docs),
            "qwen_disable_thinking": bool(args.qwen_disable_thinking),
            "force_index_from_scratch": bool(args.force_index_from_scratch),
            "force_openie_from_scratch": bool(args.force_openie_from_scratch),
        },
        "diagnostics": {
            "num_queries": len(query_solutions),
            "num_docs": len(docs),
            "missing_doc_mappings": int(missing_doc_count),
            "chunk_parquet": str(chunk_parquet),
            "graph_pickle": str(graph_pickle),
            "openie_path": str(openie_path),
            "index_reused": not need_index,
        },
        "source": {
            "code_root": str(code_root),
            "corpus_json": str(Path(args.corpus_json).resolve()),
            "per_query_report": str(Path(args.per_query_report).resolve()),
        },
    }
    save_json(args.baseline_output_json, payload)
    print(f"[fresh-index] wrote baseline cache {args.baseline_output_json}", flush=True)


if __name__ == "__main__":
    main()
