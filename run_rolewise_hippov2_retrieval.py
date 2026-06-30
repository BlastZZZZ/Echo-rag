#!/usr/bin/env python3
"""Run HippoRAG v2 retrieval for role-conditioned RCR tasks.

This runner is intentionally thin: it does not score roles, inspect gold
labels, or select the final reader context.  It only executes the flattened
``retrieval_query`` strings produced by ``build_rolewise_retrieval_tasks.py``
against an existing HippoRAG v2 index and serializes the returned ranked
passages as corpus document indices.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_EMBEDDING_NAME = os.environ.get("EMBEDDING_NAME", "Transformers/BAAI/bge-m3")
DEFAULT_EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "http://localhost:8018/v1/embeddings")
DEFAULT_LLM_NAME = os.environ.get("LLM_NAME", "qwen3-8b-train")
DEFAULT_LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8039/v1")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalize_embedding_name(embedding_name: str, embedding_base_url: Optional[str]) -> str:
    """Match the embedding-name convention used by the existing HippoRAG runs."""
    if embedding_base_url is None:
        return str(embedding_name)
    known_prefixes = ("VLLM/", "Transformers/")
    endpoint_native_substrings = ("text-embedding",)
    if str(embedding_name).startswith(known_prefixes) or any(
        token in str(embedding_name) for token in endpoint_native_substrings
    ):
        return str(embedding_name)
    return f"VLLM/{embedding_name}"


def corpus_docs(corpus_records: Sequence[Mapping[str, Any]]) -> List[str]:
    """Return the passage strings used by HippoRAG indexing."""
    docs: List[str] = []
    for record in corpus_records:
        title = str(record.get("title") or "")
        text = str(record.get("text") or "")
        docs.append(f"{title}\n{text}")
    return docs


def build_doc_to_index(docs: Sequence[str]) -> Dict[str, int]:
    """Map passage text to corpus index, rejecting ambiguous duplicate content."""
    doc_to_idx: Dict[str, int] = {}
    duplicates: Dict[str, List[int]] = {}
    for idx, doc in enumerate(docs):
        if doc in doc_to_idx:
            duplicates.setdefault(doc, [doc_to_idx[doc]]).append(idx)
            continue
        doc_to_idx[doc] = int(idx)
    if duplicates:
        sample = next(iter(duplicates.values()))
        raise ValueError(f"Corpus contains duplicate passage text; ambiguous indices include {sample[:5]}")
    return doc_to_idx


def task_queries(task_payload: Mapping[str, Any], limit_tasks: Optional[int] = None) -> List[str]:
    tasks = task_payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("Task payload must contain a tasks list")
    if limit_tasks is not None and int(limit_tasks) >= 0:
        tasks = tasks[: int(limit_tasks)]
    queries: List[str] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        query = " ".join(str(task.get("retrieval_query") or "").split())
        if not query:
            raise ValueError(f"Missing retrieval_query for task: {task}")
        queries.append(query)
    return queries


def serialize_query_solution(solution: Any, doc_to_idx: Mapping[str, int]) -> Tuple[Dict[str, Any], int]:
    """Serialize a HippoRAG QuerySolution with corpus doc indices."""
    doc_indices: List[int] = []
    missing_docs = 0
    for doc in getattr(solution, "docs", []) or []:
        doc_idx = doc_to_idx.get(str(doc))
        if doc_idx is None:
            missing_docs += 1
            continue
        doc_indices.append(int(doc_idx))

    scores_payload = getattr(solution, "doc_scores", None)
    doc_scores: List[float] = []
    if scores_payload is not None:
        scores = np.asarray(scores_payload, dtype=np.float32).tolist()
        raw_docs = list(getattr(solution, "docs", []) or [])
        for doc, score in zip(raw_docs, scores):
            if str(doc) in doc_to_idx:
                doc_scores.append(float(score))
    if doc_scores and len(doc_scores) != len(doc_indices):
        raise ValueError("Serialized doc_scores length does not match serialized doc_indices length")

    return (
        {
            "question": str(getattr(solution, "question", "") or ""),
            "doc_indices": doc_indices,
            "doc_scores": doc_scores if doc_scores else None,
        },
        missing_docs,
    )


def build_config(args: argparse.Namespace):
    from hipporag.utils.config_utils import BaseConfig

    config_field_names = {field.name for field in fields(BaseConfig)}
    kwargs = {
        "save_dir": args.save_dir,
        "llm_base_url": args.llm_base_url,
        "llm_name": args.llm_name,
        "embedding_model_name": normalize_embedding_name(args.embedding_name, args.embedding_base_url),
        "embedding_base_url": args.embedding_base_url,
        "max_new_tokens": args.max_new_tokens,
        "force_index_from_scratch": bool(args.force_index_from_scratch),
        "force_openie_from_scratch": bool(args.force_openie_from_scratch),
        "dataset": args.dataset,
        "openie_mode": args.openie_mode,
        "retrieval_mode": "hipporag_v2",
        "retrieval_top_k": args.retrieval_top_k,
        "qa_top_k": 5,
        "max_qa_steps": 3,
        "corpus_len": args.corpus_len,
        "embedding_batch_size": args.embedding_batch_size,
        "qwen_disable_thinking": bool(args.qwen_disable_thinking),
    }
    return BaseConfig(**{key: value for key, value in kwargs.items() if key in config_field_names})


def load_existing_query_solutions(path: str | Path) -> List[Mapping[str, Any]]:
    output_path = Path(path)
    if not output_path.exists():
        return []
    payload = load_json(output_path)
    rows = payload.get("query_solutions", []) if isinstance(payload, Mapping) else []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def run_rolewise_retrieval(
    *,
    args: argparse.Namespace,
    queries: Sequence[str],
    docs: Sequence[str],
    doc_to_idx: Mapping[str, int],
) -> Dict[str, Any]:
    from hipporag.HippoRAG import HippoRAG

    config = build_config(args)
    system = HippoRAG(global_config=config)

    existing_rows = load_existing_query_solutions(args.output_json) if args.resume else []
    if existing_rows and len(existing_rows) > len(queries):
        raise ValueError(f"Existing output has {len(existing_rows)} rows for {len(queries)} queries")

    query_solutions: List[Mapping[str, Any]] = list(existing_rows)
    missing_doc_count = 0
    chunk_size = max(int(args.chunk_size), 1)
    while len(query_solutions) < len(queries):
        start = len(query_solutions)
        end = min(start + chunk_size, len(queries))
        chunk_queries = list(queries[start:end])
        chunk_solutions = system.retrieve(queries=chunk_queries, num_to_retrieve=args.num_to_retrieve)
        if isinstance(chunk_solutions, tuple):
            chunk_solutions = chunk_solutions[0]
        for solution in chunk_solutions:
            row, missing = serialize_query_solution(solution, doc_to_idx)
            query_solutions.append(row)
            missing_doc_count += missing
        payload = {
            "mode": "rolewise_hipporag_v2_retrieval",
            "dataset": args.dataset,
            "num_tasks": len(queries),
            "completed_count": len(query_solutions),
            "query_solutions": query_solutions,
            "config": {
                "save_dir": args.save_dir,
                "llm_name": args.llm_name,
                "llm_base_url": args.llm_base_url,
                "embedding_model_name": normalize_embedding_name(args.embedding_name, args.embedding_base_url),
                "embedding_base_url": args.embedding_base_url,
                "retrieval_top_k": int(args.retrieval_top_k),
                "num_to_retrieve": int(args.num_to_retrieve),
                "chunk_size": int(args.chunk_size),
                "force_index_from_scratch": bool(args.force_index_from_scratch),
                "force_openie_from_scratch": bool(args.force_openie_from_scratch),
                "openie_mode": args.openie_mode,
                "qwen_disable_thinking": bool(args.qwen_disable_thinking),
            },
            "diagnostics": {
                "corpus_docs": len(docs),
                "missing_doc_mappings": int(missing_doc_count),
            },
            "source": {
                "tasks_json": args.tasks_json,
                "corpus_json": args.corpus_json,
            },
        }
        save_json(args.output_json, payload)
    return load_json(args.output_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tasks_json", required=True)
    parser.add_argument("--corpus_json", required=True)
    parser.add_argument("--save_dir", required=True, help="Existing HippoRAG v2 dataset save_dir")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--limit_tasks", type=int, default=None)
    parser.add_argument("--retrieval_top_k", type=int, default=200)
    parser.add_argument("--num_to_retrieve", type=int, default=50)
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
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()
    task_payload = load_json(args.tasks_json)
    if not isinstance(task_payload, Mapping):
        raise ValueError(f"Task JSON must be an object: {args.tasks_json}")
    corpus_records = load_json(args.corpus_json)
    if not isinstance(corpus_records, list):
        raise ValueError(f"Corpus JSON must be a list: {args.corpus_json}")
    docs = corpus_docs(corpus_records)
    args.corpus_len = len(docs)
    queries = task_queries(task_payload, args.limit_tasks)
    run_rolewise_retrieval(args=args, queries=queries, docs=docs, doc_to_idx=build_doc_to_index(docs))


if __name__ == "__main__":
    main()
