#!/usr/bin/env python3
"""Evaluate passage-level Role-Coverage Reranking.

This is the clean experimental harness for ``role_coverage_reranker.py``.  The
method core is only the set objective over passages.  This runner supplies the
three inputs needed by that objective:

* explicit query roles R_q from an external role cache;
* base relevance b(p) from a fixed baseline retrieval cache;
* passage-role compatibility c(p,r) from a fixed scorer.

It intentionally does not build evidence units, role witnesses, graph edges,
support lanes, protected prefixes, title/alias stores, or QA reader contexts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from role_coverage_reranker import role_coverage_value, select_role_coverage_topk, unique_candidates


GENERIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "his",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus_json", required=True)
    parser.add_argument("--per_query_report", required=True)
    parser.add_argument("--baseline_top200_cache", required=True)
    parser.add_argument("--roles_json_path", required=True)
    parser.add_argument("--per_query_variant", default=None)
    parser.add_argument("--limit_queries", type=int, default=None)
    parser.add_argument("--reader_top_k", type=int, default=5)
    parser.add_argument("--candidate_top_k", type=int, default=200)
    parser.add_argument(
        "--compatibility_scorer",
        choices=(
            "lexical",
            "embedding",
            "embedding_query_role",
            "llm_verifier",
            "llm_verifier_batched",
            "llm_verifier_twostage",
        ),
        default="lexical",
    )
    parser.add_argument(
        "--passage_text_mode",
        choices=("title_body", "body", "title"),
        default="title_body",
        help="Diagnostic control for which passage fields are visible to the compatibility scorer.",
    )
    parser.add_argument(
        "--role_description_mode",
        choices=("original", "mask_question_terms", "mask_query_named_terms"),
        default="original",
        help="Diagnostic control for masking question-overlap content terms from role descriptions.",
    )
    parser.add_argument("--embedding_base_url", default="http://localhost:8018/v1/embeddings")
    parser.add_argument("--embedding_model", default="/mnt/nvme/Qwen3-Embedding-8B")
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--embedding_cache_json", default=None)
    parser.add_argument("--verifier_base_url", default="http://localhost:8041/v1")
    parser.add_argument("--verifier_model", default="qwen3-8b-train")
    parser.add_argument("--verifier_cache_json", default=None)
    parser.add_argument("--verifier_timeout", type=float, default=120.0)
    parser.add_argument("--verifier_max_passage_chars", type=int, default=1200)
    parser.add_argument("--verifier_batch_size", type=int, default=10)
    parser.add_argument("--verifier_shortlist_size", type=int, default=10)
    parser.add_argument("--verifier_proposal_head_k", type=int, default=5)
    parser.add_argument(
        "--lambda_values",
        default="0.0,0.1,0.25,0.5,1.0,2.0,4.0",
        help="Comma-separated weights for base relevance in the set objective.",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument("--skip_controls", action="store_true")
    parser.add_argument("--save_json_path", required=True)
    parser.add_argument("--save_md_path", required=True)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def int_list(values: Any) -> List[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    out: List[int] = []
    for value in values:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def unique_preserve_order(values: Iterable[int]) -> List[int]:
    return unique_candidates(list(values))


def find_byte_pattern(path: str | Path, pattern: bytes, start: int = 0, chunk_size: int = 1024 * 1024) -> int:
    if not pattern:
        return -1
    with Path(path).open("rb") as handle:
        handle.seek(max(int(start), 0))
        offset = handle.tell()
        tail = b""
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return -1
            data = tail + chunk
            position = data.find(pattern)
            if position >= 0:
                return offset - len(tail) + position
            offset += len(chunk)
            tail = data[-max(len(pattern) - 1, 0) :]


def load_query_solutions_slice(path: str | Path) -> Optional[List[Mapping[str, Any]]]:
    query_key = b'"query_solutions"'
    next_key = b'"head_traces"'
    query_pos = find_byte_pattern(path, query_key)
    if query_pos < 0:
        return None
    next_pos = find_byte_pattern(path, next_key, start=query_pos + len(query_key))
    if next_pos < 0 or next_pos <= query_pos:
        return None
    with Path(path).open("rb") as handle:
        handle.seek(query_pos)
        data = handle.read(next_pos - query_pos)
    array_start = data.find(b"[")
    array_end = data.rfind(b"]")
    if array_start < 0 or array_end <= array_start:
        return None
    solutions = json.loads(data[array_start : array_end + 1])
    if not isinstance(solutions, list):
        return None
    return [item for item in solutions if isinstance(item, Mapping)]


def load_baseline_rank_cache(path: str | Path, candidate_top_k: int) -> Dict[int, Dict[str, Any]]:
    solutions = load_query_solutions_slice(path)
    if solutions is None:
        payload = load_json(path)
        solutions = payload.get("query_solutions", []) if isinstance(payload, Mapping) else []
    out: Dict[int, Dict[str, Any]] = {}
    for query_index, item in enumerate(solutions):
        if not isinstance(item, Mapping):
            continue
        docs = unique_preserve_order(int_list(item.get("doc_indices", [])))[: int(candidate_top_k)]
        raw_scores = item.get("doc_scores", [])
        score_map: Dict[int, float] = {}
        if isinstance(raw_scores, Sequence) and not isinstance(raw_scores, (str, bytes)):
            for doc_idx, score in zip(int_list(item.get("doc_indices", [])), raw_scores):
                if int(doc_idx) in score_map:
                    continue
                try:
                    score_map[int(doc_idx)] = float(score)
                except (TypeError, ValueError):
                    score_map[int(doc_idx)] = 0.0
        out[int(query_index)] = {"doc_indices": docs, "base_scores": normalize_base_scores(docs, score_map)}
    return out


def normalize_base_scores(candidate_docs: Sequence[int], score_map: Mapping[int, float]) -> Dict[int, float]:
    candidates = unique_preserve_order(candidate_docs)
    if not candidates:
        return {}
    values = [float(score_map.get(int(doc_idx), 0.0)) for doc_idx in candidates]
    if any(value > 0.0 for value in values):
        max_value = max(values)
        min_value = min(values)
        if max_value > min_value:
            return {int(doc_idx): (float(score_map.get(int(doc_idx), 0.0)) - min_value) / (max_value - min_value) for doc_idx in candidates}
        return {int(doc_idx): 1.0 for doc_idx in candidates}
    denom = float(max(len(candidates) - 1, 1))
    return {int(doc_idx): 1.0 - (rank / denom) for rank, doc_idx in enumerate(candidates)}


def rows_by_query(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    for offset, row in enumerate(rows):
        out[int(row.get("query_index", offset))] = row
    return out


def load_per_query_rows(path: str | Path, variant_name: Optional[str] = None) -> Tuple[str, Dict[int, Mapping[str, Any]]]:
    report = load_json(path)
    variants = report.get("variants", {}) if isinstance(report, Mapping) else {}
    if not isinstance(variants, Mapping) or not variants:
        raise ValueError(f"Missing variants object in {path}")
    if variant_name:
        if variant_name not in variants:
            raise KeyError(f"Variant {variant_name!r} not found in {path}")
        rows = variants[variant_name]
        if not isinstance(rows, list):
            raise ValueError(f"Variant {variant_name!r} is not a per-query row list")
        return str(variant_name), rows_by_query(rows)
    row_variants = [(str(name), rows) for name, rows in variants.items() if isinstance(rows, list)]
    if len(row_variants) != 1:
        raise ValueError(f"Cannot infer per-query variant from {list(variants)}")
    name, rows = row_variants[0]
    return name, rows_by_query(rows)


def load_roles(path: str | Path) -> Dict[int, List[Dict[str, Any]]]:
    payload = load_json(path)
    if isinstance(payload, Mapping) and isinstance(payload.get("roles_by_query"), Mapping):
        raw_roles = payload.get("roles_by_query", {})
    elif isinstance(payload, Mapping):
        raw_roles = payload
    else:
        raise ValueError(f"Unsupported role cache format: {path}")

    out: Dict[int, List[Dict[str, str]]] = {}
    for key, roles in raw_roles.items():
        try:
            query_index = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)):
            continue
        parsed: List[Dict[str, Any]] = []
        for offset, role in enumerate(roles):
            if not isinstance(role, Mapping):
                continue
            description = " ".join(str(role.get("description") or "").split())
            retrieval_text = " ".join(str(role.get("retrieval_text") or role.get("retrieval_query_text") or "").split())
            provenance_text = " ".join(str(role.get("provenance_text") or role.get("support_description") or "").split())
            query_text = " ".join(str(role.get("query") or "").split())
            if not (description or retrieval_text or provenance_text or query_text):
                continue
            parsed.append(
                {
                    "role_id": str(role.get("role_id") or f"r{offset}"),
                    "role_type": str(role.get("role_type") or role.get("support_function") or "role"),
                    "support_function": str(role.get("support_function") or role.get("role_type") or "role"),
                    "description": description or provenance_text or retrieval_text or query_text,
                    "retrieval_text": retrieval_text or query_text or description or provenance_text,
                    "provenance_text": provenance_text or description or retrieval_text or query_text,
                }
            )
        if parsed:
            out[int(query_index)] = parsed
    return out


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(value: Any) -> List[str]:
    return [
        token
        for token in normalize_text(value).split()
        if len(token) >= 3 and token not in GENERIC_STOPWORDS
    ]


def query_named_terms(query: str) -> Set[str]:
    text = str(query or "")
    terms: Set[str] = set()
    for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', text):
        terms.update(content_tokens(match.group(1) or match.group(2) or ""))
    for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9.&'-]*|\d+[A-Za-z0-9.&'-]*)(?:\s+(?:[A-Z][A-Za-z0-9.&'-]*|\d+[A-Za-z0-9.&'-]*))*", text):
        terms.update(content_tokens(match.group(0)))
    return terms


def role_description_text(role_description: str, query: str, mode: str = "original") -> str:
    if mode == "original":
        return str(role_description or "")
    if mode == "mask_question_terms":
        masked_terms = set(content_tokens(query))
    elif mode == "mask_query_named_terms":
        masked_terms = query_named_terms(query)
    else:
        raise ValueError(f"Unsupported role description mode: {mode}")
    if not masked_terms:
        return str(role_description or "")

    def keep_or_mask(match: re.Match[str]) -> str:
        token = normalize_text(match.group(0))
        return " " if token in masked_terms else match.group(0)

    masked = re.sub(r"[A-Za-z0-9]+", keep_or_mask, str(role_description or ""))
    return re.sub(r"\s+", " ", masked).strip()


def passage_text(corpus_records: Sequence[Mapping[str, Any]], doc_index: int, mode: str = "title_body") -> str:
    if int(doc_index) < 0 or int(doc_index) >= len(corpus_records):
        return ""
    record = corpus_records[int(doc_index)]
    title = str(record.get("title") or "").strip()
    body = str(record.get("text") or record.get("content") or "").strip()
    if mode == "title":
        return title
    if mode == "body":
        return body
    if mode != "title_body":
        raise ValueError(f"Unsupported passage text mode: {mode}")
    return "\n".join(part for part in (title, body) if part)


def title_for_doc(corpus_records: Sequence[Mapping[str, Any]], doc_index: int) -> str:
    if int(doc_index) < 0 or int(doc_index) >= len(corpus_records):
        return f"doc#{int(doc_index)}"
    return str(corpus_records[int(doc_index)].get("title") or f"doc#{int(doc_index)}")


def normalize_title(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def corpus_title_to_index(corpus_records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for index, record in enumerate(corpus_records):
        key = normalize_title(record.get("title"))
        if key and key not in out:
            out[key] = int(index)
    return out


def gold_doc_indices_for_row(row: Mapping[str, Any], title_to_index: Mapping[str, int]) -> List[int]:
    indices = int_list(row.get("gold_doc_indices", []))
    if indices:
        return unique_preserve_order(indices)
    titles = row.get("gold_doc_titles", [])
    if not isinstance(titles, Sequence) or isinstance(titles, (str, bytes)):
        return []
    resolved: List[int] = []
    for title in titles:
        key = normalize_title(title)
        if key in title_to_index:
            resolved.append(int(title_to_index[key]))
    return unique_preserve_order(resolved)


def lexical_compatibility(passage: str, role_description: str) -> float:
    passage_tokens = set(content_tokens(passage))
    role_tokens = set(content_tokens(role_description))
    if not passage_tokens or not role_tokens:
        return 0.0
    return len(passage_tokens & role_tokens) / float(len(role_tokens))


def load_embedding_cache(path: Optional[str]) -> Dict[str, List[float]]:
    if not path or not Path(path).exists():
        return {}
    payload = load_json(path)
    vectors = payload.get("vectors", payload) if isinstance(payload, Mapping) else {}
    if not isinstance(vectors, Mapping):
        return {}
    return {str(key): [float(value) for value in vector] for key, vector in vectors.items() if isinstance(vector, list)}


def save_embedding_cache(path: Optional[str], cache: Mapping[str, Sequence[float]]) -> None:
    if path:
        save_json(path, {"vectors": {key: list(value) for key, value in cache.items()}})


def load_verifier_cache(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not path or not Path(path).exists():
        return {}
    payload = load_json(path)
    scores = payload.get("scores", payload) if isinstance(payload, Mapping) else {}
    if not isinstance(scores, Mapping):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for key, value in scores.items():
        if isinstance(value, Mapping):
            out[str(key)] = {str(role_id): float(score) for role_id, score in value.items()}
    return out


def save_verifier_cache(path: Optional[str], cache: Mapping[str, Mapping[str, float]]) -> None:
    if path:
        save_json(path, {"scores": {key: dict(value) for key, value in cache.items()}})


def embedding_cache_key(model: str, text: str) -> str:
    return f"{model}\n{text}"


def verifier_cache_key(model: str, query: str, passage: str, roles: Sequence[Mapping[str, str]]) -> str:
    role_payload = [
        {"role_id": str(role.get("role_id") or ""), "description": str(role.get("description") or "")}
        for role in roles
    ]
    return json.dumps(
        {
            "model": str(model),
            "query": str(query or "").strip(),
            "passage": str(passage or "").strip(),
            "roles": role_payload,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def extract_json_object(text: str) -> Mapping[str, Any]:
    stripped = str(text or "").strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, Mapping):
            return parsed
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for start, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return parsed
    raise ValueError(f"No JSON object found in verifier output: {stripped[:200]}")


def request_embeddings(
    *,
    endpoint: str,
    model: str,
    texts: Sequence[str],
    timeout: float = 120.0,
) -> List[List[float]]:
    body = {"model": model, "input": list(texts)}
    request = urllib.request.Request(
        str(endpoint),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(timeout)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError("Embedding API returned no data list")
    ordered = sorted(data, key=lambda row: int(row.get("index", 0)))
    return [[float(value) for value in row.get("embedding", [])] for row in ordered]


VERIFIER_SYSTEM_PROMPT = """You judge passage support for evidence roles in multi-hop QA.

Use only the question, role descriptions, and passage. Do not answer the question.
Score each role independently:
0 = the passage does not help satisfy this role;
1 = the passage is weakly or partially relevant to this role;
2 = the passage directly helps satisfy this role.

Return only valid JSON:
{"scores":[{"role_id":"r0","score":0|1|2}]}
"""


BATCH_VERIFIER_SYSTEM_PROMPT = """You judge passage support for evidence roles in multi-hop QA.

Use only the question, role descriptions, and passages. Do not answer the question.
Score each passage-role pair independently:
0 = the passage does not help satisfy this role;
1 = the passage is weakly or partially relevant to this role;
2 = the passage directly helps satisfy this role.
Include every passage-role pair. Do not omit zero scores.

Return only valid JSON:
{"scores":[{"passage_id":"p0","role_id":"r0","score":0|1|2}]}
"""


def normalized_verifier_score(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 2.0)) / 2.0
    except (TypeError, ValueError):
        return 0.0


def request_verifier_scores(
    *,
    endpoint: str,
    model: str,
    query: str,
    passage: str,
    roles: Sequence[Mapping[str, str]],
    timeout: float,
) -> Dict[str, float]:
    role_lines = [
        {"role_id": str(role.get("role_id") or ""), "description": str(role.get("description") or "")}
        for role in roles
    ]
    body = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": str(query or ""),
                        "roles": role_lines,
                        "passage": str(passage or ""),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    request = urllib.request.Request(
        str(endpoint).rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(timeout)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = str(payload["choices"][0]["message"]["content"])
    try:
        parsed = extract_json_object(content)
    except (ValueError, json.JSONDecodeError):
        return {}
    raw_scores = parsed.get("scores", [])
    if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, (str, bytes)):
        return {}
    out: Dict[str, float] = {}
    for item in raw_scores:
        if not isinstance(item, Mapping):
            continue
        role_id = str(item.get("role_id") or "")
        if not role_id:
            continue
        out[role_id] = normalized_verifier_score(item.get("score", 0.0))
    return out


def request_batched_verifier_scores(
    *,
    endpoint: str,
    model: str,
    query: str,
    passages: Mapping[int, str],
    roles: Sequence[Mapping[str, str]],
    timeout: float,
) -> Dict[int, Dict[str, float]]:
    role_lines = [
        {"role_id": str(role.get("role_id") or ""), "description": str(role.get("description") or "")}
        for role in roles
    ]
    ordered_passages = list(passages.items())
    passage_id_to_doc = {f"p{offset}": int(doc_idx) for offset, (doc_idx, _) in enumerate(ordered_passages)}
    passage_lines = [
        {"passage_id": passage_id, "text": str(passages[doc_idx] or "")}
        for passage_id, doc_idx in passage_id_to_doc.items()
    ]
    body = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": BATCH_VERIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": str(query or ""),
                        "roles": role_lines,
                        "passages": passage_lines,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    request = urllib.request.Request(
        str(endpoint).rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(timeout)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = str(payload["choices"][0]["message"]["content"])
    try:
        parsed = extract_json_object(content)
    except (ValueError, json.JSONDecodeError):
        return {}
    raw_scores = parsed.get("scores", [])
    if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, (str, bytes)):
        return {}
    out: Dict[int, Dict[str, float]] = {}
    for item in raw_scores:
        if not isinstance(item, Mapping):
            continue
        passage_id = str(item.get("passage_id") or "")
        if passage_id in passage_id_to_doc:
            doc_idx = passage_id_to_doc[passage_id]
        else:
            try:
                doc_idx = int(passage_id)
            except ValueError:
                continue
        if doc_idx not in passages:
            continue
        role_id = str(item.get("role_id") or "")
        if not role_id:
            continue
        out.setdefault(doc_idx, {})[role_id] = normalized_verifier_score(item.get("score", 0.0))
    return out


def select_verifier_shortlist(
    *,
    candidate_docs: Sequence[int],
    roles: Sequence[Mapping[str, str]],
    role_descriptions: Mapping[str, str],
    corpus_records: Sequence[Mapping[str, Any]],
    passage_text_mode: str,
    shortlist_size: int,
    proposal_head_k: int,
) -> List[int]:
    candidates = unique_preserve_order(candidate_docs)
    budget = max(int(shortlist_size), 0)
    if budget <= 0:
        return []
    selected: List[int] = candidates[: min(max(int(proposal_head_k), 0), budget)]
    selected_set = set(selected)
    role_ids = [str(role["role_id"]) for role in roles]

    def proposal_key(doc_idx: int) -> Tuple[float, int]:
        text = passage_text(corpus_records, int(doc_idx), passage_text_mode)
        best_role_score = max(
            (lexical_compatibility(text, role_descriptions.get(role_id, "")) for role_id in role_ids),
            default=0.0,
        )
        return best_role_score, -candidates.index(int(doc_idx))

    remaining = [doc_idx for doc_idx in candidates if int(doc_idx) not in selected_set]
    for doc_idx in sorted(remaining, key=proposal_key, reverse=True):
        if len(selected) >= budget:
            break
        selected.append(int(doc_idx))
        selected_set.add(int(doc_idx))
    return selected


def get_embeddings(
    *,
    texts: Sequence[str],
    cache: Dict[str, List[float]],
    endpoint: str,
    model: str,
    batch_size: int,
) -> Dict[str, List[float]]:
    unique_texts: List[str] = []
    seen: Set[str] = set()
    for text in texts:
        normalized = str(text or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_texts.append(normalized)
    missing = [text for text in unique_texts if embedding_cache_key(model, text) not in cache]
    for start in range(0, len(missing), max(int(batch_size), 1)):
        batch = missing[start : start + max(int(batch_size), 1)]
        vectors = request_embeddings(endpoint=endpoint, model=model, texts=batch)
        if len(vectors) != len(batch):
            raise ValueError(f"Embedding API returned {len(vectors)} vectors for {len(batch)} texts")
        for text, vector in zip(batch, vectors):
            cache[embedding_cache_key(model, text)] = vector
    return {text: cache[embedding_cache_key(model, text)] for text in unique_texts}


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def build_role_scores(
    *,
    query: str,
    candidate_docs: Sequence[int],
    roles: Sequence[Mapping[str, str]],
    corpus_records: Sequence[Mapping[str, Any]],
    scorer: str,
    embedding_cache: Dict[str, List[float]],
    embedding_endpoint: str,
    embedding_model: str,
    embedding_batch_size: int,
    passage_text_mode: str = "title_body",
    role_description_mode: str = "original",
    verifier_cache: Optional[Dict[str, Dict[str, float]]] = None,
    verifier_endpoint: str = "",
    verifier_model: str = "",
    verifier_timeout: float = 120.0,
    verifier_max_passage_chars: int = 1200,
    verifier_batch_size: int = 10,
    verifier_shortlist_size: int = 10,
    verifier_proposal_head_k: int = 5,
) -> Dict[int, Dict[str, float]]:
    role_ids = [str(role["role_id"]) for role in roles]
    role_descriptions = {
        str(role["role_id"]): role_description_text(str(role["description"]), query, role_description_mode)
        for role in roles
    }
    role_scores: Dict[int, Dict[str, float]] = {}
    if scorer == "lexical":
        for doc_idx in candidate_docs:
            text = passage_text(corpus_records, int(doc_idx), passage_text_mode)
            role_scores[int(doc_idx)] = {
                role_id: lexical_compatibility(text, role_descriptions[role_id]) for role_id in role_ids
            }
        return role_scores
    if scorer in {"llm_verifier", "llm_verifier_batched", "llm_verifier_twostage"}:
        rewritten_roles = [
            {**dict(role), "description": role_descriptions[str(role["role_id"])]}
            for role in roles
        ]
        cache = verifier_cache if verifier_cache is not None else {}
        endpoint = str(verifier_endpoint or "").strip()
        if not endpoint:
            raise ValueError(f"{scorer} requires verifier_endpoint")
        max_chars = max(int(verifier_max_passage_chars), 0)
        passage_texts = {
            int(doc_idx): passage_text(corpus_records, int(doc_idx), passage_text_mode)[:max_chars]
            for doc_idx in candidate_docs
        }
        if scorer == "llm_verifier_twostage":
            shortlist = set(
                select_verifier_shortlist(
                    candidate_docs=candidate_docs,
                    roles=roles,
                    role_descriptions=role_descriptions,
                    corpus_records=corpus_records,
                    passage_text_mode=passage_text_mode,
                    shortlist_size=verifier_shortlist_size,
                    proposal_head_k=verifier_proposal_head_k,
                )
            )
            for doc_idx in candidate_docs:
                if int(doc_idx) not in shortlist:
                    role_scores[int(doc_idx)] = {role_id: 0.0 for role_id in role_ids}
                    continue
                text = passage_texts[int(doc_idx)]
                key = verifier_cache_key(verifier_model, query, text, rewritten_roles)
                if key not in cache:
                    cache[key] = request_verifier_scores(
                        endpoint=endpoint,
                        model=verifier_model,
                        query=query,
                        passage=text,
                        roles=rewritten_roles,
                        timeout=verifier_timeout,
                    )
                role_scores[int(doc_idx)] = {
                    role_id: float((cache.get(key, {}) or {}).get(role_id, 0.0)) for role_id in role_ids
                }
            return role_scores
        if scorer == "llm_verifier_batched":
            missing: Dict[int, str] = {}
            keys: Dict[int, str] = {}
            for doc_idx, text in passage_texts.items():
                key = verifier_cache_key(verifier_model, query, text, rewritten_roles)
                keys[int(doc_idx)] = key
                if key not in cache:
                    missing[int(doc_idx)] = text
            if missing:
                missing_items = list(missing.items())
                batch_size = max(int(verifier_batch_size), 1)
                for start in range(0, len(missing_items), batch_size):
                    batch = dict(missing_items[start : start + batch_size])
                    batch_scores = request_batched_verifier_scores(
                        endpoint=endpoint,
                        model=verifier_model,
                        query=query,
                        passages=batch,
                        roles=rewritten_roles,
                        timeout=verifier_timeout,
                    )
                    for doc_idx in batch:
                        cache[keys[int(doc_idx)]] = batch_scores.get(int(doc_idx), {})
            for doc_idx in candidate_docs:
                key = keys[int(doc_idx)]
                role_scores[int(doc_idx)] = {
                    role_id: float((cache.get(key, {}) or {}).get(role_id, 0.0)) for role_id in role_ids
                }
            return role_scores
        for doc_idx in candidate_docs:
            text = passage_texts[int(doc_idx)]
            key = verifier_cache_key(verifier_model, query, text, rewritten_roles)
            if key not in cache:
                cache[key] = request_verifier_scores(
                    endpoint=endpoint,
                    model=verifier_model,
                    query=query,
                    passage=text,
                    roles=rewritten_roles,
                    timeout=verifier_timeout,
                )
            role_scores[int(doc_idx)] = {
                role_id: float((cache.get(key, {}) or {}).get(role_id, 0.0)) for role_id in role_ids
            }
        return role_scores
    if scorer not in {"embedding", "embedding_query_role"}:
        raise ValueError(f"Unsupported compatibility scorer: {scorer}")

    passage_texts = {
        int(doc_idx): passage_text(corpus_records, int(doc_idx), passage_text_mode) for doc_idx in candidate_docs
    }
    if scorer == "embedding_query_role":
        role_embedding_texts = {
            role_id: f"Question: {str(query).strip()}\nRole: {role_descriptions[role_id]}"
            for role_id in role_ids
        }
    else:
        role_embedding_texts = role_descriptions
    vectors = get_embeddings(
        texts=list(passage_texts.values()) + list(role_embedding_texts.values()),
        cache=embedding_cache,
        endpoint=embedding_endpoint,
        model=embedding_model,
        batch_size=embedding_batch_size,
    )
    for doc_idx, text in passage_texts.items():
        passage_vector = vectors.get(str(text).strip(), [])
        role_scores[int(doc_idx)] = {}
        for role_id in role_ids:
            role_vector = vectors.get(str(role_embedding_texts[role_id]).strip(), [])
            role_scores[int(doc_idx)][role_id] = max(cosine_similarity(passage_vector, role_vector), 0.0)
    return role_scores


def pointwise_role_score_topk(
    *,
    candidate_docs: Sequence[int],
    role_scores: Mapping[int, Mapping[str, float]],
    base_scores: Mapping[int, float],
    role_ids: Sequence[str],
    k: int,
) -> List[int]:
    candidates = unique_preserve_order(candidate_docs)
    rank = {doc_idx: offset for offset, doc_idx in enumerate(candidates)}

    def key(doc_idx: int) -> Tuple[float, float, int]:
        role_sum = sum(float((role_scores.get(int(doc_idx), {}) or {}).get(str(role_id), 0.0)) for role_id in role_ids)
        return role_sum, float(base_scores.get(int(doc_idx), 0.0)), -int(rank.get(int(doc_idx), 10**9))

    return sorted(candidates, key=key, reverse=True)[: max(int(k), 0)]


def role_coverage_rate(
    *,
    selected_docs: Sequence[int],
    role_scores: Mapping[int, Mapping[str, float]],
    role_ids: Sequence[str],
) -> float:
    if not role_ids:
        return 0.0
    total = 0.0
    selected = unique_preserve_order(selected_docs)
    for role_id in role_ids:
        total += max(float((role_scores.get(int(doc_idx), {}) or {}).get(str(role_id), 0.0)) for doc_idx in selected) if selected else 0.0
    return total / float(len(role_ids))


def recall_from_docs(gold_docs: Sequence[int], retrieved_docs: Sequence[int], k: int) -> float:
    gold = set(int_list(gold_docs))
    if not gold:
        return 0.0
    top_docs = set(int_list(retrieved_docs[: int(k)]))
    return len(gold & top_docs) / float(len(gold))


def average(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def parse_lambda_values(value: str) -> List[float]:
    out: List[float] = []
    for part in str(value or "").split(","):
        text = part.strip()
        if text:
            out.append(float(text))
    return out


def lambda_variant_name(value: float) -> str:
    return f"role_coverage_lambda_{str(float(value)).replace('.', '_')}"


def shuffled_roles_by_query(
    roles_by_query: Mapping[int, Sequence[Mapping[str, str]]],
    *,
    seed: int,
) -> Dict[int, Sequence[Mapping[str, str]]]:
    query_ids = sorted(roles_by_query)
    shuffled = list(query_ids)
    rng = random.Random(int(seed))
    if len(shuffled) > 1:
        rng.shuffle(shuffled)
        if all(left == right for left, right in zip(query_ids, shuffled)):
            shuffled = shuffled[1:] + shuffled[:1]
    return {query_id: roles_by_query[shuffled_id] for query_id, shuffled_id in zip(query_ids, shuffled)}


def evaluate_variant(
    *,
    variant_name: str,
    rows_by_qid: Mapping[int, Mapping[str, Any]],
    rank_cache: Mapping[int, Mapping[str, Any]],
    roles_by_query: Mapping[int, Sequence[Mapping[str, str]]],
    corpus_records: Sequence[Mapping[str, Any]],
    title_to_index: Mapping[str, int],
    reader_top_k: int,
    max_examples: int,
    compatibility_scorer: str,
    embedding_cache: Dict[str, List[float]],
    embedding_endpoint: str,
    embedding_model: str,
    embedding_batch_size: int,
    verifier_cache: Dict[str, Dict[str, float]],
    verifier_endpoint: str,
    verifier_model: str,
    verifier_timeout: float,
    verifier_max_passage_chars: int,
    verifier_batch_size: int,
    verifier_shortlist_size: int,
    verifier_proposal_head_k: int,
    passage_text_mode: str,
    role_description_mode: str,
    lambda_value: Optional[float] = None,
) -> Dict[str, Any]:
    baseline_r5_values: List[float] = []
    selected_r5_values: List[float] = []
    baseline_r20_values: List[float] = []
    selected_r20_values: List[float] = []
    coverage_values: List[float] = []
    objective_values: List[float] = []
    changed_count = 0
    improved_count = 0
    worsened_count = 0
    gold_replacement_count = 0
    regret_values: List[float] = []
    recovery_values: List[float] = []
    examples: List[Dict[str, Any]] = []

    for query_index in sorted(set(rows_by_qid) & set(rank_cache) & set(roles_by_query)):
        row = rows_by_qid[int(query_index)]
        gold_docs = gold_doc_indices_for_row(row, title_to_index)
        if not gold_docs:
            continue
        candidate_docs = unique_preserve_order(int_list(rank_cache[int(query_index)].get("doc_indices", [])))
        if not candidate_docs:
            continue
        base_scores = dict(rank_cache[int(query_index)].get("base_scores", {}) or {})
        roles = list(roles_by_query.get(int(query_index), []) or [])
        role_ids = [str(role["role_id"]) for role in roles]
        if not role_ids:
            continue
        role_scores = build_role_scores(
            query=str(row.get("question") or ""),
            candidate_docs=candidate_docs,
            roles=roles,
            corpus_records=corpus_records,
            scorer=compatibility_scorer,
            embedding_cache=embedding_cache,
            embedding_endpoint=embedding_endpoint,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            passage_text_mode=passage_text_mode,
            role_description_mode=role_description_mode,
            verifier_cache=verifier_cache,
            verifier_endpoint=verifier_endpoint,
            verifier_model=verifier_model,
            verifier_timeout=verifier_timeout,
            verifier_max_passage_chars=verifier_max_passage_chars,
            verifier_batch_size=verifier_batch_size,
            verifier_shortlist_size=verifier_shortlist_size,
            verifier_proposal_head_k=verifier_proposal_head_k,
        )

        baseline_top5 = candidate_docs[: int(reader_top_k)]
        if variant_name == "base_top5":
            selected = baseline_top5
            effective_lambda = 1.0
        elif variant_name == "pointwise_role_score":
            selected = pointwise_role_score_topk(
                candidate_docs=candidate_docs,
                role_scores=role_scores,
                base_scores=base_scores,
                role_ids=role_ids,
                k=reader_top_k,
            )
            effective_lambda = 0.0
        else:
            effective_lambda = float(lambda_value if lambda_value is not None else 1.0)
            selected = select_role_coverage_topk(
                candidate_passages=candidate_docs,
                base_scores=base_scores,
                role_scores=role_scores,
                role_ids=role_ids,
                k=reader_top_k,
                base_relevance_weight=effective_lambda,
            )

        baseline_r5 = recall_from_docs(gold_docs, baseline_top5, reader_top_k)
        selected_r5 = recall_from_docs(gold_docs, selected, reader_top_k)
        baseline_r20 = recall_from_docs(gold_docs, candidate_docs, min(20, len(candidate_docs)))
        selected_augmented = unique_preserve_order(list(selected) + list(candidate_docs))
        selected_r20 = recall_from_docs(gold_docs, selected_augmented, min(20, len(selected_augmented)))
        baseline_r5_values.append(baseline_r5)
        selected_r5_values.append(selected_r5)
        baseline_r20_values.append(baseline_r20)
        selected_r20_values.append(selected_r20)
        coverage_values.append(role_coverage_rate(selected_docs=selected, role_scores=role_scores, role_ids=role_ids))
        objective_values.append(
            role_coverage_value(
                selected_passages=selected,
                base_scores=base_scores,
                role_scores=role_scores,
                role_ids=role_ids,
                base_relevance_weight=effective_lambda,
            )
        )
        if selected[: int(reader_top_k)] != baseline_top5:
            changed_count += 1
        if selected_r5 > baseline_r5:
            improved_count += 1
        elif selected_r5 < baseline_r5:
            worsened_count += 1
        lost_gold = (set(gold_docs) & set(baseline_top5)) - (set(gold_docs) & set(selected))
        if lost_gold:
            gold_replacement_count += 1
        regret_values.append(max(baseline_r5 - selected_r5, 0.0))
        recovery_values.append(max(selected_r5 - baseline_r5, 0.0))

        if len(examples) < int(max_examples) and selected_r5 != baseline_r5:
            examples.append(
                {
                    "query_index": int(query_index),
                    "question": str(row.get("question") or ""),
                    "baseline_r5": baseline_r5,
                    "selected_r5": selected_r5,
                    "delta_r5": selected_r5 - baseline_r5,
                    "gold_doc_indices": gold_docs,
                    "gold_doc_titles": [title_for_doc(corpus_records, doc_idx) for doc_idx in gold_docs],
                    "baseline_top5": baseline_top5,
                    "baseline_titles": [title_for_doc(corpus_records, doc_idx) for doc_idx in baseline_top5],
                    "selected_top5": selected,
                    "selected_titles": [title_for_doc(corpus_records, doc_idx) for doc_idx in selected],
                    "roles": roles,
                    "role_coverage_rate": role_coverage_rate(selected_docs=selected, role_scores=role_scores, role_ids=role_ids),
                }
            )

    n = len(selected_r5_values)
    return {
        "variant": variant_name,
        "summary": {
            "num_queries": n,
            "baseline_r5": average(baseline_r5_values),
            "selected_r5": average(selected_r5_values),
            "delta_r5": average(selected_r5_values) - average(baseline_r5_values),
            "baseline_r20": average(baseline_r20_values),
            "selected_augmented_r20": average(selected_r20_values),
            "delta_augmented_r20": average(selected_r20_values) - average(baseline_r20_values),
            "role_coverage_top5": average(coverage_values),
            "objective_value": average(objective_values),
            "changed_query_count": changed_count,
            "queries_improved": improved_count,
            "queries_worsened": worsened_count,
            "queries_same": n - improved_count - worsened_count,
            "GoldReplacement@5": gold_replacement_count / float(n) if n else 0.0,
            "EvidencePerturbationRegret@5": average(regret_values),
            "EvidenceRecoveryGain@5": average(recovery_values),
        },
        "examples": examples,
    }


def write_markdown(path: str | Path, payload: Mapping[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Role-Coverage Reranking Retrieval Pilot")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```text")
    config = payload.get("config", {}) if isinstance(payload.get("config"), Mapping) else {}
    for key in sorted(config):
        lines.append(f"{key}: {config[key]}")
    lines.append("```")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Variant | R@5 | Delta R@5 | RoleCoverage@5 | Changed | Improved | Worsened | GoldReplacement@5 | Regret@5 | Recovery@5 |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in payload.get("results", []) or []:
        summary = result.get("summary", {}) if isinstance(result, Mapping) else {}
        lines.append(
            "| {variant} | {r5:.4f} | {dr5:+.4f} | {cov:.4f} | {changed} | {improved} | {worsened} | {replacement:.4f} | {regret:.4f} | {recovery:.4f} |".format(
                variant=str(result.get("variant", "")),
                r5=float(summary.get("selected_r5", 0.0) or 0.0),
                dr5=float(summary.get("delta_r5", 0.0) or 0.0),
                cov=float(summary.get("role_coverage_top5", 0.0) or 0.0),
                changed=int(summary.get("changed_query_count", 0) or 0),
                improved=int(summary.get("queries_improved", 0) or 0),
                worsened=int(summary.get("queries_worsened", 0) or 0),
                replacement=float(summary.get("GoldReplacement@5", 0.0) or 0.0),
                regret=float(summary.get("EvidencePerturbationRegret@5", 0.0) or 0.0),
                recovery=float(summary.get("EvidenceRecoveryGain@5", 0.0) or 0.0),
            )
        )
    lines.append("")
    lines.append("## Cleanliness Contract")
    lines.append("")
    lines.append("```text")
    for key, value in (payload.get("cleanliness_contract", {}) or {}).items():
        lines.append(f"{key}: {value}")
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation Guard")
    lines.append("")
    lines.append(
        "This run evaluates passage-level role-coverage reranking only. Positive or negative results should be "
        "attributed to role decomposition, passage-role compatibility, or the base-relevance lambda, not to graph "
        "construction, evidence-unit witnesses, support lanes, protected-prefix assembly, or QA reader behavior."
    )
    lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    corpus_records = load_json(args.corpus_json)
    if not isinstance(corpus_records, list):
        raise ValueError(f"Corpus must be a list: {args.corpus_json}")
    title_to_index = corpus_title_to_index(corpus_records)
    per_query_variant, rows = load_per_query_rows(args.per_query_report, args.per_query_variant)
    rank_cache = load_baseline_rank_cache(args.baseline_top200_cache, args.candidate_top_k)
    roles_by_query = load_roles(args.roles_json_path)
    common_qids = sorted(set(rows) & set(rank_cache) & set(roles_by_query))
    if args.limit_queries is not None and int(args.limit_queries) >= 0:
        common_qids = common_qids[: int(args.limit_queries)]
    keep = set(common_qids)
    rows = {qid: rows[qid] for qid in keep}
    rank_cache = {qid: rank_cache[qid] for qid in keep}
    roles_by_query = {qid: roles_by_query[qid] for qid in keep}

    embedding_cache = load_embedding_cache(args.embedding_cache_json)
    verifier_cache = load_verifier_cache(args.verifier_cache_json)
    verifier_cache_entries_before = len(verifier_cache)
    shuffled_roles = shuffled_roles_by_query(roles_by_query, seed=args.random_seed)
    results: List[Dict[str, Any]] = [
        evaluate_variant(
            variant_name="base_top5",
            rows_by_qid=rows,
            rank_cache=rank_cache,
            roles_by_query=roles_by_query,
            corpus_records=corpus_records,
            title_to_index=title_to_index,
            reader_top_k=args.reader_top_k,
            max_examples=args.max_examples,
            compatibility_scorer=args.compatibility_scorer,
            embedding_cache=embedding_cache,
            embedding_endpoint=args.embedding_base_url,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            verifier_cache=verifier_cache,
            verifier_endpoint=args.verifier_base_url,
            verifier_model=args.verifier_model,
            verifier_timeout=args.verifier_timeout,
            verifier_max_passage_chars=args.verifier_max_passage_chars,
            verifier_batch_size=args.verifier_batch_size,
            verifier_shortlist_size=args.verifier_shortlist_size,
            verifier_proposal_head_k=args.verifier_proposal_head_k,
            passage_text_mode=args.passage_text_mode,
            role_description_mode=args.role_description_mode,
        )
    ]
    for lambda_value in parse_lambda_values(args.lambda_values):
        results.append(
            evaluate_variant(
                variant_name=lambda_variant_name(lambda_value),
                rows_by_qid=rows,
                rank_cache=rank_cache,
                roles_by_query=roles_by_query,
                corpus_records=corpus_records,
                title_to_index=title_to_index,
                reader_top_k=args.reader_top_k,
                max_examples=args.max_examples,
                compatibility_scorer=args.compatibility_scorer,
                embedding_cache=embedding_cache,
                embedding_endpoint=args.embedding_base_url,
                embedding_model=args.embedding_model,
                embedding_batch_size=args.embedding_batch_size,
                verifier_cache=verifier_cache,
                verifier_endpoint=args.verifier_base_url,
                verifier_model=args.verifier_model,
                verifier_timeout=args.verifier_timeout,
                verifier_max_passage_chars=args.verifier_max_passage_chars,
                verifier_batch_size=args.verifier_batch_size,
                verifier_shortlist_size=args.verifier_shortlist_size,
                verifier_proposal_head_k=args.verifier_proposal_head_k,
                passage_text_mode=args.passage_text_mode,
                role_description_mode=args.role_description_mode,
                lambda_value=lambda_value,
            )
        )
    if not args.skip_controls:
        for control_name, control_roles in (
            ("pointwise_role_score", roles_by_query),
            ("shuffled_roles", shuffled_roles),
        ):
            results.append(
                evaluate_variant(
                    variant_name=control_name,
                    rows_by_qid=rows,
                    rank_cache=rank_cache,
                    roles_by_query=control_roles,
                    corpus_records=corpus_records,
                    title_to_index=title_to_index,
                    reader_top_k=args.reader_top_k,
                    max_examples=args.max_examples,
                    compatibility_scorer=args.compatibility_scorer,
                    embedding_cache=embedding_cache,
                    embedding_endpoint=args.embedding_base_url,
                    embedding_model=args.embedding_model,
                    embedding_batch_size=args.embedding_batch_size,
                    verifier_cache=verifier_cache,
                    verifier_endpoint=args.verifier_base_url,
                    verifier_model=args.verifier_model,
                    verifier_timeout=args.verifier_timeout,
                    verifier_max_passage_chars=args.verifier_max_passage_chars,
                    verifier_batch_size=args.verifier_batch_size,
                    verifier_shortlist_size=args.verifier_shortlist_size,
                    verifier_proposal_head_k=args.verifier_proposal_head_k,
                    passage_text_mode=args.passage_text_mode,
                    role_description_mode=args.role_description_mode,
                    lambda_value=0.0,
                )
            )
    save_embedding_cache(args.embedding_cache_json, embedding_cache)
    save_verifier_cache(args.verifier_cache_json, verifier_cache)
    verifier_cache_entries_after = len(verifier_cache)
    payload = {
        "config": {
            "dataset": args.dataset,
            "corpus_json": args.corpus_json,
            "per_query_report": args.per_query_report,
            "per_query_variant": per_query_variant,
            "baseline_top200_cache": args.baseline_top200_cache,
            "roles_json_path": args.roles_json_path,
            "limit_queries": args.limit_queries,
            "evaluated_query_count": len(common_qids),
            "reader_top_k": args.reader_top_k,
            "candidate_top_k": args.candidate_top_k,
            "compatibility_scorer": args.compatibility_scorer,
            "passage_text_mode": args.passage_text_mode,
            "role_description_mode": args.role_description_mode,
            "lambda_values": args.lambda_values,
            "embedding_base_url": args.embedding_base_url
            if args.compatibility_scorer in {"embedding", "embedding_query_role"}
            else None,
            "embedding_model": args.embedding_model
            if args.compatibility_scorer in {"embedding", "embedding_query_role"}
            else None,
            "embedding_cache_json": args.embedding_cache_json
            if args.compatibility_scorer in {"embedding", "embedding_query_role"}
            else None,
            "verifier_base_url": args.verifier_base_url
            if args.compatibility_scorer in {"llm_verifier", "llm_verifier_batched", "llm_verifier_twostage"}
            else None,
            "verifier_model": args.verifier_model
            if args.compatibility_scorer in {"llm_verifier", "llm_verifier_batched", "llm_verifier_twostage"}
            else None,
            "verifier_cache_json": args.verifier_cache_json
            if args.compatibility_scorer in {"llm_verifier", "llm_verifier_batched", "llm_verifier_twostage"}
            else None,
            "verifier_max_passage_chars": args.verifier_max_passage_chars
            if args.compatibility_scorer in {"llm_verifier", "llm_verifier_batched", "llm_verifier_twostage"}
            else None,
            "verifier_batch_size": args.verifier_batch_size
            if args.compatibility_scorer == "llm_verifier_batched"
            else None,
            "verifier_shortlist_size": args.verifier_shortlist_size
            if args.compatibility_scorer == "llm_verifier_twostage"
            else None,
            "verifier_proposal_head_k": args.verifier_proposal_head_k
            if args.compatibility_scorer == "llm_verifier_twostage"
            else None,
            "verifier_cache_entries_before": verifier_cache_entries_before
            if args.compatibility_scorer
            in {"llm_verifier", "llm_verifier_batched", "llm_verifier_twostage"}
            else None,
            "verifier_cache_entries_after": verifier_cache_entries_after
            if args.compatibility_scorer
            in {"llm_verifier", "llm_verifier_batched", "llm_verifier_twostage"}
            else None,
            "verifier_cache_entries_added": verifier_cache_entries_after - verifier_cache_entries_before
            if args.compatibility_scorer
            in {"llm_verifier", "llm_verifier_batched", "llm_verifier_twostage"}
            else None,
            "skip_controls": bool(args.skip_controls),
            "random_seed": args.random_seed,
        },
        "cleanliness_contract": {
            "method_core": "F(S)=lambda*sum_b(p)+sum_r max_p c(p,r)",
            "requires_external_roles": True,
            "passage_level_selection": True,
            "imports_seer_runtime": False,
            "uses_title_alias_specific_store": False,
            "uses_relation_cue_list": False,
            "uses_gold_for_selection": False,
            "uses_evidence_units": False,
            "uses_role_witnesses": False,
            "uses_graph_edges": False,
            "runs_qa_reader": False,
        },
        "results": results,
    }
    save_json(args.save_json_path, payload)
    write_markdown(args.save_md_path, payload)


if __name__ == "__main__":
    main()
