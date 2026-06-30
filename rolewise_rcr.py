#!/usr/bin/env python3
"""Role-conditioned candidate generation for Role-Coverage Retrieval.

This module keeps the expanded RCR pilot separate from HippoRAG internals.  It
expects role-conditioned retrieval results to be materialized as a cache, then
constructs:

* a candidate pool from per-role rankings;
* a passage-role compatibility matrix from role-conditioned ranking scores;
* a pointwise role-merge baseline for ablation.

It does not call an LLM, inspect gold labels, or contain dataset-specific rules.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from role_coverage_reranker import unique_candidates


RolewiseRankCache = Dict[int, Dict[str, Dict[str, Any]]]
RoleCompatibilityCache = Dict[int, Dict[str, Dict[int, float]]]


def _as_int_list(values: Any) -> List[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    out: List[int] = []
    for value in values:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _float_score(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def role_ids_from_roles(roles: Sequence[Mapping[str, Any]]) -> List[str]:
    role_ids: List[str] = []
    seen = set()
    for offset, role in enumerate(roles):
        role_id = str(role.get("role_id") or f"r{offset}")
        if not role_id or role_id in seen:
            continue
        seen.add(role_id)
        role_ids.append(role_id)
    return role_ids


def normalize_rank_scores(
    doc_indices: Sequence[Any],
    raw_scores: Optional[Any] = None,
) -> Dict[int, float]:
    """Return normalized per-doc scores for a ranked list.

    If numeric scores are available and non-constant, min-max normalize them.
    Otherwise fall back to a deterministic rank prior in [0, 1].
    """
    docs = unique_candidates(doc_indices)
    if not docs:
        return {}

    score_map: Dict[int, float] = {}
    if isinstance(raw_scores, Mapping):
        for doc_idx in docs:
            if doc_idx in raw_scores:
                score_map[int(doc_idx)] = _float_score(raw_scores.get(doc_idx))
            elif str(doc_idx) in raw_scores:
                score_map[int(doc_idx)] = _float_score(raw_scores.get(str(doc_idx)))
    elif isinstance(raw_scores, Sequence) and not isinstance(raw_scores, (str, bytes)):
        seen = set()
        for raw_doc, raw_score in zip(doc_indices, raw_scores):
            try:
                doc_idx = int(raw_doc)
            except (TypeError, ValueError):
                continue
            if doc_idx in seen or doc_idx not in docs:
                continue
            seen.add(doc_idx)
            score_map[int(doc_idx)] = _float_score(raw_score)

    if len(score_map) == len(docs) and any(value != 0.0 for value in score_map.values()):
        values = [float(score_map[int(doc_idx)]) for doc_idx in docs]
        min_value = min(values)
        max_value = max(values)
        if max_value > min_value:
            return {
                int(doc_idx): (float(score_map[int(doc_idx)]) - min_value) / (max_value - min_value)
                for doc_idx in docs
            }
        return {int(doc_idx): 1.0 for doc_idx in docs}

    denom = float(max(len(docs) - 1, 1))
    return {int(doc_idx): 1.0 - (rank / denom) for rank, doc_idx in enumerate(docs)}


def scores_for_role_ranking(ranking: Mapping[str, Any]) -> Dict[int, float]:
    docs = _as_int_list(ranking.get("doc_indices", []))
    raw_scores = ranking.get("base_scores", ranking.get("doc_scores"))
    return normalize_rank_scores(docs, raw_scores)


def parse_rolewise_rank_cache(payload: Any) -> RolewiseRankCache:
    """Parse supported role-conditioned ranking cache formats.

    Supported canonical format:
    {
      "role_rankings_by_query": {
        "0": {
          "r0": {"doc_indices": [1, 2], "doc_scores": [0.9, 0.2]}
        }
      }
    }

    Supported row-list format:
    {
      "rankings": [
        {"query_index": 0, "role_id": "r0", "doc_indices": [1, 2]}
      ]
    }
    """
    if not isinstance(payload, Mapping):
        raise ValueError("Rolewise rank cache must be a JSON object")

    raw_by_query = payload.get("role_rankings_by_query")
    if isinstance(raw_by_query, Mapping):
        out: RolewiseRankCache = {}
        for qid, role_map in raw_by_query.items():
            try:
                query_index = int(qid)
            except (TypeError, ValueError):
                continue
            if not isinstance(role_map, Mapping):
                continue
            parsed_roles: Dict[str, Dict[str, Any]] = {}
            for role_id, ranking in role_map.items():
                if isinstance(ranking, Mapping):
                    parsed_roles[str(role_id)] = dict(ranking)
            if parsed_roles:
                out[query_index] = parsed_roles
        return out

    rows = payload.get("rankings")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        out = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            try:
                query_index = int(row.get("query_index"))
            except (TypeError, ValueError):
                continue
            role_id = str(row.get("role_id") or "")
            if not role_id:
                continue
            out.setdefault(query_index, {})[role_id] = dict(row)
        return out

    # Allow the canonical inner object directly for small fixtures.
    out = {}
    for qid, role_map in payload.items():
        try:
            query_index = int(qid)
        except (TypeError, ValueError):
            continue
        if not isinstance(role_map, Mapping):
            continue
        parsed_roles = {
            str(role_id): dict(ranking)
            for role_id, ranking in role_map.items()
            if isinstance(ranking, Mapping)
        }
        if parsed_roles:
            out[query_index] = parsed_roles
    if out:
        return out

    raise ValueError("Unsupported rolewise rank cache format")


def parse_role_compatibility_cache(payload: Any) -> RoleCompatibilityCache:
    """Parse passage-role compatibility scores keyed by query, role, and doc.

    Canonical format:
    {
      "scores_by_query": {
        "0": {
          "r0": {"123": 0.72}
        }
      }
    }
    """
    if not isinstance(payload, Mapping):
        raise ValueError("Role compatibility cache must be a JSON object")
    raw_by_query = payload.get("scores_by_query", payload)
    if not isinstance(raw_by_query, Mapping):
        raise ValueError("Role compatibility cache must contain a scores_by_query object")

    out: RoleCompatibilityCache = {}
    for qid, role_map in raw_by_query.items():
        try:
            query_index = int(qid)
        except (TypeError, ValueError):
            continue
        if not isinstance(role_map, Mapping):
            continue
        parsed_roles: Dict[str, Dict[int, float]] = {}
        for role_id, score_map in role_map.items():
            if not isinstance(score_map, Mapping):
                continue
            parsed_scores: Dict[int, float] = {}
            for doc_idx, score in score_map.items():
                try:
                    parsed_scores[int(doc_idx)] = _float_score(score)
                except (TypeError, ValueError):
                    continue
            if parsed_scores:
                parsed_roles[str(role_id)] = parsed_scores
        if parsed_roles:
            out[query_index] = parsed_roles
    return out


def build_rolewise_candidate_pool(
    *,
    roles: Sequence[Mapping[str, Any]],
    role_rankings: Mapping[str, Mapping[str, Any]],
    role_top_k: int,
    max_candidates: int,
) -> List[int]:
    """Build the role-conditioned candidate pool."""
    pool: List[int] = []
    for role_id in role_ids_from_roles(roles):
        ranking = role_rankings.get(role_id, {}) or {}
        pool.extend(_as_int_list(ranking.get("doc_indices", []))[: max(int(role_top_k), 0)])
    candidates = unique_candidates(pool)
    limit = max(int(max_candidates), 0)
    return candidates[:limit] if limit else candidates


def build_rolewise_role_scores(
    *,
    candidate_docs: Sequence[Any],
    roles: Sequence[Mapping[str, Any]],
    role_rankings: Mapping[str, Mapping[str, Any]],
    role_compatibility_scores: Optional[Mapping[str, Mapping[int, float]]] = None,
) -> Dict[int, Dict[str, float]]:
    """Build c(p,r|q) from compatibility scores or role-conditioned ranks."""
    candidates = unique_candidates(candidate_docs)
    role_ids = role_ids_from_roles(roles)
    scores: Dict[int, Dict[str, float]] = {int(doc_idx): {role_id: 0.0 for role_id in role_ids} for doc_idx in candidates}
    for role_id in role_ids:
        if role_compatibility_scores is not None:
            raw_scores = role_compatibility_scores.get(role_id, {}) or {}
            role_scores = {
                int(doc_idx): max(_float_score(score), 0.0)
                for doc_idx, score in raw_scores.items()
            }
        else:
            role_scores = scores_for_role_ranking(role_rankings.get(role_id, {}) or {})
        for doc_idx in candidates:
            scores[int(doc_idx)][role_id] = float(role_scores.get(int(doc_idx), 0.0))
    return scores


def select_pointwise_role_merge_topk(
    *,
    candidate_passages: Sequence[Any],
    base_scores: Mapping[int, float],
    role_scores: Mapping[int, Mapping[str, float]],
    role_ids: Sequence[str],
    k: int,
    base_relevance_weight: float = 0.0,
) -> List[int]:
    """Pointwise role-merge baseline, without marginal coverage."""
    candidates = unique_candidates(candidate_passages)
    rank_lookup = {int(doc_idx): rank for rank, doc_idx in enumerate(candidates)}
    roles = [str(role_id) for role_id in role_ids]

    def key(doc_idx: int) -> tuple[float, float, float, int]:
        scores = role_scores.get(int(doc_idx), {}) or {}
        role_sum = sum(_float_score(scores.get(role_id, 0.0)) for role_id in roles)
        role_max = max((_float_score(scores.get(role_id, 0.0)) for role_id in roles), default=0.0)
        base = _float_score(base_scores.get(int(doc_idx), 0.0))
        return (
            role_sum + float(base_relevance_weight) * base,
            role_max,
            base,
            -int(rank_lookup.get(int(doc_idx), 10**9)),
        )

    return sorted(candidates, key=key, reverse=True)[: max(int(k), 0)]
