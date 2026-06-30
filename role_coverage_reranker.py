#!/usr/bin/env python3
"""Role-coverage reranking core.

This module is deliberately smaller than the earlier ERGR-v0 scaffold.  It
implements only the paper-level set-selection object:

    F(S) = lambda * sum_{p in S} b(p) + sum_{r in R_q} max_{p in S} c(p, r)

It does not define evidence units, witnesses, graph edges, role extractors,
reader assembly rules, or dataset-specific activation logic.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


def unique_candidates(candidate_passages: Sequence[Any]) -> List[int]:
    """Return candidate passage ids in first-seen order, skipping invalid ids."""
    candidates: List[int] = []
    seen: Set[int] = set()
    for value in candidate_passages:
        try:
            passage_id = int(value)
        except (TypeError, ValueError):
            continue
        if passage_id in seen:
            continue
        seen.add(passage_id)
        candidates.append(passage_id)
    return candidates


def infer_role_ids(
    role_scores: Mapping[int, Mapping[str, float]],
    *,
    candidate_passages: Optional[Sequence[int]] = None,
) -> List[str]:
    """Infer role ids from the compatibility matrix in deterministic order."""
    role_ids: List[str] = []
    seen: Set[str] = set()
    passages: Iterable[int]
    if candidate_passages is None:
        passages = sorted(int(passage_id) for passage_id in role_scores.keys())
    else:
        passages = unique_candidates(candidate_passages)
    for passage_id in passages:
        for role_id in (role_scores.get(int(passage_id), {}) or {}).keys():
            role_key = str(role_id)
            if role_key in seen:
                continue
            seen.add(role_key)
            role_ids.append(role_key)
    return role_ids


def _float_score(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _passage_role_score(
    role_scores: Mapping[int, Mapping[str, float]],
    passage_id: int,
    role_id: str,
) -> float:
    score = _float_score((role_scores.get(int(passage_id), {}) or {}).get(str(role_id), 0.0))
    return max(score, 0.0)


def role_coverage_value(
    *,
    selected_passages: Sequence[Any],
    base_scores: Mapping[int, float],
    role_scores: Mapping[int, Mapping[str, float]],
    role_ids: Sequence[str],
    base_relevance_weight: float = 1.0,
) -> float:
    """Compute the role-coverage set objective for a selected passage set."""
    selected = unique_candidates(selected_passages)
    roles = [str(role_id) for role_id in role_ids]
    coverage_value = 0.0
    for role_id in roles:
        best_score = 0.0
        for passage_id in selected:
            best_score = max(best_score, _passage_role_score(role_scores, passage_id, role_id))
        coverage_value += best_score

    relevance_value = sum(_float_score(base_scores.get(int(passage_id), 0.0)) for passage_id in selected)
    return float(coverage_value + float(base_relevance_weight) * relevance_value)


def select_role_coverage_topk(
    *,
    candidate_passages: Sequence[Any],
    base_scores: Mapping[int, float],
    role_scores: Mapping[int, Mapping[str, float]],
    role_ids: Sequence[str],
    k: int = 5,
    base_relevance_weight: float = 1.0,
) -> List[int]:
    """Greedily select a reader-visible top-k set by marginal role coverage.

    The greedy step adds the candidate with the largest marginal gain under the
    objective above. Ties prefer larger total role compatibility, then larger
    base relevance, then the original candidate order.
    """
    candidates = unique_candidates(candidate_passages)
    if int(k) <= 0 or not candidates:
        return []

    roles = [str(role_id) for role_id in role_ids]
    selected: List[int] = []
    selected_set: Set[int] = set()
    covered_scores: Dict[str, float] = {role_id: 0.0 for role_id in roles}
    rank_lookup = {passage_id: rank for rank, passage_id in enumerate(candidates)}
    max_k = min(int(k), len(candidates))

    while len(selected) < max_k:
        best_passage: Optional[int] = None
        best_key: Optional[tuple[float, float, float, int]] = None
        for passage_id in candidates:
            if passage_id in selected_set:
                continue

            marginal_coverage = 0.0
            total_role_score = 0.0
            for role_id in roles:
                score = _passage_role_score(role_scores, passage_id, role_id)
                total_role_score += score
                current = float(covered_scores.get(role_id, 0.0))
                if score > current:
                    marginal_coverage += score - current

            base_relevance = _float_score(base_scores.get(int(passage_id), 0.0))
            marginal_gain = marginal_coverage + float(base_relevance_weight) * base_relevance
            key = (
                marginal_gain,
                total_role_score,
                base_relevance,
                -int(rank_lookup.get(passage_id, 10**9)),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_passage = passage_id

        if best_passage is None:
            break

        selected.append(best_passage)
        selected_set.add(best_passage)
        for role_id in roles:
            covered_scores[role_id] = max(
                float(covered_scores.get(role_id, 0.0)),
                _passage_role_score(role_scores, best_passage, role_id),
            )

    return selected
