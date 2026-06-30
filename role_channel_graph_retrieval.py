#!/usr/bin/env python3
"""Role-channel evidence exposure for HippoRAG v2.

This is the next step after Role-aware Graph Entry.  The previous one-shot
version compressed all role-selected graph entries into a single PPR reset
vector.  This module keeps role channels separate through graph retrieval.

The rank-level fusion helpers are retained for diagnostics and offline
ablations only.  ECHO-RAG v1 does not treat raw channel fusion as the final
answer-support evidence set; the main method consumes ``trace.channels`` as an
evidence exposure pool and then performs set-level evidence construction.

Boundary:
* no graph rebuild;
* no OpenIE changes;
* no gold labels;
* no baseline question-level retrieval fallback in this exposure step.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from role_aware_graph_entry import (
    build_role_passage_rankings,
    fact_indices_and_tuples,
    graph_search_with_role_entries,
    role_provenance_text,
    role_retrieval_text,
    role_graph_entry_query,
    top_fact_indices_for_scores,
    valid_roles,
)


@dataclass(frozen=True)
class RoleGraphChannel:
    role_id: str
    role_description: str
    retrieval_query: str
    selected_fact_indices: List[int]
    selected_passage_indices: List[int]
    doc_indices: List[int]
    doc_scores: List[float]
    empty_reason: str | None = None
    support_function: str = ""
    retrieval_text: str = ""
    provenance_text: str = ""
    fact_score_source: str = "computed"
    fact_score_reuse_enabled: bool = False


@dataclass(frozen=True)
class RoleFactFilterResult:
    selected_fact_indices: List[int]
    selected_facts: List[Tuple]
    fact_scores: np.ndarray | None = None


@dataclass(frozen=True)
class RoleChannelRetrievalTrace:
    question: str
    role_count: int
    channel_count: int
    # Diagnostic rank-level assembly over channels.  The ECHO final selector
    # consumes ``channels`` directly, not this raw fused top-k.
    fusion_method: str
    selected_doc_indices: List[int]
    role_queries: List[str]
    channels: List[RoleGraphChannel]
    empty_reason: str | None = None


@dataclass(frozen=True)
class RoleChannelRetrievalResult:
    solution: Any
    trace: RoleChannelRetrievalTrace


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    *,
    rrf_k: float = 60.0,
    top_k: int = 50,
) -> Tuple[List[int], List[float]]:
    """Fuse ranked lists with reciprocal rank fusion.

    This is a diagnostic IR primitive, not the ECHO-RAG v1 evidence-set
    construction mechanism.
    """
    scores: Dict[int, float] = {}
    first_seen: Dict[int, int] = {}
    order = 0
    for ranking in rankings:
        seen_in_ranking = set()
        for rank, raw_doc_idx in enumerate(ranking):
            doc_idx = int(raw_doc_idx)
            if doc_idx in seen_in_ranking:
                continue
            seen_in_ranking.add(doc_idx)
            if doc_idx not in first_seen:
                first_seen[doc_idx] = order
                order += 1
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (float(rrf_k) + float(rank) + 1.0)

    sorted_items = sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]], item[0]))
    if top_k and int(top_k) > 0:
        sorted_items = sorted_items[: int(top_k)]
    return [int(doc_idx) for doc_idx, _ in sorted_items], [float(score) for _, score in sorted_items]


def round_robin_fusion(
    rankings: Sequence[Sequence[int]],
    *,
    top_k: int = 50,
) -> Tuple[List[int], List[float]]:
    """Interleave channel rankings in role order."""
    selected: List[int] = []
    seen = set()
    max_depth = max((len(ranking) for ranking in rankings), default=0)
    for depth in range(max_depth):
        for ranking in rankings:
            if top_k and len(selected) >= int(top_k):
                break
            if depth >= len(ranking):
                continue
            doc_idx = int(ranking[depth])
            if doc_idx in seen:
                continue
            seen.add(doc_idx)
            selected.append(doc_idx)
        if top_k and len(selected) >= int(top_k):
            break
    denom = float(max(len(selected), 1))
    return selected, [1.0 - (rank / denom) for rank, _ in enumerate(selected)]


def best_channel_first_fusion(
    rankings: Sequence[Sequence[int]],
    score_rankings: Sequence[Sequence[float]] | None = None,
    *,
    top_k: int = 50,
) -> Tuple[List[int], List[float]]:
    """Append channels in descending top-score order, deduplicating passages."""
    score_rankings = score_rankings or [[] for _ in rankings]

    def channel_key(item: Tuple[int, Sequence[int]]) -> Tuple[float, int, int]:
        channel_idx, ranking = item
        scores = score_rankings[channel_idx] if channel_idx < len(score_rankings) else []
        top_score = float(scores[0]) if scores else 0.0
        return (top_score, -channel_idx, -len(ranking))

    channel_order = [
        channel_idx
        for channel_idx, _ in sorted(
            enumerate(rankings),
            key=channel_key,
            reverse=True,
        )
    ]
    selected: List[int] = []
    seen = set()
    for channel_idx in channel_order:
        for raw_doc_idx in rankings[channel_idx]:
            if top_k and len(selected) >= int(top_k):
                break
            doc_idx = int(raw_doc_idx)
            if doc_idx in seen:
                continue
            seen.add(doc_idx)
            selected.append(doc_idx)
        if top_k and len(selected) >= int(top_k):
            break
    denom = float(max(len(selected), 1))
    return selected, [1.0 - (rank / denom) for rank, _ in enumerate(selected)]


def _normalized_channel_scores(
    ranking: Sequence[int],
    scores: Sequence[float],
) -> Dict[int, float]:
    docs = []
    seen = set()
    for raw_doc_idx in ranking:
        doc_idx = int(raw_doc_idx)
        if doc_idx in seen:
            continue
        seen.add(doc_idx)
        docs.append(doc_idx)
    if not docs:
        return {}

    raw_scores = [float(scores[idx]) if idx < len(scores) else 0.0 for idx, _ in enumerate(docs)]
    if any(score != 0.0 for score in raw_scores):
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        if max_score > min_score:
            return {
                int(doc_idx): (float(score) - min_score) / (max_score - min_score)
                for doc_idx, score in zip(docs, raw_scores)
            }
        return {int(doc_idx): 1.0 for doc_idx in docs}

    denom = float(max(len(docs) - 1, 1))
    return {int(doc_idx): 1.0 - (rank / denom) for rank, doc_idx in enumerate(docs)}


def pointwise_role_score_fusion(
    rankings: Sequence[Sequence[int]],
    score_rankings: Sequence[Sequence[float]] | None = None,
    *,
    top_k: int = 50,
) -> Tuple[List[int], List[float]]:
    """Fuse by summing normalized per-channel role scores.

    This mirrors the v1 pointwise role-merge idea over graph-channel outputs:
    a passage is preferred when it scores well for one or more evidence roles.
    """
    score_rankings = score_rankings or [[] for _ in rankings]
    score_sum: Dict[int, float] = {}
    score_max: Dict[int, float] = {}
    first_seen: Dict[int, int] = {}
    order = 0
    for channel_idx, ranking in enumerate(rankings):
        scores = score_rankings[channel_idx] if channel_idx < len(score_rankings) else []
        normalized = _normalized_channel_scores(ranking, scores)
        for rank, raw_doc_idx in enumerate(ranking):
            doc_idx = int(raw_doc_idx)
            if doc_idx not in first_seen:
                first_seen[doc_idx] = order
                order += 1
            if doc_idx not in normalized:
                continue
            score = float(normalized[doc_idx])
            score_sum[doc_idx] = score_sum.get(doc_idx, 0.0) + score
            score_max[doc_idx] = max(score_max.get(doc_idx, 0.0), score)

    sorted_items = sorted(
        score_sum.items(),
        key=lambda item: (-item[1], -score_max.get(item[0], 0.0), first_seen.get(item[0], 10**9), item[0]),
    )
    if top_k and int(top_k) > 0:
        sorted_items = sorted_items[: int(top_k)]
    return [int(doc_idx) for doc_idx, _ in sorted_items], [float(score) for _, score in sorted_items]


def fuse_channel_rankings(
    channels: Sequence[RoleGraphChannel],
    *,
    fusion_method: str,
    top_k: int,
    rrf_k: float = 60.0,
) -> Tuple[List[int], List[float]]:
    rankings = [channel.doc_indices for channel in channels if channel.doc_indices]
    score_rankings = [channel.doc_scores for channel in channels if channel.doc_indices]
    if fusion_method == "rrf":
        return reciprocal_rank_fusion(rankings, rrf_k=rrf_k, top_k=top_k)
    if fusion_method == "round_robin":
        return round_robin_fusion(rankings, top_k=top_k)
    if fusion_method == "best_channel_first":
        return best_channel_first_fusion(rankings, score_rankings, top_k=top_k)
    if fusion_method == "pointwise_role_score":
        return pointwise_role_score_fusion(rankings, score_rankings, top_k=top_k)
    raise ValueError(f"Unsupported fusion_method: {fusion_method}")


def role_passage_seed_scores(
    *,
    question: str,
    role: Mapping[str, Any],
    sorted_doc_ids: np.ndarray,
    sorted_doc_scores: np.ndarray,
    role_passage_top_k: int,
) -> Tuple[List[int], Dict[int, float]]:
    rankings = build_role_passage_rankings(
        question=question,
        roles=[role],
        role_passage_rankings={str(role["role_id"]): (sorted_doc_ids, sorted_doc_scores)},
        role_passage_top_k=role_passage_top_k,
    )
    if not rankings:
        return [], {}
    ranking = rankings[0]
    return list(ranking.passage_indices), dict(ranking.passage_scores)


def _jsonable_fact(fact: Any) -> List[str]:
    if isinstance(fact, Sequence) and not isinstance(fact, (str, bytes)):
        return [str(part) for part in fact]
    return [str(fact)]


def _fact_filter_cache_key(
    *,
    system: Any,
    query: str,
    candidate_fact_indices: Sequence[int],
    candidate_facts: Sequence[Any],
) -> str:
    config = getattr(system, "global_config", None)
    payload = {
        "cache_version": 1,
        "llm_name": str(getattr(config, "llm_name", "") or ""),
        "llm_base_url": str(getattr(config, "llm_base_url", "") or ""),
        "linking_top_k": int(getattr(config, "linking_top_k", 0) or 0),
        "query": str(query),
        "candidate_fact_indices": [int(idx) for idx in candidate_fact_indices],
        "candidate_facts": [_jsonable_fact(fact) for fact in candidate_facts],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def rerank_facts_with_optional_cache(
    *,
    system: Any,
    query: str,
    query_fact_scores: np.ndarray,
    fact_filter_cache: MutableMapping[str, Any] | None = None,
) -> Tuple[List[int], List[Tuple], dict]:
    """Mirror ``HippoRAG.rerank_facts`` with an optional exact-input cache.

    The cache is deliberately keyed by the complete LLM fact-filter input:
    query, candidate fact indices, candidate facts, LLM identity, and linking
    budget.  Cache hits therefore reuse the same fact-filter decision without
    changing the retrieval semantics.
    """
    if fact_filter_cache is None:
        return system.rerank_facts(query, query_fact_scores)

    link_top_k: int = int(system.global_config.linking_top_k)
    if len(query_fact_scores) == 0 or len(system.fact_node_keys) == 0:
        return [], [], {"facts_before_rerank": [], "facts_after_rerank": []}

    try:
        if len(query_fact_scores) <= link_top_k:
            candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
        else:
            candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()

        real_candidate_fact_ids = [system.fact_node_keys[int(idx)] for idx in candidate_fact_indices]
        fact_row_dict = system.fact_embedding_store.get_rows(real_candidate_fact_ids)
        candidate_facts = [eval(fact_row_dict[fact_id]["content"]) for fact_id in real_candidate_fact_ids]

        cache_key = _fact_filter_cache_key(
            system=system,
            query=query,
            candidate_fact_indices=candidate_fact_indices,
            candidate_facts=candidate_facts,
        )
        cached = fact_filter_cache.get(cache_key)
        if isinstance(cached, Mapping):
            top_k_fact_indices = [int(idx) for idx in cached.get("top_k_fact_indices", [])]
            top_k_facts = [tuple(fact) for fact in cached.get("top_k_facts", [])]
            return (
                top_k_fact_indices,
                top_k_facts,
                {
                    "facts_before_rerank": candidate_facts,
                    "facts_after_rerank": top_k_facts,
                    "cache_hit": True,
                    "metadata": dict(cached.get("metadata", {}) or {}),
                },
            )

        top_k_fact_indices, top_k_facts, reranker_dict = system.rerank_filter(
            query,
            candidate_facts,
            candidate_fact_indices,
            len_after_rerank=link_top_k,
        )
        metadata = dict(reranker_dict.get("metadata", {}) or {}) if isinstance(reranker_dict, Mapping) else {}
        fact_filter_cache[cache_key] = {
            "top_k_fact_indices": [int(idx) for idx in top_k_fact_indices],
            "top_k_facts": [_jsonable_fact(fact) for fact in top_k_facts],
            "metadata": metadata,
        }
        return (
            top_k_fact_indices,
            top_k_facts,
            {"facts_before_rerank": candidate_facts, "facts_after_rerank": top_k_facts, "metadata": metadata},
        )
    except Exception as exc:
        if bool(getattr(system.global_config, "rerank_filter_fail_on_exception", False)):
            raise
        return [], [], {"facts_before_rerank": [], "facts_after_rerank": [], "error": str(exc)}


def _candidate_facts_from_scores(
    *,
    system: Any,
    query_fact_scores: np.ndarray,
) -> Tuple[List[int], List[Tuple]]:
    link_top_k: int = int(system.global_config.linking_top_k)
    if len(query_fact_scores) == 0 or len(system.fact_node_keys) == 0:
        return [], []
    if len(query_fact_scores) <= link_top_k:
        candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
    else:
        candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
    real_candidate_fact_ids = [system.fact_node_keys[int(idx)] for idx in candidate_fact_indices]
    fact_row_dict = system.fact_embedding_store.get_rows(real_candidate_fact_ids)
    candidate_facts = [eval(fact_row_dict[fact_id]["content"]) for fact_id in real_candidate_fact_ids]
    return [int(idx) for idx in candidate_fact_indices], [tuple(fact) for fact in candidate_facts]


def _shared_fact_filter_query(
    *,
    question: str,
    roles: Sequence[Mapping[str, Any]],
) -> str:
    role_lines = []
    for role in roles:
        role_id = str(role.get("role_id") or "")
        support_function = str(role.get("support_function") or role.get("role_type") or "")
        retrieval_text = role_retrieval_text(role)
        role_lines.append(f"- {role_id}: {support_function}. {retrieval_text}".strip())
    return (
        "Question:\n"
        f"{question}\n\n"
        "Evidence requirements:\n"
        + "\n".join(role_lines)
        + "\n\nSelect facts that are useful for satisfying these evidence requirements."
    )


def _shared_demand_assignment_query(
    *,
    question: str,
    roles: Sequence[Mapping[str, Any]],
) -> str:
    role_lines = []
    for role in roles:
        role_id = str(role.get("role_id") or "")
        support_function = str(role.get("support_function") or role.get("role_type") or "")
        retrieval_text = role_retrieval_text(role)
        role_lines.append(f"- {role_id}: {support_function}. {retrieval_text}".strip())
    return (
        "Question:\n"
        f"{question}\n\n"
        "Evidence requirements:\n"
        + "\n".join(role_lines)
        + "\n\nAssign candidate fact ids to each evidence requirement. "
        "Each requirement should receive facts that can seed graph retrieval for that specific requirement."
    )


def _shared_fact_filter_cache_key(
    *,
    system: Any,
    question: str,
    roles: Sequence[Mapping[str, Any]],
    candidate_fact_indices: Sequence[int],
    candidate_facts: Sequence[Any],
) -> str:
    shared_query = _shared_fact_filter_query(question=question, roles=roles)
    return _fact_filter_cache_key(
        system=system,
        query=shared_query,
        candidate_fact_indices=candidate_fact_indices,
        candidate_facts=candidate_facts,
    )


def _shared_demand_assignment_cache_key(
    *,
    system: Any,
    question: str,
    roles: Sequence[Mapping[str, Any]],
    candidate_fact_indices: Sequence[int],
    candidate_facts: Sequence[Any],
) -> str:
    assignment_query = _shared_demand_assignment_query(question=question, roles=roles)
    return _fact_filter_cache_key(
        system=system,
        query=assignment_query,
        candidate_fact_indices=candidate_fact_indices,
        candidate_facts=candidate_facts,
    )


def _llm_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, Sequence) and response:
        first = response[0]
        return first if isinstance(first, str) else str(first)
    return str(response)


def _llm_response_text_and_metadata(response: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(response, str):
        return response, {}
    if isinstance(response, Sequence) and response:
        first = response[0]
        text = first if isinstance(first, str) else str(first)
        metadata = response[1] if len(response) > 1 and isinstance(response[1], Mapping) else {}
        return text, dict(metadata)
    return str(response), {}


def _extract_json_object(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def _parse_assignment_payload(
    payload: Any,
    *,
    role_ids: Sequence[str],
    allowed_fact_ids: set[int],
) -> Dict[str, List[int]]:
    assignments: Dict[str, List[int]] = {role_id: [] for role_id in role_ids}

    def add(role_id: str, values: Any) -> None:
        if role_id not in assignments:
            return
        if isinstance(values, Mapping):
            values = values.get("fact_ids") or values.get("facts") or values.get("selected_fact_ids") or []
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            values = [values]
        for raw_value in values:
            try:
                fact_id = int(raw_value)
            except (TypeError, ValueError):
                continue
            if fact_id in allowed_fact_ids and fact_id not in assignments[role_id]:
                assignments[role_id].append(fact_id)

    if isinstance(payload, Mapping):
        raw_assignments = payload.get("assignments")
        if isinstance(raw_assignments, Sequence) and not isinstance(raw_assignments, (str, bytes)):
            for item in raw_assignments:
                if not isinstance(item, Mapping):
                    continue
                add(str(item.get("role_id") or item.get("id") or ""), item)
        for role_id in role_ids:
            if role_id in payload:
                add(role_id, payload[role_id])
    return assignments


def shared_question_by_demand_fact_filter(
    *,
    system: Any,
    question: str,
    roles: Sequence[Mapping[str, Any]],
    fact_filter_cache: MutableMapping[str, Any] | None = None,
) -> Dict[str, RoleFactFilterResult]:
    """Run one LLM fact-filter call and ask it to assign facts per demand.

    Compared with ``shared_question_fact_filter``, this keeps the one-call
    cost profile but makes the output demand-conditioned.  Each role receives
    its own fact ids, so terminal or constraint demands are not forced to
    compete only through a global fact list.
    """
    link_top_k: int = int(system.global_config.linking_top_k)
    role_candidates: Dict[str, Tuple[List[int], List[Tuple]]] = {}
    role_fact_scores: Dict[str, np.ndarray] = {}
    fact_by_index: Dict[int, Tuple] = {}
    union_indices: List[int] = []
    union_facts: List[Tuple] = []
    seen_indices = set()

    for role in roles:
        role_id = str(role["role_id"])
        role_query = role_graph_entry_query(question=question, role=role)
        fact_scores = np.asarray(system.get_fact_scores(role_query), dtype=float)
        role_fact_scores[role_id] = fact_scores
        candidate_indices, candidate_facts = _candidate_facts_from_scores(
            system=system,
            query_fact_scores=fact_scores,
        )
        role_candidates[role_id] = (candidate_indices, candidate_facts)
        for fact_idx, fact in zip(candidate_indices, candidate_facts):
            fact_idx = int(fact_idx)
            fact_tuple = tuple(fact)
            fact_by_index[fact_idx] = fact_tuple
            if fact_idx in seen_indices:
                continue
            seen_indices.add(fact_idx)
            union_indices.append(fact_idx)
            union_facts.append(fact_tuple)

    if not union_facts:
        return {
            role_id: RoleFactFilterResult([], [], role_fact_scores.get(role_id))
            for role_id in role_candidates
        }

    role_ids = [str(role["role_id"]) for role in roles]
    cache_key = _shared_demand_assignment_cache_key(
        system=system,
        question=question,
        roles=roles,
        candidate_fact_indices=union_indices,
        candidate_facts=union_facts,
    )
    cached = fact_filter_cache.get(cache_key) if fact_filter_cache is not None else None
    if isinstance(cached, Mapping) and isinstance(cached.get("assignments"), Mapping):
        assignments = {
            str(role_id): [int(idx) for idx in values]
            for role_id, values in cached.get("assignments", {}).items()
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes))
        }
    else:
        metadata: Dict[str, Any] = {}
        role_lines = []
        for role in roles:
            role_id = str(role["role_id"])
            support_function = str(role.get("support_function") or role.get("role_type") or "")
            role_lines.append(
                f"{role_id}: {support_function}. {role_retrieval_text(role)}".strip()
            )
        fact_lines = [
            f"{fact_idx}: {json.dumps(_jsonable_fact(fact), ensure_ascii=False)}"
            for fact_idx, fact in zip(union_indices, union_facts)
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "Assign candidate fact ids to evidence requirements for graph retrieval. "
                    "Return only valid JSON with this schema: "
                    "{\"assignments\":[{\"role_id\":\"r0\",\"fact_ids\":[1,2]}]}. "
                    "Use only fact ids from the candidate list. Select at most "
                    f"{link_top_k} facts per role."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Question:\n"
                    f"{question}\n\n"
                    "Evidence requirements:\n"
                    + "\n".join(role_lines)
                    + "\n\nCandidate facts:\n"
                    + "\n".join(fact_lines)
                ),
            },
        ]
        try:
            response, metadata = _llm_response_text_and_metadata(
                system.llm_model.infer(
                    messages=messages,
                    model=str(getattr(system.global_config, "llm_name", "")),
                    temperature=0.0,
                    max_completion_tokens=512,
                )
            )
        except Exception:
            if bool(getattr(system.global_config, "rerank_filter_fail_on_exception", False)):
                raise
            response = ""
            metadata = {}
        try:
            assignments = _parse_assignment_payload(
                _extract_json_object(response),
                role_ids=role_ids,
                allowed_fact_ids=set(union_indices),
            )
        except Exception:
            assignments = {role_id: [] for role_id in role_ids}
        if fact_filter_cache is not None:
            cache_assignments = {
                role_id: [int(idx) for idx in assignments.get(role_id, [])[:link_top_k]]
                for role_id in role_ids
            }
            fact_filter_cache[cache_key] = {
                "assignments": cache_assignments,
                "candidate_fact_indices": [int(idx) for idx in union_indices],
                "metadata": metadata,
            }

    per_role: Dict[str, RoleFactFilterResult] = {}
    for role_id, (candidate_indices, candidate_facts) in role_candidates.items():
        candidate_set = {int(idx) for idx in candidate_indices}
        assigned = [
            int(idx)
            for idx in assignments.get(role_id, [])
            if int(idx) in candidate_set and int(idx) in fact_by_index
        ]
        if not assigned and candidate_indices:
            assigned = [int(candidate_indices[0])]
        assigned = assigned[:link_top_k]
        per_role[role_id] = RoleFactFilterResult(
            selected_fact_indices=assigned,
            selected_facts=[fact_by_index[int(idx)] for idx in assigned],
            fact_scores=role_fact_scores.get(role_id),
        )
    return per_role


def shared_question_fact_filter(
    *,
    system: Any,
    question: str,
    roles: Sequence[Mapping[str, Any]],
    fact_filter_cache: MutableMapping[str, Any] | None = None,
) -> Dict[str, Tuple[List[int], List[Tuple]]]:
    """Run one fact-filter call for all evidence-requirement channels of a question.

    The LLM sees the original question, all evidence requirements, and the union of
    embedding-linked facts.  The selected facts are then projected back to each
    channel by retaining the facts that came from that channel's own embedding
    candidate list.  If the shared filter drops every fact for a channel, that
    channel falls back to its first embedding-linked fact to keep graph retrieval
    defined.
    """
    link_top_k: int = int(system.global_config.linking_top_k)
    role_candidates: Dict[str, Tuple[List[int], List[Tuple]]] = {}
    union_indices: List[int] = []
    union_facts: List[Tuple] = []
    seen_indices = set()

    for role in roles:
        role_id = str(role["role_id"])
        role_query = role_graph_entry_query(question=question, role=role)
        fact_scores = np.asarray(system.get_fact_scores(role_query), dtype=float)
        candidate_indices, candidate_facts = _candidate_facts_from_scores(
            system=system,
            query_fact_scores=fact_scores,
        )
        role_candidates[role_id] = (candidate_indices, candidate_facts)
        for fact_idx, fact in zip(candidate_indices, candidate_facts):
            if int(fact_idx) in seen_indices:
                continue
            seen_indices.add(int(fact_idx))
            union_indices.append(int(fact_idx))
            union_facts.append(tuple(fact))

    if not union_facts:
        return {role_id: ([], []) for role_id in role_candidates}

    shared_query = _shared_fact_filter_query(question=question, roles=roles)
    cache_key = _shared_fact_filter_cache_key(
        system=system,
        question=question,
        roles=roles,
        candidate_fact_indices=union_indices,
        candidate_facts=union_facts,
    )
    cached = fact_filter_cache.get(cache_key) if fact_filter_cache is not None else None
    if isinstance(cached, Mapping):
        selected_indices = [int(idx) for idx in cached.get("top_k_fact_indices", [])]
        selected_facts = [tuple(fact) for fact in cached.get("top_k_facts", [])]
    else:
        selected_indices, selected_facts, reranker_dict = system.rerank_filter(
            shared_query,
            union_facts,
            union_indices,
            len_after_rerank=min(max(link_top_k * len(roles), link_top_k), len(union_facts)),
        )
        metadata = dict(reranker_dict.get("metadata", {}) or {}) if isinstance(reranker_dict, Mapping) else {}
        selected_indices = [int(idx) for idx in selected_indices]
        selected_facts = [tuple(fact) for fact in selected_facts]
        if fact_filter_cache is not None:
            fact_filter_cache[cache_key] = {
                "top_k_fact_indices": selected_indices,
                "top_k_facts": [_jsonable_fact(fact) for fact in selected_facts],
                "metadata": metadata,
            }

    selected_set = set(selected_indices)
    per_role: Dict[str, Tuple[List[int], List[Tuple]]] = {}
    for role_id, (candidate_indices, candidate_facts) in role_candidates.items():
        kept = [
            (int(fact_idx), tuple(fact))
            for fact_idx, fact in zip(candidate_indices, candidate_facts)
            if int(fact_idx) in selected_set
        ]
        if not kept and candidate_indices:
            kept = [(int(candidate_indices[0]), tuple(candidate_facts[0]))]
        kept = kept[:link_top_k]
        per_role[role_id] = ([idx for idx, _ in kept], [fact for _, fact in kept])
    return per_role


def run_single_role_graph_channel(
    *,
    system: Any,
    question: str,
    role: Mapping[str, Any],
    role_fact_top_k: int,
    role_passage_top_k: int,
    channel_output_top_k: int,
    channel_backend: str = "hipporag_graph",
    entry_passage_node_weight: float | None = None,
    fact_filter_cache: MutableMapping[str, Any] | None = None,
    preselected_facts: Tuple[List[int], List[Tuple]] | None = None,
    precomputed_fact_scores: np.ndarray | None = None,
) -> RoleGraphChannel:
    role_id = str(role["role_id"])
    role_query = role_graph_entry_query(question=question, role=role)
    provenance_text = role_provenance_text(role)
    retrieval_text = role_retrieval_text(role)
    support_function = str(role.get("support_function") or role.get("role_type") or "")
    fact_score_source = "precomputed" if precomputed_fact_scores is not None else "computed"
    fact_scores = (
        np.asarray(precomputed_fact_scores, dtype=float)
        if precomputed_fact_scores is not None
        else np.asarray(system.get_fact_scores(role_query), dtype=float)
    )

    if channel_backend == "hipporag_graph":
        if preselected_facts is None:
            top_k_fact_indices, top_k_facts, _ = rerank_facts_with_optional_cache(
                system=system,
                query=role_query,
                query_fact_scores=fact_scores,
                fact_filter_cache=fact_filter_cache,
            )
        else:
            top_k_fact_indices, top_k_facts = preselected_facts

        sorted_dense_ids, sorted_dense_scores = system.dense_passage_retrieval(role_query)
        passage_indices, passage_scores = role_passage_seed_scores(
            question=question,
            role=role,
            sorted_doc_ids=sorted_dense_ids,
            sorted_doc_scores=sorted_dense_scores,
            role_passage_top_k=role_passage_top_k,
        )

        if len(top_k_facts) == 0 and not passage_indices:
            sorted_doc_ids, sorted_doc_scores = sorted_dense_ids, sorted_dense_scores
        elif passage_indices:
            sorted_doc_ids, sorted_doc_scores = graph_search_with_role_entries(
                system=system,
                query=role_query,
                link_top_k=system.global_config.linking_top_k,
                query_fact_scores=fact_scores,
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                passage_seed_scores=passage_scores,
                passage_node_weight=(
                    float(entry_passage_node_weight)
                    if entry_passage_node_weight is not None
                    else float(system.global_config.passage_node_weight)
                ),
            )
        else:
            sorted_doc_ids, sorted_doc_scores = system.graph_search_with_fact_entities(
                query=role_query,
                link_top_k=system.global_config.linking_top_k,
                query_fact_scores=fact_scores,
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                passage_node_weight=system.global_config.passage_node_weight,
            )
        limit = max(int(channel_output_top_k), 0)
        doc_indices = [int(doc_idx) for doc_idx in np.asarray(sorted_doc_ids).tolist()]
        doc_scores = [float(score) for score in np.asarray(sorted_doc_scores, dtype=float).tolist()]
        if limit:
            doc_indices = doc_indices[:limit]
            doc_scores = doc_scores[:limit]
        return RoleGraphChannel(
            role_id=role_id,
            role_description=provenance_text,
            retrieval_query=role_query,
            selected_fact_indices=[int(idx) for idx in top_k_fact_indices],
            selected_passage_indices=passage_indices,
            doc_indices=doc_indices,
            doc_scores=doc_scores,
            empty_reason=None if doc_indices else "no_channel_docs",
            support_function=support_function,
            retrieval_text=retrieval_text,
            provenance_text=provenance_text,
            fact_score_source=fact_score_source,
            fact_score_reuse_enabled=precomputed_fact_scores is not None,
        )

    if channel_backend != "seeded_entry":
        raise ValueError(f"Unsupported channel_backend: {channel_backend}")

    fact_indices = top_fact_indices_for_scores(fact_scores, role_fact_top_k)
    top_k_fact_indices, top_k_facts = fact_indices_and_tuples(system, fact_indices)

    sorted_dense_ids, sorted_dense_scores = system.dense_passage_retrieval(role_query)
    passage_indices, passage_scores = role_passage_seed_scores(
        question=question,
        role=role,
        sorted_doc_ids=sorted_dense_ids,
        sorted_doc_scores=sorted_dense_scores,
        role_passage_top_k=role_passage_top_k,
    )

    if not top_k_facts and not passage_indices:
        return RoleGraphChannel(
            role_id=role_id,
            role_description=provenance_text,
            retrieval_query=role_query,
            selected_fact_indices=fact_indices,
            selected_passage_indices=passage_indices,
            doc_indices=[],
            doc_scores=[],
            empty_reason="no_graph_entries",
            support_function=support_function,
            retrieval_text=retrieval_text,
            provenance_text=provenance_text,
            fact_score_source=fact_score_source,
            fact_score_reuse_enabled=precomputed_fact_scores is not None,
        )

    sorted_doc_ids, sorted_doc_scores = graph_search_with_role_entries(
        system=system,
        query=role_query,
        link_top_k=system.global_config.linking_top_k,
        query_fact_scores=fact_scores,
        top_k_facts=top_k_facts,
        top_k_fact_indices=top_k_fact_indices,
        passage_seed_scores=passage_scores,
        passage_node_weight=(
            float(entry_passage_node_weight)
            if entry_passage_node_weight is not None
            else float(system.global_config.passage_node_weight)
        ),
    )
    limit = max(int(channel_output_top_k), 0)
    doc_indices = [int(doc_idx) for doc_idx in np.asarray(sorted_doc_ids).tolist()]
    doc_scores = [float(score) for score in np.asarray(sorted_doc_scores, dtype=float).tolist()]
    if limit:
        doc_indices = doc_indices[:limit]
        doc_scores = doc_scores[:limit]
    return RoleGraphChannel(
        role_id=role_id,
        role_description=provenance_text,
        retrieval_query=role_query,
        selected_fact_indices=top_k_fact_indices,
        selected_passage_indices=passage_indices,
        doc_indices=doc_indices,
        doc_scores=doc_scores,
        support_function=support_function,
        retrieval_text=retrieval_text,
        provenance_text=provenance_text,
        fact_score_source=fact_score_source,
        fact_score_reuse_enabled=precomputed_fact_scores is not None,
    )


def role_channel_graph_retrieve(
    *,
    system: Any,
    queries: Sequence[str],
    roles_by_query: Mapping[int, Sequence[Mapping[str, Any]]],
    query_indices: Sequence[int] | None = None,
    num_to_retrieve: int = 5,
    role_fact_top_k: int = 5,
    role_passage_top_k: int = 5,
    channel_output_top_k: int = 50,
    rrf_k: float = 60.0,
    fusion_method: str = "rrf",
    channel_backend: str = "hipporag_graph",
    entry_passage_node_weight: float | None = None,
    fact_filter_cache: MutableMapping[str, Any] | None = None,
    fact_filter_strategy: str = "per_channel",
) -> List[RoleChannelRetrievalResult]:
    """Run role-preserving graph retrieval and emit channel traces.

    ``solution`` and ``trace.selected_doc_indices`` are diagnostic raw-fusion
    readouts.  Downstream ECHO selectors use ``trace.channels`` to construct the
    candidate evidence pool.
    """
    from hipporag.utils.misc_utils import QuerySolution

    if not system.ready_to_retrieve:
        system.prepare_retrieval_objects()

    qids = list(query_indices) if query_indices is not None else list(range(len(queries)))
    if len(qids) != len(queries):
        raise ValueError("query_indices length must match queries length")

    role_query_texts: List[str] = []
    roles_by_position: List[List[Dict[str, str]]] = []
    for query, qid in zip(queries, qids):
        roles = valid_roles(roles_by_query.get(int(qid), []) or [])
        roles_by_position.append(roles)
        for role in roles:
            role_query_texts.append(role_graph_entry_query(question=query, role=role))

    system.get_query_embeddings(role_query_texts)

    results: List[RoleChannelRetrievalResult] = []
    for query, roles in zip(queries, roles_by_position):
        role_queries = [role_graph_entry_query(question=query, role=role) for role in roles]
        if not roles:
            results.append(
                RoleChannelRetrievalResult(
                    solution=QuerySolution(question=query, docs=[], doc_scores=np.asarray([], dtype=float)),
                    trace=RoleChannelRetrievalTrace(
                        question=query,
                        role_count=0,
                        channel_count=0,
                        fusion_method=fusion_method,
                        selected_doc_indices=[],
                        role_queries=[],
                        channels=[],
                        empty_reason="no_valid_roles",
                    ),
                )
            )
            continue

        shared_facts_by_role: Dict[str, Any] = {}
        if fact_filter_strategy == "shared_question" and channel_backend == "hipporag_graph":
            shared_facts_by_role = shared_question_fact_filter(
                system=system,
                question=query,
                roles=roles,
                fact_filter_cache=fact_filter_cache,
            )
        elif fact_filter_strategy == "shared_question_by_demand" and channel_backend == "hipporag_graph":
            shared_facts_by_role = shared_question_by_demand_fact_filter(
                system=system,
                question=query,
                roles=roles,
                fact_filter_cache=fact_filter_cache,
            )
        elif fact_filter_strategy != "per_channel":
            raise ValueError(f"Unsupported fact_filter_strategy: {fact_filter_strategy}")

        channels: List[RoleGraphChannel] = []
        for role in roles:
            shared_entry = shared_facts_by_role.get(str(role["role_id"]))
            preselected_facts = None
            precomputed_fact_scores = None
            if isinstance(shared_entry, RoleFactFilterResult):
                preselected_facts = (shared_entry.selected_fact_indices, shared_entry.selected_facts)
                precomputed_fact_scores = shared_entry.fact_scores
            else:
                preselected_facts = shared_entry
            channels.append(
                run_single_role_graph_channel(
                    system=system,
                    question=query,
                    role=role,
                    role_fact_top_k=role_fact_top_k,
                    role_passage_top_k=role_passage_top_k,
                    channel_output_top_k=channel_output_top_k,
                    channel_backend=channel_backend,
                    entry_passage_node_weight=entry_passage_node_weight,
                    fact_filter_cache=fact_filter_cache,
                    preselected_facts=preselected_facts,
                    precomputed_fact_scores=precomputed_fact_scores,
                )
            )
        active_channels = [channel for channel in channels if channel.doc_indices]
        fused_doc_indices, fused_doc_scores = fuse_channel_rankings(
            active_channels,
            fusion_method=fusion_method,
            top_k=num_to_retrieve,
            rrf_k=rrf_k,
        )

        if not fused_doc_indices:
            results.append(
                RoleChannelRetrievalResult(
                    solution=QuerySolution(question=query, docs=[], doc_scores=np.asarray([], dtype=float)),
                    trace=RoleChannelRetrievalTrace(
                        question=query,
                        role_count=len(roles),
                        channel_count=0,
                        fusion_method=fusion_method,
                        selected_doc_indices=[],
                        role_queries=role_queries,
                        channels=channels,
                        empty_reason="no_active_channels",
                    ),
                )
            )
            continue

        top_k_docs = [
            system.chunk_embedding_store.get_row(system.passage_node_keys[int(idx)])["content"]
            for idx in fused_doc_indices[: int(num_to_retrieve)]
        ]
        results.append(
            RoleChannelRetrievalResult(
                solution=QuerySolution(
                    question=query,
                    docs=top_k_docs,
                    doc_scores=np.asarray(fused_doc_scores[: int(num_to_retrieve)], dtype=float),
                ),
                trace=RoleChannelRetrievalTrace(
                    question=query,
                    role_count=len(roles),
                    channel_count=len(active_channels),
                    fusion_method=fusion_method,
                    selected_doc_indices=fused_doc_indices[: int(num_to_retrieve)],
                    role_queries=role_queries,
                    channels=channels,
                ),
            )
        )

    return results
