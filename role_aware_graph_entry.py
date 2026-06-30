#!/usr/bin/env python3
"""Role-aware graph entry for HippoRAG v2.

This module is the first v2 step after Role-wise RCR v1.  v1 runs one complete
HippoRAG retrieval call per evidence role and merges passage rankings outside
the retriever.  Here, roles choose fact/entity entries and passage-node entries
for the existing HippoRAG graph search, so the graph/PPR stage runs once per
question.

Boundary:
* no graph rebuild;
* no OpenIE changes;
* no gold labels;
* no baseline question-level fact or passage seeds.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hipporag.utils.misc_utils import compute_mdhash_id


@dataclass(frozen=True)
class RoleFactRanking:
    role_id: str
    role_description: str
    retrieval_query: str
    fact_indices: List[int]
    fact_scores: Dict[int, float]


@dataclass(frozen=True)
class RoleAwareFactSeeds:
    merged_fact_scores: np.ndarray
    fact_indices: List[int]
    selected_by_role: Dict[str, List[int]]


@dataclass(frozen=True)
class RolePassageRanking:
    role_id: str
    role_description: str
    retrieval_query: str
    passage_indices: List[int]
    passage_scores: Dict[int, float]


@dataclass(frozen=True)
class RoleAwarePassageSeeds:
    passage_indices: List[int]
    passage_scores: Dict[int, float]
    selected_by_role: Dict[str, List[int]]


@dataclass(frozen=True)
class RoleAwareRetrievalTrace:
    question: str
    role_count: int
    role_queries: List[str]
    selected_fact_indices: List[int]
    selected_passage_indices: List[int]
    selected_by_role: Dict[str, List[int]]
    selected_passages_by_role: Dict[str, List[int]]
    empty_reason: str | None = None
    graph_search_count: int = 0
    fact_seed_budget: int = 0
    link_top_k_effective: int = 0
    seed_union_mode: str = "role_balanced_round_robin"


@dataclass(frozen=True)
class RoleAwareRetrievalResult:
    solution: Any
    trace: RoleAwareRetrievalTrace


def clean_role_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def role_retrieval_text(role: Mapping[str, Any]) -> str:
    return clean_role_text(
        role.get("retrieval_text")
        or role.get("retrieval_query_text")
        or role.get("query")
        or role.get("description")
    )


def role_provenance_text(role: Mapping[str, Any]) -> str:
    return clean_role_text(
        role.get("provenance_text")
        or role.get("support_description")
        or role.get("description")
        or role.get("retrieval_text")
    )


def role_graph_entry_query(*, question: str, role: Mapping[str, Any]) -> str:
    retrieval_text = role_retrieval_text(role)
    return f"Question: {str(question or '').strip()}\nEvidence role: {retrieval_text}"


def valid_roles(roles: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for offset, role in enumerate(roles or []):
        if not isinstance(role, Mapping):
            continue
        retrieval_text = role_retrieval_text(role)
        provenance_text = role_provenance_text(role)
        if not retrieval_text and not provenance_text:
            continue
        role_id = str(role.get("role_id") or f"r{offset}")
        if role_id in seen:
            continue
        seen.add(role_id)
        role_type = str(role.get("role_type") or role.get("support_function") or "role")
        support_function = str(role.get("support_function") or role_type)
        out.append(
            {
                "role_id": role_id,
                "role_type": role_type,
                "description": provenance_text or retrieval_text,
                "retrieval_text": retrieval_text or provenance_text,
                "provenance_text": provenance_text or retrieval_text,
                "support_function": support_function,
            }
        )
    return out


def top_fact_indices_for_scores(scores: np.ndarray, top_k: int) -> List[int]:
    if top_k <= 0 or scores.size == 0:
        return []
    k = min(int(top_k), int(scores.size))
    return np.argsort(scores)[-k:][::-1].astype(int).tolist()


def build_role_fact_rankings(
    *,
    question: str,
    roles: Sequence[Mapping[str, Any]],
    role_fact_scores: Mapping[str, np.ndarray],
    role_fact_top_k: int,
) -> List[RoleFactRanking]:
    rankings: List[RoleFactRanking] = []
    for role in valid_roles(roles):
        role_id = role["role_id"]
        scores = np.asarray(role_fact_scores.get(role_id, np.asarray([])), dtype=float)
        fact_indices = top_fact_indices_for_scores(scores, role_fact_top_k)
        rankings.append(
            RoleFactRanking(
                role_id=role_id,
                role_description=role["description"],
                retrieval_query=role_graph_entry_query(question=question, role=role),
                fact_indices=fact_indices,
                fact_scores={int(idx): float(scores[int(idx)]) for idx in fact_indices},
            )
        )
    return rankings


def merge_role_fact_seeds(
    *,
    rankings: Sequence[RoleFactRanking],
    num_facts: int,
    max_fact_seeds: int,
) -> RoleAwareFactSeeds:
    """Merge per-role fact rankings into a role-balanced seed set.

    The merge is round-robin over role rankings.  This keeps the method aligned
    with the v1 finding: roles are useful retrieval intents, so graph entry
    should not collapse all seed budget into whichever single role has the
    highest raw embedding scores.
    """
    merged_scores = np.zeros(int(num_facts), dtype=float)
    budget = max(int(max_fact_seeds), 0)
    selected: List[int] = []
    seen = set()
    selected_by_role: Dict[str, List[int]] = {ranking.role_id: [] for ranking in rankings}
    max_depth = max((len(ranking.fact_indices) for ranking in rankings), default=0)

    for depth in range(max_depth):
        for ranking in rankings:
            if budget and len(selected) >= budget:
                break
            if depth >= len(ranking.fact_indices):
                continue
            fact_idx = int(ranking.fact_indices[depth])
            if fact_idx < 0 or fact_idx >= int(num_facts):
                continue
            score = float(ranking.fact_scores.get(fact_idx, 0.0))
            merged_scores[fact_idx] = max(float(merged_scores[fact_idx]), score)
            if fact_idx in seen:
                continue
            seen.add(fact_idx)
            selected.append(fact_idx)
            selected_by_role[ranking.role_id].append(fact_idx)
        if budget and len(selected) >= budget:
            break

    return RoleAwareFactSeeds(
        merged_fact_scores=merged_scores,
        fact_indices=selected,
        selected_by_role=selected_by_role,
    )


def build_role_passage_rankings(
    *,
    question: str,
    roles: Sequence[Mapping[str, Any]],
    role_passage_rankings: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    role_passage_top_k: int,
) -> List[RolePassageRanking]:
    rankings: List[RolePassageRanking] = []
    for role in valid_roles(roles):
        role_id = role["role_id"]
        sorted_doc_ids, sorted_doc_scores = role_passage_rankings.get(
            role_id,
            (np.asarray([], dtype=int), np.asarray([], dtype=float)),
        )
        doc_ids = [int(doc_idx) for doc_idx in np.asarray(sorted_doc_ids).tolist()[: max(int(role_passage_top_k), 0)]]
        raw_scores = np.asarray(sorted_doc_scores, dtype=float).tolist()[: len(doc_ids)]
        rankings.append(
            RolePassageRanking(
                role_id=role_id,
                role_description=role["description"],
                retrieval_query=role_graph_entry_query(question=question, role=role),
                passage_indices=doc_ids,
                passage_scores={int(doc_idx): float(score) for doc_idx, score in zip(doc_ids, raw_scores)},
            )
        )
    return rankings


def merge_role_passage_seeds(
    *,
    rankings: Sequence[RolePassageRanking],
    max_passage_seeds: int,
) -> RoleAwarePassageSeeds:
    budget = max(int(max_passage_seeds), 0)
    selected: List[int] = []
    seen = set()
    selected_by_role: Dict[str, List[int]] = {ranking.role_id: [] for ranking in rankings}
    passage_scores: Dict[int, float] = {}
    max_depth = max((len(ranking.passage_indices) for ranking in rankings), default=0)

    for depth in range(max_depth):
        for ranking in rankings:
            if budget and len(selected) >= budget:
                break
            if depth >= len(ranking.passage_indices):
                continue
            passage_idx = int(ranking.passage_indices[depth])
            score = float(ranking.passage_scores.get(passage_idx, 0.0))
            passage_scores[passage_idx] = max(float(passage_scores.get(passage_idx, 0.0)), score)
            if passage_idx in seen:
                continue
            seen.add(passage_idx)
            selected.append(passage_idx)
            selected_by_role[ranking.role_id].append(passage_idx)
        if budget and len(selected) >= budget:
            break

    return RoleAwarePassageSeeds(
        passage_indices=selected,
        passage_scores={idx: float(passage_scores.get(idx, 0.0)) for idx in selected},
        selected_by_role=selected_by_role,
    )


def _parse_fact_content(value: Any) -> Tuple[str, str, str] | None:
    if isinstance(value, tuple) and len(value) == 3:
        return tuple(str(item) for item in value)  # type: ignore[return-value]
    if isinstance(value, list) and len(value) == 3:
        return tuple(str(item) for item in value)  # type: ignore[return-value]
    if not isinstance(value, str):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, (tuple, list)) and len(parsed) == 3:
        return tuple(str(item) for item in parsed)  # type: ignore[return-value]
    return None


def fact_indices_and_tuples(system: Any, fact_indices: Sequence[int]) -> Tuple[List[int], List[Tuple[str, str, str]]]:
    fact_keys = [system.fact_node_keys[int(idx)] for idx in fact_indices]
    rows = system.fact_embedding_store.get_rows(fact_keys)
    kept_indices: List[int] = []
    facts: List[Tuple[str, str, str]] = []
    for fact_idx, key in zip(fact_indices, fact_keys):
        row = rows.get(key, {}) if isinstance(rows, Mapping) else {}
        fact = _parse_fact_content(row.get("content"))
        if fact is not None:
            kept_indices.append(int(fact_idx))
            facts.append(fact)
    return kept_indices, facts


def graph_search_with_role_entries(
    *,
    system: Any,
    query: str,
    query_fact_scores: np.ndarray,
    top_k_facts: Sequence[Tuple[str, str, str]],
    top_k_fact_indices: Sequence[int],
    passage_seed_scores: Mapping[int, float],
    link_top_k: int,
    passage_node_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run HippoRAG PPR from role-selected fact/entity and passage entries."""
    linking_score_map: Dict[str, float] = {}
    phrase_scores: Dict[str, List[float]] = {}
    phrase_weights = np.zeros(len(system.graph.vs["name"]))
    passage_weights = np.zeros(len(system.graph.vs["name"]))
    number_of_occurs = np.zeros(len(system.graph.vs["name"]))
    phrases_and_ids = set()

    for rank, fact in enumerate(top_k_facts):
        subject_phrase = str(fact[0]).lower()
        object_phrase = str(fact[2]).lower()
        fact_idx = int(top_k_fact_indices[rank])
        fact_score = (
            float(query_fact_scores[fact_idx])
            if np.asarray(query_fact_scores).ndim > 0 and fact_idx < len(query_fact_scores)
            else float(query_fact_scores)
        )

        for phrase in [subject_phrase, object_phrase]:
            phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
            phrase_id = system.node_name_to_vertex_idx.get(phrase_key, None)
            if phrase_id is not None:
                weighted_fact_score = fact_score
                if len(system.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                    weighted_fact_score /= len(system.ent_node_to_chunk_ids[phrase_key])
                phrase_weights[phrase_id] += weighted_fact_score
                number_of_occurs[phrase_id] += 1
            phrases_and_ids.add((phrase, phrase_id))

    nonzero_phrase_mask = number_of_occurs > 0
    phrase_weights[nonzero_phrase_mask] /= number_of_occurs[nonzero_phrase_mask]

    for phrase, phrase_id in phrases_and_ids:
        if phrase_id is None:
            continue
        phrase_scores.setdefault(phrase, []).append(float(phrase_weights[phrase_id]))

    for phrase, scores in phrase_scores.items():
        linking_score_map[phrase] = float(np.mean(scores))

    if link_top_k and phrase_scores:
        phrase_weights, linking_score_map = system.get_top_k_weights(
            link_top_k,
            phrase_weights,
            linking_score_map,
        )

    for passage_idx, passage_score in passage_seed_scores.items():
        if int(passage_idx) < 0 or int(passage_idx) >= len(system.passage_node_keys):
            continue
        passage_node_key = system.passage_node_keys[int(passage_idx)]
        passage_node_id = system.node_name_to_vertex_idx.get(passage_node_key)
        if passage_node_id is None:
            continue
        weighted_score = float(passage_score) * float(passage_node_weight)
        passage_weights[passage_node_id] = max(float(passage_weights[passage_node_id]), weighted_score)
        passage_node_text = system.chunk_embedding_store.get_row(passage_node_key)["content"]
        linking_score_map[passage_node_text] = weighted_score

    node_weights = phrase_weights + passage_weights
    if float(np.sum(node_weights)) <= 0.0:
        raise ValueError(f"No graph entry weights for query: {query}")

    return system.run_ppr(node_weights, damping=system.global_config.damping)


def role_aware_graph_entry_retrieve(
    *,
    system: Any,
    queries: Sequence[str],
    roles_by_query: Mapping[int, Sequence[Mapping[str, Any]]],
    query_indices: Sequence[int] | None = None,
    num_to_retrieve: int = 5,
    role_fact_top_k: int = 5,
    max_fact_seeds: int | None = None,
    role_passage_top_k: int = 5,
    max_passage_seeds: int | None = None,
    include_role_passage_entries: bool = True,
    entry_passage_node_weight: float | None = None,
    max_fact_seeds_mode: str = "fixed",
    link_top_k_mode: str = "config",
) -> List[RoleAwareRetrievalResult]:
    """Run one role-aware HippoRAG graph search per question.

    ``query_indices`` maps each query position to the role-cache query id.  If it
    is omitted, positions 0..N-1 are used.
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

    # The original question is still used for HippoRAG's dense passage component
    # inside graph_search_with_fact_entities. Role queries only choose fact seeds.
    system.get_query_embeddings(list(queries) + role_query_texts)

    if max_fact_seeds_mode not in {"fixed", "total_channel_budget"}:
        raise ValueError(f"Unsupported max_fact_seeds_mode: {max_fact_seeds_mode}")
    if link_top_k_mode not in {"config", "total_channel_budget"}:
        raise ValueError(f"Unsupported link_top_k_mode: {link_top_k_mode}")

    config_link_top_k = int(getattr(system.global_config, "linking_top_k", role_fact_top_k))
    default_fact_seed_budget = (
        int(max_fact_seeds)
        if max_fact_seeds is not None
        else config_link_top_k
    )
    passage_seed_budget = (
        int(max_passage_seeds)
        if max_passage_seeds is not None
        else max(int(role_passage_top_k) * 3, int(role_passage_top_k))
    )

    results: List[RoleAwareRetrievalResult] = []
    for query, qid, roles in zip(queries, qids, roles_by_position):
        role_queries = [role_graph_entry_query(question=query, role=role) for role in roles]
        effective_fact_seed_budget = (
            len(roles) * config_link_top_k
            if max_fact_seeds_mode == "total_channel_budget"
            else default_fact_seed_budget
        )
        effective_link_top_k = (
            len(roles) * config_link_top_k
            if link_top_k_mode == "total_channel_budget"
            else config_link_top_k
        )
        if not roles:
            results.append(
                RoleAwareRetrievalResult(
                    solution=QuerySolution(question=query, docs=[], doc_scores=np.asarray([], dtype=float)),
                    trace=RoleAwareRetrievalTrace(
                        question=query,
                        role_count=0,
                        role_queries=[],
                        selected_fact_indices=[],
                        selected_passage_indices=[],
                        selected_by_role={},
                        selected_passages_by_role={},
                        empty_reason="no_valid_roles",
                        graph_search_count=0,
                        fact_seed_budget=0,
                        link_top_k_effective=0,
                    ),
                )
            )
            continue

        role_fact_scores = {
            role["role_id"]: system.get_fact_scores(role_graph_entry_query(question=query, role=role))
            for role in roles
        }
        rankings = build_role_fact_rankings(
            question=query,
            roles=roles,
            role_fact_scores=role_fact_scores,
            role_fact_top_k=role_fact_top_k,
        )
        seeds = merge_role_fact_seeds(
            rankings=rankings,
            num_facts=len(system.fact_node_keys),
            max_fact_seeds=effective_fact_seed_budget,
        )
        top_k_fact_indices, top_k_facts = fact_indices_and_tuples(system, seeds.fact_indices)
        passage_seeds = RoleAwarePassageSeeds(passage_indices=[], passage_scores={}, selected_by_role={})
        if include_role_passage_entries:
            role_passage_rankings = {
                role["role_id"]: system.dense_passage_retrieval(role_graph_entry_query(question=query, role=role))
                for role in roles
            }
            passage_rankings = build_role_passage_rankings(
                question=query,
                roles=roles,
                role_passage_rankings=role_passage_rankings,
                role_passage_top_k=role_passage_top_k,
            )
            passage_seeds = merge_role_passage_seeds(
                rankings=passage_rankings,
                max_passage_seeds=passage_seed_budget,
            )

        if not top_k_facts and not passage_seeds.passage_indices:
            results.append(
                RoleAwareRetrievalResult(
                    solution=QuerySolution(question=query, docs=[], doc_scores=np.asarray([], dtype=float)),
                    trace=RoleAwareRetrievalTrace(
                        question=query,
                        role_count=len(roles),
                        role_queries=role_queries,
                        selected_fact_indices=seeds.fact_indices,
                        selected_passage_indices=passage_seeds.passage_indices,
                        selected_by_role=seeds.selected_by_role,
                        selected_passages_by_role=passage_seeds.selected_by_role,
                        empty_reason="no_fact_seeds",
                        graph_search_count=0,
                        fact_seed_budget=effective_fact_seed_budget,
                        link_top_k_effective=effective_link_top_k,
                    ),
                )
            )
            continue

        if include_role_passage_entries:
            sorted_doc_ids, sorted_doc_scores = graph_search_with_role_entries(
                system=system,
                query=query,
                link_top_k=effective_link_top_k,
                query_fact_scores=seeds.merged_fact_scores,
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                passage_seed_scores=passage_seeds.passage_scores,
                passage_node_weight=(
                    float(entry_passage_node_weight)
                    if entry_passage_node_weight is not None
                    else float(system.global_config.passage_node_weight)
                ),
            )
        else:
            sorted_doc_ids, sorted_doc_scores = system.graph_search_with_fact_entities(
                query=query,
                link_top_k=effective_link_top_k,
                query_fact_scores=seeds.merged_fact_scores,
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                passage_node_weight=system.global_config.passage_node_weight,
            )
        top_k_docs = [
            system.chunk_embedding_store.get_row(system.passage_node_keys[int(idx)])["content"]
            for idx in sorted_doc_ids[: int(num_to_retrieve)]
        ]
        results.append(
            RoleAwareRetrievalResult(
                solution=QuerySolution(
                    question=query,
                    docs=top_k_docs,
                    doc_scores=sorted_doc_scores[: int(num_to_retrieve)],
                ),
                trace=RoleAwareRetrievalTrace(
                    question=query,
                    role_count=len(roles),
                    role_queries=role_queries,
                    selected_fact_indices=seeds.fact_indices,
                    selected_passage_indices=passage_seeds.passage_indices,
                    selected_by_role=seeds.selected_by_role,
                    selected_passages_by_role=passage_seeds.selected_by_role,
                    graph_search_count=1,
                    fact_seed_budget=effective_fact_seed_budget,
                    link_top_k_effective=effective_link_top_k,
                ),
            )
        )

    return results
