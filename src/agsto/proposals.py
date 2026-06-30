"""Native STO proposal generation for AG-STO.

The clean retriever can consume cached transition-report proposal rows for
reproduction. This module provides the package-owned fallback path: build
query-local STO proposals directly from the lightweight corpus index.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .config import AGSTOConfig
from .index import content_tokens, informative_endpoint
from .lexical import score_docs_bm25
from .proposal_roles import build_proposal_role_records, proposal_role_summary_by_doc
from .ranking import unique_ranked
from .scoring import doc_query_covered_tokens, idf_sum, score_anchor_guided_evidence_set, transition_weight_to_support_set


def _ranked_bm25_docs(*, query: str, corpus_index: Mapping[str, Any]) -> Tuple[List[int], Dict[int, float]]:
    bm25_scores = score_docs_bm25(query=query, corpus_index=corpus_index)
    ranked = [
        int(doc_idx)
        for doc_idx, _score in sorted(bm25_scores.items(), key=lambda item: (-float(item[1]), int(item[0])))
    ]
    return ranked, bm25_scores


def _retrieve_pairwise_transition_emission(
    *,
    query: str,
    corpus_index: Mapping[str, Any],
    bm25_anchor_k: int,
    max_endpoint_degree: int,
    top_k: int,
    anchor_doc_indices: Sequence[int] | None = None,
    residual_aware: bool = False,
    semantic_residual_aware: bool = False,
    semantic_query_embedding: np.ndarray | None = None,
    chunk_embedding_matrix: np.ndarray | None = None,
    semantic_residual_weight: float = 8.0,
    specificity_regularized: bool = False,
) -> Dict[str, Any]:
    """Package-owned port of the original transition pair emitter."""

    units = corpus_index["units"]
    doc_to_units = corpus_index["doc_to_units"]
    endpoint_to_units = corpus_index["endpoint_to_units"]
    endpoint_idf: Mapping[str, float] = corpus_index["endpoint_idf"]
    doc_token_idf = corpus_index["doc_token_idf"]
    query_tokens = content_tokens(query)
    bm25_scores = score_docs_bm25(query=query, corpus_index=corpus_index)
    if anchor_doc_indices:
        anchor_docs = list(dict.fromkeys(int(doc_idx) for doc_idx in anchor_doc_indices if int(doc_idx) >= 0))[
            : max(bm25_anchor_k, 1)
        ]
    else:
        anchor_docs = [
            int(doc_idx)
            for doc_idx, _score in sorted(bm25_scores.items(), key=lambda item: (-float(item[1]), int(item[0])))[
                : max(bm25_anchor_k, 1)
            ]
        ]

    pair_rows: List[Dict[str, Any]] = []
    skipped_endpoints = 0
    for anchor_doc in anchor_docs:
        semantic_residual_embedding: np.ndarray | None = None
        if semantic_residual_aware and semantic_query_embedding is not None and chunk_embedding_matrix is not None:
            query_vec = np.asarray(semantic_query_embedding, dtype=np.float32)
            anchor_vec = np.asarray(chunk_embedding_matrix[int(anchor_doc)], dtype=np.float32)
            projection = max(float(query_vec @ anchor_vec), 0.0)
            residual_vec = query_vec - projection * anchor_vec
            residual_norm = float(np.linalg.norm(residual_vec))
            if residual_norm > 1e-12:
                semantic_residual_embedding = residual_vec / residual_norm
            else:
                semantic_residual_embedding = query_vec

        anchor_endpoint_weights: Counter[str] = Counter()
        for unit_id in doc_to_units.get(int(anchor_doc), []) or []:
            unit = units[int(unit_id)]
            unit_query_overlap = query_tokens & set(unit.get("_tokens", []) or [])
            unit_weight = 1.0 + idf_sum(unit_query_overlap, doc_token_idf)
            for endpoint in unit.get("_endpoints", []) or []:
                endpoint_norm = informative_endpoint(endpoint)
                if endpoint_norm:
                    anchor_endpoint_weights[endpoint_norm] += unit_weight

        completion_scores: Counter[int] = Counter()
        completion_specificity_scores: Counter[int] = Counter()
        completion_hub_scores: Counter[int] = Counter()
        completion_endpoint_sets: Dict[int, Set[str]] = defaultdict(set)
        anchor_query_tokens = doc_query_covered_tokens(
            doc_idx=anchor_doc,
            query_tokens=query_tokens,
            corpus_index=corpus_index,
        )
        for endpoint, endpoint_weight in sorted(anchor_endpoint_weights.items()):
            neighbor_units = list(endpoint_to_units.get(endpoint, []) or [])
            if len(neighbor_units) > max_endpoint_degree:
                skipped_endpoints += 1
                continue
            specificity_credit = (
                math.log1p(endpoint_weight)
                * float(endpoint_idf.get(endpoint, 1.0))
                / math.log1p(max(len(neighbor_units), 2))
            )
            hub_credit = math.log1p(max(len(neighbor_units), 1))
            for unit_id in neighbor_units:
                doc_idx = int(units[int(unit_id)].get("doc_index", -1))
                if doc_idx < 0 or doc_idx == anchor_doc:
                    continue
                completion_scores[doc_idx] += math.log1p(endpoint_weight)
                completion_specificity_scores[doc_idx] += specificity_credit
                completion_hub_scores[doc_idx] += hub_credit
                completion_endpoint_sets[doc_idx].add(endpoint)

        best_completion = None
        best_completion_score = 0.0
        best_semantic_score = 0.0
        for doc_idx, transition_score in sorted(completion_scores.items()):
            completion_query_tokens = doc_query_covered_tokens(
                doc_idx=doc_idx,
                query_tokens=query_tokens,
                corpus_index=corpus_index,
            )
            residual_tokens = completion_query_tokens - anchor_query_tokens
            pair_covered_tokens = anchor_query_tokens | completion_query_tokens
            residual_score = idf_sum(residual_tokens, doc_token_idf)
            pair_coverage_score = idf_sum(pair_covered_tokens, doc_token_idf)
            score = (
                0.55 * float(transition_score)
                + 0.35 * float(bm25_scores.get(doc_idx, 0.0))
                + 0.2 * math.log1p(len(completion_endpoint_sets.get(doc_idx, set())))
            )
            if residual_aware:
                score += 1.15 * residual_score + 0.25 * pair_coverage_score
            if specificity_regularized:
                score += 0.35 * float(completion_specificity_scores.get(doc_idx, 0.0))
                score -= 0.03 * float(completion_hub_scores.get(doc_idx, 0.0))
            semantic_score = 0.0
            if semantic_residual_embedding is not None and chunk_embedding_matrix is not None:
                completion_vec = np.asarray(chunk_embedding_matrix[int(doc_idx)], dtype=np.float32)
                semantic_score = float(semantic_residual_embedding @ completion_vec)
                score += float(semantic_residual_weight) * semantic_score
            if best_completion is None or score > best_completion_score or (
                score == best_completion_score and doc_idx < best_completion
            ):
                best_completion = int(doc_idx)
                best_completion_score = float(score)
                best_semantic_score = float(semantic_score)
        pair_rows.append(
            {
                "anchor_doc": int(anchor_doc),
                "completion_doc": best_completion,
                "completion_score": round(best_completion_score, 6),
                "anchor_bm25_score": round(float(bm25_scores.get(anchor_doc, 0.0)), 6),
                "residual_aware": bool(residual_aware),
                "semantic_residual_aware": bool(semantic_residual_aware),
                "specificity_regularized": bool(specificity_regularized),
                "specificity_score": round(float(completion_specificity_scores.get(best_completion, 0.0)), 6)
                if best_completion is not None
                else 0.0,
                "hub_score": round(float(completion_hub_scores.get(best_completion, 0.0)), 6)
                if best_completion is not None
                else 0.0,
                "semantic_residual_score": round(float(best_semantic_score), 6) if best_completion is not None else 0.0,
            }
        )

    emitted: List[int] = []
    seen: Set[int] = set()
    for pair in pair_rows:
        for doc_idx in (pair.get("anchor_doc"), pair.get("completion_doc")):
            if doc_idx is None:
                continue
            doc_idx = int(doc_idx)
            if doc_idx in seen:
                continue
            emitted.append(doc_idx)
            seen.add(doc_idx)
            if len(emitted) >= top_k:
                break
        if len(emitted) >= top_k:
            break

    for doc_idx, _score in sorted(bm25_scores.items(), key=lambda item: (-float(item[1]), int(item[0]))):
        if doc_idx in seen:
            continue
        emitted.append(int(doc_idx))
        seen.add(int(doc_idx))
        if len(emitted) >= top_k:
            break

    return {
        "retrieved_doc_indices": emitted,
        "bm25_anchor_doc_indices": anchor_docs,
        "support_pairs": pair_rows,
        "skipped_endpoint_count": skipped_endpoints,
    }


def _rank_endpoint_specific_docs(
    *,
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
    bm25_scores: Mapping[int, float],
    limit: int,
) -> List[int]:
    doc_endpoint_tokens: Mapping[int, Sequence[str]] = corpus_index["doc_endpoint_tokens"]
    doc_to_endpoints: Mapping[int, Sequence[str]] = corpus_index["doc_to_endpoints"]
    endpoint_idf: Mapping[str, float] = corpus_index["endpoint_idf"]
    doc_token_idf: Mapping[str, float] = corpus_index["doc_token_idf"]

    rows: List[Tuple[float, float, int]] = []
    for raw_doc_idx, endpoint_tokens in doc_endpoint_tokens.items():
        doc_idx = int(raw_doc_idx)
        overlap = query_tokens & set(endpoint_tokens or [])
        if not overlap:
            continue
        endpoint_score = 0.0
        for endpoint in doc_to_endpoints.get(doc_idx, []) or []:
            if query_tokens & content_tokens(endpoint):
                endpoint_score += float(endpoint_idf.get(str(endpoint), 1.0))
        token_score = idf_sum(overlap, doc_token_idf)
        rows.append(
            (
                float(endpoint_score + token_score),
                float(bm25_scores.get(doc_idx, 0.0)),
                doc_idx,
            )
        )
    rows.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [doc_idx for _endpoint_score, _bm25_score, doc_idx in rows[: max(int(limit), 0)]]


def _rank_anchor_neighborhood_docs(
    *,
    stable_anchor_doc_indices: Sequence[int],
    corpus_index: Mapping[str, Any],
    bm25_scores: Mapping[int, float],
    max_endpoint_degree: int,
    limit: int,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    if not stable_anchor_doc_indices:
        return [], []

    doc_to_endpoints: Mapping[int, Sequence[str]] = corpus_index["doc_to_endpoints"]
    endpoint_to_docs: Mapping[str, Sequence[int]] = corpus_index["endpoint_to_docs"]
    candidate_docs: Set[int] = set()
    stable_anchor_set = {int(doc_idx) for doc_idx in stable_anchor_doc_indices}
    for anchor_doc in stable_anchor_doc_indices:
        for endpoint in doc_to_endpoints.get(int(anchor_doc), []) or []:
            docs_for_endpoint = endpoint_to_docs.get(str(endpoint), []) or []
            if len(docs_for_endpoint) > max_endpoint_degree:
                continue
            candidate_docs.update(int(doc_idx) for doc_idx in docs_for_endpoint)
    candidate_docs.difference_update(stable_anchor_set)

    rows: List[Tuple[float, float, int, List[str]]] = []
    for doc_idx in candidate_docs:
        transition_weight, shared_endpoints = transition_weight_to_support_set(
            doc_idx=int(doc_idx),
            support_docs=stable_anchor_doc_indices,
            corpus_index=corpus_index,
            max_endpoint_degree=max_endpoint_degree,
        )
        if transition_weight <= 0.0:
            continue
        rows.append(
            (
                float(transition_weight),
                float(bm25_scores.get(int(doc_idx), 0.0)),
                int(doc_idx),
                shared_endpoints,
            )
        )
    rows.sort(key=lambda item: (-item[0], -item[1], item[2]))
    ranked_docs = [doc_idx for _transition, _bm25, doc_idx, _endpoints in rows[: max(int(limit), 0)]]
    boundary_pairs = [
        {
            "anchor_docs": [int(doc_idx) for doc_idx in stable_anchor_doc_indices],
            "completion_doc": int(doc_idx),
            "direct_transition_score": round(float(transition), 6),
            "shared_endpoints_sample": list(shared_endpoints[:5]),
        }
        for transition, _bm25, doc_idx, shared_endpoints in rows[: max(int(limit), 0)]
    ]
    return ranked_docs, boundary_pairs


def _build_connected_support_search(
    *,
    query_tokens: Set[str],
    stable_anchor_doc_indices: Sequence[int],
    candidate_doc_indices: Sequence[int],
    corpus_index: Mapping[str, Any],
    bm25_scores: Mapping[int, float],
    max_endpoint_degree: int,
    limit: int,
) -> Dict[str, Any]:
    if not stable_anchor_doc_indices:
        return {"retrieved_doc_indices": [], "candidate_doc_indices": [], "top_support_sets": []}

    doc_token_idf: Mapping[str, float] = corpus_index["doc_token_idf"]
    anchor_docs = [int(doc_idx) for doc_idx in stable_anchor_doc_indices]
    anchor_set = set(anchor_docs)
    rows: List[Tuple[float, List[int], List[str]]] = []
    for doc_idx in unique_ranked(candidate_doc_indices):
        doc_idx = int(doc_idx)
        if doc_idx in anchor_set:
            continue
        transition_weight, shared_endpoints = transition_weight_to_support_set(
            doc_idx=doc_idx,
            support_docs=anchor_docs,
            corpus_index=corpus_index,
            max_endpoint_degree=max_endpoint_degree,
        )
        if transition_weight <= 0.0:
            continue
        covered = doc_query_covered_tokens(doc_idx=doc_idx, query_tokens=query_tokens, corpus_index=corpus_index)
        score = float(transition_weight) + float(bm25_scores.get(doc_idx, 0.0)) + idf_sum(covered, doc_token_idf)
        rows.append((score, unique_ranked([anchor_docs[0], doc_idx]), shared_endpoints))

    rows.sort(key=lambda item: (-item[0], item[1]))
    top_support_sets = [
        {
            "doc_indices": list(doc_indices),
            "score": round(float(score), 6),
            "shared_endpoints_sample": list(shared_endpoints[:5]),
        }
        for score, doc_indices, shared_endpoints in rows[: max(int(limit), 0)]
    ]
    retrieved_docs = unique_ranked(
        doc_idx
        for support_set in top_support_sets
        for doc_idx in support_set.get("doc_indices", []) or []
    )
    return {
        "retrieved_doc_indices": retrieved_docs,
        "candidate_doc_indices": unique_ranked(candidate_doc_indices),
        "top_support_sets": top_support_sets,
    }


def _support_mct_anchor_prior(stable_anchor_doc_indices: Sequence[int]) -> Dict[int, float]:
    return {
        int(doc_idx): 1.0 / math.sqrt(float(rank))
        for rank, doc_idx in enumerate(unique_ranked(stable_anchor_doc_indices), start=1)
    }


def _support_mct_candidate_priority(
    *,
    doc_idx: int,
    current_docs: Sequence[int],
    current_covered_tokens: Set[str],
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
    bm25_scores: Mapping[int, float],
    anchor_prior: Mapping[int, float],
    max_endpoint_degree: int,
) -> float:
    covered = doc_query_covered_tokens(doc_idx=int(doc_idx), query_tokens=query_tokens, corpus_index=corpus_index)
    transition_weight = 0.0
    if current_docs:
        transition_weight, _ = transition_weight_to_support_set(
            doc_idx=int(doc_idx),
            support_docs=current_docs,
            corpus_index=corpus_index,
            max_endpoint_degree=max_endpoint_degree,
        )
    return (
        idf_sum(covered - current_covered_tokens, corpus_index["doc_token_idf"])
        + 0.65 * math.log1p(max(float(transition_weight), 0.0))
        + 0.35 * math.log1p(max(float(bm25_scores.get(int(doc_idx), 0.0)), 0.0))
        + 0.75 * float(anchor_prior.get(int(doc_idx), 0.0))
    )


def _support_mct_frontier(
    *,
    current_docs: Sequence[int],
    candidate_doc_indices: Sequence[int],
    stable_anchor_doc_indices: Sequence[int],
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
    bm25_scores: Mapping[int, float],
    anchor_prior: Mapping[int, float],
    max_endpoint_degree: int,
    frontier_limit: int,
) -> List[int]:
    current_docs = unique_ranked(current_docs)
    current_set = set(current_docs)
    if not current_docs:
        seeds = unique_ranked(stable_anchor_doc_indices) or unique_ranked(candidate_doc_indices)
        return seeds[: max(int(frontier_limit), 1)]

    current_covered_tokens: Set[str] = set()
    for doc_idx in current_docs:
        current_covered_tokens.update(
            doc_query_covered_tokens(doc_idx=int(doc_idx), query_tokens=query_tokens, corpus_index=corpus_index)
        )
    rows: List[Tuple[float, int]] = []
    for doc_idx in unique_ranked(candidate_doc_indices):
        doc_idx = int(doc_idx)
        if doc_idx in current_set:
            continue
        transition_weight, _ = transition_weight_to_support_set(
            doc_idx=doc_idx,
            support_docs=current_docs,
            corpus_index=corpus_index,
            max_endpoint_degree=max_endpoint_degree,
        )
        if transition_weight <= 0.0:
            continue
        rows.append(
            (
                _support_mct_candidate_priority(
                    doc_idx=doc_idx,
                    current_docs=current_docs,
                    current_covered_tokens=current_covered_tokens,
                    query_tokens=query_tokens,
                    corpus_index=corpus_index,
                    bm25_scores=bm25_scores,
                    anchor_prior=anchor_prior,
                    max_endpoint_degree=max_endpoint_degree,
                ),
                doc_idx,
            )
        )
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [doc_idx for _score, doc_idx in rows[: max(int(frontier_limit), 1)]]


def _score_support_mct_state(
    *,
    doc_indices: Sequence[int],
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
    bm25_scores: Mapping[int, float],
    anchor_prior: Mapping[int, float],
    max_endpoint_degree: int,
) -> Dict[str, Any]:
    return score_anchor_guided_evidence_set(
        doc_indices=unique_ranked(doc_indices),
        query_tokens=query_tokens,
        corpus_index=corpus_index,
        bm25_scores=bm25_scores,
        anchor_prior=anchor_prior,
        specificity_prior={},
        support_prior={},
        neighborhood_prior={},
        auxiliary_prior={},
        max_endpoint_degree=max_endpoint_degree,
        chunk_embedding_matrix=None,
    )


def _complete_support_mct_state(
    *,
    doc_indices: Sequence[int],
    candidate_doc_indices: Sequence[int],
    stable_anchor_doc_indices: Sequence[int],
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
    bm25_scores: Mapping[int, float],
    anchor_prior: Mapping[int, float],
    max_endpoint_degree: int,
    max_depth: int,
    frontier_limit: int,
) -> List[int]:
    completed = unique_ranked(doc_indices)
    while len(completed) < max(int(max_depth), 1):
        frontier = _support_mct_frontier(
            current_docs=completed,
            candidate_doc_indices=candidate_doc_indices,
            stable_anchor_doc_indices=stable_anchor_doc_indices,
            query_tokens=query_tokens,
            corpus_index=corpus_index,
            bm25_scores=bm25_scores,
            anchor_prior=anchor_prior,
            max_endpoint_degree=max_endpoint_degree,
            frontier_limit=frontier_limit,
        )
        next_doc = next((int(doc_idx) for doc_idx in frontier if int(doc_idx) not in set(completed)), None)
        if next_doc is None:
            break
        completed.append(int(next_doc))
    return completed


def _build_mct_support_search(
    *,
    query_tokens: Set[str],
    stable_anchor_doc_indices: Sequence[int],
    candidate_doc_indices: Sequence[int],
    corpus_index: Mapping[str, Any],
    bm25_scores: Mapping[int, float],
    max_endpoint_degree: int,
    candidate_limit: int,
    support_set_size: int,
    support_beam_size: int,
    iterations: int,
    exploration: float,
    top_k: int,
) -> Dict[str, Any]:
    """Search support-set proposals with deterministic Monte Carlo tree search."""

    candidate_docs = unique_ranked(candidate_doc_indices)[: max(int(candidate_limit), int(top_k), 1)]
    stable_anchors = unique_ranked(stable_anchor_doc_indices)
    if not candidate_docs and not stable_anchors:
        return {"retrieved_doc_indices": [], "candidate_doc_indices": [], "top_support_sets": []}

    max_depth = max(int(support_set_size), 1)
    frontier_limit = max(int(support_beam_size) * 4, max_depth, 1)
    anchor_prior = _support_mct_anchor_prior(stable_anchors)
    nodes: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    terminal_scores: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    def node_for(doc_indices: Sequence[int]) -> Dict[str, Any]:
        docs = unique_ranked(doc_indices)
        key = tuple(sorted(docs))
        if key not in nodes:
            nodes[key] = {
                "doc_indices": docs,
                "visits": 0,
                "total_value": 0.0,
                "children": [],
                "unexpanded": _support_mct_frontier(
                    current_docs=docs,
                    candidate_doc_indices=candidate_docs,
                    stable_anchor_doc_indices=stable_anchors,
                    query_tokens=query_tokens,
                    corpus_index=corpus_index,
                    bm25_scores=bm25_scores,
                    anchor_prior=anchor_prior,
                    max_endpoint_degree=max_endpoint_degree,
                    frontier_limit=frontier_limit,
                ),
            }
        return nodes[key]

    root_docs = stable_anchors[:1]
    root = node_for(root_docs)
    for _iteration in range(max(int(iterations), 1)):
        path = [root]
        node = root
        while (
            len(node["doc_indices"]) < max_depth
            and not node["unexpanded"]
            and node["children"]
        ):
            parent_visits = max(int(node["visits"]), 1)

            def child_key(child: Mapping[str, Any]) -> Tuple[float, float, Tuple[int, ...]]:
                visits = int(child["visits"])
                mean_value = float(child["total_value"]) / float(max(visits, 1))
                exploration_bonus = float(exploration) * math.sqrt(math.log(float(parent_visits + 1)) / float(visits + 1))
                return (mean_value + exploration_bonus, mean_value, tuple(-int(doc_idx) for doc_idx in child["doc_indices"]))

            node = max(node["children"], key=child_key)
            path.append(node)

        if len(node["doc_indices"]) < max_depth and node["unexpanded"]:
            action_doc = int(node["unexpanded"].pop(0))
            child_docs = unique_ranked(list(node["doc_indices"]) + [action_doc])
            child = node_for(child_docs)
            if child not in node["children"]:
                node["children"].append(child)
            node = child
            path.append(node)

        rollout_docs = _complete_support_mct_state(
            doc_indices=node["doc_indices"],
            candidate_doc_indices=candidate_docs,
            stable_anchor_doc_indices=stable_anchors,
            query_tokens=query_tokens,
            corpus_index=corpus_index,
            bm25_scores=bm25_scores,
            anchor_prior=anchor_prior,
            max_endpoint_degree=max_endpoint_degree,
            max_depth=max_depth,
            frontier_limit=frontier_limit,
        )
        scored = _score_support_mct_state(
            doc_indices=rollout_docs,
            query_tokens=query_tokens,
            corpus_index=corpus_index,
            bm25_scores=bm25_scores,
            anchor_prior=anchor_prior,
            max_endpoint_degree=max_endpoint_degree,
        )
        value = float(scored.get("score", 0.0) or 0.0)
        terminal_key = tuple(sorted(unique_ranked(rollout_docs)))
        previous = terminal_scores.get(terminal_key)
        if previous is None or value > float(previous.get("score", 0.0) or 0.0):
            terminal_scores[terminal_key] = scored
        for path_node in path:
            path_node["visits"] = int(path_node["visits"]) + 1
            path_node["total_value"] = float(path_node["total_value"]) + value

    output_set_limit = max(int(top_k), 1)
    top_support_sets = [
        dict(score)
        for score in sorted(
            terminal_scores.values(),
            key=lambda item: (-float(item.get("score", 0.0) or 0.0), item.get("doc_indices", [])),
        )[:output_set_limit]
    ]
    retrieved_docs = unique_ranked(
        doc_idx
        for support_set in top_support_sets
        for doc_idx in support_set.get("doc_indices", []) or []
    )[: max(int(top_k), 0)]
    return {
        "retrieved_doc_indices": retrieved_docs,
        "candidate_doc_indices": candidate_docs,
        "top_support_sets": top_support_sets,
        "support_realization": "mct",
        "mct_iterations": int(max(int(iterations), 1)),
        "mct_depth": int(max_depth),
        "mct_exploration": round(float(exploration), 6),
    }


def build_native_sto_proposals(
    *,
    query: str,
    corpus_index: Mapping[str, Any],
    config: AGSTOConfig,
    anchor_doc_indices: Sequence[int] | None = None,
    native_dense_doc_indices: Sequence[int] | None = None,
    semantic_query_embedding: np.ndarray | None = None,
    chunk_embedding_matrix: np.ndarray | None = None,
    semantic_residual_weight: float = 8.0,
    support_realization: str = "direct",
    support_mct_iterations: int = 64,
    support_mct_depth: int = 2,
    support_mct_exploration: float = 1.0,
) -> Dict[str, Any]:
    """Build query-local AG-STO proposals directly from the STO index.

    The returned keys can be passed directly to ``AGSTORetriever.retrieve``.
    This is not a new method policy; it is the native proposal source for the
    same base/graph selector.
    """

    query_tokens = content_tokens(query)
    proposal_depth = max(
        int(config.proposal_candidate_depth),
        int(config.retrieval_top_k),
        int(config.evidence_set_size),
    )
    support_depth = max(int(config.support_proposal_depth), int(config.evidence_set_size))
    bm25_ranked, bm25_scores = _ranked_bm25_docs(query=query, corpus_index=corpus_index)
    dense_anchors = unique_ranked(anchor_doc_indices or native_dense_doc_indices or bm25_ranked)[:proposal_depth]
    stable_anchors = dense_anchors[: max(int(config.stable_anchor_k), 0)]
    has_semantic_pair_inputs = semantic_query_embedding is not None and chunk_embedding_matrix is not None
    support_realization = str(support_realization or "direct")
    if support_realization not in {"direct", "mct"}:
        raise ValueError(f"Unsupported support_realization={support_realization!r}. Use 'direct' or 'mct'.")

    specificity_pair_emission = _retrieve_pairwise_transition_emission(
        query=query,
        corpus_index=corpus_index,
        bm25_anchor_k=max(int(config.stable_anchor_k), 1),
        max_endpoint_degree=config.max_endpoint_degree,
        top_k=proposal_depth,
        anchor_doc_indices=stable_anchors,
        semantic_residual_aware=has_semantic_pair_inputs,
        semantic_query_embedding=semantic_query_embedding,
        chunk_embedding_matrix=chunk_embedding_matrix,
        semantic_residual_weight=semantic_residual_weight,
        specificity_regularized=True,
    )
    specificity_docs = unique_ranked(
        specificity_pair_emission.get("retrieved_doc_indices", []) or []
    )[:proposal_depth]
    endpoint_transition_docs, boundary_pairs = _rank_anchor_neighborhood_docs(
        stable_anchor_doc_indices=stable_anchors,
        corpus_index=corpus_index,
        bm25_scores=bm25_scores,
        max_endpoint_degree=config.max_endpoint_degree,
        limit=proposal_depth,
    )
    hybrid_residual_docs: List[int] = []
    hybrid_residual_emission: Dict[str, Any] = {}
    if has_semantic_pair_inputs:
        hybrid_residual_emission = _retrieve_pairwise_transition_emission(
            query=query,
            corpus_index=corpus_index,
            bm25_anchor_k=max(int(config.stable_anchor_k), 1),
            max_endpoint_degree=config.max_endpoint_degree,
            top_k=proposal_depth,
            anchor_doc_indices=stable_anchors,
            residual_aware=True,
            semantic_residual_aware=True,
            semantic_query_embedding=semantic_query_embedding,
            chunk_embedding_matrix=chunk_embedding_matrix,
            semantic_residual_weight=semantic_residual_weight,
        )
        hybrid_residual_docs = unique_ranked(
            hybrid_residual_emission.get("retrieved_doc_indices", []) or []
        )[:proposal_depth]
    neighborhood_docs = unique_ranked(
        list(stable_anchors) + list(endpoint_transition_docs) + list(specificity_docs)
    )[:proposal_depth]
    support_candidates = unique_ranked(list(endpoint_transition_docs) + list(specificity_docs) + list(bm25_ranked))
    if support_realization == "mct":
        support_search = _build_mct_support_search(
            query_tokens=query_tokens,
            stable_anchor_doc_indices=stable_anchors,
            candidate_doc_indices=support_candidates,
            corpus_index=corpus_index,
            bm25_scores=bm25_scores,
            max_endpoint_degree=config.max_endpoint_degree,
            candidate_limit=max(int(config.candidate_limit), proposal_depth, support_depth),
            support_set_size=max(int(support_mct_depth), 1),
            support_beam_size=max(int(config.beam_size), support_depth, 1),
            iterations=max(int(support_mct_iterations), 1),
            exploration=max(float(support_mct_exploration), 0.0),
            top_k=support_depth,
        )
    else:
        support_search = _build_connected_support_search(
            query_tokens=query_tokens,
            stable_anchor_doc_indices=stable_anchors,
            candidate_doc_indices=support_candidates,
            corpus_index=corpus_index,
            bm25_scores=bm25_scores,
            max_endpoint_degree=config.max_endpoint_degree,
            limit=support_depth,
        )
        support_search["support_realization"] = "direct"
    query_conditioned_neighborhood = {
        "selected_neighborhood_doc_indices": neighborhood_docs[: max(int(config.evidence_set_size), 1)],
        "retrieved_doc_indices": neighborhood_docs,
        "boundary_readout_pairs": boundary_pairs,
    }
    proposal_role_records = build_proposal_role_records(
        specificity_pairwise_transition_emission=specificity_pair_emission,
        hybrid_residual_pairwise_transition_emission=hybrid_residual_emission,
        endpoint_transition_emission=boundary_pairs,
        support_set_search=support_search,
        query_conditioned_neighborhood=query_conditioned_neighborhood,
        anchor_doc_indices=stable_anchors,
        native_dense_doc_indices=dense_anchors,
    )

    return {
        "anchor_doc_indices": list(stable_anchors),
        "native_dense_doc_indices": list(dense_anchors),
        "bm25_doc_indices": bm25_ranked[:proposal_depth],
        "specificity_doc_indices": specificity_docs,
        "endpoint_transition_doc_indices": endpoint_transition_docs,
        "hybrid_residual_doc_indices": hybrid_residual_docs,
        "query_conditioned_neighborhood": query_conditioned_neighborhood,
        "support_set_search": support_search,
        "proposal_role_records": proposal_role_records,
        "proposal_role_summary_by_doc": proposal_role_summary_by_doc(proposal_role_records),
    }
