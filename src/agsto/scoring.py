"""Evidence-set scoring utilities for AG-STO."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .ranking import unique_ranked


MISSING_TREE_EDGE_PENALTY = 2.20


def idf_sum(tokens: Iterable[str], token_idf: Mapping[str, float]) -> float:
    return float(sum(float(token_idf.get(token, 1.0)) for token in set(tokens)))


def doc_query_covered_tokens(
    *,
    doc_idx: int,
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
) -> Set[str]:
    doc_token_counts: Mapping[int, Counter[str]] = corpus_index["doc_token_counts"]
    return set(doc_token_counts.get(int(doc_idx), Counter()).keys()) & query_tokens


def doc_transition_details(
    *,
    left_doc: int,
    right_doc: int,
    corpus_index: Mapping[str, Any],
    max_endpoint_degree: int,
) -> Tuple[float, List[str]]:
    """Return strong STO transition weight between two passages."""

    if int(left_doc) == int(right_doc):
        return 0.0, []
    doc_to_endpoints: Mapping[int, Sequence[str]] = corpus_index["doc_to_endpoints"]
    endpoint_to_docs: Mapping[str, Sequence[int]] = corpus_index["endpoint_to_docs"]
    endpoint_idf: Mapping[str, float] = corpus_index["endpoint_idf"]
    left_endpoints = set(doc_to_endpoints.get(int(left_doc), []) or [])
    right_endpoints = set(doc_to_endpoints.get(int(right_doc), []) or [])
    shared = left_endpoints & right_endpoints
    if not shared:
        return 0.0, []

    usable: List[str] = []
    weight = 0.0
    for endpoint in sorted(shared):
        degree = len(endpoint_to_docs.get(endpoint, []) or [])
        if degree > max_endpoint_degree:
            continue
        usable.append(endpoint)
        weight += float(endpoint_idf.get(endpoint, 1.0))
    return float(weight), usable


def transition_weight_to_support_set(
    *,
    doc_idx: int,
    support_docs: Sequence[int],
    corpus_index: Mapping[str, Any],
    max_endpoint_degree: int,
) -> Tuple[float, List[str]]:
    total_weight = 0.0
    endpoints: Set[str] = set()
    for support_doc in support_docs:
        weight, shared = doc_transition_details(
            left_doc=int(doc_idx),
            right_doc=int(support_doc),
            corpus_index=corpus_index,
            max_endpoint_degree=max_endpoint_degree,
        )
        total_weight += weight
        endpoints.update(shared)
    return float(total_weight), sorted(endpoints)


def shared_endpoint_hub_pressure(
    *,
    doc_indices: Sequence[int],
    corpus_index: Mapping[str, Any],
    max_endpoint_degree: int,
) -> float:
    doc_to_endpoints: Mapping[int, Sequence[str]] = corpus_index["doc_to_endpoints"]
    endpoint_to_docs: Mapping[str, Sequence[int]] = corpus_index["endpoint_to_docs"]
    pressure = 0.0
    for doc_idx in doc_indices:
        for endpoint in doc_to_endpoints.get(int(doc_idx), []) or []:
            degree = len(endpoint_to_docs.get(str(endpoint), []) or [])
            if degree > max_endpoint_degree:
                pressure += math.log1p(float(degree - max_endpoint_degree))
    return float(pressure)


def semantic_set_redundancy(
    *,
    doc_indices: Sequence[int],
    chunk_embedding_matrix: np.ndarray | None,
) -> float:
    if chunk_embedding_matrix is None or len(doc_indices) < 2:
        return 0.0
    redundancy = 0.0
    valid_docs = [
        int(doc_idx)
        for doc_idx in doc_indices
        if 0 <= int(doc_idx) < int(chunk_embedding_matrix.shape[0])
    ]
    for pos, left_doc in enumerate(valid_docs):
        left_vec = np.asarray(chunk_embedding_matrix[left_doc], dtype=np.float32)
        for right_doc in valid_docs[pos + 1 :]:
            right_vec = np.asarray(chunk_embedding_matrix[right_doc], dtype=np.float32)
            redundancy += max(float(left_vec @ right_vec), 0.0)
    return float(redundancy)


def score_anchor_guided_evidence_set(
    *,
    doc_indices: Sequence[int],
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
    bm25_scores: Mapping[int, float],
    anchor_prior: Mapping[int, float],
    specificity_prior: Mapping[int, float],
    support_prior: Mapping[int, float],
    neighborhood_prior: Mapping[int, float],
    auxiliary_prior: Mapping[int, float] | None = None,
    max_endpoint_degree: int,
    chunk_embedding_matrix: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Score a graph-constrained evidence set over the STO passage graph."""

    doc_tuple = tuple(unique_ranked(doc_indices))
    doc_token_idf: Mapping[str, float] = corpus_index["doc_token_idf"]
    endpoint_to_docs: Mapping[str, Sequence[int]] = corpus_index["endpoint_to_docs"]
    covered_tokens: Set[str] = set()
    relevance = 0.0
    specificity = 0.0
    anchor_trust = 0.0
    support_agreement = 0.0
    neighborhood_agreement = 0.0
    auxiliary_agreement = 0.0
    lexical_redundancy = 0.0

    for doc_idx in doc_tuple:
        doc_tokens = doc_query_covered_tokens(doc_idx=int(doc_idx), query_tokens=query_tokens, corpus_index=corpus_index)
        lexical_redundancy += len(covered_tokens & doc_tokens)
        covered_tokens.update(doc_tokens)
        relevance += math.log1p(max(float(bm25_scores.get(int(doc_idx), 0.0)), 0.0))
        specificity += float(specificity_prior.get(int(doc_idx), 0.0))
        anchor_trust += float(anchor_prior.get(int(doc_idx), 0.0))
        support_agreement += float(support_prior.get(int(doc_idx), 0.0))
        neighborhood_agreement += float(neighborhood_prior.get(int(doc_idx), 0.0))
        auxiliary_agreement += float((auxiliary_prior or {}).get(int(doc_idx), 0.0))

    transition_edge_weights: List[float] = []
    shared_endpoints: Set[str] = set()
    transition_edge_count = 0
    for pos, left_doc in enumerate(doc_tuple):
        for right_doc in doc_tuple[pos + 1 :]:
            weight, endpoints = doc_transition_details(
                left_doc=int(left_doc),
                right_doc=int(right_doc),
                corpus_index=corpus_index,
                max_endpoint_degree=max_endpoint_degree,
            )
            if weight <= 0.0:
                continue
            transition_edge_count += 1
            transition_edge_weights.append(math.log1p(float(weight)))
            shared_endpoints.update(endpoints)

    needed_tree_edges = max(len(doc_tuple) - 1, 0)
    transition_tree_strength = sum(sorted(transition_edge_weights, reverse=True)[:needed_tree_edges])
    missing_tree_edges = max(needed_tree_edges - transition_edge_count, 0)
    coverage_idf = idf_sum(covered_tokens, doc_token_idf)
    endpoint_specificity = sum(
        float(corpus_index["endpoint_idf"].get(endpoint, 1.0))
        for endpoint in shared_endpoints
        if len(endpoint_to_docs.get(endpoint, []) or []) <= max_endpoint_degree
    )
    hub_pressure = shared_endpoint_hub_pressure(
        doc_indices=doc_tuple,
        corpus_index=corpus_index,
        max_endpoint_degree=max_endpoint_degree,
    )
    semantic_redundancy = semantic_set_redundancy(
        doc_indices=doc_tuple,
        chunk_embedding_matrix=chunk_embedding_matrix,
    )
    size_pressure = max(len(doc_tuple) - 2, 0)
    no_anchor_penalty = 0.75 if doc_tuple and anchor_trust <= 0.0 else 0.0
    query_utility = 0.60 * relevance + 1.75 * coverage_idf
    graph_support = (
        1.75 * transition_tree_strength
        + 0.55 * endpoint_specificity
        + 3.00 * specificity
        + 1.00 * anchor_trust
        + 1.25 * support_agreement
        + 2.00 * neighborhood_agreement
        + 2.00 * auxiliary_agreement
    )
    set_cost = (
        MISSING_TREE_EDGE_PENALTY * float(missing_tree_edges)
        + 0.45 * lexical_redundancy
        + 0.55 * semantic_redundancy
        + 0.18 * hub_pressure
        + 0.20 * size_pressure
        + no_anchor_penalty
    )
    score = query_utility + graph_support - set_cost
    return {
        "score": round(float(score), 6),
        "query_utility": round(float(query_utility), 6),
        "graph_support": round(float(graph_support), 6),
        "set_cost": round(float(set_cost), 6),
        "doc_indices": list(doc_tuple),
        "support_doc_count": len(doc_tuple),
        "covered_query_tokens": sorted(covered_tokens),
        "covered_query_token_count": len(covered_tokens),
        "covered_query_idf": round(float(coverage_idf), 6),
        "relevance": round(float(relevance), 6),
        "transition_strength": round(float(transition_tree_strength), 6),
        "transition_edge_count": int(transition_edge_count),
        "missing_tree_edges": int(missing_tree_edges),
        "endpoint_specificity": round(float(endpoint_specificity), 6),
        "shared_endpoint_count": len(shared_endpoints),
        "shared_endpoints_sample": sorted(shared_endpoints)[:10],
        "anchor_trust": round(float(anchor_trust), 6),
        "specificity": round(float(specificity), 6),
        "support_agreement": round(float(support_agreement), 6),
        "neighborhood_agreement": round(float(neighborhood_agreement), 6),
        "auxiliary_agreement": round(float(auxiliary_agreement), 6),
        "lexical_redundancy": round(float(lexical_redundancy), 6),
        "semantic_redundancy": round(float(semantic_redundancy), 6),
        "hub_pressure": round(float(hub_pressure), 6),
    }
