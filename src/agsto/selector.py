"""Clean AG-STO evidence-set selection."""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .completion import apply_graph_obligated_completion
from .config import AGSTOConfig
from .index import content_tokens
from .lexical import score_docs_bm25
from .proposal_roles import build_proposal_role_units, covered_role_unit_counts
from .ranking import rank_sto_proposal_consensus_docs, unique_ranked
from .scoring import (
    doc_query_covered_tokens,
    idf_sum,
    score_anchor_guided_evidence_set,
    transition_weight_to_support_set,
)


SUPPORT_PROPOSAL_WEIGHT = 1.75


def _rank_prior(doc_indices: Sequence[int], *, scale: float = 1.0) -> Dict[int, float]:
    prior: Dict[int, float] = {}
    for rank, doc_idx in enumerate(unique_ranked(doc_indices), start=1):
        prior[int(doc_idx)] = max(float(prior.get(int(doc_idx), 0.0)), float(scale) / math.sqrt(float(rank)))
    return prior


def _support_set_priors(
    support_set_search: Mapping[str, Any] | None,
) -> Tuple[Dict[int, float], Dict[int, int]]:
    support_priors: Dict[int, float] = {}
    support_best_rank: Dict[int, int] = {}
    if not isinstance(support_set_search, Mapping):
        return support_priors, support_best_rank

    for support_rank, support_set in enumerate(support_set_search.get("top_support_sets", []) or [], start=1):
        if not isinstance(support_set, Mapping):
            continue
        raw_score = max(float(support_set.get("score", 0.0) or 0.0), 0.0)
        support_credit = math.log1p(raw_score) / math.sqrt(float(support_rank))
        for doc_idx in unique_ranked(support_set.get("doc_indices", []) or []):
            doc_idx = int(doc_idx)
            support_priors[doc_idx] = max(float(support_priors.get(doc_idx, 0.0)), float(support_credit))
            support_best_rank[doc_idx] = min(int(support_best_rank.get(doc_idx, 10**9)), int(support_rank))
    return support_priors, support_best_rank


def _neighborhood_doc_priors(query_conditioned_neighborhood: Mapping[str, Any] | None) -> Dict[int, float]:
    if not isinstance(query_conditioned_neighborhood, Mapping):
        return {}

    priors: Counter[int] = Counter()
    for rank, doc_idx in enumerate(query_conditioned_neighborhood.get("selected_neighborhood_doc_indices", []) or [], start=1):
        priors[int(doc_idx)] += 1.25 / math.sqrt(float(rank))
    for rank, doc_idx in enumerate(query_conditioned_neighborhood.get("retrieved_doc_indices", []) or [], start=1):
        priors[int(doc_idx)] += 0.65 / math.sqrt(float(rank))
    for pair_rank, pair in enumerate(query_conditioned_neighborhood.get("boundary_readout_pairs", []) or [], start=1):
        if not isinstance(pair, Mapping):
            continue
        completion_doc = pair.get("completion_doc")
        if completion_doc is None:
            continue
        direct_score = max(float(pair.get("direct_transition_score", 0.0) or 0.0), 0.0)
        priors[int(completion_doc)] += (0.45 + 0.35 * math.log1p(direct_score)) / math.sqrt(float(pair_rank))
    return {int(doc_idx): float(score) for doc_idx, score in priors.items()}


def _annotate_role_unit_counts(
    score: Mapping[str, Any],
    *,
    role_units: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    annotated = dict(score)
    counts = covered_role_unit_counts(
        doc_indices=annotated.get("doc_indices", []) or [],
        role_units=role_units,
    )
    annotated["covered_role_unit_count"] = int(counts.get("role_units", 0) or 0)
    annotated["covered_pair_unit_count"] = int(counts.get("pair_units", 0) or 0)
    annotated["covered_endpoint_unit_count"] = int(counts.get("endpoint_units", 0) or 0)
    annotated["covered_support_unit_count"] = int(counts.get("support_units", 0) or 0)
    return annotated


def _role_structured_graph_constrained(candidate: Mapping[str, Any], incumbent: Mapping[str, Any]) -> bool:
    return (
        int(candidate.get("missing_tree_edges", 0) or 0) <= int(incumbent.get("missing_tree_edges", 0) or 0)
        and float(candidate.get("graph_support", 0.0) or 0.0) >= float(incumbent.get("graph_support", 0.0) or 0.0)
        and float(candidate.get("set_cost", 0.0) or 0.0) <= float(incumbent.get("set_cost", 0.0) or 0.0)
    )


def _role_structured_sort_key(score: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        -float(score.get("covered_query_idf", 0.0) or 0.0),
        -int(score.get("covered_query_token_count", 0) or 0),
        -int(score.get("covered_role_unit_count", 0) or 0),
        int(score.get("missing_tree_edges", 0) or 0),
        -float(score.get("graph_support", 0.0) or 0.0),
        float(score.get("set_cost", 0.0) or 0.0),
        -float(score.get("score", 0.0) or 0.0),
        tuple(int(doc_idx) for doc_idx in score.get("doc_indices", []) or []),
    )


def _select_role_structured_arbitration_set(
    *,
    query_tokens: Set[str],
    incumbent_set: Mapping[str, Any],
    candidate_pool: Sequence[int],
    role_units: Sequence[Mapping[str, Any]],
    corpus_index: Mapping[str, Any],
    bm25_scores: Mapping[int, float],
    anchor_prior: Mapping[int, float],
    specificity_prior: Mapping[int, float],
    support_prior: Mapping[int, float],
    neighborhood_prior: Mapping[int, float],
    auxiliary_prior: Mapping[int, float],
    evidence_set_size: int,
    max_endpoint_degree: int,
    chunk_embedding_matrix: np.ndarray | None = None,
) -> Dict[str, Any]:
    incumbent_docs = unique_ranked(incumbent_set.get("doc_indices", []) or [])
    pool = unique_ranked(list(incumbent_docs) + list(candidate_pool))
    incumbent = _annotate_role_unit_counts(incumbent_set, role_units=role_units)
    if not role_units or not pool:
        return {
            "selected": incumbent,
            "feasible_count": 0,
            "subset_count": 0,
            "applied": False,
        }

    target_size = min(max(int(evidence_set_size), 1), len(pool))
    feasible: List[Dict[str, Any]] = []
    subset_count = 0
    for subset in itertools.combinations(pool, target_size):
        subset_count += 1
        scored = _annotate_role_unit_counts(
            score_anchor_guided_evidence_set(
                doc_indices=list(subset),
                query_tokens=query_tokens,
                corpus_index=corpus_index,
                bm25_scores=bm25_scores,
                anchor_prior=anchor_prior,
                specificity_prior=specificity_prior,
                support_prior=support_prior,
                neighborhood_prior=neighborhood_prior,
                auxiliary_prior=auxiliary_prior,
                max_endpoint_degree=max_endpoint_degree,
                chunk_embedding_matrix=chunk_embedding_matrix,
            ),
            role_units=role_units,
        )
        if _role_structured_graph_constrained(scored, incumbent):
            feasible.append(scored)

    if not feasible:
        return {
            "selected": incumbent,
            "feasible_count": 0,
            "subset_count": int(subset_count),
            "applied": False,
        }

    selected = sorted(feasible, key=_role_structured_sort_key)[0]
    applied = set(selected.get("doc_indices", []) or []) != set(incumbent_docs)
    if not applied:
        selected = incumbent
    return {
        "selected": selected,
        "feasible_count": int(len(feasible)),
        "subset_count": int(subset_count),
        "applied": bool(applied),
    }


def order_anchor_guided_evidence_docs(
    *,
    selected_set: Mapping[str, Any],
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
    bm25_scores: Mapping[int, float],
    stable_anchor_doc_indices: Sequence[int],
    anchor_prior: Mapping[int, float],
    specificity_prior: Mapping[int, float],
    support_prior: Mapping[int, float],
    neighborhood_prior: Mapping[int, float],
    max_endpoint_degree: int,
) -> List[int]:
    doc_indices = unique_ranked(selected_set.get("doc_indices", []) or [])
    if not doc_indices:
        return []
    stable_anchors = [int(doc_idx) for doc_idx in unique_ranked(stable_anchor_doc_indices) if int(doc_idx) in set(doc_indices)]
    stable_anchor_set = set(stable_anchors)
    doc_token_idf: Mapping[str, float] = corpus_index["doc_token_idf"]
    doc_scores: Dict[int, float] = {}
    for doc_idx in doc_indices:
        transition_weight, _ = transition_weight_to_support_set(
            doc_idx=int(doc_idx),
            support_docs=[other_doc for other_doc in doc_indices if int(other_doc) != int(doc_idx)],
            corpus_index=corpus_index,
            max_endpoint_degree=max_endpoint_degree,
        )
        coverage = idf_sum(
            doc_query_covered_tokens(doc_idx=int(doc_idx), query_tokens=query_tokens, corpus_index=corpus_index),
            doc_token_idf,
        )
        doc_scores[int(doc_idx)] = (
            1.25 * coverage
            + 0.65 * math.log1p(max(float(bm25_scores.get(int(doc_idx), 0.0)), 0.0))
            + 1.15 * math.log1p(max(float(transition_weight), 0.0))
            + 1.35 * float(anchor_prior.get(int(doc_idx), 0.0))
            + 1.05 * float(specificity_prior.get(int(doc_idx), 0.0))
            + 0.75 * float(support_prior.get(int(doc_idx), 0.0))
            + 0.65 * float(neighborhood_prior.get(int(doc_idx), 0.0))
        )
    ranked_tail = [
        doc_idx
        for doc_idx in sorted(doc_indices, key=lambda value: (-float(doc_scores.get(int(value), 0.0)), int(value)))
        if int(doc_idx) not in stable_anchor_set
    ]
    return stable_anchors + ranked_tail


def _rank_set_expansion_docs(
    *,
    current_score: Mapping[str, Any],
    candidates: Sequence[int],
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
    bm25_scores: Mapping[int, float],
    channel_prior: Mapping[int, float],
    support_prior: Mapping[int, float],
    max_endpoint_degree: int,
    limit: int,
) -> List[int]:
    current_docs = unique_ranked(current_score.get("doc_indices", []) or [])
    current_set = set(current_docs)
    current_covered = set(current_score.get("covered_query_tokens", []) or [])
    rows: List[Tuple[float, int]] = []
    for doc_idx in candidates:
        doc_idx = int(doc_idx)
        if doc_idx in current_set:
            continue
        transition_weight, _ = transition_weight_to_support_set(
            doc_idx=doc_idx,
            support_docs=current_docs,
            corpus_index=corpus_index,
            max_endpoint_degree=max_endpoint_degree,
        )
        new_tokens = (
            doc_query_covered_tokens(doc_idx=doc_idx, query_tokens=query_tokens, corpus_index=corpus_index)
            - current_covered
        )
        marginal = (
            2.25 * idf_sum(new_tokens, corpus_index["doc_token_idf"])
            + 1.35 * math.log1p(max(float(transition_weight), 0.0))
            + 0.75 * float(channel_prior.get(doc_idx, 0.0))
            + 0.35 * math.log1p(max(float(bm25_scores.get(doc_idx, 0.0)), 0.0))
            - (0.75 if transition_weight <= 0.0 and current_docs else 0.0)
            + (0.25 if doc_idx in support_prior else 0.0)
        )
        rows.append((float(marginal), doc_idx))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [doc_idx for _priority, doc_idx in rows[: max(int(limit), 1)]]


def _selector_score_sort_key(score: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        -float(score.get("score", 0.0) or 0.0),
        int(score.get("missing_tree_edges", 10**9)),
        -int(score.get("covered_query_token_count", 0) or 0),
        tuple(int(doc_idx) for doc_idx in score.get("doc_indices", []) or []),
    )


def _run_mct_set_search(
    *,
    seed_candidates: Sequence[int],
    channel_seed_sets: Sequence[Sequence[int]],
    candidates: Sequence[int],
    effective_evidence_size: int,
    config: AGSTOConfig,
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
    bm25_scores: Mapping[int, float],
    anchor_prior: Mapping[int, float],
    specificity_prior: Mapping[int, float],
    support_prior: Mapping[int, float],
    neighborhood_prior: Mapping[int, float],
    auxiliary_prior: Mapping[int, float],
    channel_prior: Mapping[int, float],
    chunk_embedding_matrix: np.ndarray | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Replace beam expansion with deterministic MCT over the same objective."""

    target_size = min(max(int(effective_evidence_size), 1), max(len(candidates), 1))
    frontier_limit = max(int(config.beam_size), 1)
    iteration_budget = max(frontier_limit * target_size * 4, frontier_limit)

    def score_docs(doc_indices: Sequence[int]) -> Dict[str, Any]:
        return score_anchor_guided_evidence_set(
            doc_indices=unique_ranked(doc_indices),
            query_tokens=query_tokens,
            corpus_index=corpus_index,
            bm25_scores=bm25_scores,
            anchor_prior=anchor_prior,
            specificity_prior=specificity_prior,
            support_prior=support_prior,
            neighborhood_prior=neighborhood_prior,
            auxiliary_prior=auxiliary_prior,
            max_endpoint_degree=config.max_endpoint_degree,
            chunk_embedding_matrix=chunk_embedding_matrix,
        )

    def complete_docs(doc_indices: Sequence[int]) -> List[int]:
        completed = unique_ranked(doc_indices)
        while len(completed) < target_size:
            scored = score_docs(completed)
            frontier = _rank_set_expansion_docs(
                current_score=scored,
                candidates=candidates,
                query_tokens=query_tokens,
                corpus_index=corpus_index,
                bm25_scores=bm25_scores,
                channel_prior=channel_prior,
                support_prior=support_prior,
                max_endpoint_degree=config.max_endpoint_degree,
                limit=frontier_limit,
            )
            next_doc = next((int(doc_idx) for doc_idx in frontier if int(doc_idx) not in set(completed)), None)
            if next_doc is None:
                break
            completed.append(next_doc)
        return completed

    nodes: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    terminal_scores: Dict[Tuple[int, ...], Dict[str, Any]] = {}

    def node_for(doc_indices: Sequence[int]) -> Dict[str, Any]:
        docs = unique_ranked(doc_indices)
        key = tuple(sorted(docs))
        if key not in nodes:
            if docs:
                unexpanded = _rank_set_expansion_docs(
                    current_score=score_docs(docs),
                    candidates=candidates,
                    query_tokens=query_tokens,
                    corpus_index=corpus_index,
                    bm25_scores=bm25_scores,
                    channel_prior=channel_prior,
                    support_prior=support_prior,
                    max_endpoint_degree=config.max_endpoint_degree,
                    limit=frontier_limit,
                )
            else:
                unexpanded = unique_ranked(seed_candidates)[:frontier_limit]
            nodes[key] = {
                "doc_indices": docs,
                "visits": 0,
                "total_value": 0.0,
                "children": [],
                "unexpanded": [int(doc_idx) for doc_idx in unexpanded if int(doc_idx) not in set(docs)],
            }
        return nodes[key]

    root = node_for([])
    for seed_set in channel_seed_sets:
        seed_docs = unique_ranked(seed_set)
        if not seed_docs:
            continue
        completed = complete_docs(seed_docs)
        terminal_scores[tuple(sorted(completed))] = score_docs(completed)

    for _iteration in range(iteration_budget):
        path = [root]
        node = root
        while len(node["doc_indices"]) < target_size and not node["unexpanded"] and node["children"]:
            parent_visits = max(int(node["visits"]), 1)

            def child_key(child: Mapping[str, Any]) -> Tuple[float, float, Tuple[int, ...]]:
                visits = int(child["visits"])
                mean_value = float(child["total_value"]) / float(max(visits, 1))
                exploration_bonus = math.sqrt(math.log(float(parent_visits + 1)) / float(visits + 1))
                return (mean_value + exploration_bonus, mean_value, tuple(-int(doc_idx) for doc_idx in child["doc_indices"]))

            node = max(node["children"], key=child_key)
            path.append(node)

        if len(node["doc_indices"]) < target_size and node["unexpanded"]:
            action_doc = int(node["unexpanded"].pop(0))
            child_docs = unique_ranked(list(node["doc_indices"]) + [action_doc])
            child = node_for(child_docs)
            if child not in node["children"]:
                node["children"].append(child)
            node = child
            path.append(node)

        completed_docs = complete_docs(node["doc_indices"])
        scored = score_docs(completed_docs)
        value = float(scored.get("score", 0.0) or 0.0)
        terminal_key = tuple(sorted(unique_ranked(completed_docs)))
        previous = terminal_scores.get(terminal_key)
        if previous is None or float(scored.get("score", 0.0) or 0.0) > float(previous.get("score", 0.0) or 0.0):
            terminal_scores[terminal_key] = scored
        for path_node in path:
            path_node["visits"] = int(path_node["visits"]) + 1
            path_node["total_value"] = float(path_node["total_value"]) + value

    if not terminal_scores:
        fallback_docs = complete_docs(seed_candidates[:1] or candidates[:1])
        terminal_scores[tuple(sorted(fallback_docs))] = score_docs(fallback_docs)

    selected = sorted(terminal_scores.values(), key=_selector_score_sort_key)[0]
    trace = {
        "policy": "mct",
        "iterations": int(iteration_budget),
        "node_count": int(len(nodes)),
        "terminal_count": int(len(terminal_scores)),
        "frontier_limit": int(frontier_limit),
    }
    return selected, trace


def retrieve_evidence_set(
    *,
    query: str,
    corpus_index: Mapping[str, Any],
    config: AGSTOConfig,
    anchor_doc_indices: Sequence[int] | None = None,
    native_dense_doc_indices: Sequence[int] | None = None,
    bm25_doc_indices: Sequence[int] | None = None,
    specificity_doc_indices: Sequence[int] | None = None,
    endpoint_transition_doc_indices: Sequence[int] | None = None,
    hybrid_residual_doc_indices: Sequence[int] | None = None,
    query_conditioned_neighborhood: Mapping[str, Any] | None = None,
    support_set_search: Mapping[str, Any] | None = None,
    proposal_role_records: Sequence[Mapping[str, Any]] | None = None,
    proposal_role_summary_by_doc: Mapping[str, Any] | None = None,
    chunk_embedding_matrix: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Run the public AG-STO base/graph selector without prototype delegation."""

    _ = proposal_role_summary_by_doc
    query_tokens = content_tokens(query)
    bm25_scores = score_docs_bm25(query=query, corpus_index=corpus_index)
    bm25_ranked = (
        unique_ranked(bm25_doc_indices or [])
        if bm25_doc_indices
        else [doc_idx for doc_idx, _ in sorted(bm25_scores.items(), key=lambda item: (-float(item[1]), item[0]))]
    )
    dense_anchors = unique_ranked(anchor_doc_indices or native_dense_doc_indices or [])
    if not dense_anchors:
        dense_anchors = bm25_ranked[: max(config.stable_anchor_k, 1)]
    stable_anchors = dense_anchors[: max(int(config.stable_anchor_k), 0)]
    anchor_prior = _rank_prior(dense_anchors, scale=1.7)
    specificity_prior = _rank_prior(specificity_doc_indices or [], scale=1.35)
    endpoint_prior = _rank_prior(endpoint_transition_doc_indices or [], scale=0.75)
    hybrid_prior = _rank_prior(hybrid_residual_doc_indices or [], scale=0.55)
    auxiliary_prior: Counter[int] = Counter()
    for source in (endpoint_prior, hybrid_prior):
        for doc_idx, score in source.items():
            auxiliary_prior[int(doc_idx)] += float(score)
    support_prior, support_best_rank = _support_set_priors(support_set_search)
    neighborhood_prior = _neighborhood_doc_priors(query_conditioned_neighborhood)

    channel_prior: Counter[int] = Counter()
    for source in (anchor_prior, specificity_prior, endpoint_prior, hybrid_prior, support_prior, neighborhood_prior):
        for doc_idx, score in source.items():
            channel_prior[int(doc_idx)] += float(score)

    candidates: List[int] = []
    candidate_budget = max(int(config.candidate_limit), int(config.evidence_set_size), int(config.retrieval_top_k), 1)
    for sequence in (
        stable_anchors,
        dense_anchors,
        specificity_doc_indices or [],
        (query_conditioned_neighborhood.get("selected_neighborhood_doc_indices", []) or []) if isinstance(query_conditioned_neighborhood, Mapping) else [],
        (query_conditioned_neighborhood.get("retrieved_doc_indices", []) or []) if isinstance(query_conditioned_neighborhood, Mapping) else [],
        endpoint_transition_doc_indices or [],
        hybrid_residual_doc_indices or [],
        (support_set_search.get("retrieved_doc_indices", []) or []) if isinstance(support_set_search, Mapping) else [],
        support_prior.keys(),
        bm25_ranked,
    ):
        for doc_idx in sequence:
            doc_idx = int(doc_idx)
            if doc_idx < 0 or doc_idx in candidates:
                continue
            candidates.append(doc_idx)
            if len(candidates) >= candidate_budget:
                break
        if len(candidates) >= candidate_budget:
            break

    if not candidates:
        return {
            "retrieved_doc_indices": [],
            "candidate_doc_indices": [],
            "stable_anchor_doc_indices": stable_anchors,
            "selected_evidence_set": {},
            "set_search_selected_evidence_set": {},
            "set_search_policy": str(config.set_search_policy),
            "set_search_trace": {"policy": str(config.set_search_policy), "applied": False},
            "beam_size": int(config.beam_size),
        }

    effective_evidence_size = min(max(int(config.evidence_set_size), 1), max(int(config.retrieval_top_k), 1))
    effective_proposal_depth = max(int(config.proposal_candidate_depth), effective_evidence_size)
    effective_support_proposal_depth = max(int(config.support_proposal_depth), effective_evidence_size)
    seed_candidates = unique_ranked(
        list(stable_anchors)
        + list(specificity_doc_indices or [])
        + list((query_conditioned_neighborhood.get("selected_neighborhood_doc_indices", []) or []) if isinstance(query_conditioned_neighborhood, Mapping) else [])
        + list((support_set_search.get("retrieved_doc_indices", []) or []) if isinstance(support_set_search, Mapping) else [])
        + list(candidates)
    )[: max(int(config.beam_size), 1)]
    if not seed_candidates:
        seed_candidates = [
            max(
                candidates,
                key=lambda doc_idx: (
                    float(channel_prior.get(int(doc_idx), 0.0)),
                    math.log1p(max(float(bm25_scores.get(int(doc_idx), 0.0)), 0.0)),
                    -int(doc_idx),
                ),
            )
        ]

    beam_by_key: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for seed_doc in seed_candidates:
        initial = score_anchor_guided_evidence_set(
            doc_indices=[int(seed_doc)],
            query_tokens=query_tokens,
            corpus_index=corpus_index,
            bm25_scores=bm25_scores,
            anchor_prior=anchor_prior,
            specificity_prior=specificity_prior,
            support_prior=support_prior,
            neighborhood_prior=neighborhood_prior,
            auxiliary_prior=auxiliary_prior,
            max_endpoint_degree=config.max_endpoint_degree,
            chunk_embedding_matrix=chunk_embedding_matrix,
        )
        beam_by_key[(int(seed_doc),)] = initial

    channel_seed_sets = [
        unique_ranked(list(specificity_doc_indices or []))[:effective_evidence_size],
        unique_ranked(list((query_conditioned_neighborhood.get("retrieved_doc_indices", []) or []) if isinstance(query_conditioned_neighborhood, Mapping) else []))[:effective_evidence_size],
        unique_ranked(list(hybrid_residual_doc_indices or []))[:effective_evidence_size],
        unique_ranked(list(endpoint_transition_doc_indices or []))[:effective_evidence_size],
        unique_ranked(list((support_set_search.get("retrieved_doc_indices", []) or []) if isinstance(support_set_search, Mapping) else []))[:effective_evidence_size],
        unique_ranked(list(stable_anchors) + list(specificity_doc_indices or []))[:effective_evidence_size],
    ]
    for seed_set in channel_seed_sets:
        if not seed_set:
            continue
        scored_seed_set = score_anchor_guided_evidence_set(
            doc_indices=seed_set,
            query_tokens=query_tokens,
            corpus_index=corpus_index,
            bm25_scores=bm25_scores,
            anchor_prior=anchor_prior,
            specificity_prior=specificity_prior,
            support_prior=support_prior,
            neighborhood_prior=neighborhood_prior,
            auxiliary_prior=auxiliary_prior,
            max_endpoint_degree=config.max_endpoint_degree,
            chunk_embedding_matrix=chunk_embedding_matrix,
        )
        beam_by_key[tuple(sorted(seed_set))] = scored_seed_set

    while True:
        current_beams = sorted(beam_by_key.values(), key=lambda item: (-float(item["score"]), item["doc_indices"]))[: max(int(config.beam_size), 1)]
        if not current_beams:
            break
        current_min_size = min(len(item.get("doc_indices", []) or []) for item in current_beams)
        if current_min_size >= effective_evidence_size:
            break
        next_by_key = dict(beam_by_key)
        expanded_any = False
        for beam in current_beams:
            current_docs = unique_ranked(beam.get("doc_indices", []) or [])
            if len(current_docs) >= effective_evidence_size:
                continue
            expansion_docs = _rank_set_expansion_docs(
                current_score=beam,
                candidates=candidates,
                query_tokens=query_tokens,
                corpus_index=corpus_index,
                bm25_scores=bm25_scores,
                channel_prior=channel_prior,
                support_prior=support_prior,
                max_endpoint_degree=config.max_endpoint_degree,
                limit=max(int(config.beam_size), 1),
            )
            for doc_idx in expansion_docs:
                new_docs = unique_ranked(list(current_docs) + [int(doc_idx)])
                key = tuple(sorted(new_docs))
                scored = score_anchor_guided_evidence_set(
                    doc_indices=new_docs,
                    query_tokens=query_tokens,
                    corpus_index=corpus_index,
                    bm25_scores=bm25_scores,
                    anchor_prior=anchor_prior,
                    specificity_prior=specificity_prior,
                    support_prior=support_prior,
                    neighborhood_prior=neighborhood_prior,
                    auxiliary_prior=auxiliary_prior,
                    max_endpoint_degree=config.max_endpoint_degree,
                    chunk_embedding_matrix=chunk_embedding_matrix,
                )
                previous = next_by_key.get(key)
                if previous is None or float(scored["score"]) > float(previous["score"]):
                    next_by_key[key] = scored
                    expanded_any = True
        if not expanded_any:
            break
        next_size = min(current_min_size + 1, effective_evidence_size)
        grown_values = [
            item
            for item in next_by_key.values()
            if len(item.get("doc_indices", []) or []) >= next_size
        ]
        if not grown_values:
            break
        beam_by_key = {
            tuple(item["doc_indices"]): item
            for item in sorted(grown_values, key=lambda value: (-float(value["score"]), value["doc_indices"]))[: max(int(config.beam_size), 1)]
        }
        if all(len(item.get("doc_indices", []) or []) >= effective_evidence_size for item in beam_by_key.values()):
            break

    selected_evidence_set = sorted(
        beam_by_key.values(),
        key=lambda item: (
            -float(item["score"]),
            int(item.get("missing_tree_edges", 10**9)),
            -int(item.get("covered_query_token_count", 0)),
            item.get("doc_indices", []),
        ),
    )[0]
    selected_docs = order_anchor_guided_evidence_docs(
        selected_set=selected_evidence_set,
        query_tokens=query_tokens,
        corpus_index=corpus_index,
        bm25_scores=bm25_scores,
        stable_anchor_doc_indices=stable_anchors,
        anchor_prior=anchor_prior,
        specificity_prior=specificity_prior,
        support_prior=support_prior,
        neighborhood_prior=neighborhood_prior,
        max_endpoint_degree=config.max_endpoint_degree,
    )
    beam_selected_evidence_set = dict(selected_evidence_set)
    beam_selected_evidence_set["doc_indices"] = selected_docs
    set_search_trace: Dict[str, Any] = {
        "policy": "beam",
        "beam_count": int(len(beam_by_key)),
        "frontier_limit": int(max(int(config.beam_size), 1)),
    }
    set_search_selected_evidence_set = dict(beam_selected_evidence_set)
    if config.set_search_policy == "mct":
        mct_selected_evidence_set, set_search_trace = _run_mct_set_search(
            seed_candidates=seed_candidates,
            channel_seed_sets=channel_seed_sets,
            candidates=candidates,
            effective_evidence_size=effective_evidence_size,
            config=config,
            query_tokens=query_tokens,
            corpus_index=corpus_index,
            bm25_scores=bm25_scores,
            anchor_prior=anchor_prior,
            specificity_prior=specificity_prior,
            support_prior=support_prior,
            neighborhood_prior=neighborhood_prior,
            auxiliary_prior=auxiliary_prior,
            channel_prior=channel_prior,
            chunk_embedding_matrix=chunk_embedding_matrix,
        )
        mct_docs = order_anchor_guided_evidence_docs(
            selected_set=mct_selected_evidence_set,
            query_tokens=query_tokens,
            corpus_index=corpus_index,
            bm25_scores=bm25_scores,
            stable_anchor_doc_indices=stable_anchors,
            anchor_prior=anchor_prior,
            specificity_prior=specificity_prior,
            support_prior=support_prior,
            neighborhood_prior=neighborhood_prior,
            max_endpoint_degree=config.max_endpoint_degree,
        )
        set_search_selected_evidence_set = dict(mct_selected_evidence_set)
        set_search_selected_evidence_set["doc_indices"] = mct_docs
        selected_evidence_set = set_search_selected_evidence_set
        selected_docs = mct_docs

    proposal_doc_indices: List[List[int]] = [
        unique_ranked(specificity_doc_indices or [])[:effective_proposal_depth],
        unique_ranked(hybrid_residual_doc_indices or [])[:effective_proposal_depth],
        unique_ranked(
            query_conditioned_neighborhood.get("retrieved_doc_indices", []) or []
            if isinstance(query_conditioned_neighborhood, Mapping)
            else []
        )[:effective_proposal_depth],
        unique_ranked(
            support_set_search.get("retrieved_doc_indices", []) or []
            if isinstance(support_set_search, Mapping)
            else []
        )[:effective_support_proposal_depth],
    ]
    proposal_consensus_docs = rank_sto_proposal_consensus_docs(
        proposal_doc_indices=proposal_doc_indices,
        proposal_weights=[1.0, 1.0, 1.0, SUPPORT_PROPOSAL_WEIGHT],
        top_k=effective_evidence_size,
    )
    proposal_consensus_set = (
        score_anchor_guided_evidence_set(
            doc_indices=proposal_consensus_docs,
            query_tokens=query_tokens,
            corpus_index=corpus_index,
            bm25_scores=bm25_scores,
            anchor_prior=anchor_prior,
            specificity_prior=specificity_prior,
            support_prior=support_prior,
            neighborhood_prior=neighborhood_prior,
            auxiliary_prior=auxiliary_prior,
            max_endpoint_degree=config.max_endpoint_degree,
            chunk_embedding_matrix=chunk_embedding_matrix,
        )
        if proposal_consensus_docs
        else {}
    )
    if proposal_consensus_docs:
        selected_evidence_set = dict(proposal_consensus_set)
        selected_docs = proposal_consensus_docs
        selection_policy = "sto_proposal_consensus"
    else:
        selected_evidence_set = set_search_selected_evidence_set
        selection_policy = f"{config.set_search_policy}_graph_set"

    pre_reader_order_doc_indices = list(selected_docs)
    completion_trace: Dict[str, Any] = {"applied": False, "policy": str(config.completion_policy)}
    if config.completion_policy == "graph_obligated":
        completion_candidates = unique_ranked(
            list(candidates)
            + list((support_set_search.get("candidate_doc_indices", []) or []) if isinstance(support_set_search, Mapping) else [])
            + list((support_set_search.get("retrieved_doc_indices", []) or []) if isinstance(support_set_search, Mapping) else [])
            + list(specificity_doc_indices or [])
            + list(hybrid_residual_doc_indices or [])
            + list((query_conditioned_neighborhood.get("retrieved_doc_indices", []) or []) if isinstance(query_conditioned_neighborhood, Mapping) else [])
        )
        selected_docs, completion_trace = apply_graph_obligated_completion(
            selected_docs=selected_docs,
            query=query,
            query_tokens=query_tokens,
            corpus_index=corpus_index,
            bm25_scores=bm25_scores,
            stable_anchor_doc_indices=stable_anchors,
            candidate_doc_indices=completion_candidates,
            anchor_prior=anchor_prior,
            specificity_prior=specificity_prior,
            support_prior=support_prior,
            neighborhood_prior=neighborhood_prior,
            auxiliary_prior=auxiliary_prior,
            max_endpoint_degree=config.max_endpoint_degree,
            chunk_embedding_matrix=chunk_embedding_matrix,
        )
        completion_trace = {"policy": str(config.completion_policy), **completion_trace}
        selected_evidence_set = score_anchor_guided_evidence_set(
            doc_indices=selected_docs,
            query_tokens=query_tokens,
            corpus_index=corpus_index,
            bm25_scores=bm25_scores,
            anchor_prior=anchor_prior,
            specificity_prior=specificity_prior,
            support_prior=support_prior,
            neighborhood_prior=neighborhood_prior,
            auxiliary_prior=auxiliary_prior,
            max_endpoint_degree=config.max_endpoint_degree,
            chunk_embedding_matrix=chunk_embedding_matrix,
        )
    elif config.completion_policy != "none":
        raise ValueError(f"Unsupported evidence_completion_policy={config.completion_policy!r}")

    role_structured_trace: Dict[str, Any] | None = None
    role_units = build_proposal_role_units(
        proposal_role_records or [],
        support_set_search=support_set_search,
    )
    if role_units:
        role_pool = unique_ranked(
            list(set_search_selected_evidence_set.get("doc_indices", []) or [])
            + list(proposal_consensus_set.get("doc_indices", []) or [])
            + list(pre_reader_order_doc_indices)
            + list(selected_docs)
        )
        before_counts = covered_role_unit_counts(doc_indices=selected_docs, role_units=role_units)
        before_role_count = int(before_counts.get("role_units", 0) or 0)
        arbitration = _select_role_structured_arbitration_set(
            query_tokens=query_tokens,
            incumbent_set=selected_evidence_set,
            candidate_pool=role_pool,
            role_units=role_units,
            corpus_index=corpus_index,
            bm25_scores=bm25_scores,
            anchor_prior=anchor_prior,
            specificity_prior=specificity_prior,
            support_prior=support_prior,
            neighborhood_prior=neighborhood_prior,
            auxiliary_prior=auxiliary_prior,
            evidence_set_size=effective_evidence_size,
            max_endpoint_degree=config.max_endpoint_degree,
            chunk_embedding_matrix=chunk_embedding_matrix,
        )
        selected_evidence_set = dict(arbitration.get("selected", selected_evidence_set) or selected_evidence_set)
        selected_docs = order_anchor_guided_evidence_docs(
            selected_set=selected_evidence_set,
            query_tokens=query_tokens,
            corpus_index=corpus_index,
            bm25_scores=bm25_scores,
            stable_anchor_doc_indices=stable_anchors,
            anchor_prior=anchor_prior,
            specificity_prior=specificity_prior,
            support_prior=support_prior,
            neighborhood_prior=neighborhood_prior,
            max_endpoint_degree=config.max_endpoint_degree,
        )
        selected_evidence_set["doc_indices"] = selected_docs
        role_structured_trace = {
            "applied": bool(arbitration.get("applied", False)),
            "role_unit_count": int(len(role_units)),
            "candidate_pool_size": int(len(role_pool)),
            "subset_count": int(arbitration.get("subset_count", 0) or 0),
            "feasible_count": int(arbitration.get("feasible_count", 0) or 0),
            "before_role_unit_count": before_role_count,
            "after_role_unit_count": int(selected_evidence_set.get("covered_role_unit_count", 0) or 0),
        }

    result = {
        "retrieved_doc_indices": unique_ranked(
            list(selected_docs)
            + list(specificity_doc_indices or [])
            + list(hybrid_residual_doc_indices or [])
            + list((query_conditioned_neighborhood.get("retrieved_doc_indices", []) or []) if isinstance(query_conditioned_neighborhood, Mapping) else [])
            + list(endpoint_transition_doc_indices or [])
            + list((support_set_search.get("retrieved_doc_indices", []) or []) if isinstance(support_set_search, Mapping) else [])
            + list(candidates)
            + list(bm25_ranked)
        )[: max(int(config.retrieval_top_k), effective_evidence_size)],
        "selected_evidence_set": selected_evidence_set,
        "set_search_selected_evidence_set": set_search_selected_evidence_set,
        "set_search_policy": str(config.set_search_policy),
        "set_search_trace": set_search_trace,
        "beam_selected_evidence_set": beam_selected_evidence_set,
        "proposal_consensus_evidence_set": proposal_consensus_set,
        "selection_policy": selection_policy,
        "reader_ordering_policy": "proposal_consensus",
        "evidence_completion_policy": str(config.completion_policy),
        "graph_obligated_completion": completion_trace,
        "pre_reader_order_doc_indices": pre_reader_order_doc_indices,
        "candidate_doc_indices": candidates,
        "candidate_doc_count": len(candidates),
        "stable_anchor_doc_indices": stable_anchors,
        "anchor_doc_indices": dense_anchors,
        "support_best_rank_by_doc": {str(doc_idx): int(rank) for doc_idx, rank in sorted(support_best_rank.items())},
        "beam_size": int(config.beam_size),
        "evidence_set_size": int(effective_evidence_size),
        "proposal_candidate_depth": int(effective_proposal_depth),
        "support_proposal_depth": int(effective_support_proposal_depth),
        "support_proposal_weight": float(SUPPORT_PROPOSAL_WEIGHT),
        "support_retrieved_weight": 0.0,
        "native_proposal_weight": 0.0,
        "answer_obligation_weight": 0.0,
    }
    if role_structured_trace is not None:
        result["role_structured_construction"] = role_structured_trace
    return result
