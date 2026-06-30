"""Graph completion for the clean AG-STO graph policy."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .ranking import unique_ranked
from .scoring import (
    MISSING_TREE_EDGE_PENALTY,
    doc_query_covered_tokens,
    score_anchor_guided_evidence_set,
    transition_weight_to_support_set,
)


def apply_graph_obligated_completion(
    *,
    selected_docs: Sequence[int],
    query: str = "",
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
    bm25_scores: Mapping[int, float],
    stable_anchor_doc_indices: Sequence[int],
    candidate_doc_indices: Sequence[int],
    anchor_prior: Mapping[int, float],
    specificity_prior: Mapping[int, float],
    support_prior: Mapping[int, float],
    neighborhood_prior: Mapping[int, float],
    auxiliary_prior: Mapping[int, float] | None,
    max_endpoint_degree: int,
    chunk_embedding_matrix: np.ndarray | None = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Make one conservative graph-obligated replacement inside the set budget.

    This is the clean AG-STO graph policy. Diagnostic answer-obligation
    completion is intentionally not implemented here.
    """

    del query
    current_docs = unique_ranked(selected_docs)
    if not current_docs:
        return [], {"applied": False, "reason": "empty_selected_set"}

    stable_anchor_set = set(unique_ranked(stable_anchor_doc_indices))
    current_score = score_anchor_guided_evidence_set(
        doc_indices=current_docs,
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
    )
    current_tokens = set(current_score.get("covered_query_tokens", []) or [])
    selected_set = set(current_docs)
    tail_replaceable_docs = [
        int(doc_idx)
        for doc_idx in reversed(current_docs)
        if int(doc_idx) not in stable_anchor_set
    ][:1]
    if not tail_replaceable_docs:
        tail_replaceable_docs = [int(current_docs[-1])]

    best_trial: Dict[str, Any] | None = None
    best_docs: List[int] | None = None
    for candidate_doc in unique_ranked(candidate_doc_indices):
        candidate_doc = int(candidate_doc)
        if candidate_doc in selected_set:
            continue
        transition_weight, shared_endpoints = transition_weight_to_support_set(
            doc_idx=candidate_doc,
            support_docs=current_docs,
            corpus_index=corpus_index,
            max_endpoint_degree=max_endpoint_degree,
        )
        if transition_weight <= 0.0:
            continue
        channel_count = sum(
            1
            for prior in (anchor_prior, specificity_prior, support_prior, neighborhood_prior, auxiliary_prior or {})
            if float(prior.get(candidate_doc, 0.0)) > 0.0
        )
        new_tokens = doc_query_covered_tokens(
            doc_idx=candidate_doc,
            query_tokens=query_tokens,
            corpus_index=corpus_index,
        ) - current_tokens
        if channel_count < 2 and float(support_prior.get(candidate_doc, 0.0)) <= 0.0:
            continue
        if not new_tokens and float(support_prior.get(candidate_doc, 0.0)) <= 0.0:
            continue

        for dropped_doc in tail_replaceable_docs:
            trial_docs = [doc_idx for doc_idx in current_docs if int(doc_idx) != int(dropped_doc)]
            trial_docs.append(candidate_doc)
            trial_score = score_anchor_guided_evidence_set(
                doc_indices=trial_docs,
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
            )
            score_gain = float(trial_score.get("score", 0.0)) - float(current_score.get("score", 0.0))
            required_score_gain = MISSING_TREE_EDGE_PENALTY * float(max(len(current_docs) - 1, 1))
            if score_gain < required_score_gain:
                continue
            trial_trace = {
                "applied": True,
                "acceptance": "score_gain",
                "candidate_doc": int(candidate_doc),
                "dropped_doc": int(dropped_doc),
                "score_gain": round(float(score_gain), 6),
                "required_score_gain": round(float(required_score_gain), 6),
                "certified_gain": round(float(score_gain + 0.35 * math.log1p(max(float(transition_weight), 0.0))), 6),
                "selection_gain": round(float(score_gain), 6),
                "answer_obligation_weight": 0.0,
                "answer_obligation": {"score": 0.0, "intent": "disabled"},
                "dropped_answer_obligation": {"score": 0.0, "intent": "disabled"},
                "support_retrieved_rank": None,
                "candidate_transition_weight": round(float(transition_weight), 6),
                "candidate_shared_endpoints": list(shared_endpoints)[:10],
                "candidate_new_query_tokens": sorted(new_tokens),
                "candidate_channel_count": int(channel_count),
                "before_score": current_score,
                "after_score": trial_score,
            }
            if best_trial is None or (
                float(trial_trace["selection_gain"]),
                float(trial_trace["score_gain"]),
                len(new_tokens),
                float(transition_weight),
                -int(candidate_doc),
            ) > (
                float(best_trial.get("selection_gain", best_trial.get("score_gain", 0.0))),
                float(best_trial["score_gain"]),
                len(best_trial.get("candidate_new_query_tokens", []) or []),
                float(best_trial["candidate_transition_weight"]),
                -int(best_trial["candidate_doc"]),
            ):
                best_trial = trial_trace
                ordered_trial = []
                for doc_idx in current_docs:
                    if int(doc_idx) == int(dropped_doc):
                        ordered_trial.append(candidate_doc)
                    else:
                        ordered_trial.append(int(doc_idx))
                best_docs = unique_ranked(ordered_trial)

    if best_trial is None or best_docs is None:
        return current_docs, {
            "applied": False,
            "reason": "no_positive_connected_replacement",
            "before_score": current_score,
        }
    return best_docs, best_trial
