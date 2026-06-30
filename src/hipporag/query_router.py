from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class QueryRouterOutput:
    query_type: str
    seed_mix: Dict[str, float]
    edge_gate: Dict[str, float]
    node_prior: Dict[str, float]
    damping: float


def _normalize_weights(weight_dict: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(value, 0.0) for value in weight_dict.values()))
    if total <= 0:
        uniform = 1.0 / max(len(weight_dict), 1)
        return {key: uniform for key in weight_dict}
    return {key: max(value, 0.0) / total for key, value in weight_dict.items()}


def _safe_scores(scores: np.ndarray) -> np.ndarray:
    if scores is None:
        return np.array([])
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 0:
        scores = scores.reshape(1)
    return scores


def _concentration(scores: np.ndarray, top_k: int = 5) -> float:
    scores = _safe_scores(scores)
    if len(scores) == 0:
        return 0.0
    sorted_scores = np.sort(scores)[::-1]
    head = sorted_scores[: min(top_k, len(sorted_scores))]
    return float(sorted_scores[0] / (float(np.mean(head)) + 1e-8))


class HybridQueryRouter:
    """Rule-based router corrected by retrieval distribution statistics."""

    def route(
        self,
        query: str,
        entity_scores: np.ndarray,
        hyperedge_scores: np.ndarray,
        passage_scores: np.ndarray,
    ) -> QueryRouterOutput:
        query_lower = query.lower()
        entity_conc = _concentration(entity_scores)
        hyperedge_conc = _concentration(hyperedge_scores)
        passage_conc = _concentration(passage_scores)

        explanatory_cues = ["why", "how", "cause", "reason", "effect", "impact", "mechanism"]
        multi_hop_cues = [
            "which county",
            "part of",
            "related to",
            "connected to",
            "before",
            "after",
            "whose",
            "where was",
        ]

        if any(cue in query_lower for cue in explanatory_cues):
            query_type = "explanatory"
        elif any(cue in query_lower for cue in multi_hop_cues):
            query_type = "multi_hop"
        elif len(query_lower.split()) > 12 and hyperedge_conc >= entity_conc:
            query_type = "multi_hop"
        else:
            query_type = "factoid"

        if query_type == "factoid":
            seed_mix = {"entity": 0.45, "hyperedge": 0.20, "passage": 0.35}
            edge_gate = {
                "entity_hyperedge": 0.32,
                "hyperedge_hyperedge": 0.10,
                "hyperedge_passage": 0.28,
                "entity_entity": 0.10,
                "entity_passage": 0.20,
            }
            node_prior = {"entity": 0.28, "hyperedge": 0.24, "passage": 0.48}
            damping = 0.45
        elif query_type == "explanatory":
            seed_mix = {"entity": 0.20, "hyperedge": 0.48, "passage": 0.32}
            edge_gate = {
                "entity_hyperedge": 0.18,
                "hyperedge_hyperedge": 0.32,
                "hyperedge_passage": 0.30,
                "entity_entity": 0.07,
                "entity_passage": 0.13,
            }
            node_prior = {"entity": 0.18, "hyperedge": 0.47, "passage": 0.35}
            damping = 0.70
        else:
            seed_mix = {"entity": 0.28, "hyperedge": 0.45, "passage": 0.27}
            edge_gate = {
                "entity_hyperedge": 0.27,
                "hyperedge_hyperedge": 0.33,
                "hyperedge_passage": 0.20,
                "entity_entity": 0.12,
                "entity_passage": 0.08,
            }
            node_prior = {"entity": 0.24, "hyperedge": 0.46, "passage": 0.30}
            damping = 0.65

        # Retrieval-aware corrections.
        if passage_conc > max(entity_conc, hyperedge_conc) * 1.25:
            seed_mix["passage"] += 0.10
            seed_mix["hyperedge"] -= 0.05
            edge_gate["entity_passage"] += 0.04
            edge_gate["hyperedge_hyperedge"] -= 0.04
            node_prior["passage"] += 0.06
            node_prior["hyperedge"] -= 0.06
            damping = max(0.40, damping - 0.08)

        if hyperedge_conc > entity_conc * 1.10:
            seed_mix["hyperedge"] += 0.08
            seed_mix["entity"] -= 0.04
            seed_mix["passage"] -= 0.04
            edge_gate["hyperedge_hyperedge"] += 0.05
            edge_gate["entity_hyperedge"] += 0.03
            edge_gate["entity_passage"] -= 0.04
            damping = min(0.80, damping + 0.05)

        if entity_conc > passage_conc * 1.15 and len(query_lower.split()) <= 10:
            seed_mix["entity"] += 0.08
            seed_mix["passage"] -= 0.05
            edge_gate["entity_hyperedge"] += 0.04
            edge_gate["hyperedge_hyperedge"] -= 0.03
            node_prior["entity"] += 0.05
            node_prior["passage"] -= 0.05

        return QueryRouterOutput(
            query_type=query_type,
            seed_mix=_normalize_weights(seed_mix),
            edge_gate=_normalize_weights(edge_gate),
            node_prior=_normalize_weights(node_prior),
            damping=float(min(max(damping, 0.35), 0.85)),
        )

