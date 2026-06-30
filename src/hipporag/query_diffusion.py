from typing import Dict

import numpy as np

from .utils.logging_utils import get_logger

logger = get_logger(__name__)


class QueryConditionedDiffuser:
    def __init__(self, min_edge_weight: float = 1e-4):
        self.min_edge_weight = min_edge_weight

    def run(
        self,
        graph,
        reset_prob: np.ndarray,
        base_edge_weights: np.ndarray,
        edge_type_ids: np.ndarray,
        edge_target_ids: np.ndarray,
        node_type_ids: np.ndarray,
        node_match_scores: np.ndarray,
        edge_type_to_idx: Dict[str, int],
        node_type_to_idx: Dict[str, int],
        router_output,
        passage_node_idxs: np.ndarray,
        edge_weight_boosts: np.ndarray | None = None,
        directed: bool = True,
    ):
        if graph.vcount() == 0 or len(passage_node_idxs) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        edge_gate_array = np.ones(len(edge_type_to_idx), dtype=np.float32)
        for edge_type, idx in edge_type_to_idx.items():
            edge_gate_array[idx] = router_output.edge_gate.get(edge_type, 1.0)

        node_prior_array = np.ones(len(node_type_to_idx), dtype=np.float32)
        for node_type, idx in node_type_to_idx.items():
            node_prior_array[idx] = router_output.node_prior.get(node_type, 1.0)

        target_type_ids = node_type_ids[edge_target_ids]
        target_priors = node_prior_array[target_type_ids]
        target_matches = np.maximum(node_match_scores[edge_target_ids], self.min_edge_weight)
        edge_gates = edge_gate_array[edge_type_ids]

        query_edge_weights = base_edge_weights * edge_gates * target_priors * target_matches
        if edge_weight_boosts is not None and len(edge_weight_boosts) == len(query_edge_weights):
            query_edge_weights = query_edge_weights * edge_weight_boosts
        query_edge_weights = np.where(np.isfinite(query_edge_weights), query_edge_weights, self.min_edge_weight)
        query_edge_weights = np.maximum(query_edge_weights, self.min_edge_weight)

        pagerank_scores = graph.personalized_pagerank(
            vertices=range(graph.vcount()),
            damping=router_output.damping,
            directed=directed,
            weights=query_edge_weights.tolist(),
            reset=reset_prob,
            implementation="prpack",
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in passage_node_idxs], dtype=np.float32)
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores, query_edge_weights
