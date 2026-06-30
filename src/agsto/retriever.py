"""Clean AG-STO retriever facade.

The public retriever exposes only the independent AG-STO base and graph
policies. Diagnostic-only branches remain in historical analysis scripts and
are not part of this API.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .config import AGSTOConfig
from .index import build_corpus_unit_index
from .local_ppr import retrieve_local_ppr_evidence_set
from .proposals import build_native_sto_proposals
from .selector import retrieve_evidence_set


class AGSTORetriever:
    """Anchor-Guided STO Evidence Retrieval.

    AG-STO enters an STO evidence graph through stable anchors and selects a
    graph-constrained evidence set. The public API exposes only the clean base
    and graph-completion policies.
    """

    def __init__(self, *, corpus_index: Mapping[str, Any], config: AGSTOConfig | None = None) -> None:
        self.corpus_index = corpus_index
        self.config = config or AGSTOConfig()

    @classmethod
    def from_openie_docs(
        cls,
        openie_docs: Sequence[Mapping[str, Any]],
        *,
        config: AGSTOConfig | None = None,
    ) -> "AGSTORetriever":
        """Build AG-STO from OpenIE/STO documents.

        ``openie_docs`` should contain the same lightweight fields consumed by
        the transition-component retriever: ``idx``, ``passage``, and extracted
        triples/propositions.
        """

        return cls(corpus_index=build_corpus_unit_index(openie_docs), config=config)

    def retrieve(
        self,
        *,
        query: str,
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
        """Return a graph-constrained evidence set and retrieval diagnostics."""

        result = retrieve_evidence_set(
            query=query,
            corpus_index=self.corpus_index,
            config=self.config,
            anchor_doc_indices=anchor_doc_indices,
            native_dense_doc_indices=native_dense_doc_indices,
            bm25_doc_indices=bm25_doc_indices,
            specificity_doc_indices=specificity_doc_indices,
            endpoint_transition_doc_indices=endpoint_transition_doc_indices,
            hybrid_residual_doc_indices=hybrid_residual_doc_indices,
            query_conditioned_neighborhood=query_conditioned_neighborhood,
            support_set_search=support_set_search,
            proposal_role_records=proposal_role_records,
            proposal_role_summary_by_doc=proposal_role_summary_by_doc,
            chunk_embedding_matrix=chunk_embedding_matrix,
        )
        result["method"] = "AG-STO"
        result["agsto_policy"] = self.config.policy
        result["clean_api"] = True
        if proposal_role_records is not None:
            result["proposal_role_records"] = list(proposal_role_records)
        if proposal_role_summary_by_doc is not None:
            result["proposal_role_summary_by_doc"] = dict(proposal_role_summary_by_doc)
        return result

    def retrieve_native(
        self,
        *,
        query: str,
        anchor_doc_indices: Sequence[int] | None = None,
        native_dense_doc_indices: Sequence[int] | None = None,
        semantic_query_embedding: np.ndarray | None = None,
        chunk_embedding_matrix: np.ndarray | None = None,
        semantic_residual_weight: float = 8.0,
    ) -> Dict[str, Any]:
        """Generate native STO proposals and retrieve an evidence set."""

        proposals = build_native_sto_proposals(
            query=query,
            corpus_index=self.corpus_index,
            config=self.config,
            anchor_doc_indices=anchor_doc_indices,
            native_dense_doc_indices=native_dense_doc_indices,
            semantic_query_embedding=semantic_query_embedding,
            chunk_embedding_matrix=chunk_embedding_matrix,
            semantic_residual_weight=semantic_residual_weight,
        )
        return self.retrieve(
            query=query,
            chunk_embedding_matrix=chunk_embedding_matrix,
            **proposals,
        )

    def retrieve_local_ppr_native(
        self,
        *,
        query: str,
        anchor_doc_indices: Sequence[int] | None = None,
        native_dense_doc_indices: Sequence[int] | None = None,
        semantic_query_embedding: np.ndarray | None = None,
        chunk_embedding_matrix: np.ndarray | None = None,
        semantic_residual_weight: float = 8.0,
        alpha: float = 0.2,
        residual_epsilon: float = 1e-6,
    ) -> Dict[str, Any]:
        """Run the clean V13B/AG-STO bridge: native proposals + local fact PPR.

        This is intentionally not a new ``AGSTOConfig.policy``. It is a first
        packaged variant for testing whether V13B's local graph propagation can
        replace selector-side channel fusion without query-obligation prompts or
        SFB reports.
        """

        return retrieve_local_ppr_evidence_set(
            query=query,
            corpus_index=self.corpus_index,
            config=self.config,
            anchor_doc_indices=anchor_doc_indices,
            native_dense_doc_indices=native_dense_doc_indices,
            semantic_query_embedding=semantic_query_embedding,
            chunk_embedding_matrix=chunk_embedding_matrix,
            semantic_residual_weight=semantic_residual_weight,
            alpha=alpha,
            residual_epsilon=residual_epsilon,
        )
