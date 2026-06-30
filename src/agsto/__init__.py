"""Clean public API for AG-STO retrieval.

AG-STO is treated as an independent retrieval method. This package intentionally
exposes only the clean mainline policies:

- ``base``: stable anchor entry + STO proposals + graph-constrained evidence-set
  selection.
- ``graph``: ``base`` plus conservative graph-obligated completion.
- ``build_query_local_sto_graph``: query-time local STO graph admission, the
  clean replacement boundary for historical source-report candidate files.
- ``order_local_sto_source_prior``: query-rooted STO frontier source-prior
  ordering over admitted local graph docs.
- ``order_local_sto_balanced_source_prior``: root-balanced source-prior
  ordering for multi-root questions.
- ``order_local_sto_coverage_source_prior``: root-balanced source-prior that
  promotes frontier witnesses only when they cover query evidence.
- ``order_local_sto_dual_source_prior``: root-balanced identity and transition
  source-prior ordering.
- ``order_local_sto_admission_preserving_source_prior``: source-prior ordering
  that preserves online admission order while adding the primary STO witness.
- ``order_local_sto_layered_source_prior``: source-prior ordering that preserves
  role-transition chain evidence before identity/title completion.
- ``order_local_sto_root_preserving_source_prior``: source-prior ordering that
  keeps the primary transition witness while preserving later retriever roots.

Diagnostic probes such as answer-obligation, answer-role-gap, role-path,
reader-ordering, and native/support weight variants remain in analysis scripts
and are not part of this API.

Native proposal generation is available from ``agsto.proposals`` for
implementation diagnostics, but is intentionally not exported at the package
top level.
"""

from .config import AGSTOConfig
from .completion import apply_graph_obligated_completion
from .index import build_corpus_unit_index
from .lexical import rank_docs_bm25, score_docs_bm25
from .local_graph import build_query_local_sto_graph
from .local_graph import order_local_sto_balanced_source_prior
from .local_graph import order_local_sto_coverage_source_prior
from .local_graph import order_local_sto_dual_source_prior
from .local_graph import order_local_sto_admission_preserving_source_prior
from .local_graph import order_local_sto_layered_source_prior
from .local_graph import order_local_sto_root_preserving_source_prior
from .local_graph import order_local_sto_source_prior
from .local_graph import select_local_sto_evidence_docs
from .ranking import rank_sto_proposal_consensus_docs, unique_ranked
from .retriever import AGSTORetriever
from .scoring import score_anchor_guided_evidence_set

__all__ = [
    "AGSTOConfig",
    "AGSTORetriever",
    "apply_graph_obligated_completion",
    "build_corpus_unit_index",
    "build_query_local_sto_graph",
    "order_local_sto_balanced_source_prior",
    "order_local_sto_coverage_source_prior",
    "order_local_sto_dual_source_prior",
    "order_local_sto_admission_preserving_source_prior",
    "order_local_sto_layered_source_prior",
    "order_local_sto_root_preserving_source_prior",
    "order_local_sto_source_prior",
    "rank_docs_bm25",
    "rank_sto_proposal_consensus_docs",
    "score_anchor_guided_evidence_set",
    "score_docs_bm25",
    "select_local_sto_evidence_docs",
    "unique_ranked",
]
