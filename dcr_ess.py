#!/usr/bin/env python3
"""Core utilities for ECHO-RAG.

ECHO-RAG stands for Evidence-Channel Graph Retrieval for Evidence-Set
RAG.  The historical internal name was DCR-ESS.  This module intentionally
contains only pure evidence-pool and top-k evidence-set construction utilities.
It does not call an LLM, rebuild graphs, load datasets, or use gold labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set


METHOD_NAME = "ECHO-RAG"
LEGACY_METHOD_NAME = "DCR-ESS"
DEFAULT_CANDIDATE_TOP_K = 16
DEFAULT_MAX_CANDIDATES = 24
DEFAULT_READER_TOP_K = 5

CLEAN_CANDIDATE_ADMISSION = "channel_balanced"
CLEAN_SELECTOR_OBJECTIVE = "evidence_set_coverage"
CLEAN_FINAL_CONSTRUCTION = "selector_stable"
ABLATION_ONLY_CONSTRUCTIONS = frozenset({"rank_fusion", "rank_product", "isr_fusion", "isr_stable"})
ABLATION_ONLY_CANDIDATE_ADMISSIONS = frozenset(
    {
        "cross_channel",
        "terminal_bound",
        "demand_marginal_coverage",
        "demand_marginal_topk",
        "pareto_frontier",
        "pareto_repair",
        "residual_role_repair",
    }
)

CLEAN_MAINLINE_CONTRACT = {
    "candidate_admission": CLEAN_CANDIDATE_ADMISSION,
    "baseline_candidate_top_k": 0,
    "include_candidate_provenance": False,
    "selector_objective": CLEAN_SELECTOR_OBJECTIVE,
    "construction": CLEAN_FINAL_CONSTRUCTION,
    "reader_top_k": DEFAULT_READER_TOP_K,
}


def echo_rag_clean_mainline_violations(
    *,
    candidate_admission: str = CLEAN_CANDIDATE_ADMISSION,
    baseline_candidate_top_k: int = 0,
    include_candidate_provenance: bool = False,
    selector_objective: str = CLEAN_SELECTOR_OBJECTIVE,
    construction: str = CLEAN_FINAL_CONSTRUCTION,
    reader_top_k: int = DEFAULT_READER_TOP_K,
) -> List[str]:
    """Return violations of the paper-facing ECHO-RAG v1 contract.

    The contract is intentionally narrow.  It keeps ECHO-RAG as evidence-channel
    graph retrieval plus one listwise evidence-set membership decision.  Any
    hand-designed rank fusion, head seeding, prompt-objective patching, or
    candidate admission policy belongs to ablation/diagnostic code.
    """
    violations: List[str] = []
    if str(candidate_admission) != CLEAN_CANDIDATE_ADMISSION:
        violations.append(
            "candidate_admission must be channel_balanced; cross_channel/terminal_bound/demand_marginal_coverage/demand_marginal_topk/pareto_frontier/pareto_repair/residual_role_repair are ablations."
        )
    if int(baseline_candidate_top_k) != 0:
        violations.append(
            "baseline_candidate_top_k must be 0; original-question/head seeding is not the standalone method."
        )
    if bool(include_candidate_provenance):
        violations.append(
            "include_candidate_provenance must be false; provenance annotation is selector prompt augmentation."
        )
    if str(selector_objective) != CLEAN_SELECTOR_OBJECTIVE:
        violations.append(
            "selector_objective must be evidence_set_coverage; answer_resolving is a prompt ablation."
        )
    if str(construction) != CLEAN_FINAL_CONSTRUCTION:
        violations.append(
            "construction must be selector_stable; RF/RRF/ISR constructions are ablations."
        )
    if int(reader_top_k) != DEFAULT_READER_TOP_K:
        violations.append("reader_top_k must be 5 for the fixed multi-hop QA evidence-set interface.")
    return violations


def assert_echo_rag_clean_mainline(**kwargs: object) -> None:
    """Raise if a run tries to present an ablation as ECHO-RAG mainline."""
    violations = echo_rag_clean_mainline_violations(**kwargs)
    if violations:
        raise ValueError("Non-clean ECHO-RAG mainline config: " + " ".join(violations))


@dataclass(frozen=True)
class DcrEssConfig:
    """Configuration for final top-k evidence-set construction."""

    construction: str = "selector_stable"
    top_k: int = 5
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    # Ablation-only rank_fusion parameters.  They are ignored by the clean
    # selector_stable mainline and must not appear in paper-facing method runs.
    lambda_selector: float = 0.5
    rrf_k: float = 5.0


@dataclass(frozen=True)
class EchoRagConfig:
    """Paper-facing clean ECHO-RAG configuration.

    This intentionally omits RF/RRF/lambda fields.  Those fields remain only in
    ``DcrEssConfig`` for reproducing historical ablations.
    """

    construction: str = CLEAN_FINAL_CONSTRUCTION
    top_k: int = DEFAULT_READER_TOP_K
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K
    max_candidates: int = DEFAULT_MAX_CANDIDATES


@dataclass(frozen=True)
class EvidenceChannel:
    """One graph retrieval channel induced from a query evidence need."""

    channel_id: str
    description: str
    doc_indices: Sequence[int]


@dataclass(frozen=True)
class EvidenceSetSelection:
    """Final ECHO-RAG top-k evidence-set selection result."""

    candidate_docs: List[int]
    selected_docs: List[int]
    selected_ids: List[int]
    top_k_docs: List[int]
    construction: str


def unique_preserve_order(values: Iterable[int]) -> List[int]:
    """Return integer values without duplicates, preserving first occurrence."""
    out: List[int] = []
    seen = set()
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_channel_balanced_evidence_pool(
    channels: Sequence[EvidenceChannel],
    *,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    seed_docs: Sequence[int] | None = None,
) -> List[int]:
    """Build a channel-balanced evidence pool from graph retrieval channels.

    ECHO-RAG treats the channel as the retrieval object.  The pool is built by
    depth-wise channel-balanced traversal, then deduplicated while preserving
    the first exposure order. This gives each evidence demand a route into the
    candidate pool before the evidence-set selector makes the final membership
    decision.

    `seed_docs` is optional and should be used only for explicit system variants.
    The clean standalone ECHO-RAG boundary should pass no seed documents.
    """
    docs: List[int] = []
    if seed_docs:
        docs.extend(int(doc_idx) for doc_idx in seed_docs)

    max_depth = min(
        max((len(channel.doc_indices) for channel in channels), default=0),
        int(candidate_top_k),
    )
    for depth in range(max_depth):
        for channel in channels:
            if depth >= len(channel.doc_indices):
                continue
            docs.append(int(channel.doc_indices[depth]))
    return unique_preserve_order(docs)[: int(max_candidates)]


def build_demand_marginal_coverage_evidence_pool(
    channels: Sequence[EvidenceChannel],
    *,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    expansion_candidate_top_k: int | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    alpha: float = 0.35,
    seed_docs: Sequence[int] | None = None,
) -> List[int]:
    """Build a candidate pool by marginal demand coverage.

    This is an ablation candidate-materialization policy inspired by
    aspect-aware search result diversification.  Each demand channel induces a
    rank-based exposure score for a document.  The pool is then built greedily:
    documents receive higher utility when they are strong for demand channels
    that are not yet represented by the current pool.

    The function only forms the candidate pool.  It does not choose the final
    reader-facing evidence set.
    """
    selected: List[int] = []
    if seed_docs:
        selected.extend(unique_preserve_order(seed_docs))
    selected = selected[: int(max_candidates)]
    selected_set = set(selected)
    if len(selected) >= int(max_candidates):
        return selected

    depth = int(expansion_candidate_top_k if expansion_candidate_top_k is not None else candidate_top_k)
    if depth <= 0:
        return selected

    channel_ids: List[str] = []
    exposure: Dict[int, Dict[str, float]] = {}
    first_seen: Dict[int, int] = {}
    order = 0
    for channel_offset, channel in enumerate(channels):
        channel_id = str(channel.channel_id or f"channel_{channel_offset}")
        if channel_id in channel_ids:
            channel_id = f"{channel_id}#{channel_offset}"
        channel_ids.append(channel_id)
        for rank0, raw_doc_idx in enumerate(unique_preserve_order(channel.doc_indices)[:depth]):
            doc_idx = int(raw_doc_idx)
            if doc_idx not in first_seen:
                first_seen[doc_idx] = order
                order += 1
            rank = rank0 + 1
            score = 1.0 / max(math.log2(1.0 + float(rank)), 1e-12)
            exposure.setdefault(doc_idx, {})[channel_id] = max(
                float(score),
                float(exposure.get(doc_idx, {}).get(channel_id, 0.0)),
            )

    coverage: Dict[str, float] = {channel_id: 0.0 for channel_id in channel_ids}
    for doc_idx in selected:
        for channel_id, score in exposure.get(int(doc_idx), {}).items():
            coverage[channel_id] = max(float(coverage.get(channel_id, 0.0)), float(score))

    blend = min(max(float(alpha), 0.0), 1.0)
    while len(selected) < int(max_candidates):
        best_doc: int | None = None
        best_key: tuple[float, float, float, int, int] | None = None
        for doc_idx, per_channel in exposure.items():
            doc_idx = int(doc_idx)
            if doc_idx in selected_set:
                continue
            relevance = max((float(value) for value in per_channel.values()), default=0.0)
            marginal_gain = sum(
                float(per_channel.get(channel_id, 0.0))
                * max(0.0, 1.0 - float(coverage.get(channel_id, 0.0)))
                for channel_id in channel_ids
            )
            score = blend * relevance + (1.0 - blend) * marginal_gain
            key = (
                float(score),
                float(marginal_gain),
                float(relevance),
                -int(first_seen.get(doc_idx, 10**9)),
                -doc_idx,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_doc = doc_idx
        if best_doc is None:
            break
        selected.append(best_doc)
        selected_set.add(best_doc)
        for channel_id, score in exposure.get(best_doc, {}).items():
            coverage[channel_id] = max(float(coverage.get(channel_id, 0.0)), float(score))

    return selected[: int(max_candidates)]


def build_demand_marginal_topk_evidence_pool(
    channels: Sequence[EvidenceChannel],
    *,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    alpha: float = 0.35,
    seed_docs: Sequence[int] | None = None,
) -> List[int]:
    """Build a clean top-k-universe marginal coverage pool.

    This is a direct replacement for depth-wise channel interleaving.  It uses
    exactly the same per-channel depth as ``build_channel_balanced_evidence_pool``
    and changes only the materialization order within that fixed universe.
    """
    return build_demand_marginal_coverage_evidence_pool(
        channels,
        candidate_top_k=int(candidate_top_k),
        expansion_candidate_top_k=int(candidate_top_k),
        max_candidates=int(max_candidates),
        alpha=float(alpha),
        seed_docs=seed_docs,
    )


def build_pareto_frontier_evidence_pool(
    channels: Sequence[EvidenceChannel],
    *,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    expansion_candidate_top_k: int | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    seed_docs: Sequence[int] | None = None,
) -> List[int]:
    """Expose Echo role-channel candidates by Pareto rank layers.

    Each document is represented by its rank vector across role-conditioned
    retrieval channels.  A document is demoted only when another document ranks
    at least as well in every channel and strictly better in one channel.  This
    keeps candidates that are uniquely strong for one role, instead of requiring
    hand-tuned coverage/relevance weights or fixed append counts.
    """
    selected: List[int] = []
    if seed_docs:
        selected.extend(unique_preserve_order(seed_docs))
    selected = selected[: int(max_candidates)]
    selected_set = set(selected)
    if len(selected) >= int(max_candidates):
        return selected

    depth = int(expansion_candidate_top_k if expansion_candidate_top_k is not None else candidate_top_k)
    if depth <= 0:
        return selected

    channel_docs: List[List[int]] = [
        unique_preserve_order(channel.doc_indices)[:depth]
        for channel in channels
    ]
    if not channel_docs:
        return selected

    universe = build_channel_balanced_evidence_pool(
        channels,
        candidate_top_k=depth,
        max_candidates=max(depth * max(len(channel_docs), 1), int(max_candidates)),
    )
    universe = [int(doc_idx) for doc_idx in universe if int(doc_idx) not in selected_set]
    order = {int(doc_idx): pos for pos, doc_idx in enumerate(universe)}
    rank_vectors: Dict[int, tuple[int, ...]] = {}
    missing_rank = depth + 1
    for doc_idx in universe:
        ranks: List[int] = []
        for docs in channel_docs:
            try:
                ranks.append(docs.index(int(doc_idx)) + 1)
            except ValueError:
                ranks.append(missing_rank)
        rank_vectors[int(doc_idx)] = tuple(ranks)

    remaining = list(universe)
    while remaining and len(selected) < int(max_candidates):
        frontier: List[int] = []
        for doc_idx in remaining:
            doc_vector = rank_vectors[int(doc_idx)]
            dominated = False
            for other_idx in remaining:
                if int(other_idx) == int(doc_idx):
                    continue
                other_vector = rank_vectors[int(other_idx)]
                if all(a <= b for a, b in zip(other_vector, doc_vector)) and any(
                    a < b for a, b in zip(other_vector, doc_vector)
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(int(doc_idx))
        if not frontier:
            break
        frontier.sort(key=lambda doc_idx: int(order.get(int(doc_idx), 10**9)))
        for doc_idx in frontier:
            if len(selected) >= int(max_candidates):
                break
            if int(doc_idx) not in selected_set:
                selected.append(int(doc_idx))
                selected_set.add(int(doc_idx))
        frontier_set = set(frontier)
        remaining = [int(doc_idx) for doc_idx in remaining if int(doc_idx) not in frontier_set]

    return selected[: int(max_candidates)]


def build_pareto_repair_evidence_pool(
    channels: Sequence[EvidenceChannel],
    *,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    expansion_candidate_top_k: int | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    seed_docs: Sequence[int] | None = None,
) -> List[int]:
    """Repair the shallow Echo candidate pool without increasing selector load.

    The original channel-balanced pool defines the selector capacity for this
    query.  Within that same capacity, rank-dominated shallow candidates are
    replaceable; deeper Echo candidates are admitted only if the retained core
    does not dominate them across role-channel ranks.
    """
    base_docs = build_channel_balanced_evidence_pool(
        channels,
        candidate_top_k=int(candidate_top_k),
        max_candidates=int(max_candidates),
        seed_docs=seed_docs,
    )
    target_size = len(base_docs)
    if target_size <= 0:
        return base_docs

    depth = int(expansion_candidate_top_k if expansion_candidate_top_k is not None else candidate_top_k)
    if depth <= int(candidate_top_k):
        return base_docs

    expanded_docs = build_pareto_frontier_evidence_pool(
        channels,
        candidate_top_k=int(candidate_top_k),
        expansion_candidate_top_k=depth,
        max_candidates=max(depth * max(len(channels), 1), int(max_candidates)),
        seed_docs=seed_docs,
    )
    all_docs = unique_preserve_order([*base_docs, *expanded_docs])
    missing_rank = depth + 1
    channel_docs = [
        unique_preserve_order(channel.doc_indices)[:depth]
        for channel in channels
    ]
    rank_vectors: Dict[int, tuple[int, ...]] = {}
    for doc_idx in all_docs:
        ranks: List[int] = []
        for docs in channel_docs:
            try:
                ranks.append(docs.index(int(doc_idx)) + 1)
            except ValueError:
                ranks.append(missing_rank)
        rank_vectors[int(doc_idx)] = tuple(ranks)

    def dominates(left: int, right: int) -> bool:
        left_vector = rank_vectors[int(left)]
        right_vector = rank_vectors[int(right)]
        return all(a <= b for a, b in zip(left_vector, right_vector)) and any(
            a < b for a, b in zip(left_vector, right_vector)
        )

    core_docs: List[int] = []
    for doc_idx in base_docs:
        if any(dominates(other_idx, int(doc_idx)) for other_idx in base_docs if int(other_idx) != int(doc_idx)):
            continue
        core_docs.append(int(doc_idx))

    selected: List[int] = list(core_docs)
    selected_set = set(selected)
    for doc_idx in expanded_docs:
        doc_idx = int(doc_idx)
        if len(selected) >= target_size:
            break
        if doc_idx in selected_set:
            continue
        if any(dominates(core_idx, doc_idx) for core_idx in core_docs):
            continue
        selected.append(doc_idx)
        selected_set.add(doc_idx)

    for doc_idx in base_docs:
        if len(selected) >= target_size:
            break
        doc_idx = int(doc_idx)
        if doc_idx not in selected_set:
            selected.append(doc_idx)
            selected_set.add(doc_idx)

    return selected[:target_size]


def build_decomposition_candidate_pool(
    channels: Sequence[EvidenceChannel],
    *,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    seed_docs: Sequence[int] | None = None,
) -> List[int]:
    """Backward-compatible alias for the ECHO-RAG evidence-pool builder."""
    return build_channel_balanced_evidence_pool(
        channels,
        candidate_top_k=candidate_top_k,
        max_candidates=max_candidates,
        seed_docs=seed_docs,
    )


def build_cross_channel_admitted_evidence_pool(
    channels: Sequence[EvidenceChannel],
    *,
    base_candidate_top_k: int = 8,
    expansion_candidate_top_k: int = 16,
    max_candidates: int = 24,
    min_channel_support: int = 2,
    seed_docs: Sequence[int] | None = None,
) -> List[int]:
    """Ablation-only pool that admits deep candidates with channel consensus.

    The selector should see the reliable shallow evidence-channel pool, but not
    every deeper candidate exposed by one channel.  This variant first builds a
    standard shallow pool, then appends deeper candidates only when at least
    ``min_channel_support`` evidence channels independently expose the document.
    It is train-free and uses only retrieval provenance, not gold labels or
    dataset-specific lexical rules.
    """
    base_docs = build_channel_balanced_evidence_pool(
        channels,
        candidate_top_k=int(base_candidate_top_k),
        max_candidates=int(max_candidates),
        seed_docs=seed_docs,
    )
    expanded_docs = build_channel_balanced_evidence_pool(
        channels,
        candidate_top_k=int(expansion_candidate_top_k),
        max_candidates=int(max_candidates),
        seed_docs=seed_docs,
    )
    coverage = channel_coverage_map(
        channels=channels,
        candidate_docs=expanded_docs,
        candidate_top_k=int(expansion_candidate_top_k),
    )
    admitted = list(base_docs)
    min_support = max(1, int(min_channel_support))
    for doc_idx in expanded_docs:
        doc_idx = int(doc_idx)
        if len(admitted) >= int(max_candidates):
            break
        if doc_idx in admitted:
            continue
        if len(coverage.get(doc_idx, set())) < min_support:
            continue
        admitted.append(doc_idx)
    return admitted[: int(max_candidates)]


def build_terminal_channel_admitted_evidence_pool(
    channels: Sequence[EvidenceChannel],
    *,
    terminal_channel_ids: Iterable[str],
    base_candidate_top_k: int = 8,
    expansion_candidate_top_k: int = 16,
    terminal_admission_limit: int = 4,
    max_candidates: int = 24,
    seed_docs: Sequence[int] | None = None,
) -> List[int]:
    """Ablation-only pool that admits deep candidates from terminal channels.

    Terminal channels are derived from the generated role graph: a role with no
    outgoing ``must_connect_to`` edge is treated as the terminal answer/verification
    evidence obligation.  This admission policy keeps the normal shallow pool
    and appends a small number of deeper terminal-channel candidates.  It uses
    role-graph structure and retrieval provenance only, not gold answers or
    dataset-specific lexical rules.
    """
    base_docs = build_channel_balanced_evidence_pool(
        channels,
        candidate_top_k=int(base_candidate_top_k),
        max_candidates=int(max_candidates),
        seed_docs=seed_docs,
    )
    terminal_ids = {str(channel_id) for channel_id in terminal_channel_ids if str(channel_id).strip()}
    admitted = list(base_docs)
    admitted_count = 0
    limit = max(0, int(terminal_admission_limit))
    for channel in channels:
        if str(channel.channel_id) not in terminal_ids:
            continue
        for raw_doc_idx in unique_preserve_order(channel.doc_indices)[: int(expansion_candidate_top_k)]:
            if len(admitted) >= int(max_candidates) or admitted_count >= limit:
                return admitted[: int(max_candidates)]
            doc_idx = int(raw_doc_idx)
            if doc_idx in admitted:
                continue
            admitted.append(doc_idx)
            admitted_count += 1
    return admitted[: int(max_candidates)]


def selected_docs_from_ids(candidate_docs: Sequence[int], selected_ids: Sequence[int]) -> List[int]:
    """Resolve selector-local ids into global document indices."""
    docs: List[int] = []
    for raw_idx in selected_ids:
        idx = int(raw_idx)
        if idx < 0 or idx >= len(candidate_docs):
            continue
        doc_idx = int(candidate_docs[idx])
        if doc_idx not in docs:
            docs.append(doc_idx)
    return docs


def _selected_order_by_doc(candidate_docs: Sequence[int], selected_ids: Sequence[int]) -> Dict[int, int]:
    selected_order_by_doc: Dict[int, int] = {}
    for selected_order, local_idx in enumerate(selected_ids):
        idx = int(local_idx)
        if idx < 0 or idx >= len(candidate_docs):
            continue
        selected_order_by_doc.setdefault(int(candidate_docs[idx]), selected_order)
    return selected_order_by_doc


def fill_from_candidate_order(selected_docs: Sequence[int], candidate_docs: Sequence[int], *, top_k: int = 5) -> List[int]:
    """Fill a partial selected set using candidate order."""
    final = unique_preserve_order(selected_docs)
    for raw_doc_idx in candidate_docs:
        if len(final) >= int(top_k):
            break
        doc_idx = int(raw_doc_idx)
        if doc_idx not in final:
            final.append(doc_idx)
    return final[: int(top_k)]


def channel_coverage_map(
    *,
    channels: Sequence[EvidenceChannel],
    candidate_docs: Sequence[int],
    candidate_top_k: int,
) -> Dict[int, Set[str]]:
    """Map each candidate document to the evidence channels that exposed it."""
    candidates = set(int(doc_idx) for doc_idx in candidate_docs)
    coverage: Dict[int, Set[str]] = {int(doc_idx): set() for doc_idx in candidate_docs}
    for offset, channel in enumerate(channels):
        channel_id = str(channel.channel_id or f"channel_{offset}")
        for doc_idx in unique_preserve_order(channel.doc_indices)[: int(candidate_top_k)]:
            if doc_idx not in candidates:
                continue
            coverage.setdefault(int(doc_idx), set()).add(channel_id)
    return coverage


def channel_coverage_greedy_topk(
    channels: Sequence[EvidenceChannel],
    candidate_docs: Sequence[int],
    *,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    top_k: int = 5,
) -> List[int]:
    """Select a low-cost evidence set by greedy evidence-channel coverage.

    This is the deterministic ECHO-RAG-lite selector.  It is intentionally
    simple and train-free: at each step it picks the candidate that covers the
    most as-yet-uncovered evidence channels, breaking ties by candidate order.
    Once all exposed channels are covered, it fills from candidate order.
    """
    candidates = unique_preserve_order(candidate_docs)
    coverage = channel_coverage_map(
        channels=channels,
        candidate_docs=candidates,
        candidate_top_k=int(candidate_top_k),
    )
    selected: List[int] = []
    covered: Set[str] = set()
    while len(selected) < int(top_k):
        best_doc: int | None = None
        best_new_count = -1
        best_total_count = -1
        for doc_idx in candidates:
            if doc_idx in selected:
                continue
            doc_channels = coverage.get(int(doc_idx), set())
            new_count = len(doc_channels - covered)
            total_count = len(doc_channels)
            if new_count > best_new_count or (
                new_count == best_new_count and total_count > best_total_count
            ):
                best_doc = int(doc_idx)
                best_new_count = new_count
                best_total_count = total_count
        if best_doc is None:
            break
        selected.append(best_doc)
        covered.update(coverage.get(best_doc, set()))
        if len(selected) >= len(candidates):
            break
    return fill_from_candidate_order(selected, candidates, top_k=top_k)


def candidate_stable_selected_topk(
    candidate_docs: Sequence[int],
    selected_ids: Sequence[int],
    *,
    top_k: int = 5,
) -> List[int]:
    """P0: selector decides membership; candidate order decides stable order."""
    clean_candidates = unique_preserve_order(candidate_docs)
    selected = selected_docs_from_ids(clean_candidates, selected_ids)
    selected_set = set(selected)
    stable_selected = [doc_idx for doc_idx in clean_candidates if doc_idx in selected_set]
    return fill_from_candidate_order(stable_selected, clean_candidates, top_k=top_k)


def rank_product_topk(
    candidate_docs: Sequence[int],
    selected_ids: Sequence[int],
    *,
    top_k: int = 5,
) -> List[int]:
    """Ablation-only consensus ranking over candidate and selector ranks."""
    clean_candidates = unique_preserve_order(candidate_docs)
    selected_order = _selected_order_by_doc(clean_candidates, selected_ids)
    scored = []
    for candidate_rank, doc_idx in enumerate(clean_candidates):
        if doc_idx not in selected_order:
            continue
        score = 1.0 / ((float(candidate_rank) + 1.0) * (float(selected_order[doc_idx]) + 1.0))
        scored.append((score, -candidate_rank, doc_idx))
    scored.sort(reverse=True)
    selected = [int(doc_idx) for _, _, doc_idx in scored]
    return fill_from_candidate_order(selected, clean_candidates, top_k=top_k)


def isr_fusion_topk(
    candidate_docs: Sequence[int],
    selected_ids: Sequence[int],
    *,
    top_k: int = 5,
) -> List[int]:
    """Ablation-only inverse-square rank fusion."""
    clean_candidates = unique_preserve_order(candidate_docs)
    selected_order = _selected_order_by_doc(clean_candidates, selected_ids)
    scored = []
    for candidate_rank, doc_idx in enumerate(clean_candidates):
        score = 1.0 / ((float(candidate_rank) + 1.0) ** 2)
        if doc_idx in selected_order:
            score += 1.0 / ((float(selected_order[doc_idx]) + 1.0) ** 2)
        scored.append((score, -candidate_rank, doc_idx))
    scored.sort(reverse=True)
    return [int(doc_idx) for _, _, doc_idx in scored[: int(top_k)]]


def isr_stable_topk(
    candidate_docs: Sequence[int],
    selected_ids: Sequence[int],
    *,
    top_k: int = 5,
) -> List[int]:
    """Ablation-only ISR membership with candidate-order reader presentation."""
    clean_candidates = unique_preserve_order(candidate_docs)
    isr_members = set(isr_fusion_topk(clean_candidates, selected_ids, top_k=top_k))
    stable_selected = [doc_idx for doc_idx in clean_candidates if doc_idx in isr_members]
    return fill_from_candidate_order(stable_selected, clean_candidates, top_k=top_k)


def rank_fusion_topk(
    candidate_docs: Sequence[int],
    selected_ids: Sequence[int],
    *,
    top_k: int = 5,
    rrf_k: float = 5.0,
    lambda_selector: float = 0.5,
) -> List[int]:
    """Ablation-only final top-k context by rank-regularized set selection.

    Candidate order supplies the retrieval prior.  Selector order supplies the
    listwise evidence-set signal.  The two are combined with a simple reciprocal
    rank score.  This function is retained for reproducibility of old runs, not
    for the ECHO-RAG clean mainline.
    """
    selected_order_by_doc = _selected_order_by_doc(candidate_docs, selected_ids)

    scored = []
    seen = set()
    for candidate_rank, raw_doc_idx in enumerate(candidate_docs):
        doc_idx = int(raw_doc_idx)
        if doc_idx in seen:
            continue
        seen.add(doc_idx)
        score = 1.0 / (float(rrf_k) + float(candidate_rank) + 1.0)
        if doc_idx in selected_order_by_doc:
            score += float(lambda_selector) / (
                float(rrf_k) + float(selected_order_by_doc[doc_idx]) + 1.0
            )
        scored.append((score, -candidate_rank, doc_idx))

    scored.sort(reverse=True)
    return [int(doc_idx) for _, _, doc_idx in scored[: int(top_k)]]


def construct_final_topk(
    candidate_docs: Sequence[int],
    selected_ids: Sequence[int],
    *,
    construction: str = "selector_stable",
    top_k: int = 5,
    rrf_k: float = 5.0,
    lambda_selector: float = 0.5,
) -> List[int]:
    """Construct the final ECHO-RAG top-k evidence set.

    The clean mainline is ``selector_stable``: the selector decides membership,
    while candidate order decides reader-facing order.  Rank-fusion variants are
    kept only as explicit compatibility/appendix constructions.
    """
    construction_name = str(construction)
    if construction_name == "selector_stable":
        return candidate_stable_selected_topk(candidate_docs, selected_ids, top_k=top_k)
    if construction_name == "isr_stable":
        return isr_stable_topk(candidate_docs, selected_ids, top_k=top_k)
    if construction_name == "rank_fusion":
        return rank_fusion_topk(
            candidate_docs,
            selected_ids,
            top_k=top_k,
            rrf_k=rrf_k,
            lambda_selector=lambda_selector,
        )
    raise ValueError(f"Unsupported DCR-ESS construction: {construction_name}")


def select_evidence_set(
    candidate_docs: Sequence[int],
    selected_ids: Sequence[int],
    *,
    config: DcrEssConfig | EchoRagConfig | None = None,
) -> EvidenceSetSelection:
    """Resolve selector ids and construct the final ECHO-RAG top-k evidence set."""
    config = config or DcrEssConfig()
    clean_candidates = unique_preserve_order(candidate_docs)
    clean_ids = []
    for raw_idx in selected_ids:
        idx = int(raw_idx)
        if 0 <= idx < len(clean_candidates) and idx not in clean_ids:
            clean_ids.append(idx)

    selected_docs = selected_docs_from_ids(clean_candidates, clean_ids)
    top_k_docs = construct_final_topk(
        clean_candidates,
        clean_ids,
        construction=getattr(config, "construction", CLEAN_FINAL_CONSTRUCTION),
        top_k=int(getattr(config, "top_k", DEFAULT_READER_TOP_K)),
        rrf_k=float(getattr(config, "rrf_k", 5.0)),
        lambda_selector=float(getattr(config, "lambda_selector", 0.5)),
    )
    return EvidenceSetSelection(
        candidate_docs=clean_candidates,
        selected_docs=selected_docs,
        selected_ids=clean_ids,
        top_k_docs=top_k_docs,
        construction=str(config.construction),
    )


def recall_fraction(gold_docs: Iterable[int], docs: Iterable[int]) -> float:
    """Compute support recall for a selected document set."""
    gold = {int(doc) for doc in gold_docs}
    if not gold:
        return 0.0
    retrieved = {int(doc) for doc in docs}
    return len(gold & retrieved) / len(gold)


def provenance_summary(
    *,
    channels: Sequence[EvidenceChannel],
    candidate_docs: Sequence[int],
    candidate_top_k: int,
) -> Dict[int, List[Mapping[str, object]]]:
    """Describe which evidence channels exposed each candidate document."""
    candidates = set(int(doc_idx) for doc_idx in candidate_docs)
    out: Dict[int, List[Mapping[str, object]]] = {int(doc_idx): [] for doc_idx in candidate_docs}
    for channel in channels:
        channel_docs = unique_preserve_order(channel.doc_indices)[: int(candidate_top_k)]
        for rank, doc_idx in enumerate(channel_docs, start=1):
            if doc_idx not in candidates:
                continue
            out.setdefault(int(doc_idx), []).append(
                {
                    "channel_id": channel.channel_id,
                    "description": channel.description,
                    "rank": rank,
                }
            )
    return out
