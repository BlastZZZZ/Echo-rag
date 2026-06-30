"""Local-PPR evidence selection over the AG-STO corpus index.

This module is the first clean bridge between the V13B observation and the
packaged AG-STO retriever:

- use AG-STO's own STO corpus index and native proposal source;
- build a query-local fact graph from candidate OpenIE units;
- run one local Personalized PageRank push from query-grounded fact resets;
- project fact mass back to documents.

It deliberately does not consume SFB reports, query-obligation prompts, typed
schemas, QA outcomes, or gold labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Deque, Dict, List, Mapping, Sequence, Set, Tuple

from .config import AGSTOConfig
from .index import content_tokens, informative_endpoint
from .lexical import score_docs_bm25
from .proposals import build_native_sto_proposals
from .ranking import unique_ranked
from .scoring import transition_weight_to_support_set


DEFAULT_LOCAL_PPR_ALPHA = 0.2
DEFAULT_LOCAL_PPR_RESIDUAL_EPSILON = 1e-6

QUERY_FUNCTION_TOKENS = {
    "answer",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "many",
    "much",
    "name",
    "find",
}


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _unit_id(unit: Mapping[str, Any]) -> int:
    return _as_int(unit.get("_unit_int_id"), default=-1)


def _fact_units_for_docs(
    *,
    corpus_index: Mapping[str, Any],
    doc_indices: Sequence[int],
) -> List[Mapping[str, Any]]:
    allowed_docs = {int(doc_idx) for doc_idx in doc_indices if int(doc_idx) >= 0}
    units = corpus_index.get("units", []) or []
    facts: List[Mapping[str, Any]] = []
    for unit in units:
        if str(unit.get("unit_type") or "") != "openie_fact":
            continue
        if int(unit.get("doc_index", -1)) not in allowed_docs:
            continue
        if _unit_id(unit) < 0:
            continue
        facts.append(unit)
    return facts


def _fact_role_endpoints(unit: Mapping[str, Any]) -> List[str]:
    """Return normalized subject/object endpoints, excluding document title."""

    return [
        endpoint
        for endpoint in (
            informative_endpoint(unit.get("subject", "")),
            informative_endpoint(unit.get("object", "")),
        )
        if endpoint
    ]


def _fact_role_endpoint_tokens(unit: Mapping[str, Any]) -> Set[str]:
    tokens: Set[str] = set()
    for endpoint in _fact_role_endpoints(unit):
        tokens.update(content_tokens(endpoint))
    return tokens


def _query_matched_endpoints(
    *,
    query: str,
    corpus_index: Mapping[str, Any],
    max_endpoint_degree: int,
) -> Set[str]:
    """Find maximal corpus endpoints explicitly mentioned by the query."""

    query_tokens = content_tokens(query) - QUERY_FUNCTION_TOKENS
    endpoint_to_docs: Mapping[str, Sequence[int]] = corpus_index.get("endpoint_to_docs", {}) or {}
    matched: List[Tuple[str, Set[str]]] = []
    for endpoint, doc_indices in endpoint_to_docs.items():
        endpoint = str(endpoint)
        endpoint_tokens = content_tokens(endpoint)
        if not endpoint_tokens or not endpoint_tokens.issubset(query_tokens):
            continue
        if len(doc_indices or []) > max_endpoint_degree:
            continue
        matched.append((endpoint, endpoint_tokens))

    maximal: Set[str] = set()
    for endpoint, endpoint_tokens in matched:
        is_subsumed = any(
            endpoint != other_endpoint
            and endpoint_tokens < other_tokens
            for other_endpoint, other_tokens in matched
        )
        if not is_subsumed:
            maximal.add(endpoint)
    return maximal


def _candidate_docs_from_proposals(
    *,
    proposals: Mapping[str, Any],
    bm25_ranked: Sequence[int],
    config: AGSTOConfig,
) -> List[int]:
    docs: List[int] = []
    neighborhood = proposals.get("query_conditioned_neighborhood", {}) or {}
    support = proposals.get("support_set_search", {}) or {}
    for sequence in (
        proposals.get("anchor_doc_indices", []) or [],
        proposals.get("native_dense_doc_indices", []) or [],
        proposals.get("specificity_doc_indices", []) or [],
        proposals.get("endpoint_transition_doc_indices", []) or [],
        proposals.get("hybrid_residual_doc_indices", []) or [],
        neighborhood.get("selected_neighborhood_doc_indices", []) or []
        if isinstance(neighborhood, Mapping)
        else [],
        neighborhood.get("retrieved_doc_indices", []) or [] if isinstance(neighborhood, Mapping) else [],
        support.get("retrieved_doc_indices", []) or [] if isinstance(support, Mapping) else [],
        bm25_ranked,
    ):
        docs.extend(int(doc_idx) for doc_idx in sequence if int(doc_idx) >= 0)
    return unique_ranked(docs)[: max(int(config.candidate_limit), int(config.retrieval_top_k), 1)]


def _entry_docs_for_fact_grounding(
    *,
    proposals: Mapping[str, Any],
    bm25_ranked: Sequence[int],
    config: AGSTOConfig,
) -> List[int]:
    """Initial pool used only to discover query-grounded fact endpoints."""

    docs: List[int] = []
    for sequence in (
        proposals.get("native_dense_doc_indices", []) or [],
        proposals.get("specificity_doc_indices", []) or [],
        proposals.get("hybrid_residual_doc_indices", []) or [],
        list(bm25_ranked)[: max(int(config.proposal_candidate_depth), int(config.retrieval_top_k), 1)],
    ):
        docs.extend(int(doc_idx) for doc_idx in sequence if int(doc_idx) >= 0)
    return unique_ranked(docs)[: max(int(config.candidate_limit), int(config.retrieval_top_k), 1)]


def _expand_docs_from_seed_fact_endpoints(
    *,
    facts: Sequence[Mapping[str, Any]],
    seed_weights: Mapping[int, float],
    corpus_index: Mapping[str, Any],
    max_endpoint_degree: int,
) -> List[int]:
    """Expand candidate documents through endpoints of query-grounded facts."""

    if not seed_weights:
        return []
    endpoint_to_docs: Mapping[str, Sequence[int]] = corpus_index.get("endpoint_to_docs", {}) or {}
    doc_scores: Counter[int] = Counter()
    for unit in facts:
        unit_id = _unit_id(unit)
        seed_mass = float(seed_weights.get(unit_id, 0.0) or 0.0)
        if seed_mass <= 0.0:
            continue
        for endpoint in _fact_role_endpoints(unit):
            doc_indices = [int(doc_idx) for doc_idx in endpoint_to_docs.get(endpoint, []) or [] if int(doc_idx) >= 0]
            if not doc_indices or len(doc_indices) > max_endpoint_degree:
                continue
            for doc_idx in doc_indices:
                doc_scores[int(doc_idx)] += seed_mass / float(len(doc_indices))
    return [
        int(doc_idx)
        for doc_idx, _score in sorted(doc_scores.items(), key=lambda item: (-float(item[1]), int(item[0])))
    ]


def _build_fact_graph(
    *,
    facts: Sequence[Mapping[str, Any]],
    corpus_index: Mapping[str, Any],
    max_endpoint_degree: int,
) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
    endpoint_to_units: Dict[str, List[int]] = defaultdict(list)
    unit_by_id: Dict[int, Mapping[str, Any]] = {}
    doc_to_units: Dict[int, List[int]] = defaultdict(list)
    endpoint_to_docs: Mapping[str, Sequence[int]] = corpus_index.get("endpoint_to_docs", {}) or {}

    for unit in facts:
        unit_id = _unit_id(unit)
        if unit_id < 0:
            continue
        unit_by_id[unit_id] = unit
        doc_to_units[int(unit.get("doc_index", -1))].append(unit_id)
        for endpoint in unit.get("_endpoints", []) or []:
            endpoint = str(endpoint)
            if not endpoint:
                continue
            endpoint_to_units[endpoint].append(unit_id)

    adjacency: Dict[int, Set[int]] = {unit_id: set() for unit_id in unit_by_id}
    transfer_bearing_units: Set[int] = set()
    skipped_hub_endpoints = 0
    endpoint_edge_count = 0
    for endpoint, unit_ids in sorted(endpoint_to_units.items()):
        global_doc_degree = len(endpoint_to_docs.get(endpoint, []) or [])
        if global_doc_degree > max_endpoint_degree:
            skipped_hub_endpoints += 1
            continue
        clean_ids = unique_ranked(unit_ids)
        if len(clean_ids) < 2:
            continue
        for left_index, left_id in enumerate(clean_ids):
            left_doc = int(unit_by_id[left_id].get("doc_index", -1))
            for right_id in clean_ids[left_index + 1 :]:
                right_doc = int(unit_by_id[right_id].get("doc_index", -1))
                if left_doc == right_doc:
                    continue
                adjacency[left_id].add(right_id)
                adjacency[right_id].add(left_id)
                transfer_bearing_units.update([left_id, right_id])
                endpoint_edge_count += 1

    source_edge_count = 0
    for unit_ids in doc_to_units.values():
        clean_ids = [unit_id for unit_id in unique_ranked(unit_ids) if unit_id in transfer_bearing_units]
        for left_index, left_id in enumerate(clean_ids):
            for right_id in clean_ids[left_index + 1 :]:
                if right_id not in adjacency[left_id]:
                    source_edge_count += 1
                adjacency[left_id].add(right_id)
                adjacency[right_id].add(left_id)

    return (
        {unit_id: sorted(neighbors) for unit_id, neighbors in adjacency.items()},
        {
            "fact_node_count": len(unit_by_id),
            "endpoint_edge_count": int(endpoint_edge_count),
            "source_edge_count": int(source_edge_count),
            "hub_endpoint_skipped_count": int(skipped_hub_endpoints),
        },
    )


def _fact_seed_distribution(
    *,
    query: str,
    facts: Sequence[Mapping[str, Any]],
    corpus_index: Mapping[str, Any],
    max_endpoint_degree: int,
) -> Dict[int, float]:
    """Build a query-grounded reset distribution over candidate fact units.

    The reset distribution is fact-level rather than anchor-prefix-level:
    first use facts whose subject/object endpoint tokens overlap the query; if none are
    available, fall back to broader fact text overlap. Scores are closed-form
    IDF products from the corpus index, not tuned channel weights.
    """

    query_tokens = content_tokens(query) - QUERY_FUNCTION_TOKENS
    if not query_tokens:
        return {}

    token_idf: Mapping[str, float] = corpus_index.get("token_idf", {}) or {}
    endpoint_idf: Mapping[str, float] = corpus_index.get("endpoint_idf", {}) or {}
    matched_query_endpoints = _query_matched_endpoints(
        query=query,
        corpus_index=corpus_index,
        max_endpoint_degree=max_endpoint_degree,
    )
    exact_endpoint_seed_scores: Dict[int, float] = {}
    structural_seed_scores: Dict[int, float] = {}
    text_seed_scores: Dict[int, float] = {}

    def token_mass(tokens: Set[str]) -> float:
        return sum(float(token_idf.get(token, 1.0)) for token in tokens)

    def endpoint_specificity(unit: Mapping[str, Any]) -> float:
        endpoint_scores = [
            float(endpoint_idf.get(endpoint, 1.0))
            for endpoint in _fact_role_endpoints(unit)
        ]
        if not endpoint_scores:
            return 1.0
        return sum(endpoint_scores) / float(len(endpoint_scores))

    for unit in facts:
        unit_id = _unit_id(unit)
        if unit_id < 0:
            continue
        matched_unit_endpoints = matched_query_endpoints & set(_fact_role_endpoints(unit))
        if matched_unit_endpoints:
            exact_endpoint_seed_scores[unit_id] = sum(
                float(endpoint_idf.get(endpoint, 1.0))
                for endpoint in matched_unit_endpoints
            )
            continue

        endpoint_tokens = _fact_role_endpoint_tokens(unit)
        structural_overlap = query_tokens & endpoint_tokens
        if structural_overlap:
            structural_seed_scores[unit_id] = token_mass(structural_overlap) * endpoint_specificity(unit)
            continue

        text_overlap = query_tokens & set(unit.get("_tokens", []) or [])
        if text_overlap:
            text_seed_scores[unit_id] = token_mass(text_overlap)

    seed_scores = exact_endpoint_seed_scores or structural_seed_scores or text_seed_scores
    total = sum(float(score) for score in seed_scores.values() if float(score) > 0.0)
    if total <= 0.0:
        return {}
    return {
        int(unit_id): float(score) / total
        for unit_id, score in sorted(seed_scores.items())
        if float(score) > 0.0
    }


def local_ppr_push(
    *,
    out_neighbors: Mapping[int, Sequence[int]],
    seed_node_ids: Sequence[int] | None = None,
    seed_weights: Mapping[int, float] | None = None,
    alpha: float = DEFAULT_LOCAL_PPR_ALPHA,
    residual_epsilon: float = DEFAULT_LOCAL_PPR_RESIDUAL_EPSILON,
    max_pushes: int = 200_000,
) -> Dict[str, Any]:
    """Run a deterministic local-push Personalized PageRank from seed facts."""

    graph_nodes = set(out_neighbors)
    if seed_weights is not None:
        raw_seed_weights = {
            int(node_id): float(weight)
            for node_id, weight in seed_weights.items()
            if int(node_id) in graph_nodes and float(weight) > 0.0
        }
        total_seed_weight = sum(raw_seed_weights.values())
        seeds = unique_ranked(raw_seed_weights.keys())
        normalized_seed_weights = {
            int(node_id): float(weight) / float(total_seed_weight)
            for node_id, weight in raw_seed_weights.items()
        } if total_seed_weight > 0.0 else {}
    else:
        seeds = [
            int(node_id)
            for node_id in unique_ranked(seed_node_ids or [])
            if int(node_id) in graph_nodes
        ]
        normalized_seed_weights = {
            int(node_id): 1.0 / float(len(seeds))
            for node_id in seeds
        } if seeds else {}

    if not normalized_seed_weights:
        return {
            "estimate": {},
            "source_node_count": 0,
            "push_count": 0,
            "remaining_residual_mass": 0.0,
            "truncated": False,
        }

    estimate: Counter[int] = Counter()
    residual: Counter[int] = Counter(normalized_seed_weights)
    queue: Deque[int] = deque(seeds)
    queued = set(seeds)
    push_count = 0

    while queue and push_count < max_pushes:
        node_id = queue.popleft()
        queued.discard(node_id)
        degree = max(len(out_neighbors.get(node_id, []) or []), 1)
        if residual[node_id] / float(degree) <= residual_epsilon:
            continue
        mass = float(residual[node_id])
        residual[node_id] = 0.0
        estimate[node_id] += float(alpha) * mass
        share = (1.0 - float(alpha)) * mass / float(degree)
        for neighbor_id in out_neighbors.get(node_id, []) or []:
            neighbor_id = int(neighbor_id)
            residual[neighbor_id] += share
            neighbor_degree = max(len(out_neighbors.get(neighbor_id, []) or []), 1)
            if residual[neighbor_id] / float(neighbor_degree) > residual_epsilon and neighbor_id not in queued:
                queue.append(neighbor_id)
                queued.add(neighbor_id)
        push_count += 1

    return {
        "estimate": {int(node_id): float(score) for node_id, score in estimate.items() if score > 0.0},
        "source_node_count": len(seeds),
        "push_count": int(push_count),
        "remaining_residual_mass": float(sum(residual.values())),
        "truncated": bool(queue),
    }


def _ppr_sweep_cut_cluster(
    *,
    out_neighbors: Mapping[int, Sequence[int]],
    ppr_estimate: Mapping[int, float],
) -> Dict[str, Any]:
    """Induce a low-conductance local fact cluster from a PPR vector."""

    scored_nodes = [
        int(node_id)
        for node_id, score in ppr_estimate.items()
        if float(score) > 0.0 and int(node_id) in out_neighbors
    ]
    if not scored_nodes:
        return {
            "node_ids": [],
            "conductance": None,
            "sweep_size": 0,
            "cut_edges": 0,
            "volume": 0,
            "total_volume": 0,
        }

    degree = {int(node_id): len(out_neighbors.get(int(node_id), []) or []) for node_id in out_neighbors}
    total_volume = sum(int(value) for value in degree.values())
    if total_volume <= 0:
        return {
            "node_ids": unique_ranked(scored_nodes),
            "conductance": 0.0,
            "sweep_size": len(scored_nodes),
            "cut_edges": 0,
            "volume": 0,
            "total_volume": 0,
        }

    ordered = sorted(
        scored_nodes,
        key=lambda node_id: (
            -float(ppr_estimate.get(int(node_id), 0.0)) / float(max(degree.get(int(node_id), 0), 1)),
            int(node_id),
        ),
    )
    selected: Set[int] = set()
    prefix: List[int] = []
    best_prefix: List[int] = []
    best_conductance: float | None = None
    best_cut_edges = 0
    best_volume = 0
    cut_edges = 0
    volume = 0

    for node_id in ordered:
        node_id = int(node_id)
        internal_neighbors = sum(1 for neighbor_id in out_neighbors.get(node_id, []) or [] if int(neighbor_id) in selected)
        selected.add(node_id)
        prefix.append(node_id)
        node_degree = int(degree.get(node_id, 0))
        volume += node_degree
        cut_edges += node_degree - 2 * int(internal_neighbors)
        denominator = min(volume, total_volume - volume)
        if denominator <= 0:
            continue
        conductance = float(cut_edges) / float(denominator)
        if best_conductance is None or conductance < best_conductance:
            best_conductance = conductance
            best_prefix = list(prefix)
            best_cut_edges = int(cut_edges)
            best_volume = int(volume)

    if not best_prefix:
        best_prefix = unique_ranked(ordered)
        best_conductance = None
        best_cut_edges = int(cut_edges)
        best_volume = int(volume)

    return {
        "node_ids": best_prefix,
        "conductance": round(float(best_conductance), 6) if best_conductance is not None else None,
        "sweep_size": len(best_prefix),
        "cut_edges": int(best_cut_edges),
        "volume": int(best_volume),
        "total_volume": int(total_volume),
    }


def _rank_docs_by_ppr(
    *,
    facts: Sequence[Mapping[str, Any]],
    ppr_estimate: Mapping[int, float],
    candidate_docs: Sequence[int],
    cluster_node_ids: Sequence[int] | None = None,
) -> List[Dict[str, Any]]:
    doc_scores: Counter[int] = Counter()
    cluster_doc_scores: Counter[int] = Counter()
    doc_fact_counts: Counter[int] = Counter()
    cluster_doc_fact_counts: Counter[int] = Counter()
    doc_total_fact_counts: Counter[int] = Counter()
    cluster_nodes = {int(node_id) for node_id in cluster_node_ids or []}
    for unit in facts:
        unit_id = _unit_id(unit)
        doc_idx = int(unit.get("doc_index", -1))
        if unit_id < 0 or doc_idx < 0:
            continue
        doc_total_fact_counts[doc_idx] += 1
        mass = float(ppr_estimate.get(unit_id, 0.0))
        if mass <= 0.0:
            continue
        doc_scores[doc_idx] += mass
        doc_fact_counts[doc_idx] += 1
        if unit_id in cluster_nodes:
            cluster_doc_scores[doc_idx] += mass
            cluster_doc_fact_counts[doc_idx] += 1

    prior_rank = {int(doc_idx): rank for rank, doc_idx in enumerate(unique_ranked(candidate_docs), start=1)}
    rows: List[Dict[str, Any]] = []
    for doc_idx in unique_ranked(list(candidate_docs) + list(doc_scores.keys())):
        total_fact_count = max(int(doc_total_fact_counts.get(int(doc_idx), 0)), 1)
        normalized_score = float(doc_scores.get(int(doc_idx), 0.0)) / (float(total_fact_count) ** 0.5)
        cluster_normalized_score = float(cluster_doc_scores.get(int(doc_idx), 0.0)) / (float(total_fact_count) ** 0.5)
        rows.append(
            {
                "doc_index": int(doc_idx),
                "local_ppr_score": float(doc_scores.get(int(doc_idx), 0.0)),
                "local_ppr_normalized_score": float(normalized_score),
                "local_ppr_cluster_score": float(cluster_doc_scores.get(int(doc_idx), 0.0)),
                "local_ppr_cluster_normalized_score": float(cluster_normalized_score),
                "ppr_positive_fact_count": int(doc_fact_counts.get(int(doc_idx), 0)),
                "ppr_cluster_fact_count": int(cluster_doc_fact_counts.get(int(doc_idx), 0)),
                "candidate_fact_count": int(doc_total_fact_counts.get(int(doc_idx), 0)),
                "in_ppr_sweep_cluster": bool(cluster_doc_fact_counts.get(int(doc_idx), 0) > 0),
                "candidate_prior_rank": int(prior_rank.get(int(doc_idx), 10**9)),
            }
        )
    rows.sort(
        key=lambda row: (
            0 if bool(row.get("in_ppr_sweep_cluster", False)) else 1,
            -float(row["local_ppr_cluster_normalized_score"]),
            -float(row["local_ppr_cluster_score"]),
            -float(row["local_ppr_normalized_score"]),
            -float(row["local_ppr_score"]),
            int(row["candidate_prior_rank"]),
            int(row["doc_index"]),
        )
    )
    return rows


def _doc_to_fact_unit_ids(facts: Sequence[Mapping[str, Any]]) -> Dict[int, List[int]]:
    doc_to_units: Dict[int, List[int]] = defaultdict(list)
    for unit in facts:
        unit_id = _unit_id(unit)
        doc_idx = int(unit.get("doc_index", -1))
        if unit_id >= 0 and doc_idx >= 0:
            doc_to_units[doc_idx].append(unit_id)
    return {doc_idx: unique_ranked(unit_ids) for doc_idx, unit_ids in doc_to_units.items()}


def _fact_graph_doc_distances(
    *,
    selected_docs: Sequence[int],
    facts: Sequence[Mapping[str, Any]],
    fact_graph: Mapping[int, Sequence[int]],
) -> Dict[int, int]:
    doc_to_units = _doc_to_fact_unit_ids(facts)
    source_units = [
        unit_id
        for doc_idx in unique_ranked(selected_docs)
        for unit_id in doc_to_units.get(int(doc_idx), []) or []
    ]
    if not source_units:
        return {}
    unit_to_doc: Dict[int, int] = {}
    for doc_idx, unit_ids in doc_to_units.items():
        for unit_id in unit_ids:
            unit_to_doc[int(unit_id)] = int(doc_idx)
    distances: Dict[int, int] = {}
    queue: Deque[Tuple[int, int]] = deque()
    seen: Set[int] = set()
    for unit_id in unique_ranked(source_units):
        queue.append((int(unit_id), 0))
        seen.add(int(unit_id))
    while queue:
        unit_id, distance = queue.popleft()
        doc_idx = unit_to_doc.get(int(unit_id))
        if doc_idx is not None:
            distances[doc_idx] = min(int(distances.get(doc_idx, distance)), int(distance))
        for neighbor_id in fact_graph.get(int(unit_id), []) or []:
            neighbor_id = int(neighbor_id)
            if neighbor_id in seen:
                continue
            seen.add(neighbor_id)
            queue.append((neighbor_id, distance + 1))
    return distances


def _order_docs_by_graph_ppr_completion(
    *,
    query: str,
    matched_query_endpoints: Sequence[str],
    ranked_rows: Sequence[Mapping[str, Any]],
    corpus_index: Mapping[str, Any],
    facts: Sequence[Mapping[str, Any]],
    fact_graph: Mapping[int, Sequence[int]],
    max_endpoint_degree: int,
) -> List[int]:
    selected: List[int] = []
    row_by_doc = {int(row.get("doc_index", -1)): dict(row) for row in ranked_rows}
    remaining = [int(row.get("doc_index", -1)) for row in ranked_rows]
    remaining = unique_ranked(remaining)
    query_tokens = content_tokens(query) - QUERY_FUNCTION_TOKENS
    doc_endpoint_tokens: Mapping[int, Sequence[str]] = corpus_index.get("doc_endpoint_tokens", {}) or {}
    doc_title_tokens: Mapping[int, Sequence[str]] = corpus_index.get("doc_title_tokens", {}) or {}
    doc_token_counts: Mapping[int, Counter[str]] = corpus_index.get("doc_token_counts", {}) or {}
    matched_endpoint_set = {str(endpoint) for endpoint in matched_query_endpoints if str(endpoint)}
    doc_to_query_endpoints: Dict[int, Set[str]] = defaultdict(set)
    for unit in facts:
        doc_idx = int(unit.get("doc_index", -1))
        if doc_idx < 0:
            continue
        doc_to_query_endpoints[doc_idx].update(matched_endpoint_set & set(_fact_role_endpoints(unit)))

    def doc_structural_query_tokens(doc_idx: int) -> Set[str]:
        return (
            set(doc_endpoint_tokens.get(int(doc_idx), []) or [])
            | set(doc_title_tokens.get(int(doc_idx), []) or [])
            | set((doc_token_counts.get(int(doc_idx), Counter()) or Counter()).keys())
        ) & query_tokens

    covered_tokens: Set[str] = set()
    covered_endpoints: Set[str] = set()

    while remaining:
        doc_distances = _fact_graph_doc_distances(selected_docs=selected, facts=facts, fact_graph=fact_graph)
        rows: List[Tuple[Any, ...]] = []
        for doc_idx in remaining:
            transition_weight, _shared = transition_weight_to_support_set(
                doc_idx=int(doc_idx),
                support_docs=selected,
                corpus_index=corpus_index,
                max_endpoint_degree=max_endpoint_degree,
            )
            ppr_row = row_by_doc.get(int(doc_idx), {})
            new_structural_token_count = len(doc_structural_query_tokens(int(doc_idx)) - covered_tokens)
            new_endpoint_count = len(doc_to_query_endpoints.get(int(doc_idx), set()) - covered_endpoints)
            graph_distance = int(doc_distances.get(int(doc_idx), 0 if not selected else 10**9))
            rows.append(
                (
                    graph_distance,
                    0 if transition_weight > 0.0 else 1,
                    0 if bool(ppr_row.get("in_ppr_sweep_cluster", False)) else 1,
                    -int(new_endpoint_count),
                    -int(new_structural_token_count),
                    -float(ppr_row.get("local_ppr_cluster_normalized_score", 0.0) or 0.0),
                    -float(ppr_row.get("local_ppr_cluster_score", 0.0) or 0.0),
                    -float(ppr_row.get("local_ppr_normalized_score", 0.0) or 0.0),
                    -float(ppr_row.get("local_ppr_score", 0.0) or 0.0),
                    int(ppr_row.get("candidate_prior_rank", 10**9) or 10**9),
                    int(doc_idx),
                )
            )
        chosen = sorted(rows)[0][-1]
        selected.append(int(chosen))
        covered_tokens.update(doc_structural_query_tokens(int(chosen)))
        covered_endpoints.update(doc_to_query_endpoints.get(int(chosen), set()))
        remaining = [doc_idx for doc_idx in remaining if int(doc_idx) != int(chosen)]
    return selected


def retrieve_local_ppr_evidence_set(
    *,
    query: str,
    corpus_index: Mapping[str, Any],
    config: AGSTOConfig,
    anchor_doc_indices: Sequence[int] | None = None,
    native_dense_doc_indices: Sequence[int] | None = None,
    semantic_query_embedding: Any | None = None,
    chunk_embedding_matrix: Any | None = None,
    semantic_residual_weight: float = 8.0,
    alpha: float = DEFAULT_LOCAL_PPR_ALPHA,
    residual_epsilon: float = DEFAULT_LOCAL_PPR_RESIDUAL_EPSILON,
) -> Dict[str, Any]:
    """Retrieve evidence docs by applying local PPR to native AG-STO candidates."""

    proposals = build_native_sto_proposals(
        query=query,
        corpus_index=corpus_index,
        config=config,
        anchor_doc_indices=anchor_doc_indices,
        native_dense_doc_indices=native_dense_doc_indices,
        semantic_query_embedding=semantic_query_embedding,
        chunk_embedding_matrix=chunk_embedding_matrix,
        semantic_residual_weight=semantic_residual_weight,
    )
    bm25_scores = score_docs_bm25(query=query, corpus_index=corpus_index)
    bm25_ranked = [
        int(doc_idx)
        for doc_idx, _score in sorted(bm25_scores.items(), key=lambda item: (-float(item[1]), int(item[0])))
    ]
    proposal_candidate_docs = _candidate_docs_from_proposals(
        proposals=proposals,
        bm25_ranked=bm25_ranked,
        config=config,
    )
    entry_docs = _entry_docs_for_fact_grounding(
        proposals=proposals,
        bm25_ranked=bm25_ranked,
        config=config,
    )
    entry_facts = _fact_units_for_docs(corpus_index=corpus_index, doc_indices=entry_docs)
    entry_seed_weights = _fact_seed_distribution(
        query=query,
        facts=entry_facts,
        corpus_index=corpus_index,
        max_endpoint_degree=config.max_endpoint_degree,
    )
    query_grounded_expansion_docs = _expand_docs_from_seed_fact_endpoints(
        facts=entry_facts,
        seed_weights=entry_seed_weights,
        corpus_index=corpus_index,
        max_endpoint_degree=config.max_endpoint_degree,
    )
    candidate_docs = unique_ranked(
        list(entry_docs)
        + list(query_grounded_expansion_docs)
        + list(proposal_candidate_docs)
        + list(bm25_ranked)
    )[: max(int(config.candidate_limit), int(config.retrieval_top_k), 1)]
    facts = _fact_units_for_docs(corpus_index=corpus_index, doc_indices=candidate_docs)
    graph, graph_stats = _build_fact_graph(
        facts=facts,
        corpus_index=corpus_index,
        max_endpoint_degree=config.max_endpoint_degree,
    )
    matched_query_endpoints = sorted(
        _query_matched_endpoints(
            query=query,
            corpus_index=corpus_index,
            max_endpoint_degree=config.max_endpoint_degree,
        )
    )
    seed_weights = _fact_seed_distribution(
        query=query,
        facts=facts,
        corpus_index=corpus_index,
        max_endpoint_degree=config.max_endpoint_degree,
    )
    seed_ids = list(seed_weights.keys())
    ppr = local_ppr_push(
        out_neighbors=graph,
        seed_weights=seed_weights,
        alpha=alpha,
        residual_epsilon=residual_epsilon,
    )
    ppr_sweep_cluster = _ppr_sweep_cut_cluster(
        out_neighbors=graph,
        ppr_estimate=ppr.get("estimate", {}) or {},
    )
    ranked_rows = _rank_docs_by_ppr(
        facts=facts,
        ppr_estimate=ppr.get("estimate", {}) or {},
        candidate_docs=candidate_docs,
        cluster_node_ids=[],
    )
    ranked_docs = _order_docs_by_graph_ppr_completion(
        query=query,
        matched_query_endpoints=[],
        ranked_rows=ranked_rows,
        corpus_index=corpus_index,
        facts=facts,
        fact_graph=graph,
        max_endpoint_degree=config.max_endpoint_degree,
    )
    retrieved_docs = ranked_docs[: max(int(config.retrieval_top_k), 1)]
    selected_docs = retrieved_docs[: max(int(config.evidence_set_size), 1)]
    selected_score = sum(
        float(row.get("local_ppr_score", 0.0) or 0.0)
        for row in ranked_rows
        if int(row.get("doc_index", -1)) in set(selected_docs)
    )

    return {
        "method": "AG-STO-local-PPR",
        "agsto_policy": config.policy,
        "clean_api": True,
        "uses_sfb_outputs": False,
        "uses_query_obligations": False,
        "retrieved_doc_indices": retrieved_docs,
        "candidate_doc_indices": candidate_docs,
        "entry_doc_indices": entry_docs,
        "query_grounded_expansion_doc_indices": query_grounded_expansion_docs,
        "candidate_doc_count": len(candidate_docs),
        "stable_anchor_doc_indices": [],
        "native_anchor_doc_indices": unique_ranked(proposals.get("anchor_doc_indices", []) or []),
        "selected_evidence_set": {
            "doc_indices": selected_docs,
            "score": round(float(selected_score), 6),
            "selection_rule": "query_grounded_local_ppr_connected_set",
        },
        "local_ppr": {
            "alpha": float(alpha),
            "residual_epsilon": float(residual_epsilon),
            "seed_unit_ids": seed_ids,
            "seed_unit_weights": {str(unit_id): round(float(weight), 8) for unit_id, weight in seed_weights.items()},
            "seed_rule": "maximal_query_endpoint_fact_reset",
            "matched_query_endpoints": matched_query_endpoints,
            "source_node_count": int(ppr.get("source_node_count", 0)),
            "push_count": int(ppr.get("push_count", 0)),
            "remaining_residual_mass": float(ppr.get("remaining_residual_mass", 0.0)),
            "truncated": bool(ppr.get("truncated", False)),
            "sweep_cluster": {
                "node_count": len(ppr_sweep_cluster.get("node_ids", []) or []),
                "conductance": ppr_sweep_cluster.get("conductance"),
                "cut_edges": int(ppr_sweep_cluster.get("cut_edges", 0) or 0),
                "volume": int(ppr_sweep_cluster.get("volume", 0) or 0),
                "total_volume": int(ppr_sweep_cluster.get("total_volume", 0) or 0),
            },
        },
        "local_fact_graph": graph_stats,
        "doc_ppr_rows": ranked_rows[: max(int(config.retrieval_top_k), 1)],
        "native_proposals": proposals,
    }
