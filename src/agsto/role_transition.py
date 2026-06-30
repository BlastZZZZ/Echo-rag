"""Role-aware transition edges for the Source/Title/OpenIE graph.

The functions here define STO edges from the graph substrate itself. They do
not use prompts, query-specific schemas, learned weights, gold labels, or
dataset-specific rules.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .index import informative_endpoint
from .ranking import unique_ranked


ROLE_BRIDGE = "role_bridge"
SENTENCE_GROUNDED_TRANSITION = "sentence_grounded_transition"
SAME_SUBJECT = "same_subject"
SAME_OBJECT = "same_object"
TITLE_ROLE_GROUNDING = "title_role_grounding"
SOURCE_ENDPOINT_INCIDENCE = "source_endpoint_incidence"

EVIDENCE_TRANSITION_TIER = "evidence_transition"
WEAK_CONNECTIVITY_TIER = "weak_connectivity"
UNKNOWN_EDGE_TIER = "unknown"

ROLE_TRANSITION_ORDER = {
    SENTENCE_GROUNDED_TRANSITION: 0,
    ROLE_BRIDGE: 1,
    SAME_SUBJECT: 2,
    SAME_OBJECT: 3,
    TITLE_ROLE_GROUNDING: 4,
    SOURCE_ENDPOINT_INCIDENCE: 5,
}

EDGE_EVIDENCE_TIER_BY_KIND = {
    SENTENCE_GROUNDED_TRANSITION: EVIDENCE_TRANSITION_TIER,
    TITLE_ROLE_GROUNDING: EVIDENCE_TRANSITION_TIER,
    ROLE_BRIDGE: WEAK_CONNECTIVITY_TIER,
    SAME_SUBJECT: WEAK_CONNECTIVITY_TIER,
    SAME_OBJECT: WEAK_CONNECTIVITY_TIER,
    SOURCE_ENDPOINT_INCIDENCE: WEAK_CONNECTIVITY_TIER,
}

EDGE_EVIDENCE_TIER_ORDER = {
    EVIDENCE_TRANSITION_TIER: 0,
    WEAK_CONNECTIVITY_TIER: 1,
    UNKNOWN_EDGE_TIER: 2,
}


def edge_evidence_tier_for_kind(kind: str) -> str:
    return EDGE_EVIDENCE_TIER_BY_KIND.get(str(kind), UNKNOWN_EDGE_TIER)


def edge_evidence_tiers(kinds: Sequence[str]) -> List[str]:
    tiers = {edge_evidence_tier_for_kind(str(kind)) for kind in kinds}
    return sorted(tiers, key=lambda tier: (EDGE_EVIDENCE_TIER_ORDER.get(str(tier), 10), str(tier)))


def best_edge_evidence_tier(kinds: Sequence[str]) -> str:
    tiers = edge_evidence_tiers(kinds)
    return tiers[0] if tiers else UNKNOWN_EDGE_TIER


def _unit_id(unit: Mapping[str, Any]) -> int:
    try:
        return int(unit.get("_unit_int_id"))
    except (TypeError, ValueError):
        return -1


def fact_subject_endpoint(unit: Mapping[str, Any]) -> str:
    if str(unit.get("unit_type") or "") != "openie_fact":
        return ""
    return informative_endpoint(unit.get("subject", ""))


def fact_object_endpoint(unit: Mapping[str, Any]) -> str:
    if str(unit.get("unit_type") or "") != "openie_fact":
        return ""
    return informative_endpoint(unit.get("object", ""))


def fact_role_endpoints(unit: Mapping[str, Any]) -> Set[str]:
    return {endpoint for endpoint in (fact_subject_endpoint(unit), fact_object_endpoint(unit)) if endpoint}


def source_span_entity_endpoints(unit: Mapping[str, Any]) -> Set[str]:
    if str(unit.get("unit_type") or "") != "source_span":
        return set()
    title_endpoint = informative_endpoint(unit.get("title", ""))
    values = list(unit.get("span_entities", []) or [])
    if not values:
        values = [
            value
            for value in unit.get("raw_endpoints", []) or []
            if informative_endpoint(value) and informative_endpoint(value) != title_endpoint
        ]
    endpoints = {
        endpoint
        for endpoint in (informative_endpoint(value) for value in values)
        if endpoint and endpoint != title_endpoint
    }
    return endpoints


def title_alias_endpoints(title: Any) -> Set[str]:
    """Return conservative aliases for matching a page title to its own facts."""

    raw_title = str(title or "").strip()
    aliases = {endpoint for endpoint in [informative_endpoint(raw_title)] if endpoint}
    if not raw_title:
        return aliases
    without_parenthetical = re.sub(r"\s*\([^)]*\)\s*", " ", raw_title).strip()
    parenthetical_alias = informative_endpoint(without_parenthetical)
    if parenthetical_alias:
        aliases.add(parenthetical_alias)
    return aliases


def _edge_key(left_doc: int, right_doc: int) -> Tuple[int, int]:
    return tuple(sorted((int(left_doc), int(right_doc))))  # type: ignore[return-value]


def _add_transition_edge(
    *,
    edge_map: Dict[Tuple[int, int], Dict[str, Any]],
    left_doc: int,
    right_doc: int,
    endpoint: str,
    kind: str,
    left_unit_id: int | None = None,
    right_unit_id: int | None = None,
    sample_fields: Mapping[str, Any] | None = None,
) -> None:
    if int(left_doc) == int(right_doc):
        return
    key = _edge_key(int(left_doc), int(right_doc))
    row = edge_map.setdefault(
        key,
        {
            "left_doc": key[0],
            "right_doc": key[1],
            "kinds": set(),
            "endpoints": set(),
            "samples": [],
        },
    )
    row["kinds"].add(str(kind))
    row["endpoints"].add(str(endpoint))
    if len(row["samples"]) < 5:
        sample = {
            "kind": str(kind),
            "endpoint": str(endpoint),
            "left_unit_id": int(left_unit_id) if left_unit_id is not None else None,
            "right_unit_id": int(right_unit_id) if right_unit_id is not None else None,
        }
        sample.update(dict(sample_fields or {}))
        row["samples"].append(sample)


def _finalize_edge_map(edge_map: Mapping[Tuple[int, int], Mapping[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    finalized: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for key, row in edge_map.items():
        kinds = sorted(set(row.get("kinds", set())), key=lambda item: (ROLE_TRANSITION_ORDER.get(str(item), 10), str(item)))
        finalized[key] = {
            "left_doc": int(row.get("left_doc", key[0])),
            "right_doc": int(row.get("right_doc", key[1])),
            "kinds": kinds,
            "best_kind": kinds[0] if kinds else "",
            "best_kind_rank": int(ROLE_TRANSITION_ORDER.get(kinds[0], 10)) if kinds else 10,
            "evidence_tiers": edge_evidence_tiers(kinds),
            "best_evidence_tier": best_edge_evidence_tier(kinds),
            "endpoints": sorted(set(str(endpoint) for endpoint in row.get("endpoints", set()))),
            "samples": list(row.get("samples", []) or []),
        }
    return finalized


def build_role_transition_graph(
    *,
    corpus_index: Mapping[str, Any],
    max_endpoint_degree: int,
    include_title_role_grounding: bool = True,
) -> Dict[str, Any]:
    """Build a document graph induced by role-aware OpenIE endpoint transitions."""

    subject_facts_by_endpoint_doc: Dict[str, Dict[int, Mapping[str, Any]]] = defaultdict(dict)
    object_facts_by_endpoint_doc: Dict[str, Dict[int, Mapping[str, Any]]] = defaultdict(dict)
    role_docs_by_endpoint: Dict[str, Set[int]] = defaultdict(set)
    source_span_docs_by_endpoint: Dict[str, Set[int]] = defaultdict(set)
    source_span_units_by_endpoint_doc: Dict[str, Dict[int, Mapping[str, Any]]] = defaultdict(dict)
    doc_to_role_endpoints: Dict[int, Set[str]] = defaultdict(set)
    doc_titles: Dict[int, str] = {}
    doc_title_aliases: Dict[int, Set[str]] = defaultdict(set)
    title_docs_by_endpoint: Dict[str, Set[int]] = defaultdict(set)
    fact_units: List[Mapping[str, Any]] = []

    for unit in corpus_index.get("units", []) or []:
        doc_idx = int(unit.get("doc_index", -1))
        if doc_idx < 0:
            continue
        raw_title = str(unit.get("title", ""))
        title_endpoint = informative_endpoint(raw_title)
        if title_endpoint:
            doc_titles.setdefault(doc_idx, title_endpoint)
            doc_title_aliases[doc_idx].update(title_alias_endpoints(raw_title))
            title_docs_by_endpoint[title_endpoint].add(doc_idx)
        if str(unit.get("unit_type") or "") == "source_span":
            for endpoint in source_span_entity_endpoints(unit):
                source_span_docs_by_endpoint[endpoint].add(doc_idx)
                source_span_units_by_endpoint_doc[endpoint].setdefault(doc_idx, unit)
            continue
        if str(unit.get("unit_type") or "") != "openie_fact":
            continue
        subject = fact_subject_endpoint(unit)
        obj = fact_object_endpoint(unit)
        if subject:
            subject_facts_by_endpoint_doc[subject].setdefault(doc_idx, unit)
            role_docs_by_endpoint[subject].add(doc_idx)
            doc_to_role_endpoints[doc_idx].add(subject)
        if obj:
            object_facts_by_endpoint_doc[obj].setdefault(doc_idx, unit)
            role_docs_by_endpoint[obj].add(doc_idx)
            doc_to_role_endpoints[doc_idx].add(obj)
        if subject and obj:
            fact_units.append(unit)

    edge_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    sentence_transition_edge_count = 0
    for unit in fact_units:
        source_doc = int(unit.get("doc_index", -1))
        if source_doc < 0:
            continue
        source_title = doc_titles.get(source_doc, "")
        if not source_title:
            continue
        source_title_aliases = set(doc_title_aliases.get(source_doc, set()) or {source_title})
        subject = fact_subject_endpoint(unit)
        obj = fact_object_endpoint(unit)
        if subject in source_title_aliases:
            target_endpoint = obj
        elif obj in source_title_aliases:
            target_endpoint = subject
        else:
            continue
        if not target_endpoint:
            continue
        for target_doc in sorted(title_docs_by_endpoint.get(target_endpoint, set())):
            if int(target_doc) == source_doc:
                continue
            before = len(edge_map)
            _add_transition_edge(
                edge_map=edge_map,
                left_doc=source_doc,
                right_doc=int(target_doc),
                endpoint=target_endpoint,
                kind=SENTENCE_GROUNDED_TRANSITION,
                left_unit_id=_unit_id(unit),
                right_unit_id=None,
                sample_fields={
                    "source_side": "subject" if subject in source_title_aliases else "object",
                    "target_side": "object" if subject in source_title_aliases else "subject",
                    "relation": str(unit.get("relation") or ""),
                },
            )
            sentence_transition_edge_count += 1 if len(edge_map) > before else 0

    endpoints = sorted(set(subject_facts_by_endpoint_doc) | set(object_facts_by_endpoint_doc))
    skipped_hub_endpoint_count = 0
    for endpoint in endpoints:
        endpoint_doc_count = len(role_docs_by_endpoint.get(endpoint, set()))
        if endpoint_doc_count <= 0 or endpoint_doc_count > max_endpoint_degree:
            skipped_hub_endpoint_count += 1
            continue
        subject_facts = list((subject_facts_by_endpoint_doc.get(endpoint, {}) or {}).values())
        object_facts = list((object_facts_by_endpoint_doc.get(endpoint, {}) or {}).values())

        for left_index, left in enumerate(subject_facts):
            for right in subject_facts[left_index + 1 :]:
                _add_transition_edge(
                    edge_map=edge_map,
                    left_doc=int(left.get("doc_index", -1)),
                    right_doc=int(right.get("doc_index", -1)),
                    endpoint=endpoint,
                    kind=SAME_SUBJECT,
                    left_unit_id=_unit_id(left),
                    right_unit_id=_unit_id(right),
                )
        for left_index, left in enumerate(object_facts):
            for right in object_facts[left_index + 1 :]:
                _add_transition_edge(
                    edge_map=edge_map,
                    left_doc=int(left.get("doc_index", -1)),
                    right_doc=int(right.get("doc_index", -1)),
                    endpoint=endpoint,
                    kind=SAME_OBJECT,
                    left_unit_id=_unit_id(left),
                    right_unit_id=_unit_id(right),
                    sample_fields={
                        "left_doc_index": int(left.get("doc_index", -1)),
                        "right_doc_index": int(right.get("doc_index", -1)),
                        "left_relation": str(left.get("relation") or ""),
                        "right_relation": str(right.get("relation") or ""),
                    },
                )
        for left in object_facts:
            for right in subject_facts:
                _add_transition_edge(
                    edge_map=edge_map,
                    left_doc=int(left.get("doc_index", -1)),
                    right_doc=int(right.get("doc_index", -1)),
                    endpoint=endpoint,
                    kind=ROLE_BRIDGE,
                    left_unit_id=_unit_id(left),
                    right_unit_id=_unit_id(right),
                )

    title_role_edge_count = 0
    if include_title_role_grounding:
        for doc_idx, title_endpoint in sorted(doc_titles.items()):
            role_docs = sorted(role_docs_by_endpoint.get(title_endpoint, set()))
            if not role_docs or len(role_docs) > max_endpoint_degree:
                continue
            for other_doc in role_docs:
                before = len(edge_map)
                _add_transition_edge(
                    edge_map=edge_map,
                    left_doc=int(doc_idx),
                    right_doc=int(other_doc),
                    endpoint=title_endpoint,
                    kind=TITLE_ROLE_GROUNDING,
                )
                title_role_edge_count += 1 if len(edge_map) > before else 0

    source_incidence_edge_count = 0
    skipped_source_incidence_hub_endpoint_count = 0
    for endpoint in sorted(set(source_span_docs_by_endpoint) & set(role_docs_by_endpoint)):
        source_docs = set(source_span_docs_by_endpoint.get(endpoint, set()) or set())
        role_docs = set(role_docs_by_endpoint.get(endpoint, set()) or set())
        incidence_docs = source_docs | role_docs
        if not source_docs or not role_docs:
            continue
        if len(incidence_docs) > max_endpoint_degree:
            skipped_source_incidence_hub_endpoint_count += 1
            continue
        for source_doc in sorted(source_docs):
            source_unit = source_span_units_by_endpoint_doc.get(endpoint, {}).get(int(source_doc), {})
            for role_doc in sorted(role_docs):
                if int(source_doc) == int(role_doc):
                    continue
                before = len(edge_map)
                _add_transition_edge(
                    edge_map=edge_map,
                    left_doc=int(source_doc),
                    right_doc=int(role_doc),
                    endpoint=endpoint,
                    kind=SOURCE_ENDPOINT_INCIDENCE,
                    left_unit_id=_unit_id(source_unit),
                )
                source_incidence_edge_count += 1 if len(edge_map) > before else 0

    edges = _finalize_edge_map(edge_map)
    adjacency: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for key, edge in edges.items():
        left, right = key
        adjacency[left].append(edge)
        adjacency[right].append(edge)
    edge_tier_counts = {
        EVIDENCE_TRANSITION_TIER: 0,
        WEAK_CONNECTIVITY_TIER: 0,
        UNKNOWN_EDGE_TIER: 0,
    }
    for edge in edges.values():
        edge_tier_counts[str(edge.get("best_evidence_tier") or UNKNOWN_EDGE_TIER)] = (
            edge_tier_counts.get(str(edge.get("best_evidence_tier") or UNKNOWN_EDGE_TIER), 0) + 1
        )
    return {
        "edges": edges,
        "adjacency": {doc_idx: sorted(edge_rows, key=lambda row: (int(row["best_kind_rank"]), int(row["left_doc"]), int(row["right_doc"]))) for doc_idx, edge_rows in adjacency.items()},
        "doc_to_role_endpoints": {doc_idx: sorted(endpoints) for doc_idx, endpoints in doc_to_role_endpoints.items()},
        "endpoint_to_role_docs": {endpoint: sorted(doc_ids) for endpoint, doc_ids in role_docs_by_endpoint.items()},
        "doc_titles": doc_titles,
        "stats": {
            "edge_count": len(edges),
            "sentence_grounded_transition_edge_count": int(sentence_transition_edge_count),
            "skipped_hub_endpoint_count": int(skipped_hub_endpoint_count),
            "title_role_edge_count": int(title_role_edge_count),
            "source_incidence_edge_count": int(source_incidence_edge_count),
            "skipped_source_incidence_hub_endpoint_count": int(skipped_source_incidence_hub_endpoint_count),
            "edge_tier_counts": edge_tier_counts,
        },
    }


def filter_role_transition_graph(
    *,
    role_graph: Mapping[str, Any],
    allowed_kinds: Set[str],
) -> Dict[str, Any]:
    """Keep only role-transition edges whose evidence kind is explicitly allowed."""

    clean_allowed = {str(kind) for kind in allowed_kinds}
    edges: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for key, edge in (role_graph.get("edges", {}) or {}).items():
        original_kinds = [str(kind) for kind in edge.get("kinds", []) or []]
        filtered_kinds = [kind for kind in original_kinds if kind in clean_allowed]
        if not filtered_kinds:
            continue
        if isinstance(key, tuple):
            edge_key = key
        else:
            edge_key = _edge_key(int(edge.get("left_doc", -1)), int(edge.get("right_doc", -1)))
        filtered_samples = [
            sample
            for sample in edge.get("samples", []) or []
            if str(sample.get("kind", "")) in clean_allowed
        ]
        if not filtered_samples:
            filtered_samples = list(edge.get("samples", []) or [])[:1]
        filtered_kinds = sorted(
            set(filtered_kinds),
            key=lambda item: (ROLE_TRANSITION_ORDER.get(str(item), 10), str(item)),
        )
        edges[edge_key] = {
            "left_doc": int(edge.get("left_doc", edge_key[0])),
            "right_doc": int(edge.get("right_doc", edge_key[1])),
            "kinds": filtered_kinds,
            "best_kind": filtered_kinds[0],
            "best_kind_rank": int(ROLE_TRANSITION_ORDER.get(filtered_kinds[0], 10)),
            "evidence_tiers": edge_evidence_tiers(filtered_kinds),
            "best_evidence_tier": best_edge_evidence_tier(filtered_kinds),
            "endpoints": sorted(set(str(endpoint) for endpoint in edge.get("endpoints", []) or [])),
            "samples": filtered_samples[:5],
        }

    adjacency: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for key, edge in edges.items():
        left, right = key
        adjacency[left].append(edge)
        adjacency[right].append(edge)
    stats = dict(role_graph.get("stats", {}) or {})
    stats["edge_count"] = len(edges)
    stats["allowed_kinds"] = sorted(clean_allowed)
    stats["edge_tier_counts"] = {
        EVIDENCE_TRANSITION_TIER: sum(
            1 for edge in edges.values() if str(edge.get("best_evidence_tier")) == EVIDENCE_TRANSITION_TIER
        ),
        WEAK_CONNECTIVITY_TIER: sum(
            1 for edge in edges.values() if str(edge.get("best_evidence_tier")) == WEAK_CONNECTIVITY_TIER
        ),
        UNKNOWN_EDGE_TIER: sum(
            1 for edge in edges.values() if str(edge.get("best_evidence_tier")) == UNKNOWN_EDGE_TIER
        ),
    }
    return {
        **dict(role_graph),
        "edges": edges,
        "adjacency": {
            doc_idx: sorted(
                edge_rows,
                key=lambda row: (int(row["best_kind_rank"]), int(row["left_doc"]), int(row["right_doc"])),
            )
            for doc_idx, edge_rows in adjacency.items()
        },
        "stats": stats,
    }


def connected_components(nodes: Sequence[int], edges: Iterable[Tuple[int, int]]) -> List[List[int]]:
    clean_nodes = unique_ranked([int(node) for node in nodes])
    node_set = set(clean_nodes)
    adjacency: Dict[int, Set[int]] = {node: set() for node in clean_nodes}
    for left, right in edges:
        if left not in node_set or right not in node_set:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: List[List[int]] = []
    seen: Set[int] = set()
    for start in clean_nodes:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: List[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency.get(node, set()):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return components


def role_transition_edges_for_docs(
    *,
    role_graph: Mapping[str, Any],
    doc_indices: Sequence[int],
) -> Dict[str, Any]:
    docs = set(unique_ranked([int(doc_idx) for doc_idx in doc_indices]))
    edge_rows = []
    edge_keys = []
    for key, edge in (role_graph.get("edges", {}) or {}).items():
        left, right = int(edge["left_doc"]), int(edge["right_doc"])
        if left in docs and right in docs:
            edge_keys.append((left, right))
            edge_rows.append(edge)
    components = connected_components(list(docs), edge_keys)
    return {
        "connected": bool(docs) and len(components) == 1,
        "components": components,
        "edge_count": len(edge_rows),
        "kind_counts": {
            kind: sum(1 for edge in edge_rows if kind in set(edge.get("kinds", []) or []))
            for kind in (
                SENTENCE_GROUNDED_TRANSITION,
                ROLE_BRIDGE,
                SAME_SUBJECT,
                SAME_OBJECT,
                TITLE_ROLE_GROUNDING,
                SOURCE_ENDPOINT_INCIDENCE,
            )
        },
        "tier_counts": {
            tier: sum(1 for edge in edge_rows if str(edge.get("best_evidence_tier") or UNKNOWN_EDGE_TIER) == tier)
            for tier in (EVIDENCE_TRANSITION_TIER, WEAK_CONNECTIVITY_TIER, UNKNOWN_EDGE_TIER)
        },
        "edge_samples": edge_rows[:10],
    }
