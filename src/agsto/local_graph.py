"""Query-time local STO graph admission.

This module replaces the historical ``source report`` boundary with an explicit
local graph builder. It is closed-form: no prompt schemas, no learned weights,
no gold labels, and no dataset-specific rules are used.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Any, Deque, Dict, List, Mapping, Sequence, Set, Tuple

from .index import content_tokens
from .lexical import rank_docs_bm25
from .lexical import score_docs_bm25
from .query_grounding import ground_query_endpoints
from .ranking import unique_ranked
from .role_transition import (
    EVIDENCE_TRANSITION_TIER,
    ROLE_BRIDGE,
    SAME_OBJECT,
    SAME_SUBJECT,
    SENTENCE_GROUNDED_TRANSITION,
    SOURCE_ENDPOINT_INCIDENCE,
    TITLE_ROLE_GROUNDING,
    UNKNOWN_EDGE_TIER,
    WEAK_CONNECTIVITY_TIER,
    build_role_transition_graph,
    fact_object_endpoint,
    fact_subject_endpoint,
)


LOCAL_TRAVERSAL_EDGE_KINDS = frozenset(
    {
        SENTENCE_GROUNDED_TRANSITION,
        ROLE_BRIDGE,
        SAME_SUBJECT,
        TITLE_ROLE_GROUNDING,
        SOURCE_ENDPOINT_INCIDENCE,
    }
)
LOCAL_EVIDENCE_EDGE_KINDS = frozenset(
    {
        *LOCAL_TRAVERSAL_EDGE_KINDS,
        SAME_OBJECT,
    }
)
LOCAL_SELECTION_EDGE_ORDER = {
    SENTENCE_GROUNDED_TRANSITION: 0,
    ROLE_BRIDGE: 1,
    TITLE_ROLE_GROUNDING: 2,
    SAME_SUBJECT: 3,
    SAME_OBJECT: 4,
    SOURCE_ENDPOINT_INCIDENCE: 5,
}
LOCAL_SOURCE_PRIOR_EDGE_ORDER = {
    SENTENCE_GROUNDED_TRANSITION: 0,
    TITLE_ROLE_GROUNDING: 1,
    SAME_SUBJECT: 2,
    ROLE_BRIDGE: 3,
    SOURCE_ENDPOINT_INCIDENCE: 4,
}
LOCAL_SOURCE_PRIOR_EDGE_KINDS = frozenset(LOCAL_SOURCE_PRIOR_EDGE_ORDER)
LOCAL_COUNTED_EDGE_KINDS = (
    SENTENCE_GROUNDED_TRANSITION,
    ROLE_BRIDGE,
    TITLE_ROLE_GROUNDING,
    SAME_SUBJECT,
    SAME_OBJECT,
    SOURCE_ENDPOINT_INCIDENCE,
)
LOCAL_EDGE_TIERS = (
    EVIDENCE_TRANSITION_TIER,
    WEAK_CONNECTIVITY_TIER,
    UNKNOWN_EDGE_TIER,
)


def _clean_doc_indices(values: Sequence[int] | None) -> List[int]:
    return unique_ranked([int(value) for value in values or []])


def _endpoint_seed_docs(
    *,
    endpoints: Sequence[str],
    endpoint_to_docs: Mapping[str, Sequence[int]],
) -> Dict[str, List[int]]:
    return {
        str(endpoint): _clean_doc_indices(endpoint_to_docs.get(str(endpoint), []) or [])
        for endpoint in endpoints
    }


def _edge_allowed_kinds(edge: Mapping[str, Any]) -> List[str]:
    return [str(kind) for kind in edge.get("kinds", []) or [] if str(kind) in LOCAL_EVIDENCE_EDGE_KINDS]


def _edge_traversal_kinds(edge: Mapping[str, Any]) -> List[str]:
    return [str(kind) for kind in edge.get("kinds", []) or [] if str(kind) in LOCAL_TRAVERSAL_EDGE_KINDS]


def _edge_variable_flow_traversal_kinds(
    edge: Mapping[str, Any],
    *,
    from_doc: int,
    doc_variable_flow_endpoints: Mapping[int, Set[str]],
) -> List[str]:
    """Allow same-object traversal only for an active variable endpoint."""

    if SAME_OBJECT not in set(str(kind) for kind in edge.get("kinds", []) or []):
        return []
    active = set(doc_variable_flow_endpoints.get(int(from_doc), set()) or set())
    if not active:
        return []
    edge_endpoints = {str(endpoint) for endpoint in edge.get("endpoints", []) or [] if str(endpoint).strip()}
    return [SAME_OBJECT] if active & edge_endpoints else []


def _unit_by_id(corpus_index: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    units: Dict[int, Mapping[str, Any]] = {}
    for unit in corpus_index.get("units", []) or []:
        if not isinstance(unit, Mapping):
            continue
        try:
            unit_id = int(unit.get("_unit_int_id"))
        except (TypeError, ValueError):
            continue
        units[unit_id] = unit
    return units


def _fact_units_by_doc(corpus_index: Mapping[str, Any]) -> Dict[int, List[Mapping[str, Any]]]:
    facts_by_doc: Dict[int, List[Mapping[str, Any]]] = {}
    for unit in corpus_index.get("units", []) or []:
        if not isinstance(unit, Mapping):
            continue
        if str(unit.get("unit_type") or "") != "openie_fact":
            continue
        try:
            doc_idx = int(unit.get("doc_index", -1))
        except (TypeError, ValueError):
            continue
        if doc_idx < 0:
            continue
        facts_by_doc.setdefault(doc_idx, []).append(unit)
    return facts_by_doc


def _doc_query_start_endpoints(
    *,
    doc_idx: int,
    query_endpoints: Sequence[str],
    corpus_index: Mapping[str, Any],
    role_graph: Mapping[str, Any],
) -> Set[str]:
    doc = int(doc_idx)
    query_set = {str(endpoint) for endpoint in query_endpoints if str(endpoint).strip()}
    if not query_set:
        return set()
    doc_endpoints = {
        str(endpoint)
        for endpoint in (corpus_index.get("doc_to_endpoints", {}) or {}).get(doc, []) or []
        if str(endpoint).strip()
    }
    doc_endpoints.update(
        str(endpoint)
        for endpoint in (role_graph.get("doc_to_role_endpoints", {}) or {}).get(doc, []) or []
        if str(endpoint).strip()
    )
    doc_title = str((role_graph.get("doc_titles", {}) or {}).get(doc, "") or "")
    if doc_title:
        doc_endpoints.add(doc_title)
    return query_set & doc_endpoints


def _expand_doc_variable_flow_endpoint_ranks(
    *,
    doc_idx: int,
    active_endpoint_ranks: Mapping[str, int],
    facts_by_doc: Mapping[int, Sequence[Mapping[str, Any]]],
    query_token_stems: Set[str],
) -> Dict[str, int]:
    """Close active variable endpoints through facts inside one document.

    Rank 0 means the endpoint is introduced by a query-supported fact relation;
    rank 1 means it is reachable only through generic document-local flow.  The
    rank is not a weighted score: it is a deterministic admissibility tier used
    to prevent weak variables from displacing stronger source-prior evidence.
    """

    ranks: Dict[str, int] = {
        str(endpoint): min(max(int(rank), 0), 1)
        for endpoint, rank in dict(active_endpoint_ranks or {}).items()
        if str(endpoint).strip()
    }
    if not ranks:
        return {}
    facts = list(facts_by_doc.get(int(doc_idx), []) or [])
    changed = True
    while changed:
        changed = False
        for unit in facts:
            subject = fact_subject_endpoint(unit)
            obj = fact_object_endpoint(unit)
            if not subject or not obj:
                continue
            relation_supported = bool(set(query_token_stems or set()) & _unit_relation_stems(unit))
            for source, target in ((subject, obj), (obj, subject)):
                if source not in ranks:
                    continue
                proposed_rank = 0 if ranks[source] == 0 and relation_supported else 1
                current_rank = ranks.get(target)
                if current_rank is None or proposed_rank < current_rank:
                    ranks[target] = proposed_rank
                    changed = True
    return ranks


def _expand_doc_variable_flow_endpoints(
    *,
    doc_idx: int,
    active_endpoints: Set[str],
    facts_by_doc: Mapping[int, Sequence[Mapping[str, Any]]],
) -> Set[str]:
    """Close active variable endpoints through facts inside one document.

    If a current variable endpoint participates in a fact, the other endpoint
    becomes an available next-hop variable. This is the document-local
    counterpart of graph traversal and does not depend on relation-name rules.
    """

    flow = {str(endpoint) for endpoint in active_endpoints if str(endpoint).strip()}
    if not flow:
        return set()
    facts = list(facts_by_doc.get(int(doc_idx), []) or [])
    changed = True
    while changed:
        changed = False
        for unit in facts:
            subject = fact_subject_endpoint(unit)
            obj = fact_object_endpoint(unit)
            if not subject or not obj:
                continue
            if subject in flow or obj in flow:
                before = len(flow)
                flow.add(subject)
                flow.add(obj)
                changed = changed or len(flow) > before
    return flow


def _doc_pair_key(left_doc: int, right_doc: int) -> str:
    left, right = sorted((int(left_doc), int(right_doc)))
    return f"{left}:{right}"


def _variable_flow_pair_keys(local_graph: Mapping[str, Any]) -> Set[str]:
    ranks = local_graph.get("variable_flow_doc_pair_ranks", {}) or {}
    if isinstance(ranks, Mapping) and ranks:
        output = set()
        for key, rank in ranks.items():
            try:
                clean_rank = int(rank)
            except (TypeError, ValueError):
                clean_rank = 1
            if clean_rank == 0:
                output.add(str(key))
        return output
    return {
        str(value)
        for value in local_graph.get("variable_flow_doc_pair_keys", []) or []
        if str(value).strip()
    }


def _unit_relation_stems(unit: Mapping[str, Any]) -> Set[str]:
    if str(unit.get("unit_type") or "") != "openie_fact":
        return set()
    return _token_stems(content_tokens(unit.get("relation", "")))


def _same_object_relation_stems_from_sample(
    sample: Mapping[str, Any],
    *,
    from_doc: int,
) -> Set[str]:
    stems: Set[str] = set()
    for doc_key, relation_key in (
        ("left_doc_index", "left_relation"),
        ("right_doc_index", "right_relation"),
    ):
        try:
            doc_index = int(sample.get(doc_key))
        except (TypeError, ValueError):
            continue
        if doc_index == int(from_doc):
            stems.update(_token_stems(content_tokens(sample.get(relation_key, ""))))
    return stems


def _same_object_query_relation_covered(
    edge: Mapping[str, Any],
    *,
    from_doc: int,
    unit_by_id: Mapping[int, Mapping[str, Any]] | None = None,
    query_token_stems: Set[str] | None = None,
) -> bool:
    if SAME_OBJECT not in set(str(kind) for kind in edge.get("kinds", []) or []):
        return False
    query_stems = set(query_token_stems or set())
    if not query_stems:
        return False
    for sample in edge.get("samples", []) or []:
        if not isinstance(sample, Mapping) or str(sample.get("kind") or "") != SAME_OBJECT:
            continue
        if query_stems & _same_object_relation_stems_from_sample(
            sample,
            from_doc=int(from_doc),
        ):
            return True
        for key in ("left_unit_id", "right_unit_id"):
            try:
                unit_id = int(sample.get(key))
            except (TypeError, ValueError):
                continue
            unit = (unit_by_id or {}).get(unit_id)
            if not isinstance(unit, Mapping):
                continue
            try:
                doc_index = int(unit.get("doc_index", -1))
            except (TypeError, ValueError):
                doc_index = -1
            if doc_index != int(from_doc):
                continue
            if query_stems & _unit_relation_stems(unit):
                return True
    return False


def _same_object_traversal_kinds(
    edge: Mapping[str, Any],
    *,
    from_doc: int,
    unit_by_id: Mapping[int, Mapping[str, Any]],
    query_token_stems: Set[str] | None = None,
) -> List[str]:
    """Allow object-variable handoff only when introduced by a query relation.

    A bare ``same_object`` edge is weak: many pages share generic objects.  It
    becomes a valid traversal edge when the current evidence page contributes a
    fact whose relation is lexically triggered by the query, because the shared
    object is then acting as the next-hop answer variable.
    """

    return (
        [SAME_OBJECT]
        if _same_object_query_relation_covered(
            edge,
            from_doc=int(from_doc),
            unit_by_id=unit_by_id,
            query_token_stems=query_token_stems,
        )
        else []
    )


def _edge_admission_traversal_kinds(
    edge: Mapping[str, Any],
    *,
    from_doc: int,
    unit_by_id: Mapping[int, Mapping[str, Any]],
    query_token_stems: Set[str] | None = None,
) -> List[str]:
    kinds = _edge_traversal_kinds(edge)
    if kinds:
        return kinds
    return _same_object_traversal_kinds(
        edge,
        from_doc=int(from_doc),
        unit_by_id=unit_by_id,
        query_token_stems=query_token_stems,
    )


def _edge_source_prior_kinds(
    edge: Mapping[str, Any],
    *,
    from_doc: int | None = None,
    query_token_stems: Set[str] | None = None,
    enable_query_supported_same_object_handoff: bool = False,
) -> List[str]:
    """Return evidence kinds allowed to promote reader-facing source prior.

    ``same_object`` is a valid local admission edge because it exposes shared
    answer-variable bridges.  A bare ``same_object`` is deliberately not a
    source-prior promotion edge: shared objects are weaker than sentence/title
    grounded transitions and can otherwise displace stable dense/source-prior
    reader evidence.  It is allowed here only when the source side relation is
    lexically triggered by the query, i.e. it is acting as a query-supported
    object handoff rather than generic co-reference.
    """

    kinds = [
        str(kind)
        for kind in edge.get("kinds", []) or []
        if str(kind) in LOCAL_SOURCE_PRIOR_EDGE_KINDS
    ]
    if (
        bool(enable_query_supported_same_object_handoff)
        and
        from_doc is not None
        and _same_object_query_relation_covered(
            edge,
            from_doc=int(from_doc),
            query_token_stems=query_token_stems,
        )
    ):
        kinds.append(SAME_OBJECT)
    return list(dict.fromkeys(kinds))


def _edge_neighbor(edge: Mapping[str, Any], doc_idx: int) -> int:
    left = int(edge.get("left_doc", -1))
    right = int(edge.get("right_doc", -1))
    if int(doc_idx) == left:
        return right
    if int(doc_idx) == right:
        return left
    return -1


def _edge_query_token_key(
    edge: Mapping[str, Any],
    *,
    from_doc: int,
    doc_query_token_coverage: Mapping[int, Sequence[str]] | None = None,
    query_token_stems: Set[str] | None = None,
) -> Tuple[int, int]:
    query_stems = set(query_token_stems or set())
    neighbor = _edge_neighbor(edge, from_doc)
    covered = (
        doc_query_token_coverage.get(int(neighbor), ())
        if doc_query_token_coverage and neighbor >= 0
        else ()
    )
    source_covered = (
        doc_query_token_coverage.get(int(from_doc), ())
        if doc_query_token_coverage and from_doc >= 0
        else ()
    )
    covered_set = {str(token) for token in covered or ()}
    source_covered_set = {str(token) for token in source_covered or ()}
    novel_covered = covered_set - source_covered_set
    relation_stems = _edge_relation_token_stems(edge)
    relation_covered = bool(query_stems & relation_stems)
    if relation_covered:
        evidence_rank = 0
    elif novel_covered:
        evidence_rank = 1
    else:
        evidence_rank = 2
    return (
        evidence_rank,
        -len(novel_covered),
    )


def _token_stem(token: str) -> str:
    text = str(token or "")
    ordinal_aliases = {
        "1st": "first",
        "2nd": "second",
        "3rd": "third",
        "4th": "fourth",
        "5th": "fifth",
        "6th": "sixth",
        "7th": "seventh",
        "8th": "eighth",
        "9th": "ninth",
        "10th": "tenth",
    }
    if text in ordinal_aliases:
        return ordinal_aliases[text]
    for suffix in ("ing", "ied", "ed", "ers", "er", "ors", "or", "s"):
        if len(text) > len(suffix) + 3 and text.endswith(suffix):
            if suffix == "ied":
                return f"{text[:-3]}y"
            return text[: -len(suffix)]
    return text


def _token_stems(tokens: Sequence[str] | Set[str]) -> Set[str]:
    return {_token_stem(str(token)) for token in tokens if str(token).strip()}


def _edge_relation_token_stems(edge: Mapping[str, Any]) -> Set[str]:
    stems: Set[str] = set()
    if SENTENCE_GROUNDED_TRANSITION not in set(str(kind) for kind in edge.get("kinds", []) or []):
        return stems
    for sample in edge.get("samples", []) or []:
        if not isinstance(sample, Mapping):
            continue
        if str(sample.get("kind") or "") != SENTENCE_GROUNDED_TRANSITION:
            continue
        stems.update(_token_stems(content_tokens(sample.get("relation", ""))))
    return stems


def _sort_edges(
    edge_rows: Sequence[Mapping[str, Any]],
    *,
    from_doc: int,
    doc_query_token_coverage: Mapping[int, Sequence[str]] | None = None,
    query_token_stems: Set[str] | None = None,
) -> List[Mapping[str, Any]]:
    allowed_edges = [edge for edge in edge_rows if _edge_allowed_kinds(edge)]
    return sorted(
        allowed_edges,
        key=lambda edge: (
            *_edge_query_token_key(
                edge,
                from_doc=int(from_doc),
                doc_query_token_coverage=doc_query_token_coverage,
                query_token_stems=query_token_stems,
            ),
            min((LOCAL_SELECTION_EDGE_ORDER.get(kind, 10) for kind in _edge_allowed_kinds(edge)), default=10),
            _edge_neighbor(edge, from_doc),
            int(edge.get("left_doc", -1)),
            int(edge.get("right_doc", -1)),
        ),
    )


def _sort_traversal_edges(
    edge_rows: Sequence[Mapping[str, Any]],
    *,
    from_doc: int,
    doc_query_token_coverage: Mapping[int, Sequence[str]] | None = None,
    query_token_stems: Set[str] | None = None,
) -> List[Mapping[str, Any]]:
    allowed_edges = [edge for edge in edge_rows if _edge_traversal_kinds(edge)]
    return sorted(
        allowed_edges,
        key=lambda edge: (
            *_edge_query_token_key(
                edge,
                from_doc=int(from_doc),
                doc_query_token_coverage=doc_query_token_coverage,
                query_token_stems=query_token_stems,
            ),
            min((LOCAL_SELECTION_EDGE_ORDER.get(kind, 10) for kind in _edge_traversal_kinds(edge)), default=10),
            _edge_neighbor(edge, from_doc),
            int(edge.get("left_doc", -1)),
            int(edge.get("right_doc", -1)),
        ),
    )


def _sort_admission_edges(
    edge_rows: Sequence[Mapping[str, Any]],
    *,
    from_doc: int,
    unit_by_id: Mapping[int, Mapping[str, Any]],
    doc_query_token_coverage: Mapping[int, Sequence[str]] | None = None,
    query_token_stems: Set[str] | None = None,
) -> List[Mapping[str, Any]]:
    allowed_edges = [
        edge
        for edge in edge_rows
        if _edge_admission_traversal_kinds(
            edge,
            from_doc=int(from_doc),
            unit_by_id=unit_by_id,
            query_token_stems=query_token_stems,
        )
    ]
    return sorted(
        allowed_edges,
        key=lambda edge: (
            *_edge_query_token_key(
                edge,
                from_doc=int(from_doc),
                doc_query_token_coverage=doc_query_token_coverage,
                query_token_stems=query_token_stems,
            ),
            min(
                (
                    LOCAL_SELECTION_EDGE_ORDER.get(kind, 10)
                    for kind in _edge_admission_traversal_kinds(
                        edge,
                        from_doc=int(from_doc),
                        unit_by_id=unit_by_id,
                        query_token_stems=query_token_stems,
                    )
                ),
                default=10,
            ),
            _edge_neighbor(edge, from_doc),
            int(edge.get("left_doc", -1)),
            int(edge.get("right_doc", -1)),
        ),
    )


def _sort_variable_flow_admission_edges(
    edge_rows: Sequence[Mapping[str, Any]],
    *,
    from_doc: int,
    doc_variable_flow_endpoints: Mapping[int, Set[str]],
    doc_query_token_coverage: Mapping[int, Sequence[str]] | None = None,
    query_token_stems: Set[str] | None = None,
) -> List[Mapping[str, Any]]:
    allowed_edges = []
    for edge in edge_rows:
        allowed_kinds = _edge_traversal_kinds(edge)
        if not allowed_kinds:
            allowed_kinds = _edge_variable_flow_traversal_kinds(
                edge,
                from_doc=int(from_doc),
                doc_variable_flow_endpoints=doc_variable_flow_endpoints,
            )
        if allowed_kinds:
            allowed_edges.append(edge)
    return sorted(
        allowed_edges,
        key=lambda edge: (
            *_edge_query_token_key(
                edge,
                from_doc=int(from_doc),
                doc_query_token_coverage=doc_query_token_coverage,
                query_token_stems=query_token_stems,
            ),
            min(
                (
                    LOCAL_SELECTION_EDGE_ORDER.get(kind, 10)
                    for kind in [
                        *_edge_traversal_kinds(edge),
                        *_edge_variable_flow_traversal_kinds(
                            edge,
                            from_doc=int(from_doc),
                            doc_variable_flow_endpoints=doc_variable_flow_endpoints,
                        ),
                    ]
                ),
                default=10,
            ),
            _edge_neighbor(edge, from_doc),
            int(edge.get("left_doc", -1)),
            int(edge.get("right_doc", -1)),
        ),
    )


def _sort_source_prior_edges(
    edge_rows: Sequence[Mapping[str, Any]],
    *,
    from_doc: int,
    required_kinds: Set[str] | None = None,
    doc_query_token_coverage: Mapping[int, Sequence[str]] | None = None,
    query_token_stems: Set[str] | None = None,
    enable_query_supported_same_object_handoff: bool = False,
    enable_variable_flow_traversal: bool = False,
    variable_flow_doc_pair_keys: Set[str] | None = None,
) -> List[Mapping[str, Any]]:
    required = {str(kind) for kind in required_kinds or set()}
    allowed_edges = []
    for edge in edge_rows:
        allowed_kinds = _edge_source_prior_kinds(
            edge,
            from_doc=int(from_doc),
            query_token_stems=query_token_stems,
            enable_query_supported_same_object_handoff=bool(enable_query_supported_same_object_handoff),
        )
        if (
            bool(enable_variable_flow_traversal)
            and SAME_OBJECT in set(str(kind) for kind in edge.get("kinds", []) or [])
            and _doc_pair_key(int(edge.get("left_doc", -1)), int(edge.get("right_doc", -1)))
            in set(variable_flow_doc_pair_keys or set())
        ):
            allowed_kinds = list(dict.fromkeys([*allowed_kinds, SAME_OBJECT]))
        if not allowed_kinds:
            continue
        if required and not (set(allowed_kinds) & required):
            continue
        allowed_edges.append(edge)
    return sorted(
        allowed_edges,
        key=lambda edge: (
            *_edge_query_token_key(
                edge,
                from_doc=int(from_doc),
                doc_query_token_coverage=doc_query_token_coverage,
                query_token_stems=query_token_stems,
            ),
            min(
                (
                    LOCAL_SOURCE_PRIOR_EDGE_ORDER.get(kind, LOCAL_SELECTION_EDGE_ORDER.get(kind, 10))
                    for kind in _edge_source_prior_kinds(
                        edge,
                        from_doc=int(from_doc),
                        query_token_stems=query_token_stems,
                        enable_query_supported_same_object_handoff=bool(enable_query_supported_same_object_handoff),
                    )
                    + (
                        [SAME_OBJECT]
                        if bool(enable_variable_flow_traversal)
                        and SAME_OBJECT in set(str(kind) for kind in edge.get("kinds", []) or [])
                        and _doc_pair_key(int(edge.get("left_doc", -1)), int(edge.get("right_doc", -1)))
                        in set(variable_flow_doc_pair_keys or set())
                        else []
                    )
                ),
                default=10,
            ),
            _edge_neighbor(edge, from_doc),
            int(edge.get("left_doc", -1)),
            int(edge.get("right_doc", -1)),
        ),
    )


def _sort_query_coverage_edges(
    edge_rows: Sequence[Mapping[str, Any]],
    *,
    from_doc: int,
    doc_query_coverage_key: Mapping[int, Tuple[int, int]],
    required_kinds: Set[str] | None = None,
) -> List[Mapping[str, Any]]:
    """Order local STO edges by query evidence contribution before edge kind.

    This keeps the source prior structural: identity/title witnesses are only
    promoted when their neighbor document actually covers query material;
    otherwise role-transition edges remain the default tie-breaker.
    """

    required = {str(kind) for kind in required_kinds or set()}
    allowed_edges = []
    for edge in edge_rows:
        allowed_kinds = _edge_allowed_kinds(edge)
        if not allowed_kinds:
            continue
        if required and not (set(allowed_kinds) & required):
            continue
        allowed_edges.append(edge)
    return sorted(
        allowed_edges,
        key=lambda edge: (
            *doc_query_coverage_key.get(_edge_neighbor(edge, from_doc), (0, 0)),
            min((LOCAL_SELECTION_EDGE_ORDER.get(kind, 10) for kind in _edge_allowed_kinds(edge)), default=10),
            _edge_neighbor(edge, from_doc),
            int(edge.get("left_doc", -1)),
            int(edge.get("right_doc", -1)),
        ),
    )


def _induced_local_edges(
    *,
    role_graph: Mapping[str, Any],
    admitted_docs: Sequence[int],
) -> List[Dict[str, Any]]:
    admitted = set(_clean_doc_indices(admitted_docs))
    edges: List[Dict[str, Any]] = []
    for edge in (role_graph.get("edges", {}) or {}).values():
        left = int(edge.get("left_doc", -1))
        right = int(edge.get("right_doc", -1))
        allowed_kinds = _edge_allowed_kinds(edge)
        if left not in admitted or right not in admitted or not allowed_kinds:
            continue
        edges.append(
            {
                "left_doc": left,
                "right_doc": right,
                "kinds": allowed_kinds,
                "evidence_tiers": list(edge.get("evidence_tiers", []) or []),
                "best_evidence_tier": str(edge.get("best_evidence_tier") or UNKNOWN_EDGE_TIER),
                "endpoints": list(edge.get("endpoints", []) or []),
                "samples": list(edge.get("samples", []) or [])[:5],
            }
        )
    return sorted(edges, key=lambda row: (int(row["left_doc"]), int(row["right_doc"]), tuple(row["kinds"])))


def _doc_query_endpoint_coverage(
    *,
    doc_idx: int,
    query_endpoints: Sequence[str],
    corpus_index: Mapping[str, Any],
) -> Set[str]:
    doc_endpoints = set(str(endpoint) for endpoint in corpus_index.get("doc_to_endpoints", {}).get(int(doc_idx), []) or [])
    return doc_endpoints & set(str(endpoint) for endpoint in query_endpoints)


def _doc_query_token_coverage(
    *,
    doc_idx: int,
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
) -> Set[str]:
    counts = corpus_index.get("doc_token_counts", {}).get(int(doc_idx), {}) or {}
    return set(counts.keys()) & query_tokens


def _local_edge_adjacency(local_edges: Sequence[Mapping[str, Any]]) -> Dict[int, List[Mapping[str, Any]]]:
    adjacency: Dict[int, List[Mapping[str, Any]]] = {}
    for edge in local_edges:
        left = int(edge.get("left_doc", -1))
        right = int(edge.get("right_doc", -1))
        if left < 0 or right < 0 or left == right:
            continue
        adjacency.setdefault(left, []).append(edge)
        adjacency.setdefault(right, []).append(edge)
    return adjacency


def _doc_query_token_coverage_map(
    *,
    corpus_index: Mapping[str, Any],
    query_tokens: Set[str],
) -> Dict[int, Tuple[str, ...]]:
    coverage: Dict[int, Tuple[str, ...]] = {}
    for doc_idx, counts in (corpus_index.get("doc_token_counts", {}) or {}).items():
        covered = set((counts or {}).keys()) & query_tokens
        if covered:
            coverage[int(doc_idx)] = tuple(sorted(str(token) for token in covered))
    return coverage


def _local_graph_query_token_coverage(
    local_graph: Mapping[str, Any],
) -> Dict[int, Tuple[str, ...]]:
    coverage: Dict[int, Tuple[str, ...]] = {}
    for raw_doc_idx, raw_tokens in (local_graph.get("doc_query_token_coverage", {}) or {}).items():
        try:
            doc_idx = int(raw_doc_idx)
        except (TypeError, ValueError):
            continue
        tokens = tuple(sorted(str(token) for token in raw_tokens or () if str(token).strip()))
        if tokens:
            coverage[doc_idx] = tokens
    return coverage


def _local_graph_query_token_stems(local_graph: Mapping[str, Any]) -> Set[str]:
    return {str(token) for token in local_graph.get("query_token_stems", []) or [] if str(token).strip()}


def _best_edge_kind_rank(edge: Mapping[str, Any]) -> int:
    kinds = [str(kind) for kind in edge.get("kinds", []) or []]
    return min((LOCAL_SELECTION_EDGE_ORDER.get(kind, 10) for kind in kinds), default=10)


def _edge_kind_count(edge: Mapping[str, Any], kind: str) -> int:
    return 1 if str(kind) in set(str(value) for value in edge.get("kinds", []) or []) else 0


def _trace_distance(local_graph: Mapping[str, Any], doc_idx: int) -> int:
    trace = (local_graph.get("doc_admission_trace", {}) or {}).get(str(int(doc_idx)), {}) or {}
    distance = trace.get("distance")
    if distance is None:
        return 10**9
    try:
        return int(distance)
    except (TypeError, ValueError):
        return 10**9


def _selected_components(
    *,
    doc_indices: Sequence[int],
    adjacency: Mapping[int, Sequence[Mapping[str, Any]]],
) -> int:
    docs = set(_clean_doc_indices(doc_indices))
    if not docs:
        return 0
    remaining = set(docs)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            for edge in adjacency.get(int(current), []) or []:
                neighbor = _edge_neighbor(edge, int(current))
                if neighbor in remaining and neighbor in docs:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def _internal_edge_counts(
    *,
    doc_indices: Sequence[int],
    local_edges: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    docs = set(_clean_doc_indices(doc_indices))
    counts = {kind: 0 for kind in LOCAL_COUNTED_EDGE_KINDS}
    if len(docs) < 2:
        return counts
    for edge in local_edges:
        left = int(edge.get("left_doc", -1))
        right = int(edge.get("right_doc", -1))
        if left not in docs or right not in docs:
            continue
        for kind in LOCAL_COUNTED_EDGE_KINDS:
            counts[kind] += _edge_kind_count(edge, kind)
    return counts


def _edge_tier_count(edge: Mapping[str, Any], tier: str) -> int:
    return 1 if str(edge.get("best_evidence_tier") or UNKNOWN_EDGE_TIER) == str(tier) else 0


def _local_edge_tier_counts(local_edges: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {tier: 0 for tier in LOCAL_EDGE_TIERS}
    for edge in local_edges:
        tier = str(edge.get("best_evidence_tier") or UNKNOWN_EDGE_TIER)
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def _internal_edge_tier_counts(
    *,
    doc_indices: Sequence[int],
    local_edges: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    docs = set(_clean_doc_indices(doc_indices))
    counts = {tier: 0 for tier in LOCAL_EDGE_TIERS}
    if len(docs) < 2:
        return counts
    for edge in local_edges:
        left = int(edge.get("left_doc", -1))
        right = int(edge.get("right_doc", -1))
        if left not in docs or right not in docs:
            continue
        for tier in LOCAL_EDGE_TIERS:
            counts[tier] += _edge_tier_count(edge, tier)
    return counts


def _selection_feature_cache(
    *,
    admitted_docs: Sequence[int],
    query_endpoints: Sequence[str],
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
    local_graph: Mapping[str, Any],
    local_edges: Sequence[Mapping[str, Any]],
    bm25_scores: Mapping[int, float],
) -> Dict[str, Any]:
    admitted = _clean_doc_indices(admitted_docs)
    doc_features = {
        int(doc_idx): {
            "endpoints": _doc_query_endpoint_coverage(
                doc_idx=int(doc_idx),
                query_endpoints=query_endpoints,
                corpus_index=corpus_index,
            ),
            "tokens": _doc_query_token_coverage(
                doc_idx=int(doc_idx),
                query_tokens=query_tokens,
                corpus_index=corpus_index,
            ),
            "distance": _trace_distance(local_graph, int(doc_idx)),
            "rank": rank,
            "bm25": float(bm25_scores.get(int(doc_idx), 0.0) or 0.0),
        }
        for rank, doc_idx in enumerate(admitted)
    }
    pair_edge_counts: Dict[Tuple[int, int], Dict[str, int]] = {}
    connected_pairs: Set[Tuple[int, int]] = set()
    connected_neighbors: Dict[int, Set[int]] = {}
    for edge in local_edges:
        left = int(edge.get("left_doc", -1))
        right = int(edge.get("right_doc", -1))
        if left < 0 or right < 0 or left == right:
            continue
        key = tuple(sorted((left, right)))
        connected_pairs.add(key)
        connected_neighbors.setdefault(left, set()).add(right)
        connected_neighbors.setdefault(right, set()).add(left)
        counts = pair_edge_counts.setdefault(key, {kind: 0 for kind in LOCAL_COUNTED_EDGE_KINDS})
        for kind in LOCAL_COUNTED_EDGE_KINDS:
            counts[kind] += _edge_kind_count(edge, kind)
    return {
        "doc_features": doc_features,
        "pair_edge_counts": pair_edge_counts,
        "connected_pairs": connected_pairs,
        "connected_neighbors": connected_neighbors,
        "query_endpoint_set": set(str(endpoint) for endpoint in query_endpoints),
    }


def _selected_components_from_pairs(
    *,
    doc_indices: Sequence[int],
    connected_neighbors: Mapping[int, Set[int]],
) -> int:
    docs = set(_clean_doc_indices(doc_indices))
    if not docs:
        return 0
    remaining = set(docs)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            neighbors = set(connected_neighbors.get(int(current), set()) or set())
            for neighbor in sorted(neighbors & remaining):
                remaining.remove(neighbor)
                stack.append(neighbor)
    return components


def _set_covered_query_endpoints(
    *,
    doc_indices: Sequence[int],
    query_endpoints: Sequence[str],
    corpus_index: Mapping[str, Any],
) -> Set[str]:
    covered: Set[str] = set()
    for doc_idx in _clean_doc_indices(doc_indices):
        covered.update(
            _doc_query_endpoint_coverage(
                doc_idx=doc_idx,
                query_endpoints=query_endpoints,
                corpus_index=corpus_index,
            )
        )
    return covered


def _set_covered_query_tokens(
    *,
    doc_indices: Sequence[int],
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
) -> Set[str]:
    covered: Set[str] = set()
    for doc_idx in _clean_doc_indices(doc_indices):
        covered.update(
            _doc_query_token_coverage(
                doc_idx=doc_idx,
                query_tokens=query_tokens,
                corpus_index=corpus_index,
            )
        )
    return covered


def _evidence_set_key(
    *,
    doc_indices: Sequence[int],
    feature_cache: Mapping[str, Any],
) -> Tuple[Any, ...]:
    docs = _clean_doc_indices(doc_indices)
    doc_features: Mapping[int, Mapping[str, Any]] = feature_cache.get("doc_features", {}) or {}
    pair_edge_counts: Mapping[Tuple[int, int], Mapping[str, int]] = feature_cache.get("pair_edge_counts", {}) or {}
    connected_neighbors: Mapping[int, Set[int]] = feature_cache.get("connected_neighbors", {}) or {}
    query_endpoint_set: Set[str] = set(feature_cache.get("query_endpoint_set", set()) or set())
    covered_endpoints: Set[str] = set()
    covered_tokens: Set[str] = set()
    edge_counts = {kind: 0 for kind in LOCAL_COUNTED_EDGE_KINDS}
    for doc_idx in docs:
        features = doc_features.get(int(doc_idx), {}) or {}
        covered_endpoints.update(str(endpoint) for endpoint in features.get("endpoints", set()) or set())
        covered_tokens.update(str(token) for token in features.get("tokens", set()) or set())
    for left, right in combinations(sorted(docs), 2):
        counts = pair_edge_counts.get(tuple(sorted((int(left), int(right)))), {}) or {}
        for kind in LOCAL_COUNTED_EDGE_KINDS:
            edge_counts[kind] += int(counts.get(kind, 0) or 0)
    distances = [int((doc_features.get(int(doc_idx), {}) or {}).get("distance", 10**9) or 10**9) for doc_idx in docs]
    return (
        -len(covered_endpoints),
        len(query_endpoint_set - covered_endpoints),
        -len(covered_tokens),
        -int(edge_counts.get(SENTENCE_GROUNDED_TRANSITION, 0) or 0),
        -int(edge_counts.get(ROLE_BRIDGE, 0) or 0),
        _selected_components_from_pairs(doc_indices=docs, connected_neighbors=connected_neighbors),
        len(docs),
        -int(edge_counts.get(TITLE_ROLE_GROUNDING, 0) or 0),
        -int(edge_counts.get(SAME_SUBJECT, 0) or 0),
        -int(edge_counts.get(SOURCE_ENDPOINT_INCIDENCE, 0) or 0),
        max(distances) if distances else 10**9,
        sum(distances),
        sum(int((doc_features.get(int(doc_idx), {}) or {}).get("rank", 10**9) or 10**9) for doc_idx in docs),
        -sum(float((doc_features.get(int(doc_idx), {}) or {}).get("bm25", 0.0) or 0.0) for doc_idx in docs),
        tuple(sorted(docs)),
    )


def _search_local_evidence_set(
    *,
    admitted_docs: Sequence[int],
    query_endpoints: Sequence[str],
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
    local_graph: Mapping[str, Any],
    local_edges: Sequence[Mapping[str, Any]],
    adjacency: Mapping[int, Sequence[Mapping[str, Any]]],
    bm25_scores: Mapping[int, float],
    evidence_set_size: int,
) -> Tuple[int, ...]:
    admitted = _clean_doc_indices(admitted_docs)
    if not admitted:
        return tuple()
    target_size = min(max(int(evidence_set_size), 1), len(admitted))
    frontier_limit = max(128, target_size * 32)
    feature_cache = _selection_feature_cache(
        admitted_docs=admitted,
        query_endpoints=query_endpoints,
        query_tokens=query_tokens,
        corpus_index=corpus_index,
        local_graph=local_graph,
        local_edges=local_edges,
        bm25_scores=bm25_scores,
    )

    def score(doc_set: Sequence[int]) -> Tuple[Any, ...]:
        return _evidence_set_key(
            doc_indices=doc_set,
            feature_cache=feature_cache,
        )

    def expansion_candidates(doc_set: Sequence[int]) -> List[int]:
        existing = set(_clean_doc_indices(doc_set))
        doc_features: Mapping[int, Mapping[str, Any]] = feature_cache.get("doc_features", {}) or {}
        connected_neighbors: Mapping[int, Set[int]] = feature_cache.get("connected_neighbors", {}) or {}
        covered_endpoints: Set[str] = set()
        covered_tokens: Set[str] = set()
        graph_neighbors: Set[int] = set()
        for doc_idx in existing:
            features = doc_features.get(int(doc_idx), {}) or {}
            covered_endpoints.update(str(endpoint) for endpoint in features.get("endpoints", set()) or set())
            covered_tokens.update(str(token) for token in features.get("tokens", set()) or set())
            graph_neighbors.update(int(neighbor) for neighbor in connected_neighbors.get(int(doc_idx), set()) or set())
        candidates: List[int] = []
        for doc_idx in admitted:
            doc = int(doc_idx)
            if doc in existing:
                continue
            features = doc_features.get(doc, {}) or {}
            adds_endpoint = bool(set(features.get("endpoints", set()) or set()) - covered_endpoints)
            adds_token = bool(set(features.get("tokens", set()) or set()) - covered_tokens)
            if adds_endpoint or adds_token or doc in graph_neighbors:
                candidates.append(doc)
        return candidates

    current: Dict[Tuple[int, ...], Tuple[Any, ...]] = {
        (int(doc_idx),): score((int(doc_idx),))
        for doc_idx in admitted
    }
    current = dict(sorted(current.items(), key=lambda item: item[1])[:frontier_limit])
    best_by_size: Dict[int, Dict[Tuple[int, ...], Tuple[Any, ...]]] = {1: current}

    for size in range(2, target_size + 1):
        next_sets: Dict[Tuple[int, ...], Tuple[Any, ...]] = {}
        for doc_set in current:
            existing = set(doc_set)
            for doc_idx in expansion_candidates(doc_set):
                if int(doc_idx) in existing:
                    continue
                candidate = tuple(sorted((*doc_set, int(doc_idx))))
                if candidate in next_sets:
                    continue
                next_sets[candidate] = score(candidate)
        if not next_sets:
            break
        current = dict(sorted(next_sets.items(), key=lambda item: item[1])[:frontier_limit])
        best_by_size[size] = current

    best_sets: Dict[Tuple[int, ...], Tuple[Any, ...]] = {}
    for sets_for_size in best_by_size.values():
        best_sets.update(sets_for_size)
    return min(best_sets.items(), key=lambda item: item[1])[0]


def _complete_local_evidence_set(
    *,
    core_doc_indices: Sequence[int],
    admitted_docs: Sequence[int],
    query_endpoints: Sequence[str],
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
    local_graph: Mapping[str, Any],
    local_edges: Sequence[Mapping[str, Any]],
    adjacency: Mapping[int, Sequence[Mapping[str, Any]]],
    bm25_scores: Mapping[int, float],
    evidence_set_size: int,
) -> Tuple[int, ...]:
    selected = _clean_doc_indices(core_doc_indices)
    admitted = _clean_doc_indices(admitted_docs)
    target_size = min(max(int(evidence_set_size), 1), len(admitted))
    selected_set = set(selected)
    feature_cache = _selection_feature_cache(
        admitted_docs=admitted,
        query_endpoints=query_endpoints,
        query_tokens=query_tokens,
        corpus_index=corpus_index,
        local_graph=local_graph,
        local_edges=local_edges,
        bm25_scores=bm25_scores,
    )
    while len(selected) < target_size:
        candidates = [int(doc_idx) for doc_idx in admitted if int(doc_idx) not in selected_set]
        if not candidates:
            break
        best_doc = min(
            candidates,
            key=lambda doc_idx: _evidence_set_key(
                doc_indices=[*selected, int(doc_idx)],
                feature_cache=feature_cache,
            ),
        )
        selected.append(int(best_doc))
        selected_set.add(int(best_doc))
    return tuple(selected)


def order_local_sto_source_prior(
    *,
    local_graph: Mapping[str, Any],
    max_docs: int | None = None,
) -> List[int]:
    """Order admitted docs by query-rooted STO frontier traversal.

    This is the source-prior counterpart to local STO admission. It fixes the
    seed-flood behavior where all textual seeds are emitted before any graph
    neighbor. The order is deterministic and lexicographic: for each query root,
    emit the root and then its role-safe STO frontier before moving to the next
    root.
    """

    admitted_docs = _clean_doc_indices(local_graph.get("admitted_doc_indices", []) or [])
    if not admitted_docs:
        return []
    admitted_set = set(admitted_docs)
    seeds = _clean_doc_indices(local_graph.get("seed_doc_indices", []) or [])
    if not seeds:
        seeds = _clean_doc_indices(
            [
                *(local_graph.get("textual_seed_doc_indices", []) or []),
                *(local_graph.get("symbolic_seed_doc_indices", []) or []),
            ]
        )
    clean_max = len(admitted_docs) if max_docs is None else max(int(max_docs), 0)
    if clean_max <= 0:
        return []

    stats = local_graph.get("stats", {}) or {}
    try:
        closure_hops = int(stats.get("closure_hops", 0) or 0)
    except (TypeError, ValueError):
        closure_hops = 0
    enable_object_handoff = bool(stats.get("query_supported_same_object_handoff_enabled", False))
    enable_variable_flow = bool(stats.get("variable_flow_traversal_enabled", False))
    variable_flow_doc_pair_keys = _variable_flow_pair_keys(local_graph)
    adjacency = _local_edge_adjacency(list(local_graph.get("local_edges", []) or []))
    doc_query_token_coverage = _local_graph_query_token_coverage(local_graph)
    query_token_stems = _local_graph_query_token_stems(local_graph)
    ordered: List[int] = []
    seen: Set[int] = set()

    def add(doc_idx: int) -> bool:
        doc = int(doc_idx)
        if doc not in admitted_set or doc in seen or len(ordered) >= clean_max:
            return False
        seen.add(doc)
        ordered.append(doc)
        return True

    for seed in seeds:
        if len(ordered) >= clean_max:
            break
        if int(seed) not in admitted_set:
            continue
        add(int(seed))
        visited = {int(seed)}
        queue: Deque[Tuple[int, int]] = deque([(int(seed), 0)])
        while queue and len(ordered) < clean_max:
            current_doc, distance = queue.popleft()
            if distance >= closure_hops:
                continue
            for edge in _sort_source_prior_edges(
                adjacency.get(int(current_doc), []) or [],
                from_doc=int(current_doc),
                doc_query_token_coverage=doc_query_token_coverage,
                query_token_stems=query_token_stems,
                enable_query_supported_same_object_handoff=enable_object_handoff,
                enable_variable_flow_traversal=enable_variable_flow,
                variable_flow_doc_pair_keys=variable_flow_doc_pair_keys,
            ):
                neighbor = _edge_neighbor(edge, int(current_doc))
                if neighbor < 0 or neighbor not in admitted_set or neighbor in visited:
                    continue
                visited.add(neighbor)
                add(neighbor)
                queue.append((neighbor, int(distance + 1)))
                if len(ordered) >= clean_max:
                    break

    for doc_idx in admitted_docs:
        if len(ordered) >= clean_max:
            break
        add(int(doc_idx))
    return ordered


def _seed_frontier_order(
    *,
    seed: int,
    admitted_set: Set[int],
    adjacency: Mapping[int, Sequence[Mapping[str, Any]]],
    closure_hops: int,
    required_kinds: Set[str] | None = None,
    doc_query_token_coverage: Mapping[int, Sequence[str]] | None = None,
    query_token_stems: Set[str] | None = None,
    enable_query_supported_same_object_handoff: bool = False,
    enable_variable_flow_traversal: bool = False,
    variable_flow_doc_pair_keys: Set[str] | None = None,
) -> List[int]:
    ordered: List[int] = []
    visited = {int(seed)}
    queue: Deque[Tuple[int, int]] = deque([(int(seed), 0)])
    while queue:
        current_doc, distance = queue.popleft()
        if distance >= closure_hops:
            continue
        edge_rows = (
            _sort_traversal_edges(
                adjacency.get(int(current_doc), []) or [],
                from_doc=int(current_doc),
                doc_query_token_coverage=doc_query_token_coverage,
                query_token_stems=query_token_stems,
            )
            if required_kinds is None
            else _sort_source_prior_edges(
                adjacency.get(int(current_doc), []) or [],
                from_doc=int(current_doc),
                required_kinds=required_kinds,
                doc_query_token_coverage=doc_query_token_coverage,
                query_token_stems=query_token_stems,
                enable_query_supported_same_object_handoff=bool(enable_query_supported_same_object_handoff),
                enable_variable_flow_traversal=bool(enable_variable_flow_traversal),
                variable_flow_doc_pair_keys=set(variable_flow_doc_pair_keys or set()),
            )
        )
        for edge in edge_rows:
            neighbor = _edge_neighbor(edge, int(current_doc))
            if neighbor < 0 or neighbor not in admitted_set or neighbor in visited:
                continue
            visited.add(neighbor)
            ordered.append(int(neighbor))
            queue.append((int(neighbor), int(distance + 1)))
    return ordered


def _seed_query_coverage_frontier_order(
    *,
    seed: int,
    admitted_set: Set[int],
    adjacency: Mapping[int, Sequence[Mapping[str, Any]]],
    closure_hops: int,
    doc_query_coverage_key: Mapping[int, Tuple[int, int]],
) -> List[int]:
    ordered: List[int] = []
    visited = {int(seed)}
    queue: Deque[Tuple[int, int]] = deque([(int(seed), 0)])
    while queue:
        current_doc, distance = queue.popleft()
        if distance >= closure_hops:
            continue
        for edge in _sort_query_coverage_edges(
            adjacency.get(int(current_doc), []) or [],
            from_doc=int(current_doc),
            doc_query_coverage_key=doc_query_coverage_key,
        ):
            neighbor = _edge_neighbor(edge, int(current_doc))
            if neighbor < 0 or neighbor not in admitted_set or neighbor in visited:
                continue
            visited.add(neighbor)
            ordered.append(int(neighbor))
            queue.append((int(neighbor), int(distance + 1)))
    return ordered


def order_local_sto_balanced_source_prior(
    *,
    local_graph: Mapping[str, Any],
    max_docs: int | None = None,
) -> List[int]:
    """Order admitted docs by root-balanced STO frontier traversal.

    Multi-hop questions often contain multiple query roots. A source prior that
    fully expands the first root before visiting the next root can recover a
    transition document while losing the second root's own evidence. This order
    emits each root with one nearest STO transition witness per round, then
    continues round-robin over the remaining frontier.
    """

    admitted_docs = _clean_doc_indices(local_graph.get("admitted_doc_indices", []) or [])
    if not admitted_docs:
        return []
    admitted_set = set(admitted_docs)
    seeds = _clean_doc_indices(local_graph.get("seed_doc_indices", []) or [])
    if not seeds:
        seeds = _clean_doc_indices(
            [
                *(local_graph.get("textual_seed_doc_indices", []) or []),
                *(local_graph.get("symbolic_seed_doc_indices", []) or []),
            ]
        )
    seeds = [int(seed) for seed in seeds if int(seed) in admitted_set]
    if not seeds:
        seeds = admitted_docs[:]
    clean_max = len(admitted_docs) if max_docs is None else max(int(max_docs), 0)
    if clean_max <= 0:
        return []

    stats = local_graph.get("stats", {}) or {}
    try:
        closure_hops = int(stats.get("closure_hops", 0) or 0)
    except (TypeError, ValueError):
        closure_hops = 0
    adjacency = _local_edge_adjacency(list(local_graph.get("local_edges", []) or []))
    doc_query_token_coverage = _local_graph_query_token_coverage(local_graph)
    query_token_stems = _local_graph_query_token_stems(local_graph)
    frontiers = {
        int(seed): _seed_frontier_order(
            seed=int(seed),
            admitted_set=admitted_set,
            adjacency=adjacency,
            closure_hops=closure_hops,
            doc_query_token_coverage=doc_query_token_coverage,
            query_token_stems=query_token_stems,
        )
        for seed in seeds
    }
    frontier_positions = {int(seed): 0 for seed in seeds}
    ordered: List[int] = []
    seen: Set[int] = set()

    def add(doc_idx: int) -> bool:
        doc = int(doc_idx)
        if doc not in admitted_set or doc in seen or len(ordered) >= clean_max:
            return False
        seen.add(doc)
        ordered.append(doc)
        return True

    def add_next_frontier(seed: int) -> bool:
        frontier = frontiers.get(int(seed), []) or []
        pos = int(frontier_positions.get(int(seed), 0) or 0)
        while pos < len(frontier):
            candidate = int(frontier[pos])
            pos += 1
            frontier_positions[int(seed)] = pos
            if add(candidate):
                return True
        frontier_positions[int(seed)] = pos
        return False

    for seed in seeds:
        if len(ordered) >= clean_max:
            break
        add(int(seed))
        add_next_frontier(int(seed))

    progressed = True
    while progressed and len(ordered) < clean_max:
        progressed = False
        for seed in seeds:
            if len(ordered) >= clean_max:
                break
            progressed = add_next_frontier(int(seed)) or progressed

    for doc_idx in admitted_docs:
        if len(ordered) >= clean_max:
            break
        add(int(doc_idx))
    return ordered


def order_local_sto_root_preserving_source_prior(
    *,
    local_graph: Mapping[str, Any],
    max_docs: int | None = None,
) -> List[int]:
    """Order source prior by primary transition plus root preservation.

    The top-5 evidence budget cannot expand every retriever entry root before
    preserving later roots. This order keeps the first root's nearest STO
    transition witness, then admits the remaining retriever roots before
    continuing balanced frontier expansion.
    """

    admitted_docs = _clean_doc_indices(local_graph.get("admitted_doc_indices", []) or [])
    if not admitted_docs:
        return []
    admitted_set = set(admitted_docs)
    seeds = _clean_doc_indices(local_graph.get("seed_doc_indices", []) or [])
    if not seeds:
        seeds = _clean_doc_indices(
            [
                *(local_graph.get("textual_seed_doc_indices", []) or []),
                *(local_graph.get("symbolic_seed_doc_indices", []) or []),
            ]
        )
    seeds = [int(seed) for seed in seeds if int(seed) in admitted_set]
    if not seeds:
        seeds = admitted_docs[:]
    clean_max = len(admitted_docs) if max_docs is None else max(int(max_docs), 0)
    if clean_max <= 0:
        return []

    stats = local_graph.get("stats", {}) or {}
    try:
        closure_hops = int(stats.get("closure_hops", 0) or 0)
    except (TypeError, ValueError):
        closure_hops = 0
    adjacency = _local_edge_adjacency(list(local_graph.get("local_edges", []) or []))
    doc_query_token_coverage = _local_graph_query_token_coverage(local_graph)
    query_token_stems = _local_graph_query_token_stems(local_graph)
    frontiers = {
        int(seed): _seed_frontier_order(
            seed=int(seed),
            admitted_set=admitted_set,
            adjacency=adjacency,
            closure_hops=closure_hops,
            doc_query_token_coverage=doc_query_token_coverage,
            query_token_stems=query_token_stems,
        )
        for seed in seeds
    }
    frontier_positions = {int(seed): 0 for seed in seeds}
    evidence_budget = min(5, clean_max)
    ordered: List[int] = []
    seen: Set[int] = set()

    def add(doc_idx: int) -> bool:
        doc = int(doc_idx)
        if doc not in admitted_set or doc in seen or len(ordered) >= clean_max:
            return False
        seen.add(doc)
        ordered.append(doc)
        return True

    def add_next_frontier(seed: int) -> bool:
        frontier = frontiers.get(int(seed), []) or []
        pos = int(frontier_positions.get(int(seed), 0) or 0)
        while pos < len(frontier):
            candidate = int(frontier[pos])
            pos += 1
            frontier_positions[int(seed)] = pos
            if add(candidate):
                return True
        frontier_positions[int(seed)] = pos
        return False

    primary_seed = int(seeds[0])
    add(primary_seed)
    if len(ordered) < evidence_budget:
        add_next_frontier(primary_seed)

    for seed in seeds[1:]:
        if len(ordered) >= evidence_budget:
            break
        add(int(seed))

    progressed = True
    while progressed and len(ordered) < clean_max:
        progressed = False
        for seed in seeds:
            if len(ordered) >= clean_max:
                break
            progressed = add_next_frontier(int(seed)) or progressed
        for seed in seeds:
            if len(ordered) >= clean_max:
                break
            progressed = add(int(seed)) or progressed

    for doc_idx in admitted_docs:
        if len(ordered) >= clean_max:
            break
        add(int(doc_idx))
    return ordered


def order_local_sto_admission_preserving_source_prior(
    *,
    local_graph: Mapping[str, Any],
    max_docs: int | None = None,
) -> List[int]:
    """Order source prior by admission order plus the primary STO witness.

    Online STO admission already encodes the base retriever entry order. The
    only graph witness that must be promoted before preserving that order is the
    nearest STO transition from the primary root.
    """

    admitted_docs = _clean_doc_indices(local_graph.get("admitted_doc_indices", []) or [])
    if not admitted_docs:
        return []
    admitted_set = set(admitted_docs)
    clean_max = len(admitted_docs) if max_docs is None else max(int(max_docs), 0)
    if clean_max <= 0:
        return []

    seeds = _clean_doc_indices(local_graph.get("seed_doc_indices", []) or [])
    if not seeds:
        seeds = _clean_doc_indices(
            [
                *(local_graph.get("textual_seed_doc_indices", []) or []),
                *(local_graph.get("symbolic_seed_doc_indices", []) or []),
            ]
        )
    primary_seed = next((int(seed) for seed in seeds if int(seed) in admitted_set), int(admitted_docs[0]))
    stats = local_graph.get("stats", {}) or {}
    try:
        closure_hops = int(stats.get("closure_hops", 0) or 0)
    except (TypeError, ValueError):
        closure_hops = 0
    enable_object_handoff = bool(stats.get("query_supported_same_object_handoff_enabled", False))
    enable_variable_flow = bool(stats.get("variable_flow_traversal_enabled", False))
    variable_flow_doc_pair_keys = _variable_flow_pair_keys(local_graph)
    adjacency = _local_edge_adjacency(list(local_graph.get("local_edges", []) or []))
    doc_query_token_coverage = _local_graph_query_token_coverage(local_graph)
    query_token_stems = _local_graph_query_token_stems(local_graph)
    primary_frontier = _seed_frontier_order(
        seed=primary_seed,
        admitted_set=admitted_set,
        adjacency=adjacency,
        closure_hops=closure_hops,
        required_kinds=(
            {*LOCAL_SOURCE_PRIOR_EDGE_KINDS, SAME_OBJECT}
            if enable_object_handoff or enable_variable_flow
            else set(LOCAL_SOURCE_PRIOR_EDGE_KINDS)
        ),
        doc_query_token_coverage=doc_query_token_coverage,
        query_token_stems=query_token_stems,
        enable_query_supported_same_object_handoff=enable_object_handoff,
        enable_variable_flow_traversal=enable_variable_flow,
        variable_flow_doc_pair_keys=variable_flow_doc_pair_keys,
    )

    ordered: List[int] = []
    seen: Set[int] = set()

    def add(doc_idx: int) -> bool:
        doc = int(doc_idx)
        if doc not in admitted_set or doc in seen or len(ordered) >= clean_max:
            return False
        seen.add(doc)
        ordered.append(doc)
        return True

    add(primary_seed)
    if primary_frontier:
        add(int(primary_frontier[0]))
    for doc_idx in admitted_docs:
        if len(ordered) >= clean_max:
            break
        add(int(doc_idx))
    return ordered


def order_local_sto_coverage_source_prior(
    *,
    local_graph: Mapping[str, Any],
    corpus_index: Mapping[str, Any],
    query: str,
    max_docs: int | None = None,
) -> List[int]:
    """Order admitted docs by root-balanced, query-coverage STO frontier.

    The policy keeps the balanced-frontier skeleton, but orders each root's
    frontier by the neighbor document's direct query evidence contribution
    before edge kind. It is a closed-form graph/source prior, not a weighted
    channel fusion or dataset-specific switch.
    """

    admitted_docs = _clean_doc_indices(local_graph.get("admitted_doc_indices", []) or [])
    if not admitted_docs:
        return []
    admitted_set = set(admitted_docs)
    seeds = _clean_doc_indices(local_graph.get("seed_doc_indices", []) or [])
    if not seeds:
        seeds = _clean_doc_indices(
            [
                *(local_graph.get("textual_seed_doc_indices", []) or []),
                *(local_graph.get("symbolic_seed_doc_indices", []) or []),
            ]
        )
    seeds = [int(seed) for seed in seeds if int(seed) in admitted_set]
    if not seeds:
        seeds = admitted_docs[:]
    clean_max = len(admitted_docs) if max_docs is None else max(int(max_docs), 0)
    if clean_max <= 0:
        return []

    stats = local_graph.get("stats", {}) or {}
    try:
        closure_hops = int(stats.get("closure_hops", 0) or 0)
    except (TypeError, ValueError):
        closure_hops = 0
    query_tokens = content_tokens(query)
    query_token_stems = _token_stems(query_tokens)
    query_endpoints = [str(endpoint) for endpoint in local_graph.get("symbolic_anchor_endpoints", []) or []]
    doc_query_coverage_key = {
        int(doc_idx): (
            -len(
                _doc_query_endpoint_coverage(
                    doc_idx=int(doc_idx),
                    query_endpoints=query_endpoints,
                    corpus_index=corpus_index,
                )
            ),
            -len(
                _doc_query_token_coverage(
                    doc_idx=int(doc_idx),
                    query_tokens=query_tokens,
                    corpus_index=corpus_index,
                )
            ),
        )
        for doc_idx in admitted_docs
    }
    adjacency = _local_edge_adjacency(list(local_graph.get("local_edges", []) or []))
    frontiers = {
        int(seed): _seed_query_coverage_frontier_order(
            seed=int(seed),
            admitted_set=admitted_set,
            adjacency=adjacency,
            closure_hops=closure_hops,
            doc_query_coverage_key=doc_query_coverage_key,
        )
        for seed in seeds
    }
    frontier_positions = {int(seed): 0 for seed in seeds}
    ordered: List[int] = []
    seen: Set[int] = set()

    def add(doc_idx: int) -> bool:
        doc = int(doc_idx)
        if doc not in admitted_set or doc in seen or len(ordered) >= clean_max:
            return False
        seen.add(doc)
        ordered.append(doc)
        return True

    def add_next_frontier(seed: int) -> bool:
        frontier = frontiers.get(int(seed), []) or []
        pos = int(frontier_positions.get(int(seed), 0) or 0)
        while pos < len(frontier):
            candidate = int(frontier[pos])
            pos += 1
            frontier_positions[int(seed)] = pos
            if add(candidate):
                return True
        frontier_positions[int(seed)] = pos
        return False

    for seed in seeds:
        if len(ordered) >= clean_max:
            break
        add(int(seed))
        add_next_frontier(int(seed))

    progressed = True
    while progressed and len(ordered) < clean_max:
        progressed = False
        for seed in seeds:
            if len(ordered) >= clean_max:
                break
            progressed = add_next_frontier(int(seed)) or progressed

    for doc_idx in admitted_docs:
        if len(ordered) >= clean_max:
            break
        add(int(doc_idx))
    return ordered


def order_local_sto_dual_source_prior(
    *,
    local_graph: Mapping[str, Any],
    max_docs: int | None = None,
) -> List[int]:
    """Order source prior by balanced identity and transition frontiers.

    STO has two different source-prior duties: keep identity/title evidence
    close to each query root, and keep role-transition evidence close enough for
    multi-hop chaining. This order gives each root one identity-preserving
    witness and one transition witness per round before falling back to admitted
    order.
    """

    admitted_docs = _clean_doc_indices(local_graph.get("admitted_doc_indices", []) or [])
    if not admitted_docs:
        return []
    admitted_set = set(admitted_docs)
    seeds = _clean_doc_indices(local_graph.get("seed_doc_indices", []) or [])
    if not seeds:
        seeds = _clean_doc_indices(
            [
                *(local_graph.get("textual_seed_doc_indices", []) or []),
                *(local_graph.get("symbolic_seed_doc_indices", []) or []),
            ]
        )
    seeds = [int(seed) for seed in seeds if int(seed) in admitted_set]
    if not seeds:
        seeds = admitted_docs[:]
    clean_max = len(admitted_docs) if max_docs is None else max(int(max_docs), 0)
    if clean_max <= 0:
        return []

    stats = local_graph.get("stats", {}) or {}
    try:
        closure_hops = int(stats.get("closure_hops", 0) or 0)
    except (TypeError, ValueError):
        closure_hops = 0
    adjacency = _local_edge_adjacency(list(local_graph.get("local_edges", []) or []))
    doc_query_token_coverage = _local_graph_query_token_coverage(local_graph)
    query_token_stems = _local_graph_query_token_stems(local_graph)
    identity_kinds = {TITLE_ROLE_GROUNDING, SAME_SUBJECT, SOURCE_ENDPOINT_INCIDENCE}
    transition_kinds = {SENTENCE_GROUNDED_TRANSITION, ROLE_BRIDGE}
    identity_frontiers = {
        int(seed): _seed_frontier_order(
            seed=int(seed),
            admitted_set=admitted_set,
            adjacency=adjacency,
            closure_hops=closure_hops,
            required_kinds=identity_kinds,
            doc_query_token_coverage=doc_query_token_coverage,
            query_token_stems=query_token_stems,
        )
        for seed in seeds
    }
    transition_frontiers = {
        int(seed): _seed_frontier_order(
            seed=int(seed),
            admitted_set=admitted_set,
            adjacency=adjacency,
            closure_hops=closure_hops,
            required_kinds=transition_kinds,
            doc_query_token_coverage=doc_query_token_coverage,
            query_token_stems=query_token_stems,
        )
        for seed in seeds
    }
    identity_positions = {int(seed): 0 for seed in seeds}
    transition_positions = {int(seed): 0 for seed in seeds}
    ordered: List[int] = []
    seen: Set[int] = set()

    def add(doc_idx: int) -> bool:
        doc = int(doc_idx)
        if doc not in admitted_set or doc in seen or len(ordered) >= clean_max:
            return False
        seen.add(doc)
        ordered.append(doc)
        return True

    def add_next(seed: int, frontiers: Mapping[int, Sequence[int]], positions: Dict[int, int]) -> bool:
        frontier = list(frontiers.get(int(seed), []) or [])
        pos = int(positions.get(int(seed), 0) or 0)
        while pos < len(frontier):
            candidate = int(frontier[pos])
            pos += 1
            positions[int(seed)] = pos
            if add(candidate):
                return True
        positions[int(seed)] = pos
        return False

    for seed in seeds:
        if len(ordered) >= clean_max:
            break
        add(int(seed))
        add_next(int(seed), identity_frontiers, identity_positions)
        add_next(int(seed), transition_frontiers, transition_positions)

    progressed = True
    while progressed and len(ordered) < clean_max:
        progressed = False
        for seed in seeds:
            if len(ordered) >= clean_max:
                break
            progressed = add_next(int(seed), identity_frontiers, identity_positions) or progressed
            progressed = add_next(int(seed), transition_frontiers, transition_positions) or progressed

    for doc_idx in admitted_docs:
        if len(ordered) >= clean_max:
            break
        add(int(doc_idx))
    return ordered


def order_local_sto_layered_source_prior(
    *,
    local_graph: Mapping[str, Any],
    max_docs: int | None = None,
) -> List[int]:
    """Order source prior by evidence-chain layers.

    The source-prior budget is assigned in layers:

    1. query roots;
    2. one role-transition witness for each root;
    3. one identity/title witness for each root;
    4. remaining role-transition frontier;
    5. remaining identity/title frontier.

    This keeps MuSiQue-style transition chains ahead of identity expansion,
    while still letting HotpotQA-style title companions enter the early budget.
    """

    admitted_docs = _clean_doc_indices(local_graph.get("admitted_doc_indices", []) or [])
    if not admitted_docs:
        return []
    admitted_set = set(admitted_docs)
    seeds = _clean_doc_indices(local_graph.get("seed_doc_indices", []) or [])
    if not seeds:
        seeds = _clean_doc_indices(
            [
                *(local_graph.get("textual_seed_doc_indices", []) or []),
                *(local_graph.get("symbolic_seed_doc_indices", []) or []),
            ]
        )
    seeds = [int(seed) for seed in seeds if int(seed) in admitted_set]
    if not seeds:
        seeds = admitted_docs[:]
    clean_max = len(admitted_docs) if max_docs is None else max(int(max_docs), 0)
    if clean_max <= 0:
        return []

    stats = local_graph.get("stats", {}) or {}
    try:
        closure_hops = int(stats.get("closure_hops", 0) or 0)
    except (TypeError, ValueError):
        closure_hops = 0
    adjacency = _local_edge_adjacency(list(local_graph.get("local_edges", []) or []))
    doc_query_token_coverage = _local_graph_query_token_coverage(local_graph)
    query_token_stems = _local_graph_query_token_stems(local_graph)
    identity_kinds = {TITLE_ROLE_GROUNDING, SAME_SUBJECT, SOURCE_ENDPOINT_INCIDENCE}
    transition_kinds = {SENTENCE_GROUNDED_TRANSITION, ROLE_BRIDGE}
    transition_frontiers = {
        int(seed): _seed_frontier_order(
            seed=int(seed),
            admitted_set=admitted_set,
            adjacency=adjacency,
            closure_hops=closure_hops,
            required_kinds=transition_kinds,
            doc_query_token_coverage=doc_query_token_coverage,
            query_token_stems=query_token_stems,
        )
        for seed in seeds
    }
    identity_frontiers = {
        int(seed): _seed_frontier_order(
            seed=int(seed),
            admitted_set=admitted_set,
            adjacency=adjacency,
            closure_hops=closure_hops,
            required_kinds=identity_kinds,
            doc_query_token_coverage=doc_query_token_coverage,
            query_token_stems=query_token_stems,
        )
        for seed in seeds
    }
    transition_positions = {int(seed): 0 for seed in seeds}
    identity_positions = {int(seed): 0 for seed in seeds}
    ordered: List[int] = []
    seen: Set[int] = set()

    def add(doc_idx: int) -> bool:
        doc = int(doc_idx)
        if doc not in admitted_set or doc in seen or len(ordered) >= clean_max:
            return False
        seen.add(doc)
        ordered.append(doc)
        return True

    def add_next(seed: int, frontiers: Mapping[int, Sequence[int]], positions: Dict[int, int]) -> bool:
        frontier = list(frontiers.get(int(seed), []) or [])
        pos = int(positions.get(int(seed), 0) or 0)
        while pos < len(frontier):
            candidate = int(frontier[pos])
            pos += 1
            positions[int(seed)] = pos
            if add(candidate):
                return True
        positions[int(seed)] = pos
        return False

    for seed in seeds:
        if len(ordered) >= clean_max:
            break
        add(int(seed))
        add_next(int(seed), transition_frontiers, transition_positions)

    for seed in seeds:
        if len(ordered) >= clean_max:
            break
        add_next(int(seed), identity_frontiers, identity_positions)

    progressed = True
    while progressed and len(ordered) < clean_max:
        progressed = False
        for seed in seeds:
            if len(ordered) >= clean_max:
                break
            progressed = add_next(int(seed), transition_frontiers, transition_positions) or progressed
        for seed in seeds:
            if len(ordered) >= clean_max:
                break
            progressed = add_next(int(seed), identity_frontiers, identity_positions) or progressed

    for doc_idx in admitted_docs:
        if len(ordered) >= clean_max:
            break
        add(int(doc_idx))
    return ordered


def _order_selected_evidence_set(
    *,
    doc_indices: Sequence[int],
    query_endpoints: Sequence[str],
    query_tokens: Set[str],
    corpus_index: Mapping[str, Any],
    local_graph: Mapping[str, Any],
    local_edges: Sequence[Mapping[str, Any]],
    bm25_scores: Mapping[int, float],
) -> List[Dict[str, Any]]:
    remaining = set(_clean_doc_indices(doc_indices))
    adjacency = _local_edge_adjacency(local_edges)
    selected: List[int] = []
    rows: List[Dict[str, Any]] = []
    covered_endpoints: Set[str] = set()
    covered_tokens: Set[str] = set()

    def doc_key(doc_idx: int, *, connected_only: bool = False) -> Tuple[Any, ...]:
        endpoint_coverage = _doc_query_endpoint_coverage(
            doc_idx=doc_idx,
            query_endpoints=query_endpoints,
            corpus_index=corpus_index,
        )
        token_coverage = _doc_query_token_coverage(
            doc_idx=doc_idx,
            query_tokens=query_tokens,
            corpus_index=corpus_index,
        )
        connecting_edges = [
            edge
            for selected_doc in selected
            for edge in adjacency.get(int(selected_doc), []) or []
            if _edge_neighbor(edge, int(selected_doc)) == int(doc_idx)
        ]
        if connected_only and not connecting_edges:
            return (1,)
        best_edge_rank = min((_best_edge_kind_rank(edge) for edge in connecting_edges), default=10)
        return (
            0,
            -len(endpoint_coverage - covered_endpoints),
            -len(token_coverage - covered_tokens),
            best_edge_rank,
            _trace_distance(local_graph, doc_idx),
            -float(bm25_scores.get(int(doc_idx), 0.0) or 0.0),
            int(doc_idx),
        )

    while remaining:
        connected_candidates = [doc_idx for doc_idx in remaining if selected and doc_key(doc_idx, connected_only=True)[0] == 0]
        if connected_candidates:
            doc_idx = min(connected_candidates, key=lambda value: doc_key(value, connected_only=True))
        else:
            doc_idx = min(remaining, key=doc_key)
        endpoint_coverage = _doc_query_endpoint_coverage(
            doc_idx=doc_idx,
            query_endpoints=query_endpoints,
            corpus_index=corpus_index,
        )
        token_coverage = _doc_query_token_coverage(
            doc_idx=doc_idx,
            query_tokens=query_tokens,
            corpus_index=corpus_index,
        )
        new_endpoints = endpoint_coverage - covered_endpoints
        new_tokens = token_coverage - covered_tokens
        via_edge = None
        if selected:
            connecting_edges = [
                edge
                for selected_doc in selected
                for edge in adjacency.get(int(selected_doc), []) or []
                if _edge_neighbor(edge, int(selected_doc)) == int(doc_idx)
            ]
            if connecting_edges:
                via_edge = min(connecting_edges, key=_best_edge_kind_rank)
        if not selected or new_endpoints:
            phase = "endpoint_cover"
        elif via_edge is not None:
            phase = "connected_companion"
        elif new_tokens:
            phase = "content_completion"
        else:
            phase = "set_completion"
        row = {
            "doc_index": int(doc_idx),
            "phase": phase,
            "covered_query_endpoints": sorted(endpoint_coverage),
            "new_query_endpoints": sorted(new_endpoints),
            "covered_query_tokens": sorted(token_coverage),
            "new_query_tokens": sorted(new_tokens),
            "distance": _trace_distance(local_graph, int(doc_idx)),
            "bm25_score": round(float(bm25_scores.get(int(doc_idx), 0.0) or 0.0), 6),
        }
        if via_edge is not None:
            row["via_edge"] = {
                "left_doc": int(via_edge.get("left_doc", -1)),
                "right_doc": int(via_edge.get("right_doc", -1)),
                "kinds": list(via_edge.get("kinds", []) or []),
                "endpoints": list(via_edge.get("endpoints", []) or []),
            }
        rows.append(row)
        selected.append(int(doc_idx))
        remaining.remove(int(doc_idx))
        covered_endpoints.update(endpoint_coverage)
        covered_tokens.update(token_coverage)
    return rows


def select_local_sto_evidence_docs(
    *,
    query: str,
    corpus_index: Mapping[str, Any],
    local_graph: Mapping[str, Any],
    evidence_set_size: int = 5,
) -> Dict[str, Any]:
    """Select a compact evidence set from an admitted query-local STO graph.

    The selector is lexicographic rather than weighted:

    1. search directly over candidate evidence sets inside the admitted graph;
    2. prefer sets that cover grounded query endpoints and query content;
    3. break ties by STO connectivity, admission distance, and lexical order.
    """

    clean_size = max(int(evidence_set_size), 1)
    admitted_docs = _clean_doc_indices(local_graph.get("admitted_doc_indices", []) or [])
    if not admitted_docs:
        return {
            "selected_doc_indices": [],
            "selection_rows": [],
            "selection_policy": "local_sto_lexicographic_set_constructor",
        }

    bm25_scores = score_docs_bm25(query=query, corpus_index=corpus_index)
    query_tokens = content_tokens(query)
    query_endpoints = [str(endpoint) for endpoint in local_graph.get("symbolic_anchor_endpoints", []) or []]
    local_edges = list(local_graph.get("local_edges", []) or [])
    adjacency = _local_edge_adjacency(local_edges)
    core_set = _search_local_evidence_set(
        admitted_docs=admitted_docs,
        query_endpoints=query_endpoints,
        query_tokens=query_tokens,
        corpus_index=corpus_index,
        local_graph=local_graph,
        local_edges=local_edges,
        adjacency=adjacency,
        bm25_scores=bm25_scores,
        evidence_set_size=clean_size,
    )
    selected_set = _complete_local_evidence_set(
        core_doc_indices=core_set,
        admitted_docs=admitted_docs,
        query_endpoints=query_endpoints,
        query_tokens=query_tokens,
        corpus_index=corpus_index,
        local_graph=local_graph,
        local_edges=local_edges,
        adjacency=adjacency,
        bm25_scores=bm25_scores,
        evidence_set_size=clean_size,
    )
    selection_rows = _order_selected_evidence_set(
        doc_indices=selected_set,
        query_endpoints=query_endpoints,
        query_tokens=query_tokens,
        corpus_index=corpus_index,
        local_graph=local_graph,
        local_edges=local_edges,
        bm25_scores=bm25_scores,
    )
    selected_docs = _clean_doc_indices([int(row["doc_index"]) for row in selection_rows])

    return {
        "selected_doc_indices": _clean_doc_indices(selected_docs)[:clean_size],
        "selection_rows": selection_rows,
        "selection_policy": "local_sto_lexicographic_set_constructor",
        "query_endpoints": query_endpoints,
        "admitted_doc_count": len(admitted_docs),
        "local_edge_count": len(local_edges),
        "set_objective": {
            "core_doc_indices": list(core_set),
            "covered_query_endpoints": sorted(
                _set_covered_query_endpoints(
                    doc_indices=selected_docs,
                    query_endpoints=query_endpoints,
                    corpus_index=corpus_index,
                )
            ),
            "covered_query_tokens": sorted(
                _set_covered_query_tokens(
                    doc_indices=selected_docs,
                    query_tokens=query_tokens,
                    corpus_index=corpus_index,
                )
            ),
            "connected_components": _selected_components(doc_indices=selected_docs, adjacency=adjacency),
            "internal_edge_counts": _internal_edge_counts(doc_indices=selected_docs, local_edges=local_edges),
            "internal_edge_tier_counts": _internal_edge_tier_counts(
                doc_indices=selected_docs,
                local_edges=local_edges,
            ),
        },
    }


def build_query_local_sto_graph(
    *,
    query: str,
    corpus_index: Mapping[str, Any],
    role_graph: Mapping[str, Any] | None = None,
    textual_seed_doc_indices: Sequence[int] | None = None,
    textual_seed_top_k: int = 20,
    max_endpoint_degree: int = 30,
    closure_hops: int = 2,
    candidate_limit: int = 120,
    enable_query_supported_same_object_handoff: bool = False,
    enable_variable_flow_traversal: bool = False,
) -> Dict[str, Any]:
    """Build a query-local evidence graph admitted from the global STO graph.

    The admission rule is deliberately simple:

    1. enter the corpus through lexical textual seeds and grounded endpoint
       anchors;
    2. expand through role-safe STO edges only;
    3. return the induced local graph and an auditable admission trace.

    ``same_object`` is normally only a local evidence edge.  When variable-flow
    traversal is enabled, a same-object edge is traversable only if its shared
    endpoint is already active in the current document's query-rooted fact
    flow. The older query-supported object handoff rule remains a diagnostic
    ablation and is disabled by default.

    ``closure_hops`` and ``candidate_limit`` are execution bounds, not ranking
    knobs. They keep graph expansion finite and reproducible.
    """

    clean_candidate_limit = max(int(candidate_limit), 1)
    clean_closure_hops = max(int(closure_hops), 0)
    clean_textual_top_k = max(int(textual_seed_top_k), 0)

    resolved_role_graph = role_graph or build_role_transition_graph(
        corpus_index=corpus_index,
        max_endpoint_degree=max_endpoint_degree,
        include_title_role_grounding=True,
    )
    endpoint_to_docs: Mapping[str, Sequence[int]] = corpus_index.get("endpoint_to_docs", {}) or {}
    query_grounding = ground_query_endpoints(
        query=query,
        endpoint_to_docs=endpoint_to_docs,
        max_endpoint_degree=max_endpoint_degree,
    )
    query_tokens = content_tokens(query)
    query_token_stems = _token_stems(query_tokens)
    doc_query_token_coverage = _doc_query_token_coverage_map(
        corpus_index=corpus_index,
        query_tokens=query_tokens,
    )
    corpus_unit_by_id = _unit_by_id(corpus_index)
    facts_by_doc = _fact_units_by_doc(corpus_index)
    role_graph_doc_titles = {
        int(doc_idx): str(endpoint)
        for doc_idx, endpoint in (resolved_role_graph.get("doc_titles", {}) or {}).items()
        if str(endpoint or "").strip()
    }
    query_endpoints = [str(endpoint) for endpoint in query_grounding.get("query_endpoints", []) or []]
    endpoint_seed_docs = _endpoint_seed_docs(endpoints=query_endpoints, endpoint_to_docs=endpoint_to_docs)
    symbolic_seed_doc_indices = unique_ranked(
        doc_idx for endpoint in query_endpoints for doc_idx in endpoint_seed_docs.get(endpoint, []) or []
    )

    if textual_seed_doc_indices is None:
        textual_seed_docs = rank_docs_bm25(
            query=query,
            corpus_index=corpus_index,
            top_k=clean_textual_top_k,
        )
    else:
        textual_seed_docs = _clean_doc_indices(textual_seed_doc_indices)[:clean_textual_top_k]

    seed_docs = unique_ranked([*textual_seed_docs, *symbolic_seed_doc_indices])
    admitted_docs: List[int] = []
    admitted_set: Set[int] = set()
    admission_trace: Dict[int, Dict[str, Any]] = {}
    doc_variable_flow_endpoints: Dict[int, Set[str]] = {}
    doc_variable_flow_endpoint_ranks: Dict[int, Dict[str, int]] = {}
    variable_flow_doc_pair_keys: Set[str] = set()
    variable_flow_doc_pair_ranks: Dict[str, int] = {}
    variable_flow_update_count = 0
    queue: Deque[Tuple[int, int]] = deque()

    def set_doc_flow(doc_idx: int, active_endpoint_ranks: Mapping[str, int]) -> bool:
        doc = int(doc_idx)
        expanded = _expand_doc_variable_flow_endpoint_ranks(
            doc_idx=doc,
            active_endpoint_ranks=active_endpoint_ranks,
            facts_by_doc=facts_by_doc,
            query_token_stems=query_token_stems,
        )
        if not expanded:
            return False
        existing_ranks = doc_variable_flow_endpoint_ranks.setdefault(doc, {})
        improved = False
        for endpoint, rank in expanded.items():
            current = existing_ranks.get(str(endpoint))
            if current is None or int(rank) < int(current):
                existing_ranks[str(endpoint)] = int(rank)
                improved = True
        if not improved:
            return False
        doc_variable_flow_endpoints[doc] = set(existing_ranks)
        return True

    def edge_endpoint_ranks_from_doc(doc_idx: int, edge: Mapping[str, Any]) -> Dict[str, int]:
        source_ranks = doc_variable_flow_endpoint_ranks.get(int(doc_idx), {}) or {}
        ranks: Dict[str, int] = {}
        for endpoint in edge.get("endpoints", []) or []:
            endpoint_text = str(endpoint)
            if not endpoint_text.strip():
                continue
            ranks[endpoint_text] = int(source_ranks.get(endpoint_text, 1))
        return ranks

    def remember_variable_flow_pair(left_doc: int, right_doc: int, endpoint_ranks: Mapping[str, int]) -> None:
        if not endpoint_ranks:
            return
        key = _doc_pair_key(left_doc, right_doc)
        rank = min(int(value) for value in endpoint_ranks.values())
        variable_flow_doc_pair_keys.add(key)
        current = variable_flow_doc_pair_ranks.get(key)
        if current is None or rank < int(current):
            variable_flow_doc_pair_ranks[key] = rank

    def admit_seed(doc_idx: int, source: str) -> None:
        if doc_idx < 0 or len(admitted_docs) >= clean_candidate_limit:
            return
        if doc_idx not in admitted_set:
            admitted_set.add(doc_idx)
            admitted_docs.append(doc_idx)
            admission_trace[doc_idx] = {
                "distance": 0,
                "sources": [],
                "via": None,
            }
            seed_flow = _doc_query_start_endpoints(
                doc_idx=doc_idx,
                query_endpoints=query_endpoints,
                corpus_index=corpus_index,
                role_graph=resolved_role_graph,
            )
            set_doc_flow(int(doc_idx), {endpoint: 0 for endpoint in seed_flow})
            queue.append((doc_idx, 0))
        sources = admission_trace[doc_idx].setdefault("sources", [])
        if source not in sources:
            sources.append(source)

    for doc_idx in textual_seed_docs:
        admit_seed(int(doc_idx), "textual_seed")
    for doc_idx in symbolic_seed_doc_indices:
        admit_seed(int(doc_idx), "symbolic_anchor")

    adjacency: Mapping[int, Sequence[Mapping[str, Any]]] = resolved_role_graph.get("adjacency", {}) or {}
    while queue and len(admitted_docs) < clean_candidate_limit:
        current_doc, distance = queue.popleft()
        if distance >= clean_closure_hops:
            continue
        if bool(enable_variable_flow_traversal):
            edge_rows = _sort_variable_flow_admission_edges(
                adjacency.get(int(current_doc), []) or [],
                from_doc=int(current_doc),
                doc_variable_flow_endpoints=doc_variable_flow_endpoints,
                doc_query_token_coverage=doc_query_token_coverage,
                query_token_stems=query_token_stems,
            )
        elif bool(enable_query_supported_same_object_handoff):
            edge_rows = _sort_admission_edges(
                adjacency.get(int(current_doc), []) or [],
                from_doc=int(current_doc),
                unit_by_id=corpus_unit_by_id,
                doc_query_token_coverage=doc_query_token_coverage,
                query_token_stems=query_token_stems,
            )
        else:
            edge_rows = _sort_traversal_edges(
                adjacency.get(int(current_doc), []) or [],
                from_doc=int(current_doc),
                doc_query_token_coverage=doc_query_token_coverage,
                query_token_stems=query_token_stems,
            )
        for edge in edge_rows:
            if bool(enable_variable_flow_traversal):
                allowed_kinds = _edge_traversal_kinds(edge)
                variable_flow_kinds = []
                if not allowed_kinds:
                    variable_flow_kinds = _edge_variable_flow_traversal_kinds(
                        edge,
                        from_doc=int(current_doc),
                        doc_variable_flow_endpoints=doc_variable_flow_endpoints,
                    )
                    allowed_kinds = variable_flow_kinds
            elif bool(enable_query_supported_same_object_handoff):
                allowed_kinds = _edge_admission_traversal_kinds(
                    edge,
                    from_doc=int(current_doc),
                    unit_by_id=corpus_unit_by_id,
                    query_token_stems=query_token_stems,
                )
                variable_flow_kinds = []
            else:
                allowed_kinds = _edge_traversal_kinds(edge)
                variable_flow_kinds = []
            if not allowed_kinds:
                continue
            neighbor = _edge_neighbor(edge, int(current_doc))
            if neighbor < 0:
                continue
            edge_endpoints = {
                str(endpoint)
                for endpoint in edge.get("endpoints", []) or []
                if str(endpoint).strip()
            }
            edge_endpoint_ranks = edge_endpoint_ranks_from_doc(current_doc, edge)
            if neighbor in admitted_set:
                if bool(enable_variable_flow_traversal) and set_doc_flow(neighbor, edge_endpoint_ranks):
                    variable_flow_update_count += 1
                    queue.append((neighbor, int(distance + 1)))
                    if variable_flow_kinds:
                        remember_variable_flow_pair(current_doc, neighbor, edge_endpoint_ranks)
                continue
            admitted_set.add(neighbor)
            admitted_docs.append(neighbor)
            set_doc_flow(neighbor, edge_endpoint_ranks)
            if variable_flow_kinds:
                remember_variable_flow_pair(current_doc, neighbor, edge_endpoint_ranks)
            admission_trace[neighbor] = {
                "distance": int(distance + 1),
                "sources": ["sto_closure"],
                "via": {
                    "from_doc": int(current_doc),
                    "edge_kinds": allowed_kinds,
                    "endpoints": list(edge.get("endpoints", []) or []),
                },
            }
            queue.append((neighbor, int(distance + 1)))
            if len(admitted_docs) >= clean_candidate_limit:
                break

    local_edges = _induced_local_edges(role_graph=resolved_role_graph, admitted_docs=admitted_docs)
    local_edge_tier_counts = _local_edge_tier_counts(local_edges)
    return {
        "method": "query_local_sto_graph_admission",
        "query": str(query or ""),
        "query_grounding": query_grounding,
        "textual_seed_doc_indices": textual_seed_docs,
        "symbolic_anchor_endpoints": query_endpoints,
        "symbolic_endpoint_seed_doc_indices": endpoint_seed_docs,
        "symbolic_seed_doc_indices": symbolic_seed_doc_indices,
        "seed_doc_indices": seed_docs,
        "admitted_doc_indices": admitted_docs,
        "local_edges": local_edges,
        "doc_admission_trace": {str(doc_idx): trace for doc_idx, trace in sorted(admission_trace.items())},
        "doc_query_token_coverage": {
            str(doc_idx): list(doc_query_token_coverage.get(int(doc_idx), ()))
            for doc_idx in admitted_docs
            if doc_query_token_coverage.get(int(doc_idx), ())
        },
        "doc_title_endpoints": {
            str(doc_idx): str(role_graph_doc_titles.get(int(doc_idx), ""))
            for doc_idx in admitted_docs
            if role_graph_doc_titles.get(int(doc_idx), "")
        },
        "doc_variable_flow_endpoints": {
            str(doc_idx): sorted(doc_variable_flow_endpoints.get(int(doc_idx), set()))
            for doc_idx in admitted_docs
            if doc_variable_flow_endpoints.get(int(doc_idx), set())
        },
        "doc_variable_flow_endpoint_ranks": {
            str(doc_idx): dict(sorted(doc_variable_flow_endpoint_ranks.get(int(doc_idx), {}).items()))
            for doc_idx in admitted_docs
            if doc_variable_flow_endpoint_ranks.get(int(doc_idx), {})
        },
        "variable_flow_doc_pair_keys": sorted(variable_flow_doc_pair_keys),
        "variable_flow_doc_pair_ranks": dict(sorted(variable_flow_doc_pair_ranks.items())),
        "query_token_stems": sorted(query_token_stems),
        "stats": {
            "textual_seed_count": len(textual_seed_docs),
            "symbolic_anchor_count": len(query_endpoints),
            "symbolic_seed_doc_count": len(symbolic_seed_doc_indices),
            "seed_doc_count": len(seed_docs),
            "admitted_doc_count": len(admitted_docs),
            "local_edge_count": len(local_edges),
            "local_edge_tier_counts": local_edge_tier_counts,
            "closure_hops": clean_closure_hops,
            "candidate_limit": clean_candidate_limit,
            "allowed_edge_kinds": sorted(LOCAL_EVIDENCE_EDGE_KINDS),
            "traversal_edge_kinds": sorted(LOCAL_TRAVERSAL_EDGE_KINDS),
            "local_evidence_edge_kinds": sorted(LOCAL_EVIDENCE_EDGE_KINDS),
            "query_supported_same_object_handoff_enabled": bool(enable_query_supported_same_object_handoff),
            "variable_flow_traversal_enabled": bool(enable_variable_flow_traversal),
            "variable_flow_doc_count": len(doc_variable_flow_endpoints),
            "variable_flow_edge_count": len(variable_flow_doc_pair_keys),
            "variable_flow_update_count": int(variable_flow_update_count),
            "allowed_edge_tiers": [EVIDENCE_TRANSITION_TIER, WEAK_CONNECTIVITY_TIER],
            "role_graph_edge_count": int((resolved_role_graph.get("stats", {}) or {}).get("edge_count", 0) or 0),
            "role_graph_edge_tier_counts": dict(
                (resolved_role_graph.get("stats", {}) or {}).get("edge_tier_counts", {}) or {}
            ),
        },
    }
