"""Slot-state metadata helpers for AG-STO diagnostics.

These helpers expose query-chain slot metadata without changing retrieval
behavior. They intentionally do not score, filter, or reorder documents.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

from .index import content_tokens, normalize_text


def _as_int(value: Any, default: int = -1) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _maybe_int(value: Any) -> int | None:
    parsed = _as_int(value, -1)
    return parsed if parsed >= 0 else None


def _tokenize_value(value: Any, *, preserve_anchor_identity: bool = False) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if preserve_anchor_identity:
            return sorted(token for token in normalize_text(value).split() if token)
        return sorted(content_tokens(value))
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", errors="ignore")
        if preserve_anchor_identity:
            return sorted(token for token in normalize_text(text).split() if token)
        return sorted(content_tokens(text))
    if isinstance(value, Iterable):
        tokens: Set[str] = set()
        for item in value:
            tokens.update(_tokenize_value(item, preserve_anchor_identity=preserve_anchor_identity))
        return sorted(tokens)
    return sorted(content_tokens(value))


def _clean_demand_tokens(values: Any) -> List[str]:
    return _tokenize_value(values, preserve_anchor_identity=False)


def _clean_anchor_tokens(values: Any) -> List[str]:
    return _tokenize_value(values, preserve_anchor_identity=True)


def _role_family(*, anchor_tokens: Sequence[str], demand_tokens: Sequence[str]) -> str:
    if anchor_tokens and demand_tokens:
        return "sro"
    if anchor_tokens:
        return "anchor_only"
    if demand_tokens:
        return "surface"
    return "unmatched"


def build_slot_states(
    query: str,
    *,
    anchor_labels: Sequence[Any] = (),
    demand_labels: Sequence[Any] = (),
) -> List[Dict[str, Any]]:
    """Build deterministic query-chain slot states.

    If no anchors are provided, a single demand-only slot is emitted. If no
    explicit demand labels are provided, content tokens from the query are used.
    """

    query_tokens = _clean_demand_tokens(query)
    demand_tokens = _clean_demand_tokens(demand_labels) if demand_labels else query_tokens
    raw_anchor_groups = list(anchor_labels or [])
    if not raw_anchor_groups:
        raw_anchor_groups = [()]

    states: List[Dict[str, Any]] = []
    seen: Set[tuple[Any, ...]] = set()
    for raw_anchor in raw_anchor_groups:
        anchor_tokens = _clean_anchor_tokens(raw_anchor)
        role_family = _role_family(anchor_tokens=anchor_tokens, demand_tokens=demand_tokens)
        key = (tuple(anchor_tokens), tuple(demand_tokens), role_family)
        if key in seen:
            continue
        seen.add(key)
        states.append(
            {
                "slot_id": f"slot_{len(states)}",
                "anchor_tokens": anchor_tokens,
                "demand_tokens": demand_tokens,
                "role_family": role_family,
            }
        )
    return states


def _unit_doc_idx(unit: Mapping[str, Any]) -> int:
    return _as_int(unit.get("doc_idx", unit.get("doc_index")), -1)


def _unit_id(unit: Mapping[str, Any]) -> int | None:
    if "unit_id" in unit:
        return _maybe_int(unit.get("unit_id"))
    if "_unit_int_id" in unit:
        return _maybe_int(unit.get("_unit_int_id"))
    return None


def _unit_anchor_tokens(unit: Mapping[str, Any]) -> List[str]:
    tokens: Set[str] = set()
    for field in (
        "survivor_endpoint_overlap",
        "survivor_entity_token_overlap",
        "endpoint_label",
        "_endpoints",
        "raw_endpoints",
    ):
        tokens.update(_clean_anchor_tokens(unit.get(field, []) or []))
    return sorted(tokens)


def _unit_demand_tokens(unit: Mapping[str, Any]) -> List[str]:
    explicit_tokens: Set[str] = set()
    for field in ("sro_query_tokens", "entity_query_tokens", "query_tokens"):
        explicit_tokens.update(_clean_demand_tokens(unit.get(field, []) or []))
    if explicit_tokens:
        return sorted(explicit_tokens)

    text_tokens: Set[str] = set()
    for field in ("subject", "relation", "object", "title", "text"):
        text_tokens.update(_clean_demand_tokens(unit.get(field, "") or ""))
    return sorted(text_tokens)


def _link_type(anchor_overlap: Sequence[str], demand_overlap: Sequence[str]) -> str | None:
    if anchor_overlap and demand_overlap:
        return "anchor_demand"
    if anchor_overlap:
        return "anchor"
    if demand_overlap:
        return "demand"
    return None


def link_units_to_slots(
    *,
    slot_states: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    source_view: str = "source_unit",
) -> List[Dict[str, Any]]:
    """Link evidence units to compatible slot states.

    Compatibility is reported as metadata only. The caller decides whether and
    how to use it in diagnostics.
    """

    links: List[Dict[str, Any]] = []
    for unit in units or []:
        if not isinstance(unit, Mapping):
            continue
        doc_idx = _unit_doc_idx(unit)
        if doc_idx < 0:
            continue
        unit_anchor_tokens = set(_unit_anchor_tokens(unit))
        unit_demand_tokens = set(_unit_demand_tokens(unit))
        unit_id = _unit_id(unit)
        for slot in slot_states or []:
            if not isinstance(slot, Mapping):
                continue
            slot_id = str(slot.get("slot_id", ""))
            if not slot_id:
                continue
            anchor_tokens = set(_clean_anchor_tokens(slot.get("anchor_tokens", []) or []))
            demand_tokens = set(_clean_demand_tokens(slot.get("demand_tokens", []) or []))
            anchor_overlap = sorted(unit_anchor_tokens & anchor_tokens)
            demand_overlap = sorted(unit_demand_tokens & demand_tokens)
            link_type = _link_type(anchor_overlap, demand_overlap)
            if link_type is None:
                continue
            links.append(
                {
                    "doc_idx": int(doc_idx),
                    "unit_id": int(unit_id) if unit_id is not None else None,
                    "slot_id": slot_id,
                    "link_type": link_type,
                    "role_family": str(slot.get("role_family", "")),
                    "source_view": str(source_view),
                    "anchor_overlap": anchor_overlap,
                    "demand_overlap": demand_overlap,
                }
            )
    return links


def slot_coverage_for_docs(
    *,
    doc_indices: Sequence[int],
    slot_links: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summarize distinct slot coverage for selected documents."""

    selected_docs = {_as_int(doc_idx, -1) for doc_idx in doc_indices or []}
    selected_docs.discard(-1)
    slot_ids: Set[str] = set()
    slot_to_docs: Dict[str, Set[int]] = defaultdict(set)
    link_type_counts: Counter[str] = Counter()
    selected_link_count = 0
    for link in slot_links or []:
        if not isinstance(link, Mapping):
            continue
        doc_idx = _as_int(link.get("doc_idx"), -1)
        if doc_idx not in selected_docs:
            continue
        slot_id = str(link.get("slot_id", ""))
        if not slot_id:
            continue
        selected_link_count += 1
        slot_ids.add(slot_id)
        slot_to_docs[slot_id].add(doc_idx)
        link_type_counts[str(link.get("link_type", ""))] += 1

    return {
        "covered_slot_count": int(len(slot_ids)),
        "covered_slot_ids": sorted(slot_ids),
        "selected_link_count": int(selected_link_count),
        "link_type_counts": dict(sorted(link_type_counts.items(), key=lambda item: (-int(item[1]), item[0]))),
        "docs_by_slot": {
            slot_id: sorted(docs)
            for slot_id, docs in sorted(slot_to_docs.items(), key=lambda item: item[0])
        },
    }
