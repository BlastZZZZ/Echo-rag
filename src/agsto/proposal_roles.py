"""Structured role metadata for native AG-STO proposal emissions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PAIR_EMISSION_VIEWS = (
    ("specificity_pair", "specificity_pairwise_transition_emission"),
    ("hybrid_residual_pair", "hybrid_residual_pairwise_transition_emission"),
    ("semantic_pair", "semantic_pairwise_transition_emission"),
    ("pair", "pairwise_transition_emission"),
)
PAIR_VIEWS = {view for view, _field in PAIR_EMISSION_VIEWS}


def _as_doc_idx(value: Any) -> int | None:
    try:
        doc_idx = int(value)
    except (TypeError, ValueError):
        return None
    return doc_idx if doc_idx >= 0 else None


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _unique_doc_indices(values: Iterable[Any]) -> List[int]:
    docs: List[int] = []
    seen: set[int] = set()
    for value in values:
        doc_idx = _as_doc_idx(value)
        if doc_idx is None or doc_idx in seen:
            continue
        docs.append(doc_idx)
        seen.add(doc_idx)
    return docs


def _clean_endpoint_labels(values: Any) -> List[str | None]:
    if values is None:
        return [None]
    if isinstance(values, str):
        label = values.strip().lower()
        return [label] if label else [None]
    if not isinstance(values, Sequence):
        return [None]

    labels: List[str] = []
    seen: set[str] = set()
    for value in values:
        label = str(value or "").strip().lower()
        if not label or label in seen:
            continue
        labels.append(label)
        seen.add(label)
    return labels or [None]


def _record(
    *,
    doc_idx: int,
    view: str,
    role_type: str,
    partner_doc: int | None,
    endpoint_label: str | None,
    score: float,
    rank: int,
) -> Dict[str, Any]:
    return {
        "doc_idx": int(doc_idx),
        "view": str(view),
        "role_type": str(role_type),
        "partner_doc": int(partner_doc) if partner_doc is not None else None,
        "endpoint_label": str(endpoint_label) if endpoint_label else None,
        "score": float(score),
        "rank": int(rank),
    }


def _pair_support_rows(emission: Mapping[str, Any] | None) -> List[Mapping[str, Any]]:
    if not isinstance(emission, Mapping):
        return []
    rows = emission.get("support_pairs", []) or []
    return [row for row in rows if isinstance(row, Mapping)]


def _add_pair_records(
    records: List[Dict[str, Any]],
    *,
    view: str,
    emission: Mapping[str, Any] | None,
) -> None:
    for rank, pair in enumerate(_pair_support_rows(emission), start=1):
        anchor_doc = _as_doc_idx(pair.get("anchor_doc"))
        completion_doc = _as_doc_idx(pair.get("completion_doc"))
        score = _as_float(pair.get("completion_score", pair.get("score", 0.0)))
        if anchor_doc is not None:
            records.append(
                _record(
                    doc_idx=anchor_doc,
                    view=view,
                    role_type="anchor",
                    partner_doc=completion_doc,
                    endpoint_label=None,
                    score=score,
                    rank=rank,
                )
            )
        if completion_doc is not None:
            records.append(
                _record(
                    doc_idx=completion_doc,
                    view=view,
                    role_type="completion",
                    partner_doc=anchor_doc,
                    endpoint_label=None,
                    score=score,
                    rank=rank,
                )
            )


def _endpoint_rows(endpoint_transition_emission: Any) -> List[Mapping[str, Any]]:
    if isinstance(endpoint_transition_emission, Mapping):
        for field in ("support_pairs", "boundary_readout_pairs", "boundary_pairs"):
            rows = endpoint_transition_emission.get(field, []) or []
            if rows:
                return [row for row in rows if isinstance(row, Mapping)]
        return []
    if isinstance(endpoint_transition_emission, Sequence) and not isinstance(endpoint_transition_emission, (str, bytes)):
        return [row for row in endpoint_transition_emission if isinstance(row, Mapping)]
    return []


def _add_endpoint_records(
    records: List[Dict[str, Any]],
    *,
    endpoint_transition_emission: Any,
) -> None:
    for rank, pair in enumerate(_endpoint_rows(endpoint_transition_emission), start=1):
        completion_doc = _as_doc_idx(pair.get("completion_doc"))
        if completion_doc is None:
            continue

        raw_anchor_docs = pair.get("anchor_docs")
        if raw_anchor_docs is None:
            raw_anchor_docs = [pair.get("anchor_doc")]
        anchor_docs = _unique_doc_indices(raw_anchor_docs or [])
        score = _as_float(pair.get("direct_transition_score", pair.get("score", 0.0)))
        endpoint_labels = _clean_endpoint_labels(
            pair.get("endpoint", pair.get("endpoint_label", pair.get("shared_endpoints_sample")))
        )
        for anchor_doc in anchor_docs:
            for endpoint_label in endpoint_labels:
                records.append(
                    _record(
                        doc_idx=anchor_doc,
                        view="endpoint_transition",
                        role_type="endpoint_anchor",
                        partner_doc=completion_doc,
                        endpoint_label=endpoint_label,
                        score=score,
                        rank=rank,
                    )
                )
                records.append(
                    _record(
                        doc_idx=completion_doc,
                        view="endpoint_transition",
                        role_type="endpoint_completion",
                        partner_doc=anchor_doc,
                        endpoint_label=endpoint_label,
                        score=score,
                        rank=rank,
                    )
                )


def _add_support_records(
    records: List[Dict[str, Any]],
    *,
    support_set_search: Mapping[str, Any] | None,
) -> None:
    if not isinstance(support_set_search, Mapping):
        return

    for rank, support_set in enumerate(support_set_search.get("top_support_sets", []) or [], start=1):
        if not isinstance(support_set, Mapping):
            continue
        score = _as_float(support_set.get("score", 0.0))
        for doc_idx in _unique_doc_indices(support_set.get("doc_indices", []) or []):
            records.append(
                _record(
                    doc_idx=doc_idx,
                    view="support",
                    role_type="support_member",
                    partner_doc=None,
                    endpoint_label=None,
                    score=score,
                    rank=rank,
                )
            )


def _add_neighborhood_records(
    records: List[Dict[str, Any]],
    *,
    query_conditioned_neighborhood: Mapping[str, Any] | None,
) -> None:
    if not isinstance(query_conditioned_neighborhood, Mapping):
        return

    docs = _unique_doc_indices(
        list(query_conditioned_neighborhood.get("selected_neighborhood_doc_indices", []) or [])
        + list(query_conditioned_neighborhood.get("retrieved_doc_indices", []) or [])
    )
    for pair in query_conditioned_neighborhood.get("boundary_readout_pairs", []) or []:
        if not isinstance(pair, Mapping):
            continue
        docs.extend(_unique_doc_indices([pair.get("completion_doc")]))
    docs = _unique_doc_indices(docs)
    for rank, doc_idx in enumerate(docs, start=1):
        records.append(
            _record(
                doc_idx=doc_idx,
                view="neighborhood",
                role_type="neighborhood_member",
                partner_doc=None,
                endpoint_label=None,
                score=0.0,
                rank=rank,
            )
        )


def _add_dense_anchor_records(
    records: List[Dict[str, Any]],
    *,
    anchor_doc_indices: Sequence[int] | None,
    native_dense_doc_indices: Sequence[int] | None,
) -> None:
    dense_docs = _unique_doc_indices(native_dense_doc_indices or anchor_doc_indices or [])
    for rank, doc_idx in enumerate(dense_docs, start=1):
        records.append(
            _record(
                doc_idx=doc_idx,
                view="dense_anchor",
                role_type="dense_anchor",
                partner_doc=None,
                endpoint_label=None,
                score=0.0,
                rank=rank,
            )
        )


def build_proposal_role_records(
    *,
    pairwise_transition_emission: Mapping[str, Any] | None = None,
    specificity_pairwise_transition_emission: Mapping[str, Any] | None = None,
    hybrid_residual_pairwise_transition_emission: Mapping[str, Any] | None = None,
    semantic_pairwise_transition_emission: Mapping[str, Any] | None = None,
    endpoint_transition_emission: Any = None,
    support_set_search: Mapping[str, Any] | None = None,
    query_conditioned_neighborhood: Mapping[str, Any] | None = None,
    anchor_doc_indices: Sequence[int] | None = None,
    native_dense_doc_indices: Sequence[int] | None = None,
) -> List[Dict[str, Any]]:
    """Build JSON-serializable role records from native proposal emissions."""

    records: List[Dict[str, Any]] = []
    emissions = {
        "pairwise_transition_emission": pairwise_transition_emission,
        "specificity_pairwise_transition_emission": specificity_pairwise_transition_emission,
        "hybrid_residual_pairwise_transition_emission": hybrid_residual_pairwise_transition_emission,
        "semantic_pairwise_transition_emission": semantic_pairwise_transition_emission,
    }
    for view, field in PAIR_EMISSION_VIEWS:
        _add_pair_records(records, view=view, emission=emissions.get(field))
    _add_endpoint_records(records, endpoint_transition_emission=endpoint_transition_emission)
    _add_support_records(records, support_set_search=support_set_search)
    _add_neighborhood_records(records, query_conditioned_neighborhood=query_conditioned_neighborhood)
    _add_dense_anchor_records(
        records,
        anchor_doc_indices=anchor_doc_indices,
        native_dense_doc_indices=native_dense_doc_indices,
    )
    return records


def proposal_role_summary_by_doc(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Summarize role records by document using JSON-friendly string keys."""

    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, Mapping):
            continue
        doc_idx = _as_doc_idx(record.get("doc_idx"))
        if doc_idx is None:
            continue
        grouped[doc_idx].append(record)

    summary: Dict[str, Dict[str, Any]] = {}
    for doc_idx in sorted(grouped):
        doc_records = grouped[doc_idx]
        partner_docs = sorted(
            {
                partner_doc
                for partner_doc in (_as_doc_idx(record.get("partner_doc")) for record in doc_records)
                if partner_doc is not None
            }
        )
        endpoint_labels = sorted(
            {
                str(record.get("endpoint_label")).strip().lower()
                for record in doc_records
                if str(record.get("endpoint_label") or "").strip()
            }
        )
        scores = [_as_float(record.get("score", 0.0)) for record in doc_records]
        ranks = [int(record.get("rank", 10**9) or 10**9) for record in doc_records]
        summary[str(doc_idx)] = {
            "doc_idx": int(doc_idx),
            "views": sorted({str(record.get("view")) for record in doc_records if record.get("view")}),
            "role_types": sorted(
                {str(record.get("role_type")) for record in doc_records if record.get("role_type")}
            ),
            "partner_docs": partner_docs,
            "endpoint_labels": endpoint_labels,
            "record_count": int(len(doc_records)),
            "best_score": float(max(scores) if scores else 0.0),
            "best_rank": int(min(ranks) if ranks else 0),
        }
    return summary


def _unit_record(
    *,
    unit_type: str,
    docs: Sequence[int],
    view: str | None = None,
    endpoint_label: str | None = None,
    score: float = 0.0,
    rank: int = 0,
) -> Dict[str, Any]:
    return {
        "unit_type": str(unit_type),
        "docs": _unique_doc_indices(docs),
        "view": str(view) if view else None,
        "endpoint_label": str(endpoint_label) if endpoint_label else None,
        "score": round(float(score), 6),
        "rank": int(rank),
    }


def _role_unit_key(unit: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(unit.get("unit_type") or ""),
        str(unit.get("view") or ""),
        str(unit.get("endpoint_label") or ""),
        tuple(int(doc_idx) for doc_idx in unit.get("docs", []) or []),
    )


def build_proposal_role_units(
    records: Sequence[Mapping[str, Any]],
    *,
    support_set_search: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Build multi-document role units from proposal role records."""

    units_by_key: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        doc_idx = _as_doc_idx(record.get("doc_idx"))
        partner_doc = _as_doc_idx(record.get("partner_doc"))
        if doc_idx is None or partner_doc is None or doc_idx == partner_doc:
            continue
        view = str(record.get("view") or "")
        role_type = str(record.get("role_type") or "")
        score = _as_float(record.get("score", 0.0))
        rank = int(record.get("rank", 10**9) or 10**9)
        if view in PAIR_VIEWS and role_type in {"anchor", "completion"}:
            docs = [doc_idx, partner_doc] if role_type == "anchor" else [partner_doc, doc_idx]
            unit = _unit_record(unit_type="pair", docs=docs, view=view, score=score, rank=rank)
        elif view == "endpoint_transition" and role_type in {"endpoint_anchor", "endpoint_completion"}:
            docs = [doc_idx, partner_doc] if role_type == "endpoint_anchor" else [partner_doc, doc_idx]
            unit = _unit_record(
                unit_type="endpoint",
                docs=docs,
                view=view,
                endpoint_label=str(record.get("endpoint_label") or "") or None,
                score=score,
                rank=rank,
            )
        else:
            continue

        key = _role_unit_key(unit)
        previous = units_by_key.get(key)
        if previous is None or (float(unit["score"]), -int(unit["rank"])) > (
            float(previous.get("score", 0.0) or 0.0),
            -int(previous.get("rank", 10**9) or 10**9),
        ):
            units_by_key[key] = unit

    if isinstance(support_set_search, Mapping):
        for rank, support_set in enumerate(support_set_search.get("top_support_sets", []) or [], start=1):
            if not isinstance(support_set, Mapping):
                continue
            docs = _unique_doc_indices(support_set.get("doc_indices", []) or [])
            if len(docs) < 2:
                continue
            unit = _unit_record(
                unit_type="support",
                docs=docs,
                view="support",
                score=_as_float(support_set.get("score", 0.0)),
                rank=rank,
            )
            units_by_key[_role_unit_key(unit)] = unit

    return [units_by_key[key] for key in sorted(units_by_key)]


def covered_role_unit_counts(
    *,
    doc_indices: Sequence[int],
    role_units: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Count complete role units covered by a document set."""

    selected = set(_unique_doc_indices(doc_indices))
    counts: Dict[str, int] = {
        "role_units": 0,
        "pair_units": 0,
        "endpoint_units": 0,
        "support_units": 0,
    }
    covered_units: List[Mapping[str, Any]] = []
    for unit in role_units:
        unit_docs = set(_unique_doc_indices(unit.get("docs", []) or []))
        if not unit_docs or not unit_docs.issubset(selected):
            continue
        unit_type = str(unit.get("unit_type") or "")
        counts["role_units"] += 1
        if unit_type == "pair":
            counts["pair_units"] += 1
        elif unit_type == "endpoint":
            counts["endpoint_units"] += 1
        elif unit_type == "support":
            counts["support_units"] += 1
        covered_units.append(unit)
    return {
        **counts,
        "covered_units": covered_units,
    }
