from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .bridge_path_memory import build_bridge_path_records_from_hyperedge_records
from .support_signature_memory import derive_slot_support_passage_ids

CANONICAL_INTERFACE_CACHE_VERSION = 7


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).split())


def _normalize_id_list(values: Sequence[Any]) -> List[str]:
    return sorted({str(value) for value in values if str(value)})


def _coerce_id_list(values: Sequence[Any]) -> List[str]:
    return [str(value) for value in values if str(value)]


def _unique_in_order(values: Sequence[Any]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for value in values:
        normalized = str(value)
        if not normalized or normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered


def _nested_unique_id_map(value: Any) -> Dict[str, List[str]]:
    return {
        str(key): _unique_in_order(values or [])
        for key, values in dict(value or {}).items()
        if str(key)
    }


def _nested_unique_int_map(value: Any) -> Dict[str, List[int]]:
    result: Dict[str, List[int]] = {}
    for key, values in dict(value or {}).items():
        normalized_key = str(key)
        if not normalized_key:
            continue
        seen = set()
        ordered: List[int] = []
        for raw_value in values or []:
            int_value = int(raw_value)
            if int_value in seen:
                continue
            seen.add(int_value)
            ordered.append(int_value)
        result[normalized_key] = ordered
    return result


def _top_scored_ids(
    object_ids: Sequence[str],
    scores: np.ndarray,
    top_k: int,
) -> Tuple[List[str], List[float]]:
    normalized_ids = [str(value) for value in object_ids]
    score_array = np.asarray(scores, dtype=np.float32)
    if top_k <= 0 or len(normalized_ids) == 0 or score_array.size == 0:
        return [], []

    limit = min(len(normalized_ids), int(score_array.shape[0]), int(top_k))
    if limit <= 0:
        return [], []

    top_indices = np.argsort(score_array[: len(normalized_ids)])[::-1][:limit]
    selected_ids: List[str] = []
    selected_scores: List[float] = []
    seen = set()
    for local_idx in top_indices.tolist():
        object_id = normalized_ids[int(local_idx)]
        if object_id in seen:
            continue
        selected_ids.append(object_id)
        selected_scores.append(float(score_array[int(local_idx)]))
        seen.add(object_id)
    return selected_ids, selected_scores


def _rank_with_dense_fallback(
    ranked_pairs: List[Tuple[Tuple[int, ...], str]],
    top_k: int,
) -> List[str]:
    if top_k <= 0 or not ranked_pairs:
        return []
    ranked_pairs.sort(reverse=True)
    selected: List[str] = []
    seen = set()
    for _key, object_id in ranked_pairs:
        if object_id in seen:
            continue
        selected.append(object_id)
        seen.add(object_id)
        if len(selected) >= top_k:
            break
    return selected


def _record_candidate_rank_hit(
    candidate_stats: Dict[str, Any],
    *,
    hit_key: str,
    rank_key: str,
    hit_id: str,
    rank_idx: int,
) -> None:
    candidate_stats[hit_key].add(str(hit_id))
    candidate_stats[rank_key] = min(int(candidate_stats[rank_key]), int(rank_idx))


def _extract_role_text(record: Mapping[str, Any], role_name: str) -> str:
    participant_ids = [str(value) for value in record.get("participant_ids", [])]
    participant_texts = [_normalize_text(value) for value in record.get("participant_texts", [])]
    participant_roles = {
        str(participant_id): [str(role).strip().lower() for role in roles]
        for participant_id, roles in dict(record.get("participant_roles", {})).items()
    }
    participant_id_to_text = {
        participant_id: participant_texts[idx] if idx < len(participant_texts) else ""
        for idx, participant_id in enumerate(participant_ids)
    }
    for participant_id in participant_ids:
        if role_name in participant_roles.get(participant_id, []):
            participant_text = participant_id_to_text.get(participant_id, "")
            if participant_text:
                return participant_text
    if len(participant_texts) >= 2:
        if role_name == "subject":
            return participant_texts[0]
        if role_name == "object":
            return participant_texts[1]
    return ""


def _extract_bridge_signature(record: Mapping[str, Any]) -> Dict[str, Any]:
    relation = _normalize_text(record.get("relation_type", ""))
    subject = _extract_role_text(record, "subject")
    obj = _extract_role_text(record, "object")
    pair_signatures: List[Tuple[str, str, str]] = []
    subj_endpoint_signatures: List[Tuple[str, str]] = []
    obj_endpoint_signatures: List[Tuple[str, str]] = []
    if subject and obj and relation and relation != "proposition":
        pair_signatures.append((subject, relation, obj))
        subj_endpoint_signatures.append((subject, relation))
        obj_endpoint_signatures.append((obj, relation))
    participant_texts = _normalize_id_list(
        [_normalize_text(value) for value in record.get("participant_texts", []) if _normalize_text(value)]
    )
    return {
        "relation_type": relation,
        "subject_text": subject,
        "object_text": obj,
        "pair_signatures": pair_signatures,
        "subj_endpoint_signatures": subj_endpoint_signatures,
        "obj_endpoint_signatures": obj_endpoint_signatures,
        "participant_texts": participant_texts,
    }


def _slot_id_from_signature(signature: Sequence[str]) -> str:
    normalized_signature = tuple(_normalize_text(value) for value in signature)
    return "slot::" + json.dumps(list(normalized_signature), ensure_ascii=True, separators=(",", ":"))


def _build_substrate_version(
    *,
    entity_count: int,
    fact_count: int,
    bridge_count: int,
    bridge_path_count: int,
    slot_count: int,
    passage_count: int,
    graph_nodes: int,
    graph_edges: int,
    sidecar_version: int,
    passage_inventory_count: int,
    pair_signature_count: int,
    source_support_slot_count: int,
    anchor_source_support_slot_count: int,
    anchor_support_slot_count: int,
    target_support_slot_count: int,
    answer_support_slot_count: int,
) -> str:
    payload = {
        "entity_count": int(entity_count),
        "fact_count": int(fact_count),
        "bridge_count": int(bridge_count),
        "bridge_path_count": int(bridge_path_count),
        "slot_count": int(slot_count),
        "passage_count": int(passage_count),
        "graph_nodes": int(graph_nodes),
        "graph_edges": int(graph_edges),
        "sidecar_version": int(sidecar_version),
        "passage_inventory_count": int(passage_inventory_count),
        "pair_signature_count": int(pair_signature_count),
        "source_support_slot_count": int(source_support_slot_count),
        "anchor_source_support_slot_count": int(anchor_source_support_slot_count),
        "anchor_support_slot_count": int(anchor_support_slot_count),
        "target_support_slot_count": int(target_support_slot_count),
        "answer_support_slot_count": int(answer_support_slot_count),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"hyperhippo-memory-compat::{digest}"


def _coerce_record_mapping(
    records: Any,
    *,
    id_field: str = "hash_id",
    fallback_id_field: Optional[str] = None,
    copy_payloads: bool = True,
) -> Dict[str, Dict[str, Any]]:
    if records is None:
        return {}
    if isinstance(records, Mapping):
        normalized_records: Dict[str, Dict[str, Any]] = {}
        for record_id, payload in records.items():
            if not isinstance(payload, Mapping):
                continue
            if copy_payloads:
                normalized_records[str(record_id)] = dict(payload)
            else:
                normalized_records[str(record_id)] = payload if isinstance(payload, dict) else dict(payload)
        return normalized_records

    normalized_records: Dict[str, Dict[str, Any]] = {}
    for payload in records:
        if not isinstance(payload, Mapping):
            continue
        record_id = str(payload.get(id_field) or payload.get(fallback_id_field or "") or "")
        if not record_id:
            continue
        normalized_records[record_id] = dict(payload) if copy_payloads else (
            payload if isinstance(payload, dict) else dict(payload)
        )
    return normalized_records


def _load_runtime_record_mapping(
    runtime: Any,
    *,
    runtime_attr: str,
    store_attr: str,
    copy_payloads: bool = True,
) -> Dict[str, Dict[str, Any]]:
    runtime_records = getattr(runtime, runtime_attr, None)
    if runtime_records is not None:
        return _coerce_record_mapping(runtime_records, copy_payloads=copy_payloads)

    store = getattr(runtime, store_attr, None)
    if store is None:
        return {}
    if hasattr(store, "get_all_ref"):
        return _coerce_record_mapping(store.get_all_ref(), copy_payloads=copy_payloads)
    if hasattr(store, "get_all"):
        return _coerce_record_mapping(store.get_all(), copy_payloads=copy_payloads)
    return {}


def _load_runtime_support_signature_sidecar(
    runtime: Any,
    *,
    hyperedge_records: Mapping[str, Dict[str, Any]],
    passage_records: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    sidecar = getattr(runtime, "support_signature_sidecar", None) or {}
    if sidecar:
        return sidecar

    loader = getattr(runtime, "_load_support_signature_sidecar", None)
    if callable(loader):
        try:
            loaded_sidecar = loader()
        except Exception:
            loaded_sidecar = {}
        sidecar = loaded_sidecar or {}
        if sidecar:
            return sidecar

    builder = getattr(runtime, "_build_support_signature_sidecar", None)
    if callable(builder) and (hyperedge_records or passage_records):
        try:
            built_sidecar = builder()
        except Exception:
            built_sidecar = {}
        sidecar = built_sidecar or {}
    return sidecar


def _load_runtime_slot_records(
    runtime: Any,
    *,
    hyperedge_records: Mapping[str, Dict[str, Any]],
    support_signature_sidecar: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    runtime_slot_records = getattr(runtime, "slot_records", None)
    if runtime_slot_records is not None:
        return _coerce_record_mapping(runtime_slot_records, copy_payloads=False)

    slot_record_store = getattr(runtime, "slot_record_store", None)
    if slot_record_store is not None:
        if hasattr(slot_record_store, "get_all_ref"):
            return _coerce_record_mapping(slot_record_store.get_all_ref(), copy_payloads=False)
        if hasattr(slot_record_store, "get_all"):
            return _coerce_record_mapping(slot_record_store.get_all(), copy_payloads=False)

    builder = getattr(runtime, "_build_slot_records", None)
    if callable(builder) and hyperedge_records:
        try:
            built_slot_records = builder(dict(support_signature_sidecar or {}))
        except Exception:
            built_slot_records = []
        return _coerce_record_mapping(
            built_slot_records,
            id_field="hash_id",
            fallback_id_field="slot_id",
            copy_payloads=False,
        )

    return {}


def _load_runtime_bridge_path_records(
    runtime: Any,
    *,
    hyperedge_records: Mapping[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    runtime_records = getattr(runtime, "bridge_path_records", None)
    if runtime_records is not None:
        return _coerce_record_mapping(runtime_records, copy_payloads=False)

    store = getattr(runtime, "bridge_path_record_store", None)
    if store is not None:
        if hasattr(store, "get_all_ref"):
            return _coerce_record_mapping(store.get_all_ref(), copy_payloads=False)
        if hasattr(store, "get_all"):
            return _coerce_record_mapping(store.get_all(), copy_payloads=False)

    builder = getattr(runtime, "_build_bridge_path_records", None)
    if callable(builder) and hyperedge_records:
        try:
            built_bridge_path_records = builder()
        except Exception:
            built_bridge_path_records = []
        return _coerce_record_mapping(
            built_bridge_path_records,
            id_field="hash_id",
            fallback_id_field="path_id",
            copy_payloads=False,
        )

    if hyperedge_records:
        return _coerce_record_mapping(
            build_bridge_path_records_from_hyperedge_records(hyperedge_records),
            id_field="hash_id",
            fallback_id_field="path_id",
            copy_payloads=False,
        )

    return {}


def _load_runtime_support_trail_records(runtime: Any) -> Dict[str, Dict[str, Any]]:
    for attr_name in ("support_trail_records", "bridge_support_trail_records"):
        runtime_records = getattr(runtime, attr_name, None)
        if runtime_records is not None:
            return _coerce_record_mapping(runtime_records, copy_payloads=False)

    for store_attr in ("support_trail_record_store", "bridge_support_trail_record_store"):
        store = getattr(runtime, store_attr, None)
        if store is None:
            continue
        if hasattr(store, "get_all_ref"):
            return _coerce_record_mapping(store.get_all_ref(), copy_payloads=False)
        if hasattr(store, "get_all"):
            return _coerce_record_mapping(store.get_all(), copy_payloads=False)

    for builder_attr in ("_build_bridge_support_trail_records", "_build_support_trail_records"):
        builder = getattr(runtime, builder_attr, None)
        if callable(builder):
            try:
                built_records = builder()
            except Exception:
                built_records = []
            return _coerce_record_mapping(
                built_records,
                id_field="hash_id",
                fallback_id_field="trail_id",
                copy_payloads=False,
            )

    return {}


def _load_runtime_support_trail_membership_records(runtime: Any) -> Dict[str, Dict[str, Any]]:
    for attr_name in ("support_trail_membership_records", "bridge_support_trail_membership_records"):
        runtime_records = getattr(runtime, attr_name, None)
        if runtime_records is not None:
            return _coerce_record_mapping(runtime_records, copy_payloads=False)

    for store_attr in ("support_trail_membership_record_store", "bridge_support_trail_membership_record_store"):
        store = getattr(runtime, store_attr, None)
        if store is None:
            continue
        if hasattr(store, "get_all_ref"):
            return _coerce_record_mapping(store.get_all_ref(), copy_payloads=False)
        if hasattr(store, "get_all"):
            return _coerce_record_mapping(store.get_all(), copy_payloads=False)

    for builder_attr in ("_build_bridge_support_trail_membership_records", "_build_support_trail_membership_records"):
        builder = getattr(runtime, builder_attr, None)
        if callable(builder):
            try:
                built_records = builder()
            except Exception:
                built_records = []
            return _coerce_record_mapping(
                built_records,
                id_field="hash_id",
                fallback_id_field="membership_id",
                copy_payloads=False,
            )

    return {}


def _load_runtime_support_endpoint_records(runtime: Any) -> Dict[str, Dict[str, Any]]:
    for attr_name in ("support_endpoint_records", "bridge_support_endpoint_records"):
        runtime_records = getattr(runtime, attr_name, None)
        if runtime_records is not None:
            return _coerce_record_mapping(runtime_records, copy_payloads=False)

    for store_attr in ("support_endpoint_record_store", "bridge_support_endpoint_record_store"):
        store = getattr(runtime, store_attr, None)
        if store is None:
            continue
        if hasattr(store, "get_all_ref"):
            return _coerce_record_mapping(store.get_all_ref(), copy_payloads=False)
        if hasattr(store, "get_all"):
            return _coerce_record_mapping(store.get_all(), copy_payloads=False)

    for builder_attr in ("_build_bridge_support_endpoint_records", "_build_support_endpoint_records"):
        builder = getattr(runtime, builder_attr, None)
        if callable(builder):
            try:
                built_records = builder()
            except Exception:
                built_records = []
            return _coerce_record_mapping(
                built_records,
                id_field="hash_id",
                fallback_id_field="endpoint_id",
                copy_payloads=False,
            )

    return {}


@dataclass
class QueryActivationState:
    query_text: str
    query_entity_ids: List[str]
    query_entity_scores: List[float]
    query_fact_ids: List[str]
    query_fact_scores: List[float]
    query_bridge_hints: List[str]
    query_passage_ids: List[str]
    query_passage_scores: List[float]
    activation_mode: str
    substrate_version: str


@dataclass
class CandidateSet:
    channel: str
    candidate_passage_ids: List[str]
    candidate_fact_ids: List[str]
    candidate_bridge_ids: List[str]
    evidence_index_refs: List[Dict[str, Any]]
    substrate_version: str


@dataclass
class FactSlotDemand:
    fact_object_id: str
    all_slot_ids: List[str]
    exact_covered_slot_ids: List[str]
    soft_covered_slot_ids: List[str]
    covered_slot_ids: List[str]
    uncovered_slot_ids: List[str]
    covered_bridge_ids: List[str]
    uncovered_bridge_ids: List[str]


@dataclass
class DeficitFactState:
    deficit_fact_ids: List[str]
    deficit_weights: List[float]
    covered_fact_ids: List[str]
    fact_slot_demands: List[FactSlotDemand]
    fully_covered_fact_ids: List[str]
    partially_covered_fact_ids: List[str]
    exact_covered_slot_ids: List[str]
    soft_covered_slot_ids: List[str]
    covered_slot_ids: List[str]
    uncovered_slot_ids: List[str]
    covered_bridge_ids: List[str]
    uncovered_bridge_ids: List[str]
    covered_participant_entity_ids: List[str]
    uncovered_participant_entity_ids: List[str]
    evidence_index_refs: List[Dict[str, Any]]
    substrate_version: str


@dataclass
class LocalSubgraph:
    node_ids: Dict[str, List[str]]
    edge_rows: List[Dict[str, Any]]
    passage_inventories: Dict[str, Dict[str, Any]]
    fact_to_passages: Dict[str, List[str]]
    bridge_neighbors: Dict[str, List[str]]
    evidence_index_refs: List[Dict[str, Any]]
    substrate_version: str


@dataclass
class CoverageGraphView:
    entity_ids: List[str]
    fact_object_ids: List[str]
    bridge_ids: List[str]
    passage_ids: List[str]
    entity_to_passages: Dict[str, List[str]]
    fact_to_passages: Dict[str, List[str]]
    fact_to_bridge_ids: Dict[str, List[str]]
    fact_to_slot_ids: Dict[str, List[str]]
    passage_to_entities: Dict[str, List[str]]
    passage_to_fact_ids: Dict[str, List[str]]
    passage_to_slot_ids: Dict[str, List[str]]
    substrate_version: str


@dataclass
class SupportGraphView:
    fact_objects: Dict[str, Dict[str, Any]]
    bridge_objects: Dict[str, Dict[str, Any]]
    slot_objects: Dict[str, Dict[str, Any]]
    passage_inventories: Dict[str, Dict[str, Any]]
    pair_to_passages: Dict[Tuple[str, str, str], List[str]]
    subj_endpoint_to_passages: Dict[Tuple[str, str], List[str]]
    obj_endpoint_to_passages: Dict[Tuple[str, str], List[str]]
    slot_to_passages: Dict[str, List[str]]
    subject_entity_to_passages: Dict[str, List[str]]
    object_entity_to_passages: Dict[str, List[str]]
    fact_to_passages: Dict[str, List[str]]
    fact_to_slot_ids: Dict[str, List[str]]
    passage_to_slot_ids: Dict[str, List[str]]
    bridge_to_passages: Dict[str, List[str]]
    slot_to_exact_passages: Dict[str, List[str]]
    slot_to_endpoint_passages: Dict[str, List[str]]
    slot_to_entity_passages: Dict[str, List[str]]
    slot_to_source_support_passages: Dict[str, List[str]]
    slot_to_anchor_source_support_passages: Dict[str, List[str]]
    slot_to_anchor_support_passages: Dict[str, List[str]]
    slot_to_target_support_passages: Dict[str, List[str]]
    slot_to_answer_support_passages: Dict[str, List[str]]
    substrate_version: str


@dataclass
class RouteGraphView:
    fact_to_bridge_ids: Dict[str, List[str]]
    fact_to_slot_ids: Dict[str, List[str]]
    fact_to_bridge_path_ids: Dict[str, List[str]]
    bridge_to_fact_id: Dict[str, str]
    bridge_to_slot_ids: Dict[str, List[str]]
    bridge_neighbors: Dict[str, List[str]]
    slot_to_bridge_ids: Dict[str, List[str]]
    slot_to_passages: Dict[str, List[str]]
    bridge_to_passages: Dict[str, List[str]]
    bridge_path_to_passages: Dict[str, List[str]]
    bridge_path_to_facts: Dict[str, List[str]]
    slot_to_source_support_passages: Dict[str, List[str]]
    slot_to_anchor_source_support_passages: Dict[str, List[str]]
    slot_to_anchor_support_passages: Dict[str, List[str]]
    slot_to_target_support_passages: Dict[str, List[str]]
    slot_to_answer_support_passages: Dict[str, List[str]]
    substrate_version: str


@dataclass
class BridgePathGraphView:
    bridge_path_objects: Dict[str, Dict[str, Any]]
    fact_to_bridge_path_ids: Dict[str, List[str]]
    passage_to_bridge_path_ids: Dict[str, List[str]]
    entity_to_bridge_path_ids: Dict[str, List[str]]
    bridge_path_to_passages: Dict[str, List[str]]
    bridge_path_to_facts: Dict[str, List[str]]
    substrate_version: str


@dataclass
class BridgeSupportTrailGraphView:
    support_trail_objects: Dict[str, Dict[str, Any]]
    support_trail_membership_objects: Dict[str, Dict[str, Any]]
    fact_to_support_trail_membership: Dict[str, Dict[str, Any]]
    fact_to_support_trail_ids: Dict[str, List[str]]
    passage_to_support_trail_ids: Dict[str, List[str]]
    support_trail_to_passages: Dict[str, List[str]]
    support_trail_to_facts: Dict[str, List[str]]
    support_trail_to_bridge_path_ids: Dict[str, List[str]]
    substrate_version: str


@dataclass
class BridgeSupportEndpointGraphView:
    support_endpoint_objects: Dict[str, Dict[str, Any]]
    fact_to_support_endpoint_ids: Dict[str, List[str]]
    endpoint_fact_to_support_endpoint_ids: Dict[str, List[str]]
    passage_to_support_endpoint_ids: Dict[str, List[str]]
    support_endpoint_to_passages: Dict[str, List[str]]
    support_endpoint_to_facts: Dict[str, List[str]]
    support_endpoint_to_trail_ids: Dict[str, List[str]]
    support_endpoint_to_bridge_path_ids: Dict[str, List[str]]
    substrate_version: str


class CanonicalRetrievalInterface:
    def __init__(
        self,
        *,
        substrate_version: str,
        entity_payloads: Dict[str, Dict[str, Any]],
        fact_payloads: Dict[str, Dict[str, Any]],
        bridge_payloads: Dict[str, Dict[str, Any]],
        slot_payloads: Dict[str, Dict[str, Any]],
        bridge_path_payloads: Dict[str, Dict[str, Any]],
        passage_payloads: Dict[str, Dict[str, Any]],
        passage_inventories: Dict[str, Dict[str, Any]],
        entity_to_passages: Dict[str, List[str]],
        entity_to_bridges: Dict[str, List[str]],
        entity_to_bridge_path_ids: Dict[str, List[str]],
        fact_to_passages: Dict[str, List[str]],
        fact_to_bridge_ids: Dict[str, List[str]],
        fact_to_slot_ids: Dict[str, List[str]],
        fact_to_bridge_path_ids: Dict[str, List[str]],
        bridge_to_passages: Dict[str, List[str]],
        bridge_to_fact_id: Dict[str, str],
        bridge_to_slot_ids: Dict[str, List[str]],
        bridge_neighbors: Dict[str, List[str]],
        bridge_path_to_passages: Dict[str, List[str]],
        bridge_path_to_facts: Dict[str, List[str]],
        slot_to_passages: Dict[str, List[str]],
        slot_to_bridge_ids: Dict[str, List[str]],
        passage_to_bridge_path_ids: Dict[str, List[str]],
        pair_to_passages: Dict[Tuple[str, str, str], List[str]],
        subj_endpoint_to_passages: Dict[Tuple[str, str], List[str]],
        obj_endpoint_to_passages: Dict[Tuple[str, str], List[str]],
        subject_entity_to_passages: Dict[str, List[str]],
        object_entity_to_passages: Dict[str, List[str]],
        slot_to_exact_passages: Optional[Dict[str, List[str]]] = None,
        slot_to_endpoint_passages: Optional[Dict[str, List[str]]] = None,
        slot_to_entity_passages: Optional[Dict[str, List[str]]] = None,
        slot_to_support_passages: Optional[Dict[str, List[str]]] = None,
        slot_to_source_support_passages: Optional[Dict[str, List[str]]] = None,
        slot_to_anchor_source_support_passages: Optional[Dict[str, List[str]]] = None,
        slot_to_anchor_support_passages: Optional[Dict[str, List[str]]] = None,
        slot_to_target_support_passages: Optional[Dict[str, List[str]]] = None,
        slot_to_answer_support_passages: Optional[Dict[str, List[str]]] = None,
        passage_to_exact_slot_ids: Optional[Dict[str, List[str]]] = None,
        support_trail_payloads: Optional[Dict[str, Dict[str, Any]]] = None,
        support_trail_membership_payloads: Optional[Dict[str, Dict[str, Any]]] = None,
        fact_to_support_trail_membership: Optional[Dict[str, Dict[str, Any]]] = None,
        fact_to_support_trail_ids: Optional[Dict[str, List[str]]] = None,
        passage_to_support_trail_ids: Optional[Dict[str, List[str]]] = None,
        support_trail_to_passages: Optional[Dict[str, List[str]]] = None,
        support_trail_to_facts: Optional[Dict[str, List[str]]] = None,
        support_trail_to_bridge_path_ids: Optional[Dict[str, List[str]]] = None,
        support_endpoint_payloads: Optional[Dict[str, Dict[str, Any]]] = None,
        fact_to_support_endpoint_ids: Optional[Dict[str, List[str]]] = None,
        endpoint_fact_to_support_endpoint_ids: Optional[Dict[str, List[str]]] = None,
        passage_to_support_endpoint_ids: Optional[Dict[str, List[str]]] = None,
        support_endpoint_to_passages: Optional[Dict[str, List[str]]] = None,
        support_endpoint_to_facts: Optional[Dict[str, List[str]]] = None,
        support_endpoint_to_trail_ids: Optional[Dict[str, List[str]]] = None,
        support_endpoint_to_bridge_path_ids: Optional[Dict[str, List[str]]] = None,
        runtime: Any | None = None,
    ) -> None:
        self.substrate_version = substrate_version
        self.entity_payloads = entity_payloads
        self.fact_payloads = fact_payloads
        self.bridge_payloads = bridge_payloads
        self.slot_payloads = slot_payloads
        self.bridge_path_payloads = bridge_path_payloads
        self.passage_payloads = passage_payloads
        self.passage_inventories = passage_inventories
        self.entity_to_passages = entity_to_passages
        self.entity_to_bridges = entity_to_bridges
        self.entity_to_bridge_path_ids = entity_to_bridge_path_ids
        self.fact_to_passages = fact_to_passages
        self.fact_to_bridge_ids = fact_to_bridge_ids
        self.fact_to_slot_ids = fact_to_slot_ids
        self.fact_to_bridge_path_ids = fact_to_bridge_path_ids
        self.bridge_to_passages = bridge_to_passages
        self.bridge_to_fact_id = bridge_to_fact_id
        self.bridge_to_slot_ids = bridge_to_slot_ids
        self.bridge_neighbors = bridge_neighbors
        self.bridge_path_to_passages = bridge_path_to_passages
        self.bridge_path_to_facts = bridge_path_to_facts
        self.slot_to_passages = slot_to_passages
        self.slot_to_bridge_ids = slot_to_bridge_ids
        self.passage_to_bridge_path_ids = passage_to_bridge_path_ids
        self.support_trail_payloads = support_trail_payloads or {}
        self.support_trail_membership_payloads = support_trail_membership_payloads or {}
        self.fact_to_support_trail_membership = fact_to_support_trail_membership or {}
        self.fact_to_support_trail_ids = fact_to_support_trail_ids or {}
        self.passage_to_support_trail_ids = passage_to_support_trail_ids or {}
        self.support_trail_to_passages = support_trail_to_passages or {}
        self.support_trail_to_facts = support_trail_to_facts or {}
        self.support_trail_to_bridge_path_ids = support_trail_to_bridge_path_ids or {}
        self.support_endpoint_payloads = support_endpoint_payloads or {}
        self.fact_to_support_endpoint_ids = fact_to_support_endpoint_ids or {}
        self.endpoint_fact_to_support_endpoint_ids = endpoint_fact_to_support_endpoint_ids or {}
        self.passage_to_support_endpoint_ids = passage_to_support_endpoint_ids or {}
        self.support_endpoint_to_passages = support_endpoint_to_passages or {}
        self.support_endpoint_to_facts = support_endpoint_to_facts or {}
        self.support_endpoint_to_trail_ids = support_endpoint_to_trail_ids or {}
        self.support_endpoint_to_bridge_path_ids = support_endpoint_to_bridge_path_ids or {}
        self.pair_to_passages = pair_to_passages
        self.subj_endpoint_to_passages = subj_endpoint_to_passages
        self.obj_endpoint_to_passages = obj_endpoint_to_passages
        self.subject_entity_to_passages = subject_entity_to_passages
        self.object_entity_to_passages = object_entity_to_passages
        self.runtime = runtime

        self.entity_ids = sorted(entity_payloads.keys())
        self.fact_object_ids = sorted(fact_payloads.keys())
        self.bridge_ids = sorted(bridge_payloads.keys())
        self.slot_ids = sorted(slot_payloads.keys())
        self.bridge_path_ids = sorted(bridge_path_payloads.keys())
        self.support_trail_ids = sorted(self.support_trail_payloads.keys())
        self.support_trail_membership_ids = sorted(self.support_trail_membership_payloads.keys())
        self.support_endpoint_ids = sorted(self.support_endpoint_payloads.keys())
        self.passage_ids = sorted(passage_payloads.keys())
        self.slot_to_exact_passages = (
            slot_to_exact_passages
            if slot_to_exact_passages is not None
            else {
                slot_id: list(payload.get("exact_support_passage_ids", payload.get("source_ids", [])))
                for slot_id, payload in slot_payloads.items()
            }
        )
        self.slot_to_endpoint_passages = (
            slot_to_endpoint_passages
            if slot_to_endpoint_passages is not None
            else {
                slot_id: list(payload.get("endpoint_support_passage_ids", []))
                for slot_id, payload in slot_payloads.items()
            }
        )
        self.slot_to_entity_passages = (
            slot_to_entity_passages
            if slot_to_entity_passages is not None
            else {
                slot_id: list(payload.get("entity_support_passage_ids", []))
                for slot_id, payload in slot_payloads.items()
            }
        )
        self.slot_to_support_passages = (
            slot_to_support_passages
            if slot_to_support_passages is not None
            else {
                slot_id: list(payload.get("supporting_passage_ids", payload.get("source_ids", [])))
                for slot_id, payload in slot_payloads.items()
            }
        )
        self.slot_to_source_support_passages = (
            slot_to_source_support_passages
            if slot_to_source_support_passages is not None
            else {
                slot_id: list(payload.get("source_support_passage_ids", payload.get("source_ids", [])))
                for slot_id, payload in slot_payloads.items()
            }
        )
        self.slot_to_anchor_source_support_passages = (
            slot_to_anchor_source_support_passages
            if slot_to_anchor_source_support_passages is not None
            else {
                slot_id: list(
                    payload.get(
                        "anchor_source_support_passage_ids",
                        payload.get("source_support_passage_ids", payload.get("source_ids", [])),
                    )
                )
                for slot_id, payload in slot_payloads.items()
            }
        )
        self.slot_to_anchor_support_passages = (
            slot_to_anchor_support_passages
            if slot_to_anchor_support_passages is not None
            else {
                slot_id: list(payload.get("anchor_support_passage_ids", payload.get("entity_support_passage_ids", [])))
                for slot_id, payload in slot_payloads.items()
            }
        )
        self.slot_to_target_support_passages = (
            slot_to_target_support_passages
            if slot_to_target_support_passages is not None
            else {
                slot_id: list(payload.get("target_support_passage_ids", []))
                for slot_id, payload in slot_payloads.items()
            }
        )
        self.slot_to_answer_support_passages = (
            slot_to_answer_support_passages
            if slot_to_answer_support_passages is not None
            else {
                slot_id: list(payload.get("answer_support_passage_ids", payload.get("target_support_passage_ids", [])))
                for slot_id, payload in slot_payloads.items()
            }
        )
        self.passage_to_entities = {
            passage_id: list(payload.get("entity_ids", []))
            for passage_id, payload in passage_payloads.items()
        }
        self.passage_to_bridge_ids = {
            passage_id: list(payload.get("bridge_ids", []))
            for passage_id, payload in passage_payloads.items()
        }
        self.passage_to_fact_ids = {
            passage_id: list(payload.get("fact_object_ids", []))
            for passage_id, payload in passage_payloads.items()
        }
        self.passage_to_slot_ids = {
            passage_id: list(payload.get("slot_ids", []))
            for passage_id, payload in passage_payloads.items()
        }
        if passage_to_exact_slot_ids is None:
            computed_passage_to_exact_slot_ids: Dict[str, List[str]] = defaultdict(list)
            for slot_id, passage_ids in self.slot_to_exact_passages.items():
                for passage_id in passage_ids:
                    normalized_passage_id = str(passage_id)
                    if slot_id not in computed_passage_to_exact_slot_ids[normalized_passage_id]:
                        computed_passage_to_exact_slot_ids[normalized_passage_id].append(slot_id)
            self.passage_to_exact_slot_ids = {
                passage_id: sorted(slot_ids)
                for passage_id, slot_ids in computed_passage_to_exact_slot_ids.items()
            }
        else:
            self.passage_to_exact_slot_ids = passage_to_exact_slot_ids
        self._fact_slot_tier_support_lookup: Optional[Dict[str, Dict[str, List[str]]]] = None

    def _build_fact_slot_tier_support_lookup(self) -> Dict[str, Dict[str, List[str]]]:
        fact_support_lookup: Dict[str, Dict[str, List[str]]] = {}
        for fact_object_id, payload in self.fact_payloads.items():
            normalized_fact_id = str(fact_object_id)
            slot_ids = list(self.fact_to_slot_ids.get(normalized_fact_id, []))
            answer_passages: List[str] = []
            anchor_source_passages: List[str] = []
            source_slot_passages: List[str] = []
            target_passages: List[str] = []
            exact_passages: List[str] = []
            anchor_passages: List[str] = []
            for slot_id in slot_ids:
                normalized_slot_id = str(slot_id)
                answer_passages.extend(self.slot_to_answer_support_passages.get(normalized_slot_id, []))
                anchor_source_passages.extend(self.slot_to_anchor_source_support_passages.get(normalized_slot_id, []))
                source_slot_passages.extend(self.slot_to_source_support_passages.get(normalized_slot_id, []))
                target_passages.extend(self.slot_to_target_support_passages.get(normalized_slot_id, []))
                exact_passages.extend(self.slot_to_exact_passages.get(normalized_slot_id, []))
                anchor_passages.extend(self.slot_to_anchor_support_passages.get(normalized_slot_id, []))

            pair_passages: List[str] = []
            for signature in payload.get("pair_signatures", []):
                pair_passages.extend(self.pair_to_passages.get(tuple(signature), []))

            endpoint_passages: List[str] = []
            for signature in payload.get("subj_endpoint_signatures", []):
                endpoint_passages.extend(self.subj_endpoint_to_passages.get(tuple(signature), []))
            for signature in payload.get("obj_endpoint_signatures", []):
                endpoint_passages.extend(self.obj_endpoint_to_passages.get(tuple(signature), []))

            entity_passages: List[str] = []
            for participant_text in payload.get("participant_texts", []):
                normalized_participant_text = str(participant_text)
                entity_passages.extend(self.subject_entity_to_passages.get(normalized_participant_text, []))
                entity_passages.extend(self.object_entity_to_passages.get(normalized_participant_text, []))

            fact_support_lookup[normalized_fact_id] = {
                "answer": _unique_in_order(answer_passages),
                "anchor_source": _unique_in_order(anchor_source_passages),
                "source_slot": _unique_in_order(source_slot_passages),
                "target": _unique_in_order(target_passages),
                "exact": _unique_in_order(exact_passages),
                "anchor": _unique_in_order(anchor_passages),
                "pair": _unique_in_order(pair_passages),
                "endpoint": _unique_in_order(endpoint_passages),
                "entity": _unique_in_order(entity_passages),
                "source": _unique_in_order(payload.get("source_ids", [])),
            }
        return fact_support_lookup

    def _ensure_fact_slot_tier_support_lookup(self) -> Dict[str, Dict[str, List[str]]]:
        if self._fact_slot_tier_support_lookup is None:
            self._fact_slot_tier_support_lookup = self._build_fact_slot_tier_support_lookup()
        return self._fact_slot_tier_support_lookup

    @classmethod
    def from_runtime(cls, runtime: Any) -> "CanonicalRetrievalInterface":
        entity_records = _load_runtime_record_mapping(
            runtime,
            runtime_attr="entity_records",
            store_attr="entity_record_store",
            copy_payloads=False,
        )
        hyperedge_records = _load_runtime_record_mapping(
            runtime,
            runtime_attr="hyperedge_records",
            store_attr="hyperedge_record_store",
            copy_payloads=False,
        )
        passage_records = _load_runtime_record_mapping(
            runtime,
            runtime_attr="passage_records",
            store_attr="passage_record_store",
            copy_payloads=False,
        )
        sidecar = _load_runtime_support_signature_sidecar(
            runtime,
            hyperedge_records=hyperedge_records,
            passage_records=passage_records,
        )
        slot_records = _load_runtime_slot_records(
            runtime,
            hyperedge_records=hyperedge_records,
            support_signature_sidecar=sidecar,
        )
        bridge_path_records = _load_runtime_bridge_path_records(
            runtime,
            hyperedge_records=hyperedge_records,
        )
        support_trail_records = _load_runtime_support_trail_records(runtime)
        support_trail_membership_records = _load_runtime_support_trail_membership_records(runtime)
        support_endpoint_records = _load_runtime_support_endpoint_records(runtime)
        passage_inventories = {
            str(passage_id): inventory
            for passage_id, inventory in dict(sidecar.get("passage_inventories", {})).items()
        }

        entity_payloads: Dict[str, Dict[str, Any]] = {}
        for entity_id, record in entity_records.items():
            entity_payloads[str(entity_id)] = {
                "entity_id": str(entity_id),
                "canonical_name": _normalize_text(record.get("canonical_name", entity_id)),
                "aliases": _normalize_id_list(record.get("aliases", [])),
                "source_ids": _normalize_id_list(record.get("source_ids", [])),
            }

        fact_payloads: Dict[str, Dict[str, Any]] = {}
        bridge_payloads: Dict[str, Dict[str, Any]] = {}
        slot_payloads: Dict[str, Dict[str, Any]] = {}
        fact_to_passage_ids: Dict[str, set] = defaultdict(set)
        fact_to_bridge_ids: Dict[str, set] = defaultdict(set)
        fact_to_slot_ids: Dict[str, set] = defaultdict(set)
        bridge_to_passages: Dict[str, List[str]] = {}
        bridge_to_fact_id: Dict[str, str] = {}
        bridge_to_slot_ids: Dict[str, set] = defaultdict(set)
        entity_to_bridges: Dict[str, set] = defaultdict(set)
        relation_to_bridges: Dict[str, set] = defaultdict(set)
        conflict_group_to_bridges: Dict[str, set] = defaultdict(set)
        passage_to_bridge_ids: Dict[str, set] = defaultdict(set)
        passage_to_fact_ids: Dict[str, set] = defaultdict(set)
        passage_to_slot_ids: Dict[str, set] = defaultdict(set)
        slot_to_passages: Dict[str, set] = defaultdict(set)
        slot_to_bridge_ids: Dict[str, set] = defaultdict(set)
        slot_to_exact_passages: Dict[str, List[str]] = {}
        slot_to_endpoint_passages: Dict[str, List[str]] = {}
        slot_to_entity_passages: Dict[str, List[str]] = {}
        slot_to_support_passages: Dict[str, List[str]] = {}
        slot_to_source_support_passages: Dict[str, List[str]] = {}
        slot_to_anchor_source_support_passages: Dict[str, List[str]] = {}
        slot_to_anchor_support_passages: Dict[str, List[str]] = {}
        slot_to_target_support_passages: Dict[str, List[str]] = {}
        slot_to_answer_support_passages: Dict[str, List[str]] = {}
        passage_to_exact_slot_ids: Dict[str, List[str]] = defaultdict(list)
        bridge_path_payloads: Dict[str, Dict[str, Any]] = {}
        fact_to_bridge_path_ids: Dict[str, set] = defaultdict(set)
        entity_to_bridge_path_ids: Dict[str, set] = defaultdict(set)
        passage_to_bridge_path_ids: Dict[str, set] = defaultdict(set)
        bridge_path_to_passages: Dict[str, List[str]] = {}
        bridge_path_to_facts: Dict[str, List[str]] = {}
        support_trail_payloads: Dict[str, Dict[str, Any]] = {}
        support_trail_membership_payloads: Dict[str, Dict[str, Any]] = {}
        fact_to_support_trail_membership: Dict[str, Dict[str, Any]] = {}
        fact_to_support_trail_ids: Dict[str, set] = defaultdict(set)
        passage_to_support_trail_ids: Dict[str, set] = defaultdict(set)
        support_trail_to_passages: Dict[str, List[str]] = {}
        support_trail_to_facts: Dict[str, List[str]] = {}
        support_trail_to_bridge_path_ids: Dict[str, List[str]] = {}
        support_endpoint_payloads: Dict[str, Dict[str, Any]] = {}
        fact_to_support_endpoint_ids: Dict[str, set] = defaultdict(set)
        endpoint_fact_to_support_endpoint_ids: Dict[str, set] = defaultdict(set)
        passage_to_support_endpoint_ids: Dict[str, set] = defaultdict(set)
        support_endpoint_to_passages: Dict[str, List[str]] = {}
        support_endpoint_to_facts: Dict[str, List[str]] = {}
        support_endpoint_to_trail_ids: Dict[str, List[str]] = {}
        support_endpoint_to_bridge_path_ids: Dict[str, List[str]] = {}
        materialize_bridge_scaffold_slots = not bool(slot_records)

        for bridge_id, record in hyperedge_records.items():
            normalized_bridge_id = str(bridge_id)
            fact_object_id = str(record.get("fact_embedding_hash_id") or normalized_bridge_id)
            signature = _extract_bridge_signature(record)
            raw_participant_ids = [str(value) for value in record.get("participant_ids", [])]
            participant_entity_ids = _normalize_id_list(raw_participant_ids)
            source_ids = _normalize_id_list(record.get("source_ids", []))

            bridge_payloads[normalized_bridge_id] = {
                "bridge_id": normalized_bridge_id,
                "fact_object_id": fact_object_id,
                "relation_type": signature["relation_type"],
                "participant_entity_ids": participant_entity_ids,
                "participant_texts": signature["participant_texts"],
                "source_ids": source_ids,
                "pair_signatures": list(signature["pair_signatures"]),
                "subj_endpoint_signatures": list(signature["subj_endpoint_signatures"]),
                "obj_endpoint_signatures": list(signature["obj_endpoint_signatures"]),
                "summary_text": _normalize_text(record.get("summary_text", "")),
                "conflict_group": str(record.get("conflict_group", "")),
            }
            bridge_to_fact_id[normalized_bridge_id] = fact_object_id
            bridge_to_passages[normalized_bridge_id] = source_ids
            fact_to_bridge_ids[fact_object_id].add(normalized_bridge_id)
            fact_to_passage_ids[fact_object_id].update(source_ids)

            fact_payload = fact_payloads.setdefault(
                fact_object_id,
                {
                    "fact_object_id": fact_object_id,
                    "bridge_ids": set(),
                    "slot_ids": set(),
                    "relation_types": set(),
                    "participant_entity_ids": set(),
                    "participant_texts": set(),
                    "source_ids": set(),
                    "pair_signatures": set(),
                    "subj_endpoint_signatures": set(),
                    "obj_endpoint_signatures": set(),
                },
            )
            fact_payload["bridge_ids"].add(normalized_bridge_id)
            if signature["relation_type"]:
                fact_payload["relation_types"].add(signature["relation_type"])
            fact_payload["participant_entity_ids"].update(participant_entity_ids)
            fact_payload["participant_texts"].update(signature["participant_texts"])
            fact_payload["source_ids"].update(source_ids)
            fact_payload["pair_signatures"].update(signature["pair_signatures"])
            fact_payload["subj_endpoint_signatures"].update(signature["subj_endpoint_signatures"])
            fact_payload["obj_endpoint_signatures"].update(signature["obj_endpoint_signatures"])

            if materialize_bridge_scaffold_slots:
                participant_roles = {
                    str(participant_id): [str(role).strip().lower() for role in roles]
                    for participant_id, roles in dict(record.get("participant_roles", {})).items()
                }
                participant_texts = [_normalize_text(value) for value in record.get("participant_texts", [])]
                for idx, entity_id in enumerate(raw_participant_ids):
                    entity_text = participant_texts[idx] if idx < len(participant_texts) else ""
                    if not entity_text:
                        continue
                    role_names = list(participant_roles.get(str(entity_id), [])) + ["participant"]
                    for role_name in role_names:
                        slot_signature = (role_name, signature["relation_type"], entity_text)
                        slot_id = _slot_id_from_signature(slot_signature)
                        slot_payload = slot_payloads.setdefault(
                            slot_id,
                            {
                                "slot_id": slot_id,
                                "slot_kind": str(role_name),
                                "relation_type": signature["relation_type"],
                                "entity_text": entity_text,
                                "entity_ids": set(),
                                "bridge_ids": set(),
                                "fact_object_ids": set(),
                                "source_ids": set(),
                            },
                        )
                        slot_payload["entity_ids"].add(str(entity_id))
                        slot_payload["bridge_ids"].add(normalized_bridge_id)
                        slot_payload["fact_object_ids"].add(fact_object_id)
                        slot_payload["source_ids"].update(source_ids)
                        fact_payload["slot_ids"].add(slot_id)
                        fact_to_slot_ids[fact_object_id].add(slot_id)
                        bridge_to_slot_ids[normalized_bridge_id].add(slot_id)
                        slot_to_bridge_ids[slot_id].add(normalized_bridge_id)
                        for passage_id in source_ids:
                            slot_to_passages[slot_id].add(passage_id)
                            passage_to_slot_ids[passage_id].add(slot_id)

            for entity_id in participant_entity_ids:
                entity_to_bridges[entity_id].add(normalized_bridge_id)
            if signature["relation_type"]:
                relation_to_bridges[signature["relation_type"]].add(normalized_bridge_id)
            conflict_group = str(record.get("conflict_group", "")).strip()
            if conflict_group:
                conflict_group_to_bridges[conflict_group].add(normalized_bridge_id)
            for passage_id in source_ids:
                passage_to_bridge_ids[passage_id].add(normalized_bridge_id)
                passage_to_fact_ids[passage_id].add(fact_object_id)

        for bridge_path_id, record in bridge_path_records.items():
            normalized_path_id = str(record.get("path_id") or record.get("hash_id") or bridge_path_id)
            start_fact_id = str(record.get("start_fact_id", "")).strip()
            end_fact_id = str(record.get("end_fact_id", "")).strip()
            start_bridge_id = str(record.get("start_bridge_id", "")).strip()
            end_bridge_id = str(record.get("end_bridge_id", "")).strip()
            source_passage_ids = _unique_in_order(
                list(record.get("start_source_passage_ids", []))
                + list(record.get("end_source_passage_ids", []))
                + list(record.get("shared_source_passage_ids", []))
            )
            fact_ids = _unique_in_order([start_fact_id, end_fact_id])
            shared_entity_ids = _normalize_id_list(record.get("shared_entity_ids", []))
            bridge_path_payloads[normalized_path_id] = {
                "bridge_path_id": normalized_path_id,
                "path_id": normalized_path_id,
                "record_version": int(record.get("record_version", 0)),
                "start_bridge_id": start_bridge_id,
                "end_bridge_id": end_bridge_id,
                "start_fact_id": start_fact_id,
                "end_fact_id": end_fact_id,
                "shared_entity_ids": shared_entity_ids,
                "shared_entity_texts": _normalize_id_list(record.get("shared_entity_texts", [])),
                "transition_type": str(record.get("transition_type", "")),
                "start_source_passage_ids": _normalize_id_list(record.get("start_source_passage_ids", [])),
                "end_source_passage_ids": _normalize_id_list(record.get("end_source_passage_ids", [])),
                "shared_source_passage_ids": _normalize_id_list(record.get("shared_source_passage_ids", [])),
                "support_passage_pair_ids": _normalize_id_list(record.get("support_passage_pair_ids", [])),
                "start_relation_type": str(record.get("start_relation_type", "")),
                "end_relation_type": str(record.get("end_relation_type", "")),
                "start_participant_roles": dict(record.get("start_participant_roles", {})),
                "end_participant_roles": dict(record.get("end_participant_roles", {})),
                "source_overlap": bool(record.get("source_overlap", False)),
                "entity_overlap": int(record.get("entity_overlap", len(shared_entity_ids))),
                "relation_signature_pair": list(record.get("relation_signature_pair", [])),
                "path_text": str(record.get("path_text", "")),
                "path_embedding_text": str(record.get("path_embedding_text", record.get("path_text", ""))),
            }
            bridge_path_to_passages[normalized_path_id] = source_passage_ids
            bridge_path_to_facts[normalized_path_id] = fact_ids
            for fact_id in fact_ids:
                fact_to_bridge_path_ids[fact_id].add(normalized_path_id)
            for entity_id in shared_entity_ids:
                entity_to_bridge_path_ids[entity_id].add(normalized_path_id)
            for passage_id in source_passage_ids:
                passage_to_bridge_path_ids[passage_id].add(normalized_path_id)
            if start_fact_id and start_fact_id in fact_payloads:
                fact_payloads[start_fact_id].setdefault("bridge_path_ids", set()).add(normalized_path_id)
            if end_fact_id and end_fact_id in fact_payloads:
                fact_payloads[end_fact_id].setdefault("bridge_path_ids", set()).add(normalized_path_id)

        for support_trail_id, record in support_trail_records.items():
            normalized_trail_id = str(record.get("trail_id") or record.get("hash_id") or support_trail_id)
            fact_ids = _unique_in_order(record.get("fact_id_sequence", []))
            if not fact_ids:
                fact_ids = _unique_in_order(
                    [record.get("start_fact_id", "")]
                    + list(record.get("intermediate_fact_ids", []) or [])
                    + [record.get("end_fact_id", "")]
                )
            passage_ids = _unique_in_order(
                list(record.get("support_passage_ids", []) or [])
                + list(record.get("start_source_passage_ids", []) or [])
                + list(record.get("end_source_passage_ids", []) or [])
            )
            bridge_path_ids = _unique_in_order(record.get("bridge_path_id_sequence", []) or [])
            support_trail_payloads[normalized_trail_id] = {
                "support_trail_id": normalized_trail_id,
                "trail_id": normalized_trail_id,
                "record_version": int(record.get("record_version", 0)),
                "start_fact_id": str(record.get("start_fact_id", "")),
                "end_fact_id": str(record.get("end_fact_id", "")),
                "intermediate_fact_ids": _normalize_id_list(record.get("intermediate_fact_ids", [])),
                "fact_id_sequence": fact_ids,
                "bridge_path_id_sequence": bridge_path_ids,
                "hop_count": int(record.get("hop_count", len(bridge_path_ids))),
                "start_source_passage_ids": _normalize_id_list(record.get("start_source_passage_ids", [])),
                "end_source_passage_ids": _normalize_id_list(record.get("end_source_passage_ids", [])),
                "support_passage_ids": passage_ids,
                "shared_entity_ids": _normalize_id_list(record.get("shared_entity_ids", [])),
                "shared_source_passage_ids": _normalize_id_list(record.get("shared_source_passage_ids", [])),
                "transition_types": _normalize_id_list(record.get("transition_types", [])),
                "trail_text": str(record.get("trail_text", "")),
                "trail_embedding_text": str(record.get("trail_embedding_text", record.get("trail_text", ""))),
            }
            support_trail_to_passages[normalized_trail_id] = passage_ids
            support_trail_to_facts[normalized_trail_id] = fact_ids
            support_trail_to_bridge_path_ids[normalized_trail_id] = bridge_path_ids
            for fact_id in fact_ids:
                fact_to_support_trail_ids[fact_id].add(normalized_trail_id)
                if fact_id in fact_payloads:
                    fact_payloads[fact_id].setdefault("support_trail_ids", set()).add(normalized_trail_id)
            for passage_id in passage_ids:
                passage_to_support_trail_ids[passage_id].add(normalized_trail_id)

        for membership_id, record in support_trail_membership_records.items():
            normalized_membership_id = str(record.get("membership_id") or record.get("hash_id") or membership_id)
            fact_id = str(record.get("fact_id", "")).strip()
            if not normalized_membership_id or not fact_id:
                continue
            support_trail_ids = _unique_in_order(record.get("support_trail_ids", []) or [])
            support_fact_ids = _unique_in_order(record.get("support_fact_ids", []) or [])
            support_passage_ids = _unique_in_order(
                list(record.get("support_passage_ids", []) or [])
                + list(record.get("endpoint_passage_ids", []) or [])
            )
            bridge_path_ids = _unique_in_order(record.get("bridge_path_ids", []) or [])
            payload = {
                "support_trail_membership_id": normalized_membership_id,
                "membership_id": normalized_membership_id,
                "record_version": int(record.get("record_version", 0)),
                "fact_id": fact_id,
                "support_trail_ids": support_trail_ids,
                "support_trail_count": int(record.get("support_trail_count", len(support_trail_ids))),
                "support_fact_ids": support_fact_ids,
                "support_passage_ids": support_passage_ids,
                "bridge_path_ids": bridge_path_ids,
                "endpoint_fact_ids": _unique_in_order(record.get("endpoint_fact_ids", []) or []),
                "endpoint_passage_ids": _unique_in_order(record.get("endpoint_passage_ids", []) or []),
                "start_fact_ids": _unique_in_order(record.get("start_fact_ids", []) or []),
                "end_fact_ids": _unique_in_order(record.get("end_fact_ids", []) or []),
                "endpoint_fact_to_passage_ids": _nested_unique_id_map(record.get("endpoint_fact_to_passage_ids", {})),
                "endpoint_fact_to_support_trail_ids": _nested_unique_id_map(record.get("endpoint_fact_to_support_trail_ids", {})),
                "endpoint_fact_to_support_passage_ids": _nested_unique_id_map(record.get("endpoint_fact_to_support_passage_ids", {})),
                "endpoint_fact_to_bridge_path_ids": _nested_unique_id_map(record.get("endpoint_fact_to_bridge_path_ids", {})),
                "endpoint_fact_to_roles": _nested_unique_id_map(record.get("endpoint_fact_to_roles", {})),
                "endpoint_fact_to_hop_counts": _nested_unique_int_map(record.get("endpoint_fact_to_hop_counts", {})),
            }
            support_trail_membership_payloads[normalized_membership_id] = payload
            fact_to_support_trail_membership[fact_id] = payload
            fact_to_support_trail_ids[fact_id].update(support_trail_ids)
            if fact_id in fact_payloads:
                fact_payloads[fact_id].setdefault("support_trail_ids", set()).update(support_trail_ids)
                fact_payloads[fact_id].setdefault("support_trail_membership_ids", set()).add(normalized_membership_id)
            for passage_id in support_passage_ids:
                passage_to_support_trail_ids[passage_id].update(support_trail_ids)

        for endpoint_id, record in support_endpoint_records.items():
            normalized_endpoint_id = str(record.get("endpoint_id") or record.get("hash_id") or endpoint_id)
            owner_fact_id = str(record.get("owner_fact_id", "")).strip()
            endpoint_fact_id = str(record.get("endpoint_fact_id", "")).strip()
            if not normalized_endpoint_id or not owner_fact_id or not endpoint_fact_id:
                continue
            endpoint_passage_ids = _unique_in_order(record.get("endpoint_passage_ids", []) or [])
            support_passage_ids = _unique_in_order(record.get("support_passage_ids", []) or [])
            support_trail_ids = _unique_in_order(record.get("support_trail_ids", []) or [])
            support_fact_ids = _unique_in_order(record.get("support_fact_ids", []) or [])
            bridge_path_ids = _unique_in_order(record.get("bridge_path_ids", []) or [])
            payload = {
                "support_endpoint_id": normalized_endpoint_id,
                "endpoint_id": normalized_endpoint_id,
                "record_version": int(record.get("record_version", 0)),
                "owner_fact_id": owner_fact_id,
                "endpoint_fact_id": endpoint_fact_id,
                "support_trail_ids": support_trail_ids,
                "support_trail_count": int(record.get("support_trail_count", len(support_trail_ids))),
                "support_fact_ids": support_fact_ids,
                "support_passage_ids": support_passage_ids,
                "endpoint_passage_ids": endpoint_passage_ids,
                "bridge_path_ids": bridge_path_ids,
                "support_relation_families": _unique_in_order(record.get("support_relation_families", []) or []),
                "endpoint_relation_families": _unique_in_order(record.get("endpoint_relation_families", []) or []),
                "min_hop_count": int(record.get("min_hop_count", 0)),
                "hop_counts": [int(value) for value in record.get("hop_counts", []) or []],
                "owner_positions": [int(value) for value in record.get("owner_positions", []) or []],
                "endpoint_roles": _unique_in_order(record.get("endpoint_roles", []) or []),
                "transition_types": _unique_in_order(record.get("transition_types", []) or []),
            }
            support_endpoint_payloads[normalized_endpoint_id] = payload
            fact_to_support_endpoint_ids[owner_fact_id].add(normalized_endpoint_id)
            endpoint_fact_to_support_endpoint_ids[endpoint_fact_id].add(normalized_endpoint_id)
            support_endpoint_to_passages[normalized_endpoint_id] = endpoint_passage_ids
            support_endpoint_to_facts[normalized_endpoint_id] = _unique_in_order(
                [owner_fact_id, endpoint_fact_id] + support_fact_ids
            )
            support_endpoint_to_trail_ids[normalized_endpoint_id] = support_trail_ids
            support_endpoint_to_bridge_path_ids[normalized_endpoint_id] = bridge_path_ids
            if owner_fact_id in fact_payloads:
                fact_payloads[owner_fact_id].setdefault("support_endpoint_ids", set()).add(normalized_endpoint_id)
            if endpoint_fact_id in fact_payloads:
                fact_payloads[endpoint_fact_id].setdefault("credited_support_endpoint_ids", set()).add(
                    normalized_endpoint_id
                )
            for passage_id in endpoint_passage_ids:
                passage_to_support_endpoint_ids[passage_id].add(normalized_endpoint_id)

        for fact_object_id, payload in list(fact_payloads.items()):
            fact_payloads[fact_object_id] = {
                "fact_object_id": fact_object_id,
                "bridge_ids": sorted(payload["bridge_ids"]),
                "slot_ids": sorted(payload["slot_ids"]),
                "bridge_path_ids": sorted(payload.get("bridge_path_ids", set())),
                "support_trail_ids": sorted(payload.get("support_trail_ids", set())),
                "support_trail_membership_ids": sorted(payload.get("support_trail_membership_ids", set())),
                "support_endpoint_ids": sorted(payload.get("support_endpoint_ids", set())),
                "credited_support_endpoint_ids": sorted(payload.get("credited_support_endpoint_ids", set())),
                "relation_types": sorted(payload["relation_types"]),
                "participant_entity_ids": sorted(payload["participant_entity_ids"]),
                "participant_texts": sorted(payload["participant_texts"]),
                "source_ids": sorted(payload["source_ids"]),
                "pair_signatures": sorted(payload["pair_signatures"]),
                "subj_endpoint_signatures": sorted(payload["subj_endpoint_signatures"]),
                "obj_endpoint_signatures": sorted(payload["obj_endpoint_signatures"]),
            }

        if slot_records:
            slot_payloads = {}
            fact_to_slot_ids = defaultdict(set)
            bridge_to_slot_ids = defaultdict(set)
            passage_to_slot_ids = defaultdict(set)
            slot_to_passages = defaultdict(set)
            slot_to_bridge_ids = defaultdict(set)
            for slot_id, record in slot_records.items():
                normalized_slot_id = str(slot_id)
                fact_object_id = str(record.get("fact_object_id", ""))
                bridge_id = str(record.get("bridge_id", ""))
                source_ids = _normalize_id_list(record.get("source_ids", []))
                entity_id = str(record.get("entity_id", ""))
                has_materialized_support_tiers = all(
                    field_name in record
                    for field_name in (
                        "exact_support_passage_ids",
                        "endpoint_support_passage_ids",
                        "entity_support_passage_ids",
                        "supporting_passage_ids",
                    )
                )
                if has_materialized_support_tiers:
                    exact_support_passage_ids = _coerce_id_list(record.get("exact_support_passage_ids", []))
                    endpoint_support_passage_ids = _coerce_id_list(record.get("endpoint_support_passage_ids", []))
                    entity_support_passage_ids = _coerce_id_list(record.get("entity_support_passage_ids", []))
                    supporting_passage_ids = _coerce_id_list(
                        record.get("supporting_passage_ids", []) or source_ids
                    )
                    source_support_passage_ids = _coerce_id_list(
                        record.get("source_support_passage_ids", source_ids)
                    )
                    anchor_source_support_passage_ids = _coerce_id_list(
                        record.get(
                            "anchor_source_support_passage_ids",
                            record.get("source_support_passage_ids", source_ids),
                        )
                    )
                    if all(
                        field_name in record
                        for field_name in (
                            "anchor_support_passage_ids",
                            "target_support_passage_ids",
                            "answer_support_passage_ids",
                        )
                    ):
                        anchor_support_passage_ids = _coerce_id_list(record.get("anchor_support_passage_ids", []))
                        target_support_passage_ids = _coerce_id_list(record.get("target_support_passage_ids", []))
                        answer_support_passage_ids = _coerce_id_list(record.get("answer_support_passage_ids", []))
                    else:
                        derived_support_passages = derive_slot_support_passage_ids(
                            slot_kind=str(record.get("slot_kind", "")),
                            relation_type=str(record.get("relation_type", "")),
                            entity_text=str(record.get("entity_text", "")),
                            support_signature_sidecar=sidecar,
                        )
                        anchor_support_passage_ids = _normalize_id_list(
                            derived_support_passages["anchor_support_passage_ids"] or entity_support_passage_ids
                        )
                        target_support_passage_ids = _normalize_id_list(
                            derived_support_passages["target_support_passage_ids"]
                        )
                        answer_support_passage_ids = _normalize_id_list(
                            derived_support_passages["answer_support_passage_ids"] or target_support_passage_ids
                        )
                else:
                    derived_support_passages = derive_slot_support_passage_ids(
                        slot_kind=str(record.get("slot_kind", "")),
                        relation_type=str(record.get("relation_type", "")),
                        entity_text=str(record.get("entity_text", "")),
                        support_signature_sidecar=sidecar,
                    )
                    exact_support_passage_ids = _normalize_id_list(derived_support_passages["exact_support_passage_ids"])
                    endpoint_support_passage_ids = _normalize_id_list(
                        derived_support_passages["endpoint_support_passage_ids"]
                    )
                    entity_support_passage_ids = _normalize_id_list(
                        derived_support_passages["entity_support_passage_ids"]
                    )
                    supporting_passage_ids = _normalize_id_list(
                        derived_support_passages["supporting_passage_ids"] or source_ids
                    )
                    source_support_passage_ids = _normalize_id_list(
                        derived_support_passages["source_support_passage_ids"] or source_ids
                    )
                    anchor_source_support_passage_ids = _normalize_id_list(
                        derived_support_passages["anchor_source_support_passage_ids"] or source_support_passage_ids
                    )
                    anchor_support_passage_ids = _normalize_id_list(
                        derived_support_passages["anchor_support_passage_ids"] or entity_support_passage_ids
                    )
                    target_support_passage_ids = _normalize_id_list(
                        derived_support_passages["target_support_passage_ids"]
                    )
                    answer_support_passage_ids = _normalize_id_list(
                        derived_support_passages["answer_support_passage_ids"] or target_support_passage_ids
                    )
                slot_payloads[normalized_slot_id] = {
                    "slot_id": normalized_slot_id,
                    "slot_kind": str(record.get("slot_kind", "")),
                    "slot_scope": str(record.get("slot_scope", "")),
                    "relation_type": str(record.get("relation_type", "")),
                    "entity_text": str(record.get("entity_text", "")),
                    "anchor_entity_id": str(record.get("anchor_entity_id", "")),
                    "anchor_entity_text": str(record.get("anchor_entity_text", "")),
                    "target_entity_id": str(record.get("target_entity_id", "")),
                    "target_entity_text": str(record.get("target_entity_text", "")),
                    "attribute_signature": str(record.get("attribute_signature", "")),
                    "entity_ids": [entity_id] if entity_id else [],
                    "bridge_ids": [bridge_id] if bridge_id else [],
                    "fact_object_ids": [fact_object_id] if fact_object_id else [],
                    "source_ids": source_ids,
                    "owning_source_ids": _normalize_id_list(record.get("owning_source_ids", source_ids)),
                    "source_support_passage_ids": source_support_passage_ids,
                    "anchor_source_support_passage_ids": anchor_source_support_passage_ids,
                }
                slot_to_exact_passages[normalized_slot_id] = exact_support_passage_ids
                slot_to_endpoint_passages[normalized_slot_id] = endpoint_support_passage_ids
                slot_to_entity_passages[normalized_slot_id] = entity_support_passage_ids
                slot_to_support_passages[normalized_slot_id] = supporting_passage_ids
                slot_to_source_support_passages[normalized_slot_id] = source_support_passage_ids
                slot_to_anchor_source_support_passages[normalized_slot_id] = anchor_source_support_passage_ids
                slot_to_anchor_support_passages[normalized_slot_id] = anchor_support_passage_ids
                slot_to_target_support_passages[normalized_slot_id] = target_support_passage_ids
                slot_to_answer_support_passages[normalized_slot_id] = answer_support_passage_ids
                if fact_object_id:
                    fact_to_slot_ids[fact_object_id].add(normalized_slot_id)
                if bridge_id:
                    bridge_to_slot_ids[bridge_id].add(normalized_slot_id)
                    slot_to_bridge_ids[normalized_slot_id].add(bridge_id)
                for passage_id in supporting_passage_ids:
                    passage_to_slot_ids[passage_id].add(normalized_slot_id)
                    slot_to_passages[normalized_slot_id].add(passage_id)
                for passage_id in exact_support_passage_ids:
                    if normalized_slot_id not in passage_to_exact_slot_ids[passage_id]:
                        passage_to_exact_slot_ids[passage_id].append(normalized_slot_id)
            for fact_object_id, payload in list(fact_payloads.items()):
                payload["slot_ids"] = sorted(fact_to_slot_ids.get(fact_object_id, set()))
        else:
            passage_to_slot_ids = defaultdict(set)
            slot_to_passages = defaultdict(set)
            for slot_id, payload in list(slot_payloads.items()):
                derived_support_passages = derive_slot_support_passage_ids(
                    slot_kind=str(payload["slot_kind"]),
                    relation_type=str(payload["relation_type"]),
                    entity_text=str(payload["entity_text"]),
                    support_signature_sidecar=sidecar,
                )
                source_ids = sorted(payload["source_ids"])
                slot_payloads[slot_id] = {
                    "slot_id": slot_id,
                    "slot_kind": str(payload["slot_kind"]),
                    "slot_scope": str(payload.get("slot_scope", "")),
                    "relation_type": str(payload["relation_type"]),
                    "entity_text": str(payload["entity_text"]),
                    "anchor_entity_id": str(payload.get("anchor_entity_id", "")),
                    "anchor_entity_text": str(payload.get("anchor_entity_text", "")),
                    "target_entity_id": str(payload.get("target_entity_id", "")),
                    "target_entity_text": str(payload.get("target_entity_text", "")),
                    "attribute_signature": str(payload.get("attribute_signature", "")),
                    "entity_ids": sorted(payload["entity_ids"]),
                    "bridge_ids": sorted(payload["bridge_ids"]),
                    "fact_object_ids": sorted(payload["fact_object_ids"]),
                    "source_ids": source_ids,
                    "owning_source_ids": source_ids,
                    "source_support_passage_ids": list(payload.get("source_support_passage_ids", source_ids)),
                    "anchor_source_support_passage_ids": list(
                        payload.get("anchor_source_support_passage_ids", payload.get("source_support_passage_ids", source_ids))
                    ),
                }
                slot_to_exact_passages[slot_id] = list(derived_support_passages["exact_support_passage_ids"])
                slot_to_endpoint_passages[slot_id] = list(derived_support_passages["endpoint_support_passage_ids"])
                slot_to_entity_passages[slot_id] = list(derived_support_passages["entity_support_passage_ids"])
                slot_to_support_passages[slot_id] = list(derived_support_passages["supporting_passage_ids"])
                slot_to_source_support_passages[slot_id] = list(
                    derived_support_passages["source_support_passage_ids"] or source_ids
                )
                slot_to_anchor_source_support_passages[slot_id] = list(
                    derived_support_passages["anchor_source_support_passage_ids"]
                    or derived_support_passages["source_support_passage_ids"]
                    or source_ids
                )
                slot_to_anchor_support_passages[slot_id] = list(
                    derived_support_passages["anchor_support_passage_ids"]
                    or derived_support_passages["entity_support_passage_ids"]
                )
                slot_to_target_support_passages[slot_id] = list(derived_support_passages["target_support_passage_ids"])
                slot_to_answer_support_passages[slot_id] = list(
                    derived_support_passages["answer_support_passage_ids"]
                    or derived_support_passages["target_support_passage_ids"]
                )
                for passage_id in derived_support_passages["supporting_passage_ids"]:
                    passage_to_slot_ids[str(passage_id)].add(slot_id)
                    slot_to_passages[slot_id].add(str(passage_id))
                for passage_id in derived_support_passages["exact_support_passage_ids"]:
                    normalized_passage_id = str(passage_id)
                    if slot_id not in passage_to_exact_slot_ids[normalized_passage_id]:
                        passage_to_exact_slot_ids[normalized_passage_id].append(slot_id)

        entity_to_passages: Dict[str, set] = defaultdict(set)
        passage_payloads: Dict[str, Dict[str, Any]] = {}
        for passage_id, record in passage_records.items():
            normalized_passage_id = str(passage_id)
            entity_ids = _normalize_id_list(record.get("entity_ids", []))
            for entity_id in entity_ids:
                entity_to_passages[entity_id].add(normalized_passage_id)
            bridge_ids = sorted(passage_to_bridge_ids.get(normalized_passage_id, set()) | set(record.get("proposition_ids", [])))
            fact_ids = sorted(passage_to_fact_ids.get(normalized_passage_id, set()))
            slot_ids = sorted(passage_to_slot_ids.get(normalized_passage_id, set()))
            bridge_path_ids = sorted(passage_to_bridge_path_ids.get(normalized_passage_id, set()))
            passage_payloads[normalized_passage_id] = {
                "passage_id": normalized_passage_id,
                "entity_ids": entity_ids,
                "bridge_ids": [str(value) for value in bridge_ids],
                "fact_object_ids": fact_ids,
                "slot_ids": slot_ids,
                "bridge_path_ids": bridge_path_ids,
            }

        for passage_id in passage_inventories:
            passage_payloads.setdefault(
                str(passage_id),
                {
                    "passage_id": str(passage_id),
                    "entity_ids": [],
                    "bridge_ids": [],
                    "fact_object_ids": [],
                    "slot_ids": sorted(passage_to_slot_ids.get(str(passage_id), set())),
                    "bridge_path_ids": sorted(passage_to_bridge_path_ids.get(str(passage_id), set())),
                },
            )

        bridge_neighbors: Dict[str, set] = defaultdict(set)
        for bridge_ids in entity_to_bridges.values():
            bridge_id_list = sorted(bridge_ids)
            for bridge_id in bridge_id_list:
                bridge_neighbors[bridge_id].update(other for other in bridge_id_list if other != bridge_id)
        for bridge_ids in relation_to_bridges.values():
            bridge_id_list = sorted(bridge_ids)
            for bridge_id in bridge_id_list:
                bridge_neighbors[bridge_id].update(other for other in bridge_id_list if other != bridge_id)
        for bridge_ids in conflict_group_to_bridges.values():
            bridge_id_list = sorted(bridge_ids)
            for bridge_id in bridge_id_list:
                bridge_neighbors[bridge_id].update(other for other in bridge_id_list if other != bridge_id)
        for payload in passage_payloads.values():
            bridge_id_list = list(payload.get("bridge_ids", []))
            for bridge_id in bridge_id_list:
                bridge_neighbors[bridge_id].update(other for other in bridge_id_list if other != bridge_id)

        substrate_version = _build_substrate_version(
            entity_count=len(entity_payloads),
            fact_count=len(fact_payloads),
            bridge_count=len(bridge_payloads),
            bridge_path_count=len(bridge_path_payloads),
            slot_count=len(slot_payloads),
            passage_count=len(passage_payloads),
            graph_nodes=int(getattr(getattr(runtime, "graph", None), "vcount", lambda: 0)()),
            graph_edges=int(getattr(getattr(runtime, "graph", None), "ecount", lambda: 0)()),
            sidecar_version=int(sidecar.get("version", 1)),
            passage_inventory_count=len(passage_inventories),
            pair_signature_count=len(dict(sidecar.get("pair_to_passages", {}))),
            source_support_slot_count=len(slot_to_source_support_passages),
            anchor_source_support_slot_count=len(slot_to_anchor_source_support_passages),
            anchor_support_slot_count=len(slot_to_anchor_support_passages),
            target_support_slot_count=len(slot_to_target_support_passages),
            answer_support_slot_count=len(slot_to_answer_support_passages),
        )

        return cls(
            substrate_version=substrate_version,
            entity_payloads=entity_payloads,
            fact_payloads=fact_payloads,
            bridge_payloads=bridge_payloads,
            slot_payloads=slot_payloads,
            bridge_path_payloads=bridge_path_payloads,
            passage_payloads=passage_payloads,
            passage_inventories=passage_inventories,
            entity_to_passages={key: sorted(value) for key, value in entity_to_passages.items()},
            entity_to_bridges={key: sorted(value) for key, value in entity_to_bridges.items()},
            entity_to_bridge_path_ids={key: sorted(value) for key, value in entity_to_bridge_path_ids.items()},
            fact_to_passages={key: sorted(value) for key, value in fact_to_passage_ids.items()},
            fact_to_bridge_ids={key: sorted(value) for key, value in fact_to_bridge_ids.items()},
            fact_to_slot_ids={key: sorted(value) for key, value in fact_to_slot_ids.items()},
            fact_to_bridge_path_ids={key: sorted(value) for key, value in fact_to_bridge_path_ids.items()},
            bridge_to_passages=bridge_to_passages,
            bridge_to_fact_id=bridge_to_fact_id,
            bridge_to_slot_ids={key: sorted(value) for key, value in bridge_to_slot_ids.items()},
            bridge_neighbors={key: sorted(value) for key, value in bridge_neighbors.items()},
            bridge_path_to_passages=bridge_path_to_passages,
            bridge_path_to_facts=bridge_path_to_facts,
            slot_to_passages={key: sorted(value) for key, value in slot_to_passages.items()},
            slot_to_bridge_ids={key: sorted(value) for key, value in slot_to_bridge_ids.items()},
            passage_to_bridge_path_ids={
                key: sorted(value)
                for key, value in passage_to_bridge_path_ids.items()
            },
            support_trail_payloads=support_trail_payloads,
            support_trail_membership_payloads=support_trail_membership_payloads,
            fact_to_support_trail_membership=fact_to_support_trail_membership,
            fact_to_support_trail_ids={
                key: sorted(value)
                for key, value in fact_to_support_trail_ids.items()
            },
            passage_to_support_trail_ids={
                key: sorted(value)
                for key, value in passage_to_support_trail_ids.items()
            },
            support_trail_to_passages=support_trail_to_passages,
            support_trail_to_facts=support_trail_to_facts,
            support_trail_to_bridge_path_ids=support_trail_to_bridge_path_ids,
            support_endpoint_payloads=support_endpoint_payloads,
            fact_to_support_endpoint_ids={
                key: sorted(value)
                for key, value in fact_to_support_endpoint_ids.items()
            },
            endpoint_fact_to_support_endpoint_ids={
                key: sorted(value)
                for key, value in endpoint_fact_to_support_endpoint_ids.items()
            },
            passage_to_support_endpoint_ids={
                key: sorted(value)
                for key, value in passage_to_support_endpoint_ids.items()
            },
            support_endpoint_to_passages=support_endpoint_to_passages,
            support_endpoint_to_facts=support_endpoint_to_facts,
            support_endpoint_to_trail_ids=support_endpoint_to_trail_ids,
            support_endpoint_to_bridge_path_ids=support_endpoint_to_bridge_path_ids,
            pair_to_passages={
                tuple(signature): passage_refs
                for signature, passage_refs in dict(sidecar.get("pair_to_passages", {})).items()
            },
            subj_endpoint_to_passages={
                tuple(signature): passage_refs
                for signature, passage_refs in dict(sidecar.get("subj_endpoint_to_passages", {})).items()
            },
            obj_endpoint_to_passages={
                tuple(signature): passage_refs
                for signature, passage_refs in dict(sidecar.get("obj_endpoint_to_passages", {})).items()
            },
            subject_entity_to_passages={
                str(entity): passage_refs
                for entity, passage_refs in dict(sidecar.get("subject_entity_to_passages", {})).items()
            },
            object_entity_to_passages={
                str(entity): passage_refs
                for entity, passage_refs in dict(sidecar.get("object_entity_to_passages", {})).items()
            },
            slot_to_exact_passages=slot_to_exact_passages,
            slot_to_endpoint_passages=slot_to_endpoint_passages,
            slot_to_entity_passages=slot_to_entity_passages,
            slot_to_support_passages=slot_to_support_passages,
            slot_to_source_support_passages=slot_to_source_support_passages,
            slot_to_anchor_source_support_passages=slot_to_anchor_source_support_passages,
            slot_to_anchor_support_passages=slot_to_anchor_support_passages,
            slot_to_target_support_passages=slot_to_target_support_passages,
            slot_to_answer_support_passages=slot_to_answer_support_passages,
            passage_to_exact_slot_ids={
                passage_id: list(slot_ids)
                for passage_id, slot_ids in passage_to_exact_slot_ids.items()
            },
            runtime=runtime,
        )

    def get_coverage_view(self) -> CoverageGraphView:
        return CoverageGraphView(
            entity_ids=list(self.entity_ids),
            fact_object_ids=list(self.fact_object_ids),
            bridge_ids=list(self.bridge_ids),
            passage_ids=list(self.passage_ids),
            entity_to_passages=dict(self.entity_to_passages),
            fact_to_passages=dict(self.fact_to_passages),
            fact_to_bridge_ids=dict(self.fact_to_bridge_ids),
            fact_to_slot_ids=dict(self.fact_to_slot_ids),
            passage_to_entities=dict(self.passage_to_entities),
            passage_to_fact_ids=dict(self.passage_to_fact_ids),
            passage_to_slot_ids=dict(self.passage_to_slot_ids),
            substrate_version=self.substrate_version,
        )

    def get_support_view(self) -> SupportGraphView:
        return SupportGraphView(
            fact_objects=dict(self.fact_payloads),
            bridge_objects=dict(self.bridge_payloads),
            slot_objects=dict(self.slot_payloads),
            passage_inventories=dict(self.passage_inventories),
            pair_to_passages=dict(self.pair_to_passages),
            subj_endpoint_to_passages=dict(self.subj_endpoint_to_passages),
            obj_endpoint_to_passages=dict(self.obj_endpoint_to_passages),
            slot_to_passages=dict(self.slot_to_passages),
            subject_entity_to_passages=dict(self.subject_entity_to_passages),
            object_entity_to_passages=dict(self.object_entity_to_passages),
            fact_to_passages=dict(self.fact_to_passages),
            fact_to_slot_ids=dict(self.fact_to_slot_ids),
            passage_to_slot_ids=dict(self.passage_to_slot_ids),
            bridge_to_passages=dict(self.bridge_to_passages),
            slot_to_exact_passages=dict(self.slot_to_exact_passages),
            slot_to_endpoint_passages=dict(self.slot_to_endpoint_passages),
            slot_to_entity_passages=dict(self.slot_to_entity_passages),
            slot_to_source_support_passages=dict(self.slot_to_source_support_passages),
            slot_to_anchor_source_support_passages=dict(self.slot_to_anchor_source_support_passages),
            slot_to_anchor_support_passages=dict(self.slot_to_anchor_support_passages),
            slot_to_target_support_passages=dict(self.slot_to_target_support_passages),
            slot_to_answer_support_passages=dict(self.slot_to_answer_support_passages),
            substrate_version=self.substrate_version,
        )

    def get_route_view(self) -> RouteGraphView:
        return RouteGraphView(
            fact_to_bridge_ids=dict(self.fact_to_bridge_ids),
            fact_to_slot_ids=dict(self.fact_to_slot_ids),
            fact_to_bridge_path_ids=dict(self.fact_to_bridge_path_ids),
            bridge_to_fact_id=dict(self.bridge_to_fact_id),
            bridge_to_slot_ids=dict(self.bridge_to_slot_ids),
            bridge_neighbors=dict(self.bridge_neighbors),
            slot_to_bridge_ids=dict(self.slot_to_bridge_ids),
            slot_to_passages=dict(self.slot_to_passages),
            bridge_to_passages=dict(self.bridge_to_passages),
            bridge_path_to_passages=dict(self.bridge_path_to_passages),
            bridge_path_to_facts=dict(self.bridge_path_to_facts),
            slot_to_source_support_passages=dict(self.slot_to_source_support_passages),
            slot_to_anchor_source_support_passages=dict(self.slot_to_anchor_source_support_passages),
            slot_to_anchor_support_passages=dict(self.slot_to_anchor_support_passages),
            slot_to_target_support_passages=dict(self.slot_to_target_support_passages),
            slot_to_answer_support_passages=dict(self.slot_to_answer_support_passages),
            substrate_version=self.substrate_version,
        )

    def get_bridge_path_view(self) -> BridgePathGraphView:
        return BridgePathGraphView(
            bridge_path_objects=dict(self.bridge_path_payloads),
            fact_to_bridge_path_ids=dict(self.fact_to_bridge_path_ids),
            passage_to_bridge_path_ids=dict(self.passage_to_bridge_path_ids),
            entity_to_bridge_path_ids=dict(self.entity_to_bridge_path_ids),
            bridge_path_to_passages=dict(self.bridge_path_to_passages),
            bridge_path_to_facts=dict(self.bridge_path_to_facts),
            substrate_version=self.substrate_version,
        )

    def get_support_trail_view(self) -> BridgeSupportTrailGraphView:
        return BridgeSupportTrailGraphView(
            support_trail_objects=dict(self.support_trail_payloads),
            support_trail_membership_objects=dict(self.support_trail_membership_payloads),
            fact_to_support_trail_membership=dict(self.fact_to_support_trail_membership),
            fact_to_support_trail_ids=dict(self.fact_to_support_trail_ids),
            passage_to_support_trail_ids=dict(self.passage_to_support_trail_ids),
            support_trail_to_passages=dict(self.support_trail_to_passages),
            support_trail_to_facts=dict(self.support_trail_to_facts),
            support_trail_to_bridge_path_ids=dict(self.support_trail_to_bridge_path_ids),
            substrate_version=self.substrate_version,
        )

    def get_support_endpoint_view(self) -> BridgeSupportEndpointGraphView:
        return BridgeSupportEndpointGraphView(
            support_endpoint_objects=dict(self.support_endpoint_payloads),
            fact_to_support_endpoint_ids=dict(self.fact_to_support_endpoint_ids),
            endpoint_fact_to_support_endpoint_ids=dict(self.endpoint_fact_to_support_endpoint_ids),
            passage_to_support_endpoint_ids=dict(self.passage_to_support_endpoint_ids),
            support_endpoint_to_passages=dict(self.support_endpoint_to_passages),
            support_endpoint_to_facts=dict(self.support_endpoint_to_facts),
            support_endpoint_to_trail_ids=dict(self.support_endpoint_to_trail_ids),
            support_endpoint_to_bridge_path_ids=dict(self.support_endpoint_to_bridge_path_ids),
            substrate_version=self.substrate_version,
        )

    def _ensure_query_scores(self, query_text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.runtime is None:
            raise RuntimeError("Canonical retrieval interface requires a prepared runtime for query activation.")
        if hasattr(self.runtime, "get_query_embeddings"):
            self.runtime.get_query_embeddings([query_text])
        entity_scores = np.asarray(self.runtime.get_entity_scores(query_text), dtype=np.float32)
        fact_scores = np.asarray(self.runtime.get_fact_scores(query_text), dtype=np.float32)
        runtime_bridge_ids = list(getattr(self.runtime, "hyperedge_node_keys", []) or [])
        runtime_bridge_embeddings = np.asarray(getattr(self.runtime, "hyperedge_embeddings", np.array([])), dtype=np.float32)
        if hasattr(self.runtime, "get_hyperedge_scores") and runtime_bridge_ids and runtime_bridge_embeddings.size > 0:
            bridge_scores = np.asarray(self.runtime.get_hyperedge_scores(query_text), dtype=np.float32)
        else:
            bridge_scores = np.array([], dtype=np.float32)
        passage_scores = (
            np.asarray(self.runtime.get_passage_scores(query_text), dtype=np.float32)
            if hasattr(self.runtime, "get_passage_scores")
            else np.array([], dtype=np.float32)
        )
        return entity_scores, fact_scores, bridge_scores, passage_scores

    def get_query_activation_state(
        self,
        query_text: str,
        top_k_entities: int = 8,
        top_k_facts: int = 8,
        top_k_bridges: int = 8,
        top_k_passages: int = 16,
    ) -> QueryActivationState:
        entity_scores, fact_scores, bridge_scores, passage_scores = self._ensure_query_scores(query_text)

        runtime_entity_ids = list(getattr(self.runtime, "entity_node_keys", self.entity_ids))
        runtime_fact_ids = list(getattr(self.runtime, "fact_node_keys", self.fact_object_ids))
        runtime_bridge_ids = list(getattr(self.runtime, "hyperedge_node_keys", self.bridge_ids))
        runtime_passage_ids = list(getattr(self.runtime, "passage_node_keys", self.passage_ids))

        query_entity_ids, query_entity_scores = _top_scored_ids(runtime_entity_ids, entity_scores, top_k_entities)
        query_fact_ids, query_fact_scores = _top_scored_ids(runtime_fact_ids, fact_scores, top_k_facts)
        top_bridge_ids, _ = _top_scored_ids(runtime_bridge_ids, bridge_scores, top_k_bridges)
        query_passage_ids, query_passage_scores = _top_scored_ids(runtime_passage_ids, passage_scores, top_k_passages)

        bridge_hints: List[str] = []
        for fact_object_id in query_fact_ids:
            for bridge_id in self.fact_to_bridge_ids.get(fact_object_id, []):
                if bridge_id not in bridge_hints:
                    bridge_hints.append(bridge_id)
        for bridge_id in top_bridge_ids:
            if bridge_id not in bridge_hints:
                bridge_hints.append(bridge_id)

        return QueryActivationState(
            query_text=query_text,
            query_entity_ids=query_entity_ids,
            query_entity_scores=query_entity_scores,
            query_fact_ids=query_fact_ids,
            query_fact_scores=query_fact_scores,
            query_bridge_hints=bridge_hints[: max(int(top_k_bridges), 0)],
            query_passage_ids=query_passage_ids,
            query_passage_scores=query_passage_scores,
            activation_mode="prepared_runtime",
            substrate_version=self.substrate_version,
        )

    def _dense_rank_map(self, query_state: QueryActivationState) -> Dict[str, int]:
        return {str(passage_id): idx for idx, passage_id in enumerate(query_state.query_passage_ids)}

    def _query_entity_texts(self, query_state: QueryActivationState) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for entity_id in query_state.query_entity_ids:
            payload = self.entity_payloads.get(str(entity_id), {})
            canonical_name = _normalize_text(payload.get("canonical_name", ""))
            if canonical_name:
                mapping[str(entity_id)] = canonical_name
        return mapping

    def _slot_bridge_ids(self, slot_ids: Sequence[str]) -> List[str]:
        bridge_ids: List[str] = []
        for slot_id in slot_ids:
            normalized_slot_id = str(slot_id)
            payload = self.slot_payloads.get(normalized_slot_id, {})
            bridge_id = str(payload.get("bridge_id", "")).strip()
            if bridge_id:
                bridge_ids.append(bridge_id)
            bridge_ids.extend(str(value) for value in self.slot_to_bridge_ids.get(normalized_slot_id, []))
        return _unique_in_order(bridge_ids)

    def extract_candidates(
        self,
        query_state: QueryActivationState,
        channel: str,
        top_k: int,
        head_passage_ids: Optional[Sequence[str]] = None,
    ) -> CandidateSet:
        normalized_channel = str(channel).strip().lower()
        if normalized_channel == "coverage":
            return self._extract_coverage_candidates(query_state, top_k=top_k)
        if normalized_channel == "support":
            return self._extract_support_candidates(query_state, top_k=top_k)
        if normalized_channel in {"support_slot_tier", "support-slot-tier"}:
            return self._extract_slot_tier_support_candidates(query_state, top_k=top_k)
        if normalized_channel in {"support_slot_sourcepreserve", "support-slot-sourcepreserve"}:
            return self._extract_slot_sourcepreserve_candidates(query_state, top_k=top_k)
        if normalized_channel in {"bridge_path", "bridge-path"}:
            return self._extract_bridge_path_candidates(query_state, top_k=top_k)
        if normalized_channel in {
            "support_trail_membership",
            "support-trail-membership",
            "bridge_support_trail_membership",
            "bridge-support-trail-membership",
        }:
            return self._extract_support_trail_membership_candidates(
                query_state,
                top_k=top_k,
                channel="support_trail_membership",
            )
        if normalized_channel in {"support_endpoint", "support-endpoint", "bridge_support_endpoint", "bridge-support-endpoint"}:
            return self._extract_support_endpoint_candidates(
                query_state,
                top_k=top_k,
                channel="support_endpoint",
            )
        if normalized_channel in {"support_trail", "support-trail", "bridge_support_trail", "bridge-support-trail"}:
            return self._extract_support_trail_candidates(query_state, top_k=top_k)
        if normalized_channel == "route":
            return self._extract_route_candidates(
                query_state,
                top_k=top_k,
                head_passage_ids=list(head_passage_ids or []),
            )
        raise ValueError(f"Unsupported candidate channel: {channel}")

    def _extract_coverage_candidates(
        self,
        query_state: QueryActivationState,
        *,
        top_k: int,
    ) -> CandidateSet:
        dense_rank_map = self._dense_rank_map(query_state)
        dense_rank_default = len(dense_rank_map) + len(self.passage_ids) + 1
        candidate_stats: Dict[str, Dict[str, set]] = defaultdict(lambda: {"fact_hits": set(), "entity_hits": set()})

        for fact_object_id in query_state.query_fact_ids:
            for passage_id in self.fact_to_passages.get(str(fact_object_id), []):
                candidate_stats[str(passage_id)]["fact_hits"].add(str(fact_object_id))
        for entity_id in query_state.query_entity_ids:
            for passage_id in self.entity_to_passages.get(str(entity_id), []):
                candidate_stats[str(passage_id)]["entity_hits"].add(str(entity_id))
        for passage_id in query_state.query_passage_ids:
            candidate_stats.setdefault(str(passage_id), {"fact_hits": set(), "entity_hits": set()})

        ranked_passages: List[Tuple[Tuple[int, ...], str]] = []
        for passage_id, stats in candidate_stats.items():
            if passage_id not in self.passage_payloads:
                continue
            fact_hits = stats["fact_hits"]
            entity_hits = stats["entity_hits"]
            candidate_key = (
                1 if fact_hits else 0,
                len(fact_hits),
                1 if entity_hits else 0,
                len(entity_hits),
                1 if passage_id in dense_rank_map else 0,
                -int(dense_rank_map.get(passage_id, dense_rank_default)),
            )
            ranked_passages.append((candidate_key, passage_id))

        candidate_passage_ids = _rank_with_dense_fallback(ranked_passages, top_k)
        candidate_bridge_ids = _unique_in_order(
            query_state.query_bridge_hints
            + [bridge_id for fact_object_id in query_state.query_fact_ids for bridge_id in self.fact_to_bridge_ids.get(str(fact_object_id), [])]
        )[: max(int(top_k), 0)]
        evidence_index_refs = [
            {
                "kind": "query_activation",
                "fact_object_ids": list(query_state.query_fact_ids),
                "entity_ids": list(query_state.query_entity_ids),
                "dense_passage_ids": list(query_state.query_passage_ids),
            }
        ]
        return CandidateSet(
            channel="coverage",
            candidate_passage_ids=candidate_passage_ids,
            candidate_fact_ids=list(query_state.query_fact_ids[: max(int(top_k), 0)]),
            candidate_bridge_ids=candidate_bridge_ids,
            evidence_index_refs=evidence_index_refs,
            substrate_version=self.substrate_version,
        )

    def _extract_support_candidates(
        self,
        query_state: QueryActivationState,
        *,
        top_k: int,
    ) -> CandidateSet:
        dense_rank_map = self._dense_rank_map(query_state)
        dense_rank_default = len(dense_rank_map) + len(self.passage_ids) + 1
        candidate_stats = defaultdict(
            lambda: {
                "pair_hits": set(),
                "endpoint_hits": set(),
                "entity_hits": set(),
                "source_hits": set(),
                "best_fact_rank": len(query_state.query_fact_ids) + 1,
                "best_pair_rank": len(query_state.query_fact_ids) + 1,
                "best_endpoint_rank": len(query_state.query_fact_ids) + 1,
            }
        )

        for fact_rank, fact_object_id in enumerate(query_state.query_fact_ids):
            payload = self.fact_payloads.get(str(fact_object_id), {})
            for signature in payload.get("pair_signatures", []):
                for passage_id in self.pair_to_passages.get(tuple(signature), []):
                    candidate_stats[str(passage_id)]["pair_hits"].add(str(fact_object_id))
                    candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                        int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                        int(fact_rank),
                    )
                    candidate_stats[str(passage_id)]["best_pair_rank"] = min(
                        int(candidate_stats[str(passage_id)]["best_pair_rank"]),
                        int(fact_rank),
                    )
            for signature in payload.get("subj_endpoint_signatures", []):
                for passage_id in self.subj_endpoint_to_passages.get(tuple(signature), []):
                    candidate_stats[str(passage_id)]["endpoint_hits"].add(str(fact_object_id))
                    candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                        int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                        int(fact_rank),
                    )
                    candidate_stats[str(passage_id)]["best_endpoint_rank"] = min(
                        int(candidate_stats[str(passage_id)]["best_endpoint_rank"]),
                        int(fact_rank),
                    )
            for signature in payload.get("obj_endpoint_signatures", []):
                for passage_id in self.obj_endpoint_to_passages.get(tuple(signature), []):
                    candidate_stats[str(passage_id)]["endpoint_hits"].add(str(fact_object_id))
                    candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                        int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                        int(fact_rank),
                    )
                    candidate_stats[str(passage_id)]["best_endpoint_rank"] = min(
                        int(candidate_stats[str(passage_id)]["best_endpoint_rank"]),
                        int(fact_rank),
                    )
            for participant_text in payload.get("participant_texts", []):
                for passage_id in self.subject_entity_to_passages.get(str(participant_text), []):
                    candidate_stats[str(passage_id)]["entity_hits"].add(str(fact_object_id))
                    candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                        int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                        int(fact_rank),
                    )
                for passage_id in self.object_entity_to_passages.get(str(participant_text), []):
                    candidate_stats[str(passage_id)]["entity_hits"].add(str(fact_object_id))
                    candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                        int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                        int(fact_rank),
                    )
            for passage_id in payload.get("source_ids", []):
                candidate_stats[str(passage_id)]["source_hits"].add(str(fact_object_id))
                candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                    int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                    int(fact_rank),
                )

        query_entity_texts = self._query_entity_texts(query_state)
        for entity_id, entity_text in query_entity_texts.items():
            for passage_id in self.subject_entity_to_passages.get(entity_text, []):
                candidate_stats[str(passage_id)]["entity_hits"].add(entity_id)
            for passage_id in self.object_entity_to_passages.get(entity_text, []):
                candidate_stats[str(passage_id)]["entity_hits"].add(entity_id)

        ranked_passages: List[Tuple[Tuple[int, ...], str]] = []
        for passage_id, stats in candidate_stats.items():
            if passage_id not in self.passage_payloads:
                continue
            candidate_key = (
                len(stats["pair_hits"]),
                -int(stats["best_pair_rank"]),
                len(stats["endpoint_hits"]),
                -int(stats["best_endpoint_rank"]),
                len(stats["entity_hits"]),
                len(stats["source_hits"]),
                -int(stats["best_fact_rank"]),
                1 if passage_id in dense_rank_map else 0,
                -int(dense_rank_map.get(passage_id, dense_rank_default)),
            )
            ranked_passages.append((candidate_key, passage_id))

        candidate_passage_ids = _rank_with_dense_fallback(ranked_passages, top_k)
        candidate_bridge_ids = _unique_in_order(
            [
                bridge_id
                for fact_object_id in query_state.query_fact_ids
                for bridge_id in self.fact_to_bridge_ids.get(str(fact_object_id), [])
            ]
        )[: max(int(top_k), 0)]
        evidence_index_refs = [
            {
                "kind": "support_signature_lookup",
                "fact_object_ids": list(query_state.query_fact_ids),
                "entity_texts": list(query_entity_texts.values()),
            }
        ]
        return CandidateSet(
            channel="support",
            candidate_passage_ids=candidate_passage_ids,
            candidate_fact_ids=list(query_state.query_fact_ids[: max(int(top_k), 0)]),
            candidate_bridge_ids=candidate_bridge_ids,
            evidence_index_refs=evidence_index_refs,
            substrate_version=self.substrate_version,
        )

    def get_deficit_facts(
        self,
        query_state: QueryActivationState,
        coverage_head_passage_ids: Sequence[str],
    ) -> DeficitFactState:
        normalized_head = {str(passage_id) for passage_id in coverage_head_passage_ids if str(passage_id)}
        covered_fact_ids: List[str] = []
        deficit_fact_ids: List[str] = []
        fact_slot_demands: List[FactSlotDemand] = []
        fully_covered_fact_ids: List[str] = []
        partially_covered_fact_ids: List[str] = []
        exact_covered_slot_ids = set()
        soft_covered_slot_ids = set()
        covered_slot_ids = set()
        uncovered_slot_ids = set()
        covered_bridge_ids = set()
        uncovered_bridge_ids = set()
        covered_participant_entity_ids = {
            entity_id
            for passage_id in normalized_head
            for entity_id in self.passage_to_entities.get(passage_id, [])
        }
        head_slot_ids = {
            str(slot_id)
            for passage_id in normalized_head
            for slot_id in self.passage_to_slot_ids.get(passage_id, [])
        }
        exact_head_slot_ids = {
            str(slot_id)
            for passage_id in normalized_head
            for slot_id in self.passage_to_exact_slot_ids.get(passage_id, [])
        }
        uncovered_participant_entity_ids = set()

        score_map = {
            str(fact_object_id): float(score)
            for fact_object_id, score in zip(query_state.query_fact_ids, query_state.query_fact_scores)
        }
        for fact_object_id in query_state.query_fact_ids:
            normalized_fact_id = str(fact_object_id)
            supporting_passages = set(self.fact_to_passages.get(normalized_fact_id, []))
            if supporting_passages & normalized_head:
                covered_fact_ids.append(normalized_fact_id)
            else:
                deficit_fact_ids.append(normalized_fact_id)
                uncovered_participant_entity_ids.update(
                    self.fact_payloads.get(normalized_fact_id, {}).get("participant_entity_ids", [])
                )
            fact_slot_ids = [str(slot_id) for slot_id in self.fact_to_slot_ids.get(normalized_fact_id, [])]
            fact_exact_covered_slot_ids = sorted(slot_id for slot_id in fact_slot_ids if slot_id in exact_head_slot_ids)
            fact_soft_covered_slot_ids = sorted(
                slot_id
                for slot_id in fact_slot_ids
                if slot_id in head_slot_ids and slot_id not in exact_head_slot_ids
            )
            fact_covered_slot_ids = sorted(
                slot_id for slot_id in fact_slot_ids if slot_id in head_slot_ids
            )
            fact_uncovered_slot_ids = sorted(slot_id for slot_id in fact_slot_ids if slot_id not in head_slot_ids)
            fact_covered_bridge_ids = self._slot_bridge_ids(fact_covered_slot_ids)
            fact_uncovered_bridge_ids = self._slot_bridge_ids(fact_uncovered_slot_ids)
            fact_slot_demands.append(
                FactSlotDemand(
                    fact_object_id=normalized_fact_id,
                    all_slot_ids=list(fact_slot_ids),
                    exact_covered_slot_ids=fact_exact_covered_slot_ids,
                    soft_covered_slot_ids=fact_soft_covered_slot_ids,
                    covered_slot_ids=fact_covered_slot_ids,
                    uncovered_slot_ids=fact_uncovered_slot_ids,
                    covered_bridge_ids=fact_covered_bridge_ids,
                    uncovered_bridge_ids=fact_uncovered_bridge_ids,
                )
            )
            exact_covered_slot_ids.update(fact_exact_covered_slot_ids)
            soft_covered_slot_ids.update(fact_soft_covered_slot_ids)
            covered_slot_ids.update(fact_covered_slot_ids)
            uncovered_slot_ids.update(fact_uncovered_slot_ids)
            covered_bridge_ids.update(fact_covered_bridge_ids)
            uncovered_bridge_ids.update(fact_uncovered_bridge_ids)
            if fact_slot_ids and not fact_uncovered_slot_ids:
                fully_covered_fact_ids.append(normalized_fact_id)
            elif fact_covered_slot_ids and fact_uncovered_slot_ids:
                partially_covered_fact_ids.append(normalized_fact_id)

        deficit_scores = np.asarray(
            [max(score_map.get(str(fact_object_id), 0.0), 0.0) for fact_object_id in deficit_fact_ids],
            dtype=np.float32,
        )
        if deficit_scores.size > 0 and float(deficit_scores.sum()) > 0:
            deficit_scores = deficit_scores / float(deficit_scores.sum())
        elif deficit_scores.size > 0:
            deficit_scores = np.ones_like(deficit_scores, dtype=np.float32) / float(deficit_scores.size)

        return DeficitFactState(
            deficit_fact_ids=deficit_fact_ids,
            deficit_weights=[float(value) for value in deficit_scores.tolist()],
            covered_fact_ids=covered_fact_ids,
            fact_slot_demands=fact_slot_demands,
            fully_covered_fact_ids=fully_covered_fact_ids,
            partially_covered_fact_ids=partially_covered_fact_ids,
            exact_covered_slot_ids=sorted(exact_covered_slot_ids),
            soft_covered_slot_ids=sorted(soft_covered_slot_ids),
            covered_slot_ids=sorted(covered_slot_ids),
            uncovered_slot_ids=sorted(uncovered_slot_ids),
            covered_bridge_ids=sorted(covered_bridge_ids),
            uncovered_bridge_ids=sorted(uncovered_bridge_ids),
            covered_participant_entity_ids=sorted(covered_participant_entity_ids),
            uncovered_participant_entity_ids=sorted(uncovered_participant_entity_ids - covered_participant_entity_ids),
            evidence_index_refs=[
                {
                    "kind": "deficit_facts",
                    "coverage_head_passage_ids": sorted(normalized_head),
                    "covered_fact_ids": covered_fact_ids,
                    "deficit_fact_ids": deficit_fact_ids,
                    "exact_covered_slot_ids": sorted(exact_covered_slot_ids),
                    "soft_covered_slot_ids": sorted(soft_covered_slot_ids),
                    "covered_slot_ids": sorted(covered_slot_ids),
                    "uncovered_slot_ids": sorted(uncovered_slot_ids),
                    "fully_covered_fact_ids": fully_covered_fact_ids,
                    "partially_covered_fact_ids": partially_covered_fact_ids,
                }
            ],
            substrate_version=self.substrate_version,
        )

    def _extract_slot_tier_support_candidates(
        self,
        query_state: QueryActivationState,
        *,
        top_k: int,
    ) -> CandidateSet:
        dense_rank_map = self._dense_rank_map(query_state)
        dense_rank_default = len(dense_rank_map) + len(self.passage_ids) + 1
        fact_rank_default = len(query_state.query_fact_ids) + 1
        fact_support_lookup = self._ensure_fact_slot_tier_support_lookup()
        candidate_stats = defaultdict(
            lambda: {
                "answer_slot_hits": set(),
                "target_slot_hits": set(),
                "exact_slot_hits": set(),
                "anchor_slot_hits": set(),
                "pair_hits": set(),
                "endpoint_hits": set(),
                "entity_hits": set(),
                "source_hits": set(),
                "best_fact_rank": fact_rank_default,
                "best_answer_rank": fact_rank_default,
                "best_target_rank": fact_rank_default,
                "best_exact_rank": fact_rank_default,
                "best_anchor_rank": fact_rank_default,
                "best_pair_rank": fact_rank_default,
                "best_endpoint_rank": fact_rank_default,
            }
        )

        for fact_rank, fact_object_id in enumerate(query_state.query_fact_ids):
            normalized_fact_id = str(fact_object_id)
            fact_lookup = fact_support_lookup.get(normalized_fact_id, {})
            for passage_id in fact_lookup.get("answer", []):
                    _record_candidate_rank_hit(
                        candidate_stats[str(passage_id)],
                        hit_key="answer_slot_hits",
                        rank_key="best_answer_rank",
                        hit_id=normalized_fact_id,
                        rank_idx=fact_rank,
                    )
                    candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                        int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                        int(fact_rank),
                    )
            for passage_id in fact_lookup.get("target", []):
                    _record_candidate_rank_hit(
                        candidate_stats[str(passage_id)],
                        hit_key="target_slot_hits",
                        rank_key="best_target_rank",
                        hit_id=normalized_fact_id,
                        rank_idx=fact_rank,
                    )
                    candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                        int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                        int(fact_rank),
                    )
            for passage_id in fact_lookup.get("exact", []):
                    _record_candidate_rank_hit(
                        candidate_stats[str(passage_id)],
                        hit_key="exact_slot_hits",
                        rank_key="best_exact_rank",
                        hit_id=normalized_fact_id,
                        rank_idx=fact_rank,
                    )
                    candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                        int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                        int(fact_rank),
                    )
            for passage_id in fact_lookup.get("anchor", []):
                    _record_candidate_rank_hit(
                        candidate_stats[str(passage_id)],
                        hit_key="anchor_slot_hits",
                        rank_key="best_anchor_rank",
                        hit_id=normalized_fact_id,
                        rank_idx=fact_rank,
                    )
                    candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                        int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                        int(fact_rank),
                    )

            for passage_id in fact_lookup.get("pair", []):
                    _record_candidate_rank_hit(
                        candidate_stats[str(passage_id)],
                        hit_key="pair_hits",
                        rank_key="best_pair_rank",
                        hit_id=normalized_fact_id,
                        rank_idx=fact_rank,
                    )
                    candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                        int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                        int(fact_rank),
                    )
            for passage_id in fact_lookup.get("endpoint", []):
                    _record_candidate_rank_hit(
                        candidate_stats[str(passage_id)],
                        hit_key="endpoint_hits",
                        rank_key="best_endpoint_rank",
                        hit_id=normalized_fact_id,
                        rank_idx=fact_rank,
                    )
                    candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                        int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                        int(fact_rank),
                    )
            for passage_id in fact_lookup.get("entity", []):
                candidate_stats[str(passage_id)]["entity_hits"].add(normalized_fact_id)
                candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                    int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                    int(fact_rank),
                )
            for passage_id in fact_lookup.get("source", []):
                candidate_stats[str(passage_id)]["source_hits"].add(normalized_fact_id)
                candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                    int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                    int(fact_rank),
                )

        query_entity_texts = self._query_entity_texts(query_state)
        for entity_id, entity_text in query_entity_texts.items():
            for passage_id in self.subject_entity_to_passages.get(entity_text, []):
                candidate_stats[str(passage_id)]["entity_hits"].add(entity_id)
            for passage_id in self.object_entity_to_passages.get(entity_text, []):
                candidate_stats[str(passage_id)]["entity_hits"].add(entity_id)

        ranked_passages: List[Tuple[Tuple[int, ...], str]] = []
        for passage_id, stats in candidate_stats.items():
            if passage_id not in self.passage_payloads:
                continue
            candidate_key = (
                1 if stats["answer_slot_hits"] else 0,
                len(stats["answer_slot_hits"]),
                -int(stats["best_answer_rank"]),
                1 if stats["target_slot_hits"] else 0,
                len(stats["target_slot_hits"]),
                -int(stats["best_target_rank"]),
                1 if stats["exact_slot_hits"] else 0,
                len(stats["exact_slot_hits"]),
                -int(stats["best_exact_rank"]),
                1 if stats["anchor_slot_hits"] else 0,
                len(stats["anchor_slot_hits"]),
                -int(stats["best_anchor_rank"]),
                len(stats["pair_hits"]),
                -int(stats["best_pair_rank"]),
                len(stats["endpoint_hits"]),
                -int(stats["best_endpoint_rank"]),
                len(stats["entity_hits"]),
                len(stats["source_hits"]),
                -int(stats["best_fact_rank"]),
                1 if passage_id in dense_rank_map else 0,
                -int(dense_rank_map.get(passage_id, dense_rank_default)),
            )
            ranked_passages.append((candidate_key, passage_id))

        candidate_passage_ids = _rank_with_dense_fallback(ranked_passages, top_k)
        candidate_bridge_ids = _unique_in_order(
            [
                bridge_id
                for fact_object_id in query_state.query_fact_ids
                for bridge_id in self.fact_to_bridge_ids.get(str(fact_object_id), [])
            ]
        )[: max(int(top_k), 0)]
        evidence_index_refs = [
            {
                "kind": "slot_tier_support_lookup",
                "fact_object_ids": list(query_state.query_fact_ids),
                "entity_texts": list(query_entity_texts.values()),
                "support_tiers": ["answer", "target", "exact", "anchor"],
            }
        ]
        return CandidateSet(
            channel="support_slot_tier",
            candidate_passage_ids=candidate_passage_ids,
            candidate_fact_ids=list(query_state.query_fact_ids[: max(int(top_k), 0)]),
            candidate_bridge_ids=candidate_bridge_ids,
            evidence_index_refs=evidence_index_refs,
            substrate_version=self.substrate_version,
        )

    def _extract_slot_sourcepreserve_candidates(
        self,
        query_state: QueryActivationState,
        *,
        top_k: int,
    ) -> CandidateSet:
        dense_rank_map = self._dense_rank_map(query_state)
        dense_rank_default = len(dense_rank_map) + len(self.passage_ids) + 1
        fact_rank_default = len(query_state.query_fact_ids) + 1
        fact_support_lookup = self._ensure_fact_slot_tier_support_lookup()
        candidate_stats = defaultdict(
            lambda: {
                "answer_slot_hits": set(),
                "anchor_source_slot_hits": set(),
                "source_slot_hits": set(),
                "target_slot_hits": set(),
                "exact_slot_hits": set(),
                "anchor_slot_hits": set(),
                "pair_hits": set(),
                "endpoint_hits": set(),
                "entity_hits": set(),
                "source_hits": set(),
                "best_fact_rank": fact_rank_default,
                "best_answer_rank": fact_rank_default,
                "best_anchor_source_rank": fact_rank_default,
                "best_source_slot_rank": fact_rank_default,
                "best_target_rank": fact_rank_default,
                "best_exact_rank": fact_rank_default,
                "best_anchor_rank": fact_rank_default,
                "best_pair_rank": fact_rank_default,
                "best_endpoint_rank": fact_rank_default,
            }
        )

        for fact_rank, fact_object_id in enumerate(query_state.query_fact_ids):
            normalized_fact_id = str(fact_object_id)
            fact_lookup = fact_support_lookup.get(normalized_fact_id, {})
            for passage_id in fact_lookup.get("answer", []):
                _record_candidate_rank_hit(
                    candidate_stats[str(passage_id)],
                    hit_key="answer_slot_hits",
                    rank_key="best_answer_rank",
                    hit_id=normalized_fact_id,
                    rank_idx=fact_rank,
                )
                candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                    int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                    int(fact_rank),
                )
            for passage_id in fact_lookup.get("anchor_source", []):
                _record_candidate_rank_hit(
                    candidate_stats[str(passage_id)],
                    hit_key="anchor_source_slot_hits",
                    rank_key="best_anchor_source_rank",
                    hit_id=normalized_fact_id,
                    rank_idx=fact_rank,
                )
                candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                    int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                    int(fact_rank),
                )
            for passage_id in fact_lookup.get("source_slot", []):
                _record_candidate_rank_hit(
                    candidate_stats[str(passage_id)],
                    hit_key="source_slot_hits",
                    rank_key="best_source_slot_rank",
                    hit_id=normalized_fact_id,
                    rank_idx=fact_rank,
                )
                candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                    int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                    int(fact_rank),
                )
            for passage_id in fact_lookup.get("target", []):
                _record_candidate_rank_hit(
                    candidate_stats[str(passage_id)],
                    hit_key="target_slot_hits",
                    rank_key="best_target_rank",
                    hit_id=normalized_fact_id,
                    rank_idx=fact_rank,
                )
                candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                    int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                    int(fact_rank),
                )
            for passage_id in fact_lookup.get("exact", []):
                _record_candidate_rank_hit(
                    candidate_stats[str(passage_id)],
                    hit_key="exact_slot_hits",
                    rank_key="best_exact_rank",
                    hit_id=normalized_fact_id,
                    rank_idx=fact_rank,
                )
                candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                    int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                    int(fact_rank),
                )
            for passage_id in fact_lookup.get("anchor", []):
                _record_candidate_rank_hit(
                    candidate_stats[str(passage_id)],
                    hit_key="anchor_slot_hits",
                    rank_key="best_anchor_rank",
                    hit_id=normalized_fact_id,
                    rank_idx=fact_rank,
                )
                candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                    int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                    int(fact_rank),
                )
            for passage_id in fact_lookup.get("pair", []):
                _record_candidate_rank_hit(
                    candidate_stats[str(passage_id)],
                    hit_key="pair_hits",
                    rank_key="best_pair_rank",
                    hit_id=normalized_fact_id,
                    rank_idx=fact_rank,
                )
                candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                    int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                    int(fact_rank),
                )
            for passage_id in fact_lookup.get("endpoint", []):
                _record_candidate_rank_hit(
                    candidate_stats[str(passage_id)],
                    hit_key="endpoint_hits",
                    rank_key="best_endpoint_rank",
                    hit_id=normalized_fact_id,
                    rank_idx=fact_rank,
                )
                candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                    int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                    int(fact_rank),
                )
            for passage_id in fact_lookup.get("entity", []):
                candidate_stats[str(passage_id)]["entity_hits"].add(normalized_fact_id)
                candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                    int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                    int(fact_rank),
                )
            for passage_id in fact_lookup.get("source", []):
                candidate_stats[str(passage_id)]["source_hits"].add(normalized_fact_id)
                candidate_stats[str(passage_id)]["best_fact_rank"] = min(
                    int(candidate_stats[str(passage_id)]["best_fact_rank"]),
                    int(fact_rank),
                )

        query_entity_texts = self._query_entity_texts(query_state)
        for entity_id, entity_text in query_entity_texts.items():
            for passage_id in self.subject_entity_to_passages.get(entity_text, []):
                candidate_stats[str(passage_id)]["entity_hits"].add(entity_id)
            for passage_id in self.object_entity_to_passages.get(entity_text, []):
                candidate_stats[str(passage_id)]["entity_hits"].add(entity_id)

        ranked_passages: List[Tuple[Tuple[int, ...], str]] = []
        for passage_id, stats in candidate_stats.items():
            if passage_id not in self.passage_payloads:
                continue
            candidate_key = (
                1 if stats["answer_slot_hits"] else 0,
                len(stats["answer_slot_hits"]),
                -int(stats["best_answer_rank"]),
                1 if stats["anchor_source_slot_hits"] else 0,
                len(stats["anchor_source_slot_hits"]),
                -int(stats["best_anchor_source_rank"]),
                1 if stats["source_slot_hits"] else 0,
                len(stats["source_slot_hits"]),
                -int(stats["best_source_slot_rank"]),
                1 if stats["target_slot_hits"] else 0,
                len(stats["target_slot_hits"]),
                -int(stats["best_target_rank"]),
                1 if stats["exact_slot_hits"] else 0,
                len(stats["exact_slot_hits"]),
                -int(stats["best_exact_rank"]),
                1 if stats["anchor_slot_hits"] else 0,
                len(stats["anchor_slot_hits"]),
                -int(stats["best_anchor_rank"]),
                len(stats["pair_hits"]),
                -int(stats["best_pair_rank"]),
                len(stats["endpoint_hits"]),
                -int(stats["best_endpoint_rank"]),
                len(stats["entity_hits"]),
                len(stats["source_hits"]),
                -int(stats["best_fact_rank"]),
                1 if passage_id in dense_rank_map else 0,
                -int(dense_rank_map.get(passage_id, dense_rank_default)),
            )
            ranked_passages.append((candidate_key, passage_id))

        candidate_passage_ids = _rank_with_dense_fallback(ranked_passages, top_k)
        candidate_bridge_ids = _unique_in_order(
            [
                bridge_id
                for fact_object_id in query_state.query_fact_ids
                for bridge_id in self.fact_to_bridge_ids.get(str(fact_object_id), [])
            ]
        )[: max(int(top_k), 0)]
        evidence_index_refs = [
            {
                "kind": "slot_sourcepreserve_support_lookup",
                "fact_object_ids": list(query_state.query_fact_ids),
                "entity_texts": list(query_entity_texts.values()),
                "support_tiers": ["answer", "anchor_source", "source_slot", "target", "exact", "anchor"],
            }
        ]
        return CandidateSet(
            channel="support_slot_sourcepreserve",
            candidate_passage_ids=candidate_passage_ids,
            candidate_fact_ids=list(query_state.query_fact_ids[: max(int(top_k), 0)]),
            candidate_bridge_ids=candidate_bridge_ids,
            evidence_index_refs=evidence_index_refs,
            substrate_version=self.substrate_version,
        )

    def _extract_bridge_path_candidates(
        self,
        query_state: QueryActivationState,
        *,
        top_k: int,
    ) -> CandidateSet:
        dense_rank_map = self._dense_rank_map(query_state)
        dense_rank_default = len(dense_rank_map) + len(self.passage_ids) + 1
        fact_rank_default = len(query_state.query_fact_ids) + 1
        query_fact_set = {str(fact_id) for fact_id in query_state.query_fact_ids}
        query_entity_set = {str(entity_id) for entity_id in query_state.query_entity_ids}
        path_rank_pairs: List[Tuple[Tuple[int, ...], str]] = []
        passage_rank_pairs: List[Tuple[Tuple[int, ...], str]] = []
        candidate_path_ids = set()
        selected_path_stats = defaultdict(
            lambda: {
                "path_hits": set(),
                "fact_hits": set(),
                "entity_hits": set(),
                "best_fact_rank": fact_rank_default,
                "source_overlap": 0,
            }
        )

        fact_rank_map = {
            str(fact_id): rank_idx
            for rank_idx, fact_id in enumerate(query_state.query_fact_ids)
        }
        for fact_rank, fact_object_id in enumerate(query_state.query_fact_ids):
            normalized_fact_id = str(fact_object_id)
            for path_id in self.fact_to_bridge_path_ids.get(normalized_fact_id, []):
                payload = self.bridge_path_payloads.get(str(path_id), {})
                if not payload:
                    continue
                path_fact_ids = [
                    str(fact_id)
                    for fact_id in self.bridge_path_to_facts.get(str(path_id), [])
                    if str(fact_id)
                ]
                query_fact_hits = len(set(path_fact_ids) & query_fact_set)
                shared_entity_ids = {str(entity_id) for entity_id in payload.get("shared_entity_ids", [])}
                query_entity_hits = len(shared_entity_ids & query_entity_set)
                source_overlap = int(bool(payload.get("source_overlap", False)))
                support_pair_count = len(payload.get("support_passage_pair_ids", []))
                transition_priority = {
                    "relation_continuation": 4,
                    "comparison_operand_pair": 3,
                    "shared_entity_bridge": 2,
                    "same_source_chain": 1,
                    "source_to_entity_expansion": 0,
                }.get(str(payload.get("transition_type", "")), 0)
                partner_best_rank = min(
                    [fact_rank_map.get(fact_id, fact_rank_default) for fact_id in path_fact_ids]
                    or [fact_rank_default]
                )
                path_key = (
                    query_fact_hits,
                    query_entity_hits,
                    int(payload.get("entity_overlap", 0)),
                    transition_priority,
                    source_overlap,
                    support_pair_count,
                    -int(partner_best_rank),
                    -int(fact_rank),
                )
                path_rank_pairs.append((path_key, str(path_id)))
                candidate_path_ids.add(str(path_id))

        selected_path_ids = _rank_with_dense_fallback(path_rank_pairs, top_k=max(int(top_k) * 2, int(top_k)))
        for path_id in selected_path_ids:
            payload = self.bridge_path_payloads.get(str(path_id), {})
            path_fact_ids = [
                str(fact_id)
                for fact_id in self.bridge_path_to_facts.get(str(path_id), [])
                if str(fact_id)
            ]
            shared_entity_ids = {str(entity_id) for entity_id in payload.get("shared_entity_ids", [])}
            path_passage_ids = self.bridge_path_to_passages.get(str(path_id), [])
            for passage_id in path_passage_ids:
                normalized_passage_id = str(passage_id)
                stats = selected_path_stats[normalized_passage_id]
                stats["path_hits"].add(str(path_id))
                stats["fact_hits"].update(path_fact_ids)
                stats["entity_hits"].update(shared_entity_ids & query_entity_set)
                stats["source_overlap"] = max(int(stats["source_overlap"]), int(bool(payload.get("source_overlap", False))))
                best_fact_rank = min(
                    [fact_rank_map.get(fact_id, fact_rank_default) for fact_id in path_fact_ids]
                    or [fact_rank_default]
                )
                stats["best_fact_rank"] = min(int(stats["best_fact_rank"]), int(best_fact_rank))

        for passage_id, stats in selected_path_stats.items():
            if passage_id not in self.passage_payloads:
                continue
            passage_key = (
                len(stats["path_hits"]),
                len(stats["fact_hits"] & query_fact_set),
                len(stats["entity_hits"]),
                int(stats["source_overlap"]),
                -int(stats["best_fact_rank"]),
                1 if passage_id in dense_rank_map else 0,
                -int(dense_rank_map.get(passage_id, dense_rank_default)),
            )
            passage_rank_pairs.append((passage_key, passage_id))

        candidate_passage_ids = _rank_with_dense_fallback(passage_rank_pairs, top_k)
        candidate_fact_ids = _unique_in_order(
            [
                fact_id
                for path_id in selected_path_ids
                for fact_id in self.bridge_path_to_facts.get(str(path_id), [])
            ]
        )[: max(int(top_k), 0)]
        candidate_bridge_ids = _unique_in_order(
            [
                bridge_id
                for path_id in selected_path_ids
                for bridge_id in (
                    self.bridge_path_payloads.get(str(path_id), {}).get("start_bridge_id", ""),
                    self.bridge_path_payloads.get(str(path_id), {}).get("end_bridge_id", ""),
                )
                if bridge_id
            ]
        )[: max(int(top_k), 0)]
        evidence_index_refs = [
            {
                "kind": "bridge_path_lookup",
                "fact_object_ids": list(query_state.query_fact_ids),
                "entity_ids": list(query_state.query_entity_ids),
                "bridge_path_ids": list(selected_path_ids),
                "candidate_path_count": len(candidate_path_ids),
            }
        ]
        return CandidateSet(
            channel="bridge_path",
            candidate_passage_ids=candidate_passage_ids,
            candidate_fact_ids=candidate_fact_ids,
            candidate_bridge_ids=candidate_bridge_ids,
            evidence_index_refs=evidence_index_refs,
            substrate_version=self.substrate_version,
        )

    def _extract_support_trail_candidates(
        self,
        query_state: QueryActivationState,
        *,
        top_k: int,
    ) -> CandidateSet:
        if not self.support_trail_payloads and self.fact_to_support_trail_membership:
            return self._extract_support_trail_membership_candidates(
                query_state,
                top_k=top_k,
                channel="support_trail",
            )

        dense_rank_map = self._dense_rank_map(query_state)
        dense_rank_default = len(dense_rank_map) + len(self.passage_ids) + 1
        fact_rank_default = len(query_state.query_fact_ids) + 1
        query_fact_set = {str(fact_id) for fact_id in query_state.query_fact_ids}
        fact_rank_map = {
            str(fact_id): rank_idx
            for rank_idx, fact_id in enumerate(query_state.query_fact_ids)
        }
        trail_rank_pairs: List[Tuple[Tuple[int, ...], str]] = []
        candidate_trail_ids = set()
        selected_trail_stats = defaultdict(
            lambda: {
                "trail_hits": set(),
                "fact_hits": set(),
                "best_fact_rank": fact_rank_default,
                "min_hop_count": 999,
                "endpoint_hits": 0,
            }
        )

        for fact_rank, fact_object_id in enumerate(query_state.query_fact_ids):
            normalized_fact_id = str(fact_object_id)
            for trail_id in self.fact_to_support_trail_ids.get(normalized_fact_id, []):
                payload = self.support_trail_payloads.get(str(trail_id), {})
                if not payload:
                    continue
                trail_fact_ids = [
                    str(fact_id)
                    for fact_id in self.support_trail_to_facts.get(str(trail_id), [])
                    if str(fact_id)
                ]
                query_fact_hits = len(set(trail_fact_ids) & query_fact_set)
                hop_count = int(payload.get("hop_count", len(payload.get("bridge_path_id_sequence", [])) or 999))
                endpoint_hit = int(str(payload.get("end_fact_id", "")) in query_fact_set)
                best_fact_rank = min(
                    [fact_rank_map.get(fact_id, fact_rank_default) for fact_id in trail_fact_ids]
                    or [fact_rank_default]
                )
                trail_key = (
                    query_fact_hits,
                    -hop_count,
                    endpoint_hit,
                    -int(best_fact_rank),
                    -int(fact_rank),
                )
                trail_rank_pairs.append((trail_key, str(trail_id)))
                candidate_trail_ids.add(str(trail_id))

        selected_trail_ids = _rank_with_dense_fallback(
            trail_rank_pairs,
            top_k=max(int(top_k) * 2, int(top_k)),
        )
        for trail_id in selected_trail_ids:
            payload = self.support_trail_payloads.get(str(trail_id), {})
            trail_fact_ids = [
                str(fact_id)
                for fact_id in self.support_trail_to_facts.get(str(trail_id), [])
                if str(fact_id)
            ]
            trail_passage_ids = self.support_trail_to_passages.get(str(trail_id), [])
            hop_count = int(payload.get("hop_count", len(payload.get("bridge_path_id_sequence", [])) or 999))
            endpoint_passages = {str(passage_id) for passage_id in payload.get("end_source_passage_ids", [])}
            best_fact_rank = min(
                [fact_rank_map.get(fact_id, fact_rank_default) for fact_id in trail_fact_ids]
                or [fact_rank_default]
            )
            for passage_id in trail_passage_ids:
                normalized_passage_id = str(passage_id)
                stats = selected_trail_stats[normalized_passage_id]
                stats["trail_hits"].add(str(trail_id))
                stats["fact_hits"].update(trail_fact_ids)
                stats["best_fact_rank"] = min(int(stats["best_fact_rank"]), int(best_fact_rank))
                stats["min_hop_count"] = min(int(stats["min_hop_count"]), hop_count)
                stats["endpoint_hits"] = max(
                    int(stats["endpoint_hits"]),
                    int(normalized_passage_id in endpoint_passages),
                )

        passage_rank_pairs: List[Tuple[Tuple[int, ...], str]] = []
        for passage_id, stats in selected_trail_stats.items():
            if passage_id not in self.passage_payloads:
                continue
            passage_key = (
                len(stats["trail_hits"]),
                int(stats["endpoint_hits"]),
                len(stats["fact_hits"] & query_fact_set),
                -int(stats["min_hop_count"]),
                -int(stats["best_fact_rank"]),
                1 if passage_id in dense_rank_map else 0,
                -int(dense_rank_map.get(passage_id, dense_rank_default)),
            )
            passage_rank_pairs.append((passage_key, passage_id))

        candidate_passage_ids = _rank_with_dense_fallback(passage_rank_pairs, top_k)
        candidate_fact_ids = _unique_in_order(
            [
                fact_id
                for trail_id in selected_trail_ids
                for fact_id in (
                    [self.support_trail_payloads.get(str(trail_id), {}).get("end_fact_id", "")]
                    + self.support_trail_to_facts.get(str(trail_id), [])
                )
                if fact_id
            ]
        )[: max(int(top_k), 0)]
        candidate_bridge_ids = _unique_in_order(
            [
                bridge_id
                for trail_id in selected_trail_ids
                for path_id in self.support_trail_to_bridge_path_ids.get(str(trail_id), [])
                for bridge_id in (
                    self.bridge_path_payloads.get(str(path_id), {}).get("start_bridge_id", ""),
                    self.bridge_path_payloads.get(str(path_id), {}).get("end_bridge_id", ""),
                )
                if bridge_id
            ]
        )[: max(int(top_k), 0)]
        evidence_index_refs = [
            {
                "kind": "support_trail_lookup",
                "fact_object_ids": list(query_state.query_fact_ids),
                "support_trail_ids": list(selected_trail_ids),
                "candidate_trail_count": len(candidate_trail_ids),
            }
        ]
        return CandidateSet(
            channel="support_trail",
            candidate_passage_ids=candidate_passage_ids,
            candidate_fact_ids=candidate_fact_ids,
            candidate_bridge_ids=candidate_bridge_ids,
            evidence_index_refs=evidence_index_refs,
            substrate_version=self.substrate_version,
        )

    def _extract_support_trail_membership_candidates(
        self,
        query_state: QueryActivationState,
        *,
        top_k: int,
        channel: str,
    ) -> CandidateSet:
        dense_rank_map = self._dense_rank_map(query_state)
        dense_rank_default = len(dense_rank_map) + len(self.passage_ids) + 1
        fact_rank_default = len(query_state.query_fact_ids) + 1
        query_fact_set = {str(fact_id) for fact_id in query_state.query_fact_ids}
        fact_rank_map = {
            str(fact_id): rank_idx
            for rank_idx, fact_id in enumerate(query_state.query_fact_ids)
        }
        selected_memberships: List[Dict[str, Any]] = []
        selected_membership_ids = set()
        selected_trail_ids = set()
        selected_passage_stats = defaultdict(
            lambda: {
                "membership_hits": set(),
                "trail_hits": set(),
                "fact_hits": set(),
                "best_fact_rank": fact_rank_default,
                "endpoint_hits": 0,
            }
        )

        for fact_object_id in query_state.query_fact_ids:
            normalized_fact_id = str(fact_object_id)
            payload = self.fact_to_support_trail_membership.get(normalized_fact_id, {})
            membership_id = str(
                payload.get("support_trail_membership_id")
                or payload.get("membership_id")
                or ""
            )
            if not payload or not membership_id or membership_id in selected_membership_ids:
                continue
            selected_membership_ids.add(membership_id)
            selected_memberships.append(payload)

        for payload in selected_memberships:
            membership_id = str(payload.get("support_trail_membership_id") or payload.get("membership_id"))
            trail_ids = _unique_in_order(payload.get("support_trail_ids", []) or [])
            support_fact_ids = _unique_in_order(payload.get("support_fact_ids", []) or [])
            support_passage_ids = _unique_in_order(payload.get("support_passage_ids", []) or [])
            endpoint_passage_ids = {str(value) for value in payload.get("endpoint_passage_ids", []) or []}
            selected_trail_ids.update(trail_ids)
            best_fact_rank = min(
                [fact_rank_map.get(fact_id, fact_rank_default) for fact_id in support_fact_ids]
                or [fact_rank_default]
            )
            for passage_id in support_passage_ids:
                normalized_passage_id = str(passage_id)
                stats = selected_passage_stats[normalized_passage_id]
                stats["membership_hits"].add(membership_id)
                stats["trail_hits"].update(trail_ids)
                stats["fact_hits"].update(support_fact_ids)
                stats["best_fact_rank"] = min(int(stats["best_fact_rank"]), int(best_fact_rank))
                stats["endpoint_hits"] = max(
                    int(stats["endpoint_hits"]),
                    int(normalized_passage_id in endpoint_passage_ids),
                )

        passage_rank_pairs: List[Tuple[Tuple[int, ...], str]] = []
        for passage_id, stats in selected_passage_stats.items():
            if passage_id not in self.passage_payloads:
                continue
            passage_key = (
                len(stats["membership_hits"]),
                len(stats["trail_hits"]),
                int(stats["endpoint_hits"]),
                len(stats["fact_hits"] & query_fact_set),
                -int(stats["best_fact_rank"]),
                1 if passage_id in dense_rank_map else 0,
                -int(dense_rank_map.get(passage_id, dense_rank_default)),
            )
            passage_rank_pairs.append((passage_key, passage_id))

        candidate_passage_ids = _rank_with_dense_fallback(passage_rank_pairs, top_k)
        candidate_fact_ids = _unique_in_order(
            [
                fact_id
                for payload in selected_memberships
                for fact_id in payload.get("support_fact_ids", [])
                if fact_id
            ]
        )[: max(int(top_k), 0)]
        candidate_bridge_ids = _unique_in_order(
            [
                bridge_id
                for payload in selected_memberships
                for path_id in payload.get("bridge_path_ids", [])
                for bridge_id in (
                    self.bridge_path_payloads.get(str(path_id), {}).get("start_bridge_id", ""),
                    self.bridge_path_payloads.get(str(path_id), {}).get("end_bridge_id", ""),
                )
                if bridge_id
            ]
        )[: max(int(top_k), 0)]
        evidence_index_refs = [
            {
                "kind": "support_trail_membership_lookup",
                "fact_object_ids": list(query_state.query_fact_ids),
                "support_trail_membership_ids": sorted(selected_membership_ids),
                "support_trail_ids": sorted(selected_trail_ids),
                "candidate_membership_count": len(selected_membership_ids),
                "candidate_trail_count": len(selected_trail_ids),
            }
        ]
        return CandidateSet(
            channel=channel,
            candidate_passage_ids=candidate_passage_ids,
            candidate_fact_ids=candidate_fact_ids,
            candidate_bridge_ids=candidate_bridge_ids,
            evidence_index_refs=evidence_index_refs,
            substrate_version=self.substrate_version,
        )

    def _extract_support_endpoint_candidates(
        self,
        query_state: QueryActivationState,
        *,
        top_k: int,
        channel: str,
    ) -> CandidateSet:
        dense_rank_map = self._dense_rank_map(query_state)
        dense_rank_default = len(dense_rank_map) + len(self.passage_ids) + 1
        fact_rank_default = len(query_state.query_fact_ids) + 1
        fact_rank_map = {
            str(fact_id): rank_idx
            for rank_idx, fact_id in enumerate(query_state.query_fact_ids)
        }
        selected_endpoint_ids = set()
        selected_trail_ids = set()
        selected_bridge_path_ids = set()
        selected_endpoint_fact_ids = set()
        selected_passage_stats = defaultdict(
            lambda: {
                "endpoint_hits": set(),
                "trail_hits": set(),
                "bridge_path_hits": set(),
                "endpoint_fact_hits": set(),
                "best_fact_rank": fact_rank_default,
                "transition_types": set(),
                "relation_families": set(),
            }
        )

        for fact_object_id in query_state.query_fact_ids:
            owner_fact_id = str(fact_object_id)
            owner_rank = fact_rank_map.get(owner_fact_id, fact_rank_default)
            for endpoint_id in self.fact_to_support_endpoint_ids.get(owner_fact_id, []) or []:
                normalized_endpoint_id = str(endpoint_id)
                payload = self.support_endpoint_payloads.get(normalized_endpoint_id, {})
                if not payload or normalized_endpoint_id in selected_endpoint_ids:
                    continue
                selected_endpoint_ids.add(normalized_endpoint_id)
                endpoint_fact_id = str(payload.get("endpoint_fact_id", ""))
                if endpoint_fact_id:
                    selected_endpoint_fact_ids.add(endpoint_fact_id)
                support_trail_ids = _unique_in_order(payload.get("support_trail_ids", []) or [])
                bridge_path_ids = _unique_in_order(payload.get("bridge_path_ids", []) or [])
                selected_trail_ids.update(support_trail_ids)
                selected_bridge_path_ids.update(bridge_path_ids)
                for passage_id in _unique_in_order(payload.get("endpoint_passage_ids", []) or []):
                    stats = selected_passage_stats[str(passage_id)]
                    stats["endpoint_hits"].add(normalized_endpoint_id)
                    stats["trail_hits"].update(support_trail_ids)
                    stats["bridge_path_hits"].update(bridge_path_ids)
                    if endpoint_fact_id:
                        stats["endpoint_fact_hits"].add(endpoint_fact_id)
                    stats["best_fact_rank"] = min(int(stats["best_fact_rank"]), int(owner_rank))
                    stats["transition_types"].update(_unique_in_order(payload.get("transition_types", []) or []))
                    stats["relation_families"].update(
                        _unique_in_order(payload.get("endpoint_relation_families", []) or [])
                    )

        passage_rank_pairs: List[Tuple[Tuple[int, ...], str]] = []
        for passage_id, stats in selected_passage_stats.items():
            if passage_id not in self.passage_payloads:
                continue
            passage_key = (
                len(stats["endpoint_hits"]),
                len(stats["trail_hits"]),
                len(stats["bridge_path_hits"]),
                len(stats["endpoint_fact_hits"]),
                -int(stats["best_fact_rank"]),
                1 if passage_id in dense_rank_map else 0,
                -int(dense_rank_map.get(passage_id, dense_rank_default)),
            )
            passage_rank_pairs.append((passage_key, passage_id))

        candidate_passage_ids = _rank_with_dense_fallback(passage_rank_pairs, top_k)
        candidate_fact_ids = _unique_in_order(sorted(selected_endpoint_fact_ids))[: max(int(top_k), 0)]
        candidate_bridge_ids = _unique_in_order(
            [
                bridge_id
                for path_id in sorted(selected_bridge_path_ids)
                for bridge_id in (
                    self.bridge_path_payloads.get(str(path_id), {}).get("start_bridge_id", ""),
                    self.bridge_path_payloads.get(str(path_id), {}).get("end_bridge_id", ""),
                )
                if bridge_id
            ]
        )[: max(int(top_k), 0)]
        evidence_index_refs = [
            {
                "kind": "support_endpoint_lookup",
                "fact_object_ids": list(query_state.query_fact_ids),
                "support_endpoint_ids": sorted(selected_endpoint_ids),
                "support_trail_ids": sorted(selected_trail_ids),
                "bridge_path_ids": sorted(selected_bridge_path_ids),
                "candidate_endpoint_count": len(selected_endpoint_ids),
            }
        ]
        return CandidateSet(
            channel=channel,
            candidate_passage_ids=candidate_passage_ids,
            candidate_fact_ids=candidate_fact_ids,
            candidate_bridge_ids=candidate_bridge_ids,
            evidence_index_refs=evidence_index_refs,
            substrate_version=self.substrate_version,
        )

    def _extract_route_candidates(
        self,
        query_state: QueryActivationState,
        *,
        top_k: int,
        head_passage_ids: Sequence[str],
    ) -> CandidateSet:
        deficit_state = self.get_deficit_facts(query_state, coverage_head_passage_ids=head_passage_ids)
        head_entity_ids = set(deficit_state.covered_participant_entity_ids)
        bridge_rank_pairs: List[Tuple[Tuple[int, ...], str]] = []
        passage_rank_pairs: List[Tuple[Tuple[int, ...], str]] = []
        dense_rank_map = self._dense_rank_map(query_state)
        dense_rank_default = len(dense_rank_map) + len(self.passage_ids) + 1

        selected_passage_stats: Dict[str, Dict[str, set]] = defaultdict(lambda: {"bridge_hits": set(), "fact_hits": set()})

        for rank_idx, fact_object_id in enumerate(deficit_state.deficit_fact_ids):
            for bridge_id in self.fact_to_bridge_ids.get(str(fact_object_id), []):
                bridge_payload = self.bridge_payloads.get(str(bridge_id), {})
                participant_overlap = len(set(bridge_payload.get("participant_entity_ids", [])) & head_entity_ids)
                source_ids = [str(value) for value in bridge_payload.get("source_ids", [])]
                bridge_key = (
                    participant_overlap,
                    len(source_ids),
                    1 if bridge_id in query_state.query_bridge_hints else 0,
                    -rank_idx,
                )
                bridge_rank_pairs.append((bridge_key, str(bridge_id)))
                for passage_id in source_ids:
                    selected_passage_stats[str(passage_id)]["bridge_hits"].add(str(bridge_id))
                    selected_passage_stats[str(passage_id)]["fact_hits"].add(str(fact_object_id))

        for passage_id, stats in selected_passage_stats.items():
            if passage_id not in self.passage_payloads:
                continue
            passage_key = (
                len(stats["fact_hits"]),
                len(stats["bridge_hits"]),
                1 if passage_id in dense_rank_map else 0,
                -int(dense_rank_map.get(passage_id, dense_rank_default)),
            )
            passage_rank_pairs.append((passage_key, passage_id))

        candidate_bridge_ids = _rank_with_dense_fallback(bridge_rank_pairs, top_k)
        candidate_passage_ids = _rank_with_dense_fallback(passage_rank_pairs, top_k)
        evidence_index_refs = list(deficit_state.evidence_index_refs) + [
            {
                "kind": "route_bridge_lookup",
                "bridge_ids": list(candidate_bridge_ids),
            }
        ]
        return CandidateSet(
            channel="route",
            candidate_passage_ids=candidate_passage_ids,
            candidate_fact_ids=list(deficit_state.deficit_fact_ids[: max(int(top_k), 0)]),
            candidate_bridge_ids=candidate_bridge_ids,
            evidence_index_refs=evidence_index_refs,
            substrate_version=self.substrate_version,
        )

    def extract_local_subgraph(
        self,
        query_state: QueryActivationState,
        candidate_passage_ids: Sequence[str],
        channels: Sequence[str],
    ) -> LocalSubgraph:
        selected_channels = {str(channel).strip().lower() for channel in channels if str(channel).strip()}
        selected_passages = [str(passage_id) for passage_id in candidate_passage_ids if str(passage_id) in self.passage_payloads]
        selected_entities = set(query_state.query_entity_ids)
        selected_facts = set(query_state.query_fact_ids)
        selected_bridges = set(query_state.query_bridge_hints)
        selected_bridge_paths = set()

        for passage_id in selected_passages:
            selected_entities.update(self.passage_to_entities.get(passage_id, []))
            selected_facts.update(self.passage_to_fact_ids.get(passage_id, []))
            selected_bridges.update(self.passage_to_bridge_ids.get(passage_id, []))
            selected_bridge_paths.update(self.passage_to_bridge_path_ids.get(passage_id, []))

        for fact_object_id in list(selected_facts):
            selected_bridges.update(self.fact_to_bridge_ids.get(str(fact_object_id), []))
            selected_bridge_paths.update(self.fact_to_bridge_path_ids.get(str(fact_object_id), []))
            fact_payload = self.fact_payloads.get(str(fact_object_id), {})
            selected_entities.update(fact_payload.get("participant_entity_ids", []))

        for bridge_id in list(selected_bridges):
            bridge_payload = self.bridge_payloads.get(str(bridge_id), {})
            selected_entities.update(bridge_payload.get("participant_entity_ids", []))

        for path_id in list(selected_bridge_paths):
            path_payload = self.bridge_path_payloads.get(str(path_id), {})
            selected_facts.update(self.bridge_path_to_facts.get(str(path_id), []))
            start_bridge_id = str(path_payload.get("start_bridge_id", "")).strip()
            end_bridge_id = str(path_payload.get("end_bridge_id", "")).strip()
            if start_bridge_id:
                selected_bridges.add(start_bridge_id)
            if end_bridge_id:
                selected_bridges.add(end_bridge_id)
            selected_entities.update(path_payload.get("shared_entity_ids", []))

        edge_rows: List[Dict[str, Any]] = []
        for passage_id in selected_passages:
            for entity_id in self.passage_to_entities.get(passage_id, []):
                if entity_id in selected_entities:
                    edge_rows.append({"edge_type": "entity_passage", "source_id": entity_id, "target_id": passage_id})
            for fact_object_id in self.passage_to_fact_ids.get(passage_id, []):
                if fact_object_id in selected_facts:
                    edge_rows.append({"edge_type": "passage_fact", "source_id": passage_id, "target_id": fact_object_id})
            for bridge_id in self.passage_to_bridge_ids.get(passage_id, []):
                if bridge_id in selected_bridges:
                    edge_rows.append({"edge_type": "bridge_passage", "source_id": bridge_id, "target_id": passage_id})
            for path_id in self.passage_to_bridge_path_ids.get(passage_id, []):
                if path_id in selected_bridge_paths:
                    edge_rows.append({"edge_type": "bridge_path_passage", "source_id": path_id, "target_id": passage_id})

        for fact_object_id in selected_facts:
            for bridge_id in self.fact_to_bridge_ids.get(str(fact_object_id), []):
                if bridge_id in selected_bridges:
                    edge_rows.append({"edge_type": "fact_bridge", "source_id": fact_object_id, "target_id": bridge_id})

        for bridge_id in selected_bridges:
            bridge_payload = self.bridge_payloads.get(str(bridge_id), {})
            for entity_id in bridge_payload.get("participant_entity_ids", []):
                if entity_id in selected_entities:
                    edge_rows.append({"edge_type": "entity_bridge", "source_id": entity_id, "target_id": bridge_id})

        for path_id in selected_bridge_paths:
            path_payload = self.bridge_path_payloads.get(str(path_id), {})
            for fact_object_id in self.bridge_path_to_facts.get(str(path_id), []):
                if fact_object_id in selected_facts:
                    edge_rows.append({"edge_type": "fact_bridge_path", "source_id": fact_object_id, "target_id": path_id})
            for bridge_id in (
                str(path_payload.get("start_bridge_id", "")).strip(),
                str(path_payload.get("end_bridge_id", "")).strip(),
            ):
                if bridge_id and bridge_id in selected_bridges:
                    edge_rows.append({"edge_type": "bridge_path_bridge", "source_id": path_id, "target_id": bridge_id})
            for entity_id in path_payload.get("shared_entity_ids", []):
                if entity_id in selected_entities:
                    edge_rows.append({"edge_type": "entity_bridge_path", "source_id": entity_id, "target_id": path_id})

        return LocalSubgraph(
            node_ids={
                "entity_ids": sorted(selected_entities),
                "fact_object_ids": sorted(selected_facts),
                "bridge_ids": sorted(selected_bridges),
                "bridge_path_ids": sorted(selected_bridge_paths),
                "passage_ids": list(selected_passages),
            },
            edge_rows=edge_rows,
            passage_inventories={
                passage_id: dict(self.passage_inventories.get(passage_id, {}))
                for passage_id in selected_passages
                if passage_id in self.passage_inventories
            },
            fact_to_passages={
                fact_object_id: list(self.fact_to_passages.get(str(fact_object_id), []))
                for fact_object_id in sorted(selected_facts)
            },
            bridge_neighbors={
                bridge_id: list(self.bridge_neighbors.get(str(bridge_id), []))
                for bridge_id in sorted(selected_bridges)
            },
            evidence_index_refs=[
                {
                    "kind": "local_subgraph",
                    "channels": sorted(selected_channels),
                    "candidate_passage_ids": list(selected_passages),
                }
            ],
            substrate_version=self.substrate_version,
        )


def build_canonical_retrieval_interface_from_runtime(runtime: Any) -> CanonicalRetrievalInterface:
    return CanonicalRetrievalInterface.from_runtime(runtime)


def build_canonical_interface_cache_payload(
    interface: CanonicalRetrievalInterface,
) -> Dict[str, Any]:
    return {
        "cache_version": CANONICAL_INTERFACE_CACHE_VERSION,
        "substrate_version": interface.substrate_version,
        "entity_payloads": interface.entity_payloads,
        "fact_payloads": interface.fact_payloads,
        "bridge_payloads": interface.bridge_payloads,
        "slot_payloads": interface.slot_payloads,
        "bridge_path_payloads": interface.bridge_path_payloads,
        "passage_payloads": interface.passage_payloads,
        "passage_inventories": interface.passage_inventories,
        "entity_to_passages": interface.entity_to_passages,
        "entity_to_bridges": interface.entity_to_bridges,
        "entity_to_bridge_path_ids": interface.entity_to_bridge_path_ids,
        "fact_to_passages": interface.fact_to_passages,
        "fact_to_bridge_ids": interface.fact_to_bridge_ids,
        "fact_to_slot_ids": interface.fact_to_slot_ids,
        "fact_to_bridge_path_ids": interface.fact_to_bridge_path_ids,
        "bridge_to_passages": interface.bridge_to_passages,
        "bridge_to_fact_id": interface.bridge_to_fact_id,
        "bridge_to_slot_ids": interface.bridge_to_slot_ids,
        "bridge_neighbors": interface.bridge_neighbors,
        "bridge_path_to_passages": interface.bridge_path_to_passages,
        "bridge_path_to_facts": interface.bridge_path_to_facts,
        "support_trail_payloads": interface.support_trail_payloads,
        "support_trail_membership_payloads": interface.support_trail_membership_payloads,
        "fact_to_support_trail_membership": interface.fact_to_support_trail_membership,
        "fact_to_support_trail_ids": interface.fact_to_support_trail_ids,
        "passage_to_support_trail_ids": interface.passage_to_support_trail_ids,
        "support_trail_to_passages": interface.support_trail_to_passages,
        "support_trail_to_facts": interface.support_trail_to_facts,
        "support_trail_to_bridge_path_ids": interface.support_trail_to_bridge_path_ids,
        "support_endpoint_payloads": interface.support_endpoint_payloads,
        "fact_to_support_endpoint_ids": interface.fact_to_support_endpoint_ids,
        "endpoint_fact_to_support_endpoint_ids": interface.endpoint_fact_to_support_endpoint_ids,
        "passage_to_support_endpoint_ids": interface.passage_to_support_endpoint_ids,
        "support_endpoint_to_passages": interface.support_endpoint_to_passages,
        "support_endpoint_to_facts": interface.support_endpoint_to_facts,
        "support_endpoint_to_trail_ids": interface.support_endpoint_to_trail_ids,
        "support_endpoint_to_bridge_path_ids": interface.support_endpoint_to_bridge_path_ids,
        "slot_to_passages": interface.slot_to_passages,
        "slot_to_bridge_ids": interface.slot_to_bridge_ids,
        "passage_to_bridge_path_ids": interface.passage_to_bridge_path_ids,
        "pair_to_passages": interface.pair_to_passages,
        "subj_endpoint_to_passages": interface.subj_endpoint_to_passages,
        "obj_endpoint_to_passages": interface.obj_endpoint_to_passages,
        "subject_entity_to_passages": interface.subject_entity_to_passages,
        "object_entity_to_passages": interface.object_entity_to_passages,
        "slot_to_exact_passages": interface.slot_to_exact_passages,
        "slot_to_endpoint_passages": interface.slot_to_endpoint_passages,
        "slot_to_entity_passages": interface.slot_to_entity_passages,
        "slot_to_support_passages": interface.slot_to_support_passages,
        "slot_to_source_support_passages": interface.slot_to_source_support_passages,
        "slot_to_anchor_source_support_passages": interface.slot_to_anchor_source_support_passages,
        "slot_to_anchor_support_passages": interface.slot_to_anchor_support_passages,
        "slot_to_target_support_passages": interface.slot_to_target_support_passages,
        "slot_to_answer_support_passages": interface.slot_to_answer_support_passages,
        "passage_to_exact_slot_ids": interface.passage_to_exact_slot_ids,
    }


def load_canonical_interface_from_cache_payload(
    payload: Mapping[str, Any],
    *,
    runtime: Any | None = None,
) -> CanonicalRetrievalInterface:
    cache_version = int(payload.get("cache_version", 0))
    if cache_version not in {1, 3, 4, 5, 6, CANONICAL_INTERFACE_CACHE_VERSION}:
        raise ValueError(
            f"Unsupported canonical interface cache version: {payload.get('cache_version')}"
        )
    return CanonicalRetrievalInterface(
        substrate_version=str(payload["substrate_version"]),
        entity_payloads=dict(payload["entity_payloads"]),
        fact_payloads=dict(payload["fact_payloads"]),
        bridge_payloads=dict(payload["bridge_payloads"]),
        slot_payloads=dict(payload["slot_payloads"]),
        bridge_path_payloads=dict(payload.get("bridge_path_payloads", {})),
        passage_payloads=dict(payload["passage_payloads"]),
        passage_inventories=dict(payload["passage_inventories"]),
        entity_to_passages=dict(payload["entity_to_passages"]),
        entity_to_bridges=dict(payload["entity_to_bridges"]),
        entity_to_bridge_path_ids=dict(payload.get("entity_to_bridge_path_ids", {})),
        fact_to_passages=dict(payload["fact_to_passages"]),
        fact_to_bridge_ids=dict(payload["fact_to_bridge_ids"]),
        fact_to_slot_ids=dict(payload["fact_to_slot_ids"]),
        fact_to_bridge_path_ids=dict(payload.get("fact_to_bridge_path_ids", {})),
        bridge_to_passages=dict(payload["bridge_to_passages"]),
        bridge_to_fact_id=dict(payload["bridge_to_fact_id"]),
        bridge_to_slot_ids=dict(payload["bridge_to_slot_ids"]),
        bridge_neighbors=dict(payload["bridge_neighbors"]),
        bridge_path_to_passages=dict(payload.get("bridge_path_to_passages", {})),
        bridge_path_to_facts=dict(payload.get("bridge_path_to_facts", {})),
        support_trail_payloads=dict(payload.get("support_trail_payloads", {})),
        support_trail_membership_payloads=dict(payload.get("support_trail_membership_payloads", {})),
        fact_to_support_trail_membership=dict(payload.get("fact_to_support_trail_membership", {})),
        fact_to_support_trail_ids=dict(payload.get("fact_to_support_trail_ids", {})),
        passage_to_support_trail_ids=dict(payload.get("passage_to_support_trail_ids", {})),
        support_trail_to_passages=dict(payload.get("support_trail_to_passages", {})),
        support_trail_to_facts=dict(payload.get("support_trail_to_facts", {})),
        support_trail_to_bridge_path_ids=dict(payload.get("support_trail_to_bridge_path_ids", {})),
        support_endpoint_payloads=dict(payload.get("support_endpoint_payloads", {})),
        fact_to_support_endpoint_ids=dict(payload.get("fact_to_support_endpoint_ids", {})),
        endpoint_fact_to_support_endpoint_ids=dict(payload.get("endpoint_fact_to_support_endpoint_ids", {})),
        passage_to_support_endpoint_ids=dict(payload.get("passage_to_support_endpoint_ids", {})),
        support_endpoint_to_passages=dict(payload.get("support_endpoint_to_passages", {})),
        support_endpoint_to_facts=dict(payload.get("support_endpoint_to_facts", {})),
        support_endpoint_to_trail_ids=dict(payload.get("support_endpoint_to_trail_ids", {})),
        support_endpoint_to_bridge_path_ids=dict(payload.get("support_endpoint_to_bridge_path_ids", {})),
        slot_to_passages=dict(payload["slot_to_passages"]),
        slot_to_bridge_ids=dict(payload["slot_to_bridge_ids"]),
        passage_to_bridge_path_ids=dict(payload.get("passage_to_bridge_path_ids", {})),
        pair_to_passages=dict(payload["pair_to_passages"]),
        subj_endpoint_to_passages=dict(payload["subj_endpoint_to_passages"]),
        obj_endpoint_to_passages=dict(payload["obj_endpoint_to_passages"]),
        subject_entity_to_passages=dict(payload["subject_entity_to_passages"]),
        object_entity_to_passages=dict(payload["object_entity_to_passages"]),
        slot_to_exact_passages=dict(payload.get("slot_to_exact_passages", {})),
        slot_to_endpoint_passages=dict(payload.get("slot_to_endpoint_passages", {})),
        slot_to_entity_passages=dict(payload.get("slot_to_entity_passages", {})),
        slot_to_support_passages=dict(payload.get("slot_to_support_passages", {})),
        slot_to_source_support_passages=dict(payload.get("slot_to_source_support_passages", {})),
        slot_to_anchor_source_support_passages=dict(payload.get("slot_to_anchor_source_support_passages", {})),
        slot_to_anchor_support_passages=dict(payload.get("slot_to_anchor_support_passages", {})),
        slot_to_target_support_passages=dict(payload.get("slot_to_target_support_passages", {})),
        slot_to_answer_support_passages=dict(payload.get("slot_to_answer_support_passages", {})),
        passage_to_exact_slot_ids=dict(payload.get("passage_to_exact_slot_ids", {})),
        runtime=runtime,
    )
