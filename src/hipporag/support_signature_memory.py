import json
import os
from collections import defaultdict
from numbers import Integral
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .utils.misc_utils import compute_mdhash_id, text_processing

SUPPORT_SIGNATURE_SIDECAR_VERSION = 4
SLOT_TIER_SUPPORT_FIELDS = (
    "slot_scope",
    "anchor_entity_id",
    "anchor_entity_text",
    "target_entity_id",
    "target_entity_text",
    "attribute_signature",
    "owning_source_ids",
    "exact_support_passage_ids",
    "endpoint_support_passage_ids",
    "entity_support_passage_ids",
    "source_support_passage_ids",
    "anchor_source_support_passage_ids",
    "anchor_support_passage_ids",
    "target_support_passage_ids",
    "answer_support_passage_ids",
    "supporting_passage_ids",
)


def _normalize_support_text(text: str) -> str:
    processed = text_processing(text)
    if not isinstance(processed, str):
        return ""
    return " ".join(processed.split())


def _normalize_relation_text(
    text: str,
    relation_normalizer: Optional[Callable[[str], str]] = None,
) -> str:
    if relation_normalizer is None:
        return _normalize_support_text(text)
    normalized = relation_normalizer(text)
    return " ".join(str(normalized).split())


def _normalize_ref(value: Any) -> Any:
    if isinstance(value, Integral):
        return int(value)
    return str(value)


def _sort_ref_list(values) -> List[Any]:
    def sort_key(value: Any) -> Tuple[int, str]:
        if isinstance(value, Integral):
            return (0, str(int(value)).zfill(12))
        return (1, str(value))

    normalized_values = {_normalize_ref(value) for value in values}
    return [_normalize_ref(value) for value in sorted(normalized_values, key=sort_key)]


def _new_passage_inventory() -> Dict[str, set]:
    return {
        "proposition_ids": set(),
        "pair_signatures": set(),
        "subj_endpoint_signatures": set(),
        "obj_endpoint_signatures": set(),
        "slot_signatures": set(),
        "subject_slot_signatures": set(),
        "object_slot_signatures": set(),
        "participant_slot_signatures": set(),
        "relation_signatures": set(),
        "subject_entities": set(),
        "object_entities": set(),
        "participant_entities": set(),
    }


def _ensure_passage_inventory(
    passage_inventories: Dict[str, Dict[str, set]],
    passage_id: Any,
) -> Dict[str, set]:
    passage_key = str(passage_id)
    inventory = passage_inventories.get(passage_key)
    if inventory is None:
        inventory = _new_passage_inventory()
        passage_inventories[passage_key] = inventory
    return inventory


def _encode_signature(signature: Sequence[str]) -> str:
    return json.dumps(list(signature), ensure_ascii=True, separators=(",", ":"))


def _decode_signature(payload: str) -> Tuple[str, ...]:
    decoded = json.loads(payload)
    if not isinstance(decoded, list):
        raise ValueError(f"Unsupported signature payload: {payload}")
    return tuple(str(value) for value in decoded)


def _make_slot_signature(slot_kind: str, relation: str, entity_text: str) -> Optional[Tuple[str, str, str]]:
    normalized_slot_kind = " ".join(str(slot_kind).split()).strip().lower()
    normalized_relation = " ".join(str(relation).split()).strip()
    normalized_entity_text = " ".join(str(entity_text).split()).strip()
    if not normalized_slot_kind or not normalized_entity_text:
        return None
    return (normalized_slot_kind, normalized_relation, normalized_entity_text)


def _collect_slot_signatures_from_triple(subject: str, relation: str, obj: str) -> List[Tuple[str, str, str]]:
    signatures: List[Tuple[str, str, str]] = []
    for slot_kind, entity_text in [
        ("subject", subject),
        ("object", obj),
        ("participant", subject),
        ("participant", obj),
    ]:
        signature = _make_slot_signature(slot_kind, relation, entity_text)
        if signature is not None and signature not in signatures:
            signatures.append(signature)
    return signatures


def _collect_slot_signatures_from_record(record: Mapping[str, Any]) -> List[Tuple[str, str, str]]:
    relation = " ".join(str(record.get("relation_type", "")).split())
    participant_ids = [str(value) for value in list(record.get("participant_ids", []))]
    participant_texts = [
        " ".join(str(text).split())
        for text in list(record.get("participant_texts", []))
    ]
    participant_roles = {
        str(participant_id): [str(role).strip().lower() for role in roles]
        for participant_id, roles in dict(record.get("participant_roles", {})).items()
    }
    signatures: List[Tuple[str, str, str]] = []
    for idx, participant_id in enumerate(participant_ids):
        entity_text = participant_texts[idx] if idx < len(participant_texts) else ""
        if not entity_text:
            continue
        roles = participant_roles.get(participant_id, [])
        for role_name in roles:
            signature = _make_slot_signature(role_name, relation, entity_text)
            if signature is not None and signature not in signatures:
                signatures.append(signature)
        participant_signature = _make_slot_signature("participant", relation, entity_text)
        if participant_signature is not None and participant_signature not in signatures:
            signatures.append(participant_signature)
    return signatures


def _build_slot_record_id(
    *,
    bridge_id: str,
    fact_object_id: str,
    slot_kind: str,
    entity_id: str,
    entity_text: str,
) -> str:
    signature_text = json.dumps(
        [str(bridge_id), str(fact_object_id), str(slot_kind), str(entity_id), str(entity_text)],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return compute_mdhash_id(signature_text, prefix="slot-")


def _normalize_passage_id_list(values: Sequence[Any]) -> List[str]:
    return [str(value) for value in _sort_ref_list(values)]


def _derive_source_preserve_support_passage_ids(
    *,
    source_passage_ids: Sequence[Any],
    anchor_support_passage_ids: Sequence[Any],
) -> Tuple[List[str], List[str]]:
    normalized_source_passage_ids = _normalize_passage_id_list(source_passage_ids)
    normalized_anchor_support_passage_ids = set(_normalize_passage_id_list(anchor_support_passage_ids))
    anchor_source_support_passage_ids = [
        passage_id
        for passage_id in normalized_source_passage_ids
        if passage_id in normalized_anchor_support_passage_ids
    ]
    if not anchor_source_support_passage_ids:
        anchor_source_support_passage_ids = list(normalized_source_passage_ids)
    return normalized_source_passage_ids, anchor_source_support_passage_ids


def _extract_record_role_entry(record: Mapping[str, Any], role_name: str) -> Tuple[str, str]:
    participant_ids = [str(value) for value in list(record.get("participant_ids", []))]
    participant_texts = [
        " ".join(str(text).split())
        for text in list(record.get("participant_texts", []))
    ]
    participant_roles = {
        str(participant_id): [str(role).strip().lower() for role in roles]
        for participant_id, roles in dict(record.get("participant_roles", {})).items()
    }
    participant_id_to_text = {
        str(participant_id): participant_texts[idx] if idx < len(participant_texts) else ""
        for idx, participant_id in enumerate(participant_ids)
    }
    for participant_id in participant_ids:
        normalized_id = str(participant_id)
        if role_name in participant_roles.get(normalized_id, []):
            participant_text = participant_id_to_text.get(normalized_id, "")
            if participant_text:
                return normalized_id, participant_text
    if len(participant_texts) >= 2:
        if role_name == "subject":
            return participant_ids[0] if participant_ids else "", participant_texts[0]
        if role_name == "object":
            fallback_id = participant_ids[1] if len(participant_ids) >= 2 else ""
            return fallback_id, participant_texts[1]
    return "", ""


def _lookup_entity_support_passage_ids_from_maps(
    *,
    entity_text: str,
    preferred_role: str,
    subject_entity_to_passages: Mapping[str, Sequence[Any]],
    object_entity_to_passages: Mapping[str, Sequence[Any]],
    participant_entity_to_passages: Mapping[str, Sequence[Any]],
) -> List[str]:
    normalized_entity_text = " ".join(str(entity_text).split()).strip()
    normalized_role = " ".join(str(preferred_role).split()).strip().lower()
    if not normalized_entity_text:
        return []

    subject_passages = list(subject_entity_to_passages.get(normalized_entity_text, []))
    object_passages = list(object_entity_to_passages.get(normalized_entity_text, []))
    participant_passages = list(participant_entity_to_passages.get(normalized_entity_text, []))

    ordered_passages: List[Any] = []
    if normalized_role == "subject":
        ordered_passages.extend(subject_passages)
        ordered_passages.extend(participant_passages)
        ordered_passages.extend(object_passages)
    elif normalized_role == "object":
        ordered_passages.extend(object_passages)
        ordered_passages.extend(participant_passages)
        ordered_passages.extend(subject_passages)
    else:
        ordered_passages.extend(participant_passages)
        ordered_passages.extend(subject_passages)
        ordered_passages.extend(object_passages)
    return _normalize_passage_id_list(ordered_passages)


def _build_attribute_signature(
    *,
    slot_scope: str,
    relation_type: str,
    anchor_entity_text: str,
    target_entity_text: str,
) -> str:
    return json.dumps(
        [
            str(slot_scope),
            " ".join(str(relation_type).split()).strip(),
            " ".join(str(anchor_entity_text).split()).strip(),
            " ".join(str(target_entity_text).split()).strip(),
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _derive_slot_relation_context(
    *,
    slot_kind: str,
    entity_id: str,
    entity_text: str,
    subject_entity_id: str,
    subject_entity_text: str,
    object_entity_id: str,
    object_entity_text: str,
    participant_ids: Sequence[str],
    participant_texts: Sequence[str],
) -> Dict[str, str]:
    normalized_slot_kind = " ".join(str(slot_kind).split()).strip().lower()
    normalized_entity_id = str(entity_id)
    normalized_entity_text = " ".join(str(entity_text).split()).strip()
    normalized_subject_id = str(subject_entity_id)
    normalized_subject_text = " ".join(str(subject_entity_text).split()).strip()
    normalized_object_id = str(object_entity_id)
    normalized_object_text = " ".join(str(object_entity_text).split()).strip()

    slot_scope = "participant"
    if normalized_slot_kind == "subject":
        slot_scope = "anchor"
    elif normalized_slot_kind == "object":
        slot_scope = "target"
    elif normalized_subject_text and normalized_entity_text == normalized_subject_text:
        slot_scope = "anchor"
    elif normalized_object_text and normalized_entity_text == normalized_object_text:
        slot_scope = "target"

    anchor_entity_id = normalized_subject_id or normalized_entity_id
    anchor_entity_text = normalized_subject_text or normalized_entity_text
    target_entity_id = normalized_object_id
    target_entity_text = normalized_object_text
    target_role = "object" if normalized_object_text else ""

    if slot_scope == "target" and normalized_object_text:
        target_entity_id = normalized_object_id
        target_entity_text = normalized_object_text
        target_role = "object"
    elif slot_scope == "anchor" and normalized_object_text:
        target_entity_id = normalized_object_id
        target_entity_text = normalized_object_text
        target_role = "object"
    elif normalized_subject_text and normalized_object_text:
        if normalized_entity_text == normalized_subject_text:
            target_entity_id = normalized_object_id
            target_entity_text = normalized_object_text
            target_role = "object"
        elif normalized_entity_text == normalized_object_text:
            target_entity_id = normalized_object_id
            target_entity_text = normalized_object_text
            target_role = "object"

    if not target_entity_text:
        for candidate_id, candidate_text in zip(participant_ids, participant_texts):
            normalized_candidate_id = str(candidate_id)
            normalized_candidate_text = " ".join(str(candidate_text).split()).strip()
            if not normalized_candidate_text or normalized_candidate_id == normalized_entity_id:
                continue
            target_entity_id = normalized_candidate_id
            target_entity_text = normalized_candidate_text
            target_role = "participant"
            break

    return {
        "slot_scope": slot_scope,
        "anchor_entity_id": anchor_entity_id,
        "anchor_entity_text": anchor_entity_text,
        "target_entity_id": target_entity_id,
        "target_entity_text": target_entity_text,
        "target_role": target_role,
    }


def _build_participant_entity_to_passages_from_inventories(
    passage_inventories: Mapping[str, Mapping[str, Sequence[str]]],
) -> Dict[str, List[str]]:
    participant_entity_to_passages: Dict[str, set] = defaultdict(set)
    for passage_id, inventory in passage_inventories.items():
        normalized_passage_id = str(passage_id)
        for entity_text in inventory.get("participant_entities", []):
            normalized_entity = " ".join(str(entity_text).split()).strip()
            if normalized_entity:
                participant_entity_to_passages[normalized_entity].add(normalized_passage_id)
    return {
        str(entity_text): _normalize_passage_id_list(passage_ids)
        for entity_text, passage_ids in participant_entity_to_passages.items()
    }


def _prepare_slot_support_lookup_maps(
    support_signature_sidecar: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[Any, List[Any]]]:
    sidecar = dict(support_signature_sidecar or {})
    participant_entity_to_passages = dict(sidecar.get("participant_entity_to_passages", {}))
    if not participant_entity_to_passages and sidecar.get("passage_inventories"):
        participant_entity_to_passages = _build_participant_entity_to_passages_from_inventories(
            dict(sidecar.get("passage_inventories", {}))
        )

    return {
        "subject_slot_to_passages": dict(sidecar.get("subject_slot_to_passages", {})),
        "object_slot_to_passages": dict(sidecar.get("object_slot_to_passages", {})),
        "participant_slot_to_passages": dict(sidecar.get("participant_slot_to_passages", {})),
        "slot_to_passages": dict(sidecar.get("slot_to_passages", {})),
        "subj_endpoint_to_passages": dict(sidecar.get("subj_endpoint_to_passages", {})),
        "obj_endpoint_to_passages": dict(sidecar.get("obj_endpoint_to_passages", {})),
        "subject_entity_to_passages": dict(sidecar.get("subject_entity_to_passages", {})),
        "object_entity_to_passages": dict(sidecar.get("object_entity_to_passages", {})),
        "participant_entity_to_passages": participant_entity_to_passages,
        "slot_to_source_support_passages": dict(sidecar.get("slot_to_source_support_passages", {})),
        "slot_to_anchor_source_support_passages": dict(sidecar.get("slot_to_anchor_source_support_passages", {})),
        "slot_to_anchor_support_passages": dict(sidecar.get("slot_to_anchor_support_passages", {})),
        "slot_to_target_support_passages": dict(sidecar.get("slot_to_target_support_passages", {})),
        "slot_to_answer_support_passages": dict(sidecar.get("slot_to_answer_support_passages", {})),
    }


def derive_slot_support_passage_ids(
    *,
    slot_kind: str,
    relation_type: str,
    entity_text: str,
    support_signature_sidecar: Optional[Mapping[str, Any]] = None,
    prepared_lookup_maps: Optional[Mapping[str, Mapping[Any, Sequence[Any]]]] = None,
) -> Dict[str, List[str]]:
    normalized_slot_kind = " ".join(str(slot_kind).split()).strip().lower()
    normalized_relation = " ".join(str(relation_type).split()).strip()
    normalized_entity_text = " ".join(str(entity_text).split()).strip()
    slot_signature = (normalized_slot_kind, normalized_relation, normalized_entity_text)

    lookup_maps = (
        prepared_lookup_maps
        if prepared_lookup_maps is not None
        else _prepare_slot_support_lookup_maps(support_signature_sidecar)
    )
    subject_slot_to_passages = lookup_maps.get("subject_slot_to_passages", {})
    object_slot_to_passages = lookup_maps.get("object_slot_to_passages", {})
    participant_slot_to_passages = lookup_maps.get("participant_slot_to_passages", {})
    generic_slot_to_passages = lookup_maps.get("slot_to_passages", {})
    subj_endpoint_to_passages = lookup_maps.get("subj_endpoint_to_passages", {})
    obj_endpoint_to_passages = lookup_maps.get("obj_endpoint_to_passages", {})
    subject_entity_to_passages = lookup_maps.get("subject_entity_to_passages", {})
    object_entity_to_passages = lookup_maps.get("object_entity_to_passages", {})
    participant_entity_to_passages = lookup_maps.get("participant_entity_to_passages", {})
    slot_to_source_support_passages = lookup_maps.get("slot_to_source_support_passages", {})
    slot_to_anchor_source_support_passages = lookup_maps.get("slot_to_anchor_source_support_passages", {})
    slot_to_anchor_support_passages = lookup_maps.get("slot_to_anchor_support_passages", {})
    slot_to_target_support_passages = lookup_maps.get("slot_to_target_support_passages", {})
    slot_to_answer_support_passages = lookup_maps.get("slot_to_answer_support_passages", {})
    exact_slot_maps = {
        "subject": subject_slot_to_passages,
        "object": object_slot_to_passages,
        "participant": participant_slot_to_passages,
    }
    exact_support_passage_ids = exact_slot_maps.get(normalized_slot_kind, generic_slot_to_passages).get(
        slot_signature,
        generic_slot_to_passages.get(slot_signature, []),
    )

    endpoint_support_passage_ids: Sequence[Any] = []
    if normalized_relation:
        if normalized_slot_kind == "subject":
            endpoint_support_passage_ids = subj_endpoint_to_passages.get(
                (normalized_entity_text, normalized_relation),
                [],
            )
        elif normalized_slot_kind == "object":
            endpoint_support_passage_ids = obj_endpoint_to_passages.get(
                (normalized_entity_text, normalized_relation),
                [],
            )

    entity_support_passage_ids: Sequence[Any] = []
    if normalized_slot_kind == "subject":
        entity_support_passage_ids = subject_entity_to_passages.get(
            normalized_entity_text,
            [],
        )
    elif normalized_slot_kind == "object":
        entity_support_passage_ids = object_entity_to_passages.get(
            normalized_entity_text,
            [],
        )
    elif normalized_slot_kind == "participant":
        entity_support_passage_ids = participant_entity_to_passages.get(normalized_entity_text, [])
        if not entity_support_passage_ids:
            entity_support_passage_ids = list(
                dict.fromkeys(
                    list(subject_entity_to_passages.get(normalized_entity_text, []))
                    + list(object_entity_to_passages.get(normalized_entity_text, []))
                )
            )

    supporting_passage_ids = _normalize_passage_id_list(
        list(exact_support_passage_ids) + list(endpoint_support_passage_ids) + list(entity_support_passage_ids)
    )
    anchor_support_passage_ids = _normalize_passage_id_list(
        slot_to_anchor_support_passages.get(
            slot_signature,
            entity_support_passage_ids,
        )
    )
    source_support_passage_ids = _normalize_passage_id_list(
        slot_to_source_support_passages.get(
            slot_signature,
            exact_support_passage_ids,
        )
    )
    _, default_anchor_source_support_passage_ids = _derive_source_preserve_support_passage_ids(
        source_passage_ids=source_support_passage_ids,
        anchor_support_passage_ids=anchor_support_passage_ids,
    )
    anchor_source_support_passage_ids = _normalize_passage_id_list(
        slot_to_anchor_source_support_passages.get(
            slot_signature,
            default_anchor_source_support_passage_ids,
        )
    )
    target_support_passage_ids = _normalize_passage_id_list(
        slot_to_target_support_passages.get(slot_signature, [])
    )
    answer_support_passage_ids = _normalize_passage_id_list(
        slot_to_answer_support_passages.get(
            slot_signature,
            target_support_passage_ids,
        )
    )
    return {
        "exact_support_passage_ids": _normalize_passage_id_list(exact_support_passage_ids),
        "endpoint_support_passage_ids": _normalize_passage_id_list(endpoint_support_passage_ids),
        "entity_support_passage_ids": _normalize_passage_id_list(entity_support_passage_ids),
        "source_support_passage_ids": source_support_passage_ids,
        "anchor_source_support_passage_ids": anchor_source_support_passage_ids,
        "anchor_support_passage_ids": anchor_support_passage_ids,
        "target_support_passage_ids": target_support_passage_ids,
        "answer_support_passage_ids": answer_support_passage_ids,
        "supporting_passage_ids": supporting_passage_ids,
    }


def build_slot_records_from_hyperedge_records(
    hyperedge_records: Mapping[str, Dict[str, Any]],
    support_signature_sidecar: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    slot_records: Dict[str, Dict[str, Any]] = {}
    prepared_lookup_maps = _prepare_slot_support_lookup_maps(support_signature_sidecar)
    cached_support_passages: Dict[Tuple[str, str, str], Dict[str, List[str]]] = {}
    for bridge_id, record in hyperedge_records.items():
        normalized_bridge_id = str(bridge_id)
        fact_object_id = str(record.get("fact_embedding_hash_id") or normalized_bridge_id)
        relation = " ".join(str(record.get("relation_type", "")).split())
        participant_ids = [str(value) for value in list(record.get("participant_ids", []))]
        participant_texts = [
            " ".join(str(text).split())
            for text in list(record.get("participant_texts", []))
        ]
        participant_roles = {
            str(participant_id): [str(role).strip().lower() for role in roles]
            for participant_id, roles in dict(record.get("participant_roles", {})).items()
        }
        source_ids = sorted({str(value) for value in list(record.get("source_ids", [])) if str(value)})
        last_seen = int(record.get("last_seen", 0))
        support_count = int(record.get("support_count", max(1, len(source_ids) or 1)))
        conflict_group = str(record.get("conflict_group", "")).strip()
        subject_entity_id, subject_entity_text = _extract_record_role_entry(record, "subject")
        object_entity_id, object_entity_text = _extract_record_role_entry(record, "object")

        for idx, entity_id in enumerate(participant_ids):
            entity_text = participant_texts[idx] if idx < len(participant_texts) else ""
            if not entity_text:
                continue
            role_names = list(participant_roles.get(entity_id, [])) + ["participant"]
            seen_role_names = set()
            for role_name in role_names:
                normalized_role = " ".join(str(role_name).split()).strip().lower()
                if not normalized_role or normalized_role in seen_role_names:
                    continue
                seen_role_names.add(normalized_role)
                slot_id = _build_slot_record_id(
                    bridge_id=normalized_bridge_id,
                    fact_object_id=fact_object_id,
                    slot_kind=normalized_role,
                    entity_id=entity_id,
                    entity_text=entity_text,
                )
                slot_signature = (normalized_role, relation, entity_text)
                support_passages = cached_support_passages.get(slot_signature)
                if support_passages is None:
                    support_passages = derive_slot_support_passage_ids(
                        slot_kind=normalized_role,
                        relation_type=relation,
                        entity_text=entity_text,
                        prepared_lookup_maps=prepared_lookup_maps,
                    )
                    cached_support_passages[slot_signature] = support_passages
                relation_context = _derive_slot_relation_context(
                    slot_kind=normalized_role,
                    entity_id=entity_id,
                    entity_text=entity_text,
                    subject_entity_id=subject_entity_id,
                    subject_entity_text=subject_entity_text,
                    object_entity_id=object_entity_id,
                    object_entity_text=object_entity_text,
                    participant_ids=participant_ids,
                    participant_texts=participant_texts,
                )
                slot_records[slot_id] = {
                    "hash_id": slot_id,
                    "slot_id": slot_id,
                    "bridge_id": normalized_bridge_id,
                    "fact_object_id": fact_object_id,
                    "slot_kind": normalized_role,
                    "relation_type": relation,
                    "entity_id": entity_id,
                    "entity_text": entity_text,
                    "slot_scope": relation_context["slot_scope"],
                    "anchor_entity_id": relation_context["anchor_entity_id"],
                    "anchor_entity_text": relation_context["anchor_entity_text"],
                    "target_entity_id": relation_context["target_entity_id"],
                    "target_entity_text": relation_context["target_entity_text"],
                    "attribute_signature": _build_attribute_signature(
                        slot_scope=relation_context["slot_scope"],
                        relation_type=relation,
                        anchor_entity_text=relation_context["anchor_entity_text"],
                        target_entity_text=relation_context["target_entity_text"],
                    ),
                    "source_ids": list(source_ids),
                    "owning_source_ids": list(source_ids),
                    "exact_support_passage_ids": list(support_passages["exact_support_passage_ids"]),
                    "endpoint_support_passage_ids": list(support_passages["endpoint_support_passage_ids"]),
                    "entity_support_passage_ids": list(support_passages["entity_support_passage_ids"]),
                    "source_support_passage_ids": list(support_passages["source_support_passage_ids"] or source_ids),
                    "anchor_source_support_passage_ids": list(
                        support_passages["anchor_source_support_passage_ids"] or source_ids
                    ),
                    "anchor_support_passage_ids": list(support_passages["anchor_support_passage_ids"]),
                    "target_support_passage_ids": list(support_passages["target_support_passage_ids"]),
                    "answer_support_passage_ids": list(support_passages["answer_support_passage_ids"]),
                    "supporting_passage_ids": list(support_passages["supporting_passage_ids"]),
                    "supporting_passage_count": len(support_passages["supporting_passage_ids"]),
                    "answer_support_passage_count": len(support_passages["answer_support_passage_ids"]),
                    "partial_support_profile": {
                        "has_source_support": bool(support_passages["source_support_passage_ids"] or source_ids),
                        "has_anchor_source_support": bool(
                            support_passages["anchor_source_support_passage_ids"] or source_ids
                        ),
                        "has_anchor_support": bool(support_passages["anchor_support_passage_ids"]),
                        "has_target_support": bool(support_passages["target_support_passage_ids"]),
                        "has_answer_support": bool(support_passages["answer_support_passage_ids"]),
                    },
                    "support_count": support_count,
                    "last_seen": last_seen,
                    "conflict_group": conflict_group,
                }
    return [slot_records[slot_id] for slot_id in sorted(slot_records.keys())]


def build_slot_record_ids_from_hyperedge_records(
    hyperedge_records: Mapping[str, Dict[str, Any]],
) -> List[str]:
    slot_ids = set()
    for bridge_id, record in hyperedge_records.items():
        normalized_bridge_id = str(bridge_id)
        fact_object_id = str(record.get("fact_embedding_hash_id") or normalized_bridge_id)
        participant_ids = [str(value) for value in list(record.get("participant_ids", []))]
        participant_texts = [
            " ".join(str(text).split())
            for text in list(record.get("participant_texts", []))
        ]
        participant_roles = {
            str(participant_id): [str(role).strip().lower() for role in roles]
            for participant_id, roles in dict(record.get("participant_roles", {})).items()
        }

        for idx, entity_id in enumerate(participant_ids):
            entity_text = participant_texts[idx] if idx < len(participant_texts) else ""
            if not entity_text:
                continue
            role_names = list(participant_roles.get(entity_id, [])) + ["participant"]
            seen_role_names = set()
            for role_name in role_names:
                normalized_role = " ".join(str(role_name).split()).strip().lower()
                if not normalized_role or normalized_role in seen_role_names:
                    continue
                seen_role_names.add(normalized_role)
                slot_ids.add(
                    _build_slot_record_id(
                        bridge_id=normalized_bridge_id,
                        fact_object_id=fact_object_id,
                        slot_kind=normalized_role,
                        entity_id=entity_id,
                        entity_text=entity_text,
                    )
                )
    return sorted(slot_ids)


def _extract_record_role_text(record: Mapping[str, Any], role_name: str) -> str:
    _, participant_text = _extract_record_role_entry(record, role_name)
    return participant_text


def _finalize_sidecar(
    pair_to_passages: Mapping[Tuple[str, str, str], set],
    subj_endpoint_to_passages: Mapping[Tuple[str, str], set],
    obj_endpoint_to_passages: Mapping[Tuple[str, str], set],
    slot_to_passages: Mapping[Tuple[str, str, str], set],
    subject_slot_to_passages: Mapping[Tuple[str, str, str], set],
    object_slot_to_passages: Mapping[Tuple[str, str, str], set],
    participant_slot_to_passages: Mapping[Tuple[str, str, str], set],
    relation_to_passages: Mapping[str, set],
    subject_entity_to_passages: Mapping[str, set],
    object_entity_to_passages: Mapping[str, set],
    participant_entity_to_passages: Mapping[str, set],
    slot_to_source_support_passages: Mapping[Tuple[str, str, str], set],
    slot_to_anchor_source_support_passages: Mapping[Tuple[str, str, str], set],
    slot_to_anchor_support_passages: Mapping[Tuple[str, str, str], set],
    slot_to_target_support_passages: Mapping[Tuple[str, str, str], set],
    slot_to_answer_support_passages: Mapping[Tuple[str, str, str], set],
    passage_inventories: Mapping[str, Dict[str, set]],
) -> Dict[str, Any]:
    finalized_inventories: Dict[str, Dict[str, Any]] = {}
    for passage_id, inventory in passage_inventories.items():
        finalized_inventories[str(passage_id)] = {
            "proposition_ids": sorted(str(value) for value in inventory.get("proposition_ids", set())),
            "pair_signatures": sorted(inventory.get("pair_signatures", set())),
            "subj_endpoint_signatures": sorted(inventory.get("subj_endpoint_signatures", set())),
            "obj_endpoint_signatures": sorted(inventory.get("obj_endpoint_signatures", set())),
            "slot_signatures": sorted(inventory.get("slot_signatures", set())),
            "subject_slot_signatures": sorted(inventory.get("subject_slot_signatures", set())),
            "object_slot_signatures": sorted(inventory.get("object_slot_signatures", set())),
            "participant_slot_signatures": sorted(inventory.get("participant_slot_signatures", set())),
            "relation_signatures": sorted(str(value) for value in inventory.get("relation_signatures", set())),
            "subject_entities": sorted(str(value) for value in inventory.get("subject_entities", set())),
            "object_entities": sorted(str(value) for value in inventory.get("object_entities", set())),
            "participant_entities": sorted(str(value) for value in inventory.get("participant_entities", set())),
        }

    return {
        "version": SUPPORT_SIGNATURE_SIDECAR_VERSION,
        "pair_to_passages": {
            tuple(signature): _sort_ref_list(passage_refs)
            for signature, passage_refs in pair_to_passages.items()
        },
        "subj_endpoint_to_passages": {
            tuple(signature): _sort_ref_list(passage_refs)
            for signature, passage_refs in subj_endpoint_to_passages.items()
        },
        "obj_endpoint_to_passages": {
            tuple(signature): _sort_ref_list(passage_refs)
            for signature, passage_refs in obj_endpoint_to_passages.items()
        },
        "slot_to_passages": {
            tuple(signature): _sort_ref_list(passage_refs)
            for signature, passage_refs in slot_to_passages.items()
        },
        "subject_slot_to_passages": {
            tuple(signature): _sort_ref_list(passage_refs)
            for signature, passage_refs in subject_slot_to_passages.items()
        },
        "object_slot_to_passages": {
            tuple(signature): _sort_ref_list(passage_refs)
            for signature, passage_refs in object_slot_to_passages.items()
        },
        "participant_slot_to_passages": {
            tuple(signature): _sort_ref_list(passage_refs)
            for signature, passage_refs in participant_slot_to_passages.items()
        },
        "relation_to_passages": {
            str(signature): _sort_ref_list(passage_refs)
            for signature, passage_refs in relation_to_passages.items()
        },
        "subject_entity_to_passages": {
            str(entity): _sort_ref_list(passage_refs)
            for entity, passage_refs in subject_entity_to_passages.items()
        },
        "object_entity_to_passages": {
            str(entity): _sort_ref_list(passage_refs)
            for entity, passage_refs in object_entity_to_passages.items()
        },
        "participant_entity_to_passages": {
            str(entity): _sort_ref_list(passage_refs)
            for entity, passage_refs in participant_entity_to_passages.items()
        },
        "slot_to_source_support_passages": {
            tuple(signature): _sort_ref_list(passage_refs)
            for signature, passage_refs in slot_to_source_support_passages.items()
        },
        "slot_to_anchor_source_support_passages": {
            tuple(signature): _sort_ref_list(passage_refs)
            for signature, passage_refs in slot_to_anchor_source_support_passages.items()
        },
        "slot_to_anchor_support_passages": {
            tuple(signature): _sort_ref_list(passage_refs)
            for signature, passage_refs in slot_to_anchor_support_passages.items()
        },
        "slot_to_target_support_passages": {
            tuple(signature): _sort_ref_list(passage_refs)
            for signature, passage_refs in slot_to_target_support_passages.items()
        },
        "slot_to_answer_support_passages": {
            tuple(signature): _sort_ref_list(passage_refs)
            for signature, passage_refs in slot_to_answer_support_passages.items()
        },
        "passage_inventories": finalized_inventories,
    }


def build_support_signature_sidecar_from_triples(
    passage_chunk_ids: Sequence[str],
    chunk_triples_map: Mapping[str, List[Tuple[str, str, str]]],
    relation_normalizer: Optional[Callable[[str], str]] = None,
    passage_refs: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    pair_to_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    subj_endpoint_to_passages: Dict[Tuple[str, str], set] = defaultdict(set)
    obj_endpoint_to_passages: Dict[Tuple[str, str], set] = defaultdict(set)
    slot_to_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    subject_slot_to_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    object_slot_to_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    participant_slot_to_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    relation_to_passages: Dict[str, set] = defaultdict(set)
    subject_entity_to_passages: Dict[str, set] = defaultdict(set)
    object_entity_to_passages: Dict[str, set] = defaultdict(set)
    participant_entity_to_passages: Dict[str, set] = defaultdict(set)
    slot_to_source_support_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    slot_to_anchor_source_support_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    slot_to_anchor_support_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    slot_to_target_support_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    slot_to_answer_support_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    passage_inventories: Dict[str, Dict[str, set]] = {}

    for passage_idx, chunk_id in enumerate(passage_chunk_ids):
        passage_ref = passage_refs[passage_idx] if passage_refs is not None else chunk_id
        normalized_ref = _normalize_ref(passage_ref)
        inventory = _ensure_passage_inventory(passage_inventories, chunk_id)
        for triple in chunk_triples_map.get(chunk_id, []):
            if len(triple) != 3:
                continue
            subject = _normalize_support_text(triple[0])
            relation = _normalize_relation_text(triple[1], relation_normalizer=relation_normalizer)
            obj = _normalize_support_text(triple[2])
            if not subject or not relation or not obj:
                continue

            pair_signature = (subject, relation, obj)
            subj_endpoint = (subject, relation)
            obj_endpoint = (obj, relation)
            slot_signatures = _collect_slot_signatures_from_triple(subject, relation, obj)

            pair_to_passages[pair_signature].add(normalized_ref)
            subj_endpoint_to_passages[subj_endpoint].add(normalized_ref)
            obj_endpoint_to_passages[obj_endpoint].add(normalized_ref)
            relation_to_passages[relation].add(normalized_ref)
            subject_entity_to_passages[subject].add(normalized_ref)
            object_entity_to_passages[obj].add(normalized_ref)
            participant_entity_to_passages[subject].add(normalized_ref)
            participant_entity_to_passages[obj].add(normalized_ref)
            for slot_signature in slot_signatures:
                slot_to_passages[slot_signature].add(normalized_ref)
                slot_to_source_support_passages[slot_signature].add(normalized_ref)
                slot_to_anchor_source_support_passages[slot_signature].add(normalized_ref)
                inventory["slot_signatures"].add(slot_signature)
                if slot_signature[0] == "subject":
                    subject_slot_to_passages[slot_signature].add(normalized_ref)
                    inventory["subject_slot_signatures"].add(slot_signature)
                elif slot_signature[0] == "object":
                    object_slot_to_passages[slot_signature].add(normalized_ref)
                    inventory["object_slot_signatures"].add(slot_signature)
                elif slot_signature[0] == "participant":
                    participant_slot_to_passages[slot_signature].add(normalized_ref)
                    inventory["participant_slot_signatures"].add(slot_signature)

            inventory["pair_signatures"].add(pair_signature)
            inventory["subj_endpoint_signatures"].add(subj_endpoint)
            inventory["obj_endpoint_signatures"].add(obj_endpoint)
            inventory["relation_signatures"].add(relation)
            inventory["subject_entities"].add(subject)
            inventory["object_entities"].add(obj)
            inventory["participant_entities"].update([subject, obj])

    return _finalize_sidecar(
        pair_to_passages=pair_to_passages,
        subj_endpoint_to_passages=subj_endpoint_to_passages,
        obj_endpoint_to_passages=obj_endpoint_to_passages,
        slot_to_passages=slot_to_passages,
        subject_slot_to_passages=subject_slot_to_passages,
        object_slot_to_passages=object_slot_to_passages,
        participant_slot_to_passages=participant_slot_to_passages,
        relation_to_passages=relation_to_passages,
        subject_entity_to_passages=subject_entity_to_passages,
        object_entity_to_passages=object_entity_to_passages,
        participant_entity_to_passages=participant_entity_to_passages,
        slot_to_source_support_passages=slot_to_source_support_passages,
        slot_to_anchor_source_support_passages=slot_to_anchor_source_support_passages,
        slot_to_anchor_support_passages=slot_to_anchor_support_passages,
        slot_to_target_support_passages=slot_to_target_support_passages,
        slot_to_answer_support_passages=slot_to_answer_support_passages,
        passage_inventories=passage_inventories,
    )


def build_support_signature_sidecar_from_hyperedge_records(
    hyperedge_records: Mapping[str, Dict[str, Any]],
    passage_records: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    pair_to_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    subj_endpoint_to_passages: Dict[Tuple[str, str], set] = defaultdict(set)
    obj_endpoint_to_passages: Dict[Tuple[str, str], set] = defaultdict(set)
    slot_to_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    subject_slot_to_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    object_slot_to_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    participant_slot_to_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    relation_to_passages: Dict[str, set] = defaultdict(set)
    subject_entity_to_passages: Dict[str, set] = defaultdict(set)
    object_entity_to_passages: Dict[str, set] = defaultdict(set)
    participant_entity_to_passages: Dict[str, set] = defaultdict(set)
    slot_to_source_support_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    slot_to_anchor_source_support_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    slot_to_anchor_support_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    slot_to_target_support_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    slot_to_answer_support_passages: Dict[Tuple[str, str, str], set] = defaultdict(set)
    passage_inventories: Dict[str, Dict[str, set]] = {}

    for passage_id, record in passage_records.items():
        inventory = _ensure_passage_inventory(passage_inventories, passage_id)
        inventory["proposition_ids"].update(str(value) for value in record.get("proposition_ids", []))

    for hyperedge_id, record in hyperedge_records.items():
        participant_texts = [
            " ".join(str(text).split())
            for text in list(record.get("participant_texts", []))
            if " ".join(str(text).split())
        ]
        relation = " ".join(str(record.get("relation_type", "")).split())
        subject = _extract_record_role_text(record, "subject")
        obj = _extract_record_role_text(record, "object")
        use_structured_signature = bool(subject and obj and relation and relation != "proposition")
        source_ids = list(record.get("source_ids", []))
        slot_signatures = _collect_slot_signatures_from_record(record)

        for passage_id in source_ids:
            inventory = _ensure_passage_inventory(passage_inventories, passage_id)
            inventory["proposition_ids"].add(str(record.get("hash_id", hyperedge_id)))
            inventory["participant_entities"].update(participant_texts)
            for participant_text in participant_texts:
                participant_entity_to_passages[str(participant_text)].add(str(passage_id))
            for slot_signature in slot_signatures:
                slot_to_passages[slot_signature].add(str(passage_id))
                inventory["slot_signatures"].add(slot_signature)
                if slot_signature[0] == "subject":
                    subject_slot_to_passages[slot_signature].add(str(passage_id))
                    inventory["subject_slot_signatures"].add(slot_signature)
                elif slot_signature[0] == "object":
                    object_slot_to_passages[slot_signature].add(str(passage_id))
                    inventory["object_slot_signatures"].add(slot_signature)
                elif slot_signature[0] == "participant":
                    participant_slot_to_passages[slot_signature].add(str(passage_id))
                    inventory["participant_slot_signatures"].add(slot_signature)

            if use_structured_signature:
                pair_signature = (subject, relation, obj)
                subj_endpoint = (subject, relation)
                obj_endpoint = (obj, relation)

                pair_to_passages[pair_signature].add(str(passage_id))
                subj_endpoint_to_passages[subj_endpoint].add(str(passage_id))
                obj_endpoint_to_passages[obj_endpoint].add(str(passage_id))
                relation_to_passages[relation].add(str(passage_id))
                subject_entity_to_passages[subject].add(str(passage_id))
                object_entity_to_passages[obj].add(str(passage_id))

                inventory["pair_signatures"].add(pair_signature)
                inventory["subj_endpoint_signatures"].add(subj_endpoint)
                inventory["obj_endpoint_signatures"].add(obj_endpoint)
                inventory["relation_signatures"].add(relation)
                inventory["subject_entities"].add(subject)
                inventory["object_entities"].add(obj)

    for record in hyperedge_records.values():
        relation = " ".join(str(record.get("relation_type", "")).split())
        participant_ids = [str(value) for value in list(record.get("participant_ids", []))]
        participant_texts = [
            " ".join(str(text).split())
            for text in list(record.get("participant_texts", []))
        ]
        participant_roles = {
            str(participant_id): [str(role).strip().lower() for role in roles]
            for participant_id, roles in dict(record.get("participant_roles", {})).items()
        }
        subject_entity_id, subject_entity_text = _extract_record_role_entry(record, "subject")
        object_entity_id, object_entity_text = _extract_record_role_entry(record, "object")
        for idx, entity_id in enumerate(participant_ids):
            entity_text = participant_texts[idx] if idx < len(participant_texts) else ""
            if not entity_text:
                continue
            role_names = list(participant_roles.get(str(entity_id), [])) + ["participant"]
            seen_role_names = set()
            for role_name in role_names:
                normalized_role = " ".join(str(role_name).split()).strip().lower()
                if not normalized_role or normalized_role in seen_role_names:
                    continue
                seen_role_names.add(normalized_role)
                slot_signature = _make_slot_signature(normalized_role, relation, entity_text)
                if slot_signature is None:
                    continue
                relation_context = _derive_slot_relation_context(
                    slot_kind=normalized_role,
                    entity_id=str(entity_id),
                    entity_text=entity_text,
                    subject_entity_id=subject_entity_id,
                    subject_entity_text=subject_entity_text,
                    object_entity_id=object_entity_id,
                    object_entity_text=object_entity_text,
                    participant_ids=participant_ids,
                    participant_texts=participant_texts,
                )
                anchor_role = "subject" if relation_context["anchor_entity_text"] == subject_entity_text else "participant"
                target_role = relation_context["target_role"] or (
                    "object" if relation_context["target_entity_text"] == object_entity_text else "participant"
                )
                anchor_support_passages = _lookup_entity_support_passage_ids_from_maps(
                    entity_text=relation_context["anchor_entity_text"],
                    preferred_role=anchor_role,
                    subject_entity_to_passages=subject_entity_to_passages,
                    object_entity_to_passages=object_entity_to_passages,
                    participant_entity_to_passages=participant_entity_to_passages,
                )
                target_support_passages = _lookup_entity_support_passage_ids_from_maps(
                    entity_text=relation_context["target_entity_text"],
                    preferred_role=target_role,
                    subject_entity_to_passages=subject_entity_to_passages,
                    object_entity_to_passages=object_entity_to_passages,
                    participant_entity_to_passages=participant_entity_to_passages,
                )
                answer_support_passages = list(target_support_passages)
                source_support_passages = [str(value) for value in list(record.get("source_ids", [])) if str(value)]
                _, anchor_source_support_passages = _derive_source_preserve_support_passage_ids(
                    source_passage_ids=source_support_passages,
                    anchor_support_passage_ids=anchor_support_passages,
                )
                normalized_target_text = relation_context["target_entity_text"]
                if normalized_target_text and relation:
                    if target_role == "subject":
                        answer_support_passages.extend(
                            subj_endpoint_to_passages.get((normalized_target_text, relation), set())
                        )
                    elif target_role == "object":
                        answer_support_passages.extend(
                            obj_endpoint_to_passages.get((normalized_target_text, relation), set())
                        )
                slot_to_source_support_passages[slot_signature].update(source_support_passages)
                slot_to_anchor_source_support_passages[slot_signature].update(anchor_source_support_passages)
                slot_to_anchor_support_passages[slot_signature].update(anchor_support_passages)
                slot_to_target_support_passages[slot_signature].update(target_support_passages)
                slot_to_answer_support_passages[slot_signature].update(answer_support_passages)

    return _finalize_sidecar(
        pair_to_passages=pair_to_passages,
        subj_endpoint_to_passages=subj_endpoint_to_passages,
        obj_endpoint_to_passages=obj_endpoint_to_passages,
        slot_to_passages=slot_to_passages,
        subject_slot_to_passages=subject_slot_to_passages,
        object_slot_to_passages=object_slot_to_passages,
        participant_slot_to_passages=participant_slot_to_passages,
        relation_to_passages=relation_to_passages,
        subject_entity_to_passages=subject_entity_to_passages,
        object_entity_to_passages=object_entity_to_passages,
        participant_entity_to_passages=participant_entity_to_passages,
        slot_to_source_support_passages=slot_to_source_support_passages,
        slot_to_anchor_source_support_passages=slot_to_anchor_source_support_passages,
        slot_to_anchor_support_passages=slot_to_anchor_support_passages,
        slot_to_target_support_passages=slot_to_target_support_passages,
        slot_to_answer_support_passages=slot_to_answer_support_passages,
        passage_inventories=passage_inventories,
    )


def save_support_signature_sidecar(path: str, sidecar: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "version": int(sidecar.get("version", SUPPORT_SIGNATURE_SIDECAR_VERSION)),
        "pair_to_passages": {
            _encode_signature(signature): list(passage_refs)
            for signature, passage_refs in dict(sidecar.get("pair_to_passages", {})).items()
        },
        "subj_endpoint_to_passages": {
            _encode_signature(signature): list(passage_refs)
            for signature, passage_refs in dict(sidecar.get("subj_endpoint_to_passages", {})).items()
        },
        "obj_endpoint_to_passages": {
            _encode_signature(signature): list(passage_refs)
            for signature, passage_refs in dict(sidecar.get("obj_endpoint_to_passages", {})).items()
        },
        "slot_to_passages": {
            _encode_signature(signature): list(passage_refs)
            for signature, passage_refs in dict(sidecar.get("slot_to_passages", {})).items()
        },
        "subject_slot_to_passages": {
            _encode_signature(signature): list(passage_refs)
            for signature, passage_refs in dict(sidecar.get("subject_slot_to_passages", {})).items()
        },
        "object_slot_to_passages": {
            _encode_signature(signature): list(passage_refs)
            for signature, passage_refs in dict(sidecar.get("object_slot_to_passages", {})).items()
        },
        "participant_slot_to_passages": {
            _encode_signature(signature): list(passage_refs)
            for signature, passage_refs in dict(sidecar.get("participant_slot_to_passages", {})).items()
        },
        "relation_to_passages": {
            str(signature): list(passage_refs)
            for signature, passage_refs in dict(sidecar.get("relation_to_passages", {})).items()
        },
        "subject_entity_to_passages": {
            str(entity): list(passage_refs)
            for entity, passage_refs in dict(sidecar.get("subject_entity_to_passages", {})).items()
        },
        "object_entity_to_passages": {
            str(entity): list(passage_refs)
            for entity, passage_refs in dict(sidecar.get("object_entity_to_passages", {})).items()
        },
        "participant_entity_to_passages": {
            str(entity): list(passage_refs)
            for entity, passage_refs in dict(sidecar.get("participant_entity_to_passages", {})).items()
        },
        "slot_to_source_support_passages": {
            _encode_signature(signature): list(passage_refs)
            for signature, passage_refs in dict(sidecar.get("slot_to_source_support_passages", {})).items()
        },
        "slot_to_anchor_source_support_passages": {
            _encode_signature(signature): list(passage_refs)
            for signature, passage_refs in dict(sidecar.get("slot_to_anchor_source_support_passages", {})).items()
        },
        "slot_to_anchor_support_passages": {
            _encode_signature(signature): list(passage_refs)
            for signature, passage_refs in dict(sidecar.get("slot_to_anchor_support_passages", {})).items()
        },
        "slot_to_target_support_passages": {
            _encode_signature(signature): list(passage_refs)
            for signature, passage_refs in dict(sidecar.get("slot_to_target_support_passages", {})).items()
        },
        "slot_to_answer_support_passages": {
            _encode_signature(signature): list(passage_refs)
            for signature, passage_refs in dict(sidecar.get("slot_to_answer_support_passages", {})).items()
        },
        "passage_inventories": {},
    }
    for passage_id, inventory in dict(sidecar.get("passage_inventories", {})).items():
        payload["passage_inventories"][str(passage_id)] = {
            "proposition_ids": [str(value) for value in inventory.get("proposition_ids", [])],
            "pair_signatures": [_encode_signature(signature) for signature in inventory.get("pair_signatures", [])],
            "subj_endpoint_signatures": [
                _encode_signature(signature) for signature in inventory.get("subj_endpoint_signatures", [])
            ],
            "obj_endpoint_signatures": [
                _encode_signature(signature) for signature in inventory.get("obj_endpoint_signatures", [])
            ],
            "slot_signatures": [_encode_signature(signature) for signature in inventory.get("slot_signatures", [])],
            "subject_slot_signatures": [
                _encode_signature(signature) for signature in inventory.get("subject_slot_signatures", [])
            ],
            "object_slot_signatures": [
                _encode_signature(signature) for signature in inventory.get("object_slot_signatures", [])
            ],
            "participant_slot_signatures": [
                _encode_signature(signature) for signature in inventory.get("participant_slot_signatures", [])
            ],
            "relation_signatures": [str(value) for value in inventory.get("relation_signatures", [])],
            "subject_entities": [str(value) for value in inventory.get("subject_entities", [])],
            "object_entities": [str(value) for value in inventory.get("object_entities", [])],
            "participant_entities": [str(value) for value in inventory.get("participant_entities", [])],
        }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True)


def load_support_signature_sidecar(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {
            "version": SUPPORT_SIGNATURE_SIDECAR_VERSION,
            "pair_to_passages": {},
            "subj_endpoint_to_passages": {},
            "obj_endpoint_to_passages": {},
            "slot_to_passages": {},
            "subject_slot_to_passages": {},
            "object_slot_to_passages": {},
            "participant_slot_to_passages": {},
            "relation_to_passages": {},
            "subject_entity_to_passages": {},
            "object_entity_to_passages": {},
            "participant_entity_to_passages": {},
            "slot_to_source_support_passages": {},
            "slot_to_anchor_source_support_passages": {},
            "slot_to_anchor_support_passages": {},
            "slot_to_target_support_passages": {},
            "slot_to_answer_support_passages": {},
            "passage_inventories": {},
        }

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    sidecar = {
        "version": int(payload.get("version", 1)),
        "pair_to_passages": {
            _decode_signature(signature): [_normalize_ref(value) for value in passage_refs]
            for signature, passage_refs in dict(payload.get("pair_to_passages", {})).items()
        },
        "subj_endpoint_to_passages": {
            _decode_signature(signature): [_normalize_ref(value) for value in passage_refs]
            for signature, passage_refs in dict(payload.get("subj_endpoint_to_passages", {})).items()
        },
        "obj_endpoint_to_passages": {
            _decode_signature(signature): [_normalize_ref(value) for value in passage_refs]
            for signature, passage_refs in dict(payload.get("obj_endpoint_to_passages", {})).items()
        },
        "slot_to_passages": {
            _decode_signature(signature): [_normalize_ref(value) for value in passage_refs]
            for signature, passage_refs in dict(payload.get("slot_to_passages", {})).items()
        },
        "subject_slot_to_passages": {
            _decode_signature(signature): [_normalize_ref(value) for value in passage_refs]
            for signature, passage_refs in dict(payload.get("subject_slot_to_passages", {})).items()
        },
        "object_slot_to_passages": {
            _decode_signature(signature): [_normalize_ref(value) for value in passage_refs]
            for signature, passage_refs in dict(payload.get("object_slot_to_passages", {})).items()
        },
        "participant_slot_to_passages": {
            _decode_signature(signature): [_normalize_ref(value) for value in passage_refs]
            for signature, passage_refs in dict(payload.get("participant_slot_to_passages", {})).items()
        },
        "relation_to_passages": {
            str(signature): [_normalize_ref(value) for value in passage_refs]
            for signature, passage_refs in dict(payload.get("relation_to_passages", {})).items()
        },
        "subject_entity_to_passages": {
            str(entity): [_normalize_ref(value) for value in passage_refs]
            for entity, passage_refs in dict(payload.get("subject_entity_to_passages", {})).items()
        },
        "object_entity_to_passages": {
            str(entity): [_normalize_ref(value) for value in passage_refs]
            for entity, passage_refs in dict(payload.get("object_entity_to_passages", {})).items()
        },
        "participant_entity_to_passages": {
            str(entity): [_normalize_ref(value) for value in passage_refs]
            for entity, passage_refs in dict(payload.get("participant_entity_to_passages", {})).items()
        },
        "slot_to_source_support_passages": {
            _decode_signature(signature): [_normalize_ref(value) for value in passage_refs]
            for signature, passage_refs in dict(payload.get("slot_to_source_support_passages", {})).items()
        },
        "slot_to_anchor_source_support_passages": {
            _decode_signature(signature): [_normalize_ref(value) for value in passage_refs]
            for signature, passage_refs in dict(payload.get("slot_to_anchor_source_support_passages", {})).items()
        },
        "slot_to_anchor_support_passages": {
            _decode_signature(signature): [_normalize_ref(value) for value in passage_refs]
            for signature, passage_refs in dict(payload.get("slot_to_anchor_support_passages", {})).items()
        },
        "slot_to_target_support_passages": {
            _decode_signature(signature): [_normalize_ref(value) for value in passage_refs]
            for signature, passage_refs in dict(payload.get("slot_to_target_support_passages", {})).items()
        },
        "slot_to_answer_support_passages": {
            _decode_signature(signature): [_normalize_ref(value) for value in passage_refs]
            for signature, passage_refs in dict(payload.get("slot_to_answer_support_passages", {})).items()
        },
        "passage_inventories": {},
    }
    for passage_id, inventory in dict(payload.get("passage_inventories", {})).items():
        sidecar["passage_inventories"][str(passage_id)] = {
            "proposition_ids": [str(value) for value in inventory.get("proposition_ids", [])],
            "pair_signatures": [_decode_signature(signature) for signature in inventory.get("pair_signatures", [])],
            "subj_endpoint_signatures": [
                _decode_signature(signature) for signature in inventory.get("subj_endpoint_signatures", [])
            ],
            "obj_endpoint_signatures": [
                _decode_signature(signature) for signature in inventory.get("obj_endpoint_signatures", [])
            ],
            "slot_signatures": [_decode_signature(signature) for signature in inventory.get("slot_signatures", [])],
            "subject_slot_signatures": [
                _decode_signature(signature) for signature in inventory.get("subject_slot_signatures", [])
            ],
            "object_slot_signatures": [
                _decode_signature(signature) for signature in inventory.get("object_slot_signatures", [])
            ],
            "participant_slot_signatures": [
                _decode_signature(signature) for signature in inventory.get("participant_slot_signatures", [])
            ],
            "relation_signatures": [str(value) for value in inventory.get("relation_signatures", [])],
            "subject_entities": [str(value) for value in inventory.get("subject_entities", [])],
            "object_entities": [str(value) for value in inventory.get("object_entities", [])],
            "participant_entities": [str(value) for value in inventory.get("participant_entities", [])],
        }
    return sidecar
