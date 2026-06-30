import json
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .utils.misc_utils import compute_mdhash_id

BRIDGE_PATH_RECORD_VERSION = 1
BRIDGE_SUPPORT_TRAIL_RECORD_VERSION = 1
BRIDGE_SUPPORT_TRAIL_MEMBERSHIP_RECORD_VERSION = 1
BRIDGE_SUPPORT_ENDPOINT_RECORD_VERSION = 1


def _normalize_id(value: Any) -> str:
    return str(value).strip()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


def normalize_relation_family(value: Any) -> str:
    """Map noisy OpenIE relation text to a coarse retrieval-oriented family."""
    lowered = _normalize_text(value).lower().replace("_", " ").replace("-", " ")
    lowered = " ".join(lowered.split())
    if not lowered:
        return ""
    if any(token in lowered for token in ("born", "birth")):
        return "birth"
    if any(token in lowered for token in ("died", "death")):
        return "death"
    if any(token in lowered for token in ("release", "released", "publish", "published")):
        return "release"
    if any(token in lowered for token in ("establish", "founded", "found", "formed", "created")):
        return "establish"
    if any(token in lowered for token in ("start", "started", "begin", "began", "commence")):
        return "start"
    if any(token in lowered for token in ("end", "ended", "disestablish", "abolish", "dissolve", "cease")):
        return "end"
    if any(
        token in lowered
        for token in (
            "located",
            "country",
            "county",
            "city",
            "capital",
            "headquarter",
            "part of",
            "contains",
            "empty into",
            "mouth",
        )
    ) or lowered in {"in", "at", "from", "within", "is in", "was in"}:
        return "location"
    if any(
        token in lowered
        for token in (
            "written by",
            "created by",
            "directed by",
            "performed by",
            "played by",
            "starring",
            "director",
            "writer",
            "performer",
            "actor",
            "singer",
            "composer",
            "producer",
        )
    ):
        return "agent_role"
    if any(token in lowered for token in ("married", "spouse", "husband", "wife")):
        return "spouse"
    if any(token in lowered for token in ("father", "mother", "child", "son", "daughter")):
        return "kinship"
    if any(token in lowered for token in ("number", "count", "episodes", "season", "times")):
        return "count"
    compact = "".join(ch if ch.isalnum() else " " for ch in lowered)
    tokens = [token for token in compact.split() if token]
    return "other:" + "_".join(tokens[:3]) if tokens else ""


def _normalize_id_list(values: Sequence[Any]) -> List[str]:
    return sorted({_normalize_id(value) for value in values if _normalize_id(value)})


def _role_map_for_entities(record: Mapping[str, Any], entity_ids: Sequence[str]) -> Dict[str, List[str]]:
    participant_roles = {
        _normalize_id(entity_id): sorted({_normalize_text(role).lower() for role in roles if _normalize_text(role)})
        for entity_id, roles in dict(record.get("participant_roles", {})).items()
    }
    return {
        entity_id: participant_roles.get(entity_id, [])
        for entity_id in entity_ids
    }


def _participant_text_map(record: Mapping[str, Any]) -> Dict[str, str]:
    participant_ids = [_normalize_id(value) for value in record.get("participant_ids", [])]
    participant_texts = [_normalize_text(value) for value in record.get("participant_texts", [])]
    return {
        entity_id: participant_texts[idx] if idx < len(participant_texts) else ""
        for idx, entity_id in enumerate(participant_ids)
        if entity_id
    }


def _support_pair_ids(left_passages: Sequence[str], right_passages: Sequence[str]) -> List[str]:
    pairs = []
    for left in left_passages:
        for right in right_passages:
            if not left or not right:
                continue
            pairs.append(f"{left}::{right}")
    return sorted(set(pairs))


def _transition_type(
    *,
    shared_entity_ids: Sequence[str],
    shared_source_ids: Sequence[str],
    left_record: Mapping[str, Any],
    right_record: Mapping[str, Any],
) -> str:
    left_conflict = _normalize_id(left_record.get("conflict_group", ""))
    right_conflict = _normalize_id(right_record.get("conflict_group", ""))
    if left_conflict and left_conflict == right_conflict:
        return "comparison_operand_pair"
    left_relation = _normalize_text(left_record.get("relation_type", ""))
    right_relation = _normalize_text(right_record.get("relation_type", ""))
    if shared_entity_ids and left_relation and left_relation == right_relation:
        return "relation_continuation"
    if shared_source_ids:
        return "same_source_chain"
    if shared_entity_ids:
        return "shared_entity_bridge"
    return "source_to_entity_expansion"


def _path_sort_key(record: Mapping[str, Any]) -> Tuple[int, int, int, int, str]:
    return (
        int(record.get("entity_overlap", 0)),
        int(bool(record.get("source_overlap", False))),
        len(record.get("support_passage_pair_ids", [])),
        len(record.get("start_source_passage_ids", [])) + len(record.get("end_source_passage_ids", [])),
        str(record.get("path_id", "")),
    )


def _build_bridge_path_record(
    *,
    left_bridge_id: str,
    right_bridge_id: str,
    left_record: Mapping[str, Any],
    right_record: Mapping[str, Any],
    shared_entity_ids: Sequence[str],
    shared_source_ids: Sequence[str],
) -> Dict[str, Any]:
    left_fact_id = _normalize_id(left_record.get("fact_embedding_hash_id") or left_bridge_id)
    right_fact_id = _normalize_id(right_record.get("fact_embedding_hash_id") or right_bridge_id)
    left_sources = _normalize_id_list(left_record.get("source_ids", []))
    right_sources = _normalize_id_list(right_record.get("source_ids", []))
    shared_entities = _normalize_id_list(shared_entity_ids)
    left_texts = _participant_text_map(left_record)
    right_texts = _participant_text_map(right_record)
    shared_texts = []
    for entity_id in shared_entities:
        text = left_texts.get(entity_id) or right_texts.get(entity_id) or entity_id
        if text and text not in shared_texts:
            shared_texts.append(text)

    left_relation = _normalize_text(left_record.get("relation_type", ""))
    right_relation = _normalize_text(right_record.get("relation_type", ""))
    left_relation_family = normalize_relation_family(left_relation)
    right_relation_family = normalize_relation_family(right_relation)
    transition = _transition_type(
        shared_entity_ids=shared_entities,
        shared_source_ids=shared_source_ids,
        left_record=left_record,
        right_record=right_record,
    )
    support_pairs = _support_pair_ids(left_sources, right_sources)
    signature_payload = {
        "left_bridge_id": left_bridge_id,
        "right_bridge_id": right_bridge_id,
        "shared_entity_ids": shared_entities,
        "shared_source_ids": _normalize_id_list(shared_source_ids),
        "transition_type": transition,
    }
    path_id = compute_mdhash_id(
        json.dumps(signature_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        prefix="bridge-path-",
    )
    left_summary = _normalize_text(left_record.get("summary_text", ""))
    right_summary = _normalize_text(right_record.get("summary_text", ""))
    path_text_parts = [
        transition,
        " | ".join(shared_texts),
        left_summary,
        right_summary,
    ]
    path_text = " ; ".join(part for part in path_text_parts if part)
    return {
        "hash_id": path_id,
        "path_id": path_id,
        "record_version": BRIDGE_PATH_RECORD_VERSION,
        "start_bridge_id": left_bridge_id,
        "end_bridge_id": right_bridge_id,
        "start_fact_id": left_fact_id,
        "end_fact_id": right_fact_id,
        "shared_entity_ids": shared_entities,
        "shared_entity_texts": shared_texts,
        "transition_type": transition,
        "start_source_passage_ids": left_sources,
        "end_source_passage_ids": right_sources,
        "shared_source_passage_ids": _normalize_id_list(shared_source_ids),
        "support_passage_pair_ids": support_pairs,
        "start_relation_type": left_relation,
        "end_relation_type": right_relation,
        "start_relation_family": left_relation_family,
        "end_relation_family": right_relation_family,
        "start_participant_roles": _role_map_for_entities(left_record, shared_entities),
        "end_participant_roles": _role_map_for_entities(right_record, shared_entities),
        "source_overlap": bool(shared_source_ids),
        "entity_overlap": len(shared_entities),
        "relation_signature_pair": [left_relation, right_relation],
        "relation_family_pair": [left_relation_family, right_relation_family],
        "path_text": path_text,
        "path_embedding_text": path_text,
    }


def _bounded_sorted_ids(values: Sequence[str], limit: int) -> List[str]:
    unique = sorted({_normalize_id(value) for value in values if _normalize_id(value)})
    if limit <= 0:
        return []
    return unique[:limit]


def _local_neighbor_ids(values: Sequence[str], member_id: str, limit: int) -> List[str]:
    if limit <= 0:
        return []
    unique = sorted({_normalize_id(value) for value in values if _normalize_id(value)})
    member = _normalize_id(member_id)
    if not unique or not member:
        return []
    if member not in unique:
        return [value for value in unique if value != member][:limit]

    member_idx = unique.index(member)
    neighbors: List[str] = []
    offset = 1
    while len(neighbors) < limit and (member_idx - offset >= 0 or member_idx + offset < len(unique)):
        left_idx = member_idx - offset
        right_idx = member_idx + offset
        if left_idx >= 0:
            neighbors.append(unique[left_idx])
            if len(neighbors) >= limit:
                break
        if right_idx < len(unique):
            neighbors.append(unique[right_idx])
        offset += 1
    return neighbors[:limit]


def _add_pair_feature(
    pair_features: Dict[Tuple[str, str], Dict[str, set]],
    bridge_pair_counts: Dict[str, int],
    *,
    left_bridge_id: str,
    right_bridge_id: str,
    feature_key: str,
    feature_value: str,
    max_pairs_per_bridge: int,
    max_candidate_pairs: int,
) -> bool:
    if not left_bridge_id or not right_bridge_id or left_bridge_id == right_bridge_id:
        return False
    left_id, right_id = sorted((left_bridge_id, right_bridge_id))
    pair_key = (left_id, right_id)
    existing = pair_features.get(pair_key)
    if existing is None:
        if max_candidate_pairs > 0 and len(pair_features) >= max_candidate_pairs:
            return False
        if max_pairs_per_bridge > 0 and (
            bridge_pair_counts[left_id] >= max_pairs_per_bridge
            or bridge_pair_counts[right_id] >= max_pairs_per_bridge
        ):
            return False
        existing = {"shared_entity_ids": set(), "shared_source_ids": set()}
        pair_features[pair_key] = existing
        bridge_pair_counts[left_id] += 1
        bridge_pair_counts[right_id] += 1
    normalized_feature_value = _normalize_id(feature_value)
    if normalized_feature_value:
        existing[feature_key].add(normalized_feature_value)
    return True


def build_bridge_path_records_from_hyperedge_records(
    hyperedge_records: Mapping[str, Dict[str, Any]],
    *,
    max_hyperedges_per_entity: int = 64,
    max_hyperedges_per_source: int = 64,
    max_paths_per_fact: int = 32,
    max_pairs_per_bridge: int = 64,
    max_candidate_pairs: int = 0,
    max_seed_source_neighbors_per_bridge: int = 2,
    max_seed_entity_neighbors_per_bridge: int = 2,
) -> List[Dict[str, Any]]:
    """Compile source-grounded fact-transition objects from hyperedge records.

    The resulting records are addressable index-time objects. They are not
    replacement readout rules: each path carries the fact pair, shared bridge
    entity/source evidence, and passage-pair provenance needed by downstream
    retrieval channels.
    """

    if not hyperedge_records:
        return []

    normalized_records = {
        _normalize_id(bridge_id): dict(record)
        for bridge_id, record in hyperedge_records.items()
        if _normalize_id(bridge_id)
    }
    entity_to_bridges: Dict[str, List[str]] = defaultdict(list)
    source_to_bridges: Dict[str, List[str]] = defaultdict(list)
    pair_features: Dict[Tuple[str, str], Dict[str, set]] = defaultdict(
        lambda: {"shared_entity_ids": set(), "shared_source_ids": set()}
    )
    source_bridge_pair_counts: Dict[str, int] = defaultdict(int)
    entity_bridge_pair_counts: Dict[str, int] = defaultdict(int)
    source_candidate_pair_limit = max(max_candidate_pairs // 2, 1) if max_candidate_pairs > 0 else 0

    for bridge_id, record in normalized_records.items():
        for entity_id in record.get("participant_ids", []):
            normalized_entity_id = _normalize_id(entity_id)
            if normalized_entity_id:
                entity_to_bridges[normalized_entity_id].append(bridge_id)
        for source_id in record.get("source_ids", []):
            normalized_source_id = _normalize_id(source_id)
            if normalized_source_id:
                source_to_bridges[normalized_source_id].append(bridge_id)

    for bridge_id in sorted(normalized_records.keys()):
        record = normalized_records[bridge_id]
        source_neighbor_budget = int(max_seed_source_neighbors_per_bridge)
        for source_id in _normalize_id_list(record.get("source_ids", [])):
            if source_neighbor_budget <= 0:
                break
            neighbors = _local_neighbor_ids(source_to_bridges.get(source_id, []), bridge_id, source_neighbor_budget)
            for neighbor_id in neighbors:
                added = _add_pair_feature(
                    pair_features,
                    source_bridge_pair_counts,
                    left_bridge_id=bridge_id,
                    right_bridge_id=neighbor_id,
                    feature_key="shared_source_ids",
                    feature_value=source_id,
                    max_pairs_per_bridge=max_pairs_per_bridge,
                    max_candidate_pairs=source_candidate_pair_limit,
                )
                if added:
                    source_neighbor_budget -= 1
                if source_neighbor_budget <= 0:
                    break

        entity_neighbor_budget = int(max_seed_entity_neighbors_per_bridge)
        for entity_id in _normalize_id_list(record.get("participant_ids", [])):
            if entity_neighbor_budget <= 0:
                break
            neighbors = _local_neighbor_ids(entity_to_bridges.get(entity_id, []), bridge_id, entity_neighbor_budget)
            for neighbor_id in neighbors:
                added = _add_pair_feature(
                    pair_features,
                    entity_bridge_pair_counts,
                    left_bridge_id=bridge_id,
                    right_bridge_id=neighbor_id,
                    feature_key="shared_entity_ids",
                    feature_value=entity_id,
                    max_pairs_per_bridge=max_pairs_per_bridge,
                    max_candidate_pairs=max_candidate_pairs,
                )
                if added:
                    entity_neighbor_budget -= 1
                if entity_neighbor_budget <= 0:
                    break

    # Source-local links are processed first because they preserve concrete
    # passage provenance and should not be displaced by high-frequency entity
    # hubs under the per-bridge pair budget.
    for source_id, bridge_ids in source_to_bridges.items():
        bounded_bridge_ids = _bounded_sorted_ids(bridge_ids, max_hyperedges_per_source)
        for left_idx, left_bridge_id in enumerate(bounded_bridge_ids):
            for right_bridge_id in bounded_bridge_ids[left_idx + 1 :]:
                _add_pair_feature(
                    pair_features,
                    source_bridge_pair_counts,
                    left_bridge_id=left_bridge_id,
                    right_bridge_id=right_bridge_id,
                    feature_key="shared_source_ids",
                    feature_value=source_id,
                    max_pairs_per_bridge=max_pairs_per_bridge,
                    max_candidate_pairs=source_candidate_pair_limit,
                )

    for entity_id, bridge_ids in entity_to_bridges.items():
        bounded_bridge_ids = _bounded_sorted_ids(bridge_ids, max_hyperedges_per_entity)
        for left_idx, left_bridge_id in enumerate(bounded_bridge_ids):
            for right_bridge_id in bounded_bridge_ids[left_idx + 1 :]:
                _add_pair_feature(
                    pair_features,
                    entity_bridge_pair_counts,
                    left_bridge_id=left_bridge_id,
                    right_bridge_id=right_bridge_id,
                    feature_key="shared_entity_ids",
                    feature_value=entity_id,
                    max_pairs_per_bridge=max_pairs_per_bridge,
                    max_candidate_pairs=max_candidate_pairs,
                )

    path_records_by_id: Dict[str, Dict[str, Any]] = {}
    for (left_bridge_id, right_bridge_id), features in pair_features.items():
        left_record = normalized_records.get(left_bridge_id)
        right_record = normalized_records.get(right_bridge_id)
        if not left_record or not right_record:
            continue
        left_sources = _normalize_id_list(left_record.get("source_ids", []))
        right_sources = _normalize_id_list(right_record.get("source_ids", []))
        if not left_sources or not right_sources:
            continue
        record = _build_bridge_path_record(
            left_bridge_id=left_bridge_id,
            right_bridge_id=right_bridge_id,
            left_record=left_record,
            right_record=right_record,
            shared_entity_ids=features["shared_entity_ids"],
            shared_source_ids=features["shared_source_ids"],
        )
        path_records_by_id[record["hash_id"]] = record

    if max_paths_per_fact <= 0:
        return sorted(path_records_by_id.values(), key=lambda record: str(record["hash_id"]))

    fact_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in path_records_by_id.values():
        fact_to_records[str(record["start_fact_id"])].append(record)
        fact_to_records[str(record["end_fact_id"])].append(record)

    retained_path_ids = set()
    for records in fact_to_records.values():
        sorted_records = sorted(records, key=_path_sort_key, reverse=True)
        for record in sorted_records[:max_paths_per_fact]:
            retained_path_ids.add(str(record["hash_id"]))

    return [
        path_records_by_id[path_id]
        for path_id in sorted(retained_path_ids)
        if path_id in path_records_by_id
    ]


def build_bridge_path_record_ids_from_hyperedge_records(
    hyperedge_records: Mapping[str, Dict[str, Any]],
    **kwargs,
) -> List[str]:
    return [
        str(record["hash_id"])
        for record in build_bridge_path_records_from_hyperedge_records(hyperedge_records, **kwargs)
    ]


def _bridge_path_id(record: Mapping[str, Any]) -> str:
    return _normalize_id(record.get("path_id") or record.get("hash_id"))


def _bridge_path_fact_ids(record: Mapping[str, Any]) -> Tuple[str, str]:
    return (
        _normalize_id(record.get("start_fact_id", "")),
        _normalize_id(record.get("end_fact_id", "")),
    )


def _bridge_path_sources_for_fact(record: Mapping[str, Any], fact_id: str) -> List[str]:
    normalized_fact_id = _normalize_id(fact_id)
    start_fact_id, end_fact_id = _bridge_path_fact_ids(record)
    if normalized_fact_id == start_fact_id:
        return _normalize_id_list(record.get("start_source_passage_ids", []))
    if normalized_fact_id == end_fact_id:
        return _normalize_id_list(record.get("end_source_passage_ids", []))
    return []


def _bridge_path_all_sources(record: Mapping[str, Any]) -> List[str]:
    return _normalize_id_list(
        list(record.get("start_source_passage_ids", []))
        + list(record.get("end_source_passage_ids", []))
        + list(record.get("shared_source_passage_ids", []))
    )


def _bridge_path_relation_family_for_fact(record: Mapping[str, Any], fact_id: str) -> str:
    normalized_fact_id = _normalize_id(fact_id)
    start_fact_id, end_fact_id = _bridge_path_fact_ids(record)
    if normalized_fact_id == start_fact_id:
        return _normalize_text(record.get("start_relation_family", ""))
    if normalized_fact_id == end_fact_id:
        return _normalize_text(record.get("end_relation_family", ""))
    return ""


def _trail_step_sort_key(edge: Mapping[str, Any]) -> Tuple[int, int, str, str]:
    record = edge["record"]
    return (
        int(not bool(record.get("source_overlap", False))),
        -int(record.get("entity_overlap", 0)),
        str(edge["neighbor_fact_id"]),
        str(edge["bridge_path_id"]),
    )


def _build_bridge_path_adjacency(
    bridge_path_records: Sequence[Mapping[str, Any]],
    *,
    max_neighbors_per_fact: int,
) -> Dict[str, List[Dict[str, Any]]]:
    adjacency: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in bridge_path_records:
        path_id = _bridge_path_id(record)
        left_fact_id, right_fact_id = _bridge_path_fact_ids(record)
        if not path_id or not left_fact_id or not right_fact_id or left_fact_id == right_fact_id:
            continue
        left_edge = {
            "neighbor_fact_id": right_fact_id,
            "bridge_path_id": path_id,
            "record": record,
        }
        right_edge = {
            "neighbor_fact_id": left_fact_id,
            "bridge_path_id": path_id,
            "record": record,
        }
        adjacency[left_fact_id].append(left_edge)
        adjacency[right_fact_id].append(right_edge)

    bounded: Dict[str, List[Dict[str, Any]]] = {}
    for fact_id, edges in adjacency.items():
        sorted_edges = sorted(edges, key=_trail_step_sort_key)
        if max_neighbors_per_fact > 0:
            sorted_edges = sorted_edges[:max_neighbors_per_fact]
        bounded[fact_id] = sorted_edges
    return bounded


def _build_bridge_support_trail_record(
    *,
    fact_id_sequence: Sequence[str],
    bridge_path_id_sequence: Sequence[str],
    bridge_path_records_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    normalized_fact_ids = [_normalize_id(value) for value in fact_id_sequence if _normalize_id(value)]
    normalized_path_ids = [_normalize_id(value) for value in bridge_path_id_sequence if _normalize_id(value)]
    start_fact_id = normalized_fact_ids[0]
    end_fact_id = normalized_fact_ids[-1]
    intermediate_fact_ids = normalized_fact_ids[1:-1]
    hop_count = len(normalized_path_ids)

    support_passage_ids: set[str] = set()
    shared_entity_ids: set[str] = set()
    shared_source_passage_ids: set[str] = set()
    transition_types: List[str] = []
    path_texts: List[str] = []
    relation_family_by_fact: Dict[str, str] = {}
    for path_id in normalized_path_ids:
        record = bridge_path_records_by_id.get(path_id, {})
        support_passage_ids.update(_bridge_path_all_sources(record))
        shared_entity_ids.update(_normalize_id_list(record.get("shared_entity_ids", [])))
        shared_source_passage_ids.update(_normalize_id_list(record.get("shared_source_passage_ids", [])))
        transition_type = _normalize_text(record.get("transition_type", ""))
        if transition_type:
            transition_types.append(transition_type)
        path_text = _normalize_text(record.get("path_text", ""))
        if path_text:
            path_texts.append(path_text)
        for fact_id in normalized_fact_ids:
            if fact_id in relation_family_by_fact:
                continue
            relation_family = _bridge_path_relation_family_for_fact(record, fact_id)
            if relation_family:
                relation_family_by_fact[fact_id] = relation_family

    relation_family_sequence = [
        relation_family_by_fact.get(fact_id, "")
        for fact_id in normalized_fact_ids
    ]

    endpoint_passage_ids: set[str] = set()
    if normalized_path_ids:
        endpoint_record = bridge_path_records_by_id.get(normalized_path_ids[-1], {})
        endpoint_passage_ids.update(_bridge_path_sources_for_fact(endpoint_record, end_fact_id))
    start_passage_ids: set[str] = set()
    if normalized_path_ids:
        start_record = bridge_path_records_by_id.get(normalized_path_ids[0], {})
        start_passage_ids.update(_bridge_path_sources_for_fact(start_record, start_fact_id))

    signature_payload = {
        "fact_id_sequence": normalized_fact_ids,
        "bridge_path_id_sequence": normalized_path_ids,
    }
    trail_id = compute_mdhash_id(
        json.dumps(signature_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        prefix="bridge-trail-",
    )
    trail_text_parts = [
        f"hop_count={hop_count}",
        " -> ".join(normalized_fact_ids),
        " | ".join(transition_types),
        " || ".join(path_texts),
    ]
    trail_text = " ; ".join(part for part in trail_text_parts if part)
    return {
        "hash_id": trail_id,
        "trail_id": trail_id,
        "record_version": BRIDGE_SUPPORT_TRAIL_RECORD_VERSION,
        "start_fact_id": start_fact_id,
        "end_fact_id": end_fact_id,
        "intermediate_fact_ids": intermediate_fact_ids,
        "fact_id_sequence": normalized_fact_ids,
        "bridge_path_id_sequence": normalized_path_ids,
        "hop_count": hop_count,
        "start_source_passage_ids": sorted(start_passage_ids),
        "end_source_passage_ids": sorted(endpoint_passage_ids),
        "support_passage_ids": sorted(support_passage_ids),
        "shared_entity_ids": sorted(shared_entity_ids),
        "shared_source_passage_ids": sorted(shared_source_passage_ids),
        "transition_types": transition_types,
        "start_relation_family": relation_family_sequence[0] if relation_family_sequence else "",
        "end_relation_family": relation_family_sequence[-1] if relation_family_sequence else "",
        "relation_family_sequence": relation_family_sequence,
        "trail_text": trail_text,
        "trail_embedding_text": trail_text,
    }


def build_bridge_support_trail_records_from_bridge_path_records(
    bridge_path_records: Sequence[Mapping[str, Any]],
    *,
    min_depth: int = 2,
    max_depth: int = 2,
    max_neighbors_per_fact: int = 4,
    max_trails_per_start_fact: int = 8,
) -> List[Dict[str, Any]]:
    """Compile bounded multi-hop support trails over BridgePath objects.

    A support trail is an index-time object, not a query-time selector. It
    preserves the exact fact sequence, BridgePath sequence, endpoint passages,
    and shared provenance needed by downstream support assembly and factroute
    credit diagnostics.
    """

    if not bridge_path_records:
        return []
    min_depth = max(int(min_depth), 1)
    max_depth = max(int(max_depth), min_depth)
    if max_trails_per_start_fact <= 0:
        return []

    bridge_path_records_by_id = {
        _bridge_path_id(record): record
        for record in bridge_path_records
        if _bridge_path_id(record)
    }
    adjacency = _build_bridge_path_adjacency(
        bridge_path_records,
        max_neighbors_per_fact=int(max_neighbors_per_fact),
    )

    trail_records_by_id: Dict[str, Dict[str, Any]] = {}
    for start_fact_id in sorted(adjacency.keys()):
        queue = [([start_fact_id], [])]
        emitted_for_start = 0
        while queue and emitted_for_start < max_trails_per_start_fact:
            fact_sequence, path_sequence = queue.pop(0)
            current_fact_id = fact_sequence[-1]
            current_depth = len(path_sequence)
            if current_depth >= max_depth:
                continue
            for edge in adjacency.get(current_fact_id, []):
                neighbor_fact_id = _normalize_id(edge["neighbor_fact_id"])
                bridge_path_id = _normalize_id(edge["bridge_path_id"])
                if not neighbor_fact_id or neighbor_fact_id in fact_sequence or not bridge_path_id:
                    continue
                next_fact_sequence = list(fact_sequence) + [neighbor_fact_id]
                next_path_sequence = list(path_sequence) + [bridge_path_id]
                next_depth = len(next_path_sequence)
                if next_depth >= min_depth:
                    record = _build_bridge_support_trail_record(
                        fact_id_sequence=next_fact_sequence,
                        bridge_path_id_sequence=next_path_sequence,
                        bridge_path_records_by_id=bridge_path_records_by_id,
                    )
                    trail_id = str(record["hash_id"])
                    if trail_id not in trail_records_by_id:
                        trail_records_by_id[trail_id] = record
                        emitted_for_start += 1
                        if emitted_for_start >= max_trails_per_start_fact:
                            break
                if next_depth < max_depth:
                    queue.append((next_fact_sequence, next_path_sequence))

    return [
        trail_records_by_id[trail_id]
        for trail_id in sorted(trail_records_by_id)
    ]


def build_bridge_support_trail_record_ids_from_bridge_path_records(
    bridge_path_records: Sequence[Mapping[str, Any]],
    **kwargs,
) -> List[str]:
    return [
        str(record["hash_id"])
        for record in build_bridge_support_trail_records_from_bridge_path_records(
            bridge_path_records,
            **kwargs,
        )
    ]


def _support_trail_endpoint_signature(record: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(_normalize_id_list(record.get("end_source_passage_ids", [])))


def _support_trail_policy_features(
    record: Mapping[str, Any],
    bridge_path_records_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    path_records = [
        bridge_path_records_by_id.get(path_id, {})
        for path_id in _normalize_id_list(record.get("bridge_path_id_sequence", []))
    ]
    cross_source_hops = sum(1 for path_record in path_records if not bool(path_record.get("source_overlap", False)))
    same_source_hops = sum(1 for path_record in path_records if bool(path_record.get("source_overlap", False)))
    entity_overlap_sum = sum(int(path_record.get("entity_overlap", 0)) for path_record in path_records)
    shared_entity_bridge_hops = sum(
        1
        for path_record in path_records
        if _normalize_text(path_record.get("transition_type", "")) == "shared_entity_bridge"
    )
    endpoint_sources = set(_normalize_id_list(record.get("end_source_passage_ids", [])))
    start_sources = set(_normalize_id_list(record.get("start_source_passage_ids", [])))
    support_sources = set(_normalize_id_list(record.get("support_passage_ids", [])))
    return {
        "cross_source_hops": cross_source_hops,
        "same_source_hops": same_source_hops,
        "entity_overlap_sum": entity_overlap_sum,
        "shared_entity_bridge_hops": shared_entity_bridge_hops,
        "endpoint_novel_to_start": bool(endpoint_sources and endpoint_sources.isdisjoint(start_sources)),
        "support_source_count": len(support_sources),
    }


def _reserved_bridge_trail_sort_key(
    record: Mapping[str, Any],
    bridge_path_records_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[int, int, int, int, int, str]:
    features = _support_trail_policy_features(record, bridge_path_records_by_id)
    return (
        -int(features["endpoint_novel_to_start"]),
        -int(features["cross_source_hops"]),
        -int(features["shared_entity_bridge_hops"]),
        -int(features["entity_overlap_sum"]),
        int(features["same_source_hops"]),
        str(record.get("trail_id") or record.get("hash_id") or ""),
    )


def _select_reserved_bridge_support_trails(
    candidates: Sequence[Mapping[str, Any]],
    *,
    bridge_path_records_by_id: Mapping[str, Mapping[str, Any]],
    max_trails_per_start_fact: int,
    reserved_prefix_trails: int,
) -> List[Dict[str, Any]]:
    """Keep mostly default BFS trails while reserving one structural bridge slot."""

    limit = int(max_trails_per_start_fact)
    if limit <= 0:
        return []
    prefix_limit = min(max(int(reserved_prefix_trails), 0), max(limit - 1, 0))
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_endpoint_signatures: set[Tuple[str, ...]] = set()

    for candidate in candidates[:prefix_limit]:
        trail_id = _support_trail_id(candidate)
        if not trail_id or trail_id in selected_ids:
            continue
        selected.append(dict(candidate))
        selected_ids.add(trail_id)
        signature = _support_trail_endpoint_signature(candidate)
        if signature:
            selected_endpoint_signatures.add(signature)

    bridge_candidates: List[Dict[str, Any]] = []
    for candidate in candidates:
        trail_id = _support_trail_id(candidate)
        if not trail_id or trail_id in selected_ids:
            continue
        features = _support_trail_policy_features(candidate, bridge_path_records_by_id)
        if int(features["cross_source_hops"]) <= 0:
            continue
        if int(features["shared_entity_bridge_hops"]) <= 0:
            continue
        signature = _support_trail_endpoint_signature(candidate)
        if signature and signature in selected_endpoint_signatures:
            continue
        bridge_candidates.append(dict(candidate))

    if len(selected) < limit and bridge_candidates:
        bridge_record = sorted(
            bridge_candidates,
            key=lambda record: _reserved_bridge_trail_sort_key(record, bridge_path_records_by_id),
        )[0]
        trail_id = _support_trail_id(bridge_record)
        if trail_id and trail_id not in selected_ids:
            selected.append(dict(bridge_record))
            selected_ids.add(trail_id)

    for candidate in candidates:
        if len(selected) >= limit:
            break
        trail_id = _support_trail_id(candidate)
        if not trail_id or trail_id in selected_ids:
            continue
        selected.append(dict(candidate))
        selected_ids.add(trail_id)
    return selected[:limit]


def _build_depth_two_support_trail_candidates_for_start(
    *,
    start_fact_id: str,
    adjacency: Mapping[str, Sequence[Mapping[str, Any]]],
    bridge_path_records_by_id: Mapping[str, Mapping[str, Any]],
    max_candidate_trails_per_start_fact: int,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    candidate_limit = max(int(max_candidate_trails_per_start_fact), 0)
    if candidate_limit <= 0:
        return candidates
    normalized_start_id = _normalize_id(start_fact_id)
    for first_edge in adjacency.get(normalized_start_id, []):
        mid_fact_id = _normalize_id(first_edge.get("neighbor_fact_id", ""))
        first_path_id = _normalize_id(first_edge.get("bridge_path_id", ""))
        if not mid_fact_id or not first_path_id or mid_fact_id == normalized_start_id:
            continue
        for second_edge in adjacency.get(mid_fact_id, []):
            end_fact_id = _normalize_id(second_edge.get("neighbor_fact_id", ""))
            second_path_id = _normalize_id(second_edge.get("bridge_path_id", ""))
            if not end_fact_id or not second_path_id:
                continue
            if end_fact_id in {normalized_start_id, mid_fact_id}:
                continue
            record = _build_bridge_support_trail_record(
                fact_id_sequence=[normalized_start_id, mid_fact_id, end_fact_id],
                bridge_path_id_sequence=[first_path_id, second_path_id],
                bridge_path_records_by_id=bridge_path_records_by_id,
            )
            trail_id = _support_trail_id(record)
            if not trail_id or trail_id in seen_candidate_ids:
                continue
            candidates.append(record)
            seen_candidate_ids.add(trail_id)
            if len(candidates) >= candidate_limit:
                return candidates
    return candidates


def build_reserved_bridge_support_trail_records_from_bridge_path_records(
    bridge_path_records: Sequence[Mapping[str, Any]],
    *,
    min_depth: int = 2,
    max_depth: int = 2,
    candidate_max_neighbors_per_fact: int = 5,
    max_trails_per_start_fact: int = 8,
    max_candidate_trails_per_start_fact: int = 32,
    reserved_prefix_trails: int = 7,
) -> List[Dict[str, Any]]:
    """Compile depth-2 support trails with one reserved structural bridge slot.

    This is an additive index-time construction policy. It preserves the
    existing support-trail schema and leaves the default BFS builder unchanged.
    The policy keeps most default BFS trails for each start fact and reserves
    one slot for a cross-source shared-entity bridge trail when such a trail is
    present in the bounded candidate neighborhood.
    """

    if not bridge_path_records:
        return []
    min_depth = max(int(min_depth), 1)
    max_depth = max(int(max_depth), min_depth)
    if min_depth != 2 or max_depth != 2:
        raise ValueError("reserved_bridge support-trail construction currently supports depth-2 only")
    if max_trails_per_start_fact <= 0:
        return []

    bridge_path_records_by_id = {
        _bridge_path_id(record): record
        for record in bridge_path_records
        if _bridge_path_id(record)
    }
    adjacency = _build_bridge_path_adjacency(
        bridge_path_records,
        max_neighbors_per_fact=int(candidate_max_neighbors_per_fact),
    )
    retained_by_id: Dict[str, Dict[str, Any]] = {}
    candidate_limit = max(int(max_candidate_trails_per_start_fact), int(max_trails_per_start_fact))
    for start_fact_id in sorted(adjacency.keys()):
        candidates = _build_depth_two_support_trail_candidates_for_start(
            start_fact_id=start_fact_id,
            adjacency=adjacency,
            bridge_path_records_by_id=bridge_path_records_by_id,
            max_candidate_trails_per_start_fact=candidate_limit,
        )
        selected = _select_reserved_bridge_support_trails(
            candidates,
            bridge_path_records_by_id=bridge_path_records_by_id,
            max_trails_per_start_fact=int(max_trails_per_start_fact),
            reserved_prefix_trails=int(reserved_prefix_trails),
        )
        for record in selected:
            trail_id = _support_trail_id(record)
            if trail_id and trail_id not in retained_by_id:
                retained_by_id[trail_id] = record

    return [
        retained_by_id[trail_id]
        for trail_id in sorted(retained_by_id)
    ]


def build_reserved_bridge_support_trail_record_ids_from_bridge_path_records(
    bridge_path_records: Sequence[Mapping[str, Any]],
    **kwargs,
) -> List[str]:
    return [
        str(record["hash_id"])
        for record in build_reserved_bridge_support_trail_records_from_bridge_path_records(
            bridge_path_records,
            **kwargs,
        )
    ]


def _support_trail_id(record: Mapping[str, Any]) -> str:
    return _normalize_id(record.get("trail_id") or record.get("hash_id"))


def _support_trail_fact_ids(record: Mapping[str, Any]) -> List[str]:
    fact_ids = list(
        dict.fromkeys(
            _normalize_id(value)
            for value in record.get("fact_id_sequence", [])
            if _normalize_id(value)
        )
    )
    if fact_ids:
        return fact_ids
    return _normalize_id_list(
        [record.get("start_fact_id", "")]
        + list(record.get("intermediate_fact_ids", []))
        + [record.get("end_fact_id", "")]
    )


def _support_trail_relation_family_for_fact(record: Mapping[str, Any], fact_id: str) -> str:
    normalized_fact_id = _normalize_id(fact_id)
    fact_ids = _support_trail_fact_ids(record)
    relation_families = [
        _normalize_text(value)
        for value in record.get("relation_family_sequence", [])
    ]
    for idx, candidate_fact_id in enumerate(fact_ids):
        if candidate_fact_id != normalized_fact_id:
            continue
        if idx < len(relation_families):
            return relation_families[idx]
        break
    if normalized_fact_id == _normalize_id(record.get("start_fact_id", "")):
        return _normalize_text(record.get("start_relation_family", ""))
    if normalized_fact_id == _normalize_id(record.get("end_fact_id", "")):
        return _normalize_text(record.get("end_relation_family", ""))
    return ""


def _new_support_trail_membership_bucket() -> Dict[str, set]:
    return {
        "support_trail_ids": set(),
        "support_fact_ids": set(),
        "support_passage_ids": set(),
        "bridge_path_ids": set(),
        "support_relation_families": set(),
        "endpoint_fact_ids": set(),
        "endpoint_passage_ids": set(),
        "start_fact_ids": set(),
        "end_fact_ids": set(),
        "endpoint_fact_to_passage_ids": defaultdict(set),
        "endpoint_fact_to_support_trail_ids": defaultdict(set),
        "endpoint_fact_to_support_passage_ids": defaultdict(set),
        "endpoint_fact_to_bridge_path_ids": defaultdict(set),
        "endpoint_fact_to_roles": defaultdict(set),
        "endpoint_fact_to_hop_counts": defaultdict(set),
        "endpoint_fact_to_relation_families": defaultdict(set),
    }


def _add_support_trail_record_to_membership(
    membership: Dict[str, Dict[str, set]],
    record: Mapping[str, Any],
    *,
    include_endpoint_maps: bool = True,
) -> None:
    trail_id = _support_trail_id(record)
    if not trail_id:
        return
    fact_ids = _support_trail_fact_ids(record)
    if not fact_ids:
        return
    passage_ids = _normalize_id_list(record.get("support_passage_ids", []))
    bridge_path_ids = _normalize_id_list(record.get("bridge_path_id_sequence", []))
    relation_families = _normalize_id_list(record.get("relation_family_sequence", []))
    start_fact_id = _normalize_id(record.get("start_fact_id") or fact_ids[0])
    end_fact_id = _normalize_id(record.get("end_fact_id") or fact_ids[-1])
    endpoint_fact_ids = _normalize_id_list([start_fact_id, end_fact_id])
    start_passage_ids = _normalize_id_list(record.get("start_source_passage_ids", []))
    end_passage_ids = _normalize_id_list(record.get("end_source_passage_ids", []))
    endpoint_passage_ids = _normalize_id_list(start_passage_ids + end_passage_ids)
    hop_count = int(record.get("hop_count", len(bridge_path_ids)))
    endpoints = (
        [
            ("start", start_fact_id, start_passage_ids),
            ("end", end_fact_id, end_passage_ids),
        ]
        if include_endpoint_maps
        else []
    )

    for fact_id in fact_ids:
        bucket = membership[fact_id]
        bucket["support_trail_ids"].add(trail_id)
        bucket["support_fact_ids"].update(fact_ids)
        bucket["support_passage_ids"].update(passage_ids)
        bucket["bridge_path_ids"].update(bridge_path_ids)
        bucket["support_relation_families"].update(relation_families)
        bucket["endpoint_fact_ids"].update(endpoint_fact_ids)
        bucket["endpoint_passage_ids"].update(endpoint_passage_ids)
        if start_fact_id:
            bucket["start_fact_ids"].add(start_fact_id)
        if end_fact_id:
            bucket["end_fact_ids"].add(end_fact_id)
        for endpoint_role, endpoint_fact_id, endpoint_sources in endpoints:
            if not endpoint_fact_id or endpoint_fact_id == fact_id:
                continue
            bucket["endpoint_fact_to_passage_ids"][endpoint_fact_id].update(endpoint_sources)
            bucket["endpoint_fact_to_support_trail_ids"][endpoint_fact_id].add(trail_id)
            bucket["endpoint_fact_to_support_passage_ids"][endpoint_fact_id].update(passage_ids)
            bucket["endpoint_fact_to_bridge_path_ids"][endpoint_fact_id].update(bridge_path_ids)
            bucket["endpoint_fact_to_roles"][endpoint_fact_id].add(endpoint_role)
            bucket["endpoint_fact_to_hop_counts"][endpoint_fact_id].add(str(hop_count))
            endpoint_relation_family = _support_trail_relation_family_for_fact(record, endpoint_fact_id)
            if endpoint_relation_family:
                bucket["endpoint_fact_to_relation_families"][endpoint_fact_id].add(endpoint_relation_family)


def _finalize_nested_id_map(nested: Mapping[str, set]) -> Dict[str, List[str]]:
    return {
        str(key): sorted(_normalize_id(value) for value in values if _normalize_id(value))
        for key, values in sorted(nested.items())
        if str(key)
    }


def _finalize_nested_int_map(nested: Mapping[str, set]) -> Dict[str, List[int]]:
    return {
        str(key): sorted(int(value) for value in values if str(value).strip())
        for key, values in sorted(nested.items())
        if str(key)
    }


def _finalize_support_trail_membership_records(
    membership: Mapping[str, Mapping[str, set]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for fact_id in sorted(membership.keys()):
        bucket = membership[fact_id]
        support_trail_ids = sorted(bucket["support_trail_ids"])
        membership_id = compute_mdhash_id(
            json.dumps({"fact_id": fact_id}, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
            prefix="bridge-trail-membership-",
        )
        records.append(
            {
                "hash_id": membership_id,
                "membership_id": membership_id,
                "record_version": BRIDGE_SUPPORT_TRAIL_MEMBERSHIP_RECORD_VERSION,
                "fact_id": fact_id,
                "support_trail_ids": support_trail_ids,
                "support_trail_count": len(support_trail_ids),
                "support_fact_ids": sorted(bucket["support_fact_ids"]),
                "support_passage_ids": sorted(bucket["support_passage_ids"]),
                "bridge_path_ids": sorted(bucket["bridge_path_ids"]),
                "support_relation_families": sorted(bucket["support_relation_families"]),
                "endpoint_fact_ids": sorted(bucket["endpoint_fact_ids"]),
                "endpoint_passage_ids": sorted(bucket["endpoint_passage_ids"]),
                "start_fact_ids": sorted(bucket["start_fact_ids"]),
                "end_fact_ids": sorted(bucket["end_fact_ids"]),
                "endpoint_fact_to_passage_ids": _finalize_nested_id_map(bucket["endpoint_fact_to_passage_ids"]),
                "endpoint_fact_to_support_trail_ids": _finalize_nested_id_map(bucket["endpoint_fact_to_support_trail_ids"]),
                "endpoint_fact_to_support_passage_ids": _finalize_nested_id_map(bucket["endpoint_fact_to_support_passage_ids"]),
                "endpoint_fact_to_bridge_path_ids": _finalize_nested_id_map(bucket["endpoint_fact_to_bridge_path_ids"]),
                "endpoint_fact_to_roles": _finalize_nested_id_map(bucket["endpoint_fact_to_roles"]),
                "endpoint_fact_to_hop_counts": _finalize_nested_int_map(bucket["endpoint_fact_to_hop_counts"]),
                "endpoint_fact_to_relation_families": _finalize_nested_id_map(
                    bucket["endpoint_fact_to_relation_families"]
                ),
            }
        )
    return records


def build_bridge_support_trail_membership_records_from_support_trail_records(
    support_trail_records: Sequence[Mapping[str, Any]],
    *,
    include_endpoint_maps: bool = True,
) -> List[Dict[str, Any]]:
    """Compress support trails into one global membership row per fact.

    The compressed row preserves the materialized ``fact_to_support_trail_ids``
    semantics used by the canonical interface: a fact owns every support trail
    where it appears anywhere in the fact sequence, not only trails that start
    from that fact.
    """

    membership: Dict[str, Dict[str, set]] = defaultdict(_new_support_trail_membership_bucket)
    for record in support_trail_records:
        _add_support_trail_record_to_membership(
            membership,
            record,
            include_endpoint_maps=include_endpoint_maps,
        )
    return _finalize_support_trail_membership_records(membership)


def build_bridge_support_trail_membership_record_ids_from_support_trail_records(
    support_trail_records: Sequence[Mapping[str, Any]],
) -> List[str]:
    return [
        str(record["hash_id"])
        for record in build_bridge_support_trail_membership_records_from_support_trail_records(
            support_trail_records,
            include_endpoint_maps=False,
        )
    ]


def build_bridge_support_trail_membership_records_from_bridge_path_records(
    bridge_path_records: Sequence[Mapping[str, Any]],
    *,
    min_depth: int = 2,
    max_depth: int = 2,
    max_neighbors_per_fact: int = 4,
    max_trails_per_start_fact: int = 8,
    include_endpoint_maps: bool = True,
) -> List[Dict[str, Any]]:
    """Build compressed support-trail membership rows directly from BridgePath records."""

    if not bridge_path_records:
        return []
    min_depth = max(int(min_depth), 1)
    max_depth = max(int(max_depth), min_depth)
    if max_trails_per_start_fact <= 0:
        return []

    bridge_path_records_by_id = {
        _bridge_path_id(record): record
        for record in bridge_path_records
        if _bridge_path_id(record)
    }
    adjacency = _build_bridge_path_adjacency(
        bridge_path_records,
        max_neighbors_per_fact=int(max_neighbors_per_fact),
    )

    membership: Dict[str, Dict[str, set]] = defaultdict(_new_support_trail_membership_bucket)
    emitted_trail_ids: set[str] = set()
    for start_fact_id in sorted(adjacency.keys()):
        queue = [([start_fact_id], [])]
        emitted_for_start = 0
        while queue and emitted_for_start < max_trails_per_start_fact:
            fact_sequence, path_sequence = queue.pop(0)
            current_fact_id = fact_sequence[-1]
            current_depth = len(path_sequence)
            if current_depth >= max_depth:
                continue
            for edge in adjacency.get(current_fact_id, []):
                neighbor_fact_id = _normalize_id(edge["neighbor_fact_id"])
                bridge_path_id = _normalize_id(edge["bridge_path_id"])
                if not neighbor_fact_id or neighbor_fact_id in fact_sequence or not bridge_path_id:
                    continue
                next_fact_sequence = list(fact_sequence) + [neighbor_fact_id]
                next_path_sequence = list(path_sequence) + [bridge_path_id]
                next_depth = len(next_path_sequence)
                if next_depth >= min_depth:
                    record = _build_bridge_support_trail_record(
                        fact_id_sequence=next_fact_sequence,
                        bridge_path_id_sequence=next_path_sequence,
                        bridge_path_records_by_id=bridge_path_records_by_id,
                    )
                    trail_id = str(record["hash_id"])
                    if trail_id not in emitted_trail_ids:
                        emitted_trail_ids.add(trail_id)
                        _add_support_trail_record_to_membership(
                            membership,
                            record,
                            include_endpoint_maps=include_endpoint_maps,
                        )
                        emitted_for_start += 1
                        if emitted_for_start >= max_trails_per_start_fact:
                            break
                if next_depth < max_depth:
                    queue.append((next_fact_sequence, next_path_sequence))

    return _finalize_support_trail_membership_records(membership)


def build_reserved_bridge_support_trail_membership_records_from_bridge_path_records(
    bridge_path_records: Sequence[Mapping[str, Any]],
    *,
    min_depth: int = 2,
    max_depth: int = 2,
    candidate_max_neighbors_per_fact: int = 5,
    max_trails_per_start_fact: int = 8,
    max_candidate_trails_per_start_fact: int = 32,
    reserved_prefix_trails: int = 7,
    include_endpoint_maps: bool = True,
) -> List[Dict[str, Any]]:
    """Build membership rows from the additive reserved-bridge trail policy."""

    support_trail_records = build_reserved_bridge_support_trail_records_from_bridge_path_records(
        bridge_path_records,
        min_depth=min_depth,
        max_depth=max_depth,
        candidate_max_neighbors_per_fact=candidate_max_neighbors_per_fact,
        max_trails_per_start_fact=max_trails_per_start_fact,
        max_candidate_trails_per_start_fact=max_candidate_trails_per_start_fact,
        reserved_prefix_trails=reserved_prefix_trails,
    )
    return build_bridge_support_trail_membership_records_from_support_trail_records(
        support_trail_records,
        include_endpoint_maps=include_endpoint_maps,
    )


def build_reserved_bridge_support_trail_membership_record_ids_from_bridge_path_records(
    bridge_path_records: Sequence[Mapping[str, Any]],
    **kwargs,
) -> List[str]:
    record_kwargs = dict(kwargs)
    record_kwargs["include_endpoint_maps"] = False
    return [
        str(record["hash_id"])
        for record in build_reserved_bridge_support_trail_membership_records_from_bridge_path_records(
            bridge_path_records,
            **record_kwargs,
        )
    ]


def build_bridge_support_trail_membership_record_ids_from_bridge_path_records(
    bridge_path_records: Sequence[Mapping[str, Any]],
    **kwargs,
) -> List[str]:
    record_kwargs = dict(kwargs)
    record_kwargs["include_endpoint_maps"] = False
    return [
        str(record["hash_id"])
        for record in build_bridge_support_trail_membership_records_from_bridge_path_records(
            bridge_path_records,
            **record_kwargs,
        )
    ]


def _new_support_endpoint_bucket() -> Dict[str, set]:
    return {
        "support_trail_ids": set(),
        "support_fact_ids": set(),
        "support_passage_ids": set(),
        "endpoint_passage_ids": set(),
        "bridge_path_ids": set(),
        "support_relation_families": set(),
        "endpoint_relation_families": set(),
        "owner_positions": set(),
        "endpoint_roles": set(),
        "transition_types": set(),
        "hop_counts": set(),
    }


def _add_support_trail_record_to_endpoint_index(
    endpoint_index: Dict[Tuple[str, str], Dict[str, set]],
    record: Mapping[str, Any],
) -> None:
    trail_id = _support_trail_id(record)
    if not trail_id:
        return
    fact_ids = _support_trail_fact_ids(record)
    if not fact_ids:
        return
    start_fact_id = _normalize_id(record.get("start_fact_id") or fact_ids[0])
    end_fact_id = _normalize_id(record.get("end_fact_id") or fact_ids[-1])
    endpoints = [
        ("start", start_fact_id, _normalize_id_list(record.get("start_source_passage_ids", []))),
        ("end", end_fact_id, _normalize_id_list(record.get("end_source_passage_ids", []))),
    ]
    support_passage_ids = _normalize_id_list(record.get("support_passage_ids", []))
    bridge_path_ids = _normalize_id_list(record.get("bridge_path_id_sequence", []))
    relation_families = _normalize_id_list(record.get("relation_family_sequence", []))
    transition_types = _normalize_id_list(record.get("transition_types", []))
    hop_count = int(record.get("hop_count", len(bridge_path_ids)))

    for owner_position, owner_fact_id in enumerate(fact_ids):
        owner_id = _normalize_id(owner_fact_id)
        if not owner_id:
            continue
        for endpoint_role, endpoint_fact_id, endpoint_passage_ids in endpoints:
            if not endpoint_fact_id or endpoint_fact_id == owner_id:
                continue
            bucket = endpoint_index[(owner_id, endpoint_fact_id)]
            bucket["support_trail_ids"].add(trail_id)
            bucket["support_fact_ids"].update(fact_ids)
            bucket["support_passage_ids"].update(support_passage_ids)
            bucket["endpoint_passage_ids"].update(endpoint_passage_ids)
            bucket["bridge_path_ids"].update(bridge_path_ids)
            bucket["support_relation_families"].update(relation_families)
            endpoint_relation_family = _support_trail_relation_family_for_fact(record, endpoint_fact_id)
            if endpoint_relation_family:
                bucket["endpoint_relation_families"].add(endpoint_relation_family)
            bucket["owner_positions"].add(str(owner_position))
            bucket["endpoint_roles"].add(endpoint_role)
            bucket["transition_types"].update(transition_types)
            bucket["hop_counts"].add(str(hop_count))


def _finalize_support_endpoint_records(
    endpoint_index: Mapping[Tuple[str, str], Mapping[str, set]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for owner_fact_id, endpoint_fact_id in sorted(endpoint_index.keys()):
        bucket = endpoint_index[(owner_fact_id, endpoint_fact_id)]
        support_trail_ids = sorted(bucket["support_trail_ids"])
        hop_counts = sorted(int(value) for value in bucket["hop_counts"] if str(value).strip())
        endpoint_id = compute_mdhash_id(
            json.dumps(
                {
                    "owner_fact_id": owner_fact_id,
                    "endpoint_fact_id": endpoint_fact_id,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
            prefix="bridge-support-endpoint-",
        )
        records.append(
            {
                "hash_id": endpoint_id,
                "endpoint_id": endpoint_id,
                "record_version": BRIDGE_SUPPORT_ENDPOINT_RECORD_VERSION,
                "owner_fact_id": owner_fact_id,
                "endpoint_fact_id": endpoint_fact_id,
                "support_trail_ids": support_trail_ids,
                "support_trail_count": len(support_trail_ids),
                "support_fact_ids": sorted(bucket["support_fact_ids"]),
                "support_passage_ids": sorted(bucket["support_passage_ids"]),
                "endpoint_passage_ids": sorted(bucket["endpoint_passage_ids"]),
                "bridge_path_ids": sorted(bucket["bridge_path_ids"]),
                "support_relation_families": sorted(bucket["support_relation_families"]),
                "endpoint_relation_families": sorted(bucket["endpoint_relation_families"]),
                "min_hop_count": min(hop_counts) if hop_counts else 0,
                "hop_counts": hop_counts,
                "owner_positions": sorted(int(value) for value in bucket["owner_positions"] if str(value).strip()),
                "endpoint_roles": sorted(bucket["endpoint_roles"]),
                "transition_types": sorted(bucket["transition_types"]),
            }
        )
    return records


def build_bridge_support_endpoint_records_from_support_trail_records(
    support_trail_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Build owner-fact to endpoint-fact support objects from support trails."""

    endpoint_index: Dict[Tuple[str, str], Dict[str, set]] = defaultdict(_new_support_endpoint_bucket)
    for record in support_trail_records:
        _add_support_trail_record_to_endpoint_index(endpoint_index, record)
    return _finalize_support_endpoint_records(endpoint_index)


def build_bridge_support_endpoint_record_ids_from_support_trail_records(
    support_trail_records: Sequence[Mapping[str, Any]],
) -> List[str]:
    return [
        str(record["hash_id"])
        for record in build_bridge_support_endpoint_records_from_support_trail_records(
            support_trail_records
        )
    ]


def build_bridge_support_endpoint_records_from_bridge_path_records(
    bridge_path_records: Sequence[Mapping[str, Any]],
    *,
    min_depth: int = 2,
    max_depth: int = 2,
    max_neighbors_per_fact: int = 4,
    max_trails_per_start_fact: int = 8,
) -> List[Dict[str, Any]]:
    """Build compressed endpoint-credit rows directly from BridgePath records."""

    if not bridge_path_records:
        return []
    min_depth = max(int(min_depth), 1)
    max_depth = max(int(max_depth), min_depth)
    if max_trails_per_start_fact <= 0:
        return []

    bridge_path_records_by_id = {
        _bridge_path_id(record): record
        for record in bridge_path_records
        if _bridge_path_id(record)
    }
    adjacency = _build_bridge_path_adjacency(
        bridge_path_records,
        max_neighbors_per_fact=int(max_neighbors_per_fact),
    )

    endpoint_index: Dict[Tuple[str, str], Dict[str, set]] = defaultdict(_new_support_endpoint_bucket)
    emitted_trail_ids: set[str] = set()
    for start_fact_id in sorted(adjacency.keys()):
        queue = [([start_fact_id], [])]
        emitted_for_start = 0
        while queue and emitted_for_start < max_trails_per_start_fact:
            fact_sequence, path_sequence = queue.pop(0)
            current_fact_id = fact_sequence[-1]
            current_depth = len(path_sequence)
            if current_depth >= max_depth:
                continue
            for edge in adjacency.get(current_fact_id, []):
                neighbor_fact_id = _normalize_id(edge["neighbor_fact_id"])
                bridge_path_id = _normalize_id(edge["bridge_path_id"])
                if not neighbor_fact_id or neighbor_fact_id in fact_sequence or not bridge_path_id:
                    continue
                next_fact_sequence = list(fact_sequence) + [neighbor_fact_id]
                next_path_sequence = list(path_sequence) + [bridge_path_id]
                next_depth = len(next_path_sequence)
                if next_depth >= min_depth:
                    record = _build_bridge_support_trail_record(
                        fact_id_sequence=next_fact_sequence,
                        bridge_path_id_sequence=next_path_sequence,
                        bridge_path_records_by_id=bridge_path_records_by_id,
                    )
                    trail_id = str(record["hash_id"])
                    if trail_id not in emitted_trail_ids:
                        emitted_trail_ids.add(trail_id)
                        _add_support_trail_record_to_endpoint_index(endpoint_index, record)
                        emitted_for_start += 1
                        if emitted_for_start >= max_trails_per_start_fact:
                            break
                if next_depth < max_depth:
                    queue.append((next_fact_sequence, next_path_sequence))

    return _finalize_support_endpoint_records(endpoint_index)


def build_bridge_support_endpoint_record_ids_from_bridge_path_records(
    bridge_path_records: Sequence[Mapping[str, Any]],
    **kwargs,
) -> List[str]:
    return [
        str(record["hash_id"])
        for record in build_bridge_support_endpoint_records_from_bridge_path_records(
            bridge_path_records,
            **kwargs,
        )
    ]


def build_reserved_bridge_support_endpoint_records_from_bridge_path_records(
    bridge_path_records: Sequence[Mapping[str, Any]],
    *,
    min_depth: int = 2,
    max_depth: int = 2,
    candidate_max_neighbors_per_fact: int = 5,
    max_trails_per_start_fact: int = 8,
    max_candidate_trails_per_start_fact: int = 32,
    reserved_prefix_trails: int = 7,
) -> List[Dict[str, Any]]:
    """Build endpoint-credit rows from the additive reserved-bridge trail policy."""

    support_trail_records = build_reserved_bridge_support_trail_records_from_bridge_path_records(
        bridge_path_records,
        min_depth=min_depth,
        max_depth=max_depth,
        candidate_max_neighbors_per_fact=candidate_max_neighbors_per_fact,
        max_trails_per_start_fact=max_trails_per_start_fact,
        max_candidate_trails_per_start_fact=max_candidate_trails_per_start_fact,
        reserved_prefix_trails=reserved_prefix_trails,
    )
    return build_bridge_support_endpoint_records_from_support_trail_records(support_trail_records)


def build_reserved_bridge_support_endpoint_record_ids_from_bridge_path_records(
    bridge_path_records: Sequence[Mapping[str, Any]],
    **kwargs,
) -> List[str]:
    return [
        str(record["hash_id"])
        for record in build_reserved_bridge_support_endpoint_records_from_bridge_path_records(
            bridge_path_records,
            **kwargs,
        )
    ]
