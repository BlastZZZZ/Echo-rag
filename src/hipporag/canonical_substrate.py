import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import igraph as ig
import pandas as pd

from .utils.misc_utils import compute_mdhash_id, normalize_graph_text
from .support_signature_memory import (
    build_slot_records_from_hyperedge_records,
    build_support_signature_sidecar_from_hyperedge_records,
    load_support_signature_sidecar,
    save_support_signature_sidecar,
)
from .proposition_memory import build_proposition_fact_embedding_text

CANONICAL_SUBSTRATE_MANIFEST_FILENAME = "canonical_substrate_manifest.json"
CANONICAL_SUBSTRATE_SCHEMA_VERSION = 1
HYPERHIPPO_SUBSTRATE_FAMILY = "hyperhippo_memory_compat"


def get_hyperhippo_artifact_paths(working_dir: str) -> Dict[str, str]:
    return {
        "graph_pickle": os.path.join(working_dir, "graph_hyperhippo.pickle"),
        "entity_embeddings": os.path.join(working_dir, "entity_embeddings", "vdb_entity.parquet"),
        "fact_embeddings": os.path.join(working_dir, "fact_embeddings", "vdb_fact.parquet"),
        "entity_records": os.path.join(working_dir, "entity_records", "records_entity.parquet"),
        "hyperedge_records": os.path.join(working_dir, "hyperedge_records", "records_hyperedge.parquet"),
        "passage_records": os.path.join(working_dir, "passage_records", "records_passage.parquet"),
        "slot_records": os.path.join(working_dir, "slot_records", "records_slot.parquet"),
        "support_signature_sidecar": os.path.join(working_dir, "support_signature_sidecar.json"),
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _compute_sha256(path: str) -> Optional[str]:
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        return None

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_record_payloads(path: str, key_field: str = "hash_id") -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}

    df = pd.read_parquet(path)
    records: Dict[str, Dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        payload = json.loads(row["payload_json"])
        if key_field not in payload:
            raise KeyError(f"Payload in `{path}` missing key field `{key_field}`: {payload}")
        records[str(payload[key_field])] = payload
    return records


def _save_record_payloads(path: str, records: Sequence[Mapping[str, Any]], key_field: str = "hash_id") -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    rows = [
        {
            key_field: str(record[key_field]),
            "payload_json": json.dumps(dict(record), ensure_ascii=True, sort_keys=True),
        }
        for record in records
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _load_embedding_rows(path: str) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}

    df = pd.read_parquet(path)
    rows: Dict[str, Dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        hash_id = str(row["hash_id"])
        rows[hash_id] = dict(row)
    return rows


def _summarize_graph(path: str) -> Dict[str, Any]:
    summary = {
        "present": False,
        "node_count": 0,
        "edge_count": 0,
        "node_type_counts": {},
        "edge_type_counts": {},
    }
    if not path or not os.path.exists(path):
        return summary

    graph = ig.Graph.Read_Pickle(path)
    summary["present"] = True
    summary["node_count"] = int(graph.vcount())
    summary["edge_count"] = int(graph.ecount())

    if "node_type" in graph.vs.attributes():
        summary["node_type_counts"] = {
            str(node_type): int(count)
            for node_type, count in Counter(str(value) for value in graph.vs["node_type"]).items()
        }
    if "edge_type" in graph.es.attributes():
        summary["edge_type_counts"] = {
            str(edge_type): int(count)
            for edge_type, count in Counter(str(value) for value in graph.es["edge_type"]).items()
        }
    return summary


def _sample_mapping_items(mapping: Mapping[str, Sequence[str]], limit: int = 3) -> List[Tuple[str, List[str]]]:
    sampled: List[Tuple[str, List[str]]] = []
    for key in sorted(mapping.keys())[:limit]:
        sampled.append((str(key), [str(value) for value in list(mapping[key])[:5]]))
    return sampled


def _collect_sidecar_referenced_passages(sidecar: Mapping[str, Any]) -> Dict[str, List[str]]:
    referenced = {
        "pair_to_passages": [],
        "subj_endpoint_to_passages": [],
        "obj_endpoint_to_passages": [],
        "slot_to_passages": [],
        "subject_slot_to_passages": [],
        "object_slot_to_passages": [],
        "participant_slot_to_passages": [],
        "relation_to_passages": [],
        "subject_entity_to_passages": [],
        "object_entity_to_passages": [],
        "passage_inventories": [],
    }
    for field in [
        "pair_to_passages",
        "subj_endpoint_to_passages",
        "obj_endpoint_to_passages",
        "slot_to_passages",
        "subject_slot_to_passages",
        "object_slot_to_passages",
        "participant_slot_to_passages",
        "relation_to_passages",
        "subject_entity_to_passages",
        "object_entity_to_passages",
    ]:
        refs: List[str] = []
        for passage_refs in dict(sidecar.get(field, {})).values():
            refs.extend(str(value) for value in passage_refs)
        referenced[field] = sorted(set(refs))
    referenced["passage_inventories"] = sorted(str(value) for value in dict(sidecar.get("passage_inventories", {})).keys())
    return referenced


def _build_expected_entity_payloads(
    entity_records: Mapping[str, Dict[str, Any]],
    hyperedge_records: Mapping[str, Dict[str, Any]],
    passage_records: Mapping[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    source_ids_by_entity: Dict[str, set] = {}
    names_by_entity: Dict[str, set] = {}
    last_seen_by_entity: Dict[str, int] = {}

    def ensure_entity(entity_id: str) -> None:
        normalized_entity_id = str(entity_id)
        source_ids_by_entity.setdefault(normalized_entity_id, set())
        names_by_entity.setdefault(normalized_entity_id, set())
        last_seen_by_entity.setdefault(normalized_entity_id, 0)

    for entity_id, record in entity_records.items():
        ensure_entity(entity_id)
        canonical_name = " ".join(str(record.get("canonical_name", "")).split()).strip()
        if canonical_name:
            names_by_entity[str(entity_id)].add(canonical_name)
        for alias in list(record.get("aliases", [])):
            normalized_alias = " ".join(str(alias).split()).strip()
            if normalized_alias:
                names_by_entity[str(entity_id)].add(normalized_alias)
        for source_id in list(record.get("source_ids", [])):
            source_ids_by_entity[str(entity_id)].add(str(source_id))
        last_seen_by_entity[str(entity_id)] = max(
            int(record.get("last_seen", 0)),
            last_seen_by_entity[str(entity_id)],
        )

    for passage_id, record in passage_records.items():
        normalized_passage_id = str(passage_id)
        passage_last_seen = int(record.get("last_seen", 0))
        for entity_id in list(record.get("entity_ids", [])):
            ensure_entity(str(entity_id))
            source_ids_by_entity[str(entity_id)].add(normalized_passage_id)
            last_seen_by_entity[str(entity_id)] = max(last_seen_by_entity[str(entity_id)], passage_last_seen)

    for hyperedge_id, record in hyperedge_records.items():
        participant_ids = [str(value) for value in list(record.get("participant_ids", []))]
        participant_texts = [
            " ".join(str(value).split()).strip()
            for value in list(record.get("participant_texts", []))
        ]
        source_ids = [str(value) for value in list(record.get("source_ids", []))]
        hyperedge_last_seen = int(record.get("last_seen", 0))

        for idx, entity_id in enumerate(participant_ids):
            ensure_entity(entity_id)
            source_ids_by_entity[entity_id].update(source_ids)
            last_seen_by_entity[entity_id] = max(last_seen_by_entity[entity_id], hyperedge_last_seen)
            if idx < len(participant_texts) and participant_texts[idx]:
                names_by_entity[entity_id].add(participant_texts[idx])

    expected_payloads: Dict[str, Dict[str, Any]] = {}
    for entity_id, source_ids in source_ids_by_entity.items():
        candidate_names = sorted(name for name in names_by_entity.get(entity_id, set()) if name)
        hash_consistent_names = [
            name
            for name in candidate_names
            if compute_mdhash_id(normalize_graph_text(name), prefix="entity-") == entity_id
        ]
        chosen_name = ""
        if hash_consistent_names:
            chosen_name = hash_consistent_names[0]
        elif candidate_names:
            chosen_name = candidate_names[0]
        elif entity_id in entity_records:
            chosen_name = str(entity_records[entity_id].get("canonical_name", "")).strip()
        if not chosen_name:
            chosen_name = entity_id

        aliases = sorted(set(candidate_names + [chosen_name]))
        expected_payloads[entity_id] = {
            "hash_id": entity_id,
            "canonical_name": chosen_name,
            "aliases": aliases,
            "entity_type": str(entity_records.get(entity_id, {}).get("entity_type", "generic") or "generic"),
            "source_ids": sorted(source_ids),
            "support_count": len(source_ids),
            "last_seen": int(last_seen_by_entity.get(entity_id, 0)),
        }
    return expected_payloads


def find_missing_entity_payloads_from_memory(
    entity_records: Mapping[str, Dict[str, Any]],
    hyperedge_records: Mapping[str, Dict[str, Any]],
    passage_records: Mapping[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    expected_payloads = _build_expected_entity_payloads(
        entity_records=entity_records,
        hyperedge_records=hyperedge_records,
        passage_records=passage_records,
    )
    return {
        entity_id: payload
        for entity_id, payload in expected_payloads.items()
        if entity_id not in entity_records
    }


def find_missing_entity_embedding_texts_from_memory(
    entity_records: Mapping[str, Dict[str, Any]],
    hyperedge_records: Mapping[str, Dict[str, Any]],
    passage_records: Mapping[str, Dict[str, Any]],
    entity_embedding_ids: Sequence[str],
) -> Dict[str, str]:
    expected_payloads = _build_expected_entity_payloads(
        entity_records=entity_records,
        hyperedge_records=hyperedge_records,
        passage_records=passage_records,
    )
    embedding_id_set = {str(value) for value in entity_embedding_ids}
    missing_texts: Dict[str, str] = {}
    for entity_id, payload in expected_payloads.items():
        if entity_id in embedding_id_set:
            continue
        canonical_name = " ".join(str(payload.get("canonical_name", "")).split()).strip()
        if not canonical_name:
            continue
        if compute_mdhash_id(normalize_graph_text(canonical_name), prefix="entity-") != entity_id:
            continue
        missing_texts[entity_id] = canonical_name
    return missing_texts


def _extract_record_role_text(record: Mapping[str, Any], role_name: str) -> str:
    participant_ids = [str(value) for value in list(record.get("participant_ids", []))]
    participant_texts = [" ".join(str(value).split()).strip() for value in list(record.get("participant_texts", []))]
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


def build_canonical_fact_embedding_text_from_record(record: Mapping[str, Any]) -> str:
    relation_type = " ".join(str(record.get("relation_type", "")).split()).strip()
    subject_text = _extract_record_role_text(record, "subject")
    object_text = _extract_record_role_text(record, "object")
    participant_texts = [
        " ".join(str(value).split()).strip()
        for value in list(record.get("participant_texts", []))
        if " ".join(str(value).split()).strip()
    ]
    if subject_text and object_text:
        fact_participant_texts = [subject_text, object_text]
    else:
        fact_participant_texts = participant_texts[:2]
    normalized_text = " ".join(str(record.get("normalized_text", "")).split()).strip()
    summary_text = " ".join(str(record.get("summary_text", "")).split()).strip()
    return build_proposition_fact_embedding_text(
        relation_type=relation_type,
        participant_texts=fact_participant_texts,
        normalized_text=normalized_text,
        summary_text=summary_text,
    )


def _normalize_fact_embedding_lookup_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _build_normalized_fact_text_lookup(
    fact_embedding_rows: Mapping[str, Mapping[str, Any]],
) -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = {}
    for hash_id, row in fact_embedding_rows.items():
        normalized_content = _normalize_fact_embedding_lookup_text(row.get("content", ""))
        normalized_hash_id = str(hash_id or "").strip()
        if not normalized_content or not normalized_hash_id:
            continue
        lookup.setdefault(normalized_content, []).append(normalized_hash_id)
    return lookup


def find_missing_fact_id_updates_from_memory(
    hyperedge_records: Mapping[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    updated_records: Dict[str, Dict[str, Any]] = {}
    for hyperedge_id, record in hyperedge_records.items():
        fact_embedding_hash_id = str(record.get("fact_embedding_hash_id", "")).strip()
        if fact_embedding_hash_id:
            continue
        fact_text = build_canonical_fact_embedding_text_from_record(record)
        if not fact_text:
            continue
        updated_record = dict(record)
        updated_record["fact_embedding_hash_id"] = compute_mdhash_id(fact_text, prefix="fact-")
        updated_records[str(hyperedge_id)] = updated_record
    return updated_records


def find_missing_fact_id_updates_from_memory_with_fact_embeddings(
    hyperedge_records: Mapping[str, Dict[str, Any]],
    fact_embedding_rows: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Repair missing fact IDs while preserving an existing fact embedding ID space.

    Older HyperHippoRAG records may omit `fact_embedding_hash_id` after storing
    normalized participant text. Re-hashing the normalized record can drift from
    the already materialized `vdb_fact` IDs when the original fact text contained
    whitespace artifacts. Prefer a unique normalized-content match in the existing
    fact embedding store, then fall back to the canonical hash for genuinely new
    facts.
    """

    fact_embedding_rows = dict(fact_embedding_rows or {})
    fact_embedding_id_set = {str(value) for value in fact_embedding_rows.keys()}
    normalized_text_lookup = _build_normalized_fact_text_lookup(fact_embedding_rows)
    updated_records: Dict[str, Dict[str, Any]] = {}
    for hyperedge_id, record in hyperedge_records.items():
        fact_embedding_hash_id = str(record.get("fact_embedding_hash_id", "")).strip()
        if fact_embedding_hash_id:
            continue
        fact_text = build_canonical_fact_embedding_text_from_record(record)
        if not fact_text:
            continue
        canonical_fact_hash_id = compute_mdhash_id(fact_text, prefix="fact-")
        resolved_fact_hash_id = canonical_fact_hash_id
        if canonical_fact_hash_id not in fact_embedding_id_set:
            normalized_fact_text = _normalize_fact_embedding_lookup_text(fact_text)
            matched_hash_ids = sorted(set(normalized_text_lookup.get(normalized_fact_text, [])))
            if len(matched_hash_ids) == 1:
                resolved_fact_hash_id = matched_hash_ids[0]
        updated_record = dict(record)
        updated_record["fact_embedding_hash_id"] = resolved_fact_hash_id
        updated_records[str(hyperedge_id)] = updated_record
    return updated_records


def find_missing_fact_embedding_texts_from_memory(
    hyperedge_records: Mapping[str, Dict[str, Any]],
    fact_embedding_ids: Sequence[str],
) -> Dict[str, str]:
    fact_embedding_id_set = {str(value) for value in fact_embedding_ids}
    missing_texts: Dict[str, str] = {}
    for record in hyperedge_records.values():
        fact_embedding_hash_id = str(record.get("fact_embedding_hash_id", "")).strip()
        if not fact_embedding_hash_id or fact_embedding_hash_id in fact_embedding_id_set:
            continue
        fact_text = build_canonical_fact_embedding_text_from_record(record)
        if not fact_text:
            continue
        if compute_mdhash_id(fact_text, prefix="fact-") != fact_embedding_hash_id:
            continue
        missing_texts[fact_embedding_hash_id] = fact_text
    return missing_texts


def _collect_required_fact_embedding_ids(
    hyperedge_records: Mapping[str, Dict[str, Any]],
) -> List[str]:
    required_ids = []
    seen = set()
    for record in hyperedge_records.values():
        fact_embedding_hash_id = str(record.get("fact_embedding_hash_id", "")).strip()
        if not fact_embedding_hash_id or fact_embedding_hash_id in seen:
            continue
        required_ids.append(fact_embedding_hash_id)
        seen.add(fact_embedding_hash_id)
    return required_ids


def find_missing_hyperhippo_entity_payloads(working_dir: str) -> Dict[str, Dict[str, Any]]:
    artifact_paths = get_hyperhippo_artifact_paths(working_dir)
    entity_records = _load_record_payloads(artifact_paths["entity_records"])
    hyperedge_records = _load_record_payloads(artifact_paths["hyperedge_records"])
    passage_records = _load_record_payloads(artifact_paths["passage_records"])
    return find_missing_entity_payloads_from_memory(
        entity_records=entity_records,
        hyperedge_records=hyperedge_records,
        passage_records=passage_records,
    )


def find_missing_hyperhippo_entity_embedding_texts(working_dir: str) -> Dict[str, str]:
    artifact_paths = get_hyperhippo_artifact_paths(working_dir)
    entity_records = _load_record_payloads(artifact_paths["entity_records"])
    hyperedge_records = _load_record_payloads(artifact_paths["hyperedge_records"])
    passage_records = _load_record_payloads(artifact_paths["passage_records"])
    entity_embedding_rows = _load_embedding_rows(artifact_paths["entity_embeddings"])
    return find_missing_entity_embedding_texts_from_memory(
        entity_records=entity_records,
        hyperedge_records=hyperedge_records,
        passage_records=passage_records,
        entity_embedding_ids=list(entity_embedding_rows.keys()),
    )


def find_missing_hyperhippo_fact_embedding_texts(working_dir: str) -> Dict[str, str]:
    artifact_paths = get_hyperhippo_artifact_paths(working_dir)
    hyperedge_records = _load_record_payloads(artifact_paths["hyperedge_records"])
    fact_embedding_rows = _load_embedding_rows(artifact_paths["fact_embeddings"])
    return find_missing_fact_embedding_texts_from_memory(
        hyperedge_records=hyperedge_records,
        fact_embedding_ids=list(fact_embedding_rows.keys()),
    )


def find_missing_hyperhippo_fact_id_updates(working_dir: str) -> Dict[str, Dict[str, Any]]:
    artifact_paths = get_hyperhippo_artifact_paths(working_dir)
    hyperedge_records = _load_record_payloads(artifact_paths["hyperedge_records"])
    return find_missing_fact_id_updates_from_memory(hyperedge_records)


def repair_hyperhippo_entity_records(
    working_dir: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    artifact_paths = get_hyperhippo_artifact_paths(working_dir)
    target_path = output_path or artifact_paths["entity_records"]

    entity_records = _load_record_payloads(artifact_paths["entity_records"])
    hyperedge_records = _load_record_payloads(artifact_paths["hyperedge_records"])
    passage_records = _load_record_payloads(artifact_paths["passage_records"])
    expected_payloads = _build_expected_entity_payloads(
        entity_records=entity_records,
        hyperedge_records=hyperedge_records,
        passage_records=passage_records,
    )
    missing_payloads = {
        entity_id: payload
        for entity_id, payload in find_missing_entity_payloads_from_memory(
            entity_records=entity_records,
            hyperedge_records=hyperedge_records,
            passage_records=passage_records,
        ).items()
    }

    if target_path == artifact_paths["entity_records"]:
        final_payloads = dict(entity_records)
        final_payloads.update(missing_payloads)
    else:
        final_payloads = dict(expected_payloads)

    _save_record_payloads(target_path, final_payloads.values())
    return {
        "target_path": target_path,
        "missing_count": len(missing_payloads),
        "final_count": len(final_payloads),
        "sample_missing": [
            {
                "entity_id": entity_id,
                "canonical_name": payload.get("canonical_name", ""),
                "source_count": len(list(payload.get("source_ids", []))),
            }
            for entity_id, payload in list(sorted(missing_payloads.items()))[:10]
        ],
    }


def repair_missing_entity_embeddings_with_store(
    entity_records: Mapping[str, Dict[str, Any]],
    hyperedge_records: Mapping[str, Dict[str, Any]],
    passage_records: Mapping[str, Dict[str, Any]],
    entity_embedding_store,
) -> Dict[str, Any]:
    embedding_ids = list(getattr(entity_embedding_store, "hash_id_to_row", {}).keys())
    missing_texts = find_missing_entity_embedding_texts_from_memory(
        entity_records=entity_records,
        hyperedge_records=hyperedge_records,
        passage_records=passage_records,
        entity_embedding_ids=embedding_ids,
    )
    if missing_texts:
        entity_embedding_store.insert_strings(list(missing_texts.values()))
    updated_embedding_ids = list(getattr(entity_embedding_store, "hash_id_to_row", {}).keys())
    remaining_missing = find_missing_entity_embedding_texts_from_memory(
        entity_records=entity_records,
        hyperedge_records=hyperedge_records,
        passage_records=passage_records,
        entity_embedding_ids=updated_embedding_ids,
    )
    return {
        "missing_count": len(missing_texts),
        "remaining_missing_count": len(remaining_missing),
        "sample_missing": [
            {"entity_id": entity_id, "canonical_name": text}
            for entity_id, text in list(sorted(missing_texts.items()))[:10]
        ],
    }


def repair_missing_fact_ids_with_record_store(
    hyperedge_records: Mapping[str, Dict[str, Any]],
    hyperedge_record_store,
) -> Dict[str, Any]:
    updated_records = find_missing_fact_id_updates_from_memory(hyperedge_records)
    if updated_records:
        hyperedge_record_store.upsert(updated_records.values())
    remaining_missing = find_missing_fact_id_updates_from_memory(hyperedge_record_store.get_all())
    return {
        "missing_count": len(updated_records),
        "remaining_missing_count": len(remaining_missing),
        "sample_missing": [
            {
                "hyperedge_id": hyperedge_id,
                "fact_embedding_hash_id": record.get("fact_embedding_hash_id", ""),
            }
            for hyperedge_id, record in list(sorted(updated_records.items()))[:10]
        ],
    }


def repair_missing_fact_ids_with_record_and_fact_store(
    hyperedge_records: Mapping[str, Dict[str, Any]],
    hyperedge_record_store,
    fact_embedding_store,
) -> Dict[str, Any]:
    fact_embedding_rows = getattr(fact_embedding_store, "hash_id_to_row", {})
    updated_records = find_missing_fact_id_updates_from_memory_with_fact_embeddings(
        hyperedge_records=hyperedge_records,
        fact_embedding_rows=fact_embedding_rows,
    )
    if updated_records:
        hyperedge_record_store.upsert(updated_records.values())
    remaining_missing = find_missing_fact_id_updates_from_memory_with_fact_embeddings(
        hyperedge_records=hyperedge_record_store.get_all(),
        fact_embedding_rows=getattr(fact_embedding_store, "hash_id_to_row", {}),
    )
    aligned_count = sum(
        1
        for record in updated_records.values()
        if str(record.get("fact_embedding_hash_id", "")).strip() in fact_embedding_rows
    )
    return {
        "missing_count": len(updated_records),
        "remaining_missing_count": len(remaining_missing),
        "aligned_existing_fact_embedding_count": aligned_count,
        "sample_missing": [
            {
                "hyperedge_id": hyperedge_id,
                "fact_embedding_hash_id": record.get("fact_embedding_hash_id", ""),
            }
            for hyperedge_id, record in list(sorted(updated_records.items()))[:10]
        ],
    }


def repair_missing_fact_embeddings_with_store(
    hyperedge_records: Mapping[str, Dict[str, Any]],
    fact_embedding_store,
) -> Dict[str, Any]:
    embedding_ids = list(getattr(fact_embedding_store, "hash_id_to_row", {}).keys())
    missing_texts = find_missing_fact_embedding_texts_from_memory(
        hyperedge_records=hyperedge_records,
        fact_embedding_ids=embedding_ids,
    )
    if missing_texts:
        fact_embedding_store.insert_strings(list(missing_texts.values()))
    updated_embedding_ids = list(getattr(fact_embedding_store, "hash_id_to_row", {}).keys())
    remaining_missing = find_missing_fact_embedding_texts_from_memory(
        hyperedge_records=hyperedge_records,
        fact_embedding_ids=updated_embedding_ids,
    )
    return {
        "missing_count": len(missing_texts),
        "remaining_missing_count": len(remaining_missing),
        "sample_missing": [
            {"fact_embedding_id": fact_embedding_id, "fact_text": text}
            for fact_embedding_id, text in list(sorted(missing_texts.items()))[:10]
        ],
    }


def _validate_hyperhippo_substrate(
    artifact_paths: Mapping[str, str],
    entity_records: Mapping[str, Dict[str, Any]],
    hyperedge_records: Mapping[str, Dict[str, Any]],
    passage_records: Mapping[str, Dict[str, Any]],
    slot_records: Mapping[str, Dict[str, Any]],
    sidecar: Mapping[str, Any],
    graph_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    for artifact_name in ["graph_pickle", "entity_records", "hyperedge_records", "passage_records"]:
        if not os.path.exists(artifact_paths[artifact_name]):
            errors.append(f"Missing required artifact `{artifact_name}` at `{artifact_paths[artifact_name]}`.")

    has_memory_state = bool(entity_records or hyperedge_records or passage_records)
    if has_memory_state and (hyperedge_records or passage_records) and not os.path.exists(artifact_paths["support_signature_sidecar"]):
        errors.append(
            "Missing required artifact `support_signature_sidecar.json` while passage/hyperedge memory exists."
        )

    if has_memory_state and not os.path.exists(artifact_paths["entity_embeddings"]):
        errors.append(
            f"Missing required artifact `entity_embeddings` at `{artifact_paths['entity_embeddings']}`."
        )
    if has_memory_state and hyperedge_records and not os.path.exists(artifact_paths["fact_embeddings"]):
        errors.append(
            f"Missing required artifact `fact_embeddings` at `{artifact_paths['fact_embeddings']}`."
        )
    if has_memory_state and hyperedge_records and not os.path.exists(artifact_paths["slot_records"]):
        errors.append(
            f"Missing required artifact `slot_records` at `{artifact_paths['slot_records']}`."
        )

    entity_ids = set(str(value) for value in entity_records.keys())
    hyperedge_ids = set(str(value) for value in hyperedge_records.keys())
    passage_ids = set(str(value) for value in passage_records.keys())
    entity_embedding_rows = _load_embedding_rows(artifact_paths["entity_embeddings"])
    entity_embedding_ids = set(entity_embedding_rows.keys())
    fact_embedding_rows = _load_embedding_rows(artifact_paths["fact_embeddings"])
    fact_embedding_ids = set(fact_embedding_rows.keys())

    missing_hyperedges_by_passage: Dict[str, List[str]] = {}
    missing_entities_by_passage: Dict[str, List[str]] = {}
    for passage_id, record in passage_records.items():
        proposition_ids = sorted(
            {
                str(value)
                for value in list(record.get("proposition_ids", []))
                if str(value)
            }
        )
        missing_hyperedges = [value for value in proposition_ids if value not in hyperedge_ids]
        if missing_hyperedges:
            missing_hyperedges_by_passage[str(passage_id)] = missing_hyperedges

        linked_entities = sorted(
            {
                str(value)
                for value in list(record.get("entity_ids", []))
                if str(value)
            }
        )
        missing_entities = [value for value in linked_entities if value not in entity_ids]
        if missing_entities:
            missing_entities_by_passage[str(passage_id)] = missing_entities

    if missing_hyperedges_by_passage:
        errors.append(
            "Passage records reference missing hyperedges: "
            + str(_sample_mapping_items(missing_hyperedges_by_passage))
        )
    if missing_entities_by_passage:
        errors.append(
            "Passage records reference missing entities: "
            + str(_sample_mapping_items(missing_entities_by_passage))
        )

    missing_passages_by_hyperedge: Dict[str, List[str]] = {}
    missing_participants_by_hyperedge: Dict[str, List[str]] = {}
    for hyperedge_id, record in hyperedge_records.items():
        source_ids = sorted(
            {
                str(value)
                for value in list(record.get("source_ids", []))
                if str(value)
            }
        )
        missing_sources = [value for value in source_ids if value not in passage_ids]
        if missing_sources:
            missing_passages_by_hyperedge[str(hyperedge_id)] = missing_sources

        participant_ids = sorted(
            {
                str(value)
                for value in list(record.get("participant_ids", []))
                if str(value)
            }
        )
        missing_participants = [value for value in participant_ids if value not in entity_ids]
        if missing_participants:
            missing_participants_by_hyperedge[str(hyperedge_id)] = missing_participants

    if missing_passages_by_hyperedge:
        errors.append(
            "Hyperedge records reference missing passages: "
            + str(_sample_mapping_items(missing_passages_by_hyperedge))
        )
    if missing_participants_by_hyperedge:
        errors.append(
            "Hyperedge records reference missing entities: "
            + str(_sample_mapping_items(missing_participants_by_hyperedge))
        )

    hyperedges_missing_fact_ids = sorted(
        str(hyperedge_id)
        for hyperedge_id, record in hyperedge_records.items()
        if not str(record.get("fact_embedding_hash_id", "")).strip()
    )
    if hyperedges_missing_fact_ids:
        errors.append(
            "Hyperedge records missing `fact_embedding_hash_id`: "
            + str(hyperedges_missing_fact_ids[:10])
        )

    missing_entity_embedding_texts = find_missing_entity_embedding_texts_from_memory(
        entity_records=entity_records,
        hyperedge_records=hyperedge_records,
        passage_records=passage_records,
        entity_embedding_ids=list(entity_embedding_ids),
    )
    if missing_entity_embedding_texts:
        errors.append(
            "Missing entity embeddings for required entity ids: "
            + str(
                [
                    (entity_id, missing_entity_embedding_texts[entity_id])
                    for entity_id in sorted(missing_entity_embedding_texts.keys())[:10]
                ]
            )
        )

    missing_fact_embedding_texts = find_missing_fact_embedding_texts_from_memory(
        hyperedge_records=hyperedge_records,
        fact_embedding_ids=list(fact_embedding_ids),
    )
    if missing_fact_embedding_texts:
        errors.append(
            "Missing fact embeddings for required fact ids: "
            + str(
                [
                    (fact_id, missing_fact_embedding_texts[fact_id])
                    for fact_id in sorted(missing_fact_embedding_texts.keys())[:10]
                ]
            )
        )

    stale_entity_embedding_ids = sorted(entity_embedding_ids - set(_build_expected_entity_payloads(
        entity_records=entity_records,
        hyperedge_records=hyperedge_records,
        passage_records=passage_records,
    ).keys()))
    if stale_entity_embedding_ids:
        warnings.append(
            "Entity embedding store contains stale ids not referenced by current memory state: "
            + str(stale_entity_embedding_ids[:10])
        )

    required_fact_embedding_ids = set(_collect_required_fact_embedding_ids(hyperedge_records))
    stale_fact_embedding_ids = sorted(fact_embedding_ids - required_fact_embedding_ids)
    if stale_fact_embedding_ids:
        warnings.append(
            "Fact embedding store contains stale ids not referenced by current memory state: "
            + str(stale_fact_embedding_ids[:10])
        )

    missing_entity_payloads = _build_expected_entity_payloads(
        entity_records=entity_records,
        hyperedge_records=hyperedge_records,
        passage_records=passage_records,
    )
    repairable_missing_entities = sorted(
        entity_id
        for entity_id in missing_entity_payloads.keys()
        if entity_id not in entity_ids
    )
    if repairable_missing_entities:
        warnings.append(
            "Missing entity records appear repairable from existing memory state: "
            + str(repairable_missing_entities[:10])
        )

    rebuilt_sidecar = build_support_signature_sidecar_from_hyperedge_records(
        hyperedge_records=hyperedge_records,
        passage_records=passage_records,
    )
    expected_slot_records = {
        str(record["hash_id"]): dict(record)
        for record in build_slot_records_from_hyperedge_records(
            hyperedge_records,
            support_signature_sidecar=rebuilt_sidecar,
        )
    }
    expected_slot_ids = set(expected_slot_records.keys())
    actual_slot_ids = set(str(value) for value in slot_records.keys())
    missing_slot_ids = sorted(expected_slot_ids - actual_slot_ids)
    if missing_slot_ids:
        errors.append("Slot records missing required ids: " + str(missing_slot_ids[:10]))
    stale_slot_ids = sorted(actual_slot_ids - expected_slot_ids)
    if stale_slot_ids:
        warnings.append("Slot record store contains stale ids not referenced by current memory state: " + str(stale_slot_ids[:10]))
    mismatched_slot_payloads = []
    for slot_id in sorted(expected_slot_ids & actual_slot_ids):
        expected_payload = expected_slot_records[slot_id]
        actual_payload = dict(slot_records[slot_id])
        if dict(expected_payload) != actual_payload:
            mismatched_slot_payloads.append(slot_id)
            if len(mismatched_slot_payloads) >= 10:
                break
    if mismatched_slot_payloads:
        errors.append("Slot record payloads do not match canonical rebuild for ids: " + str(mismatched_slot_payloads))

    sidecar_passage_inventories = dict(sidecar.get("passage_inventories", {}))
    if passage_records and not sidecar_passage_inventories:
        errors.append("Support signature sidecar has no `passage_inventories` despite non-empty passage records.")

    if sidecar_passage_inventories:
        missing_inventories = sorted(passage_ids - set(sidecar_passage_inventories.keys()))
        if missing_inventories:
            errors.append(
                "Support signature sidecar is missing passage inventories for: "
                + str(missing_inventories[:10])
            )

        unknown_inventory_passages = sorted(set(sidecar_passage_inventories.keys()) - passage_ids)
        if unknown_inventory_passages:
            warnings.append(
                "Support signature sidecar contains inventories for unknown passages: "
                + str(unknown_inventory_passages[:10])
            )

        mismatched_inventory_props: Dict[str, List[str]] = {}
        for passage_id, record in passage_records.items():
            inventory = sidecar_passage_inventories.get(str(passage_id), {})
            expected_props = sorted(
                {
                    str(value)
                    for value in list(record.get("proposition_ids", []))
                    if str(value)
                }
            )
            actual_props = sorted(
                {
                    str(value)
                    for value in list(inventory.get("proposition_ids", []))
                    if str(value)
                }
            )
            if expected_props != actual_props:
                mismatched_inventory_props[str(passage_id)] = actual_props
        if mismatched_inventory_props:
            errors.append(
                "Support signature sidecar proposition inventories do not match passage records: "
                + str(_sample_mapping_items(mismatched_inventory_props))
            )

    referenced_passages = _collect_sidecar_referenced_passages(sidecar)
    for field_name, refs in referenced_passages.items():
        unknown_refs = sorted(value for value in refs if value not in passage_ids)
        if unknown_refs:
            errors.append(
                f"Support signature sidecar field `{field_name}` references unknown passages: {unknown_refs[:10]}"
            )

    node_type_counts = dict(graph_summary.get("node_type_counts", {}))
    if graph_summary.get("present"):
        expected_entity_count = len(entity_records)
        expected_hyperedge_count = len(hyperedge_records)
        expected_passage_count = len(passage_records)
        if node_type_counts:
            if int(node_type_counts.get("entity", 0)) != expected_entity_count:
                errors.append(
                    "Graph entity node count does not match entity record count: "
                    f"{node_type_counts.get('entity', 0)} vs {expected_entity_count}."
                )
            if int(node_type_counts.get("hyperedge", 0)) != expected_hyperedge_count:
                errors.append(
                    "Graph hyperedge node count does not match hyperedge record count: "
                    f"{node_type_counts.get('hyperedge', 0)} vs {expected_hyperedge_count}."
                )
            if int(node_type_counts.get("passage", 0)) != expected_passage_count:
                errors.append(
                    "Graph passage node count does not match passage record count: "
                    f"{node_type_counts.get('passage', 0)} vs {expected_passage_count}."
                )
        else:
            warnings.append("Graph is present but node_type counts are unavailable.")

        if graph_summary.get("node_count", 0) <= 0 and has_memory_state:
            errors.append("Graph pickle is present but contains zero nodes despite non-empty memory state.")
        if graph_summary.get("edge_count", 0) <= 0 and graph_summary.get("node_count", 0) > 1:
            warnings.append("Graph pickle contains more than one node but zero edges.")

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def audit_hyperhippo_working_dir(
    working_dir: str,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    artifact_paths = get_hyperhippo_artifact_paths(working_dir)
    entity_records = _load_record_payloads(artifact_paths["entity_records"])
    hyperedge_records = _load_record_payloads(artifact_paths["hyperedge_records"])
    passage_records = _load_record_payloads(artifact_paths["passage_records"])
    slot_records = _load_record_payloads(artifact_paths["slot_records"])
    sidecar = load_support_signature_sidecar(artifact_paths["support_signature_sidecar"])
    graph_summary = _summarize_graph(artifact_paths["graph_pickle"])
    validation = _validate_hyperhippo_substrate(
        artifact_paths=artifact_paths,
        entity_records=entity_records,
        hyperedge_records=hyperedge_records,
        passage_records=passage_records,
        slot_records=slot_records,
        sidecar=sidecar,
        graph_summary=graph_summary,
    )
    fact_embedding_rows = _load_embedding_rows(artifact_paths["fact_embeddings"])

    manifest = {
        "manifest_version": CANONICAL_SUBSTRATE_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "substrate_family": HYPERHIPPO_SUBSTRATE_FAMILY,
        "working_dir": os.path.abspath(working_dir),
        "artifacts": {
            name: {
                "path": path,
                "present": bool(path and os.path.exists(path)),
                "sha256": _compute_sha256(path),
            }
            for name, path in artifact_paths.items()
        },
        "counts": {
            "entity_records": len(entity_records),
            "entity_embeddings": len(_load_embedding_rows(artifact_paths["entity_embeddings"])),
            "fact_embeddings": len(fact_embedding_rows),
            "hyperedge_records": len(hyperedge_records),
            "passage_records": len(passage_records),
            "slot_records": len(slot_records),
            "sidecar_passage_inventories": len(dict(sidecar.get("passage_inventories", {}))),
            "graph_nodes": int(graph_summary.get("node_count", 0)),
            "graph_edges": int(graph_summary.get("edge_count", 0)),
            "graph_node_type_counts": dict(graph_summary.get("node_type_counts", {})),
            "graph_edge_type_counts": dict(graph_summary.get("edge_type_counts", {})),
            "repairable_missing_entity_records": len(find_missing_hyperhippo_entity_payloads(working_dir)),
            "repairable_missing_entity_embeddings": len(find_missing_hyperhippo_entity_embedding_texts(working_dir)),
            "repairable_missing_fact_ids": len(find_missing_hyperhippo_fact_id_updates(working_dir)),
            "repairable_missing_fact_embeddings": len(find_missing_hyperhippo_fact_embedding_texts(working_dir)),
        },
        "validation": validation,
        "extra_metadata": dict(extra_metadata or {}),
    }
    return manifest


def _build_hyperhippo_support_state_from_record_stores(
    working_dir: str,
) -> Tuple[Dict[str, str], Dict[str, Any], List[Dict[str, Any]]]:
    artifact_paths = get_hyperhippo_artifact_paths(working_dir)
    hyperedge_records = _load_record_payloads(artifact_paths["hyperedge_records"])
    passage_records = _load_record_payloads(artifact_paths["passage_records"])
    sidecar = build_support_signature_sidecar_from_hyperedge_records(
        hyperedge_records=hyperedge_records,
        passage_records=passage_records,
    )
    slot_records = build_slot_records_from_hyperedge_records(
        hyperedge_records,
        support_signature_sidecar=sidecar,
    )
    return artifact_paths, sidecar, slot_records


def rebuild_hyperhippo_support_sidecar(working_dir: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    artifact_paths, sidecar, _ = _build_hyperhippo_support_state_from_record_stores(working_dir)
    target_path = output_path or artifact_paths["support_signature_sidecar"]
    save_support_signature_sidecar(target_path, sidecar)
    return sidecar


def rebuild_hyperhippo_slot_records(working_dir: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    artifact_paths, _, slot_records = _build_hyperhippo_support_state_from_record_stores(working_dir)
    target_path = output_path or artifact_paths["slot_records"]
    _save_record_payloads(target_path, slot_records)
    return slot_records


def rebuild_hyperhippo_support_state(
    working_dir: str,
    *,
    sidecar_output_path: Optional[str] = None,
    slot_records_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    artifact_paths, sidecar, slot_records = _build_hyperhippo_support_state_from_record_stores(working_dir)
    resolved_sidecar_path = sidecar_output_path or artifact_paths["support_signature_sidecar"]
    resolved_slot_records_path = slot_records_output_path or artifact_paths["slot_records"]
    save_support_signature_sidecar(resolved_sidecar_path, sidecar)
    _save_record_payloads(resolved_slot_records_path, slot_records)
    return {
        "support_signature_sidecar_path": resolved_sidecar_path,
        "slot_records_path": resolved_slot_records_path,
        "sidecar": sidecar,
        "slot_records": slot_records,
    }


def save_hyperhippo_substrate_manifest(manifest: Mapping[str, Any], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, ensure_ascii=True, indent=2, sort_keys=True)
