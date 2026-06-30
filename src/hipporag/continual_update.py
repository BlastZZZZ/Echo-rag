from typing import Dict, Iterable, List, Tuple


class ContinualUpdater:
    """Update structured memory stores via proposition-level upserts."""

    def upsert_entity_records(
        self,
        entity_payloads: Iterable[Dict],
        entity_record_store,
    ):
        existing_records = entity_record_store.get_all()
        updated_records = {key: dict(value) for key, value in existing_records.items()}

        for payload in entity_payloads:
            entity_id = payload["hash_id"]
            existing = updated_records.get(entity_id)
            if existing is None:
                updated_records[entity_id] = dict(payload)
                continue
            existing["source_ids"] = sorted(set(existing.get("source_ids", [])) | set(payload.get("source_ids", [])))
            existing["support_count"] = existing.get("support_count", 0) + payload.get("support_count", 0)
            existing["last_seen"] = max(existing.get("last_seen", 0), payload.get("last_seen", 0))
            existing["aliases"] = sorted(set(existing.get("aliases", [])) | set(payload.get("aliases", [])))
            updated_records[entity_id] = existing

        entity_record_store.upsert(updated_records.values())

    def upsert_passage_records(
        self,
        passage_payloads: Iterable[Dict],
        passage_record_store,
    ):
        existing_records = passage_record_store.get_all()
        updated_records = {key: dict(value) for key, value in existing_records.items()}

        for payload in passage_payloads:
            passage_id = payload["hash_id"]
            existing = updated_records.get(passage_id)
            if existing is None:
                updated_records[passage_id] = dict(payload)
                continue
            existing["entity_ids"] = sorted(set(existing.get("entity_ids", [])) | set(payload.get("entity_ids", [])))
            existing["proposition_ids"] = sorted(
                set(existing.get("proposition_ids", [])) | set(payload.get("proposition_ids", []))
            )
            existing["last_seen"] = max(existing.get("last_seen", 0), payload.get("last_seen", 0))
            updated_records[passage_id] = existing

        passage_record_store.upsert(updated_records.values())

    def delete_passages(
        self,
        passage_ids: Iterable[str],
        passage_record_store,
        entity_record_store,
        hyperedge_record_store,
    ):
        passage_ids = list(dict.fromkeys(passage_ids))
        passage_payloads = passage_record_store.get_many(passage_ids)
        if not passage_payloads:
            return

        entity_records = entity_record_store.get_all()
        hyperedge_records = hyperedge_record_store.get_all()
        entity_ids_to_delete = set()
        hyperedge_ids_to_delete = set()

        for passage_id, passage_payload in passage_payloads.items():
            for entity_id in passage_payload.get("entity_ids", []):
                entity_record = entity_records.get(entity_id)
                if entity_record is None:
                    continue
                updated_source_ids = [source for source in entity_record.get("source_ids", []) if source != passage_id]
                entity_record["source_ids"] = updated_source_ids
                entity_record["support_count"] = len(updated_source_ids)
                if entity_record["support_count"] == 0:
                    entity_ids_to_delete.add(entity_id)
                    entity_records.pop(entity_id, None)
                else:
                    entity_records[entity_id] = entity_record

            for proposition_id in passage_payload.get("proposition_ids", []):
                proposition_record = hyperedge_records.get(proposition_id)
                if proposition_record is None:
                    continue
                updated_source_ids = [
                    source for source in proposition_record.get("source_ids", []) if source != passage_id
                ]
                proposition_record["source_ids"] = updated_source_ids
                proposition_record["support_count"] = len(updated_source_ids)
                if proposition_record["support_count"] == 0:
                    hyperedge_ids_to_delete.add(proposition_id)
                    hyperedge_records.pop(proposition_id, None)
                else:
                    hyperedge_records[proposition_id] = proposition_record

        passage_record_store.delete(passage_payloads.keys())
        entity_record_store.delete(entity_ids_to_delete)
        hyperedge_record_store.delete(hyperedge_ids_to_delete)
        entity_record_store.upsert(entity_records.values())
        hyperedge_record_store.upsert(hyperedge_records.values())
