import json
import os
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .utils.logging_utils import get_logger

logger = get_logger(__name__)


class RecordStore:
    """Lightweight JSON-backed parquet store for structured metadata records."""

    def __init__(self, db_dir: str, namespace: str, key_field: str = "hash_id"):
        self.namespace = namespace
        self.key_field = key_field

        if not os.path.exists(db_dir):
            logger.info(f"Creating working directory: {db_dir}")
            os.makedirs(db_dir, exist_ok=True)

        self.filename = os.path.join(db_dir, f"records_{namespace}.parquet")
        self._load_data()

    def _load_data(self):
        self.records: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            for row in df.to_dict(orient="records"):
                payload = json.loads(row["payload_json"])
                key = payload[self.key_field]
                self.records[key] = payload
            logger.info(f"Loaded {len(self.records)} records from {self.filename}")

    def _save_data(self):
        rows = [
            {
                self.key_field: key,
                "payload_json": json.dumps(record, ensure_ascii=True, sort_keys=True),
            }
            for key, record in self.records.items()
        ]
        pd.DataFrame(rows).to_parquet(self.filename, index=False)
        logger.info(f"Saved {len(self.records)} records to {self.filename}")

    def upsert(self, records: Iterable[Dict[str, Any]]):
        num_updates = 0
        for record in records:
            if self.key_field not in record:
                raise KeyError(f"Record missing key field `{self.key_field}`: {record}")
            key = record[self.key_field]
            self.records[key] = deepcopy(record)
            num_updates += 1
        if num_updates > 0:
            self._save_data()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        record = self.records.get(key)
        return deepcopy(record) if record is not None else None

    def get_many(self, keys: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        return {key: deepcopy(self.records[key]) for key in keys if key in self.records}

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        return deepcopy(self.records)

    def get_all_ref(self) -> Dict[str, Dict[str, Any]]:
        return self.records

    def get_all_ids(self) -> List[str]:
        return list(self.records.keys())

    def delete(self, keys: Iterable[str]):
        deleted = 0
        for key in list(keys):
            if key in self.records:
                del self.records[key]
                deleted += 1
        if deleted > 0:
            self._save_data()
