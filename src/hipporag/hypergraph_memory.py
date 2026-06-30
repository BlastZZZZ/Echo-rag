from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .utils.embed_utils import retrieve_knn

NODE_TYPE_TO_ID = {"entity": 0, "hyperedge": 1, "passage": 2}
EDGE_TYPE_TO_ID = {
    "entity_hyperedge": 0,
    "hyperedge_hyperedge": 1,
    "hyperedge_passage": 2,
    "entity_entity": 3,
    "entity_passage": 4,
}


def compute_freshness(last_seen: int, current_step: int, tau: float) -> float:
    if tau <= 0:
        return 1.0
    return float(np.exp(-(max(current_step - last_seen, 0) / tau)))


def _build_entity_nodes(entity_records: Dict[str, Dict], current_step: int, freshness_tau: float) -> Dict[str, Dict]:
    node_rows = {}
    for entity_id, record in entity_records.items():
        freshness = compute_freshness(record.get("last_seen", 0), current_step, freshness_tau)
        node_rows[entity_id] = {
            "name": entity_id,
            "content": record["canonical_name"],
            "node_type": "entity",
            "node_type_id": NODE_TYPE_TO_ID["entity"],
            "support_count": record.get("support_count", 1),
            "last_seen": record.get("last_seen", 0),
            "freshness": freshness,
        }
    return node_rows


def _build_passage_nodes(passage_rows: Dict[str, Dict], passage_records: Dict[str, Dict], current_step: int, freshness_tau: float) -> Dict[str, Dict]:
    node_rows = {}
    for passage_id, row in passage_rows.items():
        record = passage_records.get(passage_id, {})
        freshness = compute_freshness(record.get("last_seen", 0), current_step, freshness_tau)
        node_rows[passage_id] = {
            "name": passage_id,
            "content": row["content"],
            "node_type": "passage",
            "node_type_id": NODE_TYPE_TO_ID["passage"],
            "support_count": max(1, len(record.get("proposition_ids", []))),
            "last_seen": record.get("last_seen", 0),
            "freshness": freshness,
        }
    return node_rows


def _build_hyperedge_nodes(hyperedge_records: Dict[str, Dict], current_step: int, freshness_tau: float) -> Dict[str, Dict]:
    node_rows = {}
    for hyperedge_id, record in hyperedge_records.items():
        freshness = compute_freshness(record.get("last_seen", 0), current_step, freshness_tau)
        node_rows[hyperedge_id] = {
            "name": hyperedge_id,
            "content": record["summary_text"],
            "node_type": "hyperedge",
            "node_type_id": NODE_TYPE_TO_ID["hyperedge"],
            "support_count": record.get("support_count", 1),
            "last_seen": record.get("last_seen", 0),
            "freshness": freshness,
        }
    return node_rows


def build_node_rows(
    entity_records: Dict[str, Dict],
    passage_rows: Dict[str, Dict],
    passage_records: Dict[str, Dict],
    hyperedge_records: Dict[str, Dict],
    current_step: int,
    freshness_tau: float,
) -> Dict[str, Dict]:
    node_rows = {}
    node_rows.update(_build_entity_nodes(entity_records, current_step, freshness_tau))
    node_rows.update(_build_passage_nodes(passage_rows, passage_records, current_step, freshness_tau))
    node_rows.update(_build_hyperedge_nodes(hyperedge_records, current_step, freshness_tau))
    return node_rows


def _role_compatibility(shared_entity_id: str, h1: Dict, h2: Dict) -> float:
    roles_1 = set(h1.get("participant_roles", {}).get(shared_entity_id, []))
    roles_2 = set(h2.get("participant_roles", {}).get(shared_entity_id, []))
    if not roles_1 or not roles_2:
        return 0.0
    if roles_1 != roles_2:
        return 1.0
    return 0.5


def _context_consistency(h1: Dict, h2: Dict) -> float:
    overlap = set(h1.get("source_ids", [])) & set(h2.get("source_ids", []))
    return 1.0 if overlap else 0.0


def build_edge_records(
    entity_records: Dict[str, Dict],
    passage_records: Dict[str, Dict],
    hyperedge_records: Dict[str, Dict],
    entity_embedding_map: Dict[str, np.ndarray],
    hyperedge_embedding_map: Dict[str, np.ndarray],
    config,
) -> List[Dict]:
    edge_records: List[Dict] = []

    # EH and HP edges.
    for hyperedge_id, record in hyperedge_records.items():
        support = max(1, record.get("support_count", 1))
        for participant_id in record.get("participant_ids", []):
            edge_weight = 1.0 + 0.1 * len(record.get("participant_roles", {}).get(participant_id, []))
            edge_records.extend(
                [
                    {
                        "src": participant_id,
                        "tgt": hyperedge_id,
                        "weight": edge_weight,
                        "edge_type": "entity_hyperedge",
                        "edge_type_id": EDGE_TYPE_TO_ID["entity_hyperedge"],
                        "support_count": support,
                    },
                    {
                        "src": hyperedge_id,
                        "tgt": participant_id,
                        "weight": edge_weight,
                        "edge_type": "entity_hyperedge",
                        "edge_type_id": EDGE_TYPE_TO_ID["entity_hyperedge"],
                        "support_count": support,
                    },
                ]
            )
        for passage_id in record.get("source_ids", []):
            edge_weight = 1.0 + np.log1p(support)
            edge_records.extend(
                [
                    {
                        "src": hyperedge_id,
                        "tgt": passage_id,
                        "weight": edge_weight,
                        "edge_type": "hyperedge_passage",
                        "edge_type_id": EDGE_TYPE_TO_ID["hyperedge_passage"],
                        "support_count": support,
                    },
                    {
                        "src": passage_id,
                        "tgt": hyperedge_id,
                        "weight": edge_weight,
                        "edge_type": "hyperedge_passage",
                        "edge_type_id": EDGE_TYPE_TO_ID["hyperedge_passage"],
                        "support_count": support,
                    },
                ]
            )

    # EP edges.
    for passage_id, record in passage_records.items():
        for entity_id in record.get("entity_ids", []):
            edge_records.extend(
                [
                    {
                        "src": entity_id,
                        "tgt": passage_id,
                        "weight": 1.0,
                        "edge_type": "entity_passage",
                        "edge_type_id": EDGE_TYPE_TO_ID["entity_passage"],
                        "support_count": 1,
                    },
                    {
                        "src": passage_id,
                        "tgt": entity_id,
                        "weight": 1.0,
                        "edge_type": "entity_passage",
                        "edge_type_id": EDGE_TYPE_TO_ID["entity_passage"],
                        "support_count": 1,
                    },
                ]
            )

    # EE edges from embedding neighbors.
    entity_ids = list(entity_embedding_map.keys())
    ee_top_k = int(getattr(config, "ee_top_k", 16))
    if entity_ids and ee_top_k > 0:
        entity_vecs = np.array([entity_embedding_map[entity_id] for entity_id in entity_ids], dtype=np.float32)
        knn_results = retrieve_knn(
            query_ids=entity_ids,
            key_ids=entity_ids,
            query_vecs=entity_vecs,
            key_vecs=entity_vecs,
            k=min(len(entity_ids), max(2, ee_top_k)),
            query_batch_size=min(len(entity_ids), max(1, getattr(config, "ee_query_batch_size", 64))),
            key_batch_size=min(len(entity_ids), max(1, getattr(config, "ee_key_batch_size", 256))),
        )
        for entity_id in entity_ids:
            nn_ids, nn_scores = knn_results[entity_id]
            for neighbor_id, score in zip(nn_ids, nn_scores):
                if neighbor_id == entity_id:
                    continue
                if score < getattr(config, "ee_similarity_threshold", 0.85):
                    continue
                edge_records.append(
                    {
                        "src": entity_id,
                        "tgt": neighbor_id,
                        "weight": float(score),
                        "edge_type": "entity_entity",
                        "edge_type_id": EDGE_TYPE_TO_ID["entity_entity"],
                        "support_count": 1,
                    }
                )

    # HH edges from shared anchors and constrained semantics.
    anchor_to_hids: Dict[str, List[str]] = defaultdict(list)
    passage_to_hids: Dict[str, List[str]] = defaultdict(list)
    for hyperedge_id, record in hyperedge_records.items():
        for participant_id in record.get("participant_ids", []):
            anchor_to_hids[participant_id].append(hyperedge_id)
        for passage_id in record.get("source_ids", []):
            passage_to_hids[passage_id].append(hyperedge_id)

    candidate_pairs: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: {"shared_anchor": 0.0, "context": 0.0})
    for participant_id, hids in anchor_to_hids.items():
        unique_hids = sorted(set(hids))
        for idx, h1 in enumerate(unique_hids):
            for h2 in unique_hids[idx + 1 :]:
                candidate_pairs[(h1, h2)]["shared_anchor"] += 1.0
                candidate_pairs[(h1, h2)]["anchor_id"] = participant_id

    for _, hids in passage_to_hids.items():
        unique_hids = sorted(set(hids))
        for idx, h1 in enumerate(unique_hids):
            for h2 in unique_hids[idx + 1 :]:
                candidate_pairs[(h1, h2)]["context"] = 1.0

    outgoing_by_hid: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for (h1, h2), features in candidate_pairs.items():
        r1 = hyperedge_records[h1]
        r2 = hyperedge_records[h2]
        shared_anchor = features.get("shared_anchor", 0.0)
        anchor_id = features.get("anchor_id")
        role_compat = _role_compatibility(anchor_id, r1, r2) if anchor_id else 0.0
        summary_sim = 0.0
        if h1 in hyperedge_embedding_map and h2 in hyperedge_embedding_map:
            summary_sim = float(np.dot(hyperedge_embedding_map[h1], hyperedge_embedding_map[h2]))
        context_consistency = max(features.get("context", 0.0), _context_consistency(r1, r2))
        weight = (
            getattr(config, "hh_shared_anchor_weight", 0.35) * shared_anchor
            + getattr(config, "hh_role_compat_weight", 0.20) * role_compat
            + getattr(config, "hh_summary_sim_weight", 0.25) * max(summary_sim, 0.0)
            + getattr(config, "hh_context_consistency_weight", 0.20) * context_consistency
        )
        if weight < getattr(config, "hh_min_weight_threshold", 0.40):
            continue
        outgoing_by_hid[h1].append((h2, weight))
        outgoing_by_hid[h2].append((h1, weight))

    for h1, neighbors in outgoing_by_hid.items():
        neighbors = sorted(neighbors, key=lambda item: item[1], reverse=True)[: getattr(config, "hh_top_k_per_node", 8)]
        for h2, weight in neighbors:
            edge_records.append(
                {
                    "src": h1,
                    "tgt": h2,
                    "weight": float(weight),
                    "edge_type": "hyperedge_hyperedge",
                    "edge_type_id": EDGE_TYPE_TO_ID["hyperedge_hyperedge"],
                    "support_count": 1,
                }
            )

    return edge_records
