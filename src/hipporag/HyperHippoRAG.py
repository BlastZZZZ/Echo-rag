import json
import logging
import hashlib
import os
import pickle
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import igraph as ig
import numpy as np
from tqdm import tqdm

from .HippoRAG import HippoRAG
from .continual_update import ContinualUpdater
from .embedding_store import EmbeddingStore
from .evaluation.retrieval_eval import RetrievalRecall
from .hypergraph_memory import EDGE_TYPE_TO_ID, NODE_TYPE_TO_ID, build_edge_records, build_node_rows
from .bridge_path_memory import (
    build_bridge_path_record_ids_from_hyperedge_records,
    build_bridge_path_records_from_hyperedge_records,
    build_reserved_bridge_support_endpoint_record_ids_from_bridge_path_records,
    build_reserved_bridge_support_endpoint_records_from_bridge_path_records,
    build_reserved_bridge_support_trail_membership_record_ids_from_bridge_path_records,
    build_reserved_bridge_support_trail_membership_records_from_bridge_path_records,
    build_reserved_bridge_support_trail_record_ids_from_bridge_path_records,
    build_reserved_bridge_support_trail_records_from_bridge_path_records,
    build_bridge_support_endpoint_record_ids_from_bridge_path_records,
    build_bridge_support_endpoint_records_from_bridge_path_records,
    build_bridge_support_trail_membership_record_ids_from_bridge_path_records,
    build_bridge_support_trail_membership_records_from_bridge_path_records,
    build_bridge_support_trail_record_ids_from_bridge_path_records,
    build_bridge_support_trail_records_from_bridge_path_records,
)
from .information_extraction.proposition_extraction import PropositionExtractor
from .proposition_memory import (
    build_proposition_candidates,
    build_proposition_candidates_from_units,
    build_proposition_embedding_text,
    build_proposition_embedding_text_variants,
    build_proposition_fact_embedding_text,
    extract_evidence_snippet,
    upsert_propositions,
)
from .query_diffusion import QueryConditionedDiffuser
from .query_router import HybridQueryRouter
from .record_store import RecordStore
from .support_signature_memory import (
    SUPPORT_SIGNATURE_SIDECAR_VERSION,
    build_slot_records_from_hyperedge_records,
    build_slot_record_ids_from_hyperedge_records,
    build_support_signature_sidecar_from_hyperedge_records,
    load_support_signature_sidecar,
    save_support_signature_sidecar,
    SLOT_TIER_SUPPORT_FIELDS,
)
from .canonical_substrate import find_missing_entity_payloads_from_memory
from .canonical_substrate import repair_missing_entity_embeddings_with_store
from .canonical_substrate import repair_missing_fact_embeddings_with_store
from .canonical_substrate import repair_missing_fact_ids_with_record_and_fact_store
from .canonical_retrieval_interface import (
    build_canonical_interface_cache_payload,
    build_canonical_retrieval_interface_from_runtime,
    load_canonical_interface_from_cache_payload,
)
from .utils.misc_utils import (
    QuerySolution,
    TripleRawOutput,
    NerRawOutput,
    PropositionRawOutput,
    compute_mdhash_id,
    extract_entity_nodes,
    flatten_facts,
    min_max_normalize,
    reformat_openie_results,
    reformat_proposition_results,
    text_processing,
)
from .prompts.linking import (
    build_hyperedge_rerank_user_prompt,
    build_hyperedge_query_rewrite_user_prompt,
    get_hyperedge_rerank_system_prompt,
    get_hyperedge_query_rewrite_system_prompt,
    get_query_instruction,
)


logger = logging.getLogger(__name__)


class HyperHippoRAG(HippoRAG):
    def _get_hyperedge_source_mode(self) -> str:
        return getattr(self.global_config, "hyperedge_source_mode", "triple")

    def _use_proposition_hyperedges(self) -> bool:
        return self._get_hyperedge_source_mode() == "proposition"

    @staticmethod
    def _normalize_seed_component(component: np.ndarray) -> np.ndarray:
        component_sum = float(component.sum())
        if component_sum > 0:
            component = component / component_sum
        return component

    @staticmethod
    def _normalize_score_vector(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32)
        if scores.size == 0:
            return scores.astype(np.float32)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        if float(np.max(np.abs(scores))) <= 0:
            return np.zeros_like(scores, dtype=np.float32)
        min_val = float(np.min(scores))
        max_val = float(np.max(scores))
        if abs(max_val - min_val) < 1e-8:
            return np.ones_like(scores, dtype=np.float32) if max_val > 0 else np.zeros_like(scores, dtype=np.float32)
        return np.array(min_max_normalize(scores), dtype=np.float32)

    def _get_hyperedge_query_instruction(self) -> str:
        if self._use_proposition_hyperedges():
            return get_query_instruction("query_to_proposition")
        if getattr(self.global_config, "hyperedge_embedding_mode", "fact_reuse") == "prop_only":
            return get_query_instruction("query_to_hyperedge")
        return get_query_instruction("query_to_fact")

    def _use_hyperedge_query_v2(self) -> bool:
        return (
            not self._use_proposition_hyperedges()
            and
            getattr(self.global_config, "hyperedge_embedding_mode", "fact_reuse") == "prop_only"
            and getattr(self.global_config, "hyperedge_query_version", "v1") == "v2"
        )

    def _use_hyperedge_query_v3(self) -> bool:
        return (
            not self._use_proposition_hyperedges()
            and
            getattr(self.global_config, "hyperedge_embedding_mode", "fact_reuse") == "prop_only"
            and getattr(self.global_config, "hyperedge_query_version", "v1") == "v3"
        )

    def _use_hyperedge_query_v4(self) -> bool:
        return (
            not self._use_proposition_hyperedges()
            and
            getattr(self.global_config, "hyperedge_embedding_mode", "fact_reuse") == "prop_only"
            and getattr(self.global_config, "hyperedge_query_version", "v1") == "v4"
        )

    def _use_role_aware_hyperedge_query(self) -> bool:
        return self._use_hyperedge_query_v3() or self._use_hyperedge_query_v4()

    def _use_contextual_hyperedge_text(self) -> bool:
        return (
            getattr(self.global_config, "hyperedge_embedding_mode", "fact_reuse") == "prop_only"
            and getattr(self.global_config, "hyperedge_embedding_text_mode", "summary") == "contextual"
        )

    def _use_multi_view_hyperedge_keys(self) -> bool:
        return (
            getattr(self.global_config, "hyperedge_embedding_mode", "fact_reuse") == "prop_only"
            and getattr(self.global_config, "hyperedge_key_view_mode", "single") == "multi"
        )

    def _use_hyperedge_rerank(self) -> bool:
        return (
            getattr(self.global_config, "hyperedge_embedding_mode", "fact_reuse") == "prop_only"
            and int(getattr(self.global_config, "hyperedge_rerank_top_k", 0)) > 1
        )

    def _parse_hyperedge_query_rewrite_response(self, response: str) -> List[Dict[str, object]]:
        try:
            payload = json.loads(response)
        except (TypeError, json.JSONDecodeError):
            return []

        if isinstance(payload, dict):
            candidates = payload.get("search_queries", payload.get("queries", []))
        elif isinstance(payload, list):
            candidates = payload
        else:
            candidates = []

        if not isinstance(candidates, list):
            return []

        parsed = []
        for candidate in candidates:
            role = "rewritten"
            confidence = 1.0
            if isinstance(candidate, str):
                query_text = candidate
            elif isinstance(candidate, dict):
                query_text = candidate.get("query", candidate.get("text", ""))
                role = candidate.get("role", role)
                confidence = candidate.get("confidence", confidence)
            else:
                continue

            if not isinstance(query_text, str):
                continue
            normalized = " ".join(query_text.strip().split())
            if normalized:
                parsed.append(
                    {
                        "query": normalized,
                        "role": self._normalize_hyperedge_query_role(role),
                        "confidence": self._clamp_confidence(confidence, default=1.0),
                    }
                )
        return parsed

    @staticmethod
    def _normalize_hyperedge_query_role(role) -> str:
        if not isinstance(role, str):
            return "other"
        role_normalized = role.strip().lower()
        if not role_normalized:
            return "other"
        if "target" in role_normalized:
            return "target"
        if "bridge" in role_normalized:
            return "bridge"
        if "constraint" in role_normalized:
            return "constraint"
        if "inverse" in role_normalized or "reverse" in role_normalized:
            return "inverse"
        if "original" in role_normalized:
            return "original"
        if "rewrite" in role_normalized:
            return "rewritten"
        return "other"

    @staticmethod
    def _clamp_confidence(value, default: float = 1.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        return float(min(max(numeric, 0.0), 1.0))

    def _build_hyperedge_query_view(self, query_text: str, role: str, confidence: float) -> Dict[str, object]:
        return {
            "query": " ".join(query_text.strip().split()),
            "role": self._normalize_hyperedge_query_role(role),
            "confidence": self._clamp_confidence(confidence, default=1.0),
        }

    def _get_hyperedge_query_role_weight(self, role: str) -> float:
        role_weights = {
            "target": 1.00,
            "bridge": 0.90,
            "constraint": 0.82,
            "inverse": 0.72,
            "rewritten": 0.78,
            "original": 0.68,
            "other": 0.70,
        }
        return float(role_weights.get(role, role_weights["other"]))

    def _get_hyperedge_query_views(self, query_text: str) -> List[Dict[str, object]]:
        if not (self._use_hyperedge_query_v2() or self._use_role_aware_hyperedge_query()):
            return [self._build_hyperedge_query_view(query_text, "original", 1.0)]

        if not hasattr(self, "hyperedge_query_view_cache"):
            self.hyperedge_query_view_cache = {}
        cached_views = self.hyperedge_query_view_cache.get(query_text)
        if cached_views is not None:
            return cached_views

        max_views = max(1, int(getattr(self.global_config, "hyperedge_query_max_views", 4)))
        include_original = bool(getattr(self.global_config, "hyperedge_query_include_original", True))

        rewritten_views = []
        if getattr(self, "llm_model", None) is not None:
            query_version = getattr(self.global_config, "hyperedge_query_version", "v2")
            messages = [
                {"role": "system", "content": get_hyperedge_query_rewrite_system_prompt(query_version)},
                {
                    "role": "user",
                    "content": build_hyperedge_query_rewrite_user_prompt(
                        question=query_text,
                        max_views=max_views,
                        version=query_version,
                    ),
                },
            ]
            try:
                response, _ = self.llm_model.infer(
                    messages=messages,
                    model=self.global_config.llm_name,
                    max_completion_tokens=256,
                    response_format={"type": "json_object"},
                )
                rewritten_views = self._parse_hyperedge_query_rewrite_response(response)
            except Exception:
                rewritten_views = []

        all_views = []
        if include_original:
            all_views.append(self._build_hyperedge_query_view(query_text, "original", 1.0))
        all_views.extend(rewritten_views)

        deduped_views: List[Dict[str, object]] = []
        best_by_query: Dict[str, Dict[str, object]] = {}
        for candidate in all_views:
            normalized = candidate.get("query", "")
            if not normalized:
                continue
            existing = best_by_query.get(normalized)
            if existing is None or float(candidate.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
                best_by_query[normalized] = candidate

        deduped_views = list(best_by_query.values())

        if not deduped_views:
            deduped_views = [self._build_hyperedge_query_view(query_text, "original", 1.0)]

        if self._use_role_aware_hyperedge_query():
            deduped_views.sort(
                key=lambda view: (
                    float(self._get_hyperedge_query_role_weight(str(view.get("role", "other")))),
                    float(view.get("confidence", 0.0)),
                ),
                reverse=True,
            )
        deduped_views = deduped_views[:max_views]
        self.hyperedge_query_view_cache[query_text] = deduped_views
        return deduped_views

    def _get_hyperedge_query_view_weights(self, query_views: List[Dict[str, object]]) -> np.ndarray:
        if not query_views:
            return np.array([], dtype=np.float32)

        confidence_floor = float(
            min(max(getattr(self.global_config, "hyperedge_query_v3_confidence_floor", 0.35), 0.0), 1.0)
        )
        weights = []
        for view in query_views:
            confidence = float(view.get("confidence", 1.0))
            confidence = max(confidence_floor, min(confidence, 1.0))
            role_weight = self._get_hyperedge_query_role_weight(str(view.get("role", "other")))
            weights.append(role_weight * confidence)

        weights = np.array(weights, dtype=np.float32)
        if float(weights.sum()) <= 0:
            return np.ones(len(query_views), dtype=np.float32) / float(len(query_views))
        return weights / float(weights.sum())

    def _aggregate_multiview_scores(self, query: str, score_matrix: np.ndarray) -> np.ndarray:
        score_matrix = np.asarray(score_matrix, dtype=np.float32)
        if score_matrix.ndim == 0:
            score_matrix = score_matrix.reshape(1)
        if score_matrix.ndim == 1:
            return self._normalize_score_vector(score_matrix)
        if score_matrix.shape[1] == 1:
            return self._normalize_score_vector(score_matrix[:, 0])

        if self._use_role_aware_hyperedge_query():
            view_weights = getattr(self, "hyperedge_query_view_weights", {}).get(query)
            if view_weights is not None and len(view_weights) == score_matrix.shape[1]:
                weighted_scores = np.dot(score_matrix, view_weights)
                max_scores = np.max(score_matrix, axis=1)
                max_score_weight = float(
                    min(max(getattr(self.global_config, "hyperedge_query_v3_max_score_weight", 0.35), 0.0), 1.0)
                )
                combined_scores = max_score_weight * max_scores + (1.0 - max_score_weight) * weighted_scores
                return self._normalize_score_vector(combined_scores)
            return self._normalize_score_vector(np.max(score_matrix, axis=1))

        score_agg = getattr(self.global_config, "hyperedge_query_score_agg", "max")
        if score_agg == "mean":
            return self._normalize_score_vector(np.mean(score_matrix, axis=1))
        return self._normalize_score_vector(np.max(score_matrix, axis=1))

    def _aggregate_role_scores(
        self,
        query: str,
        score_matrix: np.ndarray,
        query_views: List[Dict[str, object]],
    ) -> Dict[str, np.ndarray]:
        role_scores: Dict[str, np.ndarray] = {}
        score_matrix = np.asarray(score_matrix, dtype=np.float32)
        if score_matrix.ndim == 0:
            score_matrix = score_matrix.reshape(1)

        if score_matrix.ndim == 1:
            default_role = "original"
            if query_views:
                default_role = str(query_views[0].get("role", default_role))
            role_scores[default_role] = self._normalize_score_vector(score_matrix)
            return role_scores

        if not query_views or len(query_views) != score_matrix.shape[1]:
            role_scores["original"] = self._aggregate_multiview_scores(query, score_matrix)
            return role_scores

        role_to_columns: Dict[str, List[int]] = defaultdict(list)
        for col_idx, view in enumerate(query_views):
            role_to_columns[str(view.get("role", "other"))].append(col_idx)

        view_weights = getattr(self, "hyperedge_query_view_weights", {}).get(query)
        for role, columns in role_to_columns.items():
            role_matrix = score_matrix[:, columns]
            if role_matrix.ndim == 1:
                role_vector = role_matrix
            elif role_matrix.shape[1] == 1:
                role_vector = role_matrix[:, 0]
            else:
                if view_weights is not None and len(view_weights) == score_matrix.shape[1]:
                    role_view_weights = np.array(view_weights[columns], dtype=np.float32)
                    role_weight_sum = float(role_view_weights.sum())
                    if role_weight_sum > 0:
                        role_view_weights = role_view_weights / role_weight_sum
                        role_vector = np.max(role_matrix * role_view_weights.reshape(1, -1), axis=1)
                    else:
                        role_vector = np.max(role_matrix, axis=1)
                else:
                    role_vector = np.max(role_matrix, axis=1)
            role_scores[role] = self._normalize_score_vector(role_vector)
        return role_scores

    def _build_v4_primary_hyperedge_scores(
        self,
        aggregated_scores: np.ndarray,
        role_scores: Dict[str, np.ndarray],
    ) -> np.ndarray:
        if len(aggregated_scores) == 0:
            return aggregated_scores

        weighted_candidates = []
        for role, weight in [
            ("target", 1.00),
            ("inverse", 0.78),
            ("original", 0.62),
            ("rewritten", 0.56),
            ("other", 0.48),
        ]:
            role_vector = role_scores.get(role)
            if role_vector is not None and len(role_vector) == len(aggregated_scores):
                weighted_candidates.append(weight * role_vector)

        if not weighted_candidates:
            return aggregated_scores

        weighted_candidates.append(0.40 * aggregated_scores)
        primary_scores = np.max(np.stack(weighted_candidates, axis=1), axis=1)
        return self._normalize_score_vector(primary_scores)

    def _get_hyperedge_score_bundle(self, query: str) -> Dict[str, object]:
        if not hasattr(self, "hyperedge_score_bundle_cache"):
            self.hyperedge_score_bundle_cache = {}
        cached_bundle = self.hyperedge_score_bundle_cache.get(query)
        if cached_bundle is not None:
            return cached_bundle

        hyperedge_key_embeddings = getattr(self, "hyperedge_view_embeddings", self.hyperedge_embeddings)
        if len(hyperedge_key_embeddings) == 0:
            empty_bundle = {
                "aggregated": np.array([], dtype=np.float32),
                "primary": np.array([], dtype=np.float32),
                "role_scores": {},
                "query_views": [],
            }
            self.hyperedge_score_bundle_cache[query] = empty_bundle
            return empty_bundle

        query_embedding = self.query_to_embedding["hyperedge"][query]
        flat_score_matrix = np.dot(hyperedge_key_embeddings, query_embedding.T)
        if np.asarray(flat_score_matrix).ndim == 0:
            flat_score_matrix = np.asarray(flat_score_matrix, dtype=np.float32).reshape(1)

        view_spans = getattr(self, "hyperedge_pos_to_view_span", [])
        if not view_spans:
            view_spans = [(idx, idx + 1) for idx in range(len(self.hyperedge_embeddings))]

        reduced_rows = []
        for start_idx, end_idx in view_spans:
            key_view_scores = np.asarray(flat_score_matrix[start_idx:end_idx], dtype=np.float32)
            if key_view_scores.ndim <= 1:
                reduced_rows.append(float(np.max(key_view_scores)))
            else:
                reduced_rows.append(np.max(key_view_scores, axis=0))
        score_matrix = np.array(reduced_rows, dtype=np.float32)
        query_views = getattr(self, "hyperedge_query_view_cache", {}).get(
            query,
            [self._build_hyperedge_query_view(query, "original", 1.0)],
        )
        aggregated_scores = self._aggregate_multiview_scores(query, score_matrix)
        role_scores = self._aggregate_role_scores(query, score_matrix, query_views)
        primary_scores = aggregated_scores
        if self._use_hyperedge_query_v4():
            primary_scores = self._build_v4_primary_hyperedge_scores(aggregated_scores, role_scores)

        bundle = {
            "aggregated": aggregated_scores,
            "primary": primary_scores,
            "role_scores": role_scores,
            "query_views": query_views,
        }
        self.hyperedge_score_bundle_cache[query] = bundle
        return bundle

    def _use_passage_rerank(self) -> bool:
        return int(getattr(self.global_config, "hyperedge_passage_rerank_top_k", 0)) > 1

    @staticmethod
    def _truncate_passage_for_rerank(text: str, max_chars: int) -> str:
        normalized = " ".join(str(text).split())
        if max_chars <= 0 or len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _parse_ranked_indices_response(response: str, num_candidates: int) -> List[int]:
        parsed_indices: List[int] = []
        try:
            payload = json.loads(response)
        except (TypeError, json.JSONDecodeError):
            payload = {}

        if isinstance(payload, dict):
            candidates = payload.get("ranked_indices", payload.get("indices", payload.get("ranking", [])))
        elif isinstance(payload, list):
            candidates = payload
        else:
            candidates = []

        if not isinstance(candidates, list):
            candidates = []

        for candidate in candidates:
            if isinstance(candidate, dict):
                candidate = candidate.get("index", candidate.get("id"))
            try:
                index = int(candidate)
            except (TypeError, ValueError):
                continue
            if 0 <= index < num_candidates and index not in parsed_indices:
                parsed_indices.append(index)

        for fallback_idx in range(num_candidates):
            if fallback_idx not in parsed_indices:
                parsed_indices.append(fallback_idx)
        return parsed_indices

    @classmethod
    def _parse_passage_rerank_response(cls, response: str, num_candidates: int) -> List[int]:
        return cls._parse_ranked_indices_response(response, num_candidates)

    def _build_hyperedge_rerank_candidate_text(self, hyperedge_id: str, max_chars: int) -> str:
        record = getattr(self, "hyperedge_records", {}).get(hyperedge_id, {})
        summary_text = str(record.get("summary_text", ""))
        relation_type = str(record.get("relation_type", ""))
        participant_texts = list(record.get("participant_texts", []))
        support_count = int(record.get("support_count", 1))
        candidate_text = str(record.get("embedding_text", "")).strip()

        if not candidate_text:
            evidence_snippets = []
            for source_id in list(record.get("source_ids", []))[:1]:
                try:
                    row = self.chunk_embedding_store.get_row(source_id)
                except Exception:
                    row = None
                if row is None:
                    continue
                snippet = self._truncate_passage_for_rerank(row.get("content", ""), max(96, max_chars // 2))
                if snippet and snippet not in evidence_snippets:
                    evidence_snippets.append(snippet)
            candidate_text = build_proposition_embedding_text(
                summary_text=summary_text,
                relation_type=relation_type,
                participant_texts=participant_texts,
                evidence_snippets=evidence_snippets,
                support_count=support_count,
            )

        compact_text = " | ".join(line.strip() for line in candidate_text.splitlines() if line.strip())
        return self._truncate_passage_for_rerank(compact_text, max_chars)

    def _rerank_hyperedge_candidates(
        self,
        query: str,
        score_bundle: Dict[str, object],
    ) -> Dict[str, np.ndarray | List[int]]:
        primary_scores = np.array(score_bundle.get("primary", np.array([])), dtype=np.float32)
        graph_size = self.graph.vcount() if getattr(self, "graph", None) is not None else 0
        default_result = {
            "adjusted_scores": primary_scores,
            "reset_component": np.zeros(graph_size, dtype=np.float32),
            "ranked_hyperedge_indices": [],
        }
        if not self._use_hyperedge_rerank():
            return default_result
        if getattr(self, "llm_model", None) is None or len(primary_scores) <= 1:
            return default_result

        if not hasattr(self, "hyperedge_rerank_cache"):
            self.hyperedge_rerank_cache = {}
        cached_result = self.hyperedge_rerank_cache.get(query)
        if cached_result is not None:
            return cached_result

        top_k = min(int(getattr(self.global_config, "hyperedge_rerank_top_k", 0)), len(primary_scores))
        if top_k <= 1:
            self.hyperedge_rerank_cache[query] = default_result
            return default_result

        max_chars = int(getattr(self.global_config, "hyperedge_rerank_max_chars", 360))
        candidate_positions = np.argsort(primary_scores)[::-1][:top_k]
        candidate_texts = []
        for local_idx, hyperedge_pos in enumerate(candidate_positions):
            hyperedge_id = self.hyperedge_node_keys[int(hyperedge_pos)]
            candidate_text = self._build_hyperedge_rerank_candidate_text(hyperedge_id, max_chars)
            candidate_texts.append(f"[{local_idx}] {candidate_text}")

        messages = [
            {"role": "system", "content": get_hyperedge_rerank_system_prompt()},
            {"role": "user", "content": build_hyperedge_rerank_user_prompt(query, candidate_texts)},
        ]

        try:
            response, _ = self.llm_model.infer(
                messages=messages,
                model=self.global_config.llm_name,
                max_completion_tokens=256,
                response_format={"type": "json_object"},
            )
            ranked_local_indices = self._parse_ranked_indices_response(response, top_k)
        except Exception:
            ranked_local_indices = list(range(top_k))

        rerank_signal = np.zeros(len(primary_scores), dtype=np.float32)
        ranked_hyperedge_indices: List[int] = []
        denom = max(top_k - 1, 1)
        for rank, local_idx in enumerate(ranked_local_indices):
            if not (0 <= local_idx < len(candidate_positions)):
                continue
            hyperedge_pos = int(candidate_positions[local_idx])
            rank_weight = 1.0 - 0.75 * (float(rank) / float(denom))
            rerank_signal[hyperedge_pos] = max(rerank_signal[hyperedge_pos], rank_weight)
            if hyperedge_pos not in ranked_hyperedge_indices:
                ranked_hyperedge_indices.append(hyperedge_pos)

        rerank_signal = self._normalize_score_vector(rerank_signal)
        if float(np.sum(rerank_signal)) > 0:
            score_mix = float(min(max(getattr(self.global_config, "hyperedge_rerank_score_mix", 0.60), 0.0), 1.0))
            adjusted_scores = self._normalize_score_vector(
                (1.0 - score_mix) * primary_scores + score_mix * rerank_signal
            )
        else:
            adjusted_scores = primary_scores

        reset_component = np.zeros(graph_size, dtype=np.float32)
        if graph_size > 0 and len(self.hyperedge_node_idxs) == len(primary_scores):
            for hyperedge_pos, signal in enumerate(rerank_signal):
                if signal <= 0:
                    continue
                node_idx = self.hyperedge_node_idxs[hyperedge_pos]
                reset_component[node_idx] = float(signal)
            if float(reset_component.sum()) > 0:
                reset_component = self._normalize_seed_component(reset_component)

        result = {
            "adjusted_scores": np.array(adjusted_scores, dtype=np.float32),
            "reset_component": np.array(reset_component, dtype=np.float32),
            "ranked_hyperedge_indices": ranked_hyperedge_indices,
        }
        self.hyperedge_rerank_cache[query] = result
        return result

    def _rerank_passage_candidates(
        self,
        query: str,
        sorted_doc_ids: np.ndarray,
        sorted_doc_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._use_passage_rerank():
            return sorted_doc_ids, sorted_doc_scores
        if getattr(self, "llm_model", None) is None or len(sorted_doc_ids) <= 1:
            return sorted_doc_ids, sorted_doc_scores

        top_k = min(int(getattr(self.global_config, "hyperedge_passage_rerank_top_k", 0)), len(sorted_doc_ids))
        if top_k <= 1:
            return sorted_doc_ids, sorted_doc_scores

        max_chars = int(getattr(self.global_config, "hyperedge_passage_rerank_max_chars", 480))
        candidate_doc_ids = list(sorted_doc_ids[:top_k])
        candidate_doc_scores = list(sorted_doc_scores[:top_k])
        candidate_passages = []
        for local_idx, doc_idx in enumerate(candidate_doc_ids):
            passage_text = self.chunk_embedding_store.get_row(self.passage_node_keys[int(doc_idx)])["content"]
            candidate_passages.append(
                f"[{local_idx}] {self._truncate_passage_for_rerank(passage_text, max_chars)}"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You rerank candidate passages for multi-hop QA retrieval. "
                    "Return strict JSON with one field: ranked_indices. "
                    "The list must contain 0-based candidate indices ordered from most useful to least useful. "
                    "Prioritize passages that directly contain answer evidence or indispensable bridge evidence."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n"
                    f"Candidates:\n" + "\n".join(candidate_passages) + "\n"
                    "Return JSON only, for example: {\"ranked_indices\": [2, 0, 1]}"
                ),
            },
        ]

        try:
            response, _ = self.llm_model.infer(
                messages=messages,
                model=self.global_config.llm_name,
                max_completion_tokens=256,
                response_format={"type": "json_object"},
            )
            ranked_local_indices = self._parse_passage_rerank_response(response, top_k)
        except Exception:
            ranked_local_indices = list(range(top_k))

        reranked_candidate_ids = [candidate_doc_ids[idx] for idx in ranked_local_indices]
        reranked_candidate_scores = [candidate_doc_scores[idx] for idx in ranked_local_indices]

        if len(sorted_doc_ids) > top_k:
            reranked_candidate_ids.extend(list(sorted_doc_ids[top_k:]))
            reranked_candidate_scores.extend(list(sorted_doc_scores[top_k:]))

        return (
            np.array(reranked_candidate_ids, dtype=np.int64),
            np.array(reranked_candidate_scores, dtype=np.float32),
        )

    def initialize_graph(self):
        self._graph_pickle_filename = os.path.join(self.working_dir, "graph_hyperhippo.pickle")
        if getattr(self.global_config, "skip_graph_load_for_canonical_eval", False):
            return ig.Graph(directed=True)
        preloaded_graph = None
        if not self.global_config.force_index_from_scratch and os.path.exists(self._graph_pickle_filename):
            preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)
        if preloaded_graph is None:
            return ig.Graph(directed=True)
        return preloaded_graph

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = self.initialize_graph()
        self.proposition_extractor = PropositionExtractor(self.llm_model)

        self.hyperedge_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "hyperedge_embeddings"),
            self.global_config.embedding_batch_size,
            "hyperedge",
        )
        self.entity_record_store = RecordStore(os.path.join(self.working_dir, "entity_records"), "entity")
        self.hyperedge_record_store = RecordStore(os.path.join(self.working_dir, "hyperedge_records"), "hyperedge")
        self.passage_record_store = RecordStore(os.path.join(self.working_dir, "passage_records"), "passage")
        self.slot_record_store = RecordStore(os.path.join(self.working_dir, "slot_records"), "slot")
        self.bridge_path_record_store = RecordStore(os.path.join(self.working_dir, "bridge_path_records"), "bridge_path")
        self.support_trail_record_store = RecordStore(
            os.path.join(self.working_dir, "support_trail_records"),
            "support_trail",
        )
        self.support_trail_membership_record_store = RecordStore(
            os.path.join(self.working_dir, "support_trail_membership_records"),
            "support_trail_membership",
        )
        self.support_endpoint_record_store = RecordStore(
            os.path.join(self.working_dir, "support_endpoint_records"),
            "support_endpoint",
        )
        self.support_signature_sidecar_path = os.path.join(self.working_dir, "support_signature_sidecar.json")
        self.canonical_retrieval_interface_cache_path = os.path.join(
            self.working_dir,
            "canonical_retrieval_interface_cache.pkl",
        )

        self.query_router = HybridQueryRouter()
        self.diffuser = QueryConditionedDiffuser(min_edge_weight=self.global_config.diffusion_min_edge_weight)
        self.updater = ContinualUpdater()
        self.current_step = self._load_current_step()

    def _load_current_step(self) -> int:
        if hasattr(self.passage_record_store, "get_all_ref"):
            passage_records = self.passage_record_store.get_all_ref()
        else:
            passage_records = self.passage_record_store.get_all()
        if not passage_records:
            return 0
        return max(record.get("last_seen", 0) for record in passage_records.values())

    def _build_support_signature_sidecar(self) -> Dict[str, object]:
        return build_support_signature_sidecar_from_hyperedge_records(
            hyperedge_records=(
                self.hyperedge_record_store.get_all_ref()
                if hasattr(self.hyperedge_record_store, "get_all_ref")
                else self.hyperedge_record_store.get_all()
            ),
            passage_records=(
                self.passage_record_store.get_all_ref()
                if hasattr(self.passage_record_store, "get_all_ref")
                else self.passage_record_store.get_all()
            ),
        )

    def _build_slot_records(self, sidecar: Dict[str, object] | None = None) -> List[Dict]:
        return build_slot_records_from_hyperedge_records(
            hyperedge_records=(
                self.hyperedge_record_store.get_all_ref()
                if hasattr(self.hyperedge_record_store, "get_all_ref")
                else self.hyperedge_record_store.get_all()
            ),
            support_signature_sidecar=sidecar,
        )

    def _build_bridge_path_records(self) -> List[Dict]:
        global_config = getattr(self, "global_config", None)
        return build_bridge_path_records_from_hyperedge_records(
            hyperedge_records=(
                self.hyperedge_record_store.get_all_ref()
                if hasattr(self.hyperedge_record_store, "get_all_ref")
                else self.hyperedge_record_store.get_all()
            ),
            max_hyperedges_per_entity=int(getattr(global_config, "bridge_path_max_hyperedges_per_entity", 64)),
            max_hyperedges_per_source=int(getattr(global_config, "bridge_path_max_hyperedges_per_source", 64)),
            max_paths_per_fact=int(getattr(global_config, "bridge_path_max_paths_per_fact", 32)),
            max_pairs_per_bridge=int(getattr(global_config, "bridge_path_max_pairs_per_bridge", 64)),
            max_candidate_pairs=int(getattr(global_config, "bridge_path_max_candidate_pairs", 500000)),
            max_seed_source_neighbors_per_bridge=int(
                getattr(global_config, "bridge_path_max_seed_source_neighbors_per_bridge", 2)
            ),
            max_seed_entity_neighbors_per_bridge=int(
                getattr(global_config, "bridge_path_max_seed_entity_neighbors_per_bridge", 2)
            ),
        )

    def _get_bridge_path_records_for_trails(self) -> List[Dict]:
        bridge_path_records = getattr(self, "bridge_path_records", None)
        if bridge_path_records:
            return list(bridge_path_records.values()) if isinstance(bridge_path_records, dict) else list(bridge_path_records)
        if hasattr(self, "bridge_path_record_store") and self.bridge_path_record_store is not None:
            records = (
                self.bridge_path_record_store.get_all_ref()
                if hasattr(self.bridge_path_record_store, "get_all_ref")
                else self.bridge_path_record_store.get_all()
            )
            if records:
                return list(records.values())
        return self._build_bridge_path_records()

    def _support_trail_construction_policy(self) -> str:
        global_config = getattr(self, "global_config", None)
        return str(getattr(global_config, "support_trail_construction_policy", "default")).strip().lower()

    def _default_support_trail_kwargs(self) -> Dict:
        global_config = getattr(self, "global_config", None)
        return {
            "min_depth": int(getattr(global_config, "support_trail_min_depth", 2)),
            "max_depth": int(getattr(global_config, "support_trail_max_depth", 2)),
            "max_neighbors_per_fact": int(getattr(global_config, "support_trail_max_neighbors_per_fact", 4)),
            "max_trails_per_start_fact": int(getattr(global_config, "support_trail_max_trails_per_start_fact", 8)),
        }

    def _reserved_bridge_support_trail_kwargs(self) -> Dict:
        global_config = getattr(self, "global_config", None)
        max_trails = int(getattr(global_config, "support_trail_max_trails_per_start_fact", 8))
        return {
            "min_depth": int(getattr(global_config, "support_trail_min_depth", 2)),
            "max_depth": int(getattr(global_config, "support_trail_max_depth", 2)),
            "candidate_max_neighbors_per_fact": int(
                getattr(
                    global_config,
                    "support_trail_reserved_candidate_max_neighbors_per_fact",
                    getattr(global_config, "support_trail_candidate_max_neighbors_per_fact", 5),
                )
            ),
            "max_trails_per_start_fact": max_trails,
            "max_candidate_trails_per_start_fact": int(
                getattr(global_config, "support_trail_max_candidate_trails_per_start_fact", 32)
            ),
            "reserved_prefix_trails": int(
                getattr(global_config, "support_trail_reserved_prefix_trails", max(max_trails - 1, 0))
            ),
        }

    def _build_bridge_support_trail_records(self) -> List[Dict]:
        if self._support_trail_construction_policy() == "reserved_bridge":
            return build_reserved_bridge_support_trail_records_from_bridge_path_records(
                self._get_bridge_path_records_for_trails(),
                **self._reserved_bridge_support_trail_kwargs(),
            )
        return build_bridge_support_trail_records_from_bridge_path_records(
            self._get_bridge_path_records_for_trails(),
            **self._default_support_trail_kwargs(),
        )

    def _build_bridge_support_trail_membership_records(self) -> List[Dict]:
        if self._support_trail_construction_policy() == "reserved_bridge":
            return build_reserved_bridge_support_trail_membership_records_from_bridge_path_records(
                self._get_bridge_path_records_for_trails(),
                **self._reserved_bridge_support_trail_kwargs(),
            )
        return build_bridge_support_trail_membership_records_from_bridge_path_records(
            self._get_bridge_path_records_for_trails(),
            **self._default_support_trail_kwargs(),
        )

    def _build_bridge_support_endpoint_records(self) -> List[Dict]:
        if self._support_trail_construction_policy() == "reserved_bridge":
            return build_reserved_bridge_support_endpoint_records_from_bridge_path_records(
                self._get_bridge_path_records_for_trails(),
                **self._reserved_bridge_support_trail_kwargs(),
            )
        return build_bridge_support_endpoint_records_from_bridge_path_records(
            self._get_bridge_path_records_for_trails(),
            **self._default_support_trail_kwargs(),
        )

    def _save_support_signature_sidecar(self, sidecar: Dict[str, object]) -> None:
        save_support_signature_sidecar(self.support_signature_sidecar_path, sidecar)

    def _sync_slot_record_store(self, slot_records: List[Dict]) -> None:
        if not hasattr(self, "slot_record_store") or self.slot_record_store is None:
            self.slot_records = {str(record["hash_id"]): dict(record) for record in slot_records}
            return
        existing_ids = set(self.slot_record_store.get_all_ids())
        target_ids = {str(record["hash_id"]) for record in slot_records}
        stale_ids = sorted(existing_ids - target_ids)
        if stale_ids:
            self.slot_record_store.delete(stale_ids)
        if slot_records:
            self.slot_record_store.upsert(slot_records)

    def _sync_bridge_path_record_store(self, bridge_path_records: List[Dict]) -> None:
        if not hasattr(self, "bridge_path_record_store") or self.bridge_path_record_store is None:
            self.bridge_path_records = {str(record["hash_id"]): dict(record) for record in bridge_path_records}
            return
        existing_ids = set(self.bridge_path_record_store.get_all_ids())
        target_ids = {str(record["hash_id"]) for record in bridge_path_records}
        stale_ids = sorted(existing_ids - target_ids)
        if stale_ids:
            self.bridge_path_record_store.delete(stale_ids)
        if bridge_path_records:
            self.bridge_path_record_store.upsert(bridge_path_records)

    def _sync_support_trail_record_store(self, support_trail_records: List[Dict]) -> None:
        if not hasattr(self, "support_trail_record_store") or self.support_trail_record_store is None:
            self.support_trail_records = {str(record["hash_id"]): dict(record) for record in support_trail_records}
            return
        existing_ids = set(self.support_trail_record_store.get_all_ids())
        target_ids = {str(record["hash_id"]) for record in support_trail_records}
        stale_ids = sorted(existing_ids - target_ids)
        if stale_ids:
            self.support_trail_record_store.delete(stale_ids)
        if support_trail_records:
            self.support_trail_record_store.upsert(support_trail_records)

    def _sync_support_trail_membership_record_store(self, support_trail_membership_records: List[Dict]) -> None:
        if not hasattr(self, "support_trail_membership_record_store") or self.support_trail_membership_record_store is None:
            self.support_trail_membership_records = {
                str(record["hash_id"]): dict(record)
                for record in support_trail_membership_records
            }
            return
        existing_ids = set(self.support_trail_membership_record_store.get_all_ids())
        target_ids = {str(record["hash_id"]) for record in support_trail_membership_records}
        stale_ids = sorted(existing_ids - target_ids)
        if stale_ids:
            self.support_trail_membership_record_store.delete(stale_ids)
        if support_trail_membership_records:
            self.support_trail_membership_record_store.upsert(support_trail_membership_records)

    def _sync_support_endpoint_record_store(self, support_endpoint_records: List[Dict]) -> None:
        if not hasattr(self, "support_endpoint_record_store") or self.support_endpoint_record_store is None:
            self.support_endpoint_records = {
                str(record["hash_id"]): dict(record)
                for record in support_endpoint_records
            }
            return
        existing_ids = set(self.support_endpoint_record_store.get_all_ids())
        target_ids = {str(record["hash_id"]) for record in support_endpoint_records}
        stale_ids = sorted(existing_ids - target_ids)
        if stale_ids:
            self.support_endpoint_record_store.delete(stale_ids)
        if support_endpoint_records:
            self.support_endpoint_record_store.upsert(support_endpoint_records)

    def _load_support_signature_sidecar(self) -> Dict[str, object]:
        return load_support_signature_sidecar(self.support_signature_sidecar_path)

    def _build_canonical_interface_cache_signature(self) -> str | None:
        artifact_paths = [
            getattr(self, "_graph_pickle_filename", None),
            getattr(getattr(self, "entity_record_store", None), "filename", None),
            getattr(getattr(self, "hyperedge_record_store", None), "filename", None),
            getattr(getattr(self, "passage_record_store", None), "filename", None),
            getattr(getattr(self, "slot_record_store", None), "filename", None),
            getattr(getattr(self, "bridge_path_record_store", None), "filename", None),
            getattr(getattr(self, "support_trail_record_store", None), "filename", None),
            getattr(getattr(self, "support_trail_membership_record_store", None), "filename", None),
            getattr(getattr(self, "support_endpoint_record_store", None), "filename", None),
            getattr(self, "support_signature_sidecar_path", None),
        ]
        signature_rows = []
        for path in artifact_paths:
            if not path or not os.path.exists(path):
                return None
            stat = os.stat(path)
            signature_rows.append(
                {
                    "path": os.path.abspath(path),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        digest = hashlib.sha256(
            json.dumps(signature_rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return digest

    def _load_cached_canonical_retrieval_interface(self):
        cache_path = getattr(self, "canonical_retrieval_interface_cache_path", None)
        cache_signature = self._build_canonical_interface_cache_signature()
        if not cache_path or not cache_signature or not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "rb") as handle:
                payload = pickle.load(handle)
            if str(payload.get("cache_signature", "")) != cache_signature:
                logger.info("Canonical retrieval interface cache miss due to signature mismatch: %s", cache_path)
                return None
            interface_payload = payload.get("interface_payload")
            if not isinstance(interface_payload, dict):
                return None
            interface = load_canonical_interface_from_cache_payload(interface_payload, runtime=self)
            logger.info("Loaded canonical retrieval interface from cache: %s", cache_path)
            return interface
        except Exception:
            logger.exception("Failed to load canonical retrieval interface cache: %s", cache_path)
            return None

    def _save_cached_canonical_retrieval_interface(self, interface) -> None:
        cache_path = getattr(self, "canonical_retrieval_interface_cache_path", None)
        cache_signature = self._build_canonical_interface_cache_signature()
        if not cache_path or not cache_signature or interface is None:
            return
        payload = {
            "cache_signature": cache_signature,
            "interface_payload": build_canonical_interface_cache_payload(interface),
        }
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        logger.info("Saved canonical retrieval interface cache: %s", cache_path)

    def _slot_record_store_has_required_support_tiers(self) -> bool:
        if not hasattr(self, "slot_record_store") or self.slot_record_store is None:
            return False
        current_records = getattr(self.slot_record_store, "records", {})
        if not current_records:
            return False
        for record in current_records.values():
            if not all(field_name in record for field_name in SLOT_TIER_SUPPORT_FIELDS):
                return False
        return True

    def _initialize_query_retrieval_caches(self) -> None:
        self.query_to_embedding = {"triple": {}, "entity": {}, "hyperedge": {}, "passage": {}}
        self.hyperedge_query_view_cache = {}
        self.hyperedge_query_view_weights = {}
        self.hyperedge_score_bundle_cache = {}
        self.hyperedge_rerank_cache = {}

    def _repair_fact_substrate_objects(
        self,
        hyperedge_records: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        if not hyperedge_records:
            return hyperedge_records
        repair_missing_fact_ids_with_record_and_fact_store(
            hyperedge_records=hyperedge_records,
            hyperedge_record_store=self.hyperedge_record_store,
            fact_embedding_store=self.fact_embedding_store,
        )
        hyperedge_records = (
            self.hyperedge_record_store.get_all_ref()
            if hasattr(self.hyperedge_record_store, "get_all_ref")
            else self.hyperedge_record_store.get_all()
        )
        repair_missing_fact_embeddings_with_store(
            hyperedge_records=hyperedge_records,
            fact_embedding_store=self.fact_embedding_store,
        )
        return hyperedge_records

    def _load_prepare_support_state(
        self,
        *,
        sync_multiview_hyperedges: bool,
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
        hyperedge_records = (
            self.hyperedge_record_store.get_all_ref()
            if hasattr(self.hyperedge_record_store, "get_all_ref")
            else self.hyperedge_record_store.get_all()
        )
        passage_records = (
            self.passage_record_store.get_all_ref()
            if hasattr(self.passage_record_store, "get_all_ref")
            else self.passage_record_store.get_all()
        )
        hyperedge_records = self._repair_fact_substrate_objects(hyperedge_records)
        chunk_rows = (
            self.chunk_embedding_store.hash_id_to_row
            if hasattr(self.chunk_embedding_store, "hash_id_to_row")
            else self.chunk_embedding_store.get_all_id_to_rows()
        )
        passage_rows = {pid: chunk_rows[pid] for pid in passage_records if pid in chunk_rows}
        if sync_multiview_hyperedges:
            hyperedge_records = self._sync_multiview_hyperedge_embeddings(hyperedge_records, passage_rows)
        self.hyperedge_records = hyperedge_records
        self.passage_records = passage_records
        self.support_signature_sidecar = self._load_support_signature_sidecar()
        if (
            (
                not self.support_signature_sidecar.get("passage_inventories")
                or int(self.support_signature_sidecar.get("version", 0)) < SUPPORT_SIGNATURE_SIDECAR_VERSION
            )
            and (hyperedge_records or passage_records)
        ):
            self.support_signature_sidecar = self._build_support_signature_sidecar()
            self._save_support_signature_sidecar(self.support_signature_sidecar)
        if hyperedge_records:
            if hasattr(self, "slot_record_store") and self.slot_record_store is not None:
                current_slot_ids = set(self.slot_record_store.get_all_ids())
                expected_slot_ids = set(build_slot_record_ids_from_hyperedge_records(hyperedge_records))
                if (
                    expected_slot_ids != current_slot_ids
                    or not current_slot_ids
                    or not self._slot_record_store_has_required_support_tiers()
                ):
                    expected_slot_records = self._build_slot_records(self.support_signature_sidecar)
                    self._sync_slot_record_store(expected_slot_records)
                self.slot_records = (
                    self.slot_record_store.get_all_ref()
                    if hasattr(self.slot_record_store, "get_all_ref")
                    else self.slot_record_store.get_all()
                )
            else:
                expected_slot_records = self._build_slot_records(self.support_signature_sidecar)
                self.slot_records = {str(record["hash_id"]): dict(record) for record in expected_slot_records}
        else:
            self.slot_records = {}
        if hyperedge_records:
            expected_bridge_path_ids = set(build_bridge_path_record_ids_from_hyperedge_records(hyperedge_records))
            if hasattr(self, "bridge_path_record_store") and self.bridge_path_record_store is not None:
                current_bridge_path_ids = set(self.bridge_path_record_store.get_all_ids())
                if expected_bridge_path_ids != current_bridge_path_ids:
                    expected_bridge_path_records = self._build_bridge_path_records()
                    self._sync_bridge_path_record_store(expected_bridge_path_records)
                self.bridge_path_records = (
                    self.bridge_path_record_store.get_all_ref()
                    if hasattr(self.bridge_path_record_store, "get_all_ref")
                    else self.bridge_path_record_store.get_all()
                )
            else:
                expected_bridge_path_records = self._build_bridge_path_records()
                self.bridge_path_records = {
                    str(record["hash_id"]): dict(record)
                    for record in expected_bridge_path_records
                }
        else:
            self.bridge_path_records = {}
        if self.bridge_path_records:
            bridge_path_record_values = (
                list(self.bridge_path_records.values())
                if isinstance(self.bridge_path_records, dict)
                else list(self.bridge_path_records)
            )
            if self._support_trail_construction_policy() == "reserved_bridge":
                expected_support_trail_membership_ids = set(
                    build_reserved_bridge_support_trail_membership_record_ids_from_bridge_path_records(
                        bridge_path_record_values,
                        **self._reserved_bridge_support_trail_kwargs(),
                    )
                )
            else:
                expected_support_trail_membership_ids = set(
                    build_bridge_support_trail_membership_record_ids_from_bridge_path_records(
                        bridge_path_record_values,
                        **self._default_support_trail_kwargs(),
                    )
                )
            if (
                hasattr(self, "support_trail_membership_record_store")
                and self.support_trail_membership_record_store is not None
            ):
                current_support_trail_membership_ids = set(self.support_trail_membership_record_store.get_all_ids())
                if expected_support_trail_membership_ids != current_support_trail_membership_ids:
                    expected_support_trail_membership_records = self._build_bridge_support_trail_membership_records()
                    self._sync_support_trail_membership_record_store(expected_support_trail_membership_records)
                self.support_trail_membership_records = (
                    self.support_trail_membership_record_store.get_all_ref()
                    if hasattr(self.support_trail_membership_record_store, "get_all_ref")
                    else self.support_trail_membership_record_store.get_all()
                )
            else:
                expected_support_trail_membership_records = self._build_bridge_support_trail_membership_records()
                self.support_trail_membership_records = {
                    str(record["hash_id"]): dict(record)
                    for record in expected_support_trail_membership_records
                }
            if self._support_trail_construction_policy() == "reserved_bridge":
                expected_support_endpoint_ids = set(
                    build_reserved_bridge_support_endpoint_record_ids_from_bridge_path_records(
                        bridge_path_record_values,
                        **self._reserved_bridge_support_trail_kwargs(),
                    )
                )
            else:
                expected_support_endpoint_ids = set(
                    build_bridge_support_endpoint_record_ids_from_bridge_path_records(
                        bridge_path_record_values,
                        **self._default_support_trail_kwargs(),
                    )
                )
            if hasattr(self, "support_endpoint_record_store") and self.support_endpoint_record_store is not None:
                current_support_endpoint_ids = set(self.support_endpoint_record_store.get_all_ids())
                if expected_support_endpoint_ids != current_support_endpoint_ids:
                    expected_support_endpoint_records = self._build_bridge_support_endpoint_records()
                    self._sync_support_endpoint_record_store(expected_support_endpoint_records)
                self.support_endpoint_records = (
                    self.support_endpoint_record_store.get_all_ref()
                    if hasattr(self.support_endpoint_record_store, "get_all_ref")
                    else self.support_endpoint_record_store.get_all()
                )
            else:
                expected_support_endpoint_records = self._build_bridge_support_endpoint_records()
                self.support_endpoint_records = {
                    str(record["hash_id"]): dict(record)
                    for record in expected_support_endpoint_records
                }
            if self._support_trail_construction_policy() == "reserved_bridge":
                expected_support_trail_ids = set(
                    build_reserved_bridge_support_trail_record_ids_from_bridge_path_records(
                        bridge_path_record_values,
                        **self._reserved_bridge_support_trail_kwargs(),
                    )
                )
            else:
                expected_support_trail_ids = set(
                    build_bridge_support_trail_record_ids_from_bridge_path_records(
                        bridge_path_record_values,
                        **self._default_support_trail_kwargs(),
                    )
                )
            if hasattr(self, "support_trail_record_store") and self.support_trail_record_store is not None:
                current_support_trail_ids = set(self.support_trail_record_store.get_all_ids())
                if expected_support_trail_ids != current_support_trail_ids:
                    expected_support_trail_records = self._build_bridge_support_trail_records()
                    self._sync_support_trail_record_store(expected_support_trail_records)
                self.support_trail_records = (
                    self.support_trail_record_store.get_all_ref()
                    if hasattr(self.support_trail_record_store, "get_all_ref")
                    else self.support_trail_record_store.get_all()
                )
            else:
                expected_support_trail_records = self._build_bridge_support_trail_records()
                self.support_trail_records = {
                    str(record["hash_id"]): dict(record)
                    for record in expected_support_trail_records
                }
        else:
            self.support_trail_records = {}
            self.support_trail_membership_records = {}
            self.support_endpoint_records = {}
        self.support_passage_inventories = self.support_signature_sidecar.get("passage_inventories", {})
        return hyperedge_records, passage_records, chunk_rows

    def _prepare_fact_bridge_lookup(
        self,
        hyperedge_records: Dict[str, Dict],
        *,
        canonical_only: bool = False,
    ) -> None:
        if canonical_only:
            fact_node_keys = []
            seen_fact_ids = set()
            for record in hyperedge_records.values():
                fact_hash_id = record.get("fact_embedding_hash_id")
                if (
                    not fact_hash_id
                    or fact_hash_id in seen_fact_ids
                    or fact_hash_id not in self.fact_embedding_store.hash_id_to_row
                ):
                    continue
                fact_node_keys.append(fact_hash_id)
                seen_fact_ids.add(fact_hash_id)
            self.fact_node_keys = fact_node_keys
        else:
            self.fact_node_keys = list(self.fact_embedding_store.get_all_ids())
        self.fact_hash_id_to_hyperedge_keys = defaultdict(list)
        for hyperedge_id, record in hyperedge_records.items():
            fact_hash_id = record.get("fact_embedding_hash_id")
            if fact_hash_id:
                self.fact_hash_id_to_hyperedge_keys[fact_hash_id].append(hyperedge_id)

    def _prepare_canonical_query_scoring_objects(
        self,
        *,
        passage_records: Dict[str, Dict],
    ) -> None:
        self._prepare_canonical_query_scoring_objects_from_node_ids(
            entity_node_keys=self.entity_record_store.get_all_ids(),
            fact_node_keys=self.fact_node_keys,
            passage_node_keys=passage_records.keys(),
        )

    def _prepare_canonical_query_scoring_objects_from_node_ids(
        self,
        *,
        entity_node_keys,
        fact_node_keys,
        passage_node_keys,
    ) -> None:
        self.entity_node_keys = [
            node_key
            for node_key in entity_node_keys
            if node_key in self.entity_embedding_store.hash_id_to_row
        ]
        self.entity_embeddings = (
            np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys), dtype=np.float32)
            if self.entity_node_keys
            else np.array([], dtype=np.float32)
        )

        self.passage_node_keys = [
            node_key
            for node_key in passage_node_keys
            if node_key in self.chunk_embedding_store.hash_id_to_row
        ]
        self.passage_embeddings = (
            np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys), dtype=np.float32)
            if self.passage_node_keys
            else np.array([], dtype=np.float32)
        )

        self.fact_node_keys = [
            node_key
            for node_key in fact_node_keys
            if node_key in self.fact_embedding_store.hash_id_to_row
        ]
        self.fact_embeddings = (
            np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys), dtype=np.float32)
            if self.fact_node_keys
            else np.array([], dtype=np.float32)
        )
        self.hyperedge_node_keys = []
        self.hyperedge_embeddings = np.array([], dtype=np.float32)
        self.hyperedge_view_embeddings = np.array([], dtype=np.float32)
        self.hyperedge_pos_to_view_span = []

    def prepare_canonical_retrieval_interface(self):
        self._initialize_query_retrieval_caches()
        cached_interface = self._load_cached_canonical_retrieval_interface()
        if cached_interface is not None:
            self._prepare_canonical_query_scoring_objects_from_node_ids(
                entity_node_keys=list(cached_interface.entity_ids),
                fact_node_keys=list(cached_interface.fact_object_ids),
                passage_node_keys=list(cached_interface.passage_ids),
            )
            self.canonical_retrieval_interface = cached_interface
            self.canonical_interface_ready = True
            return
        hyperedge_records, passage_records, _ = self._load_prepare_support_state(
            sync_multiview_hyperedges=False,
        )
        self._prepare_fact_bridge_lookup(hyperedge_records, canonical_only=True)
        self._prepare_canonical_query_scoring_objects(passage_records=passage_records)
        logger.info("Building canonical retrieval interface from runtime state.")
        self.canonical_retrieval_interface = build_canonical_retrieval_interface_from_runtime(self)
        self._save_cached_canonical_retrieval_interface(self.canonical_retrieval_interface)
        self.canonical_interface_ready = True

    def save_openie_results(self, all_openie_info: List[dict]):
        if not all_openie_info:
            with open(self.openie_results_path, "w") as f:
                json.dump({"docs": [], "avg_ent_chars": 0, "avg_ent_words": 0}, f)
            return
        super().save_openie_results(all_openie_info)

    def merge_openie_results(
        self,
        all_openie_info: List[dict],
        chunks_to_save: Dict[str, dict],
        ner_results_dict: Dict[str, NerRawOutput],
        triple_results_dict: Dict[str, TripleRawOutput],
        proposition_results_dict: Dict[str, PropositionRawOutput] | None = None,
    ) -> List[dict]:
        merged = super().merge_openie_results(
            all_openie_info=all_openie_info,
            chunks_to_save=chunks_to_save,
            ner_results_dict=ner_results_dict,
            triple_results_dict=triple_results_dict,
        )
        if proposition_results_dict:
            self.merge_proposition_results(merged, proposition_results_dict)
        return merged

    @staticmethod
    def merge_proposition_results(
        all_openie_info: List[dict],
        proposition_results_dict: Dict[str, PropositionRawOutput],
    ) -> List[dict]:
        if not proposition_results_dict:
            return all_openie_info

        info_by_idx = {item.get("idx"): item for item in all_openie_info if "idx" in item}
        for chunk_id, proposition_output in proposition_results_dict.items():
            row = info_by_idx.get(chunk_id)
            if row is None:
                continue
            row["propositions"] = list(proposition_output.propositions or [])
        return all_openie_info

    def _build_entity_payloads(
        self,
        chunk_to_entity_ids: Dict[str, List[str]],
        step_id: int,
    ) -> List[Dict]:
        entity_to_sources: Dict[str, Set[str]] = defaultdict(set)
        all_entity_rows = self.entity_embedding_store.get_all_id_to_rows()

        for chunk_id, entity_ids in chunk_to_entity_ids.items():
            for entity_id in entity_ids:
                entity_to_sources[entity_id].add(chunk_id)

        payloads = []
        for entity_id, source_ids in entity_to_sources.items():
            row = all_entity_rows.get(entity_id)
            if row is None:
                continue
            canonical_name = row["content"]
            payloads.append(
                {
                    "hash_id": entity_id,
                    "canonical_name": canonical_name,
                    "aliases": [canonical_name],
                    "entity_type": "generic",
                    "source_ids": sorted(source_ids),
                    "support_count": len(source_ids),
                    "last_seen": step_id,
                }
            )
        return payloads

    def _build_passage_payloads(
        self,
        chunk_ids: List[str],
        chunk_to_hids: Dict[str, List[str]],
        chunk_to_entity_ids: Dict[str, List[str]],
        step_id: int,
    ) -> List[Dict]:
        chunk_rows = self.chunk_embedding_store.get_rows(chunk_ids)
        payloads = []
        for chunk_id in chunk_ids:
            row = chunk_rows[chunk_id]
            payloads.append(
                {
                    "hash_id": chunk_id,
                    "content": row["content"],
                    "entity_ids": sorted(set(chunk_to_entity_ids.get(chunk_id, []))),
                    "proposition_ids": sorted(set(chunk_to_hids.get(chunk_id, []))),
                    "last_seen": step_id,
                    "timestamp": step_id,
                    "reliability": 1.0,
                }
            )
        return payloads

    def _materialize_candidate_fact_embeddings(self, candidates) -> None:
        missing_fact_texts: Dict[str, str] = {}
        for candidate in candidates:
            fact_hash_id = str(getattr(candidate, "fact_embedding_hash_id", "")).strip()
            if not fact_hash_id or fact_hash_id in self.fact_embedding_store.hash_id_to_row:
                continue
            fact_text = build_proposition_fact_embedding_text(
                relation_type=str(getattr(candidate, "relation_type", "")),
                participant_texts=list(getattr(candidate, "participant_texts", [])),
                normalized_text=str(getattr(candidate, "normalized_text", "")),
                summary_text=str(getattr(candidate, "summary_text", "")),
            )
            if not fact_text:
                continue
            if compute_mdhash_id(fact_text, prefix="fact-") != fact_hash_id:
                continue
            missing_fact_texts[fact_hash_id] = fact_text
        if missing_fact_texts:
            self.fact_embedding_store.insert_strings(list(missing_fact_texts.values()))

    def _get_hyperedge_embedding_map(self, hyperedge_records: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        hyperedge_embedding_map = {}
        fact_hash_ids = []
        hyperedge_hash_ids = []
        embedding_mode = self.global_config.hyperedge_embedding_mode

        for record in hyperedge_records.values():
            fact_hash_id = record.get("fact_embedding_hash_id")
            hyperedge_hash_id = record.get("embedding_hash_id")
            if embedding_mode == "fact_reuse" and fact_hash_id in self.fact_embedding_store.hash_id_to_row:
                fact_hash_ids.append(fact_hash_id)
            if hyperedge_hash_id in self.hyperedge_embedding_store.hash_id_to_row:
                hyperedge_hash_ids.append(hyperedge_hash_id)

        fact_hash_to_embedding = {}
        if fact_hash_ids:
            fact_embeddings = self.fact_embedding_store.get_embeddings(fact_hash_ids)
            fact_hash_to_embedding = {hash_id: emb for hash_id, emb in zip(fact_hash_ids, fact_embeddings)}

        hyperedge_hash_to_embedding = {}
        if hyperedge_hash_ids:
            hyperedge_embeddings = self.hyperedge_embedding_store.get_embeddings(hyperedge_hash_ids)
            hyperedge_hash_to_embedding = {
                hash_id: emb for hash_id, emb in zip(hyperedge_hash_ids, hyperedge_embeddings)
            }

        for hyperedge_id, record in hyperedge_records.items():
            fact_hash_id = record.get("fact_embedding_hash_id")
            hyperedge_hash_id = record.get("embedding_hash_id")
            if embedding_mode == "fact_reuse" and fact_hash_id in fact_hash_to_embedding:
                hyperedge_embedding_map[hyperedge_id] = fact_hash_to_embedding[fact_hash_id]
            elif hyperedge_hash_id in hyperedge_hash_to_embedding:
                hyperedge_embedding_map[hyperedge_id] = hyperedge_hash_to_embedding[hyperedge_hash_id]
        return hyperedge_embedding_map

    def _build_record_contextual_embedding_text(
        self,
        record: Dict,
        passage_rows: Dict[str, Dict],
    ) -> Tuple[str, List[str]]:
        evidence_top_k = max(1, int(getattr(self.global_config, "hyperedge_embedding_evidence_top_k", 2)))
        evidence_max_chars = max(64, int(getattr(self.global_config, "hyperedge_embedding_evidence_max_chars", 240)))
        evidence_snippets = []
        for source_id in list(record.get("source_ids", []))[:evidence_top_k]:
            row = passage_rows.get(source_id)
            if row is None:
                continue
            snippet = extract_evidence_snippet(row.get("content", ""), max_chars=evidence_max_chars)
            if snippet and snippet not in evidence_snippets:
                evidence_snippets.append(snippet)

        embedding_text = build_proposition_embedding_text(
            summary_text=str(record.get("summary_text", "")),
            relation_type=str(record.get("relation_type", "")),
            participant_texts=list(record.get("participant_texts", [])),
            evidence_snippets=evidence_snippets,
            support_count=int(record.get("support_count", 1)),
        )
        return embedding_text, evidence_snippets

    def _build_record_hyperedge_view_texts(
        self,
        record: Dict,
        passage_rows: Dict[str, Dict],
    ) -> List[str]:
        _, evidence_snippets = self._build_record_contextual_embedding_text(record, passage_rows)
        return build_proposition_embedding_text_variants(
            summary_text=str(record.get("summary_text", "")),
            relation_type=str(record.get("relation_type", "")),
            participant_texts=list(record.get("participant_texts", [])),
            evidence_snippets=evidence_snippets,
            support_count=int(record.get("support_count", 1)),
        )

    def _sync_contextual_hyperedge_embeddings(
        self,
        hyperedge_records: Dict[str, Dict],
        passage_rows: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        if not self._use_contextual_hyperedge_text():
            return hyperedge_records
        if not hyperedge_records:
            return hyperedge_records

        updated_records = []
        texts_to_insert = []
        synced_records = {key: dict(value) for key, value in hyperedge_records.items()}

        for hyperedge_id, record in synced_records.items():
            embedding_text, evidence_snippets = self._build_record_contextual_embedding_text(record, passage_rows)
            embedding_hash_id = compute_mdhash_id(embedding_text, prefix="hyperedge-")
            record_needs_update = (
                record.get("embedding_text") != embedding_text
                or record.get("embedding_hash_id") != embedding_hash_id
                or list(record.get("evidence_snippets", [])) != evidence_snippets
            )
            if embedding_hash_id not in self.hyperedge_embedding_store.hash_id_to_row:
                texts_to_insert.append(embedding_text)
            if record_needs_update:
                record["embedding_text"] = embedding_text
                record["evidence_snippets"] = evidence_snippets
                record["embedding_hash_id"] = embedding_hash_id
                updated_records.append(record)
                synced_records[hyperedge_id] = record

        if texts_to_insert:
            self.hyperedge_embedding_store.insert_strings(list(dict.fromkeys(texts_to_insert)))
        if updated_records:
            self.hyperedge_record_store.upsert(updated_records)
        return synced_records

    def _sync_multiview_hyperedge_embeddings(
        self,
        hyperedge_records: Dict[str, Dict],
        passage_rows: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        if not self._use_multi_view_hyperedge_keys():
            return hyperedge_records
        if not hyperedge_records:
            return hyperedge_records

        updated_records = []
        texts_to_insert = []
        synced_records = {key: dict(value) for key, value in hyperedge_records.items()}

        for hyperedge_id, record in synced_records.items():
            view_texts = self._build_record_hyperedge_view_texts(record, passage_rows)
            view_hash_ids = [compute_mdhash_id(text, prefix="hyperedge-") for text in view_texts]
            record_needs_update = (
                list(record.get("embedding_texts", [])) != view_texts
                or list(record.get("embedding_hash_ids", [])) != view_hash_ids
            )
            for text, hash_id in zip(view_texts, view_hash_ids):
                if hash_id not in self.hyperedge_embedding_store.hash_id_to_row:
                    texts_to_insert.append(text)
            if record_needs_update:
                record["embedding_texts"] = view_texts
                record["embedding_hash_ids"] = view_hash_ids
                updated_records.append(record)
                synced_records[hyperedge_id] = record

        if texts_to_insert:
            self.hyperedge_embedding_store.insert_strings(list(dict.fromkeys(texts_to_insert)))
        if updated_records:
            self.hyperedge_record_store.upsert(updated_records)
        return synced_records

    def _resolve_hyperedge_embedding(self, record: Dict):
        embedding_mode = self.global_config.hyperedge_embedding_mode
        fact_hash_id = record.get("fact_embedding_hash_id")
        embedding_hash_id = record.get("embedding_hash_id")

        if (
            embedding_mode == "fact_reuse"
            and fact_hash_id in self.fact_embedding_store.hash_id_to_row
        ):
            return self.fact_embedding_store.get_embedding(fact_hash_id)
        if embedding_hash_id in self.hyperedge_embedding_store.hash_id_to_row:
            return self.hyperedge_embedding_store.get_embedding(embedding_hash_id)
        return None

    def _deduplicate_edge_records(self, edge_records: List[Dict]) -> List[Dict]:
        deduped = {}
        for record in edge_records:
            key = (record["src"], record["tgt"], record["edge_type"])
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = dict(record)
                continue
            existing["weight"] = max(existing["weight"], record["weight"])
            existing["support_count"] = max(existing.get("support_count", 1), record.get("support_count", 1))
            deduped[key] = existing
        return list(deduped.values())

    def rebuild_graph_from_memory(self):
        entity_records = self.entity_record_store.get_all()
        hyperedge_records = self.hyperedge_record_store.get_all()
        passage_records = self.passage_record_store.get_all()
        missing_entity_payloads = find_missing_entity_payloads_from_memory(
            entity_records=entity_records,
            hyperedge_records=hyperedge_records,
            passage_records=passage_records,
        )
        if missing_entity_payloads:
            self.entity_record_store.upsert(missing_entity_payloads.values())
            entity_records.update(missing_entity_payloads)
        repair_missing_entity_embeddings_with_store(
            entity_records=entity_records,
            hyperedge_records=hyperedge_records,
            passage_records=passage_records,
            entity_embedding_store=self.entity_embedding_store,
        )
        hyperedge_records = self._repair_fact_substrate_objects(hyperedge_records)
        chunk_rows = self.chunk_embedding_store.get_all_id_to_rows()
        passage_rows = {pid: chunk_rows[pid] for pid in passage_records if pid in chunk_rows}
        hyperedge_records = self._sync_contextual_hyperedge_embeddings(hyperedge_records, passage_rows)

        node_rows = build_node_rows(
            entity_records=entity_records,
            passage_rows=passage_rows,
            passage_records=passage_records,
            hyperedge_records=hyperedge_records,
            current_step=self.current_step,
            freshness_tau=self.global_config.freshness_tau,
        )

        self.graph = ig.Graph(directed=True)
        if node_rows:
            node_attr_lists = defaultdict(list)
            for node_id, row in node_rows.items():
                row = dict(row)
                row["name"] = node_id
                for key, value in row.items():
                    node_attr_lists[key].append(value)
            self.graph.add_vertices(n=len(node_rows), attributes=dict(node_attr_lists))

        entity_ids = [entity_id for entity_id in entity_records if entity_id in self.entity_embedding_store.hash_id_to_row]
        entity_embedding_map = {}
        if entity_ids:
            entity_embeddings = self.entity_embedding_store.get_embeddings(entity_ids)
            entity_embedding_map = {entity_id: emb for entity_id, emb in zip(entity_ids, entity_embeddings)}

        hyperedge_embedding_map = self._get_hyperedge_embedding_map(hyperedge_records)
        edge_records = build_edge_records(
            entity_records=entity_records,
            passage_records=passage_records,
            hyperedge_records=hyperedge_records,
            entity_embedding_map=entity_embedding_map,
            hyperedge_embedding_map=hyperedge_embedding_map,
            config=self.global_config,
        )
        edge_records = self._deduplicate_edge_records(edge_records)

        if edge_records and self.graph.vcount() > 0:
            current_node_ids = {vertex["name"] for vertex in self.graph.vs}
            valid_edges = []
            edge_attrs = defaultdict(list)
            for edge in edge_records:
                if edge["src"] not in current_node_ids or edge["tgt"] not in current_node_ids:
                    continue
                valid_edges.append((edge["src"], edge["tgt"]))
                for key, value in edge.items():
                    if key in {"src", "tgt"}:
                        continue
                    edge_attrs[key].append(value)
            if valid_edges:
                self.graph.add_edges(valid_edges, attributes=dict(edge_attrs))

        self.save_igraph()
        support_signature_sidecar = self._build_support_signature_sidecar()
        self._save_support_signature_sidecar(support_signature_sidecar)
        self._sync_slot_record_store(self._build_slot_records(support_signature_sidecar))
        self.ready_to_retrieve = False

    def _prepare_new_memory(
        self,
        chunk_ids: List[str],
        chunk_triples: List[List[Tuple[str, str, str]]],
        step_id: int,
        chunk_propositions: List[List[Dict]] | None = None,
    ):
        chunk_rows = self.chunk_embedding_store.get_rows(chunk_ids)
        chunk_id_to_text = {chunk_id: chunk_rows[chunk_id]["content"] for chunk_id in chunk_ids if chunk_id in chunk_rows}

        if self._use_proposition_hyperedges():
            candidates, chunk_to_hids, chunk_to_entity_ids = build_proposition_candidates_from_units(
                chunk_ids,
                chunk_propositions or [[] for _ in chunk_ids],
                step_id,
                chunk_id_to_text=chunk_id_to_text if self._use_contextual_hyperedge_text() else None,
                evidence_max_chars=int(getattr(self.global_config, "hyperedge_embedding_evidence_max_chars", 240)),
            )
            entity_nodes = sorted(
                {
                    participant_text
                    for candidate in candidates
                    for participant_text in candidate.participant_texts
                    if participant_text
                }
            )
        else:
            entity_nodes, _ = extract_entity_nodes(chunk_triples)
            candidates, chunk_to_hids, chunk_to_entity_ids = build_proposition_candidates(
                chunk_ids,
                chunk_triples,
                step_id,
                chunk_id_to_text=chunk_id_to_text if self._use_contextual_hyperedge_text() else None,
                evidence_max_chars=int(getattr(self.global_config, "hyperedge_embedding_evidence_max_chars", 240)),
            )

        if entity_nodes:
            self.entity_embedding_store.insert_strings(entity_nodes)
        if candidates:
            self._materialize_candidate_fact_embeddings(candidates)

        if candidates:
            if self.global_config.hyperedge_embedding_mode == "prop_only" or self._use_proposition_hyperedges():
                texts_to_insert = [candidate.embedding_text for candidate in candidates]
            else:
                texts_to_insert = [
                    candidate.embedding_text
                    for candidate in candidates
                    if candidate.fact_embedding_hash_id not in self.fact_embedding_store.hash_id_to_row
                ]
            if texts_to_insert:
                self.hyperedge_embedding_store.insert_strings(texts_to_insert)
        updated_hyperedge_records = upsert_propositions(
            candidates=candidates,
            existing_records=self.hyperedge_record_store.get_all(),
            merge_threshold=self.global_config.proposition_merge_threshold,
        )
        self.hyperedge_record_store.upsert(updated_hyperedge_records)

        entity_payloads = self._build_entity_payloads(chunk_to_entity_ids, step_id)
        passage_payloads = self._build_passage_payloads(chunk_ids, chunk_to_hids, chunk_to_entity_ids, step_id)
        self.updater.upsert_entity_records(entity_payloads, self.entity_record_store)
        self.updater.upsert_passage_records(passage_payloads, self.passage_record_store)

    def index(self, docs: List[str]):
        if self.global_config.openie_mode == "offline":
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()
        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k: chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self._use_proposition_hyperedges():
            ner_results_dict, _ = reformat_openie_results(all_openie_info)
            existing_info_by_idx = {item["idx"]: item for item in all_openie_info if "idx" in item}
            proposition_rows_to_process = {
                chunk_id: row
                for chunk_id, row in chunk_to_rows.items()
                if "propositions" not in existing_info_by_idx.get(chunk_id, {})
            }
            if proposition_rows_to_process:
                proposition_results_dict = self.proposition_extractor.batch_extract_propositions(
                    proposition_rows_to_process,
                    ner_results_dict,
                )
                self.merge_proposition_results(all_openie_info, proposition_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        _, triple_results_dict = reformat_openie_results(all_openie_info)
        proposition_results_dict = reformat_proposition_results(all_openie_info)

        if len(new_openie_rows) == 0:
            has_memory = bool(
                self.entity_record_store.get_all_ids()
                or self.hyperedge_record_store.get_all_ids()
                or self.passage_record_store.get_all_ids()
            )
            if not has_memory and len(chunk_to_rows) > 0:
                chunk_ids = list(chunk_to_rows.keys())
                chunk_triples = [
                    [text_processing(t) for t in triple_results_dict.get(chunk_id, TripleRawOutput(chunk_id, None, [], {})).triples]
                    for chunk_id in chunk_ids
                ]
                chunk_propositions = [
                    proposition_results_dict.get(chunk_id, PropositionRawOutput(chunk_id, None, [], {})).propositions
                    for chunk_id in chunk_ids
                ]
                self.current_step += 1
                self._prepare_new_memory(
                    chunk_ids,
                    chunk_triples,
                    step_id=self.current_step,
                    chunk_propositions=chunk_propositions,
                )
                self.rebuild_graph_from_memory()
            elif (self.graph.vcount() == 0 and has_memory) or (has_memory and self._use_contextual_hyperedge_text()):
                self.rebuild_graph_from_memory()
            return

        new_chunk_ids = list(new_openie_rows.keys())
        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in new_chunk_ids]
        chunk_propositions = [
            proposition_results_dict.get(chunk_id, PropositionRawOutput(chunk_id, None, [], {})).propositions
            for chunk_id in new_chunk_ids
        ]

        self.current_step += 1
        self._prepare_new_memory(
            new_chunk_ids,
            chunk_triples,
            step_id=self.current_step,
            chunk_propositions=chunk_propositions,
        )
        self.rebuild_graph_from_memory()

    def delete(self, docs_to_delete: List[str]):
        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]
        if not docs_to_delete:
            return

        chunk_ids_to_delete = [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete]

        all_openie_info, _ = self.load_existing_openie([])
        filtered_openie_info = [doc for doc in all_openie_info if doc["idx"] not in set(chunk_ids_to_delete)]
        self.save_openie_results(filtered_openie_info)

        self.updater.delete_passages(
            passage_ids=chunk_ids_to_delete,
            passage_record_store=self.passage_record_store,
            entity_record_store=self.entity_record_store,
            hyperedge_record_store=self.hyperedge_record_store,
        )
        self.chunk_embedding_store.delete(chunk_ids_to_delete)
        self.rebuild_graph_from_memory()

    def prepare_retrieval_objects(self):
        self._initialize_query_retrieval_caches()

        self.node_name_to_vertex_idx = {vertex["name"]: idx for idx, vertex in enumerate(self.graph.vs)}
        hyperedge_records, passage_records, _ = self._load_prepare_support_state(
            sync_multiview_hyperedges=True,
        )
        self._prepare_fact_bridge_lookup(hyperedge_records, canonical_only=False)

        self.entity_node_keys = [
            node_key
            for node_key in self.entity_record_store.get_all_ids()
            if node_key in self.node_name_to_vertex_idx and node_key in self.entity_embedding_store.hash_id_to_row
        ]
        self.entity_node_idxs = [self.node_name_to_vertex_idx[node_key] for node_key in self.entity_node_keys]
        self.entity_embeddings = (
            np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys), dtype=np.float32)
            if self.entity_node_keys
            else np.array([], dtype=np.float32)
        )

        self.hyperedge_node_keys = []
        hyperedge_embeddings = []
        hyperedge_view_embeddings = []
        hyperedge_pos_to_view_span = []
        for node_key in self.hyperedge_record_store.get_all_ids():
            if node_key not in self.node_name_to_vertex_idx:
                continue
            record = hyperedge_records.get(node_key)
            if record is None:
                continue
            embedding = self._resolve_hyperedge_embedding(record)
            if embedding is None:
                continue
            self.hyperedge_node_keys.append(node_key)
            hyperedge_embeddings.append(embedding)
            view_hash_ids = list(record.get("embedding_hash_ids", []))
            view_embeddings_for_record = []
            if self._use_multi_view_hyperedge_keys():
                for view_hash_id in view_hash_ids:
                    if view_hash_id in self.hyperedge_embedding_store.hash_id_to_row:
                        view_embeddings_for_record.append(self.hyperedge_embedding_store.get_embedding(view_hash_id))
            if not view_embeddings_for_record:
                view_embeddings_for_record = [embedding]
            start_idx = len(hyperedge_view_embeddings)
            hyperedge_view_embeddings.extend(view_embeddings_for_record)
            hyperedge_pos_to_view_span.append((start_idx, len(hyperedge_view_embeddings)))
        self.hyperedge_node_idxs = [self.node_name_to_vertex_idx[node_key] for node_key in self.hyperedge_node_keys]
        self.hyperedge_embeddings = (
            np.array(hyperedge_embeddings, dtype=np.float32)
            if hyperedge_embeddings
            else np.array([], dtype=np.float32)
        )
        self.hyperedge_view_embeddings = (
            np.array(hyperedge_view_embeddings, dtype=np.float32)
            if hyperedge_view_embeddings
            else np.array([], dtype=np.float32)
        )
        self.hyperedge_pos_to_view_span = hyperedge_pos_to_view_span

        self.passage_node_keys = [
            node_key
            for node_key in self.passage_record_store.get_all_ids()
            if node_key in self.node_name_to_vertex_idx and node_key in self.chunk_embedding_store.hash_id_to_row
        ]
        self.passage_node_idxs = [self.node_name_to_vertex_idx[node_key] for node_key in self.passage_node_keys]
        self.passage_embeddings = (
            np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys), dtype=np.float32)
            if self.passage_node_keys
            else np.array([], dtype=np.float32)
        )
        self.entity_node_key_to_pos = {node_key: idx for idx, node_key in enumerate(self.entity_node_keys)}
        self.passage_node_key_to_pos = {node_key: idx for idx, node_key in enumerate(self.passage_node_keys)}
        self.hyperedge_pos_to_entity_vertex_idxs: List[List[int]] = []
        self.hyperedge_pos_to_passage_vertex_idxs: List[List[int]] = []
        for hyperedge_key in self.hyperedge_node_keys:
            record = hyperedge_records.get(hyperedge_key, {})
            entity_vertex_idxs = []
            for entity_id in record.get("participant_ids", []):
                entity_pos = self.entity_node_key_to_pos.get(entity_id)
                if entity_pos is not None:
                    entity_vertex_idxs.append(self.entity_node_idxs[entity_pos])
            passage_vertex_idxs = []
            for passage_id in record.get("source_ids", []):
                passage_pos = self.passage_node_key_to_pos.get(passage_id)
                if passage_pos is not None:
                    passage_vertex_idxs.append(self.passage_node_idxs[passage_pos])
            self.hyperedge_pos_to_entity_vertex_idxs.append(entity_vertex_idxs)
            self.hyperedge_pos_to_passage_vertex_idxs.append(passage_vertex_idxs)
        self.fact_embeddings = (
            np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys), dtype=np.float32)
            if self.fact_node_keys
            else np.array([], dtype=np.float32)
        )

        self.node_type_ids = np.array(self.graph.vs["node_type_id"], dtype=np.int64) if self.graph.vcount() > 0 else np.array([], dtype=np.int64)
        self.edge_type_ids = np.array(self.graph.es["edge_type_id"], dtype=np.int64) if self.graph.ecount() > 0 else np.array([], dtype=np.int64)
        self.base_edge_weights = np.array(self.graph.es["weight"], dtype=np.float32) if self.graph.ecount() > 0 else np.array([], dtype=np.float32)
        self.edge_source_ids = np.array([edge.source for edge in self.graph.es], dtype=np.int64) if self.graph.ecount() > 0 else np.array([], dtype=np.int64)
        self.edge_target_ids = np.array([edge.target for edge in self.graph.es], dtype=np.int64) if self.graph.ecount() > 0 else np.array([], dtype=np.int64)
        cached_interface = self._load_cached_canonical_retrieval_interface()
        if cached_interface is not None:
            self.canonical_retrieval_interface = cached_interface
        else:
            logger.info("Building canonical retrieval interface from runtime state.")
            self.canonical_retrieval_interface = build_canonical_retrieval_interface_from_runtime(self)
            self._save_cached_canonical_retrieval_interface(self.canonical_retrieval_interface)
        self.canonical_interface_ready = True

        self.ready_to_retrieve = True

    def get_canonical_retrieval_interface(self):
        if not getattr(self, "canonical_interface_ready", False) or getattr(self, "canonical_retrieval_interface", None) is None:
            self.prepare_canonical_retrieval_interface()
        return getattr(self, "canonical_retrieval_interface", None)

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        if not hasattr(self, "hyperedge_query_view_cache"):
            self.hyperedge_query_view_cache = {}
        if not hasattr(self, "hyperedge_query_view_weights"):
            self.hyperedge_query_view_weights = {}
        if not hasattr(self, "hyperedge_score_bundle_cache"):
            self.hyperedge_score_bundle_cache = {}

        all_query_strings = []
        for query in queries:
            query_text = query.question if isinstance(query, QuerySolution) else query
            if (
                query_text not in self.query_to_embedding["entity"]
                or query_text not in self.query_to_embedding["hyperedge"]
                or query_text not in self.query_to_embedding["passage"]
            ):
                all_query_strings.append(query_text)
                self.hyperedge_score_bundle_cache.pop(query_text, None)

        all_query_strings = list(dict.fromkeys(all_query_strings))
        if not all_query_strings:
            return

        structured_query_instruction = get_query_instruction("query_to_fact")
        hyperedge_query_instruction = self._get_hyperedge_query_instruction()
        query_embeddings_for_structured = self.embedding_model.batch_encode(
            all_query_strings,
            instruction=structured_query_instruction,
            norm=True,
        )
        for query_text, embedding in zip(all_query_strings, query_embeddings_for_structured):
            self.query_to_embedding["triple"][query_text] = embedding
            self.query_to_embedding["entity"][query_text] = embedding

        if hyperedge_query_instruction == structured_query_instruction:
            for query_text, embedding in zip(all_query_strings, query_embeddings_for_structured):
                self.query_to_embedding["hyperedge"][query_text] = embedding
        else:
            if self._use_hyperedge_query_v2() or self._use_role_aware_hyperedge_query():
                hyperedge_query_texts = []
                query_to_span = {}
                query_view_weights = {}
                for query_text in all_query_strings:
                    query_views = self._get_hyperedge_query_views(query_text)
                    start_idx = len(hyperedge_query_texts)
                    hyperedge_query_texts.extend([str(view["query"]) for view in query_views])
                    query_to_span[query_text] = (start_idx, len(hyperedge_query_texts))
                    query_view_weights[query_text] = self._get_hyperedge_query_view_weights(query_views)

                query_embeddings_for_hyperedge = self.embedding_model.batch_encode(
                    hyperedge_query_texts,
                    instruction=hyperedge_query_instruction,
                    norm=True,
                )
                for query_text, (start_idx, end_idx) in query_to_span.items():
                    self.query_to_embedding["hyperedge"][query_text] = np.array(
                        query_embeddings_for_hyperedge[start_idx:end_idx],
                        dtype=np.float32,
                    )
                    self.hyperedge_query_view_weights[query_text] = np.array(
                        query_view_weights[query_text],
                        dtype=np.float32,
                    )
            else:
                query_embeddings_for_hyperedge = self.embedding_model.batch_encode(
                    all_query_strings,
                    instruction=hyperedge_query_instruction,
                    norm=True,
                )
                for query_text, embedding in zip(all_query_strings, query_embeddings_for_hyperedge):
                    self.query_to_embedding["hyperedge"][query_text] = embedding

        query_embeddings_for_passage = self.embedding_model.batch_encode(
            all_query_strings,
            instruction=get_query_instruction("query_to_passage"),
            norm=True,
        )
        for query_text, embedding in zip(all_query_strings, query_embeddings_for_passage):
            self.query_to_embedding["passage"][query_text] = embedding

    def get_entity_scores(self, query: str) -> np.ndarray:
        if len(self.entity_embeddings) == 0:
            return np.array([])
        query_embedding = self.query_to_embedding["entity"][query]
        scores = np.dot(self.entity_embeddings, query_embedding.T)
        scores = np.squeeze(scores) if scores.ndim == 2 else scores
        return self._normalize_score_vector(scores)

    def get_hyperedge_scores(self, query: str) -> np.ndarray:
        bundle = self._get_hyperedge_score_bundle(query)
        if self._use_hyperedge_query_v4():
            return np.array(bundle["primary"], dtype=np.float32)
        return np.array(bundle["aggregated"], dtype=np.float32)

    def get_passage_scores(self, query: str) -> np.ndarray:
        if len(self.passage_embeddings) == 0:
            return np.array([])
        query_embedding = self.query_to_embedding["passage"][query]
        scores = np.dot(self.passage_embeddings, query_embedding.T)
        scores = np.squeeze(scores) if scores.ndim == 2 else scores
        return self._normalize_score_vector(scores)

    def dense_passage_retrieval(self, query: str):
        query_doc_scores = self.get_passage_scores(query)
        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores

    def _topk_reset_component(self, scores: np.ndarray, node_idxs: List[int], top_k: int) -> np.ndarray:
        reset_component = np.zeros(self.graph.vcount(), dtype=np.float32)
        if len(scores) == 0 or len(node_idxs) == 0:
            return reset_component
        top_indices = np.argsort(scores)[::-1][: min(top_k, len(scores))]
        for score_idx in top_indices:
            reset_component[node_idxs[score_idx]] = float(scores[score_idx])
        component_sum = float(reset_component.sum())
        if component_sum > 0:
            reset_component /= component_sum
        return reset_component

    def _build_reranked_fact_hyperedge_reset_component(
        self,
        query: str,
        query_fact_scores: np.ndarray,
    ) -> np.ndarray:
        reset_component = np.zeros(self.graph.vcount(), dtype=np.float32)
        if not getattr(self.global_config, "use_reranked_fact_hyperedge_seed", False):
            return reset_component
        if len(query_fact_scores) == 0 or self.graph.vcount() == 0:
            return reset_component

        top_k_fact_indices, _, _ = self.rerank_facts(query, query_fact_scores)
        if len(top_k_fact_indices) == 0:
            return reset_component

        hyperedge_weights: Dict[str, float] = {}
        for fact_idx in top_k_fact_indices:
            if fact_idx < 0 or fact_idx >= len(self.fact_node_keys):
                continue
            fact_hash_id = self.fact_node_keys[fact_idx]
            fact_score = float(query_fact_scores[fact_idx]) if fact_idx < len(query_fact_scores) else 0.0
            for hyperedge_id in self.fact_hash_id_to_hyperedge_keys.get(fact_hash_id, []):
                if hyperedge_id not in self.node_name_to_vertex_idx:
                    continue
                hyperedge_weights[hyperedge_id] = max(hyperedge_weights.get(hyperedge_id, 0.0), fact_score)

        if not hyperedge_weights:
            return reset_component

        for hyperedge_id, score in hyperedge_weights.items():
            reset_component[self.node_name_to_vertex_idx[hyperedge_id]] = score

        if float(reset_component.sum()) <= 0:
            nonzero_hyperedges = list(hyperedge_weights.keys())
            uniform_weight = 1.0 / len(nonzero_hyperedges)
            for hyperedge_id in nonzero_hyperedges:
                reset_component[self.node_name_to_vertex_idx[hyperedge_id]] = uniform_weight

        return self._normalize_seed_component(reset_component)

    def _build_role_projection_from_hyperedge_scores(
        self,
        hyperedge_scores: np.ndarray,
        top_k: int,
    ) -> Dict[str, np.ndarray]:
        hyperedge_node_scores = np.zeros(self.graph.vcount(), dtype=np.float32)
        passage_node_scores = np.zeros(self.graph.vcount(), dtype=np.float32)
        entity_node_scores = np.zeros(self.graph.vcount(), dtype=np.float32)
        if len(hyperedge_scores) == 0 or self.graph.vcount() == 0:
            return {
                "hyperedge_node_scores": hyperedge_node_scores,
                "passage_node_scores": passage_node_scores,
                "entity_node_scores": entity_node_scores,
            }

        top_indices = np.argsort(hyperedge_scores)[::-1][: min(max(top_k, 1), len(hyperedge_scores))]
        for score_idx in top_indices:
            score = float(hyperedge_scores[score_idx])
            if score <= 0:
                continue
            hyperedge_node_scores[self.hyperedge_node_idxs[score_idx]] = max(
                hyperedge_node_scores[self.hyperedge_node_idxs[score_idx]],
                score,
            )
            for passage_node_idx in self.hyperedge_pos_to_passage_vertex_idxs[score_idx]:
                passage_node_scores[passage_node_idx] = max(passage_node_scores[passage_node_idx], score)
            for entity_node_idx in self.hyperedge_pos_to_entity_vertex_idxs[score_idx]:
                entity_node_scores[entity_node_idx] = max(entity_node_scores[entity_node_idx], score)

        return {
            "hyperedge_node_scores": self._normalize_score_vector(hyperedge_node_scores),
            "passage_node_scores": self._normalize_score_vector(passage_node_scores),
            "entity_node_scores": self._normalize_score_vector(entity_node_scores),
        }

    def _build_v4_role_conditioning(
        self,
        score_bundle: Dict[str, object],
        entity_scores: np.ndarray,
        passage_scores: np.ndarray,
    ) -> Dict[str, np.ndarray] | None:
        if not self._use_hyperedge_query_v4():
            return None
        if self.graph.vcount() == 0 or self.graph.ecount() == 0:
            return None

        primary_scores = np.array(score_bundle.get("primary", np.array([])), dtype=np.float32)
        if len(primary_scores) == 0:
            return None

        role_scores = score_bundle.get("role_scores", {})
        bridge_scores = np.array(role_scores.get("bridge", np.zeros_like(primary_scores)), dtype=np.float32)
        constraint_scores = np.array(role_scores.get("constraint", np.zeros_like(primary_scores)), dtype=np.float32)
        inverse_scores = np.array(role_scores.get("inverse", np.zeros_like(primary_scores)), dtype=np.float32)

        role_top_k = int(getattr(self.global_config, "hyperedge_query_v4_role_top_k", 12))
        primary_projection = self._build_role_projection_from_hyperedge_scores(primary_scores, role_top_k)
        bridge_projection = self._build_role_projection_from_hyperedge_scores(bridge_scores, role_top_k)
        constraint_projection = self._build_role_projection_from_hyperedge_scores(constraint_scores, role_top_k)
        inverse_projection = self._build_role_projection_from_hyperedge_scores(inverse_scores, role_top_k)
        bridge_reset_component = self._topk_reset_component(
            bridge_scores,
            self.hyperedge_node_idxs,
            min(role_top_k, max(1, getattr(self.global_config, "hyperedge_seed_top_k", role_top_k))),
        )

        primary_hyperedge_support = np.maximum(
            primary_projection["hyperedge_node_scores"],
            0.55 * inverse_projection["hyperedge_node_scores"],
        )
        primary_passage_support = np.maximum(
            primary_projection["passage_node_scores"],
            0.45 * inverse_projection["passage_node_scores"],
        )
        primary_entity_support = np.maximum(
            primary_projection["entity_node_scores"],
            0.45 * inverse_projection["entity_node_scores"],
        )
        passage_support = np.maximum(
            np.maximum(0.70 * primary_passage_support, constraint_projection["passage_node_scores"]),
            0.40 * bridge_projection["passage_node_scores"],
        )
        entity_support = np.maximum(
            np.maximum(0.55 * primary_entity_support, constraint_projection["entity_node_scores"]),
            0.25 * bridge_projection["entity_node_scores"],
        )

        adjusted_entity_scores = np.array(entity_scores, dtype=np.float32)
        if len(adjusted_entity_scores) == len(self.entity_node_idxs) and len(adjusted_entity_scores) > 0:
            entity_support_values = entity_support[self.entity_node_idxs]
            entity_score_boost = float(getattr(self.global_config, "hyperedge_query_v4_entity_score_boost", 0.30))
            adjusted_entity_scores = self._normalize_score_vector(
                adjusted_entity_scores * (1.0 + entity_score_boost * entity_support_values)
            )

        adjusted_passage_scores = np.array(passage_scores, dtype=np.float32)
        if len(adjusted_passage_scores) == len(self.passage_node_idxs) and len(adjusted_passage_scores) > 0:
            passage_support_values = passage_support[self.passage_node_idxs]
            passage_score_boost = float(getattr(self.global_config, "hyperedge_query_v4_passage_score_boost", 0.65))
            adjusted_passage_scores = self._normalize_score_vector(
                adjusted_passage_scores * (1.0 + passage_score_boost * passage_support_values)
            )

        edge_weight_boosts = np.ones_like(self.base_edge_weights, dtype=np.float32)
        source_primary_h = primary_hyperedge_support[self.edge_source_ids]
        target_primary_h = primary_hyperedge_support[self.edge_target_ids]
        source_bridge_h = bridge_projection["hyperedge_node_scores"][self.edge_source_ids]
        target_bridge_h = bridge_projection["hyperedge_node_scores"][self.edge_target_ids]
        source_constraint_h = constraint_projection["hyperedge_node_scores"][self.edge_source_ids]
        target_constraint_h = constraint_projection["hyperedge_node_scores"][self.edge_target_ids]
        source_passage_support = passage_support[self.edge_source_ids]
        target_passage_support = passage_support[self.edge_target_ids]
        source_entity_support = entity_support[self.edge_source_ids]
        target_entity_support = entity_support[self.edge_target_ids]

        hh_mask = self.edge_type_ids == EDGE_TYPE_TO_ID["hyperedge_hyperedge"]
        if np.any(hh_mask):
            hh_bridge_signal = np.maximum(source_bridge_h, target_bridge_h)
            hh_target_bridge_signal = (
                np.sqrt(np.maximum(source_primary_h, 0.0) * np.maximum(target_bridge_h, 0.0))
                + np.sqrt(np.maximum(source_bridge_h, 0.0) * np.maximum(target_primary_h, 0.0))
            )
            edge_weight_boosts[hh_mask] += (
                float(getattr(self.global_config, "hyperedge_query_v4_bridge_hh_boost", 1.10)) * hh_bridge_signal[hh_mask]
                + float(getattr(self.global_config, "hyperedge_query_v4_target_bridge_hh_boost", 0.70))
                * hh_target_bridge_signal[hh_mask]
            )

        hp_mask = self.edge_type_ids == EDGE_TYPE_TO_ID["hyperedge_passage"]
        if np.any(hp_mask):
            hp_target_signal = np.maximum(source_primary_h, target_primary_h)
            hp_bridge_signal = np.maximum(source_bridge_h, target_bridge_h)
            hp_constraint_signal = np.maximum.reduce(
                [source_constraint_h, target_constraint_h, source_passage_support, target_passage_support]
            )
            edge_weight_boosts[hp_mask] += (
                float(getattr(self.global_config, "hyperedge_query_v4_target_edge_boost", 0.55))
                * hp_target_signal[hp_mask]
                + 0.35 * float(getattr(self.global_config, "hyperedge_query_v4_target_edge_boost", 0.55))
                * hp_bridge_signal[hp_mask]
                + float(getattr(self.global_config, "hyperedge_query_v4_constraint_edge_boost", 0.65))
                * hp_constraint_signal[hp_mask]
            )

        eh_mask = self.edge_type_ids == EDGE_TYPE_TO_ID["entity_hyperedge"]
        if np.any(eh_mask):
            eh_target_signal = np.maximum(source_primary_h, target_primary_h)
            eh_constraint_signal = np.maximum.reduce(
                [source_constraint_h, target_constraint_h, source_entity_support, target_entity_support]
            )
            edge_weight_boosts[eh_mask] += (
                0.75 * float(getattr(self.global_config, "hyperedge_query_v4_target_edge_boost", 0.55))
                * eh_target_signal[eh_mask]
                + float(getattr(self.global_config, "hyperedge_query_v4_constraint_edge_boost", 0.65))
                * eh_constraint_signal[eh_mask]
            )

        ep_mask = self.edge_type_ids == EDGE_TYPE_TO_ID["entity_passage"]
        if np.any(ep_mask):
            ep_constraint_signal = np.maximum.reduce(
                [source_passage_support, target_passage_support, source_entity_support, target_entity_support]
            )
            edge_weight_boosts[ep_mask] += (
                0.50 * float(getattr(self.global_config, "hyperedge_query_v4_constraint_edge_boost", 0.65))
                * ep_constraint_signal[ep_mask]
            )

        edge_weight_boosts = np.clip(
            edge_weight_boosts,
            1.0,
            float(max(1.0, getattr(self.global_config, "hyperedge_query_v4_max_edge_boost", 3.0))),
        ).astype(np.float32)

        return {
            "adjusted_entity_scores": adjusted_entity_scores,
            "adjusted_passage_scores": adjusted_passage_scores,
            "bridge_hyperedge_reset_component": bridge_reset_component,
            "edge_weight_boosts": edge_weight_boosts,
        }

    def build_query_seeds_and_match_scores(
        self,
        entity_scores: np.ndarray,
        hyperedge_scores: np.ndarray,
        passage_scores: np.ndarray,
        router_output,
        reranked_hyperedge_reset: np.ndarray | None = None,
        reranked_fact_hyperedge_reset: np.ndarray | None = None,
        query_role_conditioning: Dict[str, np.ndarray] | None = None,
    ):
        entity_scores_for_seed = np.array(entity_scores, dtype=np.float32)
        passage_scores_for_seed = np.array(passage_scores, dtype=np.float32)
        if query_role_conditioning is not None:
            adjusted_entity_scores = query_role_conditioning.get("adjusted_entity_scores")
            adjusted_passage_scores = query_role_conditioning.get("adjusted_passage_scores")
            if adjusted_entity_scores is not None and len(adjusted_entity_scores) == len(entity_scores_for_seed):
                entity_scores_for_seed = np.array(adjusted_entity_scores, dtype=np.float32)
            if adjusted_passage_scores is not None and len(adjusted_passage_scores) == len(passage_scores_for_seed):
                passage_scores_for_seed = np.array(adjusted_passage_scores, dtype=np.float32)

        node_match_scores = np.zeros(self.graph.vcount(), dtype=np.float32)
        for score_idx, node_idx in enumerate(self.entity_node_idxs):
            if score_idx < len(entity_scores_for_seed):
                node_match_scores[node_idx] = float(entity_scores_for_seed[score_idx])
        for score_idx, node_idx in enumerate(self.hyperedge_node_idxs):
            if score_idx < len(hyperedge_scores):
                node_match_scores[node_idx] = float(hyperedge_scores[score_idx])
        for score_idx, node_idx in enumerate(self.passage_node_idxs):
            if score_idx < len(passage_scores_for_seed):
                node_match_scores[node_idx] = float(passage_scores_for_seed[score_idx])

        if self.global_config.freshness_as_node_prior and self.graph.vcount() > 0 and "freshness" in self.graph.vs.attributes():
            node_match_scores *= np.array(self.graph.vs["freshness"], dtype=np.float32)

        entity_reset = self._topk_reset_component(entity_scores_for_seed, self.entity_node_idxs, self.global_config.entity_seed_top_k)
        hyperedge_reset = self._topk_reset_component(
            hyperedge_scores,
            self.hyperedge_node_idxs,
            self.global_config.hyperedge_seed_top_k,
        )
        if query_role_conditioning is not None:
            bridge_hyperedge_reset_component = query_role_conditioning.get("bridge_hyperedge_reset_component")
            if (
                bridge_hyperedge_reset_component is not None
                and len(bridge_hyperedge_reset_component) == self.graph.vcount()
                and float(np.sum(bridge_hyperedge_reset_component)) > 0
            ):
                bridge_seed_mix = float(
                    min(max(getattr(self.global_config, "hyperedge_query_v4_bridge_seed_mix", 0.18), 0.0), 1.0)
                )
                hyperedge_reset = self._normalize_seed_component(
                    (1.0 - bridge_seed_mix) * hyperedge_reset
                    + bridge_seed_mix * np.array(bridge_hyperedge_reset_component, dtype=np.float32)
                )
        if reranked_hyperedge_reset is not None and float(reranked_hyperedge_reset.sum()) > 0:
            mix_weight = float(
                min(max(getattr(self.global_config, "hyperedge_rerank_seed_mix", 0.55), 0.0), 1.0)
            )
            hyperedge_reset = self._normalize_seed_component(
                (1.0 - mix_weight) * hyperedge_reset + mix_weight * reranked_hyperedge_reset
            )
        if reranked_fact_hyperedge_reset is not None and float(reranked_fact_hyperedge_reset.sum()) > 0:
            mix_weight = float(
                min(max(getattr(self.global_config, "reranked_fact_hyperedge_seed_mix", 0.65), 0.0), 1.0)
            )
            hyperedge_reset = self._normalize_seed_component(
                (1.0 - mix_weight) * hyperedge_reset + mix_weight * reranked_fact_hyperedge_reset
            )
        passage_reset = self._topk_reset_component(
            passage_scores_for_seed,
            self.passage_node_idxs,
            self.global_config.passage_seed_top_k,
        )

        reset_prob = (
            router_output.seed_mix["entity"] * entity_reset
            + router_output.seed_mix["hyperedge"] * hyperedge_reset
            + router_output.seed_mix["passage"] * passage_reset
        )
        if float(reset_prob.sum()) <= 0:
            if self.graph.vcount() == 0:
                return reset_prob, node_match_scores
            reset_prob = np.ones(self.graph.vcount(), dtype=np.float32) / self.graph.vcount()
        else:
            reset_prob /= float(reset_prob.sum())
        return reset_prob, node_match_scores

    def hypergraph_search(
        self,
        query: str,
        router_output,
        entity_scores: np.ndarray,
        hyperedge_scores: np.ndarray,
        passage_scores: np.ndarray,
        reranked_hyperedge_reset: np.ndarray | None = None,
        reranked_fact_hyperedge_reset: np.ndarray | None = None,
        query_role_conditioning: Dict[str, np.ndarray] | None = None,
    ):
        effective_passage_scores = np.array(passage_scores, dtype=np.float32)
        if query_role_conditioning is not None:
            adjusted_passage_scores = query_role_conditioning.get("adjusted_passage_scores")
            if adjusted_passage_scores is not None and len(adjusted_passage_scores) == len(effective_passage_scores):
                effective_passage_scores = np.array(adjusted_passage_scores, dtype=np.float32)

        if len(effective_passage_scores) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        if self.graph.vcount() == 0 or self.graph.ecount() == 0 or len(self.passage_node_idxs) == 0:
            sorted_doc_ids = np.argsort(effective_passage_scores)[::-1]
            sorted_doc_scores = effective_passage_scores[sorted_doc_ids.tolist()]
            return sorted_doc_ids, sorted_doc_scores, np.array([], dtype=np.float32)

        reset_prob, node_match_scores = self.build_query_seeds_and_match_scores(
            entity_scores=entity_scores,
            hyperedge_scores=hyperedge_scores,
            passage_scores=effective_passage_scores,
            router_output=router_output,
            reranked_hyperedge_reset=reranked_hyperedge_reset,
            reranked_fact_hyperedge_reset=reranked_fact_hyperedge_reset,
            query_role_conditioning=query_role_conditioning,
        )
        sorted_doc_ids, sorted_doc_scores, query_edge_weights = self.diffuser.run(
            graph=self.graph,
            reset_prob=reset_prob,
            base_edge_weights=self.base_edge_weights,
            edge_type_ids=self.edge_type_ids,
            edge_target_ids=self.edge_target_ids,
            node_type_ids=self.node_type_ids,
            node_match_scores=node_match_scores,
            edge_type_to_idx=EDGE_TYPE_TO_ID,
            node_type_to_idx=NODE_TYPE_TO_ID,
            router_output=router_output,
            passage_node_idxs=np.array(self.passage_node_idxs, dtype=np.int64),
            edge_weight_boosts=(
                query_role_conditioning.get("edge_weight_boosts")
                if query_role_conditioning is not None
                else None
            ),
            directed=True,
        )
        if len(sorted_doc_ids) == 0:
            sorted_doc_ids = np.argsort(effective_passage_scores)[::-1]
            sorted_doc_scores = effective_passage_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores, query_edge_weights

    def retrieve(self, queries: List[str], num_to_retrieve: int = None, gold_docs: List[List[str]] = None):
        retrieve_start_time = time.time()
        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config) if gold_docs is not None else None

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)
        retrieval_results = []

        for query in tqdm(queries, desc="Retrieving", total=len(queries)):
            search_start = time.time()
            entity_scores = self.get_entity_scores(query)
            hyperedge_score_bundle = self._get_hyperedge_score_bundle(query)
            hyperedge_scores = self.get_hyperedge_scores(query)
            hyperedge_rerank = self._rerank_hyperedge_candidates(query=query, score_bundle=hyperedge_score_bundle)
            if len(hyperedge_rerank.get("adjusted_scores", np.array([]))) == len(hyperedge_scores):
                hyperedge_scores = np.array(hyperedge_rerank["adjusted_scores"], dtype=np.float32)
                hyperedge_score_bundle = dict(hyperedge_score_bundle)
                hyperedge_score_bundle["primary"] = hyperedge_scores
            passage_scores = self.get_passage_scores(query)
            reranked_hyperedge_reset = hyperedge_rerank.get("reset_component")
            reranked_fact_hyperedge_reset = None
            if getattr(self.global_config, "use_reranked_fact_hyperedge_seed", False):
                query_fact_scores = self.get_fact_scores(query)
                reranked_fact_hyperedge_reset = self._build_reranked_fact_hyperedge_reset_component(
                    query=query,
                    query_fact_scores=query_fact_scores,
                )
            router_output = self.query_router.route(query, entity_scores, hyperedge_scores, passage_scores)
            query_role_conditioning = self._build_v4_role_conditioning(
                score_bundle=hyperedge_score_bundle,
                entity_scores=entity_scores,
                passage_scores=passage_scores,
            )
            sorted_doc_ids, sorted_doc_scores, _ = self.hypergraph_search(
                query=query,
                router_output=router_output,
                entity_scores=entity_scores,
                hyperedge_scores=hyperedge_scores,
                passage_scores=passage_scores,
                reranked_hyperedge_reset=reranked_hyperedge_reset,
                reranked_fact_hyperedge_reset=reranked_fact_hyperedge_reset,
                query_role_conditioning=query_role_conditioning,
            )
            sorted_doc_ids, sorted_doc_scores = self._rerank_passage_candidates(
                query=query,
                sorted_doc_ids=sorted_doc_ids,
                sorted_doc_scores=sorted_doc_scores,
            )
            self.ppr_time += time.time() - search_start

            top_k_docs = [
                self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                for idx in sorted_doc_ids[:num_to_retrieve]
            ]
            retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        self.all_retrieval_time += time.time() - retrieve_start_time

        if retrieval_recall_evaluator is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, _ = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs,
                retrieved_docs=[result.docs for result in retrieval_results],
                k_list=k_list,
            )
            return retrieval_results, overall_retrieval_result

        return retrieval_results
