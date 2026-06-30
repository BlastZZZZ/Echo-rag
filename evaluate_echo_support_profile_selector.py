#!/usr/bin/env python3
"""Evaluate ECHO-RAG with provenance-aware evidence set construction.

This is an explicit v2-style selector experiment, not the ECHO-RAG v1 mainline.
It keeps the same demand-channel retrieval outputs and the same demand-balanced
candidate pool as v1, but changes the selector input from a plain passage list
to passage text plus channel-derived demand provenance profiles.

The demand provenance profile is retrieval provenance only.  It is not gold
supervision, not a support label, and not a dataset-specific rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from dcr_ess import METHOD_NAME, candidate_stable_selected_topk, unique_preserve_order
from evaluate_ress_listwise_selector import (
    NOTHINK_EXTRA_BODY,
    build_candidate_docs,
    evidence_channels_from_trace_channels,
    passage_from_corpus_row,
    truncate_text,
)

from evaluate_role_aware_graph_entry import attach_r5, summarize_r5
from evaluate_role_coverage_reranker import (
    corpus_title_to_index,
    gold_doc_indices_for_row,
    load_baseline_rank_cache,
    load_json,
    load_per_query_rows,
    save_json,
    title_for_doc,
    unique_preserve_order as unique_docs,
)
from evaluate_rolewise_rcr import make_row, normalize_gold_answers
from rerun_qa_nothink_from_perquery import make_openai_client
from role_channel_graph_retrieval import RoleGraphChannel


EXTRA_BODY_MODES = {"auto", "qwen_disable", "none"}


def is_qwen_model(model: str) -> bool:
    return "qwen" in str(model or "").lower()


def resolve_extra_body(model: str, mode: str) -> Dict[str, Any]:
    normalized = str(mode or "auto").strip().lower()
    if normalized == "none":
        return {}
    if normalized == "qwen_disable":
        return dict(NOTHINK_EXTRA_BODY)
    if normalized == "auto" and is_qwen_model(model):
        return dict(NOTHINK_EXTRA_BODY)
    return {}


PROFILE_MODES = {
    "full",
    "no_profile",
    "shuffled",
    "channel_count_only",
    "rank_only",
    "no_role_descriptions",
    "no_title_group",
    "text_only_equal_length",
}

SELECTION_PROMPT_MODES = {
    "coverage",
    "direct_answer_first",
    "domain_answer_judge",
    "domain_answer_scorecard",
    "echo_residual_role_optimizer",
    "domain_symbol_route",
    "minimal_sufficient_set",
}

PASSAGE_EXCERPT_MODES = {
    "prefix",
    "query_focus",
}

TEXT_ONLY_EQUAL_LENGTH_PROFILE_LINES = [
    "profile_control: text-only equal-length placeholder; retrieval provenance withheld",
    "channel_count: masked",
    "best_channel_rank: masked",
    "exposed_by:",
    "- masked provenance placeholder for length control",
    "- masked provenance placeholder for length control",
    "- masked provenance placeholder for length control",
]

QUERY_FOCUS_STOPWORDS = {
    "about",
    "after",
    "also",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "type",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}

DIRECT_SYMBOL_CUES = {
    "abbreviated",
    "abbreviation",
    "attribute",
    "command",
    "configuration",
    "field",
    "function",
    "header",
    "identifier",
    "method",
    "metric",
    "member",
    "parameter",
    "routing",
    "statistics",
    "struct",
    "traffic",
    "variable",
}

CODE_LIKE_RE = re.compile(
    r"`[^`]+`|[A-Za-z_][A-Za-z0-9_]*\(\)|[A-Za-z]+_[A-Za-z0-9_]+|[A-Z][A-Z0-9_]{2,}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus_json", required=True)
    parser.add_argument("--per_query_report", required=True)
    parser.add_argument("--baseline_top200_cache", required=True)
    parser.add_argument("--role_channel_json", required=True)
    parser.add_argument("--save_json_path", required=True)
    parser.add_argument("--save_md_path", required=True)
    parser.add_argument("--per_query_variant", default=None)
    parser.add_argument("--variant_name", default="echo_demand_exposure_construction")
    parser.add_argument(
        "--profile_mode",
        choices=sorted(PROFILE_MODES),
        default="full",
        help="Demand provenance profile ablation mode.",
    )
    parser.add_argument(
        "--profile_shuffle_seed",
        type=int,
        default=1729,
        help="Base seed for deterministic within-query profile shuffling.",
    )
    parser.add_argument(
        "--selection_prompt_mode",
        choices=sorted(SELECTION_PROMPT_MODES),
        default="coverage",
        help="Selector prompt policy. 'coverage' preserves the original demand-coverage prompt.",
    )
    parser.add_argument("--limit_queries", type=int, default=-1)
    parser.add_argument("--reader_top_k", type=int, default=5)
    parser.add_argument("--candidate_top_k", type=int, default=12)
    parser.add_argument("--baseline_candidate_top_k", type=int, default=0)
    parser.add_argument("--max_candidates", type=int, default=24)
    parser.add_argument(
        "--candidate_admission",
        choices=[
            "channel_balanced",
            "demand_marginal_coverage",
            "demand_marginal_topk",
            "pareto_frontier",
            "pareto_repair",
            "residual_role_repair",
        ],
        default="channel_balanced",
        help="Candidate-pool materialization over demand-channel lists.",
    )
    parser.add_argument(
        "--demand_marginal_expansion_top_k",
        type=int,
        default=0,
        help="demand_marginal_coverage only: channel depth used to build the candidate universe. 0 reuses --candidate_top_k.",
    )
    parser.add_argument(
        "--demand_marginal_alpha",
        type=float,
        default=0.35,
        help="demand_marginal_coverage only: blend between best-channel relevance and uncovered-demand marginal gain.",
    )
    parser.add_argument(
        "--symbol_route_extra_candidates",
        type=int,
        default=0,
        help=(
            "For direct symbol/name lookup questions only, add this many lexical "
            "code-symbol candidates to the ECHO candidate pool. Default 0 keeps "
            "existing behavior unchanged."
        ),
    )
    parser.add_argument(
        "--symbol_route_baseline_top_k",
        type=int,
        default=0,
        help="For direct symbol/name lookup questions only, seed this many baseline ranked documents.",
    )
    parser.add_argument(
        "--symbol_route_max_candidates",
        type=int,
        default=0,
        help="Maximum candidate count after symbol-route augmentation. 0 reuses --max_candidates.",
    )
    parser.add_argument(
        "--symbol_route_prepend_candidates",
        action="store_true",
        help="Place symbol-route candidates before channel candidates for direct-symbol queries.",
    )
    parser.add_argument("--max_passage_chars", type=int, default=760)
    parser.add_argument(
        "--passage_excerpt_mode",
        choices=sorted(PASSAGE_EXCERPT_MODES),
        default="prefix",
        help="How candidate passage text is truncated for selector prompts.",
    )
    parser.add_argument("--selector_model", default="qwen3-32b-judge")
    parser.add_argument("--selector_base_url", default="http://localhost:8045/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument(
        "--extra_body_mode",
        choices=sorted(EXTRA_BODY_MODES),
        default="auto",
        help="Use Qwen enable_thinking=false only for Qwen models by default; use none for strict OpenAI calls.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0)
    parser.add_argument("--selector_cache_path", required=True)
    return parser.parse_args()


def load_cache(path: Path) -> MutableMapping[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, MutableMapping):
        raise ValueError(f"Selector cache must be a JSON object: {path}")
    return payload


def write_cache(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def channel_from_mapping(raw: Mapping[str, Any]) -> RoleGraphChannel:
    return RoleGraphChannel(
        role_id=str(raw.get("role_id") or ""),
        role_description=str(raw.get("role_description") or ""),
        retrieval_query=str(raw.get("retrieval_query") or ""),
        selected_fact_indices=[int(value) for value in raw.get("selected_fact_indices", []) or []],
        selected_passage_indices=[int(value) for value in raw.get("selected_passage_indices", []) or []],
        doc_indices=[int(value) for value in raw.get("doc_indices", []) or []],
        doc_scores=[float(value) for value in raw.get("doc_scores", []) or []],
        empty_reason=raw.get("empty_reason"),
        support_function=str(raw.get("support_function") or ""),
        retrieval_text=str(raw.get("retrieval_text") or ""),
        provenance_text=str(raw.get("provenance_text") or ""),
    )


def channels_for_trace(trace: Mapping[str, Any]) -> List[RoleGraphChannel]:
    raw_channels = trace.get("channels", [])
    if not isinstance(raw_channels, Sequence) or isinstance(raw_channels, (str, bytes)):
        return []
    return [channel_from_mapping(channel) for channel in raw_channels if isinstance(channel, Mapping)]


def channel_profile_for_candidates(
    *,
    channels: Sequence[Any],
    candidate_docs: Sequence[int],
    candidate_top_k: int,
) -> Dict[int, Dict[str, Any]]:
    candidate_set = {int(doc_idx) for doc_idx in candidate_docs}
    profile: Dict[int, Dict[str, Any]] = {
        int(doc_idx): {
            "channel_count": 0,
            "best_rank": None,
            "function_diversity": 0,
            "support_functions": [],
            "channels": [],
        }
        for doc_idx in candidate_docs
    }
    for channel_idx, channel in enumerate(channels):
        role_id = str(getattr(channel, "role_id", "") or f"channel_{channel_idx}")
        description = str(
            getattr(channel, "role_description", None)
            or getattr(channel, "retrieval_query", None)
            or role_id
        )
        support_function = str(getattr(channel, "support_function", "") or "").strip()
        function_key = support_function or role_id or f"channel_{channel_idx}"
        if support_function and support_function not in description:
            description = f"{support_function}: {description}"
        for rank0, raw_doc_idx in enumerate(unique_preserve_order(getattr(channel, "doc_indices", []) or [])[: int(candidate_top_k)]):
            doc_idx = int(raw_doc_idx)
            if doc_idx not in candidate_set:
                continue
            rank = int(rank0) + 1
            item = profile.setdefault(
                doc_idx,
                {
                    "channel_count": 0,
                    "best_rank": None,
                    "function_diversity": 0,
                    "support_functions": [],
                    "channels": [],
                },
            )
            item["channels"].append(
                {
                    "role_id": role_id,
                    "description": description,
                    "support_function": function_key,
                    "rank": rank,
                }
            )
            item["channel_count"] = int(item.get("channel_count", 0)) + 1
            best_rank = item.get("best_rank")
            item["best_rank"] = rank if best_rank is None else min(int(best_rank), rank)
    for item in profile.values():
        channels = item.get("channels", []) if isinstance(item, Mapping) else []
        functions = []
        seen = set()
        if isinstance(channels, Sequence) and not isinstance(channels, (str, bytes)):
            for channel_item in channels:
                if not isinstance(channel_item, Mapping):
                    continue
                function = str(channel_item.get("support_function") or "").strip()
                if function and function not in seen:
                    seen.add(function)
                    functions.append(function)
        item["support_functions"] = functions
        item["function_diversity"] = len(functions)
    return profile


def _profile_channels(profile: Mapping[str, Any] | Any) -> List[Mapping[str, Any]]:
    if not isinstance(profile, Mapping):
        return []
    raw_channels = profile.get("channels", [])
    if not isinstance(raw_channels, Sequence) or isinstance(raw_channels, (str, bytes)):
        return []
    return [item for item in raw_channels if isinstance(item, Mapping)]


def profile_role_ids(profile: Mapping[str, Any] | Any) -> List[str]:
    roles: List[str] = []
    for channel in _profile_channels(profile):
        role_id = str(channel.get("role_id") or "").strip()
        if role_id and role_id not in roles:
            roles.append(role_id)
    return roles


def profile_rank_sum(profile: Mapping[str, Any] | Any) -> int:
    total = 0
    for channel in _profile_channels(profile):
        try:
            total += int(channel.get("rank") or 10**6)
        except (TypeError, ValueError):
            total += 10**6
    return total


def role_ids_for_docs(docs: Sequence[int], *, profiles: Mapping[int, Mapping[str, Any]]) -> List[str]:
    roles: List[str] = []
    for raw_doc_idx in docs:
        for role_id in profile_role_ids(profiles.get(int(raw_doc_idx), {})):
            if role_id not in roles:
                roles.append(role_id)
    return roles


def best_role_rank_for_docs(docs: Sequence[int], *, profiles: Mapping[int, Mapping[str, Any]]) -> Dict[str, int]:
    best: Dict[str, int] = {}
    for raw_doc_idx in docs:
        for channel in _profile_channels(profiles.get(int(raw_doc_idx), {})):
            role_id = str(channel.get("role_id") or "").strip()
            if not role_id:
                continue
            try:
                rank = int(channel.get("rank") or 10**6)
            except (TypeError, ValueError):
                rank = 10**6
            best[role_id] = min(best.get(role_id, 10**6), rank)
    return best


def echo_residual_role_select_docs(
    *,
    candidate_docs: Sequence[int],
    backbone_docs: Sequence[int],
    profiles: Mapping[int, Mapping[str, Any]],
    top_k: int,
) -> Tuple[List[int], Dict[str, Any]]:
    budget = min(max(0, int(top_k)), len(unique_preserve_order([*backbone_docs, *candidate_docs])))
    if budget <= 0:
        return [], {
            "selected_ids": [],
            "selection_method": "echo_residual_role_optimizer",
            "objective": None,
        }

    incumbent = unique_preserve_order(backbone_docs)[:budget]
    for doc_idx in unique_preserve_order(candidate_docs):
        if len(incumbent) >= budget:
            break
        if int(doc_idx) not in incumbent:
            incumbent.append(int(doc_idx))

    universe = unique_preserve_order([*incumbent, *candidate_docs])
    candidate_positions = {int(doc_idx): pos for pos, doc_idx in enumerate(candidate_docs)}
    universe_positions = {int(doc_idx): pos for pos, doc_idx in enumerate(universe)}
    incumbent_set = set(int(doc_idx) for doc_idx in incumbent)
    incumbent_rank_by_role = best_role_rank_for_docs(incumbent, profiles=profiles)
    incumbent_role_set = set(incumbent_rank_by_role)
    incumbent_incidence = sum(len(profile_role_ids(profiles.get(int(doc_idx), {}))) for doc_idx in incumbent)

    best_docs: Tuple[int, ...] | None = None
    best_key: Tuple[int, int, int, int, int, int] | None = None
    for docs in combinations(universe, min(budget, len(universe))):
        selected_rank_by_role = best_role_rank_for_docs(docs, profiles=profiles)
        selected_role_set = set(selected_rank_by_role)
        if not incumbent_role_set.issubset(selected_role_set):
            continue
        if any(
            int(selected_rank_by_role.get(role_id, 10**6)) > int(incumbent_rank)
            for role_id, incumbent_rank in incumbent_rank_by_role.items()
        ):
            continue
        new_role_gain = len(selected_role_set - incumbent_role_set)
        rank_improvement_by_role = {
            role_id: int(incumbent_rank) - int(selected_rank_by_role.get(role_id, 10**6))
            for role_id, incumbent_rank in incumbent_rank_by_role.items()
        }
        rank_improvement_count = sum(1 for gain in rank_improvement_by_role.values() if int(gain) > 0)
        total_rank_improvement = sum(max(0, int(gain)) for gain in rank_improvement_by_role.values())
        if new_role_gain <= 0:
            continue
        displacement = sum(1 for doc_idx in docs if int(doc_idx) not in incumbent_set)
        incidence = sum(len(profile_role_ids(profiles.get(int(doc_idx), {}))) for doc_idx in docs)
        retrieval_prior = -sum(int(universe_positions.get(int(doc_idx), 10**6)) for doc_idx in docs)
        key = (
            new_role_gain,
            rank_improvement_count,
            total_rank_improvement,
            -displacement,
            incidence - incumbent_incidence,
            retrieval_prior,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_docs = tuple(int(doc_idx) for doc_idx in docs)

    if best_docs is None:
        best_docs = tuple(int(doc_idx) for doc_idx in incumbent)
        best_key = (
            0,
            0,
            0,
            0,
            0,
            -sum(int(universe_positions.get(int(doc_idx), 10**6)) for doc_idx in incumbent),
        )

    selected_set = set(best_docs)
    selected_docs = [int(doc_idx) for doc_idx in universe if int(doc_idx) in selected_set][:budget]
    selected_ids = [candidate_positions[int(doc_idx)] for doc_idx in selected_docs if int(doc_idx) in candidate_positions]
    return selected_docs, {
        "selected_ids": selected_ids,
        "selection_method": "echo_residual_role_optimizer",
        "objective": list(best_key),
        "objective_fields": [
            "new_echo_role_gain",
            "improved_echo_role_count",
            "total_echo_rank_improvement",
            "negative_incumbent_displacement",
            "echo_role_incidence_gain",
            "negative_candidate_position_sum",
        ],
        "objective_family": "echo_residual_role_preserving_set_selection",
        "incumbent_doc_indices": [int(doc_idx) for doc_idx in incumbent],
        "incumbent_role_ids": sorted(incumbent_role_set),
        "selected_role_ids": sorted(role_ids_for_docs(selected_docs, profiles=profiles)),
        "incumbent_best_role_ranks": {role_id: int(rank) for role_id, rank in sorted(incumbent_rank_by_role.items())},
        "selected_best_role_ranks": {
            role_id: int(rank)
            for role_id, rank in sorted(best_role_rank_for_docs(selected_docs, profiles=profiles).items())
        },
        "displaced_incumbent_doc_indices": [int(doc_idx) for doc_idx in incumbent if int(doc_idx) not in selected_set],
        "added_candidate_doc_indices": [int(doc_idx) for doc_idx in selected_docs if int(doc_idx) not in incumbent_set],
    }


def source_selected_docs(source_row: Mapping[str, Any], *, top_k: int) -> List[int]:
    for key in ("reader_doc_indices_topk", "retrieved_doc_indices_top5", "selected_doc_indices"):
        raw_docs = source_row.get(key)
        if isinstance(raw_docs, Sequence) and not isinstance(raw_docs, (str, bytes)):
            docs = unique_preserve_order(int(doc_idx) for doc_idx in raw_docs)
            if docs:
                return docs[: int(top_k)]
    return []


def residual_role_repair_candidate_docs(
    *,
    source_row: Mapping[str, Any],
    candidate_docs: Sequence[int],
    channels: Sequence[Any],
    candidate_top_k: int,
    expansion_top_k: int,
    max_candidates: int,
    reader_top_k: int,
) -> tuple[List[int], Dict[str, Any]]:
    selected_docs = source_selected_docs(source_row, top_k=int(reader_top_k))
    if not selected_docs:
        return list(unique_preserve_order(candidate_docs))[: int(max_candidates)], {
            "active": False,
            "reason": "missing_source_selected_docs",
        }
    selected_set = set(int(doc_idx) for doc_idx in selected_docs)
    depth = max(int(candidate_top_k), int(expansion_top_k))
    residual_docs: List[int] = []
    uncovered_roles: List[str] = []
    covered_roles: List[str] = []
    role_head_docs: List[int] = []
    for channel_idx, channel in enumerate(channels):
        role_id = str(getattr(channel, "role_id", "") or f"channel_{channel_idx}")
        docs = unique_preserve_order(getattr(channel, "doc_indices", []) or [])[:depth]
        if not docs:
            continue
        head_doc = int(docs[0])
        role_head_docs.append(head_doc)
        if head_doc in selected_set:
            covered_roles.append(role_id)
            continue
        uncovered_roles.append(role_id)
        residual_docs.append(head_doc)
    clean_candidates = unique_preserve_order(candidate_docs)[: int(max_candidates)]
    clean_residual_docs = unique_preserve_order(residual_docs)
    if not clean_residual_docs:
        return clean_candidates, {
            "active": True,
            "source_selected_docs": selected_docs,
            "covered_role_ids": covered_roles,
            "uncovered_role_ids": uncovered_roles,
            "role_head_docs": role_head_docs,
            "residual_docs": [],
            "added_residual_docs": [],
            "candidate_count_before": len(clean_candidates),
            "candidate_count_after": len(clean_candidates),
            "reason": "no_residual_docs",
        }
    combined = unique_preserve_order([*selected_docs, *clean_residual_docs, *candidate_docs])[: int(max_candidates)]
    return combined, {
        "active": True,
        "source_selected_docs": selected_docs,
        "covered_role_ids": covered_roles,
        "uncovered_role_ids": uncovered_roles,
        "role_head_docs": role_head_docs,
        "residual_docs": clean_residual_docs,
        "added_residual_docs": [doc_idx for doc_idx in clean_residual_docs if doc_idx not in set(candidate_docs)],
        "candidate_count_before": len(clean_candidates),
        "candidate_count_after": len(combined),
    }


def title_counts(candidate_docs: Sequence[int], corpus_records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for raw_doc_idx in unique_preserve_order(candidate_docs):
        doc_idx = int(raw_doc_idx)
        if 0 <= doc_idx < len(corpus_records):
            title = title_for_doc(corpus_records, doc_idx).strip()
            if title:
                key = title.casefold()
                counts[key] = counts.get(key, 0) + 1
    return counts


def query_focus_terms(question: str) -> List[str]:
    terms: List[str] = []
    for raw in re.findall(r"[A-Za-z0-9]+", str(question).casefold()):
        if len(raw) < 4 or raw in QUERY_FOCUS_STOPWORDS:
            continue
        variants = [raw]
        if raw.endswith("ies") and len(raw) > 5:
            variants.append(raw[:-3] + "y")
        if raw.endswith("ing") and len(raw) > 6:
            variants.append(raw[:-3])
        if raw.endswith("ed") and len(raw) > 5:
            variants.append(raw[:-2])
        if raw.endswith("s") and len(raw) > 4:
            variants.append(raw[:-1])
        for term in variants:
            if len(term) >= 4 and term not in QUERY_FOCUS_STOPWORDS and term not in terms:
                terms.append(term)
    return terms


def clean_code_like_term(raw: str) -> str:
    term = str(raw or "").strip().strip("`").strip()
    if term.endswith("()"):
        term = term[:-2]
    return term.strip()


def code_like_terms(question: str) -> List[str]:
    terms: List[str] = []
    for match in CODE_LIKE_RE.finditer(str(question or "")):
        term = clean_code_like_term(match.group(0))
        if len(term) < 2:
            continue
        if term not in terms:
            terms.append(term)
    return terms


def is_direct_symbol_question(question: str) -> bool:
    """Return true for lookup questions asking for code symbols, fields, or names.

    The route is intentionally narrow.  Generic "what is the name" wording is
    not sufficient unless the question also contains technical/code cues.
    """
    text = str(question or "")
    lowered = text.casefold()
    has_code_token = bool(code_like_terms(text))
    has_direct_name = any(
        phrase in lowered
        for phrase in (
            "what is the name",
            "what is the term",
            "term used",
            "what term refers",
            "what function",
            "what method",
            "what field",
            "what member",
            "what parameter",
            "what attribute",
            "what command",
            "what configuration",
        )
    )
    has_technical_cue = any(re.search(rf"\b{re.escape(cue)}\b", lowered) for cue in DIRECT_SYMBOL_CUES)
    has_what_is_lookup = bool(re.search(r"\bwhat\s+(is|are|was|were)\b", lowered))
    return bool(
        (has_code_token and (has_direct_name or has_what_is_lookup))
        or (has_direct_name and has_technical_cue)
        or (has_what_is_lookup and has_technical_cue)
    )


def compact_for_symbol_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()


def query_phrases_for_symbol_route(question: str) -> List[str]:
    terms = query_focus_terms(question)
    phrases: List[str] = []
    for width in (4, 3, 2):
        for offset in range(0, max(0, len(terms) - width + 1)):
            phrase = " ".join(terms[offset : offset + width])
            if len(phrase) >= 9 and phrase not in phrases:
                phrases.append(phrase)
    return phrases


def symbol_candidate_score(question: str, passage: str) -> Tuple[float, Dict[str, Any]]:
    """Score a candidate using only question and passage text, never gold labels."""
    compact_passage = compact_for_symbol_match(passage)
    raw_passage = str(passage or "")
    focus_terms = query_focus_terms(question)
    code_terms = code_like_terms(question)
    phrases = query_phrases_for_symbol_route(question)
    score = 0.0
    matched_terms: List[str] = []
    matched_phrases: List[str] = []
    matched_code_terms: List[str] = []
    for term in focus_terms:
        if term and term in compact_passage:
            score += 1.0
            matched_terms.append(term)
            if len(term) >= 8:
                score += 0.35
    for phrase in phrases:
        if phrase and phrase in compact_passage:
            score += 2.5
            matched_phrases.append(phrase)
    for term in code_terms:
        cleaned = clean_code_like_term(term)
        if not cleaned:
            continue
        pattern = re.escape(cleaned)
        if re.search(pattern, raw_passage, flags=re.IGNORECASE):
            score += 8.0
            matched_code_terms.append(cleaned)
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\(\)|[A-Za-z]+_[A-Za-z0-9_]+|[A-Z][A-Z0-9_]{2,}", raw_passage):
        score += min(2.0, 0.25 * len(matched_terms))
    return score, {
        "matched_terms": matched_terms,
        "matched_phrases": matched_phrases,
        "matched_code_terms": matched_code_terms,
    }


def symbol_route_candidate_docs(
    *,
    question: str,
    candidate_docs: Sequence[int],
    baseline_docs: Sequence[int],
    corpus_passages: Sequence[str],
    extra_candidates: int,
    baseline_top_k: int,
    max_candidates: int,
    prepend_candidates: bool,
) -> Tuple[List[int], Dict[str, Any]]:
    """Add lexical/code-symbol candidates for direct lookup questions only."""
    original = [int(doc_idx) for doc_idx in unique_preserve_order(candidate_docs)]
    if not is_direct_symbol_question(question):
        return original[: int(max_candidates)], {
            "symbol_route_active": False,
            "symbol_route_reason": "not_direct_symbol_question",
            "symbol_candidates": [],
        }
    symbol_docs: List[int] = []
    for doc_idx in unique_preserve_order(baseline_docs)[: max(0, int(baseline_top_k))]:
        symbol_docs.append(int(doc_idx))
    scored: List[Tuple[float, int, int, Dict[str, Any]]] = []
    for doc_idx, passage in enumerate(corpus_passages):
        score, details = symbol_candidate_score(question, passage)
        if score <= 0.0:
            continue
        scored.append((score, -len(str(passage or "")), int(doc_idx), details))
    scored.sort(reverse=True)
    for _score, _neg_len, doc_idx, _details in scored:
        if doc_idx not in symbol_docs:
            symbol_docs.append(doc_idx)
        if len(symbol_docs) >= max(0, int(baseline_top_k)) + max(0, int(extra_candidates)):
            break
    if prepend_candidates:
        combined = unique_preserve_order([*symbol_docs, *original])
    else:
        combined = unique_preserve_order([*original, *symbol_docs])
    max_total = max(1, int(max_candidates))
    kept = [int(doc_idx) for doc_idx in combined[:max_total]]
    return kept, {
        "symbol_route_active": True,
        "symbol_route_reason": "direct_symbol_question",
        "symbol_candidates": symbol_docs,
        "symbol_candidates_added": [doc_idx for doc_idx in symbol_docs if doc_idx not in original],
        "symbol_candidate_count_before": len(original),
        "symbol_candidate_count_after": len(kept),
    }


def query_focused_excerpt(text: str, question: str, max_chars: int) -> str:
    """Return a query-relevant passage window without using answers or gold labels."""
    passage = str(text or "")
    budget = int(max_chars)
    if budget <= 0:
        return ""
    if len(passage) <= budget:
        return passage
    terms = query_focus_terms(question)
    if not terms:
        return truncate_text(passage, budget)
    lowered = passage.casefold()
    starts = set()
    for term in terms:
        for match in re.finditer(re.escape(term), lowered):
            starts.add(max(0, min(match.start() - budget // 3, len(passage) - budget)))
    if not starts:
        return truncate_text(passage, budget)

    def score(start: int) -> Tuple[int, int, int]:
        window = lowered[start : start + budget]
        covered = sum(1 for term in terms if term in window)
        count = sum(window.count(term) for term in terms)
        return covered, count, -start

    best_start = max(starts, key=score)
    excerpt = passage[best_start : best_start + budget].strip()
    if best_start > 0:
        excerpt = "... " + excerpt
    if best_start + budget < len(passage):
        excerpt = excerpt + " ..."
    return excerpt


def passage_excerpt(text: str, question: str, max_chars: int, mode: str) -> str:
    if str(mode) == "prefix":
        return truncate_text(text, int(max_chars))
    if str(mode) == "query_focus":
        return query_focused_excerpt(text, question, int(max_chars))
    raise ValueError(f"Unknown passage_excerpt_mode: {mode}")


def demand_exposure_prompt(
    *,
    question: str,
    role_descriptions: Sequence[str],
    candidate_docs: Sequence[int],
    corpus_passages: Sequence[str],
    corpus_records: Sequence[Mapping[str, Any]],
    profiles: Mapping[int, Mapping[str, Any]],
    max_passage_chars: int,
    top_k: int,
    profile_mode: str = "full",
    selection_prompt_mode: str = "coverage",
    passage_excerpt_mode: str = "prefix",
) -> List[Dict[str, str]]:
    if str(selection_prompt_mode) not in SELECTION_PROMPT_MODES:
        raise ValueError(f"Unknown selection_prompt_mode: {selection_prompt_mode}")
    if str(passage_excerpt_mode) not in PASSAGE_EXCERPT_MODES:
        raise ValueError(f"Unknown passage_excerpt_mode: {passage_excerpt_mode}")
    role_text = "\n".join(f"- {role}" for role in role_descriptions if str(role).strip())
    title_count_by_key = title_counts(candidate_docs, corpus_records)
    candidate_blocks: List[str] = []
    include_title_group = str(profile_mode) != "no_title_group"
    for local_id, raw_doc_idx in enumerate(candidate_docs):
        doc_idx = int(raw_doc_idx)
        title = title_for_doc(corpus_records, doc_idx)
        title_key = title.strip().casefold()
        profile = profiles.get(doc_idx, {})
        channels = profile.get("channels", []) if isinstance(profile.get("channels"), list) else []
        channel_lines = []
        for item in channels:
            if not isinstance(item, Mapping):
                continue
            desc = str(item.get("description") or item.get("role_id") or "").strip()
            rank = int(item.get("rank", 0) or 0)
            if desc and rank > 0:
                channel_lines.append(f"- rank {rank}: {desc}")
        if not channel_lines:
            channel_lines.append("- not exposed within the profiled channel depth")
        profile_lines = [f"candidate_position: {local_id}"]
        if include_title_group:
            profile_lines.append(f"title_group_size: {title_count_by_key.get(title_key, 0)}")
        if str(profile_mode) == "text_only_equal_length":
            profile_lines.extend(TEXT_ONLY_EQUAL_LENGTH_PROFILE_LINES)
        elif str(profile_mode) == "no_profile":
            profile_lines.append("profile_ablation: demand-exposure profile hidden")
        else:
            profile_lines.extend(
                [
                    f"channel_count: {int(profile.get('channel_count', 0) or 0)}",
                    f"best_channel_rank: {profile.get('best_rank')}",
                    "exposed_by:",
                    *channel_lines,
                ]
            )
        profile_text = "\n".join(profile_lines)
        passage = passage_excerpt(corpus_passages[doc_idx], question, int(max_passage_chars), str(passage_excerpt_mode))
        if str(selection_prompt_mode) in {"domain_answer_judge", "domain_answer_scorecard"}:
            candidate_blocks.append(
                f"[{local_id}] title={title}\nPassage:\n{passage}\nDemand-exposure profile (optional tie-break only):\n{profile_text}"
            )
        else:
            candidate_blocks.append(
                f"[{local_id}] title={title}\nDemand-exposure profile:\n{profile_text}\nPassage:\n{passage}"
            )
    if str(selection_prompt_mode) in {"domain_answer_judge", "domain_answer_scorecard", "domain_symbol_route", "minimal_sufficient_set"}:
        system = "You are a strict QA evidence selector. Maximize answerability from the selected passages. Return only valid JSON."
    else:
        system = "You are a strict graph-exposure-aware multi-hop evidence-set constructor. Return only valid JSON."
    profile_guidance = "Use the demand-exposure profile as graph-retrieval provenance, not as ground truth. "
    if str(profile_mode) == "text_only_equal_length":
        profile_guidance = (
            "The profile blocks are neutral text-only length controls and contain no graph-retrieval provenance; "
            "select from the question and passage text only. "
        )
    if str(selection_prompt_mode) == "direct_answer_first":
        selection_policy = (
            "Selection policy:\n"
            "1. First select passages that directly answer the question or contain the most specific answer-bearing evidence.\n"
            "2. Do not replace a precise answer-bearing passage with broader background, index-like, neighboring, or merely related passages.\n"
            "3. Use demand-exposure provenance as a tie-breaker after textual answer support, not as ground truth.\n"
            "4. Cover complementary evidence demands only when the question genuinely requires multiple facts; for lookup-style questions, one precise answer passage is more valuable than broad coverage.\n"
            "5. Avoid redundant same-topic passages unless each adds necessary answer support."
        )
    elif str(selection_prompt_mode) == "domain_answer_judge":
        selection_policy = (
            "Selection policy:\n"
            "Silently judge each candidate against the exact question, not the general topic.\n"
            "Priority 1: DIRECT_ANSWER passages that explicitly name, define, quantify, or state the requested answer.\n"
            "Priority 2: NECESSARY_BRIDGE passages that are required to connect entities or conditions when the question truly needs multiple facts.\n"
            "Priority 3: SUPPORTING_DETAIL passages that add answer-relevant constraints.\n"
            "Lowest priority: BACKGROUND, index-like, neighboring, same-title, or merely related passages that do not help answer the exact question.\n"
            "For lookup-style questions, prefer the passage that contains the requested value over broader coverage. "
            "Use demand-exposure provenance only to break ties between candidates with the same textual support class. "
            "Do not select a passage just because it has high channel_count, early rank, or covers many demand hints."
        )
    elif str(selection_prompt_mode) == "domain_answer_scorecard":
        selection_policy = (
            "Selection policy:\n"
            "First, silently score every candidate against the exact question using these classes:\n"
            " DIRECT_ANSWER = explicitly states the requested answer value;\n"
            " ANSWER_ENTITY = names the requested entity/event/object and gives close local context;\n"
            " NECESSARY_BRIDGE = required to connect entities or conditions for a multi-fact question;\n"
            " SUPPORTING_DETAIL = relevant but not sufficient alone;\n"
            " BACKGROUND = broad, index-like, neighboring, same-title, or only topically related.\n"
            "Select DIRECT_ANSWER candidates before bridges or background. "
            "For lookup-style questions, one direct answer passage outranks several broad coverage passages. "
            "Only add NECESSARY_BRIDGE passages when the direct answer cannot be interpreted without them. "
            "Use demand-exposure provenance only as a tie-breaker inside the same class. "
            "The order of selected_ids is the reader order: put the strongest answer-bearing passage first."
        )
    elif str(selection_prompt_mode) == "domain_symbol_route":
        selection_policy = (
            "Selection policy for direct symbol/name lookup questions:\n"
            "1. Select passages that explicitly name the requested function, method, field, member, parameter, "
            "command, abbreviation, metric, or code-like term.\n"
            "2. Prefer passages where the requested symbol appears near definitional wording such as 'is', "
            "'called', 'refers to', 'returns', 'used to', 'method', 'field', or 'number of'.\n"
            "3. Do not select broad topical background, neighboring chunks, table-of-contents/index-like text, "
            "or same-subsystem passages when they do not contain the requested symbol/value.\n"
            "4. If several passages contain plausible symbols, order selected_ids with the most exact "
            "question-answering passage first for the reader.\n"
            "5. Add bridge or context passages only after the direct symbol/value passage is selected. "
            "Use demand-exposure provenance only as a tie-breaker inside the same textual support class."
        )
    elif str(selection_prompt_mode) == "minimal_sufficient_set":
        selection_policy = (
            "Selection policy:\n"
            "Build the smallest answer-sufficient support set within the fixed budget. "
            "First select passages that state the answer or bind an entity needed to identify the answer. "
            "Then add only passages that supply a missing bridge, comparison branch, constraint, or verification fact required by the question. "
            "Use demand-exposure provenance to identify which support function a passage may serve, while treating passage text as the authority. "
            "Avoid selecting two passages for the same support function when one passage already provides the needed evidence. "
            "If fewer than the budget are necessary, fill remaining ids with the strongest non-redundant supporting passages. "
            "Order selected_ids so the reader sees answer-bearing evidence before bridge or verification context."
        )
    else:
        selection_policy = (
            "A good set should cover complementary evidence demands, preserve early/high-profile support candidates when their text is relevant, "
            "avoid spending budget on non-supporting or redundant same-title passages, and not select answer-looking text without bridge/support context."
        )
    if str(selection_prompt_mode) in {"domain_answer_judge", "domain_answer_scorecard", "domain_symbol_route", "minimal_sufficient_set"}:
        user = (
            "Question:\n"
            f"{question}\n\n"
            "Candidate passages with optional demand-exposure profiles:\n"
            + "\n\n".join(candidate_blocks)
            + "\n\n"
            "Optional demand hints generated from the question. These are not a coverage checklist:\n"
            f"{role_text}\n\n"
            f"Select exactly {top_k} candidate ids as a fixed-budget answer-support evidence set. "
            + profile_guidance
            + selection_policy
            + " "
            "Return only JSON with schema {\"selected_ids\": [0, 1, 2, 3, 4]}."
        )
    else:
        user = (
            "Question:\n"
            f"{question}\n\n"
            "Evidence demands generated from the question:\n"
            f"{role_text}\n\n"
            "Candidate passages with demand-exposure profiles:\n"
            + "\n\n".join(candidate_blocks)
            + "\n\n"
            f"Select exactly {top_k} candidate ids as a fixed-budget answer-support evidence set. "
            + profile_guidance
            + selection_policy
            + " "
            "Return only JSON with schema {\"selected_ids\": [0, 1, 2, 3, 4]}."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def support_profile_prompt(**kwargs: Any) -> List[Dict[str, str]]:
    """Backward-compatible alias for older tests/scripts."""
    return demand_exposure_prompt(**kwargs)


def _copy_profile(profile: Mapping[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(dict(profile), ensure_ascii=False))


def apply_profile_mode(
    *,
    profiles: Mapping[int, Mapping[str, Any]],
    candidate_docs: Sequence[int],
    profile_mode: str,
    query_index: int,
    shuffle_seed: int,
) -> Dict[int, Dict[str, Any]]:
    """Apply profile ablations without changing candidate passages or order."""
    mode = str(profile_mode)
    if mode not in PROFILE_MODES:
        raise ValueError(f"Unknown profile_mode: {profile_mode}")
    docs = [int(doc_idx) for doc_idx in candidate_docs]
    normalized: Dict[int, Dict[str, Any]] = {
        doc_idx: _copy_profile(profiles.get(doc_idx, {}))
        for doc_idx in docs
    }
    if mode == "full" or mode == "no_title_group":
        return normalized
    if mode == "no_profile" or mode == "text_only_equal_length":
        return {
            doc_idx: {
                "channel_count": 0,
                "best_rank": None,
                "channels": [],
            }
            for doc_idx in docs
        }
    if mode == "shuffled":
        if len(docs) <= 1:
            return normalized
        keyed = []
        for doc_idx in docs:
            raw = f"{int(shuffle_seed)}:{int(query_index)}:{doc_idx}".encode("utf-8")
            keyed.append((hashlib.sha256(raw).hexdigest(), doc_idx))
        shuffled_docs = [doc_idx for _, doc_idx in sorted(keyed)]
        if shuffled_docs == docs:
            shuffled_docs = shuffled_docs[1:] + shuffled_docs[:1]
        return {
            target_doc: _copy_profile(normalized.get(source_doc, {}))
            for target_doc, source_doc in zip(docs, shuffled_docs)
        }
    if mode == "channel_count_only":
        return {
            doc_idx: {
                "channel_count": int(normalized.get(doc_idx, {}).get("channel_count", 0) or 0),
                "best_rank": None,
                "channels": [],
            }
            for doc_idx in docs
        }
    if mode == "rank_only":
        return {
            doc_idx: {
                "channel_count": 0,
                "best_rank": normalized.get(doc_idx, {}).get("best_rank"),
                "channels": [],
            }
            for doc_idx in docs
        }
    if mode == "no_role_descriptions":
        out: Dict[int, Dict[str, Any]] = {}
        for doc_idx in docs:
            profile = _copy_profile(normalized.get(doc_idx, {}))
            stripped_channels = []
            for item in profile.get("channels", []) if isinstance(profile.get("channels"), list) else []:
                if not isinstance(item, Mapping):
                    continue
                stripped_channels.append(
                    {
                        "role_id": item.get("role_id"),
                        "description": str(item.get("role_id") or "role"),
                        "rank": item.get("rank"),
                    }
                )
            profile["channels"] = stripped_channels
            out[doc_idx] = profile
        return out
    raise ValueError(f"Unhandled profile_mode: {profile_mode}")


def cache_key(
    messages: Sequence[Mapping[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    extra_body: Mapping[str, Any],
) -> str:
    raw = json.dumps(
        {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def is_context_length_error(exc: BaseException) -> bool:
    text = str(exc).casefold()
    return (
        "maximum context length" in text
        or "context length" in text
        or "reduce the length of the messages" in text
        or "requested" in text
        and "tokens" in text
        and "maximum" in text
    )


def parse_selected_ids(text: str, num_candidates: int, top_k: int) -> List[int]:
    raw = str(text or "").strip()
    try:
        payload: Any = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        payload = json.loads(match.group(0)) if match else {}
    values = payload.get("selected_ids") if isinstance(payload, Mapping) else []
    out: List[int] = []
    if isinstance(values, list):
        for value in values:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < int(num_candidates) and idx not in out:
                out.append(idx)
            if len(out) >= int(top_k):
                break
    return out


def call_selector(
    *,
    client: Any,
    messages: Sequence[Mapping[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    extra_body: Mapping[str, Any],
    retries: int,
    retry_wait_seconds: float,
) -> Tuple[str, Dict[str, Any]]:
    last_error: BaseException | None = None
    for attempt in range(max(1, int(retries))):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=dict(extra_body),
            )
            choice = response.choices[0]
            usage = getattr(response, "usage", None)
            return str(choice.message.content or ""), {
                "finish_reason": getattr(choice, "finish_reason", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            }
        except Exception as exc:
            last_error = exc
            if is_context_length_error(exc):
                break
            if attempt + 1 >= max(1, int(retries)):
                break
            time.sleep(float(retry_wait_seconds) * (2**attempt))
    raise RuntimeError(f"Demand-exposure constructor call failed after {retries} attempts: {last_error}") from last_error


def select_docs(
    *,
    client: Any,
    question: str,
    role_descriptions: Sequence[str],
    candidate_docs: Sequence[int],
    backbone_docs: Sequence[int],
    corpus_passages: Sequence[str],
    corpus_records: Sequence[Mapping[str, Any]],
    profiles: Mapping[int, Mapping[str, Any]],
    top_k: int,
    max_passage_chars: int,
    profile_mode: str,
    selection_prompt_mode: str,
    passage_excerpt_mode: str,
    model: str,
    temperature: float,
    max_tokens: int,
    extra_body: Mapping[str, Any],
    retries: int,
    retry_wait_seconds: float,
    cache: MutableMapping[str, Any],
    cache_path: Path,
) -> Tuple[List[int], Dict[str, Any]]:
    if str(selection_prompt_mode) == "echo_residual_role_optimizer":
        return echo_residual_role_select_docs(
            candidate_docs=candidate_docs,
            backbone_docs=backbone_docs,
            profiles=profiles,
            top_k=int(top_k),
        )
    passage_budget = max(1, int(max_passage_chars))
    min_passage_budget = min(240, passage_budget)
    while True:
        messages = demand_exposure_prompt(
            question=question,
            role_descriptions=role_descriptions,
            candidate_docs=candidate_docs,
            corpus_passages=corpus_passages,
            corpus_records=corpus_records,
            profiles=profiles,
            max_passage_chars=int(passage_budget),
            top_k=int(top_k),
            profile_mode=profile_mode,
            selection_prompt_mode=selection_prompt_mode,
            passage_excerpt_mode=passage_excerpt_mode,
        )
        key = cache_key(messages, model, temperature, max_tokens, extra_body)
        cached = cache.get(key)
        if cached is not None:
            break
        try:
            content, metadata = call_selector(
                client=client,
                messages=messages,
                model=model,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                extra_body=extra_body,
                retries=int(retries),
                retry_wait_seconds=float(retry_wait_seconds),
            )
        except RuntimeError as exc:
            if not is_context_length_error(exc) or passage_budget <= min_passage_budget:
                raise
            next_budget = max(min_passage_budget, min(passage_budget - 80, int(passage_budget * 0.85)))
            if next_budget >= passage_budget:
                next_budget = passage_budget - 1
            print(
                "[echo-demand-exposure] selector prompt exceeded context; "
                f"reducing max_passage_chars {passage_budget}->{next_budget}",
                flush=True,
            )
            passage_budget = int(next_budget)
            continue
        selected_ids = parse_selected_ids(content, len(candidate_docs), int(top_k))
        cached = {
            "content": content,
            "metadata": metadata,
            "selected_ids": selected_ids,
            "prompt_max_passage_chars": int(passage_budget),
        }
        cache[key] = cached
        write_cache(cache_path, cache)
        break
    selected_ids = [int(idx) for idx in cached.get("selected_ids", []) if 0 <= int(idx) < len(candidate_docs)]
    selected_docs = candidate_stable_selected_topk(candidate_docs, selected_ids, top_k=int(top_k))
    return selected_docs[: int(top_k)], dict(cached)


def make_md(payload: Mapping[str, Any]) -> str:
    summary = payload.get("summaries", {}).get(payload.get("variant_name"), {})
    diagnostics = payload.get("diagnostics", {}) if isinstance(payload.get("diagnostics"), Mapping) else {}
    config = payload.get("config", {}) if isinstance(payload.get("config"), Mapping) else {}
    return "\n".join(
        [
            "# ECHO Demand-Provenance-Aware Evidence Set Construction",
            "",
            "## Boundary",
            "",
            "This is an evidence-set construction experiment over fixed ECHO demand-channel retrieval outputs.",
            "It changes the evidence construction input only: passages are annotated with channel-derived demand provenance profiles.",
            "Candidate-pool materialization is controlled only by saved channel lists and the reported admission parameters.",
            "It does not rebuild graphs, rerun retrieval, train, tune, or use gold labels during selection.",
            "",
            "## Config",
            "",
            "```text",
            f"dataset: {payload.get('dataset')}",
            f"candidate_top_k: {config.get('candidate_top_k')}",
            f"candidate_admission: {config.get('candidate_admission')}",
            f"demand_marginal_expansion_top_k: {config.get('demand_marginal_expansion_top_k')}",
            f"demand_marginal_alpha: {config.get('demand_marginal_alpha')}",
            f"profile_candidate_top_k: {config.get('profile_candidate_top_k')}",
            f"max_candidates: {config.get('max_candidates')}",
            f"symbol_route_extra_candidates: {config.get('symbol_route_extra_candidates')}",
            f"symbol_route_baseline_top_k: {config.get('symbol_route_baseline_top_k')}",
            f"symbol_route_max_candidates: {config.get('symbol_route_max_candidates')}",
            f"symbol_route_prepend_candidates: {config.get('symbol_route_prepend_candidates')}",
            f"max_passage_chars: {config.get('max_passage_chars')}",
            f"passage_excerpt_mode: {config.get('passage_excerpt_mode')}",
            f"reader_top_k: {config.get('reader_top_k')}",
            f"profile_mode: {config.get('profile_mode')}",
            f"selection_prompt_mode: {config.get('selection_prompt_mode')}",
            f"limit_queries: {config.get('limit_queries')}",
            f"selector_model: {config.get('selector_model')}",
            "```",
            "",
            "## Retrieval",
            "",
            "| Variant | R@5 | dR@5 | Changed | Improved | Worsened |",
            "|---|---:|---:|---:|---:|---:|",
            (
                f"| {payload.get('variant_name')} | "
                f"{float(summary.get('selected_r5', 0.0)):.4f} | "
                f"{float(summary.get('delta_r5', 0.0)):+.4f} | "
                f"{int(summary.get('changed_query_count', 0))} | "
                f"{int(summary.get('improved_query_count', 0))} | "
                f"{int(summary.get('worsened_query_count', 0))} |"
            ),
            "",
            "## Diagnostics",
            "",
            "```json",
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    corpus_records = load_json(args.corpus_json)
    if not isinstance(corpus_records, list):
        raise ValueError(f"Corpus must be a list: {args.corpus_json}")
    corpus_passages = [passage_from_corpus_row(row) for row in corpus_records]
    title_to_index = corpus_title_to_index(corpus_records)
    per_query_variant, rows_by_qid = load_per_query_rows(args.per_query_report, args.per_query_variant)
    base_rank_cache = load_baseline_rank_cache(args.baseline_top200_cache, candidate_top_k=max(int(args.max_candidates), int(args.reader_top_k)))
    role_payload = load_json(args.role_channel_json)
    traces = role_payload.get("traces", []) if isinstance(role_payload, Mapping) else []
    if not isinstance(traces, list):
        raise ValueError("role_channel_json must contain a traces list")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip() if str(args.api_key).strip().upper() == "ENV" else args.api_key
    extra_body = resolve_extra_body(args.selector_model, args.extra_body_mode)
    client = make_openai_client(base_url=args.selector_base_url, api_key=api_key)
    cache_path = Path(args.selector_cache_path)
    cache = load_cache(cache_path)
    cache_entries_before = len(cache)

    baseline_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    selector_traces: List[Dict[str, Any]] = []
    processed_queries = 0
    symbol_route_query_count = 0
    symbol_route_added_candidate_count = 0
    for trace_idx, trace in enumerate(traces):
        if int(args.limit_queries) >= 0 and processed_queries >= int(args.limit_queries):
            break
        if not isinstance(trace, Mapping):
            continue
        try:
            qid = int(trace.get("query_index"))
        except (TypeError, ValueError):
            continue
        if qid not in rows_by_qid or qid not in base_rank_cache:
            continue
        source_row = rows_by_qid[qid]
        question = str(source_row.get("question") or trace.get("question") or "")
        gold_docs = gold_doc_indices_for_row(source_row, title_to_index)
        gold_answers = normalize_gold_answers(source_row.get("gold_answers"))
        baseline_docs = unique_docs(base_rank_cache[qid].get("doc_indices", []))
        baseline_topk = baseline_docs[: int(args.reader_top_k)]
        base_row = make_row(
            query_index=qid,
            question=question,
            gold_answers=gold_answers,
            gold_doc_indices=gold_docs,
            reader_doc_indices=baseline_topk,
            corpus_records=corpus_records,
        )
        baseline_rows.append(attach_r5(base_row, gold_docs, args.reader_top_k))

        channels = channels_for_trace(trace)
        role_descriptions = [
            str(channel.role_description or channel.retrieval_query)
            for channel in channels
            if str(channel.role_description or channel.retrieval_query).strip()
        ]
        candidate_docs = build_candidate_docs(
            baseline_docs=baseline_docs,
            channels=channels,
            candidate_top_k=int(args.candidate_top_k),
            baseline_candidate_top_k=int(args.baseline_candidate_top_k),
            max_candidates=int(args.max_candidates),
            candidate_admission=str(args.candidate_admission),
            demand_marginal_expansion_top_k=int(args.demand_marginal_expansion_top_k),
            demand_marginal_alpha=float(args.demand_marginal_alpha),
        )
        residual_role_repair_diag: Dict[str, Any] = {
            "active": False,
            "reason": "disabled",
        }
        if str(args.candidate_admission) == "residual_role_repair":
            candidate_docs, residual_role_repair_diag = residual_role_repair_candidate_docs(
                source_row=source_row,
                candidate_docs=candidate_docs,
                channels=channels,
                candidate_top_k=int(args.candidate_top_k),
                expansion_top_k=int(args.demand_marginal_expansion_top_k),
                max_candidates=int(args.max_candidates),
                reader_top_k=int(args.reader_top_k),
            )
        symbol_route_max_candidates = (
            int(args.symbol_route_max_candidates)
            if int(args.symbol_route_max_candidates) > 0
            else int(args.max_candidates)
        )
        symbol_route_diag: Dict[str, Any] = {
            "symbol_route_active": False,
            "symbol_route_reason": "disabled",
            "symbol_candidates": [],
        }
        if int(args.symbol_route_extra_candidates) > 0 or str(args.selection_prompt_mode) == "domain_symbol_route":
            candidate_docs, symbol_route_diag = symbol_route_candidate_docs(
                question=question,
                candidate_docs=candidate_docs,
                baseline_docs=baseline_docs,
                corpus_passages=corpus_passages,
                extra_candidates=int(args.symbol_route_extra_candidates),
                baseline_top_k=int(args.symbol_route_baseline_top_k),
                max_candidates=symbol_route_max_candidates,
                prepend_candidates=bool(args.symbol_route_prepend_candidates),
            )
        if bool(symbol_route_diag.get("symbol_route_active")):
            symbol_route_query_count += 1
            symbol_route_added_candidate_count += len(symbol_route_diag.get("symbol_candidates_added") or [])
        profile_candidate_top_k = int(args.candidate_top_k)
        if str(args.candidate_admission) in {"demand_marginal_coverage", "pareto_frontier", "pareto_repair", "residual_role_repair"} and int(args.demand_marginal_expansion_top_k) > 0:
            profile_candidate_top_k = max(profile_candidate_top_k, int(args.demand_marginal_expansion_top_k))
        raw_profiles = channel_profile_for_candidates(
            channels=channels,
            candidate_docs=candidate_docs,
            candidate_top_k=int(profile_candidate_top_k),
        )
        profiles = apply_profile_mode(
            profiles=raw_profiles,
            candidate_docs=candidate_docs,
            profile_mode=args.profile_mode,
            query_index=qid,
            shuffle_seed=int(args.profile_shuffle_seed),
        )
        selector_backbone_docs = (
            source_selected_docs(source_row, top_k=int(args.reader_top_k))
            if str(args.candidate_admission) == "residual_role_repair"
            else baseline_topk
        )
        if (
            str(args.candidate_admission) == "residual_role_repair"
            and residual_role_repair_diag.get("reason") == "no_residual_docs"
        ):
            selected_docs = source_selected_docs(source_row, top_k=int(args.reader_top_k))
            selected_ids = [candidate_docs.index(doc_idx) for doc_idx in selected_docs if doc_idx in candidate_docs]
            selector_payload = {
                "selected_ids": selected_ids,
                "metadata": {"finish_reason": "reused_original_echo_selection"},
                "selection_method": "residual_role_repair_noop_reuse_original",
            }
        else:
            selected_docs, selector_payload = select_docs(
                client=client,
                question=question,
                role_descriptions=role_descriptions,
                candidate_docs=candidate_docs,
                backbone_docs=selector_backbone_docs,
                corpus_passages=corpus_passages,
                corpus_records=corpus_records,
                profiles=profiles,
                top_k=int(args.reader_top_k),
                max_passage_chars=int(args.max_passage_chars),
                profile_mode=str(args.profile_mode),
                selection_prompt_mode=(
                    "domain_symbol_route"
                    if str(args.selection_prompt_mode) == "domain_symbol_route"
                    and bool(symbol_route_diag.get("symbol_route_active"))
                    else (
                        "domain_answer_judge"
                        if str(args.selection_prompt_mode) == "domain_symbol_route"
                        else str(args.selection_prompt_mode)
                    )
                ),
                passage_excerpt_mode=str(args.passage_excerpt_mode),
                model=args.selector_model,
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                extra_body=extra_body,
                retries=int(args.retries),
                retry_wait_seconds=float(args.retry_wait_seconds),
                cache=cache,
                cache_path=cache_path,
            )
        selected_row = make_row(
            query_index=qid,
            question=question,
            gold_answers=gold_answers,
            gold_doc_indices=gold_docs,
            reader_doc_indices=selected_docs,
            corpus_records=corpus_records,
        )
        selected_row["selector_variant"] = args.variant_name
        selected_row["candidate_count"] = len(candidate_docs)
        selected_rows.append(attach_r5(selected_row, gold_docs, args.reader_top_k))
        selector_traces.append(
            {
                "query_index": qid,
                "candidate_docs": candidate_docs,
                "selected_doc_indices": selected_docs,
                "selected_ids": selector_payload.get("selected_ids", []),
                "finish_reason": selector_payload.get("metadata", {}).get("finish_reason"),
                "prompt_max_passage_chars": selector_payload.get("prompt_max_passage_chars"),
                "selection_method": selector_payload.get("selection_method"),
                "selection_objective": selector_payload.get("objective"),
                "selection_objective_fields": selector_payload.get("objective_fields"),
                "selection_objective_family": selector_payload.get("objective_family"),
                "incumbent_doc_indices": selector_payload.get("incumbent_doc_indices"),
                "incumbent_role_ids": selector_payload.get("incumbent_role_ids"),
                "selected_role_ids": selector_payload.get("selected_role_ids"),
                "incumbent_best_role_ranks": selector_payload.get("incumbent_best_role_ranks"),
                "selected_best_role_ranks": selector_payload.get("selected_best_role_ranks"),
                "displaced_incumbent_doc_indices": selector_payload.get("displaced_incumbent_doc_indices"),
                "added_candidate_doc_indices": selector_payload.get("added_candidate_doc_indices"),
                "residual_role_repair": residual_role_repair_diag,
                "symbol_route": symbol_route_diag,
            }
        )
        processed_queries += 1
        if (trace_idx + 1) % 10 == 0:
            print(f"[echo-demand-exposure] processed {trace_idx + 1}/{len(traces)} cache_size={len(cache)}", flush=True)

    summary = summarize_r5(selected_rows, baseline_rows)
    payload: Dict[str, Any] = {
        "mode": "echo_demand_exposure_construction",
        "method_name": f"{METHOD_NAME} provenance-aware evidence set construction",
        "dataset": args.dataset,
        "variant_name": args.variant_name,
        "per_query_variant": per_query_variant,
        "summaries": {args.variant_name: summary},
        "variants": {
            "hipporag_v2": baseline_rows,
            args.variant_name: selected_rows,
        },
        "selector_traces": selector_traces,
        "diagnostics": {
            "num_queries": len(selected_rows),
            "cache_entries_before": int(cache_entries_before),
            "cache_entries_after": int(len(cache)),
            "cache_entries_added": int(len(cache) - cache_entries_before),
            "symbol_route_query_count": int(symbol_route_query_count),
            "symbol_route_added_candidate_count": int(symbol_route_added_candidate_count),
        },
        "config": {
            "reader_top_k": int(args.reader_top_k),
            "candidate_top_k": int(args.candidate_top_k),
            "candidate_admission": str(args.candidate_admission),
            "demand_marginal_expansion_top_k": int(args.demand_marginal_expansion_top_k),
            "demand_marginal_alpha": float(args.demand_marginal_alpha),
            "profile_candidate_top_k": (
                max(int(args.candidate_top_k), int(args.demand_marginal_expansion_top_k))
                if str(args.candidate_admission) in {"demand_marginal_coverage", "pareto_frontier", "pareto_repair", "residual_role_repair"}
                and int(args.demand_marginal_expansion_top_k) > 0
                else int(args.candidate_top_k)
            ),
            "baseline_candidate_top_k": int(args.baseline_candidate_top_k),
            "max_candidates": int(args.max_candidates),
            "symbol_route_extra_candidates": int(args.symbol_route_extra_candidates),
            "symbol_route_baseline_top_k": int(args.symbol_route_baseline_top_k),
            "symbol_route_max_candidates": int(args.symbol_route_max_candidates),
            "symbol_route_prepend_candidates": bool(args.symbol_route_prepend_candidates),
            "max_passage_chars": int(args.max_passage_chars),
            "passage_excerpt_mode": str(args.passage_excerpt_mode),
            "limit_queries": int(args.limit_queries),
            "profile_mode": str(args.profile_mode),
            "profile_shuffle_seed": int(args.profile_shuffle_seed),
            "selection_prompt_mode": str(args.selection_prompt_mode),
            "selector_model": args.selector_model,
            "selector_base_url": args.selector_base_url,
            "extra_body_mode": args.extra_body_mode,
            "extra_body": extra_body,
            "temperature": float(args.temperature),
            "max_tokens": int(args.max_tokens),
        },
        "source": {
            "corpus_json": args.corpus_json,
            "per_query_report": args.per_query_report,
            "baseline_top200_cache": args.baseline_top200_cache,
            "role_channel_json": args.role_channel_json,
        },
    }
    return payload


def main() -> None:
    args = parse_args()
    payload = run(args)
    save_json(args.save_json_path, payload)
    Path(args.save_md_path).write_text(make_md(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
