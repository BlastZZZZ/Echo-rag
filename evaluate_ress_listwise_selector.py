#!/usr/bin/env python3
"""Evaluate ECHO-RAG with an LLM listwise evidence-set selector.

The selector sees a fixed evidence pool from channel-conditioned graph
retrieval, then selects the final top-k evidence set in one listwise decision.
It does not use gold labels, dataset-specific lexical rules, or graph
rebuilding.  Historical result files may still use RCR-RESS or DCR-ESS names;
paper-facing artifacts should use ECHO-RAG terminology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from dcr_ess import (
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_MAX_CANDIDATES,
    EvidenceChannel,
    METHOD_NAME,
    assert_echo_rag_clean_mainline,
    build_channel_balanced_evidence_pool,
    build_cross_channel_admitted_evidence_pool,
    build_demand_marginal_coverage_evidence_pool,
    build_demand_marginal_topk_evidence_pool,
    build_pareto_frontier_evidence_pool,
    build_pareto_repair_evidence_pool,
    build_terminal_channel_admitted_evidence_pool,
    candidate_stable_selected_topk,
    provenance_summary,
    unique_preserve_order,
)
from evaluate_role_coverage_reranker import (
    corpus_title_to_index,
    gold_doc_indices_for_row,
    load_baseline_rank_cache,
    load_json,
    load_per_query_rows,
    load_roles,
    save_json,
)
from evaluate_role_aware_graph_entry import attach_r5, summarize_r5
from evaluate_rolewise_rcr import make_row, normalize_gold_answers
from rerun_qa_nothink_from_perquery import make_openai_client
from role_channel_graph_retrieval import RoleGraphChannel


NOTHINK_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus_json", required=True)
    parser.add_argument("--per_query_report", required=True)
    parser.add_argument("--baseline_top200_cache", required=True)
    parser.add_argument("--role_channel_json", required=True)
    parser.add_argument(
        "--roles_json_path",
        default=None,
        help="Optional role cache used by terminal_bound candidate admission.",
    )
    parser.add_argument("--save_json_path", required=True)
    parser.add_argument("--save_md_path", required=True)
    parser.add_argument("--per_query_variant", default=None)
    parser.add_argument("--variant_name", default="dcr_ess_listwise")
    parser.add_argument("--reader_top_k", type=int, default=5)
    parser.add_argument("--candidate_top_k", type=int, default=DEFAULT_CANDIDATE_TOP_K)
    parser.add_argument(
        "--candidate_admission",
        choices=[
            "channel_balanced",
            "cross_channel",
            "terminal_bound",
            "demand_marginal_coverage",
            "demand_marginal_topk",
            "pareto_frontier",
            "pareto_repair",
            "residual_role_repair",
        ],
        default="channel_balanced",
        help="Candidate-pool construction. The default is the clean ECHO-RAG channel-balanced pool.",
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
        "--admission_base_top_k",
        type=int,
        default=8,
        help="cross_channel only: shallow pool depth preserved before admitting deeper consensus candidates.",
    )
    parser.add_argument(
        "--admission_min_channel_support",
        type=int,
        default=2,
        help="cross_channel only: minimum number of evidence channels required for a deep candidate.",
    )
    parser.add_argument(
        "--terminal_admission_limit",
        type=int,
        default=4,
        help="terminal_bound only: maximum number of deep terminal-channel candidates to append.",
    )
    parser.add_argument("--baseline_candidate_top_k", type=int, default=0)
    parser.add_argument("--max_candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--max_passage_chars", type=int, default=900)
    parser.add_argument(
        "--include_candidate_provenance",
        action="store_true",
        help="Annotate each candidate with the role channels that retrieved it.",
    )
    parser.add_argument("--selector_model", default="qwen3-32b-judge")
    parser.add_argument("--selector_base_url", default="http://localhost:8045/v1")
    parser.add_argument(
        "--selector_objective",
        choices=["evidence_set_coverage", "answer_resolving"],
        default="evidence_set_coverage",
        help=(
            "Selector objective. The default is the clean ECHO-RAG evidence-set "
            "coverage objective; answer_resolving is kept as an explicit ablation."
        ),
    )
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0)
    parser.add_argument("--selector_cache_path", required=True)
    parser.add_argument(
        "--allow_ablation_variants",
        action="store_true",
        help="Allow non-mainline ablations such as head seeds, admission variants, provenance prompts, or answer_resolving.",
    )
    args = parser.parse_args()
    try:
        validate_clean_mainline_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def validate_clean_mainline_args(args: argparse.Namespace) -> None:
    """Reject ablation knobs unless the caller explicitly opts into ablations."""
    if bool(getattr(args, "allow_ablation_variants", False)):
        return
    assert_echo_rag_clean_mainline(
        candidate_admission=getattr(args, "candidate_admission", "channel_balanced"),
        baseline_candidate_top_k=int(getattr(args, "baseline_candidate_top_k", 0)),
        include_candidate_provenance=bool(getattr(args, "include_candidate_provenance", False)),
        selector_objective=getattr(args, "selector_objective", "evidence_set_coverage"),
        construction="selector_stable",
        reader_top_k=int(getattr(args, "reader_top_k", 5)),
    )


def passage_from_corpus_row(row: Mapping[str, Any]) -> str:
    title = str(row.get("title") or row.get("Title") or "").strip()
    raw_text = row.get("text") if "text" in row else row.get("Text")
    if isinstance(raw_text, list):
        text = " ".join(str(item) for item in raw_text).strip()
    else:
        text = str(raw_text or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def truncate_text(text: str, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def build_candidate_docs(
    *,
    baseline_docs: Sequence[int],
    channels: Sequence[Any],
    candidate_top_k: int,
    baseline_candidate_top_k: int,
    max_candidates: int,
    candidate_admission: str = "channel_balanced",
    admission_base_top_k: int = 8,
    admission_min_channel_support: int = 2,
    terminal_channel_ids: Sequence[str] | None = None,
    terminal_admission_limit: int = 4,
    demand_marginal_expansion_top_k: int = 0,
    demand_marginal_alpha: float = 0.35,
) -> List[int]:
    evidence_channels = evidence_channels_from_trace_channels(channels)
    seed_docs = unique_preserve_order(baseline_docs)[: int(baseline_candidate_top_k)]
    if str(candidate_admission) == "cross_channel":
        return build_cross_channel_admitted_evidence_pool(
            evidence_channels,
            base_candidate_top_k=int(admission_base_top_k),
            expansion_candidate_top_k=int(candidate_top_k),
            max_candidates=int(max_candidates),
            min_channel_support=int(admission_min_channel_support),
            seed_docs=seed_docs,
        )
    if str(candidate_admission) == "terminal_bound":
        return build_terminal_channel_admitted_evidence_pool(
            evidence_channels,
            terminal_channel_ids=terminal_channel_ids or [],
            base_candidate_top_k=int(admission_base_top_k),
            expansion_candidate_top_k=int(candidate_top_k),
            terminal_admission_limit=int(terminal_admission_limit),
            max_candidates=int(max_candidates),
            seed_docs=seed_docs,
        )
    if str(candidate_admission) == "demand_marginal_coverage":
        expansion_top_k = (
            int(demand_marginal_expansion_top_k)
            if int(demand_marginal_expansion_top_k) > 0
            else int(candidate_top_k)
        )
        return build_demand_marginal_coverage_evidence_pool(
            evidence_channels,
            candidate_top_k=int(candidate_top_k),
            expansion_candidate_top_k=expansion_top_k,
            max_candidates=int(max_candidates),
            alpha=float(demand_marginal_alpha),
            seed_docs=seed_docs,
        )
    if str(candidate_admission) == "demand_marginal_topk":
        return build_demand_marginal_topk_evidence_pool(
            evidence_channels,
            candidate_top_k=int(candidate_top_k),
            max_candidates=int(max_candidates),
            alpha=float(demand_marginal_alpha),
            seed_docs=seed_docs,
        )
    if str(candidate_admission) == "pareto_frontier":
        expansion_top_k = (
            int(demand_marginal_expansion_top_k)
            if int(demand_marginal_expansion_top_k) > 0
            else int(candidate_top_k)
        )
        return build_pareto_frontier_evidence_pool(
            evidence_channels,
            candidate_top_k=int(candidate_top_k),
            expansion_candidate_top_k=expansion_top_k,
            max_candidates=int(max_candidates),
            seed_docs=seed_docs,
        )
    if str(candidate_admission) == "pareto_repair":
        expansion_top_k = (
            int(demand_marginal_expansion_top_k)
            if int(demand_marginal_expansion_top_k) > 0
            else int(candidate_top_k)
        )
        return build_pareto_repair_evidence_pool(
            evidence_channels,
            candidate_top_k=int(candidate_top_k),
            expansion_candidate_top_k=expansion_top_k,
            max_candidates=int(max_candidates),
            seed_docs=seed_docs,
        )
    if str(candidate_admission) == "residual_role_repair":
        return build_channel_balanced_evidence_pool(
            evidence_channels,
            candidate_top_k=int(candidate_top_k),
            max_candidates=int(max_candidates),
            seed_docs=seed_docs,
        )
    return build_channel_balanced_evidence_pool(
        evidence_channels,
        candidate_top_k=int(candidate_top_k),
        max_candidates=int(max_candidates),
        seed_docs=seed_docs,
    )


def terminal_role_ids(roles: Sequence[Mapping[str, Any]]) -> List[str]:
    """Return terminal role ids from a generated evidence-role graph."""
    role_ids = [str(role.get("role_id") or "") for role in roles if str(role.get("role_id") or "").strip()]
    terminals: List[str] = []
    for role in roles:
        role_id = str(role.get("role_id") or "")
        if not role_id:
            continue
        outgoing = [str(value) for value in role.get("must_connect_to", []) or [] if str(value).strip()]
        if not outgoing and role_id not in terminals:
            terminals.append(role_id)
    if not terminals and role_ids:
        terminals.append(role_ids[-1])
    return terminals


def evidence_channels_from_trace_channels(channels: Sequence[Any]) -> List[EvidenceChannel]:
    evidence_channels: List[EvidenceChannel] = []
    for channel_idx, channel in enumerate(channels):
        role_id = str(getattr(channel, "role_id", "") or f"channel_{channel_idx}")
        description = str(
            getattr(channel, "role_description", None)
            or getattr(channel, "retrieval_query", None)
            or ""
        )
        support_function = str(getattr(channel, "support_function", "") or "").strip()
        if support_function and support_function not in description:
            description = f"{support_function}: {description}"
        doc_indices = [int(doc_idx) for doc_idx in getattr(channel, "doc_indices", []) or []]
        evidence_channels.append(
            EvidenceChannel(
                channel_id=role_id,
                description=description,
                doc_indices=doc_indices,
            )
        )
    return evidence_channels


def channel_from_mapping(raw: Mapping[str, Any]) -> RoleGraphChannel:
    """Parse one serialized demand-channel trace without importing fusion ablations."""
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


def candidate_role_provenance(
    *,
    candidate_docs: Sequence[int],
    channels: Sequence[Any],
    candidate_top_k: int,
) -> Dict[int, List[str]]:
    evidence_channels = evidence_channels_from_trace_channels(channels)
    raw_provenance = provenance_summary(
        channels=evidence_channels,
        candidate_docs=[int(doc_idx) for doc_idx in candidate_docs],
        candidate_top_k=int(candidate_top_k),
    )
    provenance: Dict[int, List[str]] = {int(doc_idx): [] for doc_idx in candidate_docs}
    for doc_idx, entries in raw_provenance.items():
        for entry in entries:
            description = str(entry.get("description") or "").strip()
            rank = int(entry.get("rank", 0) or 0)
            if not description or rank <= 0:
                continue
            provenance.setdefault(int(doc_idx), []).append(f"{description} (rank {rank})")
    return provenance


def selector_prompt(
    *,
    question: str,
    role_descriptions: Sequence[str],
    candidate_docs: Sequence[int],
    corpus_passages: Sequence[str],
    max_passage_chars: int,
    top_k: int,
    candidate_provenance: Mapping[int, Sequence[str]] | None = None,
    selector_objective: str = "evidence_set_coverage",
) -> List[Dict[str, str]]:
    channels_text = "\n".join(f"- {role}" for role in role_descriptions)
    candidates = []
    for local_id, doc_idx in enumerate(candidate_docs):
        passage = truncate_text(corpus_passages[int(doc_idx)], max_passage_chars)
        provenance = ""
        if candidate_provenance is not None:
            roles = [str(role) for role in candidate_provenance.get(int(doc_idx), []) if str(role).strip()]
            if roles:
                provenance = "Retrieved by evidence channels:\n" + "\n".join(f"- {role}" for role in roles) + "\n"
        candidates.append(f"[{local_id}] {provenance}{passage}")
    objective = str(selector_objective or "evidence_set_coverage")
    if objective == "answer_resolving":
        instruction = (
            f"Select exactly {top_k} candidate ids that together form the best top-k evidence context. "
            "The selected set must make the final answer binding unambiguous: include the passage that resolves the "
            "question's terminal answer and the bridge evidence that ties it to the question. Prefer passages that "
            "cover distinct evidence channels, but do not select a merely related passage if it introduces a competing "
            "entity, date, organization, or value without resolving why it is the answer. Preserve useful direct "
            "evidence and avoid redundancy unless it disambiguates the final answer. Return only JSON with schema "
            "{\"selected_ids\": [0, 1, 2, 3, 4]}."
        )
        system = (
            "You are a strict multi-hop answer-resolving evidence-set selector. "
            "Return only valid JSON."
        )
    else:
        instruction = (
            f"Select exactly {top_k} candidate ids that together form the best top-k evidence context. "
            "Prefer passages that cover distinct evidence channels, preserve useful direct evidence, and avoid "
            "redundant passages that repeat the same channel without adding a missing bridge or answer evidence. "
            "The selected set should answer the question as a multi-hop evidence set, but do not over-prioritize "
            "isolated answer-looking passages unless they are tied to the evidence channels. Return only JSON with "
            "schema {\"selected_ids\": [0, 1, 2, 3, 4]}."
        )
        system = "You are a strict multi-hop evidence-set selector. Return only valid JSON."

    user = (
        "Question:\n"
        f"{question}\n\n"
        "Evidence channels that a good multi-hop evidence set should cover:\n"
        f"{channels_text}\n\n"
        "Candidate passages:\n"
        + "\n\n".join(candidates)
        + "\n\n"
        + instruction
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def cache_key(messages: Sequence[Mapping[str, str]], model: str, temperature: float, max_tokens: int) -> str:
    raw = json.dumps(
        {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": NOTHINK_EXTRA_BODY,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_selected_ids(text: str, num_candidates: int, top_k: int) -> List[int]:
    raw = str(text or "").strip()
    payload: Any
    try:
        payload = json.loads(raw)
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
            if 0 <= idx < num_candidates and idx not in out:
                out.append(idx)
            if len(out) >= top_k:
                break
    return out


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


def call_selector(
    *,
    client: Any,
    messages: Sequence[Mapping[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_wait_seconds: float,
) -> Tuple[str, Dict[str, Any]]:
    last_error: BaseException | None = None
    for attempt in range(max(1, retries)):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=dict(NOTHINK_EXTRA_BODY),
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
            if attempt + 1 >= max(1, retries):
                break
            time.sleep(retry_wait_seconds * (2**attempt))
    raise RuntimeError(f"Selector call failed after {retries} attempts: {last_error}") from last_error


def select_docs(
    *,
    client: Any,
    question: str,
    role_descriptions: Sequence[str],
    candidate_docs: Sequence[int],
    corpus_passages: Sequence[str],
    top_k: int,
    max_passage_chars: int,
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_wait_seconds: float,
    cache: MutableMapping[str, Any],
    cache_path: Path,
    candidate_provenance: Mapping[int, Sequence[str]] | None = None,
    selector_objective: str = "evidence_set_coverage",
) -> Tuple[List[int], Dict[str, Any]]:
    messages = selector_prompt(
        question=question,
        role_descriptions=role_descriptions,
        candidate_docs=candidate_docs,
        corpus_passages=corpus_passages,
        max_passage_chars=max_passage_chars,
        top_k=top_k,
        candidate_provenance=candidate_provenance,
        selector_objective=selector_objective,
    )
    key = cache_key(messages, model, temperature, max_tokens)
    cached = cache.get(key)
    if cached is None:
        content, metadata = call_selector(
            client=client,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            retry_wait_seconds=retry_wait_seconds,
        )
        selected_ids = parse_selected_ids(content, len(candidate_docs), top_k)
        cached = {"content": content, "metadata": metadata, "selected_ids": selected_ids}
        cache[key] = cached
        write_cache(cache_path, cache)
    selected_ids = [int(idx) for idx in cached.get("selected_ids", []) if 0 <= int(idx) < len(candidate_docs)]
    selected_docs = candidate_stable_selected_topk(candidate_docs, selected_ids, top_k=int(top_k))
    return selected_docs[:top_k], dict(cached)


def make_md(payload: Mapping[str, Any]) -> str:
    summary = payload.get("summaries", {}).get(payload.get("variant_name"), {})
    config = payload.get("config", {})
    return "\n".join(
        [
            "# DCR-ESS Listwise Selector",
            "",
            "## Boundary",
            "",
            "This selector makes one listwise evidence-set decision over fixed channel-conditioned graph retrieval candidates.",
            "It does not rebuild graphs, rerun OpenIE, or use gold labels during selection.",
            "",
            "## Config",
            "",
            "```text",
            f"dataset: {payload.get('dataset')}",
            f"role_channel_json: {payload.get('source', {}).get('role_channel_json')}",
            f"candidate_top_k: {config.get('candidate_top_k')}",
            f"candidate_admission: {config.get('candidate_admission')}",
            f"admission_base_top_k: {config.get('admission_base_top_k')}",
            f"admission_min_channel_support: {config.get('admission_min_channel_support')}",
            f"terminal_admission_limit: {config.get('terminal_admission_limit')}",
            f"demand_marginal_expansion_top_k: {config.get('demand_marginal_expansion_top_k')}",
            f"demand_marginal_alpha: {config.get('demand_marginal_alpha')}",
            f"baseline_candidate_top_k: {config.get('baseline_candidate_top_k')}",
            f"max_candidates: {config.get('max_candidates')}",
            f"reader_top_k: {config.get('reader_top_k')}",
            f"selector_model: {config.get('selector_model')}",
            f"selector_objective: {config.get('selector_objective')}",
            f"allow_ablation_variants: {config.get('allow_ablation_variants')}",
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
        ]
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    validate_clean_mainline_args(args)
    corpus_records = load_json(args.corpus_json)
    if not isinstance(corpus_records, list):
        raise ValueError(f"Corpus must be a list: {args.corpus_json}")
    corpus_passages = [passage_from_corpus_row(row) for row in corpus_records]
    title_to_index = corpus_title_to_index(corpus_records)
    per_query_variant, rows_by_qid = load_per_query_rows(args.per_query_report, args.per_query_variant)
    base_rank_cache = load_baseline_rank_cache(args.baseline_top200_cache, candidate_top_k=max(args.candidate_top_k, args.reader_top_k))
    role_payload = load_json(args.role_channel_json)
    traces = role_payload.get("traces", []) if isinstance(role_payload, Mapping) else []
    if not isinstance(traces, list):
        raise ValueError("role_channel_json must contain a traces list")
    roles_by_query: Dict[int, Sequence[Mapping[str, Any]]] = {}
    if getattr(args, "roles_json_path", None):
        roles_by_query = load_roles(args.roles_json_path)
    client = make_openai_client(base_url=args.selector_base_url, api_key=args.api_key)
    cache_path = Path(args.selector_cache_path)
    cache = load_cache(cache_path)
    baseline_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    selector_traces: List[Dict[str, Any]] = []

    for trace_idx, trace in enumerate(traces):
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
        baseline_docs = unique_preserve_order(base_rank_cache[qid].get("doc_indices", []))
        baseline_top5 = baseline_docs[: int(args.reader_top_k)]
        base_row = make_row(
            query_index=qid,
            question=question,
            gold_answers=gold_answers,
            gold_doc_indices=gold_docs,
            reader_doc_indices=baseline_top5,
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
            candidate_admission=getattr(args, "candidate_admission", "channel_balanced"),
            admission_base_top_k=int(getattr(args, "admission_base_top_k", 8)),
            admission_min_channel_support=int(getattr(args, "admission_min_channel_support", 2)),
            terminal_channel_ids=terminal_role_ids(roles_by_query.get(qid, [])),
            terminal_admission_limit=int(getattr(args, "terminal_admission_limit", 4)),
            demand_marginal_expansion_top_k=int(getattr(args, "demand_marginal_expansion_top_k", 0)),
            demand_marginal_alpha=float(getattr(args, "demand_marginal_alpha", 0.35)),
        )
        provenance = None
        if args.include_candidate_provenance:
            provenance_candidate_top_k = int(args.candidate_top_k)
            if (
                str(getattr(args, "candidate_admission", "channel_balanced")) == "demand_marginal_coverage"
                and int(getattr(args, "demand_marginal_expansion_top_k", 0)) > 0
            ):
                provenance_candidate_top_k = max(
                    provenance_candidate_top_k,
                    int(getattr(args, "demand_marginal_expansion_top_k", 0)),
                )
            provenance = candidate_role_provenance(
                candidate_docs=candidate_docs,
                channels=channels,
                candidate_top_k=int(provenance_candidate_top_k),
            )
        selected_docs, selector_payload = select_docs(
            client=client,
            question=question,
            role_descriptions=role_descriptions,
            candidate_docs=candidate_docs,
            corpus_passages=corpus_passages,
            top_k=int(args.reader_top_k),
            max_passage_chars=int(args.max_passage_chars),
            model=args.selector_model,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            retries=int(args.retries),
            retry_wait_seconds=float(args.retry_wait_seconds),
            cache=cache,
            cache_path=cache_path,
            candidate_provenance=provenance,
            selector_objective=args.selector_objective,
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
            }
        )
        if (trace_idx + 1) % 10 == 0:
            print(f"[dcr-ess-listwise] processed {trace_idx + 1}/{len(traces)} cache_size={len(cache)}", flush=True)

    summary = summarize_r5(selected_rows, baseline_rows)
    payload: Dict[str, Any] = {
        "mode": "dcr_ess_listwise_selector",
        "legacy_mode": "ress_listwise_selector",
        "method_name": METHOD_NAME,
        "legacy_method_name": "DCR-ESS",
        "dataset": args.dataset,
        "variant_name": args.variant_name,
        "per_query_variant": per_query_variant,
        "summaries": {args.variant_name: summary},
        "variants": {
            "hipporag_v2": baseline_rows,
            args.variant_name: selected_rows,
        },
        "selector_traces": selector_traces,
        "config": {
            "reader_top_k": int(args.reader_top_k),
            "candidate_top_k": int(args.candidate_top_k),
            "candidate_admission": getattr(args, "candidate_admission", "channel_balanced"),
            "admission_base_top_k": int(getattr(args, "admission_base_top_k", 8)),
            "admission_min_channel_support": int(getattr(args, "admission_min_channel_support", 2)),
            "terminal_admission_limit": int(getattr(args, "terminal_admission_limit", 4)),
            "demand_marginal_expansion_top_k": int(getattr(args, "demand_marginal_expansion_top_k", 0)),
            "demand_marginal_alpha": float(getattr(args, "demand_marginal_alpha", 0.35)),
            "baseline_candidate_top_k": int(args.baseline_candidate_top_k),
            "max_candidates": int(args.max_candidates),
            "max_passage_chars": int(args.max_passage_chars),
            "include_candidate_provenance": bool(args.include_candidate_provenance),
            "selector_model": args.selector_model,
            "selector_base_url": args.selector_base_url,
            "selector_objective": args.selector_objective,
            "allow_ablation_variants": bool(getattr(args, "allow_ablation_variants", False)),
            "temperature": float(args.temperature),
            "max_tokens": int(args.max_tokens),
        },
        "source": {
            "corpus_json": args.corpus_json,
            "per_query_report": args.per_query_report,
            "baseline_top200_cache": args.baseline_top200_cache,
            "role_channel_json": args.role_channel_json,
            "roles_json_path": getattr(args, "roles_json_path", None),
        },
    }
    write_cache(cache_path, cache)
    return payload


def main() -> None:
    args = parse_args()
    payload = run(args)
    save_json(args.save_json_path, payload)
    Path(args.save_md_path).write_text(make_md(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
