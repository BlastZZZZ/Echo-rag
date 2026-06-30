#!/usr/bin/env python3
"""Rerun QA reader only from a saved per-query report with Qwen thinking off.

This script intentionally does not run retrieval. It reconstructs the fixed
reader-facing Top-5 passages stored in a per-query JSON report, sends the same
HippoRAG RAG-QA prompt to an OpenAI-compatible reader, and passes
``chat_template_kwargs.enable_thinking=false`` in ``extra_body``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
from hipporag.prompts.prompt_template_manager import PromptTemplateManager


DEFAULT_PER_QUERY = (
    "outputs_pure_index_runtime_support_comparevariant_2wiki_limit100_qa_"
    "v8store_v2repair_headanchor_budget20_cachealigned_20260426/"
    "2wiki_v8store_v2repair_headanchor_limit100_perquery.json"
)
DEFAULT_CORPUS = "reproduce/dataset/2wikimultihopqa_corpus.json"
DEFAULT_OUTPUT_JSON = "QA_NOTHINK_REEVAL_2WIKI_LIMIT100_HEADANCHOR_20260427.json"
DEFAULT_OUTPUT_MD = "QA_NOTHINK_REEVAL_2WIKI_LIMIT100_HEADANCHOR_20260427.md"
DEFAULT_CACHE = "run_logs/qa_nothink_2wiki_limit100_headanchor_20260427_cache.json"
NOTHINK_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
READER_PROMPT_MODES = {"hipporag", "memory", "short_span"}
READER_PASSAGE_MODES = {"full", "query_focus", "query_focus_multi"}
EXTRA_BODY_MODES = {"auto", "qwen_disable", "none"}
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


@dataclass(frozen=True)
class ReaderTask:
    variant: str
    row_index: int
    row: Mapping[str, Any]
    messages: List[Dict[str, str]]
    gold_answers: List[str]
    doc_indices: List[int]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def normalize_text_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def passage_from_corpus_row(row: Mapping[str, Any]) -> str:
    title = str(row.get("title") or row.get("Title") or "").strip()
    text = normalize_text_field(row.get("text") if "text" in row else row.get("Text")).strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def load_corpus_passages(corpus_path: Path) -> Dict[int, str]:
    raw = load_json(corpus_path)
    if not isinstance(raw, list):
        raise ValueError(f"Corpus must be a list, got {type(raw).__name__}: {corpus_path}")

    passages: Dict[int, str] = {}
    for position, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"Corpus row {position} is not an object: {type(item).__name__}")
        passage = passage_from_corpus_row(item)
        passages[position] = passage
        idx = item.get("idx")
        if isinstance(idx, int) and idx not in passages:
            passages[idx] = passage
    return passages


def load_role_channel_memory(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    traces = payload.get("traces", []) if isinstance(payload, Mapping) else []
    out: Dict[int, Dict[str, Any]] = {}
    for trace in traces:
        if not isinstance(trace, Mapping):
            continue
        try:
            qid = int(trace.get("query_index"))
        except (TypeError, ValueError):
            continue
        roles: List[str] = []
        doc_roles: Dict[int, List[str]] = {}
        for channel in trace.get("channels", []) or []:
            if not isinstance(channel, Mapping):
                continue
            desc = " ".join(
                str(
                    channel.get("role_description")
                    or channel.get("provenance_text")
                    or channel.get("retrieval_text")
                    or channel.get("role_id")
                    or ""
                ).split()
            )
            if not desc:
                continue
            if desc not in roles:
                roles.append(desc)
            for rank, doc_idx in enumerate(as_int_list(channel.get("doc_indices")), start=1):
                doc_roles.setdefault(int(doc_idx), [])
                label = f"rank {rank}: {desc}"
                if label not in doc_roles[int(doc_idx)]:
                    doc_roles[int(doc_idx)].append(label)
        out[qid] = {"roles": roles, "doc_roles": doc_roles}
    return out


def as_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return []
    out: List[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def row_reader_doc_indices(row: Mapping[str, Any]) -> List[int]:
    return as_int_list(row.get("reader_doc_indices_topk") or row.get("retrieved_doc_indices_top5"))[:5]


def normalize_gold_answers(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                out.extend(str(inner) for inner in item if inner is not None)
            elif item is not None:
                out.append(str(item))
        return out
    return [str(value)]


def merge_gold_answers(*answer_lists: Sequence[Any]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for answers in answer_lists:
        for answer in normalize_gold_answers(answers):
            key = " ".join(str(answer or "").casefold().split())
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(str(answer))
    return out


def sample_gold_answers(sample: Mapping[str, Any]) -> List[str]:
    if "answer" in sample or "gold_ans" in sample:
        gold = sample["answer"] if "answer" in sample else sample["gold_ans"]
    elif "reference" in sample:
        gold = sample["reference"]
    elif "obj" in sample:
        gold = [sample.get("obj"), sample.get("possible_answers"), sample.get("o_wiki_title"), sample.get("o_aliases")]
    else:
        return []
    aliases = sample.get("answer_aliases", [])
    return merge_gold_answers(normalize_gold_answers(gold), normalize_gold_answers(aliases))


def load_answer_aliases(dataset_path: Optional[Path]) -> Dict[int, List[str]]:
    if dataset_path is None or not dataset_path.exists():
        return {}
    raw = load_json(dataset_path)
    if not isinstance(raw, list):
        raise ValueError(f"Dataset alias file must be a list: {dataset_path}")
    aliases: Dict[int, List[str]] = {}
    for idx, sample in enumerate(raw):
        if not isinstance(sample, Mapping):
            continue
        answers = sample_gold_answers(sample)
        if answers:
            aliases[idx] = answers
    return aliases


def default_dataset_path(dataset: str) -> Path:
    return PROJECT_ROOT / "reproduce" / "dataset" / f"{dataset}.json"


def build_prompt_messages(
    question: str,
    passages: Sequence[str],
    dataset: str,
    prompt_template_manager: Optional[PromptTemplateManager] = None,
    reader_prompt_mode: str = "hipporag",
    doc_indices: Optional[Sequence[int]] = None,
    memory_profile: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, str]]:
    if str(reader_prompt_mode) not in READER_PROMPT_MODES:
        raise ValueError(f"Unknown reader_prompt_mode: {reader_prompt_mode}")
    if str(reader_prompt_mode) == "short_span":
        context_blocks = []
        for idx, passage in enumerate(passages, start=1):
            context_blocks.append(f"[{idx}]\n{passage}")
        user = (
            "Context passages:\n"
            + "\n\n".join(context_blocks)
            + "\n\n"
            f"Question: {question}\n\n"
            "Answer with the shortest exact phrase from the context that answers the question.\n"
            "Prefer copying the answer span verbatim from the context.\n"
            "Do not explain. Do not output a full sentence if a noun phrase is sufficient.\n"
            "For type/kind/category questions, answer with the specific phrase named in the context, not a generic category.\n"
            "If the answer is a name, place, crop, object, method, or label, output only that name or label.\n"
            "Return exactly one line in this format:\n"
            "Answer: <short phrase>"
        )
        return [
            {
                "role": "system",
                "content": "You are an extractive QA reader. Copy concise answer spans from the provided context.",
            },
            {"role": "user", "content": user},
        ]
    if str(reader_prompt_mode) == "memory":
        manager = prompt_template_manager or PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        doc_indices = list(doc_indices or range(len(passages)))
        doc_roles = {}
        if isinstance(memory_profile, Mapping) and isinstance(memory_profile.get("doc_roles"), Mapping):
            doc_roles = memory_profile.get("doc_roles", {})
        prompt_user = ""
        for rank, passage in enumerate(passages, start=1):
            doc_idx = int(doc_indices[rank - 1]) if rank - 1 < len(doc_indices) else rank - 1
            roles = doc_roles.get(doc_idx, []) if isinstance(doc_roles, Mapping) else []
            role_text = "\n".join(f"- {role}" for role in roles[:6]) or "- selected by ECHOv2 base evidence memory"
            memory_block = (
                "\n".join(
                    [
                        f"[Memory {rank}]",
                        "Episodic memory:",
                        f"- ECHOv2 reader support rank {rank}",
                        f"- corpus_doc_index {doc_idx}",
                        "Semantic demand provenance:",
                        role_text,
                        "Veridical passage:",
                        str(passage or ""),
                    ]
                )
            )
            prompt_user += f"Wikipedia Title: {memory_block}\n\n"
        prompt_user += "Question: " + question + "\nThought: "
        template_dataset = dataset if manager.is_template_name_valid(name=f"rag_qa_{dataset}") else "musique"
        rendered = manager.render(name=f"rag_qa_{template_dataset}", prompt_user=prompt_user)
        if not isinstance(rendered, list):
            raise TypeError(f"RAG QA template must render chat messages, got {type(rendered).__name__}")
        return [{"role": str(item["role"]), "content": str(item["content"])} for item in rendered]
    manager = prompt_template_manager or PromptTemplateManager(
        role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
    )
    prompt_user = ""
    for passage in passages:
        prompt_user += f"Wikipedia Title: {passage}\n\n"
    prompt_user += "Question: " + question + "\nThought: "

    template_dataset = dataset if manager.is_template_name_valid(name=f"rag_qa_{dataset}") else "musique"
    rendered = manager.render(name=f"rag_qa_{template_dataset}", prompt_user=prompt_user)
    if not isinstance(rendered, list):
        raise TypeError(f"RAG QA template must render chat messages, got {type(rendered).__name__}")
    return [{"role": str(item["role"]), "content": str(item["content"])} for item in rendered]


def truncate_text(text: str, max_chars: int) -> str:
    passage = str(text or "")
    budget = int(max_chars)
    if budget <= 0:
        return ""
    if len(passage) <= budget:
        return passage
    return passage[:budget].rstrip() + " ..."


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


def query_focused_excerpt(text: str, question: str, max_chars: int) -> str:
    """Return a question-relevant text window without using answers or gold labels."""
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


def query_focused_multi_excerpt(text: str, question: str, max_chars: int, window_count: int = 3) -> str:
    """Return several question-relevant windows to reduce single-window misses."""
    passage = str(text or "")
    budget = int(max_chars)
    if budget <= 0:
        return ""
    if len(passage) <= budget:
        return passage
    terms = query_focus_terms(question)
    if not terms:
        return truncate_text(passage, budget)

    count = max(1, int(window_count))
    separator = "\n...\n"
    segment_budget = max(180, (budget - (count - 1) * len(separator)) // count)
    lowered = passage.casefold()
    starts = set()
    for term in terms:
        for match in re.finditer(re.escape(term), lowered):
            starts.add(max(0, min(match.start() - segment_budget // 3, len(passage) - segment_budget)))
    if not starts:
        return truncate_text(passage, budget)

    def score(start: int) -> Tuple[int, int, int]:
        window = lowered[start : start + segment_budget]
        covered = sum(1 for term in terms if term in window)
        count_hits = sum(window.count(term) for term in terms)
        return covered, count_hits, -start

    selected: List[int] = []
    for start in sorted(starts, key=score, reverse=True):
        if all(abs(start - prev) >= segment_budget // 2 for prev in selected):
            selected.append(start)
        if len(selected) >= count:
            break
    selected.sort()

    snippets: List[str] = []
    for start in selected:
        end = min(len(passage), start + segment_budget)
        snippet = passage[start:end].strip()
        if start > 0:
            snippet = "... " + snippet
        if end < len(passage):
            snippet = snippet + " ..."
        snippets.append(snippet)
    return separator.join(snippets)


def reader_passage_excerpt(passage: str, question: str, mode: str, max_chars: int) -> str:
    mode = str(mode)
    if mode not in READER_PASSAGE_MODES:
        raise ValueError(f"Unknown reader_passage_mode: {mode}")
    if mode == "full":
        return str(passage or "")

    text = str(passage or "")
    title, sep, body = text.partition("\n")
    focus_fn = query_focused_multi_excerpt if mode == "query_focus_multi" else query_focused_excerpt
    if sep and title.strip() and body.strip():
        focused = focus_fn(body, question, int(max_chars))
        return f"{title.strip()}\n{focused}".strip()
    return focus_fn(text, question, int(max_chars))


def make_reader_tasks(
    per_query_data: Mapping[str, Any],
    corpus_passages: Mapping[int, str],
    dataset: str,
    variants: Optional[Sequence[str]] = None,
    answer_aliases_by_qid: Optional[Mapping[int, Sequence[str]]] = None,
    reader_prompt_mode: str = "hipporag",
    reader_passage_mode: str = "full",
    reader_max_passage_chars: int = 1200,
    role_memory_by_qid: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> List[ReaderTask]:
    raw_variants = per_query_data.get("variants")
    if not isinstance(raw_variants, Mapping):
        raise ValueError("Per-query report must contain a 'variants' object")

    selected = list(variants) if variants else list(raw_variants.keys())
    answer_aliases_by_qid = answer_aliases_by_qid or {}
    role_memory_by_qid = role_memory_by_qid or {}
    manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
    tasks: List[ReaderTask] = []
    for variant in selected:
        rows = rows_for_variant(raw_variants, variant)
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"Variant {variant} row {row_index} is not an object")
            doc_indices = row_reader_doc_indices(row)
            missing = [doc_idx for doc_idx in doc_indices if doc_idx not in corpus_passages]
            if missing:
                raise KeyError(f"Missing corpus doc indices for {variant} row {row_index}: {missing[:5]}")
            question = str(row.get("question") or "")
            try:
                query_index = int(row.get("query_index", row_index))
            except (TypeError, ValueError):
                query_index = row_index
            passages = [
                reader_passage_excerpt(
                    corpus_passages[doc_idx],
                    question,
                    mode=reader_passage_mode,
                    max_chars=reader_max_passage_chars,
                )
                for doc_idx in doc_indices
            ]
            messages = build_prompt_messages(
                question,
                passages,
                dataset,
                manager,
                reader_prompt_mode=reader_prompt_mode,
                doc_indices=doc_indices,
                memory_profile=role_memory_by_qid.get(query_index, {}),
            )
            gold_answers = merge_gold_answers(
                normalize_gold_answers(row.get("gold_answers")),
                answer_aliases_by_qid.get(query_index, []),
            )
            tasks.append(
                ReaderTask(
                    variant=variant,
                    row_index=row_index,
                    row=row,
                    messages=messages,
                    gold_answers=gold_answers,
                    doc_indices=doc_indices,
                )
            )
    return tasks


def rows_for_variant(raw_variants: Mapping[str, Any], variant: str) -> List[Any]:
    payload = raw_variants.get(variant)
    if payload is None:
        raise KeyError(f"Variant not found in per-query report: {variant}")
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    raise ValueError(f"Variant rows must be a list or an object containing a rows list for {variant}")


def parse_answer(response_content: Any) -> Tuple[str, bool]:
    text = "" if response_content is None else str(response_content)
    try:
        return text.split("Answer:", 1)[1].strip(), True
    except Exception:
        return text, False


def contains_think(text: Any) -> bool:
    return "<think>" in str(text or "").lower()


def is_qwen_model(model: str) -> bool:
    return "qwen" in str(model or "").lower()


def resolve_api_key_arg(api_key_arg: str) -> str:
    if str(api_key_arg or "").strip().upper() == "ENV":
        return os.environ.get("OPENAI_API_KEY", "").strip()
    return str(api_key_arg or "")


def resolve_extra_body(model: str, mode: str) -> Dict[str, Any]:
    normalized = str(mode or "auto").strip().lower()
    if normalized == "none":
        return {}
    if normalized == "qwen_disable":
        return dict(NOTHINK_EXTRA_BODY)
    if normalized == "auto" and is_qwen_model(model):
        return dict(NOTHINK_EXTRA_BODY)
    return {}


def cache_key(
    messages: Sequence[Mapping[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    extra_body: Mapping[str, Any],
) -> str:
    payload = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": extra_body,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_openai_client(base_url: str, api_key: str) -> Any:
    from openai import OpenAI

    try:
        import httpx

        http_client = httpx.Client(timeout=httpx.Timeout(300.0, read=300.0), trust_env=False)
        return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
    except Exception:
        return OpenAI(base_url=base_url, api_key=api_key)


def call_reader(
    client: Any,
    messages: Sequence[Mapping[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    extra_body: Mapping[str, Any],
    retries: int,
    retry_wait_seconds: float,
) -> Tuple[str, Dict[str, Any]]:
    last_error: Optional[BaseException] = None
    for attempt in range(max(1, retries)):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=dict(extra_body),
            )
            choice = response.choices[0]
            content = choice.message.content
            usage = getattr(response, "usage", None)
            metadata = {
                "finish_reason": getattr(choice, "finish_reason", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            }
            return "" if content is None else str(content), metadata
        except Exception as exc:  # pragma: no cover - exercised in live runs
            last_error = exc
            if attempt + 1 >= max(1, retries):
                break
            time.sleep(retry_wait_seconds * (2**attempt))
    raise RuntimeError(f"Reader call failed after {retries} attempts: {last_error}") from last_error


def metric_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / float(len(rows))


def recall_at_5(row: Mapping[str, Any]) -> float:
    gold = set(as_int_list(row.get("gold_doc_indices")))
    docs = set(row_reader_doc_indices(row))
    if not gold:
        return 0.0
    return float(len(gold & docs)) / float(len(gold))


def evaluate_predictions(gold_answers: Sequence[List[str]], predictions: Sequence[str]) -> Tuple[List[float], List[float]]:
    _, em_rows = QAExactMatch().calculate_metric_scores(list(gold_answers), list(predictions))
    _, f1_rows = QAF1Score().calculate_metric_scores(list(gold_answers), list(predictions))
    return [float(row["ExactMatch"]) for row in em_rows], [float(row["F1"]) for row in f1_rows]


def summarize_variant(rows: Sequence[Mapping[str, Any]], reevaluated_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "num_queries": 0,
            "R@5": 0.0,
            "original_ExactMatch": 0.0,
            "original_F1": 0.0,
            "nothink_ExactMatch": 0.0,
            "nothink_F1": 0.0,
            "delta_ExactMatch": 0.0,
            "delta_F1": 0.0,
            "original_think_leak_count": 0,
            "nothink_think_leak_count": 0,
            "nothink_parse_failure_count": 0,
        }
    original_em = metric_mean(rows, "ExactMatch")
    original_f1 = metric_mean(rows, "F1")
    nothink_em = metric_mean(reevaluated_rows, "ExactMatch")
    nothink_f1 = metric_mean(reevaluated_rows, "F1")
    return {
        "num_queries": len(rows),
        "R@5": sum(recall_at_5(row) for row in rows) / float(len(rows)),
        "original_ExactMatch": original_em,
        "original_F1": original_f1,
        "nothink_ExactMatch": nothink_em,
        "nothink_F1": nothink_f1,
        "delta_ExactMatch": nothink_em - original_em,
        "delta_F1": nothink_f1 - original_f1,
        "original_think_leak_count": sum(1 for row in rows if contains_think(row.get("predicted_answer"))),
        "nothink_think_leak_count": sum(1 for row in reevaluated_rows if contains_think(row.get("response_content"))),
        "nothink_parse_failure_count": sum(1 for row in reevaluated_rows if not row.get("parsed_with_answer_marker")),
    }


def run_reevaluation(
    tasks: Sequence[ReaderTask],
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    cache_path: Path,
    concurrency: int,
    retries: int,
    retry_wait_seconds: float,
    extra_body: Mapping[str, Any],
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    cache: MutableMapping[str, Any]
    if cache_path.exists():
        cache = load_json(cache_path)
        if not isinstance(cache, MutableMapping):
            raise ValueError(f"Cache file must contain a JSON object: {cache_path}")
    else:
        cache = {}

    cache_lock = threading.Lock()
    client = make_openai_client(base_url=base_url, api_key=api_key)

    def run_one(task: ReaderTask) -> Tuple[Tuple[str, int], Dict[str, Any], bool]:
        key = cache_key(task.messages, model, temperature, max_tokens, extra_body)
        with cache_lock:
            cached = cache.get(key)
        cache_hit = cached is not None
        if cached is None:
            response_content, metadata = call_reader(
                client=client,
                messages=task.messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                retries=retries,
                retry_wait_seconds=retry_wait_seconds,
            )
            cached = {"response_content": response_content, "metadata": metadata}
            with cache_lock:
                cache[key] = cached
                write_json_atomic(cache_path, cache)

        response_content = str(cached.get("response_content") or "")
        predicted_answer, parsed = parse_answer(response_content)
        row_payload = {
            "query_index": task.row.get("query_index", task.row_index),
            "row_index": task.row_index,
            "question": task.row.get("question"),
            "gold_answers": task.gold_answers,
            "gold_doc_indices": as_int_list(task.row.get("gold_doc_indices")),
            "reader_doc_indices_topk": task.doc_indices,
            "original_predicted_answer": task.row.get("predicted_answer"),
            "original_ExactMatch": float(task.row.get("ExactMatch", 0.0) or 0.0),
            "original_F1": float(task.row.get("F1", 0.0) or 0.0),
            "response_content": response_content,
            "predicted_answer": predicted_answer,
            "parsed_with_answer_marker": parsed,
            "think_leak": contains_think(response_content),
            "cache_hit": cache_hit,
            "metadata": cached.get("metadata") or {},
        }
        return (task.variant, task.row_index), row_payload, cache_hit

    results: Dict[Tuple[str, int], Dict[str, Any]] = {}
    total = len(tasks)
    completed = 0
    hits = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        future_to_task = {executor.submit(run_one, task): task for task in tasks}
        for future in as_completed(future_to_task):
            key, row_payload, cache_hit = future.result()
            results[key] = row_payload
            completed += 1
            hits += int(cache_hit)
            if completed == total or completed % 10 == 0:
                print(f"[nothink-qa] completed {completed}/{total} cache_hits={hits}", flush=True)
    return results


def attach_metrics_by_variant(
    per_query_data: Mapping[str, Any],
    raw_results: Mapping[Tuple[str, int], Mapping[str, Any]],
    variants: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    raw_variants = per_query_data.get("variants")
    if not isinstance(raw_variants, Mapping):
        raise ValueError("Per-query report must contain a 'variants' object")
    selected = list(variants) if variants else list(raw_variants.keys())

    out: Dict[str, Dict[str, Any]] = {}
    for variant in selected:
        original_rows = rows_for_variant(raw_variants, variant)
        ordered_rows = [dict(raw_results[(variant, idx)]) for idx in range(len(original_rows))]
        gold_answers = [normalize_gold_answers(row.get("gold_answers")) for row in ordered_rows]
        predictions = [str(row.get("predicted_answer") or "") for row in ordered_rows]
        em_scores, f1_scores = evaluate_predictions(gold_answers, predictions)
        for row, em, f1 in zip(ordered_rows, em_scores, f1_scores):
            row["ExactMatch"] = em
            row["F1"] = f1
            row["delta_ExactMatch"] = em - float(row.get("original_ExactMatch", 0.0) or 0.0)
            row["delta_F1"] = f1 - float(row.get("original_F1", 0.0) or 0.0)
        out[variant] = {
            "metrics": summarize_variant(original_rows, ordered_rows),
            "rows": ordered_rows,
        }
    return out


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def short_text(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# Qwen3 No-Think Reader-Only Re-Evaluation",
        "",
        "This report fixes the saved reader-facing Top-5 contexts and only reruns the QA reader with `chat_template_kwargs.enable_thinking=false`.",
        "",
        "## Config",
        "",
    ]
    meta = report.get("meta", {})
    for key in (
        "per_query_path",
        "corpus_path",
        "answer_alias_dataset_path",
        "answer_alias_queries",
        "dataset",
        "model",
        "base_url",
        "temperature",
        "max_tokens",
        "concurrency",
    ):
        lines.append(f"- `{key}`: `{meta.get(key)}`")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Variant | R@5 fixed | Original EM | No-think EM | dEM | Original F1 | No-think F1 | dF1 | Orig `<think>` | New `<think>` | Parse misses |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    variants = report.get("variants", {})
    for variant, payload in variants.items():
        metrics = payload.get("metrics", {})
        lines.append(
            "| {variant} | {r5} | {oem} | {nem} | {dem} | {of1} | {nf1} | {df1} | {otl} | {ntl} | {pm} |".format(
                variant=variant,
                r5=fmt(metrics.get("R@5")),
                oem=fmt(metrics.get("original_ExactMatch")),
                nem=fmt(metrics.get("nothink_ExactMatch")),
                dem=fmt(metrics.get("delta_ExactMatch"), digits=4),
                of1=fmt(metrics.get("original_F1")),
                nf1=fmt(metrics.get("nothink_F1")),
                df1=fmt(metrics.get("delta_F1"), digits=4),
                otl=metrics.get("original_think_leak_count"),
                ntl=metrics.get("nothink_think_leak_count"),
                pm=metrics.get("nothink_parse_failure_count"),
            )
        )

    focus_indices = set(meta.get("focus_query_indices") or [])
    if focus_indices:
        lines.extend(
            [
                "",
                "## Focus Queries",
                "",
                "| Variant | q | Orig EM/F1 | New EM/F1 | Orig answer | No-think answer | Raw leaked `<think>` |",
                "|---|---:|---:|---:|---|---|---:|",
            ]
        )
        for variant, payload in variants.items():
            for row in payload.get("rows", []):
                if int(row.get("query_index", -1)) not in focus_indices:
                    continue
                lines.append(
                    "| {variant} | {q} | {oem}/{of1} | {nem}/{nf1} | {opred} | {npred} | {leak} |".format(
                        variant=variant,
                        q=row.get("query_index"),
                        oem=fmt(row.get("original_ExactMatch"), digits=1),
                        of1=fmt(row.get("original_F1")),
                        nem=fmt(row.get("ExactMatch"), digits=1),
                        nf1=fmt(row.get("F1")),
                        opred=short_text(row.get("original_predicted_answer")),
                        npred=short_text(row.get("predicted_answer")),
                        leak=row.get("think_leak"),
                    )
                )

    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- R@5 is copied from the fixed Top-5 contexts and is not re-measured by retrieval.",
            "- EM/F1 changes here are reader-configuration sensitivity, not retrieval-method changes.",
            "- Any residual parse miss means the reader still failed to emit `Answer:` under the original HippoRAG prompt.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query", type=Path, default=Path(DEFAULT_PER_QUERY))
    parser.add_argument("--corpus", type=Path, default=Path(DEFAULT_CORPUS))
    parser.add_argument("--dataset", default="2wikimultihopqa")
    parser.add_argument(
        "--answer-alias-dataset",
        type=Path,
        default=None,
        help="Optional original dataset JSON used to enrich gold_answers with answer_aliases by query_index. Defaults to reproduce/dataset/<dataset>.json when present.",
    )
    parser.add_argument("--variant", action="append", default=None, help="Variant to rerun. Defaults to all variants.")
    parser.add_argument(
        "--role-channel-json",
        type=Path,
        default=None,
        help="Optional ECHOv2 role_channel.json used by --reader-prompt-mode memory.",
    )
    parser.add_argument("--output-json", type=Path, default=Path(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", type=Path, default=Path(DEFAULT_OUTPUT_MD))
    parser.add_argument("--cache", type=Path, default=Path(DEFAULT_CACHE))
    parser.add_argument("--model", default="qwen3-8b-train")
    parser.add_argument("--base-url", default="http://localhost:8041/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--extra-body-mode",
        choices=sorted(EXTRA_BODY_MODES),
        default="auto",
        help="Use Qwen enable_thinking=false only for Qwen models by default; use none for strict OpenAI calls.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-wait-seconds", type=float, default=2.0)
    parser.add_argument(
        "--reader-prompt-mode",
        choices=sorted(READER_PROMPT_MODES),
        default="hipporag",
        help="Reader prompt style.",
    )
    parser.add_argument(
        "--reader-passage-mode",
        choices=sorted(READER_PASSAGE_MODES),
        default="full",
        help="Reader context construction. 'query_focus' extracts a question-relevant window from each fixed Top-5 passage.",
    )
    parser.add_argument(
        "--reader-max-passage-chars",
        type=int,
        default=1200,
        help="Per-passage character budget used by --reader-passage-mode query_focus.",
    )
    parser.add_argument("--focus-query-index", type=int, action="append", default=[43, 66])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_query_path = args.per_query.resolve()
    corpus_path = args.corpus.resolve()
    per_query_data = load_json(per_query_path)
    corpus_passages = load_corpus_passages(corpus_path)
    alias_dataset_path = args.answer_alias_dataset.resolve() if args.answer_alias_dataset else default_dataset_path(args.dataset)
    answer_aliases = load_answer_aliases(alias_dataset_path)
    role_memory = load_role_channel_memory(args.role_channel_json.resolve() if args.role_channel_json else None)
    tasks = make_reader_tasks(
        per_query_data=per_query_data,
        corpus_passages=corpus_passages,
        dataset=args.dataset,
        variants=args.variant,
        answer_aliases_by_qid=answer_aliases,
        reader_prompt_mode=args.reader_prompt_mode,
        reader_passage_mode=args.reader_passage_mode,
        reader_max_passage_chars=args.reader_max_passage_chars,
        role_memory_by_qid=role_memory,
    )
    print(f"[nothink-qa] tasks={len(tasks)} variants={args.variant or 'all'}", flush=True)
    api_key = resolve_api_key_arg(args.api_key)
    extra_body = resolve_extra_body(args.model, args.extra_body_mode)
    raw_results = run_reevaluation(
        tasks=tasks,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cache_path=args.cache,
        concurrency=args.concurrency,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
        extra_body=extra_body,
    )
    variants = attach_metrics_by_variant(per_query_data, raw_results, variants=args.variant)
    report = {
        "meta": {
            "per_query_path": str(per_query_path),
            "corpus_path": str(corpus_path),
            "answer_alias_dataset_path": str(alias_dataset_path) if alias_dataset_path.exists() else None,
            "answer_alias_queries": len(answer_aliases),
            "dataset": args.dataset,
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "reader_prompt_mode": args.reader_prompt_mode,
            "reader_passage_mode": args.reader_passage_mode,
            "reader_max_passage_chars": args.reader_max_passage_chars,
            "role_channel_json": str(args.role_channel_json.resolve()) if args.role_channel_json else None,
            "role_memory_queries": len(role_memory),
            "concurrency": args.concurrency,
            "extra_body_mode": args.extra_body_mode,
            "extra_body": extra_body,
            "cache_path": str(args.cache.resolve()),
            "focus_query_indices": args.focus_query_index,
            "num_tasks": len(tasks),
        },
        "variants": variants,
    }
    write_json_atomic(args.output_json, report)
    args.output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"[nothink-qa] wrote {args.output_json}", flush=True)
    print(f"[nothink-qa] wrote {args.output_md}", flush=True)
    for variant, payload in variants.items():
        metrics = payload["metrics"]
        print(
            "[nothink-qa] {variant} R@5={r5:.4f} EM {oem:.4f}->{nem:.4f} "
            "F1 {of1:.4f}->{nf1:.4f} think {otl}->{ntl}".format(
                variant=variant,
                r5=metrics["R@5"],
                oem=metrics["original_ExactMatch"],
                nem=metrics["nothink_ExactMatch"],
                of1=metrics["original_F1"],
                nf1=metrics["nothink_F1"],
                otl=metrics["original_think_leak_count"],
                ntl=metrics["nothink_think_leak_count"],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
