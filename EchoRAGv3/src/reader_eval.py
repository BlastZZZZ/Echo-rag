#!/usr/bin/env python3
"""Evaluate raw context versus source-grounded proof annotation variants.

This is a reader-facing test of the proof-state evidence compiler.  It does
not change retrieval, reorder documents, train a ranker, add an answer-selection
rule, or use gold answers in the prompt.  The paired variants share the same
Top-5 passages and prompt contract.  The only runtime difference is the content
of the proof annotation section.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score
except Exception:
    import re
    import string

    def _normalize_answer(text: Any) -> str:
        value = "" if text is None else str(text).lower()
        value = "".join(ch for ch in value if ch not in set(string.punctuation))
        value = re.sub(r"\b(a|an|the)\b", " ", value)
        return " ".join(value.split())

    class QAExactMatch:
        def calculate_metric_scores(self, gold: Sequence[Sequence[str]], predictions: Sequence[str]) -> tuple[float, list[dict[str, float]]]:
            rows = []
            for answers, prediction in zip(gold, predictions):
                pred = _normalize_answer(prediction)
                score = float(any(pred == _normalize_answer(answer) for answer in answers))
                rows.append({"ExactMatch": score})
            return sum(row["ExactMatch"] for row in rows) / float(len(rows) or 1), rows

    class QAF1Score:
        def calculate_metric_scores(self, gold: Sequence[Sequence[str]], predictions: Sequence[str]) -> tuple[float, list[dict[str, float]]]:
            rows = []
            for answers, prediction in zip(gold, predictions):
                pred_tokens = _normalize_answer(prediction).split()
                best = 0.0
                for answer in answers:
                    gold_tokens = _normalize_answer(answer).split()
                    if not pred_tokens or not gold_tokens:
                        best = max(best, float(pred_tokens == gold_tokens))
                        continue
                    common = set(pred_tokens) & set(gold_tokens)
                    overlap = sum(min(pred_tokens.count(token), gold_tokens.count(token)) for token in common)
                    if overlap == 0:
                        continue
                    precision = overlap / float(len(pred_tokens))
                    recall = overlap / float(len(gold_tokens))
                    best = max(best, 2 * precision * recall / (precision + recall))
                rows.append({"F1": best})
            return sum(row["F1"] for row in rows) / float(len(rows) or 1), rows

import proof_adapter


@dataclass(frozen=True)
class ReaderTask:
    variant: str
    row_index: int
    query_index: int
    question: str
    gold_answers: list[str]
    selected_doc_indices: list[int]
    messages: list[dict[str, str]]
    proof_annotation_status: str
    annotation_source_query_index: int | None


DEFAULT_VARIANTS = [
    "raw_context_empty_annotation",
    "raw_context_plus_proof_annotation",
    "shuffled_proof_annotation",
    "status_matched_shuffled_proof_annotation",
    "sentence_snippet_annotation",
    "atom_bag_annotation",
]

PROOF_FORMATS = {"flat_atoms", "chain_trace", "grounded_atoms"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed-chain-json", type=Path, required=True)
    parser.add_argument("--corpus-json", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, default=None)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default="ENV")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-wait-seconds", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--variant",
        action="append",
        choices=DEFAULT_VARIANTS,
        default=None,
        help="Variant to run. Defaults to the full proof-annotation control matrix.",
    )
    parser.add_argument(
        "--proof-format",
        choices=sorted(PROOF_FORMATS),
        default="flat_atoms",
        help="Rendering used for raw_context_plus_proof_annotation and shuffled_proof_annotation.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def normalize_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return "" if value is None else str(value)


def passage_from_corpus_row(row: Mapping[str, Any]) -> str:
    title = str(row.get("title") or row.get("Title") or "").strip()
    text = normalize_text(row.get("text") if "text" in row else row.get("Text")).strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def load_corpus(path: Path) -> dict[int, str]:
    raw = read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Corpus must be a list: {path}")
    out: dict[int, str] = {}
    for position, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        passage = passage_from_corpus_row(item)
        out[position] = passage
        idx = item.get("idx")
        if isinstance(idx, int) and idx not in out:
            out[idx] = passage
    return out


def string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    out: list[str] = []
    for item in value:
        if item is None:
            continue
        text = " ".join(str(item).split())
        if text and text not in out:
            out.append(text)
    return out


def answer_aliases(path: Path | None) -> dict[int, list[str]]:
    if path is None or not path.exists():
        return {}
    raw = read_json(path)
    if not isinstance(raw, list):
        return {}
    out: dict[int, list[str]] = {}
    for idx, row in enumerate(raw):
        if not isinstance(row, Mapping):
            continue
        candidates: list[Any] = []
        for key in ("answer", "gold_ans", "reference", "answer_aliases", "possible_answers", "o_aliases"):
            if key in row:
                candidates.append(row.get(key))
        aliases: list[str] = []
        for item in candidates:
            for text in string_list(item):
                key = " ".join(text.casefold().split())
                if key and all(" ".join(old.casefold().split()) != key for old in aliases):
                    aliases.append(text)
        if aliases:
            out[idx] = aliases
    return out


def merge_answers(*groups: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for answer in group:
            key = " ".join(str(answer or "").casefold().split())
            if key and key not in seen:
                seen.add(key)
                out.append(str(answer))
    return out


def as_int_list(value: Any, *, limit: int | None = None) -> list[int]:
    if value is None or isinstance(value, (str, bytes)):
        return []
    out: list[int] = []
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number not in out:
            out.append(number)
        if limit is not None and len(out) >= limit:
            break
    return out


def truncate(text: str, max_chars: int = 2200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def binding_lines(items: Any) -> list[str]:
    lines: list[str] = []
    for item in items or []:
        if not isinstance(item, Mapping):
            continue
        variable = " ".join(str(item.get("variable") or "").split())
        value = " ".join(str(item.get("value") or "").split())
        if variable and value:
            lines.append(f"{variable} = {value}")
    return lines


def flat_proof_annotation_text(row: Mapping[str, Any]) -> tuple[str, str]:
    atoms = proof_adapter.proof_atoms(row)
    status = proof_adapter.proof_annotation_status(row, atoms)
    if not atoms:
        return status, "(none)"

    lines = [f"status: {status}"]
    for idx, atom in enumerate(atoms, start=1):
        subjects = ", ".join(atom.get("subject_values") or []) or "(implicit)"
        objects = ", ".join(atom.get("object_values") or []) or "(none)"
        relation = str(atom.get("relation_text") or "").strip() or "(relation unspecified)"
        span = str(atom.get("exact_source_span") or "").strip()
        doc = atom.get("source_doc_index")
        lines.append(f"- atom {idx}: {subjects} -- {relation} --> {objects}")
        if atom.get("derived_atom"):
            derived_from = ", ".join(str(item) for item in atom.get("derived_from_atom_indices") or [])
            lines.append(f"  derived_from_atoms: {derived_from}")
        else:
            lines.append(f"  source_doc_index: {doc}")
        if span:
            lines.append(f"  source_span: {span}")

    missing = proof_adapter.missing_obligations(row)
    if missing:
        lines.append("missing_obligations:")
        for item in missing:
            desc = " ".join(str(item.get("description") or item.get("role_id") or "").split())
            operator = str(item.get("operator") or "").strip()
            lines.append(f"- {item.get('role_id')}: {operator} {desc}".strip())
    return status, "\n".join(lines)


def chain_trace_proof_annotation_text(row: Mapping[str, Any]) -> tuple[str, str]:
    atoms = proof_adapter.proof_atoms(row)
    status = proof_adapter.proof_annotation_status(row, atoms)
    if not atoms:
        return status, "(none)"

    obligations = {str(item.get("role_id")): item for item in proof_adapter.role_obligations(row)}
    lines = [f"status: {status}", "proof_chain_trace:"]
    for idx, atom in enumerate(atoms, start=1):
        subjects = ", ".join(atom.get("subject_values") or []) or "(implicit)"
        objects = ", ".join(atom.get("object_values") or []) or "(none)"
        relation = str(atom.get("relation_text") or "").strip() or "(relation unspecified)"
        span = str(atom.get("exact_source_span") or "").strip()
        doc = atom.get("source_doc_index")
        role_id = str(atom.get("role_id") or "")
        obligation = obligations.get(role_id, {})
        operator = str(obligation.get("operator") or "").strip()
        role_type = str(obligation.get("role_type") or "").strip()
        description = " ".join(str(obligation.get("description") or "").split())
        consumes = binding_lines(atom.get("consumes"))
        produces = binding_lines(atom.get("produces"))
        title_scoped_consumes = binding_lines(atom.get("title_scoped_consumes"))
        lines.append(f"- step {idx}:")
        lines.append(f"  role_id: {role_id}")
        if role_type:
            lines.append(f"  role_type: {role_type}")
        if operator:
            lines.append(f"  operator: {operator}")
        if description:
            lines.append(f"  obligation: {description}")
        if consumes:
            lines.append("  consumes:")
            lines.extend(f"    - {line}" for line in consumes)
        if title_scoped_consumes:
            lines.append("  title_scoped_consumes:")
            lines.extend(f"    - {line}" for line in title_scoped_consumes)
        lines.append(f"  claim: {subjects} -- {relation} --> {objects}")
        if produces:
            lines.append("  produces:")
            lines.extend(f"    - {line}" for line in produces)
        if atom.get("derived_atom"):
            derived_from = ", ".join(str(item) for item in atom.get("derived_from_atom_indices") or [])
            lines.append(f"  derived_from_atoms: {derived_from}")
        else:
            lines.append(f"  source_doc_index: {doc}")
        if span:
            lines.append(f"  source_span: {span}")

    missing = proof_adapter.missing_obligations(row)
    if missing:
        lines.append("missing_obligations:")
        for item in missing:
            desc = " ".join(str(item.get("description") or item.get("role_id") or "").split())
            operator = str(item.get("operator") or "").strip()
            lines.append(f"- {item.get('role_id')}: {operator} {desc}".strip())
    return status, "\n".join(lines)


def proof_annotation_text(row: Mapping[str, Any], proof_format: str = "flat_atoms") -> tuple[str, str]:
    if proof_format == "chain_trace":
        return chain_trace_proof_annotation_text(row)
    if proof_format == "grounded_atoms":
        return atom_bag_annotation_text(row)
    if proof_format == "flat_atoms":
        return flat_proof_annotation_text(row)
    raise ValueError(f"Unknown proof format: {proof_format}")


def atom_bag_annotation_text(row: Mapping[str, Any]) -> tuple[str, str]:
    atoms = proof_adapter.proof_atoms(row)
    status = proof_adapter.proof_annotation_status(row, atoms)
    if not atoms:
        return status, "(none)"

    lines = [f"status: {status}", "atom_bag:"]
    for idx, atom in enumerate(atoms, start=1):
        subjects = ", ".join(atom.get("subject_values") or []) or "(implicit)"
        objects = ", ".join(atom.get("object_values") or []) or "(none)"
        relation = str(atom.get("relation_text") or "").strip() or "(relation unspecified)"
        span = str(atom.get("exact_source_span") or "").strip()
        doc = atom.get("source_doc_index")
        lines.append(f"- atom {idx}: {subjects} -- {relation} --> {objects}")
        lines.append(f"  source_doc_index: {doc}")
        if span:
            lines.append(f"  source_span: {span}")
    return status, "\n".join(lines)


def sentence_snippet_annotation_text(row: Mapping[str, Any]) -> tuple[str, str]:
    atoms = proof_adapter.proof_atoms(row)
    status = proof_adapter.proof_annotation_status(row, atoms)
    snippets: list[tuple[Any, str]] = []
    seen: set[tuple[str, str]] = set()
    for atom in atoms:
        span = " ".join(str(atom.get("exact_source_span") or "").split())
        if not span:
            continue
        doc = atom.get("source_doc_index")
        key = (str(doc), span)
        if key in seen:
            continue
        seen.add(key)
        snippets.append((doc, span))
    if not snippets:
        return status, "(none)"

    lines = [f"status: {status}", "source_snippets:"]
    for idx, (doc, span) in enumerate(snippets, start=1):
        lines.append(f"- snippet {idx}:")
        lines.append(f"  source_doc_index: {doc}")
        lines.append(f"  source_span: {span}")
    return status, "\n".join(lines)


def annotation_payloads(rows: Sequence[Mapping[str, Any]], proof_format: str = "flat_atoms") -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        qid = int(row.get("query_index", row_index))
        status, typed_text = proof_annotation_text(row, proof_format=proof_format)
        _, snippet_text = sentence_snippet_annotation_text(row)
        _, atom_bag_text = atom_bag_annotation_text(row)
        payloads.append(
            {
                "query_index": qid,
                "status": status,
                "typed": typed_text,
                "sentence_snippet": snippet_text,
                "atom_bag": atom_bag_text,
                "has_annotation": typed_text.strip() != "(none)",
            }
        )

    non_empty = [item for item in payloads if bool(item["has_annotation"])]
    for idx, item in enumerate(payloads):
        shuffled = "(none)"
        status_matched_shuffled = "(none)"
        source_qid: int | None = None
        status_matched_source_qid: int | None = None
        if non_empty:
            candidates = [candidate for candidate in non_empty if candidate["query_index"] != item["query_index"]]
            if candidates:
                selected = candidates[idx % len(candidates)]
                shuffled = str(selected["typed"])
                source_qid = int(selected["query_index"])
                if bool(item["has_annotation"]):
                    status_matched_shuffled = shuffled
                    status_matched_source_qid = source_qid
        item["shuffled"] = shuffled
        item["shuffled_source_query_index"] = source_qid
        item["status_matched_shuffled"] = status_matched_shuffled
        item["status_matched_shuffled_source_query_index"] = status_matched_source_qid
    return payloads


def build_messages(
    *,
    question: str,
    doc_indices: Sequence[int],
    corpus: Mapping[int, str],
    proof_annotation: str,
) -> list[dict[str, str]]:
    context_blocks: list[str] = []
    for rank, doc_idx in enumerate(doc_indices, start=1):
        passage = truncate(corpus[int(doc_idx)])
        context_blocks.append(f"[{rank}] corpus_doc_index={int(doc_idx)}\n{passage}")
    user = (
        "Context passages:\n"
        + "\n\n".join(context_blocks)
        + "\n\n"
        "Source-grounded proof annotation:\n"
        + proof_annotation
        + "\n\n"
        f"Question: {question}\n\n"
        "Answer with the shortest exact phrase supported by the context passages.\n"
        "Use the proof annotation only as a source-grounded guide; do not treat it as a new source.\n"
        "Do not explain. Return exactly one line:\n"
        "Answer: <short phrase>"
    )
    return [
        {
            "role": "system",
            "content": "You are an extractive multi-hop QA reader. Copy concise answers from the provided context.",
        },
        {"role": "user", "content": user},
    ]


def make_tasks(
    rows: Sequence[Mapping[str, Any]],
    corpus: Mapping[int, str],
    aliases_by_qid: Mapping[int, Sequence[str]],
    variants: Sequence[str] | None = None,
    proof_format: str = "flat_atoms",
) -> list[ReaderTask]:
    tasks: list[ReaderTask] = []
    selected_variants = list(variants or DEFAULT_VARIANTS)
    payloads = annotation_payloads(rows, proof_format=proof_format)
    for row_index, row in enumerate(rows):
        qid = int(row.get("query_index", row_index))
        question = str(row.get("question") or "")
        doc_indices = as_int_list(row.get("selected_doc_indices"), limit=5)
        missing = [doc for doc in doc_indices if doc not in corpus]
        if missing:
            raise KeyError(f"Missing corpus doc indices at query {qid}: {missing[:5]}")
        gold_answers = merge_answers(string_list(row.get("gold_answers")), aliases_by_qid.get(qid, []))
        payload = payloads[row_index]
        status = str(payload["status"])
        variant_payloads: dict[str, str] = {
            "raw_context_empty_annotation": "(none)",
            "raw_context_plus_proof_annotation": str(payload["typed"]),
            "shuffled_proof_annotation": str(payload["shuffled"]),
            "status_matched_shuffled_proof_annotation": str(payload["status_matched_shuffled"]),
            "sentence_snippet_annotation": str(payload["sentence_snippet"]),
            "atom_bag_annotation": str(payload["atom_bag"]),
        }
        source_qids: dict[str, int | None] = {
            "raw_context_empty_annotation": None,
            "raw_context_plus_proof_annotation": qid,
            "shuffled_proof_annotation": payload.get("shuffled_source_query_index"),
            "status_matched_shuffled_proof_annotation": payload.get("status_matched_shuffled_source_query_index"),
            "sentence_snippet_annotation": qid,
            "atom_bag_annotation": qid,
        }
        for variant in selected_variants:
            proof_text = variant_payloads[variant]
            messages = build_messages(
                question=question,
                doc_indices=doc_indices,
                corpus=corpus,
                proof_annotation=proof_text,
            )
            tasks.append(
                ReaderTask(
                    variant=variant,
                    row_index=row_index,
                    query_index=qid,
                    question=question,
                    gold_answers=gold_answers,
                    selected_doc_indices=doc_indices,
                    messages=messages,
                    proof_annotation_status=status,
                    annotation_source_query_index=source_qids[variant],
                )
            )
    return tasks


def resolve_api_key(value: str) -> str:
    text = str(value or "").strip()
    if text.upper() == "ENV":
        return os.environ.get("OPENAI_API_KEY", "").strip()
    if text.startswith("FILE:"):
        return Path(text[len("FILE:") :]).read_text(encoding="utf-8").strip()
    return text


def cache_key(messages: Sequence[Mapping[str, str]], model: str, temperature: float, max_tokens: int) -> str:
    payload = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_answer(text: Any) -> tuple[str, bool]:
    value = "" if text is None else str(text)
    if "Answer:" in value:
        return value.split("Answer:", 1)[1].strip(), True
    return value.strip(), False


def call_reader(
    client: Any,
    *,
    base_url: str,
    api_key: str,
    messages: Sequence[Mapping[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_wait_seconds: float,
) -> tuple[str, dict[str, Any]]:
    last_error: BaseException | None = None
    for attempt in range(max(1, retries)):
        try:
            if os.environ.get("PROOF_READER_USE_CURL") == "1":
                return call_reader_with_curl(
                    base_url=base_url,
                    api_key=api_key,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            response = client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            usage = getattr(response, "usage", None)
            return str(choice.message.content or ""), {
                "finish_reason": getattr(choice, "finish_reason", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            }
        except Exception as exc:  # pragma: no cover - live API path
            last_error = exc
            if attempt + 1 >= max(1, retries):
                break
            time.sleep(float(retry_wait_seconds) * (2**attempt))
    raise RuntimeError(f"reader call failed after {retries} attempts: {last_error}") from last_error


def call_reader_with_curl(
    *,
    base_url: str,
    api_key: str,
    messages: Sequence[Mapping[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as payload_file:
        json.dump(payload, payload_file, ensure_ascii=False)
        payload_path = payload_file.name
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as config_file:
        config_file.write(f'url = "{url}"\n')
        config_file.write('request = "POST"\n')
        config_file.write(f'header = "Authorization: Bearer {api_key}"\n')
        config_file.write('header = "Content-Type: application/json"\n')
        config_file.write(f'data-binary = "@{payload_path}"\n')
        config_path = config_file.name
    try:
        command = ["curl", "-sS", "-m", "300"]
        resolve_ip = os.environ.get("PROOF_READER_RESOLVE_IP", "").strip()
        if resolve_ip:
            host = urlsplit(url).hostname
            if host:
                command.extend(["--resolve", f"{host}:443:{resolve_ip}"])
        command.extend(["-K", config_path])
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"curl exited {completed.returncode}")
        data = json.loads(completed.stdout)
        if "error" in data:
            raise RuntimeError(str(data["error"]))
        choice = data["choices"][0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return str(message.get("content") or ""), {
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "api_transport": "curl",
        }
    finally:
        for path in (payload_path, config_path):
            try:
                os.unlink(path)
            except OSError:
                pass


def make_client(base_url: str, api_key: str) -> Any:
    from openai import OpenAI

    try:
        import httpx

        http_client = httpx.Client(timeout=httpx.Timeout(300.0, read=300.0), trust_env=False)
        return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0)
    except Exception:  # pragma: no cover - dependency-specific client path
        return OpenAI(base_url=base_url, api_key=api_key, timeout=300.0, max_retries=0)


def run_tasks(
    tasks: Sequence[ReaderTask],
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    cache_path: Path,
    concurrency: int,
    retries: int,
    retry_wait_seconds: float,
) -> list[dict[str, Any]]:
    cache: MutableMapping[str, Any]
    if cache_path.exists():
        loaded = read_json(cache_path)
        cache = loaded if isinstance(loaded, MutableMapping) else {}
    else:
        cache = {}
    cache_lock = threading.Lock()
    client = None if os.environ.get("PROOF_READER_USE_CURL") == "1" else make_client(base_url, api_key)
    grouped_tasks: dict[str, list[ReaderTask]] = {}
    representative_by_key: dict[str, ReaderTask] = {}
    for task in tasks:
        key = cache_key(task.messages, model, temperature, max_tokens)
        grouped_tasks.setdefault(key, []).append(task)
        representative_by_key.setdefault(key, task)

    def row_from_cached(task: ReaderTask, cached: Mapping[str, Any], cache_hit: bool) -> dict[str, Any]:
        answer, parsed = parse_answer(cached.get("response_content"))
        return {
            "variant": task.variant,
            "row_index": task.row_index,
            "query_index": task.query_index,
            "question": task.question,
            "gold_answers": task.gold_answers,
            "selected_doc_indices": task.selected_doc_indices,
            "proof_annotation_status": task.proof_annotation_status,
            "annotation_source_query_index": task.annotation_source_query_index,
            "response_content": cached.get("response_content"),
            "predicted_answer": answer,
            "parsed_with_answer_marker": parsed,
            "cache_hit": cache_hit,
            "metadata": cached.get("metadata") or {},
        }

    def run_one_key(key: str) -> tuple[str, Mapping[str, Any], bool]:
        task = representative_by_key[key]
        with cache_lock:
            cached = cache.get(key)
        cache_hit = cached is not None
        if cached is None:
            response, metadata = call_reader(
                client,
                base_url=base_url,
                api_key=api_key,
                messages=task.messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                retries=retries,
                retry_wait_seconds=retry_wait_seconds,
            )
            cached = {"response_content": response, "metadata": metadata}
            with cache_lock:
                cache[key] = cached
                write_json_atomic(cache_path, cache)
        return key, cached, cache_hit

    rows: list[dict[str, Any]] = []
    completed = 0
    hits = 0
    total_unique = len(grouped_tasks)
    if int(concurrency) <= 1:
        for key in grouped_tasks:
            key, cached, cache_hit = run_one_key(key)
            for task in grouped_tasks[key]:
                rows.append(row_from_cached(task, cached, cache_hit))
            completed += 1
            hits += int(cache_hit)
            if completed == total_unique or completed % 20 == 0:
                print(
                    f"[proof-reader] completed {completed}/{total_unique} unique_prompts "
                    f"rows={len(rows)}/{len(tasks)} cache_hits={hits}",
                    flush=True,
                )
        rows.sort(key=lambda item: (str(item["variant"]), int(item["row_index"])))
        return rows

    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        future_to_key = {executor.submit(run_one_key, key): key for key in grouped_tasks}
        for future in as_completed(future_to_key):
            key, cached, cache_hit = future.result()
            for task in grouped_tasks[key]:
                rows.append(row_from_cached(task, cached, cache_hit))
            completed += 1
            hits += int(cache_hit)
            if completed == total_unique or completed % 20 == 0:
                print(
                    f"[proof-reader] completed {completed}/{total_unique} unique_prompts "
                    f"rows={len(rows)}/{len(tasks)} cache_hits={hits}",
                    flush=True,
                )
    rows.sort(key=lambda item: (str(item["variant"]), int(item["row_index"])))
    return rows


def score_rows(rows: Sequence[dict[str, Any]]) -> None:
    gold = [string_list(row.get("gold_answers")) for row in rows]
    predictions = [str(row.get("predicted_answer") or "") for row in rows]
    _, em = QAExactMatch().calculate_metric_scores(gold, predictions)
    _, f1 = QAF1Score().calculate_metric_scores(gold, predictions)
    for row, em_row, f1_row in zip(rows, em, f1):
        row["ExactMatch"] = float(em_row["ExactMatch"])
        row["F1"] = float(f1_row["F1"])


def mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / float(len(rows))


def bootstrap_ci(values: Sequence[float], samples: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(13)
    n = len(values)
    means: list[float] = []
    for _ in range(max(1, int(samples))):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / float(n))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return [lo, hi]


def paired_summary(rows: Sequence[Mapping[str, Any]], bootstrap_samples: int) -> dict[str, Any]:
    by_variant: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row.get("variant")), []).append(row)
    sorted_variants = {
        variant: sorted(payload, key=lambda item: int(item["row_index"])) for variant, payload in by_variant.items()
    }
    raw = sorted_variants.get("raw_context_empty_annotation", [])
    proof = sorted_variants.get("raw_context_plus_proof_annotation", [])
    if not raw:
        raise ValueError("Raw baseline variant is missing")

    raw_by_row = {int(row["row_index"]): row for row in raw}

    def variant_metrics(payload: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "num_queries": len(payload),
            "ExactMatch": mean(payload, "ExactMatch"),
            "F1": mean(payload, "F1"),
            "parse_failure_count": sum(1 for row in payload if not row.get("parsed_with_answer_marker")),
        }

    def compare_to_raw(payload: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        deltas_f1: list[float] = []
        deltas_em: list[float] = []
        f1_wlt = {"win": 0, "loss": 0, "tie": 0}
        em_wlt = {"win": 0, "loss": 0, "tie": 0}
        by_status: dict[str, dict[str, Any]] = {}
        for row in payload:
            raw_row = raw_by_row.get(int(row["row_index"]))
            if raw_row is None:
                raise ValueError(f"Missing raw baseline row {row['row_index']}")
            df1 = float(row.get("F1", 0.0)) - float(raw_row.get("F1", 0.0))
            dem = float(row.get("ExactMatch", 0.0)) - float(raw_row.get("ExactMatch", 0.0))
            deltas_f1.append(df1)
            deltas_em.append(dem)
            f1_wlt["win" if df1 > 1e-9 else "loss" if df1 < -1e-9 else "tie"] += 1
            em_wlt["win" if dem > 1e-9 else "loss" if dem < -1e-9 else "tie"] += 1
            status = str(row.get("proof_annotation_status") or "")
            bucket = by_status.setdefault(status, {"rows": 0, "delta_f1_values": [], "delta_em_values": []})
            bucket["rows"] += 1
            bucket["delta_f1_values"].append(df1)
            bucket["delta_em_values"].append(dem)

        status_summary = {}
        for status, bucket in by_status.items():
            rows_count = int(bucket["rows"])
            status_summary[status] = {
                "rows": rows_count,
                "delta_F1": sum(bucket["delta_f1_values"]) / float(rows_count or 1),
                "delta_ExactMatch": sum(bucket["delta_em_values"]) / float(rows_count or 1),
            }
        return {
            "delta_ExactMatch": sum(deltas_em) / float(len(deltas_em) or 1),
            "delta_F1": sum(deltas_f1) / float(len(deltas_f1) or 1),
            "delta_ExactMatch_ci95": bootstrap_ci(deltas_em, bootstrap_samples),
            "delta_F1_ci95": bootstrap_ci(deltas_f1, bootstrap_samples),
            "F1_WLT": f1_wlt,
            "EM_WLT": em_wlt,
            "by_proof_annotation_status": status_summary,
            "delta_F1_values_by_row": {
                str(int(row["row_index"])): float(row.get("F1", 0.0)) - float(raw_by_row[int(row["row_index"])].get("F1", 0.0))
                for row in payload
            },
            "delta_ExactMatch_values_by_row": {
                str(int(row["row_index"])): float(row.get("ExactMatch", 0.0))
                - float(raw_by_row[int(row["row_index"])].get("ExactMatch", 0.0))
                for row in payload
            },
        }

    variant_summaries = {variant: variant_metrics(payload) for variant, payload in sorted_variants.items()}
    comparisons_to_raw = {
        variant: compare_to_raw(payload)
        for variant, payload in sorted_variants.items()
        if variant != "raw_context_empty_annotation"
    }

    proof_vs_controls: dict[str, dict[str, Any]] = {}
    proof_comparison = comparisons_to_raw.get("raw_context_plus_proof_annotation")
    if proof_comparison is not None:
        proof_f1 = proof_comparison["delta_F1_values_by_row"]
        proof_em = proof_comparison["delta_ExactMatch_values_by_row"]
        for variant, comparison in comparisons_to_raw.items():
            if variant == "raw_context_plus_proof_annotation":
                continue
            common = sorted(set(proof_f1) & set(comparison["delta_F1_values_by_row"]), key=int)
            dd_f1 = [proof_f1[key] - comparison["delta_F1_values_by_row"][key] for key in common]
            dd_em = [proof_em[key] - comparison["delta_ExactMatch_values_by_row"][key] for key in common]
            proof_vs_controls[variant] = {
                "rows": len(common),
                "delta_delta_F1": sum(dd_f1) / float(len(dd_f1) or 1),
                "delta_delta_F1_ci95": bootstrap_ci(dd_f1, bootstrap_samples),
                "delta_delta_ExactMatch": sum(dd_em) / float(len(dd_em) or 1),
                "delta_delta_ExactMatch_ci95": bootstrap_ci(dd_em, bootstrap_samples),
            }

    proof_comparison = proof_comparison or {
        "delta_ExactMatch": 0.0,
        "delta_F1": 0.0,
        "delta_ExactMatch_ci95": [0.0, 0.0],
        "delta_F1_ci95": [0.0, 0.0],
        "F1_WLT": {"win": 0, "loss": 0, "tie": 0},
        "EM_WLT": {"win": 0, "loss": 0, "tie": 0},
        "by_proof_annotation_status": {},
    }

    return {
        "num_queries": len(raw),
        "raw": variant_summaries.get("raw_context_empty_annotation", {}),
        "proof_annotation": variant_summaries.get("raw_context_plus_proof_annotation", {}),
        "delta_ExactMatch": proof_comparison["delta_ExactMatch"],
        "delta_F1": proof_comparison["delta_F1"],
        "delta_ExactMatch_ci95": proof_comparison["delta_ExactMatch_ci95"],
        "delta_F1_ci95": proof_comparison["delta_F1_ci95"],
        "F1_WLT": proof_comparison["F1_WLT"],
        "EM_WLT": proof_comparison["EM_WLT"],
        "by_proof_annotation_status": proof_comparison["by_proof_annotation_status"],
        "variants": variant_summaries,
        "comparisons_to_raw": {
            variant: {key: value for key, value in comparison.items() if not key.endswith("_values_by_row")}
            for variant, comparison in comparisons_to_raw.items()
        },
        "proof_vs_controls_delta_delta": proof_vs_controls,
    }


def build_markdown(report: Mapping[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# Proof Annotation Reader Evaluation",
        "",
        "Paired variants share the same selected Top-5 and reader prompt. "
        "They differ only in the proof annotation section.",
        "",
        "## Main Pair",
        "",
        "| dataset | n | raw EM | proof EM | dEM | raw F1 | proof F1 | dF1 | F1 W/L/T |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {report['meta'].get('dataset')} | {s['num_queries']} | "
            f"{s['raw']['ExactMatch']:.4f} | {s['proof_annotation']['ExactMatch']:.4f} | {s['delta_ExactMatch']:+.4f} | "
            f"{s['raw']['F1']:.4f} | {s['proof_annotation']['F1']:.4f} | {s['delta_F1']:+.4f} | "
            f"{s['F1_WLT']['win']}/{s['F1_WLT']['loss']}/{s['F1_WLT']['tie']} |"
        ),
        "",
        "## Controls",
        "",
        "| variant | EM | F1 | dEM vs raw | dF1 vs raw | CI95 dF1 | W/L/T |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    variants = s.get("variants", {})
    comparisons = s.get("comparisons_to_raw", {})
    for variant, metrics in variants.items():
        if variant == "raw_context_empty_annotation":
            lines.append(
                f"| {variant} | {metrics.get('ExactMatch', 0.0):.4f} | {metrics.get('F1', 0.0):.4f} | "
                f"+0.0000 | +0.0000 | [+0.0000, +0.0000] | 0/0/{metrics.get('num_queries', 0)} |"
            )
            continue
        comparison = comparisons.get(variant, {})
        ci = comparison.get("delta_F1_ci95", [0.0, 0.0])
        wlt = comparison.get("F1_WLT", {"win": 0, "loss": 0, "tie": 0})
        lines.append(
            f"| {variant} | {metrics.get('ExactMatch', 0.0):.4f} | {metrics.get('F1', 0.0):.4f} | "
            f"{comparison.get('delta_ExactMatch', 0.0):+.4f} | {comparison.get('delta_F1', 0.0):+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {wlt.get('win', 0)}/{wlt.get('loss', 0)}/{wlt.get('tie', 0)} |"
        )

    proof_vs_controls = s.get("proof_vs_controls_delta_delta", {})
    if proof_vs_controls:
        lines.extend(
            [
                "",
                "## Typed Proof vs Controls",
                "",
                "| control | delta-delta F1 | CI95 delta-delta F1 | delta-delta EM |",
                "|---|---:|---|---:|",
            ]
        )
        for control, comparison in proof_vs_controls.items():
            ci = comparison.get("delta_delta_F1_ci95", [0.0, 0.0])
            lines.append(
                f"| {control} | {comparison.get('delta_delta_F1', 0.0):+.4f} | "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {comparison.get('delta_delta_ExactMatch', 0.0):+.4f} |"
            )

    lines.extend(
        [
        "",
        "## Boundary",
        "",
        "- No retrieval change.",
        "- No document reordering.",
        "- No answer-selection rule.",
        "- No proof-only replacement.",
        "- Gold answers are used only for scoring.",
        "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args.api_key)
    if not api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    payload = read_json(args.typed_chain_json)
    rows = [row for row in payload.get("rows", []) if isinstance(row, Mapping)]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    dataset = str(args.dataset or payload.get("dataset") or args.typed_chain_json.stem)
    aliases = answer_aliases(args.dataset_json)
    corpus = load_corpus(args.corpus_json)
    variants = args.variant or DEFAULT_VARIANTS
    tasks = make_tasks(rows, corpus, aliases, variants=variants, proof_format=args.proof_format)
    raw_rows = run_tasks(
        tasks,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cache_path=args.cache,
        concurrency=args.concurrency,
        retries=args.retries,
        retry_wait_seconds=args.retry_wait_seconds,
    )
    score_rows(raw_rows)
    report = {
        "meta": {
            "typed_chain_json": str(args.typed_chain_json.resolve()),
            "corpus_json": str(args.corpus_json.resolve()),
            "dataset_json": str(args.dataset_json.resolve()) if args.dataset_json else None,
            "dataset": dataset,
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "rows": len(rows),
            "variants": variants,
            "proof_format": args.proof_format,
            "boundary": {
                "retrieval_change": False,
                "document_reordering": False,
                "answer_selection_rule": False,
                "proof_only_replacement": False,
                "gold_in_prompt": False,
            },
        },
        "summary": paired_summary(raw_rows, args.bootstrap_samples),
        "rows": raw_rows,
    }
    write_json_atomic(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
