#!/usr/bin/env python3
"""Build a fixed-prompt ERGR role cache with an OpenAI-compatible chat API.

This script only decomposes questions into evidence-role descriptions. It does
not see gold answers, gold titles, retrieval results, QA predictions, or SEER
support stores.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROLE_TYPES = {
    "grounding",
    "bridge",
    "constraint",
    "answer_bearing",
    "comparison",
    "verification",
}


SYSTEM_PROMPT = """You decompose multi-hop QA questions into evidence roles.

An evidence role describes what a retrieved evidence unit must establish for the
answer to be derivable. Do not answer the question. Do not use dataset names.
Do not mention gold passages. Use only the question text. Do not introduce
named entities, titles, dates, numbers, or answer values that are absent from
the question. If an entity must be discovered by retrieval, describe it as a
variable such as "the bridge entity", "that person", "that organization", or
"the answer entity".

Return only valid JSON with this schema:
{
  "roles": [
    {
      "role_id": "r0",
      "role_type": "grounding|bridge|constraint|answer_bearing|comparison|verification",
      "description": "short natural-language evidence requirement",
      "inputs": ["variables already visible or introduced earlier"],
      "outputs": ["variables introduced or resolved by this role"],
      "must_connect_to": ["role ids"],
      "required": true
    }
  ]
}

Keep 2 to 5 roles. Use role_type only as a coarse functional label; do not use
hand-written relation categories such as birth/death/director/writer as labels.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="qwen3-8b-train")
    parser.add_argument("--base_url", default="http://localhost:8041/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--parse_retries", type=int, default=3)
    parser.add_argument("--request_retries", type=int, default=6)
    parser.add_argument("--retry_sleep_seconds", type=float, default=2.0)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_key_file", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--qwen_disable_thinking", action="store_true")
    parser.add_argument(
        "--strict_question_grounding",
        action="store_true",
        help=(
            "Drop generated roles whose descriptions or auxiliary fields contain "
            "named surfaces absent from the question. This guards against "
            "answer/entity injection during demand generation."
        ),
    )
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key and str(args.api_key).strip().upper() == "ENV":
        return os.environ.get("OPENAI_API_KEY", "").strip()
    if args.api_key:
        return str(args.api_key).strip()
    if args.api_key_file:
        key_path = Path(args.api_key_file)
        if key_path.exists():
            return key_path.read_text(encoding="utf-8").strip()
    return os.environ.get("OPENAI_API_KEY", "").strip()


def repair_missing_commas_between_fields(text: str) -> str:
    """Repair a common LLM JSON error: missing comma before the next object key."""
    lines = str(text or "").splitlines()
    repaired: List[str] = []
    for idx, line in enumerate(lines):
        repaired.append(line)
        stripped = line.rstrip()
        if not stripped:
            continue
        next_line = lines[idx + 1].lstrip() if idx + 1 < len(lines) else ""
        if not re.match(r'"[^"]+"\s*:', next_line):
            continue
        if stripped.endswith((",", "{", "[", ":")):
            continue
        if stripped.endswith(("}", "]", '"')) or stripped in {"true", "false", "null"}:
            repaired[-1] = line.rstrip() + ","
    return "\n".join(repaired)


def repair_array_quote_bracket_transposition(text: str) -> str:
    """Repair ``']"`` to ``'"]`` inside malformed single-string arrays."""
    return str(text or "").replace("']\"", "'\"]")


def json_repair_variants(text: str) -> List[str]:
    """Generate conservative malformed-JSON repair candidates."""
    variants: List[str] = []
    seen = set()

    def add(candidate: str) -> None:
        if candidate not in seen:
            seen.add(candidate)
            variants.append(candidate)

    add(str(text or ""))
    idx = 0
    while idx < len(variants):
        candidate = variants[idx]
        idx += 1
        add(candidate.replace("\\'", "'"))
        add(repair_missing_commas_between_fields(candidate))
        add(repair_array_quote_bracket_transposition(candidate))
    return variants


def extract_json_object(text: str) -> Mapping[str, Any]:
    stripped = str(text or "").strip()
    candidates = json_repair_variants(stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        body = match.group(0)
        candidates.extend(json_repair_variants(body))
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, Mapping):
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    preview = stripped[:200]
    if last_error is not None:
        raise ValueError(f"No JSON object found in model output: {preview}") from last_error
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, Mapping):
            return parsed
    except json.JSONDecodeError:
        pass
    raise ValueError(f"No JSON object found in model output: {preview}")


GENERIC_SURFACES = {
    "answer",
    "answer entity",
    "bridge",
    "bridge entity",
    "birthplace",
    "both",
    "calculate",
    "cause",
    "child",
    "clarify",
    "combine",
    "compare",
    "comparison",
    "confirm",
    "constraint",
    "count",
    "date",
    "define",
    "describe",
    "determine",
    "duration",
    "ensure",
    "evidence",
    "establish",
    "explain",
    "extract",
    "exclude",
    "filter",
    "find",
    "further",
    "grounding",
    "identify",
    "identifies",
    "institution",
    "list",
    "link",
    "location",
    "locate",
    "name",
    "map",
    "person",
    "place",
    "people",
    "people y",
    "provide",
    "question",
    "region",
    "region c",
    "retrieve",
    "role",
    "shared",
    "specific",
    "source",
    "successor",
    "summarize",
    "support",
    "target",
    "team",
    "verification",
    "verify",
}


def normalize_surface(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def candidate_named_surfaces(text: str) -> List[str]:
    """Extract conservative named/date/number surfaces for leakage auditing."""
    surfaces: List[str] = []
    seen = set()

    def add(value: str) -> None:
        surface = str(value or "").replace(".", " ")
        surface = " ".join(surface.split()).strip(" ,.;:()[]{}")
        words = surface.split()
        while words and normalize_surface(words[0]) in GENERIC_SURFACES:
            words = words[1:]
        surface = " ".join(words).strip(" ,.;:()[]{}")
        key = normalize_surface(surface)
        if len(surface) < 2 or not key or key in seen or key in GENERIC_SURFACES:
            return
        seen.add(key)
        surfaces.append(surface)

    for match in re.finditer(r"[\"`“”‘’]([^\"`“”‘’]{2,})[\"`“”‘’]", str(text or "")):
        add(match.group(1))
    capitalized = r"(?:[A-Z][A-Za-z0-9&.\-']+|[A-Z]{2,}|\d{2,4})"
    for match in re.finditer(rf"\b{capitalized}(?:\s+{capitalized})*\b", str(text or "")):
        add(match.group(0))
    for match in re.finditer(r"\b\d{2,4}(?:[-/]\d{1,2})?(?:[-/]\d{1,2})?\b", str(text or "")):
        add(match.group(0))
    return surfaces


def unsupported_question_surfaces(text: str, question: str) -> List[str]:
    """Return named surfaces in text that are not licensed by the question."""
    question_norm = normalize_surface(question)
    question_possessive_norm = re.sub(r"\b([a-z0-9]+) s\b", r"\1", question_norm)
    question_tokens = set(question_possessive_norm.split())
    unsupported: List[str] = []
    seen = set()
    for surface in candidate_named_surfaces(text):
        key = normalize_surface(surface)
        if not key or key in seen:
            continue
        seen.add(key)
        possessive_key = re.sub(r"\b([a-z0-9]+) s\b", r"\1", key)
        if key in GENERIC_SURFACES or key in question_norm:
            continue
        if possessive_key in GENERIC_SURFACES or possessive_key in question_norm or possessive_key in question_possessive_norm:
            continue
        surface_tokens = set(possessive_key.split())
        if surface_tokens and surface_tokens <= question_tokens:
            continue
        unsupported.append(surface)
    return unsupported


def role_grounding_text(role: Mapping[str, Any]) -> str:
    # Retrieval uses the demand description, while inputs/outputs are legacy
    # bookkeeping fields. Grounding audits therefore focus on the text that
    # actually enters the graph-retrieval query.
    return str(role.get("description") or "")


def normalize_roles(
    payload: Mapping[str, Any],
    *,
    question: str = "",
    strict_question_grounding: bool = False,
) -> List[Dict[str, Any]]:
    raw_roles = payload.get("roles", [])
    if not isinstance(raw_roles, Sequence) or isinstance(raw_roles, (str, bytes)):
        return []
    roles: List[Dict[str, Any]] = []
    for offset, role in enumerate(raw_roles[:5]):
        if not isinstance(role, Mapping):
            continue
        description = " ".join(str(role.get("description") or "").split())
        if not description:
            continue
        role_type = str(role.get("role_type") or "bridge").strip()
        if role_type not in ROLE_TYPES:
            role_type = "bridge"
        normalized_role = {
            "role_id": str(role.get("role_id") or f"r{offset}"),
            "role_type": role_type,
            "description": description,
            "inputs": [str(value) for value in role.get("inputs", []) or [] if str(value).strip()],
            "outputs": [str(value) for value in role.get("outputs", []) or [] if str(value).strip()],
            "must_connect_to": [str(value) for value in role.get("must_connect_to", []) or [] if str(value).strip()],
            "required": bool(role.get("required", True)),
            "weight": float(role.get("weight", 1.0) or 1.0),
        }
        if question:
            unsupported = unsupported_question_surfaces(role_grounding_text(normalized_role), question)
            normalized_role["question_grounded"] = not unsupported
            normalized_role["unsupported_question_surfaces"] = unsupported
            if strict_question_grounding and unsupported:
                continue
        roles.append(normalized_role)
    return roles


def fallback_roles_for_question(question: str) -> List[Dict[str, Any]]:
    """Question-only fallback for rare role-generation parse failures."""
    clean_question = " ".join(str(question or "").split())
    description = "Find evidence that directly answers the question."
    if clean_question:
        description = f"Find evidence that directly answers: {clean_question}"
    return [
        {
            "role_id": "r0",
            "role_type": "answer_bearing",
            "description": description,
            "inputs": ["question"],
            "outputs": ["answer"],
            "must_connect_to": [],
            "required": True,
            "weight": 1.0,
        }
    ]


def _usage_from_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _sum_usage(usages: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    for usage in usages:
        if not isinstance(usage, Mapping):
            continue
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if isinstance(prompt, (int, float)):
            total_prompt += int(prompt)
        if isinstance(completion, (int, float)):
            total_completion += int(completion)
        if isinstance(total, (int, float)):
            total_tokens += int(total)
    if total_tokens <= 0:
        total_tokens = total_prompt + total_completion
    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
    }


def _attach_metadata(raw: Any, metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        enriched = dict(raw)
        enriched["_metadata"] = dict(metadata)
        return enriched
    return {"raw_output": raw, "_metadata": dict(metadata)}


def call_chat_completion_response(
    *,
    base_url: str,
    model: str,
    question: str,
    temperature: float,
    timeout: float,
    max_tokens: int,
    api_key: str = "",
    request_retries: int = 6,
    retry_sleep_seconds: float = 2.0,
    qwen_disable_thinking: bool = False,
) -> Mapping[str, Any]:
    endpoint = str(base_url).rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ],
    }
    if qwen_disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    attempts = max(1, int(request_retries))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                raw = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504} or attempt + 1 >= attempts:
                raise
            time.sleep(float(retry_sleep_seconds) * (attempt + 1))
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                raise
            time.sleep(float(retry_sleep_seconds) * (attempt + 1))
    else:
        raise RuntimeError(f"chat completion request failed: {last_error}")
    payload = json.loads(raw)
    content = str(payload["choices"][0]["message"]["content"] or "")
    return content, _usage_from_payload(payload)


def call_chat_completion_content(
    *,
    base_url: str,
    model: str,
    question: str,
    temperature: float,
    timeout: float,
    max_tokens: int,
    api_key: str = "",
    request_retries: int = 6,
    retry_sleep_seconds: float = 2.0,
    qwen_disable_thinking: bool = False,
) -> str:
    content, _ = call_chat_completion_response(
        base_url=base_url,
        model=model,
        question=question,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
        api_key=api_key,
        request_retries=request_retries,
        retry_sleep_seconds=retry_sleep_seconds,
        qwen_disable_thinking=qwen_disable_thinking,
    )
    return content


def call_chat_completion(
    *,
    base_url: str,
    model: str,
    question: str,
    temperature: float,
    timeout: float,
    max_tokens: int,
    api_key: str = "",
    request_retries: int = 6,
    retry_sleep_seconds: float = 2.0,
    qwen_disable_thinking: bool = False,
) -> Mapping[str, Any]:
    content = call_chat_completion_content(
        base_url=base_url,
        model=model,
        question=question,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
        api_key=api_key,
        request_retries=request_retries,
        retry_sleep_seconds=retry_sleep_seconds,
        qwen_disable_thinking=qwen_disable_thinking,
    )
    return extract_json_object(content)


def load_existing(path: str | Path) -> Dict[str, Any]:
    if not Path(path).exists():
        return {"roles_by_query": {}, "raw_outputs": {}, "usage_by_query": {}}
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        return {"roles_by_query": {}, "raw_outputs": {}, "usage_by_query": {}}
    return {
        "roles_by_query": dict(payload.get("roles_by_query", {}) or {}),
        "raw_outputs": dict(payload.get("raw_outputs", {}) or {}),
        "usage_by_query": dict(payload.get("usage_by_query", {}) or {}),
    }


def save_role_cache(
    *,
    path: str | Path,
    args: argparse.Namespace,
    roles_by_query: Mapping[str, Any],
    raw_outputs: Mapping[str, Any],
    usage_by_query: Mapping[str, Any] | None = None,
) -> None:
    usage_payload = dict(usage_by_query or {})
    usage_totals = _sum_usage(
        usage.get("total", usage) if isinstance(usage, Mapping) else {}
        for usage in usage_payload.values()
    )
    save_json(
        path,
        {
            "config": {
                "dataset_json": args.dataset_json,
                "limit": args.limit,
                "model": args.model,
                "base_url": args.base_url,
                "temperature": args.temperature,
                "timeout": args.timeout,
                "max_tokens": args.max_tokens,
                "concurrency": args.concurrency,
                "checkpoint_every": args.checkpoint_every,
                "parse_retries": args.parse_retries,
                "request_retries": args.request_retries,
                "retry_sleep_seconds": args.retry_sleep_seconds,
                "api_key_file": args.api_key_file,
                "api_key_provided": bool(args.api_key or args.api_key_file),
                "qwen_disable_thinking": bool(args.qwen_disable_thinking),
                "strict_question_grounding": bool(args.strict_question_grounding),
                "prompt": SYSTEM_PROMPT,
            },
            "roles_by_query": dict(roles_by_query),
            "raw_outputs": dict(raw_outputs),
            "usage_by_query": usage_payload,
            "usage_totals": usage_totals,
        },
    )


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args)
    dataset = load_json(args.dataset_json)
    if not isinstance(dataset, list):
        raise ValueError(f"Dataset must be a list: {args.dataset_json}")
    rows = dataset[: int(args.limit)] if args.limit is not None and int(args.limit) >= 0 else dataset
    output = load_existing(args.output_json) if args.resume else {"roles_by_query": {}, "raw_outputs": {}, "usage_by_query": {}}
    roles_by_query: Dict[str, Any] = dict(output.get("roles_by_query", {}) or {})
    raw_outputs: Dict[str, Any] = dict(output.get("raw_outputs", {}) or {})
    usage_by_query: Dict[str, Any] = dict(output.get("usage_by_query", {}) or {})

    pending: List[tuple[int, Mapping[str, Any]]] = []
    for query_index, row in enumerate(rows):
        key = str(query_index)
        if key in roles_by_query and roles_by_query[key]:
            continue
        pending.append((query_index, row))

    def process_one(query_index: int, row: Mapping[str, Any]) -> tuple[str, List[Dict[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
        key = str(query_index)
        question = str(row.get("question") or row.get("query") or "").strip()
        if not question:
            return key, [], {"error": "empty question"}, {"attempts": [], "total": {}}
        last_error: Exception | None = None
        last_content = ""
        usage_attempts: List[Mapping[str, Any]] = []
        try:
            for attempt in range(max(1, int(args.parse_retries))):
                last_content, usage = call_chat_completion_response(
                    base_url=args.base_url,
                    model=args.model,
                    question=question,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                    api_key=api_key,
                    request_retries=args.request_retries,
                    retry_sleep_seconds=args.retry_sleep_seconds,
                    qwen_disable_thinking=bool(args.qwen_disable_thinking),
                )
                if usage:
                    usage_attempts.append({"attempt": attempt, **usage})
                try:
                    raw = extract_json_object(last_content)
                    roles = normalize_roles(
                        raw,
                        question=question,
                        strict_question_grounding=bool(args.strict_question_grounding),
                    )
                    if roles:
                        usage_total = _sum_usage(usage_attempts)
                        metadata = {"usage": usage_total, "usage_attempts": list(usage_attempts)}
                        return key, roles, _attach_metadata(raw, metadata), {"attempts": list(usage_attempts), "total": usage_total}
                    last_error = ValueError("parsed JSON contained no valid roles")
                except ValueError as exc:
                    last_error = exc
                if attempt + 1 < max(1, int(args.parse_retries)):
                    time.sleep(0.25 * (attempt + 1))
            usage_total = _sum_usage(usage_attempts)
            metadata = {"usage": usage_total, "usage_attempts": list(usage_attempts)}
            return key, fallback_roles_for_question(question), {
                "error": str(last_error) if last_error is not None else "role parsing failed",
                "question": question,
                "raw_content": last_content,
                "fallback_role": True,
                "_metadata": metadata,
            }, {"attempts": list(usage_attempts), "total": usage_total}
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            usage_total = _sum_usage(usage_attempts)
            metadata = {"usage": usage_total, "usage_attempts": list(usage_attempts)}
            return key, fallback_roles_for_question(question), {
                "error": str(exc),
                "question": question,
                "fallback_role": True,
                "_metadata": metadata,
            }, {"attempts": list(usage_attempts), "total": usage_total}

    total = len(pending)
    completed = 0
    checkpoint_every = max(1, int(args.checkpoint_every))
    concurrency = max(1, int(args.concurrency))
    print(f"[role-cache] pending={total} concurrency={concurrency}", flush=True)

    if concurrency == 1:
        for query_index, row in pending:
            key, roles, raw, usage = process_one(query_index, row)
            roles_by_query[key] = roles
            raw_outputs[key] = raw
            usage_by_query[key] = usage
            completed += 1
            if args.sleep_seconds > 0:
                time.sleep(float(args.sleep_seconds))
            if completed % checkpoint_every == 0 or completed == total:
                save_role_cache(path=args.output_json, args=args, roles_by_query=roles_by_query, raw_outputs=raw_outputs, usage_by_query=usage_by_query)
                print(f"[role-cache] completed {completed}/{total}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_query = {executor.submit(process_one, query_index, row): query_index for query_index, row in pending}
            for future in as_completed(future_to_query):
                key, roles, raw, usage = future.result()
                roles_by_query[key] = roles
                raw_outputs[key] = raw
                usage_by_query[key] = usage
                completed += 1
                if completed % checkpoint_every == 0 or completed == total:
                    save_role_cache(path=args.output_json, args=args, roles_by_query=roles_by_query, raw_outputs=raw_outputs, usage_by_query=usage_by_query)
                    print(f"[role-cache] completed {completed}/{total}", flush=True)

    save_role_cache(path=args.output_json, args=args, roles_by_query=roles_by_query, raw_outputs=raw_outputs, usage_by_query=usage_by_query)


if __name__ == "__main__":
    main()
