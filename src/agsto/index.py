"""STO corpus indexing utilities for AG-STO."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple


WEAK_ENDPOINTS = {
    "he",
    "she",
    "it",
    "they",
    "him",
    "her",
    "them",
    "the river",
    "the city",
    "the town",
    "the village",
    "the state",
    "the country",
    "the region",
    "the province",
    "the district",
    "the county",
    "the company",
    "the club",
    "the team",
    "the band",
    "the film",
    "the album",
    "the song",
    "the book",
    "the novel",
    "the play",
    "the church",
    "the cathedral",
    "the school",
    "the university",
    "the station",
    "the airport",
    "the building",
    "the series",
}


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "both",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "many",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(value: Any) -> set[str]:
    return {
        token
        for token in normalize_text(value).split()
        if len(token) >= 3 and token not in STOPWORDS
    }


def prefixed_title_body(text: str, title: str) -> Tuple[str, str] | None:
    clean_title = str(title or "").strip()
    clean_text = str(text or "").strip()
    if not clean_title or not clean_text.startswith(clean_title):
        return None
    if len(clean_text) > len(clean_title) and clean_text[len(clean_title)].isalnum():
        return None
    body = clean_text[len(clean_title) :].strip(" \t\n\r:-")
    return clean_title, body


def split_title_and_body(
    passage: str,
    *,
    entities: Sequence[str] | None = None,
    explicit_title: str | None = None,
) -> Tuple[str, str]:
    text = str(passage or "").strip()
    if not text:
        return "", ""

    if explicit_title:
        prefixed = prefixed_title_body(text, explicit_title)
        return prefixed if prefixed is not None else (str(explicit_title).strip(), text)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines[0], "\n".join(lines[1:]).strip()

    # Real OpenIE caches often store "Title Body..." on one line.  Promoting
    # the visible prefix entity to title keeps STO title grounding meaningful.
    single_line = lines[0] if lines else text
    for entity in sorted((str(value).strip() for value in entities or []), key=len, reverse=True):
        prefixed = prefixed_title_body(single_line, entity)
        if prefixed is not None:
            return prefixed
    return "", single_line


def split_source_spans(body: str, *, max_chars: int = 420) -> List[Dict[str, Any]]:
    text = re.sub(r"\s+", " ", str(body or "")).strip()
    if not text:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    spans: List[Dict[str, Any]] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > max_chars:
            spans.append({"span_index": len(spans), "span_text": current})
            current = sentence
        elif current:
            current = f"{current} {sentence}"
        else:
            current = sentence
    if current:
        spans.append({"span_index": len(spans), "span_text": current})
    return spans


def extracted_triples(doc: Mapping[str, Any]) -> List[Tuple[str, str, str]]:
    triples = doc.get("extracted_triples") or []
    clean: List[Tuple[str, str, str]] = []
    for triple in triples:
        if isinstance(triple, Sequence) and not isinstance(triple, (str, bytes)) and len(triple) >= 3:
            clean.append((str(triple[0]), str(triple[1]), str(triple[2])))
    return clean


def informative_endpoint(value: Any) -> str:
    normalized = normalize_text(value)
    if not normalized:
        return ""
    if normalized in WEAK_ENDPOINTS:
        return ""
    tokens = normalized.split()
    if not tokens:
        return ""
    if len(normalized) < 3 and not normalized.isdigit():
        return ""
    return normalized


def unit_title_tokens(unit: Mapping[str, Any]) -> Set[str]:
    return content_tokens(unit.get("title", ""))


def light_units_for_doc(openie_doc: Mapping[str, Any], *, doc_index: int) -> List[Dict[str, Any]]:
    """Build retrieval-only STO units without audit-only grounding fields."""

    chunk_id = str(openie_doc.get("idx") or f"doc-{doc_index}")
    passage = str(openie_doc.get("passage") or "")
    entities = [str(value) for value in openie_doc.get("extracted_entities", []) or [] if str(value).strip()]
    title, body = split_title_and_body(
        passage,
        entities=entities,
        explicit_title=str(openie_doc.get("title") or openie_doc.get("doc_title") or "").strip() or None,
    )
    if not body:
        body = passage
    spans = split_source_spans(body)
    if not spans and body:
        spans = [{"span_index": 0, "span_text": body}]

    units: List[Dict[str, Any]] = []
    for span in spans:
        span_text = str(span.get("span_text", ""))
        normalized_span = normalize_text(span_text)
        span_entities: List[str] = []
        seen_span_entities: Set[str] = set()
        for entity in entities:
            normalized_entity = normalize_text(entity)
            if not normalized_entity or normalized_entity in seen_span_entities:
                continue
            if f" {normalized_entity} " not in f" {normalized_span} ":
                continue
            span_entities.append(str(entity))
            seen_span_entities.add(normalized_entity)
        endpoint_entities: List[str] = []
        seen_endpoint_entities: Set[str] = set()
        for value in [title, *span_entities]:
            normalized_endpoint = normalize_text(value)
            if not normalized_endpoint or normalized_endpoint in seen_endpoint_entities:
                continue
            endpoint_entities.append(str(value))
            seen_endpoint_entities.add(normalized_endpoint)
        units.append(
            {
                "unit_type": "source_span",
                "doc_index": int(doc_index),
                "chunk_id": chunk_id,
                "title": title,
                "text": f"{title} {span_text}",
                "span_entities": span_entities,
                "endpoint_entities": endpoint_entities,
                "raw_endpoints": endpoint_entities,
            }
        )
    for triple_index, triple in enumerate(extracted_triples(openie_doc)):
        subject, relation, obj = triple
        units.append(
            {
                "unit_type": "openie_fact",
                "doc_index": int(doc_index),
                "chunk_id": chunk_id,
                "title": title,
                "triple_index": int(triple_index),
                "subject": subject,
                "relation": relation,
                "object": obj,
                "text": f"{title} {subject} {relation} {obj}",
                "raw_endpoints": [subject, obj, title],
            }
        )
    return units


def light_unit_endpoint_set(unit: Mapping[str, Any]) -> Set[str]:
    return {
        endpoint
        for endpoint in (informative_endpoint(value) for value in unit.get("raw_endpoints", []) or [])
        if endpoint
    }


def light_unit_text(unit: Mapping[str, Any]) -> str:
    return str(unit.get("text") or "")


def build_corpus_unit_index(openie_docs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build the lightweight Source/Title/OpenIE substrate used by AG-STO."""

    units: List[Dict[str, Any]] = []
    doc_to_units: Dict[int, List[int]] = defaultdict(list)
    endpoint_to_units: Dict[str, List[int]] = defaultdict(list)
    doc_to_endpoints: Dict[int, Set[str]] = defaultdict(set)
    endpoint_to_docs: Dict[str, Set[int]] = defaultdict(set)
    token_to_units: Dict[str, List[int]] = defaultdict(list)
    token_to_docs: Dict[str, Set[int]] = defaultdict(set)
    doc_token_counts: Dict[int, Counter[str]] = defaultdict(Counter)
    doc_title_tokens: Dict[int, Set[str]] = defaultdict(set)
    doc_endpoint_tokens: Dict[int, Set[str]] = defaultdict(set)
    doc_text_parts: Dict[int, List[str]] = defaultdict(list)
    doc_relation_text_parts: Dict[int, List[str]] = defaultdict(list)

    for doc_index, openie_doc in enumerate(openie_docs):
        doc_text_parts[int(doc_index)].append(str(openie_doc.get("passage") or ""))
        for local_unit in light_units_for_doc(openie_doc, doc_index=doc_index):
            unit_id = len(units)
            unit = dict(local_unit)
            endpoints = light_unit_endpoint_set(unit)
            tokens = content_tokens(light_unit_text(unit))
            endpoint_tokens: Set[str] = set()
            for endpoint in endpoints:
                endpoint_tokens.update(content_tokens(endpoint))
            unit["_unit_int_id"] = unit_id
            unit["_endpoints"] = sorted(endpoints)
            unit["_tokens"] = sorted(tokens)
            unit["_endpoint_tokens"] = sorted(endpoint_tokens)
            unit["_title_tokens"] = sorted(unit_title_tokens(unit))
            units.append(unit)
            doc_to_units[int(doc_index)].append(unit_id)
            doc_text_parts[int(doc_index)].append(light_unit_text(unit))
            if unit.get("unit_type") == "openie_fact":
                doc_relation_text_parts[int(doc_index)].append(
                    " ".join(
                        str(unit.get(key) or "")
                        for key in ("subject", "relation", "object")
                        if str(unit.get(key) or "").strip()
                    )
                )
            doc_token_counts[int(doc_index)].update(tokens)
            doc_title_tokens[int(doc_index)].update(unit.get("_title_tokens", []) or [])
            doc_endpoint_tokens[int(doc_index)].update(endpoint_tokens)
            for endpoint in endpoints:
                endpoint_to_units[endpoint].append(unit_id)
                doc_to_endpoints[int(doc_index)].add(endpoint)
                endpoint_to_docs[endpoint].add(int(doc_index))
            for token in tokens:
                token_to_units[token].append(unit_id)
                token_to_docs[token].add(int(doc_index))

    unit_count = max(len(units), 1)
    token_idf = {
        token: math.log((unit_count + 1.0) / (len(unit_ids) + 0.5)) + 1.0
        for token, unit_ids in token_to_units.items()
    }
    doc_count = max(len(openie_docs), 1)
    doc_token_idf = {
        token: math.log(1.0 + (doc_count - len(doc_ids) + 0.5) / (len(doc_ids) + 0.5))
        for token, doc_ids in token_to_docs.items()
    }
    endpoint_idf = {
        endpoint: math.log((doc_count + 1.0) / (len(doc_ids) + 0.5)) + 1.0
        for endpoint, doc_ids in endpoint_to_docs.items()
    }
    doc_lengths = {doc_idx: sum(counter.values()) for doc_idx, counter in doc_token_counts.items()}
    avg_doc_length = sum(doc_lengths.values()) / float(max(len(doc_lengths), 1))

    return {
        "units": units,
        "doc_to_units": dict(doc_to_units),
        "endpoint_to_units": dict(endpoint_to_units),
        "doc_to_endpoints": {doc_idx: sorted(endpoints) for doc_idx, endpoints in doc_to_endpoints.items()},
        "endpoint_to_docs": {endpoint: sorted(doc_ids) for endpoint, doc_ids in endpoint_to_docs.items()},
        "endpoint_idf": endpoint_idf,
        "token_to_units": dict(token_to_units),
        "token_to_docs": {token: sorted(doc_ids) for token, doc_ids in token_to_docs.items()},
        "token_idf": token_idf,
        "doc_token_idf": doc_token_idf,
        "doc_token_counts": dict(doc_token_counts),
        "doc_title_tokens": {doc_idx: sorted(tokens) for doc_idx, tokens in doc_title_tokens.items()},
        "doc_endpoint_tokens": {doc_idx: sorted(tokens) for doc_idx, tokens in doc_endpoint_tokens.items()},
        "doc_texts": {
            int(doc_idx): " ".join(part for part in parts if str(part).strip())
            for doc_idx, parts in doc_text_parts.items()
        },
        "doc_relation_texts": {
            int(doc_idx): " ".join(part for part in parts if str(part).strip())
            for doc_idx, parts in doc_relation_text_parts.items()
        },
        "doc_lengths": doc_lengths,
        "avg_doc_length": avg_doc_length,
    }
