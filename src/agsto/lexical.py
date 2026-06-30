"""Lexical document scoring over the AG-STO STO index."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Set

from .index import content_tokens


def score_docs_bm25(
    *,
    query: str,
    corpus_index: Mapping[str, Any],
    k1: float = 1.2,
    b: float = 0.75,
) -> Dict[int, float]:
    """Document-level BM25 scores over the STO-derived token inventory."""

    query_tokens = content_tokens(query)
    if not query_tokens:
        return {}
    token_to_docs = corpus_index["token_to_docs"]
    doc_token_counts: Mapping[int, Counter[str]] = corpus_index["doc_token_counts"]
    doc_token_idf: Mapping[str, float] = corpus_index["doc_token_idf"]
    doc_lengths: Mapping[int, int] = corpus_index["doc_lengths"]
    doc_title_tokens: Mapping[int, Sequence[str]] = corpus_index["doc_title_tokens"]
    avg_doc_length = float(corpus_index.get("avg_doc_length", 1.0) or 1.0)

    candidate_docs: Set[int] = set()
    for token in query_tokens:
        candidate_docs.update(int(doc_idx) for doc_idx in token_to_docs.get(token, []) or [])

    doc_scores: Dict[int, float] = {}
    for doc_idx in candidate_docs:
        counts = doc_token_counts.get(doc_idx, Counter())
        doc_len = float(doc_lengths.get(doc_idx, 0) or 0)
        score = 0.0
        for token in query_tokens:
            tf = float(counts.get(token, 0))
            if tf <= 0.0:
                continue
            idf = float(doc_token_idf.get(token, 0.0))
            denom = tf + k1 * (1.0 - b + b * doc_len / avg_doc_length)
            score += idf * (tf * (k1 + 1.0)) / max(denom, 1e-9)
        title_overlap = query_tokens & set(doc_title_tokens.get(doc_idx, []) or [])
        score += 1.25 * sum(float(doc_token_idf.get(token, 0.0)) for token in title_overlap)
        if score > 0.0:
            doc_scores[doc_idx] = score
    return doc_scores


def rank_docs_bm25(
    *,
    query: str,
    corpus_index: Mapping[str, Any],
    top_k: int,
) -> List[int]:
    """Rank documents by BM25 over the STO-derived token inventory."""

    doc_scores = score_docs_bm25(query=query, corpus_index=corpus_index)
    return [
        doc_idx
        for doc_idx, _ in sorted(doc_scores.items(), key=lambda item: (-float(item[1]), item[0]))[:top_k]
    ]
