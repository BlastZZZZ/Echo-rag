"""Ranking helpers for AG-STO evidence selection."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set


def unique_ranked(values: Iterable[int]) -> List[int]:
    """Return non-negative document ids in first-seen order without duplicates."""

    seen: Set[int] = set()
    result: List[int] = []
    for value in values:
        doc_idx = int(value)
        if doc_idx < 0 or doc_idx in seen:
            continue
        seen.add(doc_idx)
        result.append(doc_idx)
    return result


def rank_sto_proposal_consensus_docs(
    *,
    proposal_doc_indices: Sequence[Sequence[int]],
    proposal_weights: Sequence[float] | None = None,
    top_k: int,
) -> List[int]:
    """Rank docs by agreement among STO evidence-set proposals."""

    scores: Counter[int] = Counter()
    best_rank: Dict[int, int] = {}
    channel_count: Counter[int] = Counter()
    weights = list(proposal_weights or [])
    if len(weights) < len(proposal_doc_indices):
        weights.extend([1.0] * (len(proposal_doc_indices) - len(weights)))
    for proposal_index, proposal in enumerate(proposal_doc_indices):
        proposal_weight = max(float(weights[proposal_index]), 0.0)
        seen_in_channel: Set[int] = set()
        for rank, doc_idx in enumerate(unique_ranked(proposal), start=1):
            doc_idx = int(doc_idx)
            scores[doc_idx] += proposal_weight / float(rank + 1)
            best_rank[doc_idx] = min(int(best_rank.get(doc_idx, 10**9)), int(rank))
            if doc_idx not in seen_in_channel:
                channel_count[doc_idx] += 1
                seen_in_channel.add(doc_idx)
    return [
        doc_idx
        for doc_idx, _ in sorted(
            scores.items(),
            key=lambda item: (
                -float(item[1]),
                -int(channel_count.get(int(item[0]), 0)),
                int(best_rank.get(int(item[0]), 10**9)),
                int(item[0]),
            ),
        )[: max(int(top_k), 1)]
    ]
