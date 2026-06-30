import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .utils.misc_utils import compute_mdhash_id, text_processing


@dataclass
class PropositionCandidate:
    hash_id: str
    summary_text: str
    relation_type: str
    participant_ids: List[str]
    participant_texts: List[str]
    participant_roles: Dict[str, List[str]]
    source_ids: List[str]
    support_count: int
    last_seen: int
    embedding_text: str
    evidence_snippets: List[str]
    embedding_hash_id: str
    fact_embedding_hash_id: str
    conflict_group: str
    proposition_type: str = "triple"
    normalized_text: str = ""


def _normalize_token(text: str) -> str:
    processed = text_processing(text)
    if not isinstance(processed, str):
        return ""
    return " ".join(processed.split())


def _normalize_free_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _normalize_signature_text(text: str) -> str:
    processed = text_processing(text)
    if not isinstance(processed, str):
        return ""
    return " ".join(processed.split())


def _canonicalize_entities(entities: Iterable[str]) -> List[str]:
    canonical_entities: List[str] = []
    for entity in entities or []:
        normalized = _normalize_token(entity)
        if normalized and normalized not in canonical_entities:
            canonical_entities.append(normalized)
    return canonical_entities


def _build_default_embedding_text(
    summary_text: str,
    relation_type: str,
    participant_texts: List[str],
    evidence_snippets: List[str],
    support_count: int,
) -> str:
    if evidence_snippets:
        return build_proposition_embedding_text(
            summary_text=summary_text,
            relation_type=relation_type,
            participant_texts=participant_texts,
            evidence_snippets=evidence_snippets,
            support_count=support_count,
        )
    return summary_text


def extract_evidence_snippet(text: str, max_chars: int = 240) -> str:
    normalized = " ".join(str(text).split())
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def build_proposition_embedding_text(
    summary_text: str,
    relation_type: str,
    participant_texts: List[str],
    evidence_snippets: List[str],
    support_count: int,
) -> str:
    participant_texts = list(participant_texts or [])
    subject_text = participant_texts[0] if len(participant_texts) > 0 else ""
    object_text = participant_texts[1] if len(participant_texts) > 1 else ""
    is_structured_triple = bool(
        relation_type
        and relation_type != "proposition"
        and len(participant_texts) <= 2
    )

    lines = [f"Proposition: {summary_text}"]
    if relation_type:
        lines.append(f"Relation: {relation_type}")
    if is_structured_triple:
        if subject_text:
            lines.append(f"Subject: {subject_text}")
        if object_text:
            lines.append(f"Object: {object_text}")
    elif participant_texts:
        lines.append("Entities: " + ", ".join(participant_texts))
    lines.append(f"Support Count: {max(int(support_count), 1)}")

    cleaned_snippets = []
    for snippet in evidence_snippets or []:
        normalized = " ".join(str(snippet).split())
        if normalized and normalized not in cleaned_snippets:
            cleaned_snippets.append(normalized)
    if cleaned_snippets:
        lines.append("Evidence:")
        for idx, snippet in enumerate(cleaned_snippets, start=1):
            lines.append(f"{idx}. {snippet}")

    return "\n".join(lines)


def build_proposition_fact_embedding_text(
    relation_type: str,
    participant_texts: List[str],
    normalized_text: str = "",
    summary_text: str = "",
) -> str:
    participant_texts = [str(value) for value in list(participant_texts or [])]
    normalized_relation = str(relation_type or "").strip()
    normalized_fact_text = str(normalized_text or "").strip()
    normalized_summary_text = str(summary_text or "").strip()

    if normalized_relation and normalized_relation != "proposition" and len(participant_texts) >= 2:
        return str((participant_texts[0], normalized_relation, participant_texts[1]))
    return normalized_fact_text or normalized_summary_text


def build_proposition_embedding_text_variants(
    summary_text: str,
    relation_type: str,
    participant_texts: List[str],
    evidence_snippets: List[str],
    support_count: int,
) -> List[str]:
    participant_texts = list(participant_texts or [])
    subject_text = participant_texts[0] if len(participant_texts) > 0 else ""
    object_text = participant_texts[1] if len(participant_texts) > 1 else ""
    is_structured_triple = bool(
        relation_type
        and relation_type != "proposition"
        and len(participant_texts) <= 2
    )

    variants: List[str] = []
    if summary_text:
        variants.append(summary_text)

    structured_text = build_proposition_embedding_text(
        summary_text=summary_text,
        relation_type=relation_type,
        participant_texts=participant_texts,
        evidence_snippets=[],
        support_count=support_count,
    )
    if structured_text:
        variants.append(structured_text)

    if is_structured_triple and (object_text or subject_text):
        object_first_lines = [
            f"Object-Centric Proposition: {object_text}" if object_text else "Object-Centric Proposition:",
            f"Relation: {relation_type}",
        ]
        if subject_text:
            object_first_lines.append(f"Subject: {subject_text}")
        object_first_text = "\n".join(line for line in object_first_lines if line)
        if object_first_text:
            variants.append(object_first_text)
    elif participant_texts:
        variants.append("Entity-Centric Proposition: " + ", ".join(participant_texts))

    if evidence_snippets:
        contextual_text = build_proposition_embedding_text(
            summary_text=summary_text,
            relation_type=relation_type,
            participant_texts=participant_texts,
            evidence_snippets=evidence_snippets,
            support_count=support_count,
        )
        if contextual_text:
            variants.append(contextual_text)

    deduped_variants = []
    for variant in variants:
        normalized = "\n".join(line.rstrip() for line in str(variant).splitlines()).strip()
        if normalized and normalized not in deduped_variants:
            deduped_variants.append(normalized)
    return deduped_variants


def build_proposition_candidates(
    chunk_ids: List[str],
    chunk_triples: List[List[Tuple[str, str, str]]],
    step_id: int,
    chunk_id_to_text: Dict[str, str] | None = None,
    evidence_max_chars: int = 240,
) -> Tuple[List[PropositionCandidate], Dict[str, List[str]], Dict[str, List[str]]]:
    """Build canonical proposition candidates from extracted triples."""

    candidates: List[PropositionCandidate] = []
    chunk_to_hids: Dict[str, List[str]] = defaultdict(list)
    chunk_to_eids: Dict[str, List[str]] = defaultdict(list)

    for chunk_id, triples in zip(chunk_ids, chunk_triples):
        for triple in triples:
            if len(triple) != 3:
                continue

            subject = _normalize_token(triple[0])
            relation = _normalize_token(triple[1])
            obj = _normalize_token(triple[2])

            if not subject or not relation or not obj:
                continue

            subject_id = compute_mdhash_id(subject, prefix="entity-")
            object_id = compute_mdhash_id(obj, prefix="entity-")
            signature = json.dumps(
                {"subject_id": subject_id, "relation_type": relation, "object_id": object_id},
                ensure_ascii=True,
                sort_keys=True,
            )
            hash_id = compute_mdhash_id(signature, prefix="hyperedge-")
            summary_text = f"{subject} [{relation}] {obj}"
            evidence_snippets = []
            if chunk_id_to_text is not None and chunk_id in chunk_id_to_text:
                snippet = extract_evidence_snippet(chunk_id_to_text[chunk_id], max_chars=evidence_max_chars)
                if snippet:
                    evidence_snippets.append(snippet)
            embedding_text = _build_default_embedding_text(
                summary_text=summary_text,
                relation_type=relation,
                participant_texts=[subject, obj],
                evidence_snippets=evidence_snippets,
                support_count=1,
            )
            conflict_group = compute_mdhash_id(
                json.dumps(sorted([subject_id, object_id]), ensure_ascii=True),
                prefix="conflict-",
            )

            candidate = PropositionCandidate(
                hash_id=hash_id,
                summary_text=summary_text,
                relation_type=relation,
                participant_ids=[subject_id, object_id],
                participant_texts=[subject, obj],
                participant_roles={
                    subject_id: ["subject"],
                    object_id: ["object"],
                },
                source_ids=[chunk_id],
                support_count=1,
                last_seen=step_id,
                embedding_text=embedding_text,
                evidence_snippets=evidence_snippets,
                embedding_hash_id=compute_mdhash_id(embedding_text, prefix="hyperedge-"),
                fact_embedding_hash_id=compute_mdhash_id(
                    build_proposition_fact_embedding_text(
                        relation_type=relation,
                        participant_texts=[subject, obj],
                        summary_text=summary_text,
                    ),
                    prefix="fact-",
                ),
                conflict_group=conflict_group,
                proposition_type="triple",
                normalized_text=summary_text,
            )
            candidates.append(candidate)
            chunk_to_hids[chunk_id].append(hash_id)
            chunk_to_eids[chunk_id].extend([subject_id, object_id])

    chunk_to_eids = {key: sorted(set(value)) for key, value in chunk_to_eids.items()}
    chunk_to_hids = {key: sorted(set(value)) for key, value in chunk_to_hids.items()}
    return candidates, chunk_to_hids, chunk_to_eids


def build_proposition_candidates_from_units(
    chunk_ids: List[str],
    chunk_propositions: List[List[Dict]],
    step_id: int,
    chunk_id_to_text: Dict[str, str] | None = None,
    evidence_max_chars: int = 240,
) -> Tuple[List[PropositionCandidate], Dict[str, List[str]], Dict[str, List[str]]]:
    candidates: List[PropositionCandidate] = []
    chunk_to_hids: Dict[str, List[str]] = defaultdict(list)
    chunk_to_eids: Dict[str, List[str]] = defaultdict(list)

    for chunk_id, propositions in zip(chunk_ids, chunk_propositions):
        for proposition in propositions:
            if not isinstance(proposition, dict):
                continue

            summary_text = _normalize_free_text(proposition.get("text", ""))
            if not summary_text:
                continue

            participant_texts = _canonicalize_entities(proposition.get("entities", []))
            participant_ids = [compute_mdhash_id(entity, prefix="entity-") for entity in participant_texts]
            normalized_text = _normalize_signature_text(summary_text)
            signature = json.dumps(
                {
                    "normalized_text": normalized_text,
                    "participant_ids": participant_ids,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            hash_id = compute_mdhash_id(signature, prefix="hyperedge-")

            evidence_snippets = []
            if chunk_id_to_text is not None and chunk_id in chunk_id_to_text:
                snippet = extract_evidence_snippet(chunk_id_to_text[chunk_id], max_chars=evidence_max_chars)
                if snippet:
                    evidence_snippets.append(snippet)

            embedding_text = _build_default_embedding_text(
                summary_text=summary_text,
                relation_type="proposition",
                participant_texts=participant_texts,
                evidence_snippets=evidence_snippets,
                support_count=1,
            )
            conflict_payload = participant_ids if participant_ids else [normalized_text]
            conflict_group = compute_mdhash_id(
                json.dumps(sorted(conflict_payload), ensure_ascii=True),
                prefix="conflict-",
            )
            participant_roles = {participant_id: ["participant"] for participant_id in participant_ids}

            candidate = PropositionCandidate(
                hash_id=hash_id,
                summary_text=summary_text,
                relation_type="proposition",
                participant_ids=participant_ids,
                participant_texts=participant_texts,
                participant_roles=participant_roles,
                source_ids=[chunk_id],
                support_count=1,
                last_seen=step_id,
                embedding_text=embedding_text,
                evidence_snippets=evidence_snippets,
                embedding_hash_id=compute_mdhash_id(embedding_text, prefix="hyperedge-"),
                fact_embedding_hash_id=compute_mdhash_id(
                    build_proposition_fact_embedding_text(
                        relation_type="proposition",
                        participant_texts=participant_texts,
                        normalized_text=normalized_text,
                        summary_text=summary_text,
                    ),
                    prefix="fact-",
                ),
                conflict_group=conflict_group,
                proposition_type="proposition",
                normalized_text=normalized_text,
            )
            candidates.append(candidate)
            chunk_to_hids[chunk_id].append(hash_id)
            chunk_to_eids[chunk_id].extend(participant_ids)

    chunk_to_eids = {key: sorted(set(value)) for key, value in chunk_to_eids.items()}
    chunk_to_hids = {key: sorted(set(value)) for key, value in chunk_to_hids.items()}
    return candidates, chunk_to_hids, chunk_to_eids


def score_proposition_merge(candidate: PropositionCandidate, existing_record: Dict) -> float:
    """Simple merge score for exact proposition matches on canonicalized triples."""

    same_relation = float(candidate.relation_type == existing_record["relation_type"])
    same_participants = float(candidate.participant_ids == existing_record["participant_ids"])
    candidate_normalized = candidate.normalized_text or candidate.summary_text
    existing_normalized = existing_record.get("normalized_text", existing_record.get("summary_text", ""))
    same_summary = float(candidate_normalized == existing_normalized)
    return 0.4 * same_relation + 0.4 * same_participants + 0.2 * same_summary


def upsert_propositions(
    candidates: Iterable[PropositionCandidate],
    existing_records: Dict[str, Dict],
    merge_threshold: float,
) -> List[Dict]:
    """Merge candidates into proposition records using canonicalized signatures."""

    staged_records = {key: dict(value) for key, value in existing_records.items()}

    for candidate in candidates:
        existing_record = staged_records.get(candidate.hash_id)
        candidate_payload = {
            "hash_id": candidate.hash_id,
            "summary_text": candidate.summary_text,
            "relation_type": candidate.relation_type,
            "participant_ids": list(candidate.participant_ids),
            "participant_texts": list(candidate.participant_texts),
            "participant_roles": dict(candidate.participant_roles),
            "source_ids": list(candidate.source_ids),
            "support_count": candidate.support_count,
            "last_seen": candidate.last_seen,
            "embedding_text": candidate.embedding_text,
            "evidence_snippets": list(candidate.evidence_snippets),
            "embedding_hash_id": candidate.embedding_hash_id,
            "fact_embedding_hash_id": candidate.fact_embedding_hash_id,
            "conflict_group": candidate.conflict_group,
            "proposition_type": candidate.proposition_type,
            "normalized_text": candidate.normalized_text or candidate.summary_text,
        }

        if existing_record is None:
            staged_records[candidate.hash_id] = candidate_payload
            continue

        merge_score = score_proposition_merge(candidate, existing_record)
        if merge_score < merge_threshold:
            # Conservative fallback: keep deterministic id but preserve latest source signal.
            candidate_payload["support_count"] = max(1, candidate_payload["support_count"])
            staged_records[candidate.hash_id] = candidate_payload
            continue

        merged_source_ids = sorted(set(existing_record.get("source_ids", [])) | set(candidate.source_ids))
        merged_support_count = existing_record.get("support_count", 0) + candidate.support_count
        merged_evidence_snippets = []
        for snippet in list(existing_record.get("evidence_snippets", [])) + list(candidate.evidence_snippets):
            normalized = " ".join(str(snippet).split())
            if normalized and normalized not in merged_evidence_snippets:
                merged_evidence_snippets.append(normalized)
        merged_embedding_text = _build_default_embedding_text(
            summary_text=candidate.summary_text,
            relation_type=candidate.relation_type,
            participant_texts=list(candidate.participant_texts),
            evidence_snippets=merged_evidence_snippets,
            support_count=merged_support_count,
        )
        merged_roles = {
            participant_id: sorted(
                set(existing_record.get("participant_roles", {}).get(participant_id, []))
                | set(candidate.participant_roles.get(participant_id, []))
            )
            for participant_id in candidate.participant_ids
        }

        existing_record.update(
            {
                "source_ids": merged_source_ids,
                "support_count": merged_support_count,
                "last_seen": max(existing_record.get("last_seen", 0), candidate.last_seen),
                "participant_roles": merged_roles,
                "embedding_text": merged_embedding_text,
                "evidence_snippets": merged_evidence_snippets,
                "embedding_hash_id": compute_mdhash_id(merged_embedding_text, prefix="hyperedge-"),
                "fact_embedding_hash_id": candidate.fact_embedding_hash_id,
                "conflict_group": existing_record.get("conflict_group", candidate.conflict_group),
                "proposition_type": existing_record.get("proposition_type", candidate.proposition_type),
                "normalized_text": existing_record.get("normalized_text", candidate.normalized_text or candidate.summary_text),
            }
        )
        staged_records[candidate.hash_id] = existing_record

    return list(staged_records.values())
