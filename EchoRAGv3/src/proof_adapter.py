#!/usr/bin/env python3
"""Export typed proof-chain rows as proof-state annotations.

This does not rerun retrieval, call a reader, train a ranker, or use gold
answers at runtime.  It only projects an existing typed proof-chain diagnostic
row into:

1. EchoRAG-ProofAnnotation: a query-local source-grounded annotation over the
   existing evidence context.
2. ComoRAG-ProofStateRecord: a proof-state memory record that exposes
   obligations, grounded atoms, proof links, and unresolved obligations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed_chain_json", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--save_echorag_jsonl", required=True)
    parser.add_argument("--save_comorag_jsonl", required=True)
    parser.add_argument("--save_summary_json", required=True)
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = " ".join(str(item or "").split())
        if text and text not in out:
            out.append(text)
    return out


def selected_doc_index_set(row: Mapping[str, Any]) -> set[int]:
    out: set[int] = set()
    for item in row.get("selected_doc_indices") or []:
        try:
            out.add(int(item))
        except Exception:
            continue
    return out


def role_id(role: Mapping[str, Any]) -> str:
    return str(role.get("role_id") or "").strip()


def proof_atoms(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    selected_docs = selected_doc_index_set(row)
    for atom in row.get("typed_atom_proof_atoms") or row.get("typed_atoms") or []:
        if not isinstance(atom, Mapping):
            continue
        if not atom.get("source_authorization") or not str(atom.get("exact_source_span") or "").strip():
            continue
        if selected_docs:
            try:
                source_doc_index = int(atom.get("source_doc_index"))
            except Exception:
                continue
            if source_doc_index not in selected_docs:
                continue
        atoms.append(
            {
                "atom_index": atom.get("atom_index"),
                "role_id": atom.get("role_id"),
                "source_doc_index": atom.get("source_doc_index"),
                "subject_values": string_list(atom.get("subject_values")),
                "relation_text": " ".join(str(atom.get("relation_text") or "").split()),
                "object_values": string_list(atom.get("object_values")),
                "consumes": [
                    {"variable": item.get("variable"), "value": item.get("value")}
                    for item in atom.get("consumes") or []
                    if isinstance(item, Mapping)
                ],
                "produces": [
                    {"variable": item.get("variable"), "value": item.get("value")}
                    for item in atom.get("produces") or []
                    if isinstance(item, Mapping)
                ],
                "source_authorization": atom.get("source_authorization"),
                "source_operator": atom.get("source_operator"),
                "derived_atom": bool(atom.get("derived_atom")),
                "derived_from_atom_indices": list(atom.get("derived_from_atom_indices") or []),
                "comparison_operation": dict(atom.get("comparison_operation") or {}),
                "comparison_inputs": list(atom.get("comparison_inputs") or []),
                "exact_source_span": " ".join(str(atom.get("exact_source_span") or "").split()),
            }
        )
    return atoms


def proof_doc_indices(atoms: Sequence[Mapping[str, Any]]) -> list[int]:
    out: list[int] = []
    for atom in atoms:
        try:
            doc_idx = int(atom.get("source_doc_index"))
        except Exception:
            continue
        if doc_idx not in out:
            out.append(doc_idx)
    return out


def role_obligations(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for diag in row.get("role_diagnostics") or row.get("roles") or []:
        if not isinstance(diag, Mapping):
            continue
        rid = str(diag.get("role_id") or "")
        if not rid:
            continue
        out.append(
            {
                "role_id": rid,
                "role_type": diag.get("role_type"),
                "operator": diag.get("operator"),
                "description": diag.get("description"),
                "inputs": list(diag.get("inputs") or []),
                "outputs": list(diag.get("outputs") or []),
            }
        )
    return out


def missing_obligations(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    selector_meta = row.get("ta_preselector")
    coverage = selector_meta if isinstance(selector_meta, Mapping) else row.get("typed_atom_chain_role_coverage")
    missing_ids = []
    if isinstance(coverage, Mapping):
        missing_ids = list(coverage.get("missing_required_role_ids") or [])
    by_id = {item["role_id"]: item for item in role_obligations(row)}
    return [by_id.get(str(rid), {"role_id": str(rid)}) for rid in missing_ids]


def terminal_outputs(row: Mapping[str, Any]) -> list[str]:
    values = string_list(row.get("typed_atom_complete_terminal_outputs"))
    if not values:
        values = string_list(row.get("typed_atom_terminal_outputs"))
    return values


def complete(row: Mapping[str, Any]) -> bool:
    return bool(row.get("typed_atom_chain_complete"))


def proof_annotation_status(row: Mapping[str, Any], atoms: Sequence[Mapping[str, Any]]) -> str:
    if complete(row) and atoms:
        return "complete_annotation"
    if row.get("typed_atom_connected_chain"):
        return "partial_annotation_missing_obligations"
    if atoms:
        return "disconnected_source_closed_atoms"
    return "no_source_closed_atoms"


def proof_memory_status(row: Mapping[str, Any], atoms: Sequence[Mapping[str, Any]]) -> str:
    if not complete(row) or not atoms:
        return "partial_proof_state_record"
    if len(terminal_outputs(row)) > 1 or row.get("typed_atom_ambiguous_complete_terminal_set"):
        return "ambiguous_terminal_proof_state_record"
    return "complete_proof_state_record"


def source_closed_atoms(atoms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for atom in atoms:
        status = "derived" if atom.get("derived_atom") else ("closed" if atom.get("source_authorization") else "open")
        rows.append(
            {
                "role_id": atom.get("role_id"),
                "source_doc_index": atom.get("source_doc_index"),
                "subject_values": atom.get("subject_values", []),
                "relation_text": atom.get("relation_text"),
                "object_values": atom.get("object_values", []),
                "source_close_status": status,
                "source_authorization": atom.get("source_authorization"),
                "derived_atom": bool(atom.get("derived_atom")),
                "derived_from_atom_indices": list(atom.get("derived_from_atom_indices") or []),
            }
        )
    return rows


def echorag_view(row: Mapping[str, Any], *, dataset: str) -> dict[str, Any]:
    atoms = proof_atoms(row)
    return {
        "algorithm": "EchoRAG-ProofAnnotation",
        "dataset": dataset,
        "query_index": row.get("query_index"),
        "question": row.get("question"),
        "proof_annotation_status": proof_annotation_status(row, atoms),
        "selected_doc_indices": list(row.get("selected_doc_indices") or []),
        "proof_doc_indices": proof_doc_indices(atoms),
        "proof_complete": complete(row),
        "answer_candidates": terminal_outputs(row),
        "missing_obligations": missing_obligations(row),
        "proof_atoms": atoms,
    }


def comorag_view(row: Mapping[str, Any], *, dataset: str) -> dict[str, Any]:
    atoms = proof_atoms(row)
    obligations = role_obligations(row)
    missing = missing_obligations(row)
    return {
        "algorithm": "ComoRAG-ProofStateRecord",
        "source_id": f"{dataset}:{row.get('query_index')}",
        "dataset": dataset,
        "query_index": row.get("query_index"),
        "question": row.get("question"),
        "proof_memory_status": proof_memory_status(row, atoms),
        "memory_object": {
            "Z_obligations": obligations,
            "E_source_closed_atoms": source_closed_atoms(atoms),
            "P_proof_doc_indices": proof_doc_indices(atoms),
            "b_answerability": {
                "proof_complete": complete(row),
                "connected_chain": bool(row.get("typed_atom_connected_chain")),
                "ambiguous_terminal_set": bool(row.get("typed_atom_ambiguous_complete_terminal_set")),
                "answer_candidates": terminal_outputs(row),
            },
        },
        "missing_obligations": missing,
    }


def summarize(echo_rows: Sequence[Mapping[str, Any]], como_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "boundary": {
            "uses_gold_at_runtime": False,
            "reader_replay": False,
            "retrieval_expansion": False,
            "ranker_training": False,
        },
        "echorag_proof_annotation_status": dict(
            Counter(str(row.get("proof_annotation_status")) for row in echo_rows)
        ),
        "comorag_proof_memory_status": dict(Counter(str(row.get("proof_memory_status")) for row in como_rows)),
        "rows": len(echo_rows),
    }


def main() -> None:
    args = parse_args()
    payload = read_json(args.typed_chain_json)
    dataset = str(args.dataset or payload.get("dataset") or "")
    rows = [row for row in payload.get("rows", []) or [] if isinstance(row, Mapping)]
    echo_rows = [echorag_view(row, dataset=dataset) for row in rows]
    como_rows = [comorag_view(row, dataset=dataset) for row in rows]
    write_jsonl(args.save_echorag_jsonl, echo_rows)
    write_jsonl(args.save_comorag_jsonl, como_rows)
    write_json(args.save_summary_json, summarize(echo_rows, como_rows))
    print(f"[proof-adapters] wrote {args.save_echorag_jsonl}")
    print(f"[proof-adapters] wrote {args.save_comorag_jsonl}")
    print(f"[proof-adapters] wrote {args.save_summary_json}")


if __name__ == "__main__":
    main()
