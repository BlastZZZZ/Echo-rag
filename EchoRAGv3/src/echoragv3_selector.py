#!/usr/bin/env python3
"""EchoRAGv3: complete EchoRAGv2 Top-5 with typed-atom evidence roles.

EchoRAGv3 is the completion-only pre-reader version of EchoRAGv2-TA.  It does
not expand retrieval, train a selector, call an LLM, use gold answers for
selection, or add an answer gate.  It keeps the EchoRAGv2 Top-5 backbone and
uses source-grounded typed atoms only to complete missing required roles from
the existing candidate pool.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed-atoms-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--corpus-json", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["completion", "reselect"], default="completion")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def as_int_list(value: Any) -> list[int]:
    if value is None or isinstance(value, (str, bytes)):
        return []
    out: list[int] = []
    for item in value:
        try:
            number = int(item)
        except Exception:
            continue
        if number not in out:
            out.append(number)
    return out


def string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if value is None or isinstance(value, (str, bytes)):
        return []
    out: list[str] = []
    for item in value:
        text = " ".join(str(item or "").split())
        if text and text not in out:
            out.append(text)
    return out


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def normalize_for_answer(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def load_corpus(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    raw = read_json(path)
    if not isinstance(raw, list):
        return {}
    out: dict[int, str] = {}
    for position, row in enumerate(raw):
        if not isinstance(row, Mapping):
            continue
        title = str(row.get("title") or row.get("Title") or "").strip()
        text_value = row.get("text") if "text" in row else row.get("Text")
        if isinstance(text_value, list):
            text = " ".join(str(item) for item in text_value)
        else:
            text = str(text_value or "")
        passage = (title + "\n" + text).strip() if title else text.strip()
        out[position] = passage
        if isinstance(row.get("idx"), int):
            out[int(row["idx"])] = passage
    return out


def answer_present(answers: Sequence[str], passages: Sequence[str]) -> bool:
    haystack = normalize_for_answer(" ".join(passages))
    if not haystack:
        return False
    for answer in answers:
        needle = normalize_for_answer(answer)
        if needle and needle in haystack:
            return True
    return False


def required_role_ids(row: Mapping[str, Any]) -> list[str]:
    roles: list[str] = []
    for role in row.get("roles") or row.get("role_diagnostics") or []:
        if not isinstance(role, Mapping):
            continue
        if role.get("required", True) is False:
            continue
        rid = str(role.get("role_id") or "").strip()
        if rid and rid not in roles:
            roles.append(rid)
    if roles:
        return roles
    for atom in row.get("typed_atoms") or row.get("typed_atom_proof_atoms") or []:
        if not isinstance(atom, Mapping):
            continue
        rid = str(atom.get("role_id") or "").strip()
        if rid and rid not in roles:
            roles.append(rid)
    return roles


def source_grounded_atom(atom: Mapping[str, Any]) -> bool:
    return bool(atom.get("source_authorization")) and bool(str(atom.get("exact_source_span") or "").strip())


def atoms_by_doc(row: Mapping[str, Any], candidate_docs: Sequence[int]) -> dict[int, list[Mapping[str, Any]]]:
    candidate_set = set(candidate_docs)
    out: dict[int, list[Mapping[str, Any]]] = {doc: [] for doc in candidate_docs}
    for atom in row.get("typed_atoms") or row.get("typed_atom_proof_atoms") or []:
        if not isinstance(atom, Mapping) or not source_grounded_atom(atom):
            continue
        try:
            doc = int(atom.get("source_doc_index"))
        except Exception:
            continue
        if doc in candidate_set:
            out.setdefault(doc, []).append(atom)
    return out


def atom_role_ids(atoms: Sequence[Mapping[str, Any]], required: set[str]) -> set[str]:
    out: set[str] = set()
    for atom in atoms:
        rid = str(atom.get("role_id") or "").strip()
        if rid and (not required or rid in required):
            out.add(rid)
    return out


def select_docs_by_reselection(row: Mapping[str, Any], top_k: int = 5) -> tuple[list[int], dict[str, Any]]:
    candidate_docs = as_int_list(row.get("candidate_docs"))
    original_selected = as_int_list(row.get("selected_doc_indices"))
    if not candidate_docs:
        candidate_docs = original_selected

    required = set(required_role_ids(row))
    by_doc = atoms_by_doc(row, candidate_docs)
    position = {doc: idx for idx, doc in enumerate(candidate_docs)}
    selected: list[int] = []
    uncovered = set(required)
    remaining = list(candidate_docs)

    while len(selected) < top_k and remaining:
        ranked = []
        for doc in remaining:
            roles = atom_role_ids(by_doc.get(doc, []), required)
            new_roles = roles & uncovered if uncovered else set()
            ranked.append(
                (
                    len(new_roles),
                    len(roles),
                    len(by_doc.get(doc, [])),
                    -position.get(doc, 10**9),
                    doc,
                )
            )
        best = max(ranked)
        if required and best[0] == 0:
            break
        doc = int(best[-1])
        selected.append(doc)
        remaining.remove(doc)
        uncovered -= atom_role_ids(by_doc.get(doc, []), required)

    for doc in candidate_docs:
        if len(selected) >= top_k:
            break
        if doc not in selected:
            selected.append(doc)

    covered = set()
    for doc in selected:
        covered.update(atom_role_ids(by_doc.get(doc, []), required))
    metadata = {
        "method": "greedy_required_role_coverage_then_retrieval_order",
        "top_k": top_k,
        "original_selected_doc_indices": original_selected[:top_k],
        "candidate_pool_size": len(candidate_docs),
        "required_role_ids": sorted(required),
        "covered_required_role_ids": sorted(covered & required),
        "missing_required_role_ids": sorted(required - covered),
        "selected_atom_count": sum(len(by_doc.get(doc, [])) for doc in selected),
        "changed_from_original": selected != original_selected[:top_k],
        "uses_gold_at_runtime": False,
    }
    return selected, metadata


def covered_roles_for_docs(
    docs: Sequence[int],
    by_doc: Mapping[int, Sequence[Mapping[str, Any]]],
    required: set[str],
) -> set[str]:
    covered: set[str] = set()
    for doc in docs:
        covered.update(atom_role_ids(by_doc.get(doc, []), required))
    return covered


def replacement_doc(
    selected: Sequence[int],
    by_doc: Mapping[int, Sequence[Mapping[str, Any]]],
    required: set[str],
) -> int:
    ranked = []
    for selected_position, doc in enumerate(selected):
        roles = atom_role_ids(by_doc.get(doc, []), required)
        ranked.append((len(roles), len(by_doc.get(doc, [])), -selected_position, doc))
    return int(min(ranked)[-1])


def select_docs_by_completion(row: Mapping[str, Any], top_k: int = 5) -> tuple[list[int], dict[str, Any]]:
    candidate_docs = as_int_list(row.get("candidate_docs"))
    original_selected = as_int_list(row.get("selected_doc_indices"))[:top_k]
    if not candidate_docs:
        candidate_docs = list(original_selected)
    selected = list(original_selected)
    for doc in candidate_docs:
        if len(selected) >= top_k:
            break
        if doc not in selected:
            selected.append(doc)

    required = set(required_role_ids(row))
    by_doc = atoms_by_doc(row, candidate_docs)
    position = {doc: idx for idx, doc in enumerate(candidate_docs)}

    covered = covered_roles_for_docs(selected, by_doc, required)
    while required - covered:
        candidates = []
        for doc in candidate_docs:
            if doc in selected:
                continue
            doc_roles = atom_role_ids(by_doc.get(doc, []), required)
            if not doc_roles:
                continue
            for displaced in selected:
                trial = list(selected)
                trial[trial.index(displaced)] = doc
                trial_covered = covered_roles_for_docs(trial, by_doc, required)
                coverage_gain = len(trial_covered) - len(covered)
                if coverage_gain <= 0:
                    continue
                rank = (
                    coverage_gain,
                    len(doc_roles & (required - covered)),
                    len(doc_roles),
                    len(by_doc.get(doc, [])),
                    -position.get(doc, 10**9),
                    selected.index(displaced),
                )
                candidates.append((rank, doc, displaced, trial_covered))
        if not candidates:
            break
        _, promoted, displaced, covered = max(candidates, key=lambda item: item[0])
        selected[selected.index(int(displaced))] = int(promoted)

    metadata = {
        "method": "complete_missing_required_roles_from_candidate_pool",
        "top_k": top_k,
        "original_selected_doc_indices": original_selected,
        "candidate_pool_size": len(candidate_docs),
        "required_role_ids": sorted(required),
        "covered_required_role_ids": sorted(covered & required),
        "missing_required_role_ids": sorted(required - covered),
        "selected_atom_count": sum(len(by_doc.get(doc, [])) for doc in selected),
        "changed_from_original": selected != original_selected,
        "uses_gold_at_runtime": False,
    }
    return selected, metadata


def select_docs(row: Mapping[str, Any], top_k: int = 5, mode: str = "completion") -> tuple[list[int], dict[str, Any]]:
    if mode == "reselect":
        return select_docs_by_reselection(row, top_k=top_k)
    return select_docs_by_completion(row, top_k=top_k)


def posthoc_gold_surface(row: Mapping[str, Any], selected_docs: Sequence[int], corpus: Mapping[int, str]) -> bool | None:
    if not corpus:
        return None
    passages = [corpus[doc] for doc in selected_docs if doc in corpus]
    return answer_present(string_list(row.get("gold_answers")), passages)


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    changed = sum(1 for row in rows if row.get("ta_preselector", {}).get("changed_from_original"))
    selected_atoms = [int(row.get("ta_preselector", {}).get("selected_atom_count", 0) or 0) for row in rows]
    missing = [len(row.get("ta_preselector", {}).get("missing_required_role_ids", [])) for row in rows]
    gold_values = [row.get("gold_surface_in_selected_docs") for row in rows if row.get("gold_surface_in_selected_docs") is not None]
    return {
        "rows": len(rows),
        "changed_rows": changed,
        "changed_rate": changed / float(len(rows) or 1),
        "avg_selected_atom_count": sum(selected_atoms) / float(len(selected_atoms) or 1),
        "rows_with_all_required_roles_covered": sum(1 for item in missing if item == 0),
        "avg_missing_required_roles": sum(missing) / float(len(missing) or 1),
        "gold_surface_in_selected_docs_count_posthoc": sum(1 for value in gold_values if value is True),
        "boundary": {
            "candidate_pool_expansion": False,
            "reader_change": False,
            "answer_gate": False,
            "selector_training": False,
            "gold_used_for_selection": False,
        },
    }


def build_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    return "\n".join(
        [
            "# EchoRAGv2-TA Preselector Summary",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| rows | {summary['rows']} |",
            f"| changed rows | {summary['changed_rows']} |",
            f"| changed rate | {summary['changed_rate']:.4f} |",
            f"| avg selected atom count | {summary['avg_selected_atom_count']:.4f} |",
            f"| rows with all required roles covered | {summary['rows_with_all_required_roles_covered']} |",
            f"| avg missing required roles | {summary['avg_missing_required_roles']:.4f} |",
            f"| gold surface in selected docs post-hoc | {summary['gold_surface_in_selected_docs_count_posthoc']} |",
            "",
            "Boundary: no candidate expansion, no reader change, no answer gate, no selector training, no gold used for selection.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    payload = read_json(args.typed_atoms_json)
    corpus = load_corpus(args.corpus_json)
    rows = []
    for row in payload.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        out = dict(row)
        selected, metadata = select_docs(row, top_k=int(args.top_k), mode=str(args.mode))
        out["selected_doc_indices"] = selected
        out["ta_preselector"] = metadata
        gold_surface = posthoc_gold_surface(out, selected, corpus)
        if gold_surface is not None:
            out["gold_surface_in_selected_docs"] = gold_surface
        rows.append(out)

    output = dict(payload)
    output["selector_variant"] = "echoragv3_completion_v1" if args.mode == "completion" else "echoragv2_ta_reselect_v1"
    output["rows"] = rows
    output["summary"] = summarize(rows)
    config = dict(output.get("config", {}) or {})
    config["echoragv3"] = {
        "method": "complete_missing_required_roles_from_candidate_pool"
        if args.mode == "completion"
        else "greedy_required_role_coverage_then_retrieval_order",
        "top_k": int(args.top_k),
        "mode": str(args.mode),
        "uses_gold_at_runtime": False,
    }
    output["config"] = config
    write_json_atomic(args.output_json, output)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(output), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
