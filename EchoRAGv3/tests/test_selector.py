import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import echoragv3_selector as selector


def _atom(doc, role, span="evidence", auth="operator:test"):
    return {
        "source_doc_index": doc,
        "role_id": role,
        "exact_source_span": span,
        "source_authorization": auth,
    }


def test_selects_documents_covering_required_roles_before_rank_fill():
    row = {
        "candidate_docs": [10, 11, 12, 13, 14, 15],
        "selected_doc_indices": [10, 11, 12, 13, 14],
        "roles": [
            {"role_id": "r0", "required": True},
            {"role_id": "r1", "required": True},
        ],
        "typed_atoms": [_atom(12, "r0"), _atom(15, "r1")],
    }

    selected, meta = selector.select_docs(row, top_k=5, mode="reselect")

    assert selected[:2] == [12, 15]
    assert selected == [12, 15, 10, 11, 13]
    assert meta["covered_required_role_ids"] == ["r0", "r1"]
    assert meta["uses_gold_at_runtime"] is False


def test_ignores_atoms_outside_candidate_pool():
    row = {
        "candidate_docs": [1, 2, 3],
        "selected_doc_indices": [1, 2, 3],
        "roles": [{"role_id": "r0", "required": True}],
        "typed_atoms": [_atom(99, "r0")],
    }

    selected, meta = selector.select_docs(row, top_k=2, mode="reselect")

    assert selected == [1, 2]
    assert meta["missing_required_role_ids"] == ["r0"]


def test_ungrounded_atoms_do_not_affect_selection():
    row = {
        "candidate_docs": [1, 2, 3],
        "selected_doc_indices": [1, 2, 3],
        "roles": [{"role_id": "r0", "required": True}],
        "typed_atoms": [
            _atom(3, "r0", span=""),
            _atom(2, "r0", auth=""),
        ],
    }

    selected, meta = selector.select_docs(row, top_k=2, mode="reselect")

    assert selected == [1, 2]
    assert meta["selected_atom_count"] == 0


def test_completion_only_replaces_doc_when_original_top5_misses_required_role():
    row = {
        "candidate_docs": [10, 11, 12, 13, 14, 15],
        "selected_doc_indices": [10, 11, 12, 13, 14],
        "roles": [
            {"role_id": "r0", "required": True},
            {"role_id": "r1", "required": True},
        ],
        "typed_atoms": [_atom(10, "r0"), _atom(15, "r1")],
    }

    selected, meta = selector.select_docs(row, top_k=5, mode="completion")

    assert selected == [10, 11, 12, 13, 15]
    assert meta["covered_required_role_ids"] == ["r0", "r1"]
    assert meta["changed_from_original"] is True


def test_completion_only_preserves_original_when_roles_already_covered():
    row = {
        "candidate_docs": [10, 11, 12, 13, 14, 15],
        "selected_doc_indices": [10, 11, 12, 13, 14],
        "roles": [{"role_id": "r0", "required": True}],
        "typed_atoms": [_atom(10, "r0"), _atom(15, "r0")],
    }

    selected, meta = selector.select_docs(row, top_k=5, mode="completion")

    assert selected == [10, 11, 12, 13, 14]
    assert meta["changed_from_original"] is False


def test_completion_only_rejects_role_swaps_without_net_coverage_gain():
    row = {
        "candidate_docs": [1, 2],
        "selected_doc_indices": [1],
        "roles": [
            {"role_id": "r0", "required": True},
            {"role_id": "r1", "required": True},
        ],
        "typed_atoms": [_atom(1, "r0"), _atom(2, "r1")],
    }

    selected, meta = selector.select_docs(row, top_k=1, mode="completion")

    assert selected == [1]
    assert meta["covered_required_role_ids"] == ["r0"]
    assert meta["missing_required_role_ids"] == ["r1"]
    assert meta["changed_from_original"] is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
