from types import SimpleNamespace

import pytest

import evaluate_dcr_ess_listwise_selector as selector
import evaluate_ress_listwise_selector as selector_impl


def test_parse_args_defaults_to_clean_standalone_candidate_pool(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_dcr_ess_listwise_selector.py",
            "--dataset",
            "toy",
            "--corpus_json",
            str(tmp_path / "corpus.json"),
            "--per_query_report",
            str(tmp_path / "per_query.json"),
            "--baseline_top200_cache",
            str(tmp_path / "baseline.json"),
            "--role_channel_json",
            str(tmp_path / "roles.json"),
            "--save_json_path",
            str(tmp_path / "out.json"),
            "--save_md_path",
            str(tmp_path / "out.md"),
            "--selector_cache_path",
            str(tmp_path / "selector_cache.json"),
        ],
    )

    args = selector.parse_args()

    assert args.baseline_candidate_top_k == 0
    assert args.candidate_admission == "channel_balanced"
    assert args.admission_base_top_k == 8
    assert args.admission_min_channel_support == 2
    assert args.terminal_admission_limit == 4
    assert args.selector_objective == "evidence_set_coverage"
    assert args.allow_ablation_variants is False


def test_parse_args_rejects_ablation_candidate_admission_without_explicit_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_dcr_ess_listwise_selector.py",
            "--dataset",
            "toy",
            "--corpus_json",
            str(tmp_path / "corpus.json"),
            "--per_query_report",
            str(tmp_path / "per_query.json"),
            "--baseline_top200_cache",
            str(tmp_path / "baseline.json"),
            "--role_channel_json",
            str(tmp_path / "roles.json"),
            "--save_json_path",
            str(tmp_path / "out.json"),
            "--save_md_path",
            str(tmp_path / "out.md"),
            "--selector_cache_path",
            str(tmp_path / "selector_cache.json"),
            "--candidate_admission",
            "cross_channel",
        ],
    )

    with pytest.raises(SystemExit):
        selector.parse_args()


def test_parse_args_allows_ablation_candidate_admission_only_with_explicit_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_dcr_ess_listwise_selector.py",
            "--dataset",
            "toy",
            "--corpus_json",
            str(tmp_path / "corpus.json"),
            "--per_query_report",
            str(tmp_path / "per_query.json"),
            "--baseline_top200_cache",
            str(tmp_path / "baseline.json"),
            "--role_channel_json",
            str(tmp_path / "roles.json"),
            "--save_json_path",
            str(tmp_path / "out.json"),
            "--save_md_path",
            str(tmp_path / "out.md"),
            "--selector_cache_path",
            str(tmp_path / "selector_cache.json"),
            "--candidate_admission",
            "cross_channel",
            "--allow_ablation_variants",
        ],
    )

    args = selector.parse_args()

    assert args.candidate_admission == "cross_channel"
    assert args.allow_ablation_variants is True


def test_build_candidate_docs_uses_explicit_baseline_seed_only_when_requested():
    channels = [
        SimpleNamespace(role_id="r0", role_description="first", retrieval_query="", doc_indices=[1, 2, 3]),
        SimpleNamespace(role_id="r1", role_description="second", retrieval_query="", doc_indices=[2, 4, 5]),
    ]

    standalone_docs = selector.build_candidate_docs(
        baseline_docs=[9, 8],
        channels=channels,
        candidate_top_k=2,
        baseline_candidate_top_k=0,
        max_candidates=10,
    )
    seeded_docs = selector.build_candidate_docs(
        baseline_docs=[9, 8],
        channels=channels,
        candidate_top_k=2,
        baseline_candidate_top_k=1,
        max_candidates=10,
    )

    assert standalone_docs == [1, 2, 4]
    assert seeded_docs == [9, 1, 2, 4]


def test_build_candidate_docs_supports_cross_channel_admission():
    channels = [
        SimpleNamespace(role_id="r0", role_description="first", retrieval_query="", doc_indices=[1, 2, 3, 9]),
        SimpleNamespace(role_id="r1", role_description="second", retrieval_query="", doc_indices=[4, 2, 5, 9]),
        SimpleNamespace(role_id="r2", role_description="third", retrieval_query="", doc_indices=[6, 7, 8, 9]),
    ]

    docs = selector.build_candidate_docs(
        baseline_docs=[],
        channels=channels,
        candidate_top_k=4,
        baseline_candidate_top_k=0,
        max_candidates=10,
        candidate_admission="cross_channel",
        admission_base_top_k=1,
        admission_min_channel_support=3,
    )

    assert docs == [1, 4, 6, 9]


def test_build_candidate_docs_supports_terminal_bound_admission():
    channels = [
        SimpleNamespace(role_id="r0", role_description="first", retrieval_query="", doc_indices=[1, 2, 3]),
        SimpleNamespace(role_id="r1", role_description="answer", retrieval_query="", doc_indices=[4, 5, 6]),
    ]

    docs = selector.build_candidate_docs(
        baseline_docs=[],
        channels=channels,
        candidate_top_k=3,
        baseline_candidate_top_k=0,
        max_candidates=10,
        candidate_admission="terminal_bound",
        admission_base_top_k=1,
        terminal_channel_ids=["r1"],
        terminal_admission_limit=2,
    )

    assert docs == [1, 4, 5, 6]


def test_terminal_role_ids_uses_role_graph_sinks_with_fallback_to_last_role():
    roles = [
        {"role_id": "r0", "must_connect_to": ["r1"]},
        {"role_id": "r1", "must_connect_to": []},
    ]
    cyclic_roles = [
        {"role_id": "r0", "must_connect_to": ["r1"]},
        {"role_id": "r1", "must_connect_to": ["r0"]},
    ]

    assert selector.terminal_role_ids(roles) == ["r1"]
    assert selector.terminal_role_ids(cyclic_roles) == ["r1"]


def test_candidate_role_provenance_records_role_and_rank():
    channels = [
        SimpleNamespace(
            role_id="r0",
            role_description="identify composer",
            retrieval_query="fallback role 0",
            doc_indices=[10, 11, 12],
        ),
        SimpleNamespace(
            role_id="r1",
            role_description="find birthplace",
            retrieval_query="fallback role 1",
            doc_indices=[12, 10, 13],
        ),
    ]
    provenance = selector.candidate_role_provenance(
        candidate_docs=[10, 12],
        channels=channels,
        candidate_top_k=2,
    )
    assert provenance[10] == ["identify composer (rank 1)", "find birthplace (rank 2)"]
    assert provenance[12] == ["find birthplace (rank 1)"]


def test_selector_prompt_can_include_candidate_provenance():
    messages = selector.selector_prompt(
        question="Where was the composer born?",
        role_descriptions=["identify composer", "find birthplace"],
        candidate_docs=[0],
        corpus_passages=["Composer\nThe composer was born in Berlin."],
        max_passage_chars=200,
        top_k=1,
        candidate_provenance={0: ["find birthplace (rank 1)"]},
    )
    user = messages[1]["content"]
    assert "Retrieved by evidence channels" in user
    assert "find birthplace (rank 1)" in user
    assert "The composer was born in Berlin." in user


def test_selector_prompt_defaults_to_evidence_set_coverage_objective():
    messages = selector.selector_prompt(
        question="Where was the composer born?",
        role_descriptions=["identify composer", "find birthplace"],
        candidate_docs=[0, 1],
        corpus_passages=[
            "Composer\nThe composer wrote the score.",
            "Birthplace\nThe composer was born in Berlin.",
        ],
        max_passage_chars=200,
        top_k=2,
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "multi-hop evidence-set selector" in system
    assert "answer-resolving evidence-set selector" not in system
    assert "cover distinct evidence channels" in user
    assert "isolated answer-looking passages" in user


def test_selector_prompt_supports_answer_resolving_ablation():
    messages = selector.selector_prompt(
        question="Where was the composer born?",
        role_descriptions=["identify composer", "find birthplace"],
        candidate_docs=[0, 1],
        corpus_passages=[
            "Composer\nThe composer wrote the score.",
            "Birthplace\nThe composer was born in Berlin.",
        ],
        max_passage_chars=200,
        top_k=2,
        selector_objective="answer_resolving",
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "answer-resolving evidence-set selector" in system
    assert "final answer binding" in user
    assert "terminal answer" in user
    assert "competing entity, date, organization, or value" in user


def test_select_docs_treats_selector_output_as_membership_not_reader_order(monkeypatch, tmp_path):
    def fake_call_selector(**kwargs):
        return '{"selected_ids": [2, 0]}', {"finish_reason": "stop"}

    monkeypatch.setattr(selector_impl, "call_selector", fake_call_selector)
    cache = {}

    docs, payload = selector.select_docs(
        client=object(),
        question="Where was the composer born?",
        role_descriptions=["identify composer", "find birthplace"],
        candidate_docs=[10, 11, 12, 13],
        corpus_passages=[""] * 14,
        top_k=3,
        max_passage_chars=200,
        model="fake-selector",
        temperature=0.0,
        max_tokens=64,
        retries=1,
        retry_wait_seconds=0.0,
        cache=cache,
        cache_path=tmp_path / "cache.json",
    )

    assert payload["selected_ids"] == [2, 0]
    assert docs == [10, 12, 11]
