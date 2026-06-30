import evaluate_echo_support_profile_selector as selector


class DummyChannel:
    def __init__(self, role_id, role_description, doc_indices, support_function=""):
        self.role_id = role_id
        self.role_description = role_description
        self.retrieval_query = role_description
        self.doc_indices = doc_indices
        self.support_function = support_function


def test_channel_profile_for_candidates_records_rank_and_channel_count():
    profiles = selector.channel_profile_for_candidates(
        channels=[
            DummyChannel("r0", "find bridge", [10, 11]),
            DummyChannel("r1", "find answer", [11, 12]),
        ],
        candidate_docs=[10, 11, 12],
        candidate_top_k=2,
    )

    assert profiles[10]["channel_count"] == 1
    assert profiles[10]["best_rank"] == 1
    assert profiles[11]["channel_count"] == 2
    assert profiles[11]["best_rank"] == 1
    assert profiles[11]["function_diversity"] == 2
    assert profiles[11]["support_functions"] == ["r0", "r1"]
    assert [item["description"] for item in profiles[11]["channels"]] == ["find bridge", "find answer"]


def test_channel_profile_records_support_function_diversity_when_available():
    profiles = selector.channel_profile_for_candidates(
        channels=[
            DummyChannel("r0", "find bridge", [10], support_function="bridge_binding"),
            DummyChannel("r1", "find answer", [10], support_function="terminal_answer"),
            DummyChannel("r2", "find answer duplicate", [10], support_function="terminal_answer"),
        ],
        candidate_docs=[10],
        candidate_top_k=1,
    )

    assert profiles[10]["channel_count"] == 3
    assert profiles[10]["function_diversity"] == 2
    assert profiles[10]["support_functions"] == ["bridge_binding", "terminal_answer"]
    assert profiles[10]["channels"][0]["support_function"] == "bridge_binding"


def test_support_profile_prompt_contains_profile_and_passage_text():
    messages = selector.demand_exposure_prompt(
        question="Who is the sibling?",
        role_descriptions=["identify actor", "find sibling"],
        candidate_docs=[0, 1],
        corpus_passages=[
            "Actor\nSusie was played by Natalie Wood.",
            "Sibling\nLana Wood is Natalie Wood's sister.",
        ],
        corpus_records=[
            {"title": "Actor", "text": "Susie was played by Natalie Wood."},
            {"title": "Sibling", "text": "Lana Wood is Natalie Wood's sister."},
        ],
        profiles={
            0: {"channel_count": 1, "best_rank": 1, "channels": [{"description": "identify actor", "rank": 1}]},
            1: {"channel_count": 1, "best_rank": 1, "channels": [{"description": "find sibling", "rank": 1}]},
        },
        max_passage_chars=200,
        top_k=2,
    )

    user = messages[1]["content"]
    assert "Demand-exposure profile" in user
    assert "channel_count: 1" in user
    assert "best_channel_rank: 1" in user
    assert "Susie was played by Natalie Wood" in user
    assert "fixed-budget answer-support evidence set" in user
    assert "Return only JSON" in user


def test_direct_answer_first_prompt_keeps_profile_and_prioritizes_answer_bearing_passages():
    messages = selector.demand_exposure_prompt(
        question="What did the farm start with?",
        role_descriptions=["find farm activity", "find direct answer"],
        candidate_docs=[0, 1],
        corpus_passages=[
            "Background\nThe farm has many visitor activities.",
            "Answer\nThe family started with U-pick berries.",
        ],
        corpus_records=[
            {"title": "Background", "text": "The farm has many visitor activities."},
            {"title": "Answer", "text": "The family started with U-pick berries."},
        ],
        profiles={
            0: {"channel_count": 2, "best_rank": 1, "channels": [{"description": "find farm activity", "rank": 1}]},
            1: {"channel_count": 1, "best_rank": 2, "channels": [{"description": "find direct answer", "rank": 2}]},
        },
        max_passage_chars=200,
        top_k=2,
        selection_prompt_mode="direct_answer_first",
    )

    user = messages[1]["content"]
    assert "Demand-exposure profile" in user
    assert "directly answer the question" in user
    assert "precise answer-bearing passage" in user
    assert "Use demand-exposure provenance as a tie-breaker" in user
    assert "Cover complementary evidence demands only when the question genuinely requires multiple facts" in user
    assert "Return only JSON" in user


def test_domain_answer_judge_prompt_puts_passage_before_profile_and_demotes_provenance():
    messages = selector.demand_exposure_prompt(
        question="What did the farm start with?",
        role_descriptions=["cover farm history", "find visitor activity"],
        candidate_docs=[0],
        corpus_passages=["Answer\nThe family started with U-pick berries."],
        corpus_records=[{"title": "Answer", "text": "The family started with U-pick berries."}],
        profiles={
            0: {"channel_count": 3, "best_rank": 1, "channels": [{"description": "cover farm history", "rank": 1}]},
        },
        max_passage_chars=200,
        top_k=1,
        selection_prompt_mode="domain_answer_judge",
    )

    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "QA evidence selector" in system
    assert user.index("Passage:") < user.index("Demand-exposure profile")
    assert "These are not a coverage checklist" in user
    assert "DIRECT_ANSWER passages" in user
    assert "Use demand-exposure provenance only to break ties" in user
    assert "Do not select a passage just because it has high channel_count" in user
    assert "Return only JSON" in user


def test_domain_answer_scorecard_prompt_orders_selected_ids_for_reader():
    messages = selector.demand_exposure_prompt(
        question="What did the farm start with?",
        role_descriptions=["cover farm history"],
        candidate_docs=[0],
        corpus_passages=["Answer\nThe Bohners started with U-pick berries."],
        corpus_records=[{"title": "Answer", "text": "The Bohners started with U-pick berries."}],
        profiles={0: {"channel_count": 1, "best_rank": 1, "channels": [{"description": "cover farm history", "rank": 1}]}},
        max_passage_chars=200,
        top_k=1,
        selection_prompt_mode="domain_answer_scorecard",
        passage_excerpt_mode="query_focus",
    )

    user = messages[1]["content"]
    assert "DIRECT_ANSWER = explicitly states the requested answer value" in user
    assert "BACKGROUND = broad" in user
    assert "The order of selected_ids is the reader order" in user
    assert "Use demand-exposure provenance only as a tie-breaker" in user


def test_direct_symbol_question_detection_is_narrow():
    assert selector.is_direct_symbol_question(
        "What function is used to stop the response timer by calling del_timer_sync()?"
    )
    assert selector.is_direct_symbol_question(
        "What is the term used to describe metrics, often abbreviated as STATS?"
    )
    assert not selector.is_direct_symbol_question(
        "What is the name of the actor whose sibling appeared in the film?"
    )


def test_symbol_route_candidate_docs_adds_exact_symbol_chunks_only_for_direct_lookup():
    corpus = [
        "Background\nThe IPv6 routing chapter discusses forwarding.",
        "Answer\nstruct dst_entry *ip6_route_output returns the destination cache entry in the Tx path.",
        "Other\nThe ip6_route_input method handles the receive path.",
    ]

    docs, diag = selector.symbol_route_candidate_docs(
        question=(
            "What is the main IPv6 routing subsystem lookup method used in the "
            "transmission path that returns the destination cache entry?"
        ),
        candidate_docs=[0, 2],
        baseline_docs=[],
        corpus_passages=corpus,
        extra_candidates=2,
        baseline_top_k=0,
        max_candidates=4,
        prepend_candidates=True,
    )

    assert diag["symbol_route_active"] is True
    assert 1 in docs
    assert docs.index(1) < docs.index(0)


def test_symbol_route_candidate_docs_stays_unchanged_for_generic_multihop_question():
    docs, diag = selector.symbol_route_candidate_docs(
        question="Which country is the birthplace of the author's father?",
        candidate_docs=[10, 11],
        baseline_docs=[12],
        corpus_passages=["a", "b", "c"],
        extra_candidates=2,
        baseline_top_k=1,
        max_candidates=4,
        prepend_candidates=True,
    )

    assert docs == [10, 11]
    assert diag["symbol_route_active"] is False


def test_domain_symbol_route_prompt_prioritizes_exact_requested_symbol():
    messages = selector.demand_exposure_prompt(
        question="What function is used to stop the response timer?",
        role_descriptions=["find timer callback"],
        candidate_docs=[0],
        corpus_passages=["Answer\nThe timer is stopped by calling del_timer_sync()."],
        corpus_records=[{"title": "Answer", "text": "The timer is stopped by calling del_timer_sync()."}],
        profiles={0: {"channel_count": 1, "best_rank": 1, "channels": [{"description": "find timer callback", "rank": 1}]}},
        max_passage_chars=200,
        top_k=1,
        selection_prompt_mode="domain_symbol_route",
        passage_excerpt_mode="query_focus",
    )

    user = messages[1]["content"]
    assert "direct symbol/name lookup" in user
    assert "function, method, field" in user
    assert "most exact question-answering passage first" in user


def test_query_focused_excerpt_can_surface_late_question_match():
    text = (
        "Opening background that does not answer the question. "
        + "filler " * 160
        + "The Bohners eventually decided to start with U-pick berries and shiitake mushrooms."
    )

    excerpt = selector.query_focused_excerpt(
        text,
        "What type of farming did the Bohners decide to start with?",
        max_chars=220,
    )

    assert "Bohners eventually decided" in excerpt
    assert "U-pick berries" in excerpt
    assert excerpt.startswith("...")


def test_parse_selected_ids_filters_invalid_and_deduplicates():
    assert selector.parse_selected_ids('{"selected_ids": [2, "0", 2, 99]}', num_candidates=3, top_k=3) == [2, 0]


def test_apply_profile_mode_shuffled_preserves_docs_but_rebinds_profiles():
    profiles = {
        10: {"channel_count": 1, "best_rank": 1, "channels": [{"description": "a", "rank": 1}]},
        11: {"channel_count": 2, "best_rank": 2, "channels": [{"description": "b", "rank": 2}]},
        12: {"channel_count": 3, "best_rank": 3, "channels": [{"description": "c", "rank": 3}]},
    }
    shuffled = selector.apply_profile_mode(
        profiles=profiles,
        candidate_docs=[10, 11, 12],
        profile_mode="shuffled",
        query_index=7,
        shuffle_seed=13,
    )

    assert set(shuffled) == {10, 11, 12}
    assert sorted(item["channel_count"] for item in shuffled.values()) == [1, 2, 3]
    assert any(shuffled[doc]["channel_count"] != profiles[doc]["channel_count"] for doc in profiles)


def test_apply_profile_mode_no_profile_hides_exposure_fields():
    profiles = {
        10: {"channel_count": 1, "best_rank": 1, "channels": [{"description": "a", "rank": 1}]},
    }
    hidden = selector.apply_profile_mode(
        profiles=profiles,
        candidate_docs=[10],
        profile_mode="no_profile",
        query_index=0,
        shuffle_seed=0,
    )

    assert hidden[10]["channel_count"] == 0
    assert hidden[10]["best_rank"] is None
    assert hidden[10]["channels"] == []


def test_text_only_equal_length_prompt_masks_provenance_values():
    profiles = selector.apply_profile_mode(
        profiles={
            0: {"channel_count": 2, "best_rank": 1, "channels": [{"description": "find bridge", "rank": 1}]},
        },
        candidate_docs=[0],
        profile_mode="text_only_equal_length",
        query_index=0,
        shuffle_seed=0,
    )
    messages = selector.demand_exposure_prompt(
        question="Who is the sibling?",
        role_descriptions=["identify actor"],
        candidate_docs=[0],
        corpus_passages=["Actor\nSusie was played by Natalie Wood."],
        corpus_records=[{"title": "Actor", "text": "Susie was played by Natalie Wood."}],
        profiles=profiles,
        max_passage_chars=200,
        top_k=1,
        profile_mode="text_only_equal_length",
    )

    user = messages[1]["content"]
    assert "text-only equal-length placeholder" in user
    assert "channel_count: masked" in user
    assert "best_channel_rank: masked" in user
    assert "find bridge" not in user
    assert "rank 1" not in user
    assert "select from the question and passage text only" in user
