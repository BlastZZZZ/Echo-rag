def get_query_instruction(linking_method):
    instructions = {
        'ner_to_node': 'Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.',
        'query_to_node': 'Given a question, retrieve relevant phrases that are mentioned in this question.',
        'query_to_fact': 'Given a question, retrieve relevant triplet facts that matches this question.',
        'query_to_hyperedge': 'Given a question, retrieve relevant proposition summaries written as subject [relation] object statements that help answer this question.',
        'query_to_proposition': 'Given a question, retrieve relevant natural-language propositions that preserve complete context and help answer this question.',
        'query_to_sentence': 'Given a question, retrieve relevant sentences that best answer the question.',
        'query_to_passage': 'Given a question, retrieve relevant documents that best answer the question.',
    }
    default_instruction = 'Given a question, retrieve relevant documents that best answer the question.'
    return instructions.get(linking_method, default_instruction)


def get_hyperedge_query_rewrite_system_prompt(version: str = "v2"):
    if version in {"v3", "v4"}:
        extra_guidance = ""
        if version == "v4":
            extra_guidance = (
                " Prefer at most one strong target view, one bridge view, one constraint view, and one inverse view. "
                "Avoid redundant paraphrases because each view will control a different part of retrieval."
            )
        return (
            "You rewrite QA questions into proposition-style retrieval queries for matching proposition summaries. "
            + "Each rewritten query must be a short statement in the form subject [relation] object. "
            + "Use variables such as ?x, ?y, ?bridge when the answer or bridge entity is unknown. "
            + "Preserve named entities and key relation words from the original question. "
            + "For multi-hop questions, explicitly separate target, bridge, inverse, and constraint views when useful. "
            + extra_guidance
            + "Return strict JSON with one field: search_queries."
        )
    return (
        "You rewrite QA questions into proposition-style retrieval queries for matching proposition summaries. "
        "Each rewritten query must be a short statement in the form subject [relation] object. "
        "Use variables such as ?x, ?y, ?bridge when the answer or bridge entity is unknown. "
        "Preserve named entities and key relation words from the original question. "
        "For multi-hop questions, cover target relations, bridge relations, and key constraints. "
        "Return strict JSON with one field: search_queries."
    )


def build_hyperedge_query_rewrite_user_prompt(question: str, max_views: int, version: str = "v2"):
    if version in {"v3", "v4"}:
        role_hint = "4. Prefer roles from this set: target, bridge, constraint, inverse.\n"
        if version == "v4":
            role_hint = (
                "4. Prefer one clear target view first, then add bridge, constraint, and inverse only when they help multi-hop retrieval.\n"
                "5. Prefer 2 to 4 views instead of many weak paraphrases.\n"
            )
        return (
            f"Question: {question}\n"
            + f"Return 2 to {max_views} unique proposition-style retrieval queries.\n"
            + "Rules:\n"
            + "1. Each query must be concise and easy to match against a proposition summary.\n"
            + "2. Use the literal format subject [relation] object whenever possible.\n"
            + "3. Keep useful entities and constraints from the question.\n"
            + role_hint
            + "6. Confidence must be a number between 0 and 1.\n"
            + "7. Do not answer the question.\n"
            + "8. Output JSON only, for example: "
            + "{\"search_queries\": [{\"query\": \"?x [relation] entity\", \"role\": \"target\", \"confidence\": 0.92}, "
            + "{\"query\": \"entity [related to] ?bridge\", \"role\": \"bridge\", \"confidence\": 0.73}]}"
        )
    return (
        f"Question: {question}\n"
        f"Return 2 to {max_views} unique proposition-style retrieval queries.\n"
        "Rules:\n"
        "1. Each query must be concise and easy to match against a proposition summary.\n"
        "2. Use the literal format subject [relation] object whenever possible.\n"
        "3. Keep useful entities and constraints from the question.\n"
        "4. Do not answer the question.\n"
        "5. Output JSON only, for example: "
        "{\"search_queries\": [\"?x [relation] entity\", \"entity [relation] ?y\"]}"
    )


def get_hyperedge_rerank_system_prompt():
    return (
        "You rerank candidate propositions for multi-hop QA retrieval. "
        "Each candidate proposition may be directly useful, act as a bridge, or enforce a key constraint. "
        "Prefer propositions that match the asked relation, identify indispensable bridge entities, "
        "or preserve key constraints such as time, location, comparison, and type. "
        "Return strict JSON with one field: ranked_indices. "
        "The list must contain 0-based candidate indices ordered from most useful to least useful."
    )


def build_hyperedge_rerank_user_prompt(question: str, candidates: list[str]):
    return (
        f"Question: {question}\n"
        + "Candidates:\n"
        + "\n".join(candidates)
        + "\nReturn JSON only, for example: {\"ranked_indices\": [2, 0, 1]}"
    )
