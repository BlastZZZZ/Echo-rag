# strict_full Provenance

`strict_full` is the saved strict EchoRAGv2 full-channel baseline, not the later MEA/memory acceptance layer.

## Artifact Root

```text
outputs_echov2_qwen8_bgem3_limit1000_20260602_passageseed
```

`strict_full` corresponds to:

```text
{dataset}/echov2_qwen8_bgem3_shared_question/qa.json
```

`strict_fact` corresponds to:

```text
{dataset}/echov2_qwen8_bgem3_shared_question_by_demand/qa.json
```

## Code Root

The original local run was traced to a local `echov2_runnable_code_20260527` working tree. This repository is a cleaned, relocatable copy of that code path.

## Code Chain

| Stage | Code |
| --- | --- |
| role cache | `build_ergr_role_cache_v1.py` |
| baseline/index | `tmp_build_qwen8_bgem3_index_and_baseline.py` |
| role-channel retrieval | `evaluate_role_channel_graph_retrieval.py` |
| passage-seed graph entry | `role_channel_graph_retrieval.py`, `role_aware_graph_entry.py` |
| selector | `evaluate_echo_support_profile_selector.py` |
| QA reader | `rerun_qa_nothink_from_perquery.py` |

The passage-seed graph entry mechanism is implemented by dense role-passage retrieval feeding `passage_seed_scores` into `graph_search_with_role_entries`, which writes passage-node weights into the PPR entry vector.

## Saved Full Results

| Dataset | Variant | R@5 | EM | F1 |
| --- | --- | ---: | ---: | ---: |
| MuSiQue | strict_full | 0.6384 | 0.3400 | 0.4241 |
| HotpotQA | strict_full | 0.9050 | 0.5980 | 0.7209 |
| 2Wiki | strict_full | 0.9077 | 0.5980 | 0.6785 |
| Macro | strict_full | 0.8171 | 0.5120 | 0.6078 |
