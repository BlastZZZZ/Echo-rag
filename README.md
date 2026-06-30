# EchoRAGv2 Standalone Slim

This is a GitHub-size slim bundle of the EchoRAGv2 strict pipeline used by the Qwen3-8B + BGE-M3 experiments.

The core code and cached MuSiQue selector smoke test can run without referring back to the original local HippoRAG experiment tree. Full three-dataset reruns need the dataset JSONs to be hydrated into `reproduce/dataset/`.

## What Is Included

- EchoRAGv2 role decomposition, role-channel graph retrieval, provenance-aware selector, and fixed-context QA reader.
- The local HippoRAG v2 source tree under `src/hipporag`.
- Minimal dataset files for the cached MuSiQue smoke test.
- Gold-row reports for MuSiQue, HotpotQA, and 2Wiki.
- A cached MuSiQue limit-100 selector smoke input under `standalone_inputs/musique_limit100`.
- Relative-path runner scripts under `scripts/`.

## Main Entry Points

| Stage | File |
| --- | --- |
| Evidence-role cache | `build_ergr_role_cache_v1.py` |
| Passage-seed role graph entry | `role_aware_graph_entry.py` |
| Role-channel graph retrieval | `evaluate_role_channel_graph_retrieval.py` |
| Demand provenance selector | `evaluate_echo_support_profile_selector.py` |
| Fixed Top-5 Qwen no-think QA | `rerun_qa_nothink_from_perquery.py` |
| Baseline/index builder | `tmp_build_qwen8_bgem3_index_and_baseline.py` |

## Install

```bash
cd Echo-rag
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD"
```

To use an existing Python environment instead of `.venv`, set `PY` explicitly:

```bash
export PY=python3
export PYTHONPATH="$PWD/src:$PWD"
```

## API Settings

The runner scripts automatically load a local `.env` file from the repository root. Copy the template and edit it:

```bash
cp .env.example .env
```

For local vLLM:

```bash
OPENAI_API_KEY=EMPTY
LLM_BASE_URL=http://localhost:8002/v1
LLM_NAME=qwen3-8b
API_KEY_ARG=ENV
```

For a remote OpenAI-compatible API:

```bash
OPENAI_API_KEY=sk-your-key
LLM_BASE_URL=https://api.example.com/v1
LLM_NAME=your-model-name
API_KEY_ARG=ENV
```

`LLM_BASE_URL` should point to the `/v1` endpoint, not `/chat/completions`. The `.env` file is ignored by git.

## Cached Smoke Test

This checks the package wiring using bundled MuSiQue limit-100 inputs. It replays the selector stage from a seeded selector cache, so it should not need to call an LLM if all cache keys match.

```bash
cd Echo-rag
bash scripts/run_musique_selector_smoke.sh
```

Expected output:

- `standalone_runs/musique_limit100_echo_v2_smoke/echo_v2_selector_musique_limit100.json`
- `standalone_runs/musique_limit100_echo_v2_smoke/echo_v2_selector_musique_limit100.md`

## Full Strict Pipeline

The slim package excludes most full dataset JSONs to keep the archive below 24 MB. If you are on the original workstation, hydrate them from the full local package first:

```bash
cd Echo-rag
bash scripts/hydrate_full_data_from_local.sh ../echoragv2_standalone_20260630
```

You can also pass any directory that contains `reproduce/dataset/*.json`.

Then run:

```bash
LLM_NAME=qwen3-8b \
LLM_BASE_URL=http://localhost:8002/v1 \
EMBEDDING_NAME=Transformers/BAAI/bge-m3 \
bash scripts/run_echov2_strict_qwen8_bgem3.sh
```

Defaults:

- datasets: `musique hotpotqa 2wikimultihopqa`
- limit: `1000`
- variants:
  - `echov2_qwen8_bgem3_shared_question`
  - `echov2_qwen8_bgem3_shared_question_by_demand`
- output root: `outputs_echov2_qwen8_bgem3_limit1000_20260602_passageseed`

## Provenance

The local strict_full result was traced to:

```text
outputs_echov2_qwen8_bgem3_limit1000_20260602_passageseed/
  {dataset}/echov2_qwen8_bgem3_shared_question/qa.json
```

See `docs/STRICT_FULL_PROVENANCE.md` for the exact code path and metrics.

## GitHub Notes

Generated outputs are ignored by `.gitignore`. This slim package intentionally omits most dataset JSONs; check dataset redistribution permissions before hydrating and publishing full data.
