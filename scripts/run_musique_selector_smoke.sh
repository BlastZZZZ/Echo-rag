#!/usr/bin/env bash
set -euo pipefail

unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${ROOT}/.env"
  set +a
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
PY="${PY:-${ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${PYTHON:-python3}"
fi
API_KEY_ARG="${API_KEY_ARG:-ENV}"
SELECTOR_MODEL="${SELECTOR_MODEL:-qwen3-32b-judge}"
SELECTOR_BASE_URL="${SELECTOR_BASE_URL:-${LLM_BASE_URL:-http://127.0.1.1:8045/v1}}"

export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

DATASET="musique"
INPUT_DIR="${ROOT}/standalone_inputs/musique_limit100"
OUT_DIR="${ROOT}/standalone_runs/musique_limit100_echo_v2_smoke"
CACHE_PATH="${OUT_DIR}/cache/selector_cache.json"

CORPUS_JSON="${ROOT}/reproduce/dataset/musique_corpus.json"
PER_QUERY_REPORT="${ROOT}/RCR_DATASET_GOLD_ROWS_MUSIQUE_FULL_20260429.json"
BASELINE_CACHE="${INPUT_DIR}/baseline_top200_musique_limit100_compact.json"
ROLE_CHANNEL_JSON="${INPUT_DIR}/echo_role_channel_musique_limit100_shared_question_20260514.json"
SEED_SELECTOR_CACHE="${INPUT_DIR}/echo_v2_selector_musique_limit100_shared_question_20260514_cache.json"

mkdir -p "${OUT_DIR}/cache" "${OUT_DIR}/logs"
cd "${ROOT}"

test -s "${CORPUS_JSON}"
test -s "${PER_QUERY_REPORT}"
test -s "${BASELINE_CACHE}"
test -s "${ROLE_CHANNEL_JSON}"
test -s "${SEED_SELECTOR_CACHE}"

if [[ ! -s "${CACHE_PATH}" ]]; then
  cp "${SEED_SELECTOR_CACHE}" "${CACHE_PATH}"
fi

echo "[preflight] ROOT=${ROOT}"
echo "[preflight] PY=${PY}"
echo "[preflight] input_dir=${INPUT_DIR}"
echo "[preflight] out_dir=${OUT_DIR}"
echo "[preflight] selector=${SELECTOR_MODEL} ${SELECTOR_BASE_URL}"

"${PY}" "${ROOT}/evaluate_echo_support_profile_selector.py" \
  --dataset "${DATASET}" \
  --corpus_json "${CORPUS_JSON}" \
  --per_query_report "${PER_QUERY_REPORT}" \
  --per_query_variant dataset_gold_rows \
  --baseline_top200_cache "${BASELINE_CACHE}" \
  --role_channel_json "${ROLE_CHANNEL_JSON}" \
  --save_json_path "${OUT_DIR}/echo_v2_selector_musique_limit100.json" \
  --save_md_path "${OUT_DIR}/echo_v2_selector_musique_limit100.md" \
  --variant_name echo_v2_shared_question_fact_filter \
  --profile_mode full \
  --limit_queries 100 \
  --reader_top_k 5 \
  --candidate_top_k 12 \
  --baseline_candidate_top_k 0 \
  --max_candidates 24 \
  --max_passage_chars 760 \
  --selector_model "${SELECTOR_MODEL}" \
  --selector_base_url "${SELECTOR_BASE_URL}" \
  --api_key "${API_KEY_ARG}" \
  --temperature 0.0 \
  --max_tokens 512 \
  --retries 3 \
  --retry_wait_seconds 2 \
  --selector_cache_path "${CACHE_PATH}" \
  2>&1 | tee "${OUT_DIR}/logs/selector_smoke.log"
