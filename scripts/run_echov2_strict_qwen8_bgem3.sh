#!/usr/bin/env bash
set -euo pipefail

unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_ROOT="${CODE_ROOT:-${ROOT}}"
DATA_ROOT="${DATA_ROOT:-${ROOT}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${ROOT}}"
PY="${PY:-${ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${PYTHON:-python3}"
fi

DATE_TAG="${DATE_TAG:-20260602_passageseed}"
DATASETS="${DATASETS:-musique hotpotqa 2wikimultihopqa}"
LIMIT_QUERIES="${LIMIT_QUERIES:-1000}"
MAX_INDEX_DOCS="${MAX_INDEX_DOCS:-0}"
FORCE_INDEX_FROM_SCRATCH="${FORCE_INDEX_FROM_SCRATCH:-0}"
FORCE_OPENIE_FROM_SCRATCH="${FORCE_OPENIE_FROM_SCRATCH:-0}"

LLM_NAME="${LLM_NAME:-qwen3-8b}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8002/v1}"
EMBEDDING_NAME="${EMBEDDING_NAME:-Transformers/BAAI/bge-m3}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

RUN_ROOT="${ARTIFACT_ROOT}/outputs_echov2_qwen8_bgem3_limit${LIMIT_QUERIES}_${DATE_TAG}"
LOG_ROOT="${ARTIFACT_ROOT}/run_logs/echov2_qwen8_bgem3_limit${LIMIT_QUERIES}_${DATE_TAG}"
INDEX_ROOT="${INDEX_ROOT:-}"

export PYTHONPATH="${CODE_ROOT}/src:${CODE_ROOT}:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

dataset_tag() {
  case "$1" in
    2wikimultihopqa) echo "2wiki" ;;
    hotpotqa) echo "hotpot" ;;
    musique) echo "musique" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

gold_rows_path() {
  case "$1" in
    2wikimultihopqa) echo "${DATA_ROOT}/RCR_DATASET_GOLD_ROWS_2WIKI_FULL_20260429.json" ;;
    hotpotqa) echo "${DATA_ROOT}/RCR_DATASET_GOLD_ROWS_HOTPOT_FULL_20260429.json" ;;
    musique) echo "${DATA_ROOT}/RCR_DATASET_GOLD_ROWS_MUSIQUE_FULL_20260429.json" ;;
    *) echo "unknown dataset: $1" >&2; exit 2 ;;
  esac
}

corpus_json_path() {
  echo "${DATA_ROOT}/reproduce/dataset/$1"_corpus.json
}

dataset_json_path() {
  echo "${DATA_ROOT}/reproduce/dataset/$1.json"
}

dataset_root() {
  echo "${RUN_ROOT}/$1"
}

role_cache_path() {
  echo "$(dataset_root "$1")/roles/role_cache_limit${LIMIT_QUERIES}_${LLM_NAME}_${DATE_TAG}.json"
}

index_save_dir() {
  if [[ -n "${INDEX_ROOT}" ]]; then
    echo "${INDEX_ROOT}/$1/index"
  else
    echo "$(dataset_root "$1")/index"
  fi
}

baseline_cache_path() {
  echo "$(dataset_root "$1")/baseline/baseline_top200_limit${LIMIT_QUERIES}_${DATE_TAG}.json"
}

preflight() {
  test -x "${PY}" || command -v "${PY}" >/dev/null
  test -d "${CODE_ROOT}"
  test -d "${DATA_ROOT}"
  for dataset in ${DATASETS}; do
    test -s "$(corpus_json_path "${dataset}")"
    test -s "$(dataset_json_path "${dataset}")"
    test -s "$(gold_rows_path "${dataset}")"
  done
  echo "[preflight] code_root=${CODE_ROOT}"
  echo "[preflight] data_root=${DATA_ROOT}"
  echo "[preflight] run_root=${RUN_ROOT}"
  echo "[preflight] index_root=${INDEX_ROOT:-${RUN_ROOT}}"
  echo "[preflight] llm=${LLM_NAME} ${LLM_BASE_URL}"
  echo "[preflight] embedding=${EMBEDDING_NAME} base_url='${EMBEDDING_BASE_URL}'"
  echo "[preflight] datasets=${DATASETS} limit_queries=${LIMIT_QUERIES}"
}

build_roles() {
  local dataset="$1"
  local role_json log_file empty_count role_count role_status
  role_json="$(role_cache_path "${dataset}")"
  log_file="${LOG_ROOT}/$(dataset_tag "${dataset}")_roles_${DATE_TAG}.log"
  mkdir -p "$(dirname "${role_json}")"

  if [[ -s "${role_json}" ]]; then
    role_status="$("${PY}" -c "import json,sys; d=json.load(open(sys.argv[1])); r=d.get('roles_by_query',{}); print(len(r), sum(1 for v in r.values() if not v))" "${role_json}")"
    role_count="${role_status%% *}"
    empty_count="${role_status##* }"
    if [[ "${role_count}" -ge "${LIMIT_QUERIES}" && "${empty_count}" == "0" ]]; then
      echo "[$(date '+%F %T')] SKIP roles ${dataset} complete count=${role_count} empty=${empty_count}" | tee "${log_file}"
      return 0
    fi
  fi

  {
    echo "[$(date '+%F %T')] START roles ${dataset}"
    "${PY}" "${CODE_ROOT}/build_ergr_role_cache_v1.py" \
      --dataset_json "$(dataset_json_path "${dataset}")" \
      --output_json "${role_json}" \
      --limit "${LIMIT_QUERIES}" \
      --model "${LLM_NAME}" \
      --base_url "${LLM_BASE_URL}" \
      --temperature 0.0 \
      --timeout 180 \
      --max_tokens 512 \
      --concurrency 4 \
      --checkpoint_every 25 \
      --parse_retries 3 \
      --request_retries 6 \
      --retry_sleep_seconds 2 \
      --api_key EMPTY \
      --qwen_disable_thinking \
      --resume
    echo "[$(date '+%F %T')] DONE roles ${dataset} -> ${role_json}"
  } 2>&1 | tee "${log_file}"
}

build_index_and_baseline() {
  local dataset="$1"
  local log_file
  log_file="${LOG_ROOT}/$(dataset_tag "${dataset}")_index_baseline_${DATE_TAG}.log"
  {
    if [[ -s "$(baseline_cache_path "${dataset}")" && "${FORCE_INDEX_FROM_SCRATCH}" != "1" && "${FORCE_OPENIE_FROM_SCRATCH}" != "1" ]]; then
      echo "[$(date '+%F %T')] SKIP index/baseline ${dataset} existing $(baseline_cache_path "${dataset}")"
      return 0
    fi
    echo "[$(date '+%F %T')] START index/baseline ${dataset}"
    "${PY}" "${CODE_ROOT}/tmp_build_qwen8_bgem3_index_and_baseline.py" \
      --code-root "${CODE_ROOT}" \
      --dataset "${dataset}" \
      --corpus-json "$(corpus_json_path "${dataset}")" \
      --per-query-report "$(gold_rows_path "${dataset}")" \
      --save-dir "$(index_save_dir "${dataset}")" \
      --baseline-output-json "$(baseline_cache_path "${dataset}")" \
      --llm-name "${LLM_NAME}" \
      --llm-base-url "${LLM_BASE_URL}" \
      --embedding-name "${EMBEDDING_NAME}" \
      --embedding-base-url "${EMBEDDING_BASE_URL}" \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --retrieval-top-k 200 \
      --limit-queries "${LIMIT_QUERIES}" \
      --max-index-docs "${MAX_INDEX_DOCS}" \
      --qwen-disable-thinking
    echo "[$(date '+%F %T')] DONE index/baseline ${dataset}"
  } 2>&1 | tee "${log_file}"
}

run_variant() {
  local dataset="$1"
  local fact_strategy="$2"
  local variant_label="$3"
  local ds_root role_channel_json role_channel_md selector_json selector_md qa_json qa_md selector_cache qa_cache checkpoint_dir fact_filter_cache log_file

  ds_root="$(dataset_root "${dataset}")/${variant_label}"
  mkdir -p "${ds_root}/cache"

  role_channel_json="${ds_root}/role_channel.json"
  role_channel_md="${ds_root}/role_channel.md"
  selector_json="${ds_root}/selector.json"
  selector_md="${ds_root}/selector.md"
  qa_json="${ds_root}/qa.json"
  qa_md="${ds_root}/qa.md"
  selector_cache="${ds_root}/cache/selector_cache.json"
  qa_cache="${ds_root}/cache/qa_cache.json"
  checkpoint_dir="${ds_root}/cache/role_channel_checkpoints"
  fact_filter_cache="${ds_root}/cache/fact_filter_cache.json"
  log_file="${LOG_ROOT}/$(dataset_tag "${dataset}")_${variant_label}_${DATE_TAG}.log"

  {
    if [[ -s "${qa_json}" ]]; then
      echo "[$(date '+%F %T')] SKIP variant ${dataset} ${variant_label} existing qa ${qa_json}"
      return 0
    fi

    if [[ ! -s "${role_channel_json}" ]]; then
      echo "[$(date '+%F %T')] START role-channel ${dataset} strategy=${fact_strategy}"
      "${PY}" "${CODE_ROOT}/evaluate_role_channel_graph_retrieval.py" \
        --dataset "${dataset}" \
        --corpus_json "$(corpus_json_path "${dataset}")" \
        --per_query_report "$(gold_rows_path "${dataset}")" \
        --per_query_variant dataset_gold_rows \
        --baseline_top200_cache "$(baseline_cache_path "${dataset}")" \
        --roles_json_path "$(role_cache_path "${dataset}")" \
        --save_dir "$(index_save_dir "${dataset}")" \
        --output_json "${role_channel_json}" \
        --save_md_path "${role_channel_md}" \
        --limit_queries "${LIMIT_QUERIES}" \
        --reader_top_k 5 \
        --retrieval_top_k 200 \
        --num_to_retrieve 50 \
        --role_fact_top_k 5 \
        --role_passage_top_k 5 \
        --channel_output_top_k 50 \
        --rrf_k 60 \
        --fusion_method rrf \
        --channel_backend hipporag_graph \
        --chunk_size 5 \
        --llm_name "${LLM_NAME}" \
        --llm_base_url "${LLM_BASE_URL}" \
        --embedding_name "${EMBEDDING_NAME}" \
        --embedding_base_url "${EMBEDDING_BASE_URL}" \
        --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --llm_timeout_seconds 180 \
        --llm_read_timeout_seconds 180 \
        --llm_max_retries 2 \
        --rerank_filter_exception_policy raise \
        --rerank_filter_mode dspy \
        --fact_filter_strategy "${fact_strategy}" \
        --openie_mode online \
        --chunk_checkpoint_dir "${checkpoint_dir}" \
        --resume_chunk_checkpoints \
        --fact_filter_cache_json "${fact_filter_cache}" \
        --qwen_disable_thinking
    fi

    if [[ ! -s "${selector_json}" ]]; then
      echo "[$(date '+%F %T')] START selector ${dataset} variant=${variant_label}"
      "${PY}" "${CODE_ROOT}/evaluate_echo_support_profile_selector.py" \
        --dataset "${dataset}" \
        --corpus_json "$(corpus_json_path "${dataset}")" \
        --per_query_report "$(gold_rows_path "${dataset}")" \
        --per_query_variant dataset_gold_rows \
        --baseline_top200_cache "$(baseline_cache_path "${dataset}")" \
        --role_channel_json "${role_channel_json}" \
        --save_json_path "${selector_json}" \
        --save_md_path "${selector_md}" \
        --variant_name "${variant_label}" \
        --profile_mode full \
        --selection_prompt_mode coverage \
        --limit_queries "${LIMIT_QUERIES}" \
        --reader_top_k 5 \
        --candidate_top_k 12 \
        --baseline_candidate_top_k 0 \
        --max_candidates 24 \
        --max_passage_chars 760 \
        --selector_model "${LLM_NAME}" \
        --selector_base_url "${LLM_BASE_URL}" \
        --api_key EMPTY \
        --temperature 0.0 \
        --max_tokens 512 \
        --retries 3 \
        --retry_wait_seconds 2 \
        --selector_cache_path "${selector_cache}"
    fi

    echo "[$(date '+%F %T')] START QA ${dataset} variant=${variant_label}"
    "${PY}" "${CODE_ROOT}/rerun_qa_nothink_from_perquery.py" \
      --per-query "${selector_json}" \
      --corpus "$(corpus_json_path "${dataset}")" \
      --dataset "${dataset}" \
      --answer-alias-dataset "$(dataset_json_path "${dataset}")" \
      --variant "${variant_label}" \
      --output-json "${qa_json}" \
      --output-md "${qa_md}" \
      --cache "${qa_cache}" \
      --model "${LLM_NAME}" \
      --base-url "${LLM_BASE_URL}" \
      --api-key EMPTY \
      --temperature 0.0 \
      --max-tokens 400 \
      --concurrency 2 \
      --retries 8 \
      --retry-wait-seconds 2
    echo "[$(date '+%F %T')] DONE ${dataset} variant=${variant_label}"
  } 2>&1 | tee "${log_file}"
}

run_dataset() {
  local dataset="$1"
  build_roles "${dataset}"
  build_index_and_baseline "${dataset}"
  run_variant "${dataset}" "shared_question" "echov2_qwen8_bgem3_shared_question"
  run_variant "${dataset}" "shared_question_by_demand" "echov2_qwen8_bgem3_shared_question_by_demand"
}

preflight
for dataset in ${DATASETS}; do
  run_dataset "${dataset}"
done
echo "[$(date '+%F %T')] ALL DONE ${DATASETS}"
