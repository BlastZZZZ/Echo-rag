#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${PY:-${ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${PYTHON:-python3}"
fi

export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"
cd "${ROOT}"

"${PY}" -m py_compile \
  build_ergr_role_cache_v1.py \
  role_aware_graph_entry.py \
  role_channel_graph_retrieval.py \
  evaluate_role_channel_graph_retrieval.py \
  evaluate_echo_support_profile_selector.py \
  rerun_qa_nothink_from_perquery.py \
  rerun_qa_openai_from_perquery.py \
  tmp_build_qwen8_bgem3_index_and_baseline.py

"${PY}" - <<'PY'
import build_ergr_role_cache_v1
import role_aware_graph_entry
import role_channel_graph_retrieval
import evaluate_role_channel_graph_retrieval
import evaluate_echo_support_profile_selector
import rerun_qa_nothink_from_perquery

print("static imports ok")
PY
