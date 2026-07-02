#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "usage: scripts/run_reader_eval.sh <typed_chain_json> <corpus_json> <dataset_json> <dataset_label> <output_json> <output_md> <cache_json> [limit]" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python}"
limit="${8:-0}"
model="${OPENAI_MODEL:-gpt-4o-mini}"
base_url="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

"$python_bin" "$project_root/src/reader_eval.py" \
  --typed-chain-json "$1" \
  --corpus-json "$2" \
  --dataset-json "$3" \
  --dataset "$4" \
  --output-json "$5" \
  --output-md "$6" \
  --cache "$7" \
  --model "$model" \
  --base-url "$base_url" \
  --api-key ENV \
  --temperature 0.0 \
  --max-tokens 80 \
  --concurrency "${OPENAI_CONCURRENCY:-2}" \
  --retries "${OPENAI_RETRIES:-5}" \
  --retry-wait-seconds "${OPENAI_RETRY_WAIT_SECONDS:-4}" \
  --limit "$limit" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-1000}" \
  --proof-format grounded_atoms \
  --variant raw_context_empty_annotation \
  --variant raw_context_plus_proof_annotation
