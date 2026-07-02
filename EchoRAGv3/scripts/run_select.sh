#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: scripts/run_select.sh <typed_atoms_json> <corpus_json> <output_json> <output_md> [top_k]" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python}"
top_k="${5:-5}"

"$python_bin" "$project_root/src/echoragv3_selector.py" \
  --typed-atoms-json "$1" \
  --corpus-json "$2" \
  --output-json "$3" \
  --output-md "$4" \
  --top-k "$top_k" \
  --mode completion
