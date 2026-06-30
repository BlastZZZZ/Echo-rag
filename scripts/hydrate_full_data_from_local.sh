#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/hydrate_full_data_from_local.sh ../full-package-or-source-root" >&2
  exit 2
fi

SOURCE_ROOT="$1"

SOURCE_DATASET_DIR="${SOURCE_ROOT}/reproduce/dataset"
TARGET_DATASET_DIR="${ROOT}/reproduce/dataset"

if [[ ! -d "${SOURCE_DATASET_DIR}" ]]; then
  echo "source dataset directory not found: ${SOURCE_DATASET_DIR}" >&2
  echo "usage: bash scripts/hydrate_full_data_from_local.sh ../full-package-or-source-root" >&2
  exit 2
fi

mkdir -p "${TARGET_DATASET_DIR}"

for name in \
  musique.json musique_corpus.json \
  hotpotqa.json hotpotqa_corpus.json \
  2wikimultihopqa.json 2wikimultihopqa_corpus.json \
  2wiki.json 2wiki_corpus.json \
  nq.json nq_corpus.json \
  nq_rear.json nq_rear_corpus.json \
  popqa.json popqa_corpus.json \
  hgrag_agriculture.json hgrag_agriculture_corpus.json hgrag_agriculture_mapping_audit.json \
  hgrag_legal.json hgrag_legal_corpus.json hgrag_legal_mapping_audit.json \
  hgrag_cs.json hgrag_cs_corpus.json hgrag_cs_mapping_audit.json \
  hgrag_mix.json hgrag_mix_corpus.json hgrag_mix_mapping_audit.json \
  hgrag_hypertension.json hgrag_hypertension_corpus.json hgrag_hypertension_mapping_audit.json
do
  if [[ -s "${SOURCE_DATASET_DIR}/${name}" ]]; then
    cp -a "${SOURCE_DATASET_DIR}/${name}" "${TARGET_DATASET_DIR}/${name}"
  fi
done

echo "hydrated dataset files from ${SOURCE_DATASET_DIR}"
echo "target: ${TARGET_DATASET_DIR}"
