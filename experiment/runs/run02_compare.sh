#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${LAB_ROOT}/env.sh"

ROOT="${SEARCH_R1_STAGE02_CACHE}"
EVAL_DIR="${ROOT}/evaluation"
DATA_DIR="${LAB_ROOT}/data/stage02"
INDEX_DIR="${SEARCH_R1_LAB_CACHE}/indexes/stage02-fixed96-e5-small-v2"
PRE="${EVAL_DIR}/qwen-base-pre.jsonl"
POST="${EVAL_DIR}/tiny-grpo-20step-post.jsonl"
CHECKPOINT="${ROOT}/02-tiny-grpo-20step/checkpoints/actor/global_step_20"
METRICS="${ROOT}/metrics.json"
REPORT="${LAB_ROOT}/results/02-tiny-grpo/results.md"

mkdir -p "${EVAL_DIR}"
for path in "${DATA_DIR}/corpus.jsonl" "${DATA_DIR}/val_eval.jsonl" \
  "${INDEX_DIR}/e5_Flat.index" "${CHECKPOINT}/config.json"; do
  test -e "${path}" || { echo "missing required path: ${path}" >&2; exit 1; }
done

if [[ ! -s "${PRE}" ]]; then
  CUDA_VISIBLE_DEVICES=0 python3 -u "${LAB_ROOT}/scripts/run_eval.py" \
    --model "${SEARCH_R1_BASE_MODEL_PATH}" \
    --model-label qwen-base-pre \
    --corpus "${DATA_DIR}/corpus.jsonl" \
    --eval "${DATA_DIR}/val_eval.jsonl" \
    --index "${INDEX_DIR}/e5_Flat.index" \
    --retriever-model "${SEARCH_R1_RETRIEVER_MODEL}" \
    --model-device cuda:0 --retriever-device cpu --mode both \
    --topk 3 --max-search-turns 2 --max-new-tokens 256 --output "${PRE}"
fi

CUDA_VISIBLE_DEVICES=0 python3 -u "${LAB_ROOT}/scripts/run_eval.py" \
  --model "${CHECKPOINT}" \
  --model-label tiny-grpo-20step-post \
  --corpus "${DATA_DIR}/corpus.jsonl" \
  --eval "${DATA_DIR}/val_eval.jsonl" \
  --index "${INDEX_DIR}/e5_Flat.index" \
  --retriever-model "${SEARCH_R1_RETRIEVER_MODEL}" \
  --model-device cuda:0 --retriever-device cpu --mode both \
  --topk 3 --max-search-turns 2 --max-new-tokens 256 --output "${POST}"

python3 -u "${LAB_ROOT}/scripts/summarize_stage02.py" \
  --pre-results "${PRE}" \
  --post-results "${POST}" \
  --run-root "${ROOT}" \
  --retriever-preflight "${ROOT}/retriever-preflight.json" \
  --metrics "${METRICS}" \
  --markdown "${REPORT}" \
  --require-pass

echo "Stage 02 comparison accepted: ${REPORT}"
