#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${LAB_ROOT}/env.sh"

RUN_NAME="${RUN_NAME:-01-base-vs-rl}"
RUN_DIR="${SEARCH_R1_LAB_CACHE}/experiments/${RUN_NAME}"
INDEX_DIR="${SEARCH_R1_LAB_CACHE}/indexes/synthetic-e5-small-v2"
CORPUS="${LAB_ROOT}/data/corpus.jsonl"
EVAL_DATA="${LAB_ROOT}/data/eval.jsonl"
BASE_RESULTS="${RUN_DIR}/qwen-base.jsonl"
GRPO_RESULTS="${RUN_DIR}/search-r1-grpo.jsonl"
METRICS="${RUN_DIR}/metrics.json"
REPORT="${LAB_ROOT}/results/01-base-vs-rl/results.md"

mkdir -p "${RUN_DIR}" "${INDEX_DIR}"

for model_path in "${SEARCH_R1_BASE_MODEL_PATH}" "${SEARCH_R1_MODEL_PATH}"; do
  test -f "${model_path}/config.json" || {
    echo "missing model: ${model_path}" >&2
    exit 1
  }
done

python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 1
PY

python3 -m unittest discover -s "${LAB_ROOT}/tests" -v \
  2>&1 | tee "${RUN_DIR}/tests.log"

python3 -u "${LAB_ROOT}/scripts/build_index.py" \
  --corpus "${CORPUS}" \
  --index "${INDEX_DIR}/e5_Flat.index" \
  --metadata "${INDEX_DIR}/metadata.json" \
  --model "${SEARCH_R1_RETRIEVER_MODEL}" \
  --device cpu \
  2>&1 | tee "${RUN_DIR}/build_index.log"

CUDA_VISIBLE_DEVICES=0 python3 -u "${LAB_ROOT}/scripts/run_eval.py" \
  --model "${SEARCH_R1_BASE_MODEL_PATH}" \
  --model-label qwen-base \
  --corpus "${CORPUS}" \
  --eval "${EVAL_DATA}" \
  --index "${INDEX_DIR}/e5_Flat.index" \
  --retriever-model "${SEARCH_R1_RETRIEVER_MODEL}" \
  --model-device cuda:0 \
  --retriever-device cpu \
  --mode both \
  --topk 3 \
  --max-search-turns 2 \
  --max-new-tokens 256 \
  --output "${BASE_RESULTS}" \
  2>&1 | tee "${RUN_DIR}/qwen-base.log"

CUDA_VISIBLE_DEVICES=0 python3 -u "${LAB_ROOT}/scripts/run_eval.py" \
  --model "${SEARCH_R1_MODEL_PATH}" \
  --model-label search-r1-grpo \
  --corpus "${CORPUS}" \
  --eval "${EVAL_DATA}" \
  --index "${INDEX_DIR}/e5_Flat.index" \
  --retriever-model "${SEARCH_R1_RETRIEVER_MODEL}" \
  --model-device cuda:0 \
  --retriever-device cpu \
  --mode both \
  --topk 3 \
  --max-search-turns 2 \
  --max-new-tokens 256 \
  --output "${GRPO_RESULTS}" \
  2>&1 | tee "${RUN_DIR}/search-r1-grpo.log"

python3 -u "${LAB_ROOT}/scripts/summarize_stage01.py" \
  --base-results "${BASE_RESULTS}" \
  --grpo-results "${GRPO_RESULTS}" \
  --metrics "${METRICS}" \
  --markdown "${REPORT}" \
  2>&1 | tee "${RUN_DIR}/summarize.log"

echo "experiment complete: ${RUN_DIR}"
echo "report: ${REPORT}"
