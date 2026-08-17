#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${LAB_ROOT}/env.sh"

RUN_NAME="${RUN_NAME:-00-search-smoke}"
RUN_DIR="${SEARCH_R1_LAB_CACHE}/experiments/${RUN_NAME}"
INDEX_DIR="${SEARCH_R1_LAB_CACHE}/indexes/synthetic-e5-small-v2"
CORPUS="${LAB_ROOT}/data/corpus.jsonl"
EVAL_DATA="${LAB_ROOT}/data/eval.jsonl"
RESULTS="${RUN_DIR}/trajectories.jsonl"
METRICS="${RUN_DIR}/metrics.json"
REPORT="${LAB_ROOT}/learning/experiments/00-search-smoke/results.md"

mkdir -p "${RUN_DIR}" "${INDEX_DIR}"

test -f "${SEARCH_R1_MODEL_PATH}/config.json" || {
  echo "missing model: ${SEARCH_R1_MODEL_PATH}" >&2
  exit 1
}

python3 - <<'PY'
import faiss
import torch
import transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("faiss", faiss.__version__)
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
  --model "${SEARCH_R1_MODEL_PATH}" \
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
  --output "${RESULTS}" \
  2>&1 | tee "${RUN_DIR}/run_eval.log"

python3 -u "${LAB_ROOT}/scripts/summarize.py" \
  --results "${RESULTS}" \
  --metrics "${METRICS}" \
  --markdown "${REPORT}" \
  2>&1 | tee "${RUN_DIR}/summarize.log"

echo "experiment complete: ${RUN_DIR}"
echo "report: ${REPORT}"
