#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd "${LAB_ROOT}/.." && pwd)
source "${LAB_ROOT}/env.sh"
STAGE_ROOT="${SEARCH_R1_STAGE04_CACHE:-/data/cache/search-r1/stage04-official}"
RUN_ROOT="${STAGE_ROOT}/runs"
RETRIEVER_DIR="${STAGE_ROOT}/retriever/wiki-18-e5"
TRAIN_PYTHON="${SEARCH_R1_TRAIN_VENV}/bin/python"
mkdir -p "${RUN_ROOT}"
: >"${RUN_ROOT}/retrieval-requests.jsonl"

PYTHONPATH="${STAGE_ROOT}/python:${REPO_ROOT}/.deps:${LAB_ROOT}" "${TRAIN_PYTHON}" -u "${LAB_ROOT}/scripts/retrieval_server_logged.py" \
  --index "${RETRIEVER_DIR}/e5_Flat.index" --corpus "${RETRIEVER_DIR}/wiki-18.jsonl" \
  --model "${STAGE_ROOT}/models/e5-base-v2" --faiss-gpu --port 8014 \
  --request-log "${RUN_ROOT}/retrieval-requests.jsonl" >"${RUN_ROOT}/retriever.log" 2>&1 &
RETRIEVER_PID=$!
cleanup() { kill "${RETRIEVER_PID}" 2>/dev/null || true; wait "${RETRIEVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

for _ in $(seq 1 1800); do
  curl -fsS http://127.0.0.1:8014/health >"${RUN_ROOT}/retriever-health.json" && break
  kill -0 "${RETRIEVER_PID}" 2>/dev/null || { tail -200 "${RUN_ROOT}/retriever.log" >&2; exit 1; }
  sleep 2
done
curl -fsS http://127.0.0.1:8014/health

PYTHONPATH="${STAGE_ROOT}/python:${REPO_ROOT}/.deps:${LAB_ROOT}" "${TRAIN_PYTHON}" "${LAB_ROOT}/scripts/preflight_stage04_retriever.py" \
  --validation "${LAB_ROOT}/data/stage04/val.parquet" --url http://127.0.0.1:8014/retrieve \
  --output "${RUN_ROOT}/retriever-preflight.json"

MODEL_PATH="${SEARCH_R1_BASE_MODEL_PATH}" EVAL_NAME=baseline "${LAB_ROOT}/runs/run04_eval.sh"
UPDATES=1 "${LAB_ROOT}/runs/run04_ppo.sh"
MODEL_PATH="${RUN_ROOT}/stage04-nq-ppo-1step/checkpoints/actor/global_step_1" EVAL_NAME=ppo-1step "${LAB_ROOT}/runs/run04_eval.sh"
UPDATES=5 "${LAB_ROOT}/runs/run04_ppo.sh"
MODEL_PATH="${RUN_ROOT}/stage04-nq-ppo-5step/checkpoints/actor/global_step_5" EVAL_NAME=ppo-5step "${LAB_ROOT}/runs/run04_eval.sh"

"${TRAIN_PYTHON}" "${LAB_ROOT}/scripts/summarize_stage04.py" \
  --root "${RUN_ROOT}" --output-json "${LAB_ROOT}/results/04-nq-ppo/metrics.json" \
  --output-md "${LAB_ROOT}/results/04-nq-ppo/results.md" --require-pass
