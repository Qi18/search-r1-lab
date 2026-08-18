#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd "${LAB_ROOT}/.." && pwd)
source "${LAB_ROOT}/env.sh"
STAGE04_ROOT="${SEARCH_R1_STAGE04_CACHE:-/data/cache/search-r1/stage04-official}"
STAGE_ROOT="${SEARCH_R1_STAGE05_CACHE:-/data/cache/search-r1/stage05-v01}"
RUN_ROOT="${STAGE_ROOT}/runs"
RETRIEVER_DIR="${STAGE04_ROOT}/retriever/wiki-18-e5"
TRAIN_PYTHON="${SEARCH_R1_TRAIN_VENV}/bin/python"
mkdir -p "${RUN_ROOT}/formal"
echo $$ >"${RUN_ROOT}/formal/supervisor.pid"
: >"${RUN_ROOT}/formal/retrieval-requests.jsonl"

PYTHONPATH="${STAGE04_ROOT}/python:${REPO_ROOT}/.deps:${LAB_ROOT}" "${TRAIN_PYTHON}" -u \
  "${LAB_ROOT}/scripts/retrieval_server_logged.py" --index "${RETRIEVER_DIR}/e5_Flat.index" \
  --corpus "${RETRIEVER_DIR}/wiki-18.jsonl" --model "${STAGE04_ROOT}/models/e5-base-v2" \
  --faiss-gpu --port 8015 --request-log "${RUN_ROOT}/formal/retrieval-requests.jsonl" \
  >"${RUN_ROOT}/formal/retriever.log" 2>&1 &
RETRIEVER_PID=$!
cleanup() { kill "${RETRIEVER_PID}" 2>/dev/null || true; wait "${RETRIEVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for _ in $(seq 1 1800); do
  curl -fsS http://127.0.0.1:8015/health >"${RUN_ROOT}/formal/retriever-health.json" && break
  kill -0 "${RETRIEVER_PID}" 2>/dev/null || { tail -200 "${RUN_ROOT}/formal/retriever.log" >&2; exit 1; }
  sleep 2
done
curl -fsS http://127.0.0.1:8015/health

SWANLAB_LIVE=true ALGORITHM=ppo UPDATES=305 "${LAB_ROOT}/runs/run05_train.sh"
MODEL_PATH="${RUN_ROOT}/ppo-305step/checkpoints/actor/global_step_300" \
  EVAL_NAME=ppo-300step "${LAB_ROOT}/runs/run05_eval.sh"
SWANLAB_LIVE=true ALGORITHM=grpo UPDATES=305 "${LAB_ROOT}/runs/run05_train.sh"
MODEL_PATH="${RUN_ROOT}/grpo-305step/checkpoints/actor/global_step_300" \
  EVAL_NAME=grpo-300step "${LAB_ROOT}/runs/run05_eval.sh"
echo FINAL_STAGE05_FORMAL_PASS
