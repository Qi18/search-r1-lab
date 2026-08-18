#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${LAB_ROOT}/env.sh"
STAGE_ROOT="${SEARCH_R1_STAGE04_CACHE:-/data/cache/search-r1/stage04-official}"
SWANLAB_RUNTIME="${SEARCH_R1_SWANLAB_RUNTIME:-/data/cache/search-r1/swanlab-runtime}"
SWANLAB_PYTHON="${SEARCH_R1_SWANLAB_PYTHON:-/usr/bin/python3}"
LOGDIR="${STAGE_ROOT}/swanlab"
mkdir -p "${LOGDIR}"

for run in train-1 train-5 summary; do
  PYTHONPATH="${SWANLAB_RUNTIME}:${PYTHONPATH}" "${SWANLAB_PYTHON}" "${LAB_ROOT}/scripts/backfill_stage04_swanlab.py" \
    --run-root "${STAGE_ROOT}/runs" --metrics "${LAB_ROOT}/results/04-nq-ppo/metrics.json" \
    --logdir "${LOGDIR}" --manifest "${LOGDIR}/${run}.json" --run "${run}" --mode online
done
