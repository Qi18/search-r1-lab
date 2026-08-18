#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${LAB_ROOT}/env.sh"
STAGE_ROOT="${SEARCH_R1_STAGE05_CACHE:-/data/cache/search-r1/stage05-v01}"
RUN_ROOT="${STAGE_ROOT}/runs"
RESULT_ROOT="${LAB_ROOT}/results/05-multidata-ppo-grpo"
SWAN_ROOT="${STAGE_ROOT}/swanlab-backfill"
SWAN_PYTHON=/usr/bin/python3
mkdir -p "${SWAN_ROOT}" "${RESULT_ROOT}"
for run in ppo grpo summary; do
  PYTHONPATH="/data/cache/search-r1/swanlab-runtime:${PYTHONPATH:-}" "${SWAN_PYTHON}" \
    "${LAB_ROOT}/scripts/backfill_stage05_swanlab.py" --run-root "${RUN_ROOT}" \
    --metrics "${RESULT_ROOT}/metrics.json" --logdir "${SWAN_ROOT}/${run}" \
    --manifest "${SWAN_ROOT}/${run}.json" --run "${run}"
done
"${SWAN_PYTHON}" - <<PY
import json
from pathlib import Path
root=Path("${SWAN_ROOT}")
payload={name: json.loads((root/f"{name}.json").read_text()) for name in ("ppo", "grpo", "summary")}
Path("${RESULT_ROOT}/swanlab.json").write_text(json.dumps(payload, indent=2)+"\n")
PY
