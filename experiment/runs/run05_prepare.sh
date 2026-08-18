#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${LAB_ROOT}/env.sh"
STAGE_ROOT="${SEARCH_R1_STAGE05_CACHE:-/data/cache/search-r1/stage05-v01}"
SOURCE="${SEARCH_R1_STAGE04_CACHE:-/data/cache/search-r1/stage04-official}/data/nq_hotpotqa_train"
DATA_DIR="${LAB_ROOT}/data/stage05"
TRAIN_PYTHON="${SEARCH_R1_TRAIN_VENV}/bin/python"

for path in "${SOURCE}/train.parquet" "${SOURCE}/test.parquet"; do
  test -s "${path}" || { echo "missing official asset: ${path}" >&2; exit 1; }
done
mkdir -p "${STAGE_ROOT}/runs"
PYTHONPATH="${STAGE_ROOT}/python:${SEARCH_R1_REPO_HOME}/.deps:${LAB_ROOT}" "${TRAIN_PYTHON}" \
  "${LAB_ROOT}/scripts/prepare_stage05_data.py" --source "${SOURCE}" --output "${DATA_DIR}" \
  --train-per-source 256 --val-per-source 16 --seed 505

cp "${LAB_ROOT}/data/stage04/retriever-manifest.json" "${DATA_DIR}/retriever-manifest.json"
"${TRAIN_PYTHON}" - <<PY
import json
from pathlib import Path
p=Path("${DATA_DIR}/manifest.json")
m=json.loads(p.read_text())
assert m["frozen_counts"]["train"] == {"hotpotqa": 256, "nq": 256}
assert len(m["frozen_counts"]["validation"]) == 7 and sum(m["frozen_counts"]["validation"].values()) == 112
print("STAGE05_DATA_PASS")
PY
