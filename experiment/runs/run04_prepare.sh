#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd "${LAB_ROOT}/.." && pwd)
source "${LAB_ROOT}/env.sh"

STAGE_ROOT="${SEARCH_R1_STAGE04_CACHE:-/data/cache/search-r1/stage04-official}"
RAW_DATA="${STAGE_ROOT}/data/nq_hotpotqa_train"
RETRIEVER_DIR="${STAGE_ROOT}/retriever/wiki-18-e5"
MODEL_DIR="${STAGE_ROOT}/models/e5-base-v2"
DATA_DIR="${LAB_ROOT}/data/stage04"
TRAIN_PYTHON="${SEARCH_R1_TRAIN_VENV}/bin/python"

for path in "${RAW_DATA}/train.parquet" "${RAW_DATA}/test.parquet" "${RETRIEVER_DIR}/part_aa" "${RETRIEVER_DIR}/part_ab" "${RETRIEVER_DIR}/wiki-18.jsonl.gz"; do
  test -s "${path}" || { echo "missing asset: ${path}" >&2; exit 1; }
done

if [[ ! -s "${RETRIEVER_DIR}/e5_Flat.index" ]]; then
  cat "${RETRIEVER_DIR}/part_aa" "${RETRIEVER_DIR}/part_ab" >"${RETRIEVER_DIR}/e5_Flat.index.tmp"
  mv "${RETRIEVER_DIR}/e5_Flat.index.tmp" "${RETRIEVER_DIR}/e5_Flat.index"
fi
if [[ ! -s "${RETRIEVER_DIR}/wiki-18.jsonl" ]]; then
  gzip -dk "${RETRIEVER_DIR}/wiki-18.jsonl.gz"
fi

source "${SEARCH_R1_TRAIN_VENV}/bin/activate"
HF_ENDPOINT="${HF_ENDPOINT}" hf download intfloat/e5-base-v2 \
  config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt \
  --local-dir "${MODEL_DIR}" --max-workers 4
PYTHONPATH="${STAGE_ROOT}/python:${REPO_ROOT}/.deps:${LAB_ROOT}" "${TRAIN_PYTHON}" "${LAB_ROOT}/scripts/prepare_stage04_data.py" \
  --source "${RAW_DATA}" --output "${DATA_DIR}" --train-size 512 --val-size 128 --seed 404

"${TRAIN_PYTHON}" - <<PY
import hashlib, json
from pathlib import Path
root=Path("${RETRIEVER_DIR}")
def sha(path):
 d=hashlib.sha256()
 with path.open("rb") as h:
  for chunk in iter(lambda:h.read(16*1024*1024),b""): d.update(chunk)
 return d.hexdigest()
payload={"source":{"index":"PeterJinGo/wiki-18-e5-index","corpus":"PeterJinGo/wiki-18-corpus","retriever":"intfloat/e5-base-v2"},"bytes":{"index":(root/"e5_Flat.index").stat().st_size,"corpus":(root/"wiki-18.jsonl").stat().st_size},"sha256":{"index":sha(root/"e5_Flat.index"),"corpus":sha(root/"wiki-18.jsonl")}}
Path("${DATA_DIR}/retriever-manifest.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps(payload,indent=2))
PY
