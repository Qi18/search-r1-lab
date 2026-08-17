#!/usr/bin/env bash

_SEARCH_R1_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SEARCH_R1_EXPERIMENT_HOME="${SEARCH_R1_EXPERIMENT_HOME:-${_SEARCH_R1_ENV_DIR}}"
export SEARCH_R1_REPO_HOME="${SEARCH_R1_REPO_HOME:-$(cd "${SEARCH_R1_EXPERIMENT_HOME}/.." && pwd)}"
export SEARCH_R1_LAB_HOME="${SEARCH_R1_LAB_HOME:-${SEARCH_R1_EXPERIMENT_HOME}}"
export SEARCH_R1_LAB_CACHE="${SEARCH_R1_LAB_CACHE:-/data/cache/search-r1-lab}"
export SEARCH_R1_MODEL_PATH="${SEARCH_R1_MODEL_PATH:-/data/cache/search-r1/models/SearchR1-qwen2.5-3b-em-grpo}"
export SEARCH_R1_BASE_MODEL_PATH="${SEARCH_R1_BASE_MODEL_PATH:-/data/cache/search-r1/models/Qwen2.5-3B}"
export SEARCH_R1_RETRIEVER_MODEL="${SEARCH_R1_RETRIEVER_MODEL:-intfloat/e5-small-v2}"
export HF_HOME="${HF_HOME:-/data/cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SEARCH_R1_REPO_HOME}/.deps:${SEARCH_R1_EXPERIMENT_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
unset _SEARCH_R1_ENV_DIR
