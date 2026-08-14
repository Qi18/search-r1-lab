#!/usr/bin/env bash

export SEARCH_R1_LAB_HOME="${SEARCH_R1_LAB_HOME:-/data/projects/search-r1-lab}"
export SEARCH_R1_LAB_CACHE="${SEARCH_R1_LAB_CACHE:-/data/cache/search-r1-lab}"
export SEARCH_R1_MODEL_PATH="${SEARCH_R1_MODEL_PATH:-/data/cache/search-r1/models/SearchR1-qwen2.5-3b-em-grpo}"
export SEARCH_R1_RETRIEVER_MODEL="${SEARCH_R1_RETRIEVER_MODEL:-intfloat/e5-small-v2}"
export HF_HOME="${HF_HOME:-/data/cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SEARCH_R1_LAB_HOME}/.deps:${SEARCH_R1_LAB_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
