#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd "${LAB_ROOT}/.." && pwd)
source "${LAB_ROOT}/env.sh"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
EVAL_NAME="${EVAL_NAME:?EVAL_NAME is required}"
STAGE_ROOT="${SEARCH_R1_STAGE05_CACHE:-/data/cache/search-r1/stage05-v01}"
RUN_DIR="${STAGE_ROOT}/runs/eval/${EVAL_NAME}"
TRAIN_PYTHON="${SEARCH_R1_TRAIN_VENV}/bin/python"
mkdir -p "${RUN_DIR}"
: >"${RUN_DIR}/trajectories.jsonl"
curl -fsS http://127.0.0.1:8015/health

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS TOKENIZERS_PARALLELISM=true HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN
cd "${REPO_ROOT}"
PYTHONPATH="${SEARCH_R1_STAGE04_CACHE:-/data/cache/search-r1/stage04-official}/python:${REPO_ROOT}/.deps:${LAB_ROOT}" \
"${TRAIN_PYTHON}" -u experiment/scripts/main_ppo_stage05.py \
  data.train_files="${LAB_ROOT}/data/stage05/train.parquet" data.val_files="${LAB_ROOT}/data/stage05/val.parquet" \
  data.train_batch_size=32 data.val_batch_size=56 data.max_prompt_length=4096 data.max_response_length=500 \
  data.max_start_length=2048 data.max_obs_length=500 algorithm.adv_estimator=grpo algorithm.no_think_rl=false \
  actor_rollout_ref.model.path="${MODEL_PATH}" actor_rollout_ref.model.enable_gradient_checkpointing=false \
  actor_rollout_ref.model.use_remove_padding=false actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.n_agent=1 actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  trainer.logger="['console']" +trainer.val_only=true +trainer.val_before_train=true \
  +trainer.trajectory_output="${RUN_DIR}/trajectories.jsonl" trainer.n_gpus_per_node=8 trainer.nnodes=1 \
  trainer.total_epochs=1 trainer.total_training_steps=1 trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${RUN_DIR}/unused" max_turns=4 do_search=true \
  retriever.url=http://127.0.0.1:8015/retrieve retriever.topk=3 2>&1 | tee "${RUN_DIR}/eval.log"

test "$(wc -l <"${RUN_DIR}/trajectories.jsonl")" -eq 112
for source in nq hotpotqa triviaqa popqa 2wikimultihopqa musique bamboogle; do
  grep -q "val/test_score/${source}" "${RUN_DIR}/eval.log"
done
echo "STAGE05_EVAL_PASS ${EVAL_NAME}"
