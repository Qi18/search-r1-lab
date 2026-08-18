#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd "${LAB_ROOT}/.." && pwd)
source "${LAB_ROOT}/env.sh"

UPDATES="${UPDATES:-1}"
case "${UPDATES}" in 1|5) ;; *) echo "UPDATES must be 1 or 5" >&2; exit 2;; esac
STAGE_ROOT="${SEARCH_R1_STAGE04_CACHE:-/data/cache/search-r1/stage04-official}"
RUN_DIR="${STAGE_ROOT}/runs/stage04-nq-ppo-${UPDATES}step"
DATA_DIR="${LAB_ROOT}/data/stage04"
TRAIN_PYTHON="${SEARCH_R1_TRAIN_VENV}/bin/python"
ACTOR="${RUN_DIR}/checkpoints/actor/global_step_${UPDATES}"
CRITIC="${RUN_DIR}/checkpoints/critic/global_step_${UPDATES}"
mkdir -p "${RUN_DIR}"
curl -fsS http://127.0.0.1:8014/health

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS TOKENIZERS_PARALLELISM=true HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN

nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 >"${RUN_DIR}/gpu.csv" &
MONITOR_PID=$!
cleanup() { kill "${MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

cd "${REPO_ROOT}"
PYTHONPATH="${STAGE_ROOT}/python:${REPO_ROOT}/.deps:${LAB_ROOT}" "${TRAIN_PYTHON}" -u experiment/scripts/main_ppo_stage04.py \
  data.train_files="${DATA_DIR}/train.parquet" data.val_files="${DATA_DIR}/val.parquet" \
  data.train_data_num=null data.val_data_num=null data.train_batch_size=64 data.val_batch_size=64 \
  data.max_prompt_length=4096 data.max_response_length=500 data.max_start_length=1024 data.max_obs_length=500 \
  data.shuffle_train_dataloader=true algorithm.adv_estimator=gae algorithm.no_think_rl=false \
  algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.model.path="${SEARCH_R1_BASE_MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=64 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.n_agent=1 actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=64 actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.state_masking=true critic.model.path="${SEARCH_R1_BASE_MODEL_PATH}" \
  critic.model.enable_gradient_checkpointing=true critic.model.use_remove_padding=false critic.optim.lr=1e-5 \
  critic.optim.lr_warmup_steps_ratio=0.0 critic.ppo_micro_batch_size=8 \
  critic.model.fsdp_config.param_offload=true critic.model.fsdp_config.grad_offload=true \
  critic.model.fsdp_config.optimizer_offload=true trainer.critic_warmup=0 \
  trainer.logger="['console']" +trainer.val_only=false +trainer.val_before_train=false \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 trainer.save_freq="${UPDATES}" trainer.test_freq=-1 \
  trainer.project_name=Search-R1 trainer.experiment_name="stage04-nq-ppo-${UPDATES}step" \
  trainer.total_epochs=100 trainer.total_training_steps="$((UPDATES + 1))" \
  trainer.default_hdfs_dir=null trainer.default_local_dir="${RUN_DIR}/checkpoints" \
  max_turns=2 do_search=true retriever.url=http://127.0.0.1:8014/retrieve retriever.topk=3 \
  2>&1 | tee "${RUN_DIR}/train.log"

cleanup
trap - EXIT INT TERM
"${TRAIN_PYTHON}" "${LAB_ROOT}/scripts/check_stage04_run.py" \
  --log "${RUN_DIR}/train.log" --actor "${ACTOR}" --critic "${CRITIC}" \
  --gpu-csv "${RUN_DIR}/gpu.csv" --step "${UPDATES}" --output "${RUN_DIR}/acceptance.json"
