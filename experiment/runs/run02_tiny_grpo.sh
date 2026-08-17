#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd "${LAB_ROOT}/.." && pwd)
source "${LAB_ROOT}/env.sh"

UPDATES="${UPDATES:-1}"
RUN_NAME="${RUN_NAME:-02-tiny-grpo-${UPDATES}step}"
RUN_DIR="${SEARCH_R1_STAGE02_CACHE}/${RUN_NAME}"
DATA_DIR="${LAB_ROOT}/data/stage02"
INDEX_DIR="${SEARCH_R1_LAB_CACHE}/indexes/stage02-fixed96-e5-small-v2"
CHECKPOINT="${RUN_DIR}/checkpoints/actor/global_step_${UPDATES}"
TRAIN_PYTHON="${SEARCH_R1_TRAIN_VENV}/bin/python"
TRAIN_LOG="${RUN_DIR}/train.log"
RETRIEVER_LOG="${RUN_DIR}/retriever.log"
RETRIEVER_PID=""

case "${UPDATES}" in
  1|5|20) ;;
  *) echo "UPDATES must be one of: 1, 5, 20" >&2; exit 2 ;;
esac

mkdir -p "${RUN_DIR}" "${INDEX_DIR}"

cleanup() {
  if [[ -n "${RETRIEVER_PID}" ]] && kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
    kill "${RETRIEVER_PID}" 2>/dev/null || true
    wait "${RETRIEVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for path in \
  "${TRAIN_PYTHON}" \
  "${SEARCH_R1_BASE_MODEL_PATH}/config.json" \
  "${DATA_DIR}/train.parquet" \
  "${DATA_DIR}/val.parquet" \
  "${DATA_DIR}/corpus.jsonl"; do
  test -e "${path}" || { echo "missing required path: ${path}" >&2; exit 1; }
done

"${SEARCH_R1_TRAIN_VENV}/bin/pip" check
"${TRAIN_PYTHON}" - <<'PY'
import flash_attn
import ray
import torch
import vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("ray", ray.__version__, "vllm", vllm.__version__, "flash_attn", flash_attn.__version__)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
PY

PYTHONPATH="${REPO_ROOT}/.deps:${LAB_ROOT}" python3 "${LAB_ROOT}/scripts/build_index.py" \
  --corpus "${DATA_DIR}/corpus.jsonl" \
  --index "${INDEX_DIR}/e5_Flat.index" \
  --metadata "${INDEX_DIR}/metadata.json" \
  --model "${SEARCH_R1_RETRIEVER_MODEL}" \
  --device cpu

PYTHONPATH="${REPO_ROOT}/.deps:${LAB_ROOT}" python3 -u "${LAB_ROOT}/scripts/serve_retriever.py" \
  --corpus "${DATA_DIR}/corpus.jsonl" \
  --index "${INDEX_DIR}/e5_Flat.index" \
  --model "${SEARCH_R1_RETRIEVER_MODEL}" \
  --device cpu \
  --port 8012 \
  >"${RETRIEVER_LOG}" 2>&1 &
RETRIEVER_PID=$!

for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8012/health >"${RUN_DIR}/retriever-health.json"; then
    break
  fi
  if ! kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
    echo "retriever exited during startup" >&2
    tail -n 100 "${RETRIEVER_LOG}" >&2
    exit 1
  fi
  sleep 1
done
curl -fsS http://127.0.0.1:8012/health

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

TOTAL_TRAINING_STEPS=$((UPDATES + 1))

cd "${REPO_ROOT}"
"${TRAIN_PYTHON}" -u -m verl.trainer.main_ppo \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  data.train_data_num=null \
  data.val_data_num=null \
  data.train_batch_size=32 \
  data.val_batch_size=32 \
  data.max_prompt_length=1024 \
  data.max_response_length=256 \
  data.max_start_length=512 \
  data.max_obs_length=256 \
  data.shuffle_train_dataloader=true \
  algorithm.adv_estimator=grpo \
  algorithm.no_think_rl=false \
  actor_rollout_ref.model.path="${SEARCH_R1_BASE_MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.grad_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.n_agent=4 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.state_masking=true \
  trainer.logger="['console']" \
  +trainer.val_only=false \
  +trainer.val_before_train=true \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq="${UPDATES}" \
  trainer.test_freq="${UPDATES}" \
  trainer.project_name=Search-R1 \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.total_epochs=100 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${RUN_DIR}/checkpoints" \
  max_turns=2 \
  do_search=true \
  retriever.url=http://127.0.0.1:8012/retrieve \
  retriever.topk=3 \
  2>&1 | tee "${TRAIN_LOG}"

"${TRAIN_PYTHON}" "${LAB_ROOT}/scripts/check_stage02_run.py" \
  --log "${TRAIN_LOG}" \
  --checkpoint "${CHECKPOINT}" \
  --expected-step "${UPDATES}" \
  --output "${RUN_DIR}/acceptance.json"

echo "Stage 02 run accepted: ${RUN_DIR}"
