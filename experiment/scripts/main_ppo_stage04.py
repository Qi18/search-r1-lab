from __future__ import annotations

import json
import re
from pathlib import Path

import hydra
import numpy as np
import ray
import torch

from verl import DataProto
from verl.trainer.main_ppo import RewardManager


ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


class RecordingRewardManager(RewardManager):
    def __init__(self, tokenizer, num_examine: int, output: str | None = None) -> None:
        super().__init__(tokenizer=tokenizer, num_examine=num_examine)
        self.output = Path(output) if output else None

    def __call__(self, data: DataProto):
        rewards = super().__call__(data)
        if self.output is None:
            return rewards
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("a", encoding="utf-8") as handle:
            for i in range(len(data)):
                item = data[i]
                prompt_length = item.batch["prompts"].shape[-1]
                valid_length = int(item.batch["attention_mask"][prompt_length:].sum().item())
                response = self.tokenizer.decode(item.batch["responses"][:valid_length])
                matches = ANSWER.findall(response)
                row = {
                    "id": str(item.non_tensor_batch.get("id", i)),
                    "question": str(item.non_tensor_batch.get("question", "")),
                    "golden_answers": list(item.non_tensor_batch.get("golden_answers", [])),
                    "data_source": str(item.non_tensor_batch.get("data_source", "unknown")),
                    "response": response,
                    "prediction": matches[-1].strip() if matches else "",
                    "search_count": len(re.findall(r"<search>", response, re.IGNORECASE)),
                    "reward": float(rewards[i, max(valid_length - 1, 0)].item()),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return rewards


@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from omegaconf import OmegaConf
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

    OmegaConf.resolve(config)
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)
    global_pool_id = "global_pool"
    val_only = bool(config.trainer.get("val_only", False))
    role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}
    mapping = {Role.ActorRollout: global_pool_id}
    if not val_only:
        role_worker_mapping.update({
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        })
        mapping.update({Role.Critic: global_pool_id, Role.RefPolicy: global_pool_id})

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
        mapping=mapping,
    )
    reward_fn = RecordingRewardManager(tokenizer, num_examine=0)
    val_reward_fn = RecordingRewardManager(
        tokenizer,
        num_examine=1,
        output=config.trainer.get("trajectory_output"),
    )
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
