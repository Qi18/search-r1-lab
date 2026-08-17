from __future__ import annotations

import random

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from search_r1_lab.seeded_ppo import SeededActorRolloutRefWorker, seed_everything
from verl.trainer.main_ppo import RewardManager


@hydra.main(
    config_path="../../verl/trainer/config",
    config_name="ppo_trainer",
    version_base=None,
)
def main(config) -> None:
    seed = int(config.actor_rollout_ref.rollout.get("seed", 0))
    seed_everything(seed)
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "PYTHONHASHSEED": str(seed),
                    "SEARCH_R1_STAGE03_SEED": str(seed),
                }
            }
        )
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config) -> None:
    seed = int(config.actor_rollout_ref.rollout.get("seed", 0))
    seed_everything(seed)
    print(f"stage03_controller_seed={seed}")
    print(OmegaConf.to_yaml(config, resolve=True))
    OmegaConf.resolve(config)

    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.workers.fsdp_workers import CriticWorker

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)
    assert config.actor_rollout_ref.actor.strategy == "fsdp"
    assert config.actor_rollout_ref.actor.strategy == config.critic.strategy

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(SeededActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(SeededActorRolloutRefWorker),
    }
    pool_id = "global_pool"
    resource_pool_spec = {
        pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes
    }
    mapping = {
        Role.ActorRollout: pool_id,
        Role.Critic: pool_id,
        Role.RefPolicy: pool_id,
    }

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        ),
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
