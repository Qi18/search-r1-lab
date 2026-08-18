from __future__ import annotations

import atexit
import json
import math
import re
from pathlib import Path

import hydra
import ray

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
            for index in range(len(data)):
                item = data[index]
                prompt_length = item.batch["prompts"].shape[-1]
                valid_length = int(item.batch["attention_mask"][prompt_length:].sum().item())
                response = self.tokenizer.decode(item.batch["responses"][:valid_length])
                matches = ANSWER.findall(response)
                handle.write(json.dumps({
                    "id": str(item.non_tensor_batch.get("id", index)),
                    "question": str(item.non_tensor_batch.get("question", "")),
                    "golden_answers": list(item.non_tensor_batch.get("golden_answers", [])),
                    "data_source": str(item.non_tensor_batch.get("data_source", "unknown")),
                    "response": response,
                    "prediction": matches[-1].strip() if matches else "",
                    "search_count": len(re.findall(r"<search>", response, re.IGNORECASE)),
                    "reward": float(rewards[index, max(valid_length - 1, 0)].item()),
                }, ensure_ascii=False) + "\n")
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
    tokenizer = hf_tokenizer(copy_local_path_from_hdfs(config.actor_rollout_ref.model.path))
    val_only = bool(config.trainer.get("val_only", False))
    disable_validation = bool(config.trainer.get("disable_validation", False))
    algorithm = str(config.algorithm.adv_estimator)
    pool = "global_pool"
    role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}
    mapping = {Role.ActorRollout: pool}
    if not val_only:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = pool
        if algorithm == "gae":
            role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
            mapping[Role.Critic] = pool

    reward_fn = RecordingRewardManager(tokenizer, num_examine=0)
    val_reward_fn = None if disable_validation else RecordingRewardManager(
        tokenizer, num_examine=1, output=config.trainer.get("trajectory_output")
    )
    if bool(config.trainer.get("swanlab_live", False)):
        import swanlab
        import verl.trainer.ppo.ray_trainer as trainer_module
        from verl.utils.logger.aggregate_logger import LocalLogger

        swanlab_logdir = str(config.trainer.get("swanlab_logdir", "swanlog"))
        swanlab_config = {
            "algorithm": algorithm,
            "base_model": str(config.actor_rollout_ref.model.path),
            "train_batch_size": int(config.data.train_batch_size),
            "n_agent": int(config.actor_rollout_ref.rollout.n_agent),
            "max_turns": int(config.max_turns),
            "topk": int(config.retriever.topk),
            "learning_rate": float(config.actor_rollout_ref.actor.optim.lr),
            "total_training_steps": int(config.trainer.total_training_steps) - 1,
            "gpus": int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes),
            "state_masking": bool(config.actor_rollout_ref.actor.state_masking),
        }

        class SwanlabConsoleTracking:
            def __init__(self, project_name, experiment_name, default_backend="console", config=None):
                self.console = LocalLogger(print_to_console=True)
                self.run = swanlab.init(
                    project=project_name,
                    experiment_name=experiment_name,
                    description="Live Stage05 official-v0.1 route; metrics emitted directly by veRL.",
                    group="stage05-official-v01-formal",
                    tags=["Search-R1", "Stage05", algorithm.upper(), "live"],
                    logdir=swanlab_logdir,
                    mode="online",
                    config=swanlab_config,
                    reinit=True,
                )
                atexit.register(swanlab.finish)

            def log(self, data, step, backend=None):
                self.console.log(data=data, step=step)
                finite = {key: float(value) for key, value in data.items() if isinstance(value, (int, float)) and math.isfinite(float(value))}
                self.run.log(finite, step=int(step))

        trainer_module.Tracking = SwanlabConsoleTracking
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=ResourcePoolManager(
            resource_pool_spec={pool: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}, mapping=mapping
        ),
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
