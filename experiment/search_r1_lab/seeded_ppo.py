from __future__ import annotations

import os
import random

import numpy as np
import torch

from verl.third_party.vllm import LLM, vllm_version
from verl.single_controller.base import Worker
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout
from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
from vllm import SamplingParams


def seed_everything(seed: int) -> None:
    """Seed the controller or one distributed worker."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SeededVLLMRollout(vLLMRollout):
    """Search-R1 vLLM rollout with the experiment seed passed to the engine."""

    def __init__(self, actor_module, config, tokenizer, model_hf_config, **kwargs):
        super(vLLMRollout, self).__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine)

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size()
        seed = int(config.get("seed", 0))

        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            load_format=config.load_format,
            seed=seed,
        )
        self.inference_engine.offload_model_weights()

        sampling_kwargs = {
            "n": 1,
            "logprobs": 1,
            "max_tokens": config.response_length,
        }
        if vllm_version in ("0.4.2", "0.5.4", "0.6.3"):
            sampling_kwargs["detokenize"] = False
        for key in config.keys():
            if hasattr(SamplingParams(), str(key)):
                sampling_kwargs[key] = config.get(key)

        print(f"stage03_seed={seed}, sampling_kwargs={sampling_kwargs}")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        self.pad_token_id = tokenizer.pad_token_id


class SeededFSDPVLLMShardingManager(FSDPVLLMShardingManager):
    """Use seed + data-parallel rank for rollout CUDA RNG state."""

    def __init__(self, *args, seed: int, **kwargs):
        super().__init__(*args, **kwargs)
        if self.device_mesh is not None:
            training_state = torch.cuda.get_rng_state()
            dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(int(seed) + dp_rank)
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(training_state)


class SeededActorRolloutRefWorker(Worker):
    """Official FSDP worker with experiment-scoped deterministic RNG setup."""

    def __init__(self, config, role: str):
        Worker.__init__(self)
        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        from torch.distributed.device_mesh import init_device_mesh
        from verl.workers.sharding_manager.fsdp_ulysses import (
            FSDPUlyssesShardingManager,
        )

        world_size = torch.distributed.get_world_size()
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get(
            "ulysses_sequence_parallel_size", 1
        )
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(
            self.ulysses_device_mesh
        )

        self.role = role
        assert role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = role in ["ref", "actor_rollout_ref"]
        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            fsdp_config = self.config.actor.fsdp_config
            self._is_offload_param = fsdp_config.get("param_offload", False)
            self._is_offload_grad = fsdp_config.get("grad_offload", False)
            self._is_offload_optimizer = fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        data_parallel_size = self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= data_parallel_size
            self.config.actor.ppo_micro_batch_size //= data_parallel_size
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_micro_batch_size *= self.config.rollout.n
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= data_parallel_size
            self.config.rollout.log_prob_micro_batch_size *= self.config.rollout.n
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= data_parallel_size
            self.config.ref.log_prob_micro_batch_size *= self.config.rollout.n

        rank = int(os.environ.get("RANK", "0"))
        seed_everything(int(config.rollout.get("seed", 0)) + rank)

    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh

        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0
        rollout_device_mesh = init_device_mesh(
            "cuda", mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )

        if self.config.rollout.name != "vllm":
            return super()._build_rollout()

        rollout = SeededVLLMRollout(
            actor_module=self.actor_module_fsdp,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            model_hf_config=self.actor_model_config,
        )
        if torch.distributed.get_world_size() == 1:
            self.config.rollout.load_format = "dummy_hf"
        manager = SeededFSDPVLLMShardingManager(
            module=self.actor_module_fsdp,
            inference_engine=rollout.inference_engine,
            model_config=self.actor_model_config,
            full_params="hf" in self.config.rollout.load_format,
            device_mesh=rollout_device_mesh,
            seed=int(self.config.rollout.get("seed", 0)),
        )
        return rollout, manager


# Ray's colocated worker factory requires the direct base class to remain Worker.
# Reuse the official implementation while keeping the two seeded overrides above.
for _name, _value in ActorRolloutRefWorker.__dict__.items():
    if _name not in {"__dict__", "__weakref__", "__init__", "_build_rollout"}:
        setattr(SeededActorRolloutRefWorker, _name, _value)
