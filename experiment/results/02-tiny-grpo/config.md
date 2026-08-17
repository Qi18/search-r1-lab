# Stage 02 配置

## 运行环境

- GPU：8 x NVIDIA L20；
- 训练环境：`/data/venvs/search-r1-stage02`；
- PyTorch：2.4.0 + CUDA 12.1；
- vLLM：0.6.3；
- Ray：2.31.0；
- Transformers：4.47.1；
- FlashAttention：2.6.3；
- Base model：`Qwen2.5-3B Base`；
- Retriever：`intfloat/e5-small-v2`，CPU FlatIP 索引。

## 固定数据

- 训练：64 条 Stage 01 合成事实；
- 验证：32 条新增合成事实，与训练问题和实体完全分离；
- 检索库：96 篇文档；
- `data_source` 使用 `nq`，仅为了复用官方 EM reward 路由；数据本身不是 NQ。

## Tiny GRPO

| 参数 | 值 |
| --- | ---: |
| train batch | 32 |
| validation batch | 32 |
| rollouts / question (`n_agent`) | 4 |
| global PPO mini batch | 32 |
| global PPO micro batch | 8 |
| max turns | 2 |
| top-k | 3 |
| max response / observation | 256 / 256 |
| learning rate | 1e-6 |
| KL coefficient | 0.001 |
| checkpoint steps | 1, 5, 20 |

veRL Worker 会先将全局 mini/micro batch 除以 8 卡，因此 micro batch 不能取 4；这里使用 8，对应每卡 micro batch 1。

官方循环从 `global_steps=1` 开始，并在自增后判断退出。为准确完成 N 次优化，脚本向 `trainer.total_training_steps` 传入 `N+1`，checkpoint 仍保存在 `global_step_N`。

Retriever 使用独立 CPU 服务监听 `127.0.0.1:8012`，不占训练 GPU，也不会碰端口 8000 上已有的旧 smoke 服务。训练时日志只写本地 console；完成后从日志回填 SwanLab，并标记为 `backfill`。
