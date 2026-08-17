# Stage 02: Tiny GRPO

目标是从 `Qwen2.5-3B Base` 出发，在 8 x L20 上完成 Search-R1 官方 veRL/GRPO 闭环，而不是继续评测已有的官方 GRPO checkpoint。

分级运行：

```bash
source experiment/env.sh
PYTHONPATH="$(pwd)/experiment" python3 experiment/scripts/generate_stage02_data.py \
  --stage01-dir experiment/data/stage01 \
  --output-dir experiment/data/stage02

UPDATES=1 bash experiment/runs/run02_tiny_grpo.sh
UPDATES=5 bash experiment/runs/run02_tiny_grpo.sh
UPDATES=20 bash experiment/runs/run02_tiny_grpo.sh
```

训练和保存完成后，使用同一冻结验证集回载 20-step checkpoint：

```bash
bash experiment/runs/run02_compare.sh
```

Git 只保存固定数据、配置、验收摘要和结论。模型、checkpoint、日志与完整轨迹保存在：

```text
/data/cache/search-r1/experiments/02-tiny-grpo/
```

历史指标可回填为 SwanLab 离线实验，再使用 `swanlab sync` 上传：

```bash
PYTHONPATH=/data/cache/search-r1/swanlab-runtime:experiment/scripts \
  python3 experiment/scripts/backfill_stage02_swanlab.py \
  --run-root /data/cache/search-r1/experiments/02-tiny-grpo \
  --logdir /data/cache/search-r1/experiments/02-tiny-grpo/swanlog \
  --manifest /data/cache/search-r1/experiments/02-tiny-grpo/swanlab-backfill-manifest.json
```

最终验收以 20-step checkpoint 可回载、同一冻结验证集的训练前后对照，以及 reward/KL/loss 均为有限值为准。
