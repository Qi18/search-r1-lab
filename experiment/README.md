# Search-R1 L20 Experiment

基于官方完整 Search-R1 的独立实验区。官方源码保留在仓库根目录，所有自定义脚本、数据和结论均位于 `experiment/`。

## Stage 状态

| Stage | 状态 | 作用 |
| --- | --- | --- |
| 00 | PASS | 搜索闭环冒烟 |
| 01 | PASS | Base/GRPO × Search on/off 四象限 |
| 02 | PASS | Tiny GRPO 训练与 checkpoint 回载 |
| 03 | PASS | seed 与可复现性门禁 |
| 04 | NEXT | 官方 Preliminary：NQ + PPO |
| 05 | PENDING | 官方 v0.1：多数据集 PPO/GRPO |
| 06 | PENDING | 官方 v0.2：masking、长训练和模型规模 |
| 07 | PENDING | 官方 v0.3：reward/backbone/retriever/data 消融 |
| 08 | PENDING | 自定义鲁棒性扩展 |

详细配置和验收条件见 `TRAINING_PLAN.md`。

## 已有入口

```bash
source experiment/env.sh

# Stage00
bash experiment/runs/runsmoke.sh

# Stage01
bash experiment/runs/run01_base_vs_rl.sh

# Stage02
UPDATES=20 bash experiment/runs/run02_tiny_grpo.sh

# Stage03 seed gate 单次运行
SEED=3103 REPLICA=a UPDATES=2 bash experiment/runs/run03_seed_smoke.sh
```

Stage04 尚未创建运行脚本；下一步先做官方 NQ/Wiki-18/E5 数据与索引 preflight，不把未验收命令写成可运行入口。

## 目录作用

| 路径 | 作用 |
| --- | --- |
| `README.md` | Stage 状态和运行入口 |
| `TRAINING_PLAN.md` | 对齐官方 Preliminary/v0.1/v0.2/v0.3 的实验路线 |
| `FINAL_REPORT.md` | 当前已验证结果和结论边界 |
| `env.sh` | 模型、缓存和 Python 路径 |
| `data/` | 可提交的小型数据与 manifest |
| `runs/` | 一键运行脚本 |
| `scripts/` | 数据、索引、评测、验收和 SwanLab 工具 |
| `search_r1_lab/` | 实验辅助模块 |
| `tests/` | 协议与指标回归测试 |
| `results/` | 各 Stage 结果摘要 |

## 产物边界

Git 只保存源码、小型数据、配置、manifest 和结果摘要。模型、checkpoint、大型索引、原始日志与完整 rollout 位于 `/data/cache/search-r1/`。

## References

- Official code: https://github.com/PeterGriffinJin/Search-R1
- Paper 1: https://arxiv.org/abs/2503.09516
- Paper 2: https://arxiv.org/abs/2505.15117
- Official experiment log: https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/experiment_log.md
