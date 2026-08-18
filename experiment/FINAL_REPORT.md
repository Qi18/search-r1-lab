# Search-R1 当前实验报告

更新时间：2026-08-18

## 当前结论

Stage00–04 已完成。项目已从合成数据门禁进入官方真实数据：NQ、Wiki-18/E5 Retriever、PPO Actor/Critic、checkpoint 保存与独立回载评测均已打通。

下一阶段是官方 v0.1 路线：从 NQ 扩展多数据集，并比较 PPO/GRPO。

## 已完成阶段

| Stage | 关键结果 | 结论边界 |
| --- | --- | --- |
| 00 | search EM 62.5%，no-search EM 0%，Hit@3 100% | 只有 8 条合成问题 |
| 01 | Base+Search EM 4.7%，官方 GRPO+Search EM 78.1% | 64 条合成事实，Retriever 条件命中率 100% |
| 02 | 20-step 后冻结 val Search EM 28.125%，Base 为 6.25% | train64/val32，不能外推真实 QA |
| 03 | 同 seed 训练轨迹哈希一致，异 seed 哈希变化 | GPU 更新不保证位级确定 |
| 04 | Base/1-step/5-step EM 3.125%/3.125%/5.469%，5-step F1 8.95% | 官方 NQ + Wiki-18，但仅 5 updates / val128 |

## Stage03 可复现性边界

- seed=3103 的两次 2-step 训练期检索轨迹完全一致；reward 都为 `0.383 → 0.242`。
- seed=3104 得到不同轨迹；reward 为 `0.617 → 0.672`。
- 相同 seed 的训练后 val EM 为 0.34375 与 0.375，说明微小 CUDA/FSDP 数值差异会放大到生成结果。
- 后续正式实验使用多个不同 seed 汇总均值/方差；相同 seed replica 不作为独立样本。

## 路线调整

原计划把 `state_masking`、format reward、Retriever 和模型规模消融提前放在 Stage03，同时把 NQ/PPO 放在最后。这与作者真实实验演进相反。

现调整为：

```text
04 Preliminary: NQ + PPO
05 v0.1: 多数据集 + PPO/GRPO
06 v0.2: masking + 1005 steps + 3B/7B/14B
07 v0.3: reward/backbone/retriever/data scaling
08 我们自己的鲁棒性扩展
```

Stage03 已按“可复现性门禁”收口。原 Stage03 的 format reward、Retriever 和规模消融移动到官方 v0.3 对应的 Stage07；多跳、噪声和故障注入移动到 Stage08。

## Stage04 训练过程

1. 校验官方 NQ 数据、Wiki-18 corpus、E5 index 和冻结 train512/val128 的 SHA；
2. 启动 21,015,324 文档的 GPU Retriever，完成 Hit@k 与延迟 preflight；
3. 在同一 val128 上建立 Qwen2.5-3B Base+Search 基线；
4. 从同一 Base 独立训练 PPO 1-step 与 5-step，启用 GAE、Critic、Reference、state masking、max_turns2/topk3；
5. 保存 Actor/Critic checkpoint，分别回载 Actor 并生成完整 128 条评测轨迹；
6. 强制检查有限指标、真实搜索、无 OOM/Traceback、双 checkpoint、相同验证 ID 和回载完整性。

## Stage04 验收结论

- Retriever PASS：Hit@1 52.34%，Hit@3 64.06%，平均/P95 延迟 3.0/11.8ms。
- PPO 1-step、5-step 均 PASS；峰值显存 43,973/44,145MiB，checkpoint step 分别耗时 173.3/169.3 秒。
- 三组评测均为同一 val128：Base EM 3.125%，1-step EM 3.125%，5-step EM 5.469%。
- SwanLab 已上传训练 1-step、5-step 与总验收三个 run。

## Stage04 实验结论

- 1-step 没有带来可测的 EM/F1 变化，它的价值是验证真实 PPO 与 checkpoint 闭环。
- 5-step 相对 Base：EM +2.344 个百分点，Token F1 从 5.98% 到 8.95%，搜索率从 82.81% 到 89.06%，平均搜索从 1.63 到 1.83。
- 这说明短程 PPO 已改变搜索行为，并在本次冻结样本上伴随指标改善；但训练 batch reward 并非单调，且单 seed/val128/5 updates 不足以证明稳定泛化或论文级复现。

## 下一验收点

Stage05 先在 NQ/HotpotQA 上建立 PPO/GRPO 对照，再决定是否扩展到七数据集和 305-step 配置。

详细结果位于 `experiment/results/`；大文件位于 `/data/cache/search-r1/`；训练指标同步到 SwanLab 项目 `Search-R1`。
