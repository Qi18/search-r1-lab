# Search-R1 当前实验报告

更新时间：2026-08-18

## 当前结论

Stage00–04 已完成，Stage05 官方 v0.1 的多数据集 PPO/GRPO 短程门禁已 PASS。项目已打通 NQ + HotpotQA 训练格式、七数据集评测、Wiki-18/E5 Retriever、PPO/GRPO、checkpoint 保存与独立回载。

Stage05 的 1-step 结果只证明工程闭环；下一步执行 305-step 长程路线，观察损耗、reward、搜索行为和七数据集指标趋势。

## 已完成阶段

| Stage | 关键结果 | 结论边界 |
| --- | --- | --- |
| 00 | search EM 62.5%，no-search EM 0%，Hit@3 100% | 只有 8 条合成问题 |
| 01 | Base+Search EM 4.7%，官方 GRPO+Search EM 78.1% | 64 条合成事实，Retriever 条件命中率 100% |
| 02 | 20-step 后冻结 val Search EM 28.125%，Base 为 6.25% | train64/val32，不能外推真实 QA |
| 03 | 同 seed 训练轨迹哈希一致，异 seed 哈希变化 | GPU 更新不保证位级确定 |
| 04 | Base/1-step/5-step EM 3.125%/3.125%/5.469%，5-step F1 8.95% | 官方 NQ + Wiki-18，但仅 5 updates / val128 |
| 05 short gate | Base/PPO/GRPO EM 均 6.25%；PPO/GRPO 训练与七集回载 PASS | 每集 16 条、仅 1 update，不代表收敛 |

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

## Stage05 短程训练过程

1. 从官方 `PeterJinGo/nq_hotpotqa_train` 冻结 train512：NQ/HotpotQA 各 256；
2. 从官方 test 冻结 val112：NQ、HotpotQA、TriviaQA、PopQA、2WikiMultiHopQA、MuSiQue、Bamboogle 各 16；
3. 使用 21,015,324 文档的 Wiki-18、E5-base-v2、topk3 和 `max_turns=4`；
4. PPO 按官方 v0.1 使用 GAE + Critic + `n_agent=1`；GRPO 使用组内优势 + `n_agent=5`；两者都启用 state masking；
5. 分别完成 1-step、checkpoint 保存和同一 val112 的独立回载；
6. 检查有限指标、真实搜索、state mask、无 OOM、checkpoint 和七数据集 ID 一致性。

## Stage05 短程验收结论

- Retriever PASS：Hit@1 28.57%，Hit@3 42.86%，平均延迟约 3.5ms/query。
- PPO PASS：policy loss 0.099、value loss 2.143、KL 0.004、reward 0.062、峰值显存 37,469MiB。
- GRPO PASS：policy loss 0.063、KL loss 0.000、reward 0.138、峰值显存 40,334MiB；GRPO 不训练 Critic。
- Base、PPO 1-step、GRPO 1-step 在 val112 上 EM 均为 6.25%、F1 均为 12.3%、搜索率均为 85.7%。
- 三组相同说明 1-step 权重变化尚不足以改变确定性验证轨迹；它不是“算法没有效果”的证据。

## Stage05 工程结论

- 官方 v0.1 的多数据集数据契约、两种 RL 算法、四轮 Agent loop、真实检索、state masking 和 checkpoint 回载已经成立。
- GRPO 每个问题采样 5 条轨迹，本次单步处理约 27,619 token，PPO 约 5,439 token；因此 GRPO 200.7 秒与 PPO 168.7 秒不能只按算法公式解释，二者采样量不同。
- PPO 的 value loss 2.143 与极低 explained variance 只反映随机初始化 Critic 的首步状态；需要多步曲线判断 Critic 是否校准。
- GRPO 首步 KL loss 为 0 符合从 Reference 同权重启动的预期；后续更新后才应出现偏离。

## 下一验收点

执行官方 v0.1 的 305-step PPO/GRPO 长程路线。受 8 x L20 48GB 约束，batch 从官方 512 缩到 32，其他关键语义保持为 lr 1e-6、warmup ratio 0.95、max_turns4、topk3、state masking、PPO n_agent1、GRPO n_agent5。每 100 步保存 checkpoint；当前 veRL PPO 实现没有优化器级 resume，checkpoint 可回载评测，但不能宣称精确断点续训。

详细结果位于 `experiment/results/`；大文件位于 `/data/cache/search-r1/`；训练指标同步到 SwanLab 项目 `Search-R1`。
