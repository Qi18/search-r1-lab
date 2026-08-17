# Stage 03: Agent RL 消融

目标：解释 Stage 02 的提升来自哪里，而不是继续堆训练步数。

## 执行顺序

1. **03-0 可复现性门禁**：同 seed 重复 2-step，异 seed 运行 2-step；比较检索轨迹代理哈希和训练指标。
2. **03-A 行为探针**：Base/GRPO 的 search on/off、topk 变化，仅评测不训练。
3. **03-B 核心训练消融**：Base、`state_masking=false`、`state_masking=true`、格式奖励；每项三个 seed。
4. **03-C 多跳实验**：换多跳数据后比较 `max_turns=1/2/3`，当前单跳固定语料不用于论证多轮搜索价值。

固定项：Qwen2.5-3B、train64/val32/corpus96、GRPO、batch32、`n_agent=4`、topk3、学习率 1e-6。

结果写入本目录；运行产物写入 `/data/cache/search-r1/experiments/03-ablations/`，不提交模型权重。
