# Search-R1 实验路线

## 目标

以官方完整 Search-R1 为基座，先用低成本实验验证环境、评测、训练和可复现性，再严格按照作者的实验演进复现 Preliminary、v0.1、v0.2、v0.3。

原则：官方复现与自定义扩展分开；上一阶段未通过验收，不扩大数据、模型或训练步数。

## 当前状态

| Stage | 状态 | 作用 | 类型 |
| --- | --- | --- | --- |
| 00 | PASS | 搜索 Agent 链路冒烟 | 我们的前置门禁 |
| 01 | PASS | Base/GRPO × Search on/off 四象限 | 我们的前置门禁 |
| 02 | PASS | 1/5/20-step Tiny GRPO 与 checkpoint 回载 | 我们的前置门禁 |
| 03 | PASS | seed 传递与可复现性边界 | 我们的前置门禁 |
| 04 | NEXT | NQ + PPO 小步复现 | 官方 Preliminary |
| 05 | PENDING | 多数据集 PPO/GRPO | 官方 v0.1 |
| 06 | PENDING | masking、长训练、模型规模 | 官方 v0.2 |
| 07 | PENDING | reward/backbone/retriever/data 消融 | 官方 v0.3 |
| 08 | PENDING | 噪声、故障和新方法 | 我们的扩展 |

## 总体路线

```text
00 搜索闭环
  -> 01 四象限归因
  -> 02 Tiny GRPO
  -> 03 可复现性门禁
  -> 04 Preliminary: NQ + PPO
  -> 05 v0.1: 多数据集 + PPO/GRPO
  -> 06 v0.2: masking + 1005 steps + 3B/7B/14B
  -> 07 v0.3: reward/backbone/retriever/data scaling
  -> 08 自定义 Agent RL 扩展
```

## Stage 00：搜索闭环

验证 `<search> → Retriever → <information> → <answer>` 真实工作，并区分模型生成搜索标签与环境实际调用工具。

- 数据：8 条固定合成 QA。
- 结果：search EM 62.5%，no-search EM 0%；Hit@3 100%。
- 产物：`results/00-search-smoke/`。

## Stage 01：四象限归因

比较 Qwen2.5-3B Base / 官方 GRPO checkpoint 与 Search enabled / disabled，分离基础模型、Retriever 和 RL 行为的贡献。

- 数据：64 条固定合成 QA，四组共 256 条轨迹。
- 结果：Base+Search EM 4.7%，GRPO+Search EM 78.1%。
- 边界：Retriever 对已发出的查询都能命中，因此主要测协议执行和证据利用，不代表真实 QA 泛化。
- 产物：`results/01-base-vs-rl/`。

## Stage 02：Tiny GRPO

从 Qwen2.5-3B Base 出发，跑通官方 veRL/FSDP/vLLM/GRPO 主链路。

- 数据：train64 / val32 / corpus96。
- 训练：1、5、20 optimizer steps；batch32；`n_agent=4`；`max_turns=2`；topk3。
- 验收：checkpoint 可保存、回载、评测；Reward/KL/Loss 有限；无 OOM、NaN、Ray 死锁。
- 20-step 结果：冻结验证集 Search EM 从 6.25% 提升到 28.125%。
- 产物：`results/02-tiny-grpo/`，大文件位于 `/data/cache/search-r1/experiments/02-tiny-grpo/`。

## Stage 03：可复现性门禁

Stage03 不再提前承载官方 v0.3 消融，只负责证明后续实验中的 seed 是有效变量。

- 同 seed=3103 两次 2-step：训练期检索轨迹哈希一致，reward 都为 `0.383 → 0.242`。
- 异 seed=3104：轨迹哈希变化，reward 为 `0.617 → 0.672`。
- 边界：FSDP/CUDA 更新不是位级确定，同 seed 的训练后 val EM 为 0.34375 / 0.375。
- 结论：正式实验使用不同 seed 统计均值/方差；同 seed replica 只做复现性诊断。
- 产物：`results/03-ablations/`。

## Stage 04：官方 Preliminary——NQ + PPO

对应作者最早的实验：只在 Natural Questions 上用 PPO 做少量训练，先确认真实 Wikipedia 检索下能学出搜索行为。

### 执行

1. 下载官方 `nq_hotpotqa_train` 数据、Wiki-18 corpus 和 E5 index；记录版本与 SHA-256。
2. 启动官方 Retriever Server，验收 NQ query 的 Hit@k 和延迟。
3. 固定 NQ 验证集，跑 Qwen Base + Search 基线。
4. 用 3B Base 跑 1-step、短程 PPO，再回载评测。
5. 根据实测显存和每步耗时决定是否继续，不直接运行长任务。

### 验收

- 使用官方数据格式、Agent loop、Reward 和评测脚本；
- PPO 的 Actor、Critic、Reference、Rollout 全部执行；
- 训练前后使用同一 NQ 验证集；
- 报告 EM/F1、搜索率、Hit@k、KL、显存和每步耗时。

## Stage 05：官方 v0.1——多数据集 PPO/GRPO

复现作者从 NQ 扩展到多数据集，并对比 PPO 与 GRPO 的阶段。

官方脚本基准：

- `total_training_steps=305`
- learning rate `1e-6`
- `n_agent=5`
- `max_turns=4`
- Retriever topk3
- `state_masking=true`

先在 NQ/HotpotQA 路线上对齐 PPO/GRPO，再扩展到论文中的七数据集评测。比较效果、稳定性、显存、吞吐和 checkpoint 成本。

## Stage 06：官方 v0.2——稳定性与规模

复现作者修复 retrieved-token masking 和 GRPO sample indexing 后的长训练路线。

- 先做 `state_masking=true/false` 回归，证明 observation token 不进入 policy loss 的影响；
- 目标配置：`total_training_steps=1005`、lr `1e-6`、`n_agent=5`、`max_turns=4`、topk3；
- 先完成 3B；7B/14B 只在 3B 时间、显存和恢复机制验收后启动；
- 比较短训练与长训练的搜索率、EM/F1、KL 和训练稳定性。

## Stage 07：官方 v0.3——系统消融

把原 Stage03 中过早规划的消融移动到这里，与作者第二篇论文保持一致。

### Reward design

- 只有答案 reward；
- 答案 + format reward；
- 官方格式配置：`structure_format_score=0.2`、`final_format_score=0.1`、`retrieval_score=0`；
- v0.3 GRPO 学习率 `5e-7`。

### LLM backbone

- 通用 Base vs reasoning model；
- 3B / 7B / 14B，资源允许后再考虑 32B。

### Search engine

- 不同 Retriever 的训练动态；
- 训练时与推理时 Retriever 不一致时的泛化。

### Data scaling

- 固定其余配置，改变训练数据量；
- 至少 3 个不同 seed，报告均值、方差、成本和代表性轨迹。

## Stage 08：自定义扩展

只有官方主线复现后才进入：

- 噪声文档、无答案和错误证据；
- Retriever 超时、空结果和服务失败；
- 多跳任务中的 `max_turns=1/2/3`；
- topk、reranker、在线搜索与本地搜索切换；
- 新 reward、路由或轨迹优化方法。

这些结果必须标记为“我们的扩展”，不与官方复现混写。

## 立即执行顺序

```text
Stage04-0 真实数据/索引 preflight
  -> Stage04-1 NQ Base + Search 冻结基线
  -> Stage04-2 NQ PPO 1-step
  -> Stage04-3 NQ PPO 短程训练与回载
```

所有大模型、索引、checkpoint、原始日志和完整轨迹写入 `/data/cache/search-r1/`；Git 只保存脚本、配置、manifest、指标和结论。
