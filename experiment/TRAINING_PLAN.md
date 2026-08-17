# Search-R1 训练与学习方案

## 目标

以官方完整 Search-R1 代码为基座，按成本逐步增加的方式完成：

1. 理解搜索 Agent 的生成、工具调用和环境反馈闭环；
2. 分离基础模型、Retriever 和 RL 策略各自的贡献；
3. 在 8xL20 上跑通可回载、可评测的 Tiny GRPO；
4. 通过消融实验理解 Search-R1 为什么有效；
5. 最后再进行 NQ/HotpotQA 规模复现和 PPO 对照。

所有阶段必须独立可运行、可测量、可恢复。上一阶段未通过验收门槛时，不进入下一阶段。

## 当前基线（2026-08-17）

### 运行环境

- GPU：8 x NVIDIA L20，每张约 46 GB；
- 数据盘：`/data` 可用空间约 3.5 TB；
- 官方 Search-R1 基线：`598e61bd1d36895726d28a8d06b3a15bed19f5d3`；
- 实验分支：`experiment`；
- 已缓存模型：`/data/cache/search-r1/models/SearchR1-qwen2.5-3b-em-grpo`；
- Base 模型：`/data/cache/search-r1/models/Qwen2.5-3B`；
- 两个模型权重合计约 19 GB；
- 当前实验单测：5 passed。

### 已完成的 Stage 00 结果

参考命令：

```bash
source experiment/env.sh
bash experiment/runs/runsmoke.sh
```

现有 8 条合成问题结果：

| Mode | EM | Contains | F1 | Valid answer | Generated search | Retriever request | Hit@k | Avg turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-search | 0.0% | 0.0% | 3.1% | 100.0% | 100.0% | 0.0% | 0.0% | 1.00 |
| search | 62.5% | 100.0% | 83.3% | 100.0% | 100.0% | 100.0% | 100.0% | 1.00 |

模型在两种模式下都会生成搜索动作，但 no-search 没有调用 Retriever。该结果证明检索观察能够显著改善当前合成任务，但暂时不能直接证明 RL 策略本身贡献了多少。

## 总体路线

```text
00 inference smoke（已完成）
  -> 01 Base vs RL 四象限评测
  -> 02 Tiny GRPO 训练
  -> 03 Agent RL 消融
  -> 04 NQ/HotpotQA 规模复现
  -> 05 PPO 对照
```

| 阶段 | 核心问题 | GPU | 主要产物 |
| --- | --- | ---: | --- |
| 00 | 搜索闭环是否真实工作 | 1 | 轨迹、EM/F1、检索指标 |
| 01 | 检索与 RL 各贡献多少 | 1 | 四象限基线报告 |
| 02 | GRPO 训练链路是否闭环 | 8 | 可回载 Checkpoint |
| 03 | 哪些 Agent/RL 设计真正有效 | 8 | 消融矩阵和失败分析 |
| 04 | 是否能复现官方任务趋势 | 8 | NQ/HotpotQA 结果 |
| 05 | GRPO 与 PPO 的成本效果差异 | 8 | 算法对照报告 |

## Stage 00：补强推理 Smoke

### 已验证语义

- `generated_search_tag_rate`：模型是否生成 `<search>`；
- `retriever_request_rate`：环境是否真实调用 Retriever；
- `retrieval_hit_rate`：返回结果是否命中目标证据；
- `valid_answer_rate`：轨迹是否包含合法 `<answer>`；
- `avg_search_turns`：每条轨迹平均生成的搜索动作数。

本次运行中，两种模式的搜索标签生成率均为 100%；no-search 的 Retriever 请求率为 0%，search 为 100%。旧 `Search calls` 指标混合了模型动作和工具调用，现已拆分。

### 验收结果

- 5 个单测通过；
- 16 条 JSONL 轨迹完整；
- 能区分搜索动作与 Retriever 请求；
- search 模式 Hit@3 为 100%；
- 无 CUDA OOM、NaN 或损坏结果；
- 命令、日志、指标和报告均可恢复。

## Stage 01：Base vs RL 四象限评测

### 实验矩阵

使用相同问题、Retriever、Prompt、采样参数和答案评分器：

| Model | Retriever | 目的 |
| --- | --- | --- |
| Qwen2.5-3B Base | disabled | 测量模型原始知识 |
| Qwen2.5-3B Base | enabled | 测量单纯增加检索的收益 |
| Search-R1 GRPO | disabled | 测量 RL 对直接回答的影响 |
| Search-R1 GRPO | enabled | 测量完整 Search-R1 效果 |

8 条合成数据 pilot 已完成，下一步扩展到 50-200 条固定 QA。

### Pilot 结果（2026-08-17）

| Model | Retriever | EM | Contains | Valid | Request | Hit@3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B Base | disabled | 0.0% | 0.0% | 75.0% | 0.0% | 0.0% |
| Qwen2.5-3B Base | enabled | 12.5% | 62.5% | 75.0% | 75.0% | 83.3% |
| Search-R1 GRPO | disabled | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% |
| Search-R1 GRPO | enabled | 62.5% | 100.0% | 100.0% | 100.0% | 100.0% |

初步观察：

- 开启搜索时，GRPO 相对 Base 的 EM 高 50.0 个百分点、F1 高 53.9 个百分点；
- Base 只有 6/8 样本实际请求 Retriever，其中 5/6 命中目标证据；
- GRPO 的协议有效率、请求率和 Hit@3 都是 100%；
- Pilot 链路已完成，但 Base 有 2 条格式失败，且样本量只有 8 条，Stage 01 尚未最终验收。

### 必须保持不变的控制变量

- 问题集合和答案归一化；
- Retriever 索引与 `topk`；
- Prompt 模板；
- `temperature`、最大生成长度；
- 最大搜索轮数；
- EM/F1 计算逻辑；
- 运行硬件和批处理策略。

### 输出指标

- EM、Contains、F1；
- 搜索标签生成率；
- Retriever 实际调用率；
- Hit@1、Hit@3；
- 平均搜索轮数；
- 平均输入/输出 Token；
- 平均延迟；
- 格式正确率；
- 失败类型分布。

### 验收门槛

- 四组都能稳定生成完整轨迹；
- 每组至少拥有完全相同的有效样本数；
- 能定量分离 Retriever 收益和 RL 策略收益；
- 每个失败样本都能回看 Prompt、动作、观察和答案；
- 结果写入 `experiment/results/01-base-vs-rl/results.md`。

## Stage 02：Tiny GRPO

### 原则

不要直接修改官方 `train_grpo.sh`。新增独立入口并保留官方配置作为对照。

建议文件：

```text
experiment/runs/run02_tiny_grpo.sh
experiment/results/02-tiny-grpo/
├── README.md
├── config.md
├── metrics.json
├── results.md
└── failure-cases.md
```

大模型、Checkpoint、原始日志和轨迹继续放在：

```text
/data/cache/search-r1/experiments/02-tiny-grpo/
```

### 起始配置

| 参数 | 建议值 |
| --- | --- |
| Base model | Qwen2.5-3B Base |
| Train data | 64-128 QA |
| Validation data | 32 QA |
| GPU | 8 x L20 |
| train_batch_size | 32 |
| rollout.n_agent | 4 |
| ppo_mini_batch_size | 32 |
| ppo_micro_batch_size | 4 或 8 |
| max_turns | 2 |
| retriever.topk | 3 |
| max_response_length | 256 |
| max_obs_length | 256 |
| actor learning rate | 1e-6 |
| KL coefficient | 0.001 |
| total_training_steps | 先 1，再 5，最后 20 |
| save_freq | 10 |
| test_freq | 5 |

具体 batch 参数必须满足当前 veRL 的整除约束；如果配置检查失败，优先缩小 micro batch，不直接扩大训练规模。

### 分级启动

```text
preflight
  -> 1 optimizer step
  -> 5 optimizer steps
  -> 20 optimizer steps
  -> checkpoint reload
  -> pre/post evaluation
```

### 必须观察

- Reward 均值、方差和分量；
- KL、Policy Loss、Gradient Norm；
- Rollout 长度和搜索次数；
- 格式正确率；
- Retriever 请求与命中率；
- GPU 峰值显存；
- 每步耗时；
- Ray Worker 和 vLLM 健康状态。

### 验收门槛

- 数据加载、Rollout、Reward、Advantage 和优化器全部执行；
- Reward、KL 和 Loss 均为有限值；
- 没有 OOM、NaN、Ray 死锁；
- Checkpoint 能被 Stage 00/01 评测器重新加载；
- 训练前后使用同一冻结验证集对比；
- 至少保存一个成功轨迹和一个失败轨迹；
- 结果写入 `experiment/results/02-tiny-grpo/results.md`。

## Stage 03：Agent RL 消融

Tiny GRPO 通过后，每次只改变一个变量。

### 优先实验

1. Qwen Base vs Search-R1 GRPO；
2. Retriever disabled vs enabled；
3. `max_turns = 1 / 2 / 3`；
4. `topk = 1 / 3 / 5`；
5. `state_masking = true / false`；
6. `n_agent = 2 / 4 / 8`；
7. 只有答案奖励 vs 答案奖励加格式奖励；
8. 正常检索结果 vs 注入噪声结果。

### 要回答的问题

- RL 学到的是搜索格式，还是搜索策略？
- 模型是否会进行无意义的频繁搜索？
- Retriever 质量下降时，模型能否纠错？
- 哪些 Token 应进入策略梯度？
- 多轮搜索是否真的优于单轮搜索？
- Reward 改善来自答案正确，还是格式投机？
- 更大的 rollout group 是否带来更可靠的相对优势？

### 验收门槛

- 每个实验只有一个主要变量发生变化；
- 至少运行 3 个随机种子或明确标记单次试验；
- 保存配置、聚合指标和代表性轨迹；
- 结论包含收益、代价和失败边界；
- 结果写入 `experiment/results/03-ablations/`。

## Stage 04：NQ/HotpotQA 规模复现

仅在 Tiny GRPO 稳定后启动：

1. 处理 NQ/HotpotQA；
2. 构建或下载 Wikipedia Corpus 与索引；
3. 冻结验证集和 Retriever 版本；
4. 在 8xL20 上运行 Qwen2.5-3B GRPO；
5. 对比 Base、检索增强 Base、官方 Checkpoint 和自训练 Checkpoint；
6. 评测 EM/F1 与搜索行为。

不要直接照搬 H100 的时间估算。先根据 Tiny GRPO 实测：

- 每步耗时；
- 每条 Rollout 的 Token 数；
- Retriever P50/P95 延迟；
- GPU 峰值显存；
- Checkpoint 大小；
- 每 100 步的存储增长。

再估算官方 `1005 steps` 的时间和成本。

### 验收门槛

- 数据、索引、模型和代码版本全部固定；
- 全程没有不可恢复的训练中断；
- Checkpoint 可恢复训练并可独立评测；
- 与冻结基线使用相同评测脚本；
- 报告效果、吞吐、显存、耗时和失败样本。

## Stage 05：PPO 对照

GRPO 跑通后再运行 PPO，因为 PPO 额外引入 Critic，训练状态和显存成本更高。

重点比较：

- GRPO 是否更省显存；
- PPO Critic 是否让训练更稳定；
- 两者收敛速度；
- 搜索调用率、Hit@k 和最终 EM/F1；
- Checkpoint 大小和恢复成本；
- 相同 GPU 时间下的最终收益。

PPO 不作为第一条训练主线，只作为算法对照。

## 源码学习顺序

### 第一层：Agent 闭环

1. `search_r1/llm_agent/generation.py`
2. `search_r1/llm_agent/tensor_helper.py`
3. `search_r1/search/retrieval_server.py`
4. `search_r1/search/retrieval.py`

目标：能够解释一条轨迹如何完成：

```text
prompt
  -> <search>
  -> Retriever
  -> <information>
  -> continued reasoning
  -> <answer>
```

### 第二层：训练编排

1. `verl/trainer/main_ppo.py`
2. `verl/trainer/ppo/ray_trainer.py`
3. `verl/workers/`
4. Reward 与 Advantage 计算代码

目标：能够解释 Ray Worker、Actor、Reference、Critic、Rollout 和 Reward 如何协作。

### 第三层：GRPO 关键机制

重点理解：

- 同一问题为何生成多条轨迹；
- 相对奖励如何转换成 Advantage；
- KL 约束如何影响更新；
- `state_masking` 为什么决定哪些 Token 参与训练；
- 搜索观察是否应进入策略梯度；
- 稀疏答案奖励如何影响前序搜索动作。

## 实验记录规范

每个实验目录至少包含：

```text
README.md          # 问题、假设和运行方式
config.md          # 完整参数与代码/模型/数据版本
metrics.json       # 机器可读指标
results.md         # 结果、结论和下一步
failure-cases.md   # 代表性失败轨迹
```

每次运行必须记录：

- Git commit 和 branch；
- 模型 ID 或本地路径；
- 数据与索引版本；
- 完整命令；
- GPU 数量和显存；
- 开始/结束时间；
- 日志与 Checkpoint 路径；
- 训练和验证指标；
- 是否通过验收门槛。

Git 只保存源码、小型固定数据、配置和总结。模型、索引、原始日志、完整轨迹和 Checkpoint 保存在 `/data/cache/search-r1/`。

## 推荐节奏

### Day 1

- 阅读 Agent 闭环代码；
- 重跑 Stage 00；
- 拆分搜索标签生成率和 Retriever 请求率；
- 保存一条完整成功轨迹和失败轨迹。

### Day 2

- 完成 Base vs RL 四象限；
- 输出失败类型；
- 确认 Retriever 和 RL 各自贡献。

### Day 3-4

- 建立隔离的 veRL/vLLM 环境；
- 依次完成 1、5、20 个 Tiny GRPO steps；
- 验证 Checkpoint 回载。

### Day 5 以后

- 执行 `max_turns/topk/state_masking/reward` 消融；
- 根据实测吞吐决定是否启动 NQ/HotpotQA 全量复现；
- GRPO 稳定后再加入 PPO 对照。

## 下一步

Stage 01 pilot 已完成，当前主任务是扩大评测规模：

1. 准备 50-200 条固定 QA 与对应证据语料；
2. 重跑四象限并检查格式失败、请求率和 Hit@1/Hit@3；
3. 对失败样本做协议、检索和答案三层归因；
4. Stage 01 通过验收后再开始 Tiny GRPO。
