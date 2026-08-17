# Search-R1 L20 Experiment

A small, reproducible laboratory for understanding and reproducing Search-R1.
The repository follows NanoChat's experiment style: one readable shell entry
point, explicit stage ordering, unbuffered logs, cache isolation, and a checked
in result report.

The first experiment asks one concrete question:

> Does an official Search-R1 GRPO checkpoint answer synthetic facts more
> accurately when its search tool is enabled?

## Quick start

The default L20 setup expects:

- 1 CUDA GPU for the language model; the small E5 retriever runs on CPU
- the official Search-R1 Qwen2.5-3B GRPO checkpoint
- Python 3.10 with PyTorch, Transformers, NumPy, and FAISS

```bash
source experiment/env.sh
bash experiment/runs/runsmoke.sh
```

Override paths without editing scripts:

```bash
SEARCH_R1_MODEL_PATH=/path/to/model \
SEARCH_R1_LAB_CACHE=/path/to/cache \
bash experiment/runs/runsmoke.sh
```

## Reference pipeline

```text
environment check
  -> dense index build
  -> no-search baseline
  -> Search-R1 multi-turn search
  -> EM/F1/tool metrics
  -> experiment/learning/experiments/00-search-smoke/results.md
```

Large models, indexes, raw logs, and generated trajectories live under
`SEARCH_R1_LAB_CACHE`. Git tracks only source, small synthetic fixtures, plans,
and compact result summaries.

## 目录与文件职责

| 路径 | 作用 | 主要输入 | 主要输出或修改时机 |
| --- | --- | --- | --- |
| `experiment/README.md` | 实验目录总入口，说明如何启动、各模块边界和阅读顺序 | 当前可运行入口与目录结构 | 新增实验阶段或目录时更新 |
| `experiment/TRAINING_PLAN.md` | 从推理 Smoke、四象限评测到 Tiny GRPO、消融和完整复现的训练路线 | 当前硬件、模型、数据和阶段结果 | 阶段目标、参数或验收标准变化时更新 |
| `experiment/FINAL_REPORT.md` | 保存最近一次已经验证的项目状态和结论 | 已完成实验的日志与指标 | 一个阶段完成并得到稳定结论后更新 |
| `experiment/env.sh` | 统一模型、缓存、Hugging Face、仓库和 Python 模块路径 | L20 本地目录与可选环境变量 | 环境路径变化时更新；不得写入 Token 或私钥 |
| `experiment/data/` | 保存可提交 Git 的小型确定性语料和 QA Fixture | 手工构造或裁剪的小数据集 | 增加 Smoke、回归或边界样本时更新 |
| `experiment/runs/` | 实验编排入口，按固定顺序串联检查、索引、推理、评测和报告 | 环境变量、数据、模型和脚本 | 新增一个可复现实验时新增一个入口 |
| `experiment/scripts/` | 可独立执行的数据处理、索引、评测和汇总程序 | JSONL、模型、索引、轨迹 | 某个实验步骤需要复用或单独调试时修改 |
| `experiment/search_r1_lab/` | 实验专用 Python 逻辑，封装协议、Retriever、Agent 循环、指标和 I/O | Prompt、模型输出、检索索引 | 修改实验行为或增加可测试能力时更新 |
| `experiment/tests/` | 固定协议解析和指标行为，防止实验重构改变统计口径 | 小型确定性输入 | 修改协议或指标前先补测试 |
| `experiment/learning/` | 源码学习记录、实验设计和小型结论归档 | 代码阅读与实验结果 | 每次形成可复用认识后更新 |
| `experiment/learning/experiments/` | 按编号保存每个实验的结果、配置和失败样本 | 单次实验产物 | 每个阶段建立独立目录，不能覆盖其他实验 |

## 关键模块职责

### `experiment/runs/runsmoke.sh`

当前参考实验的唯一编排入口，负责：

1. 加载 `experiment/env.sh`；
2. 检查 CUDA、PyTorch、Transformers 和 FAISS；
3. 运行协议与指标测试；
4. 构建或复用 E5 + FAISS 索引；
5. 执行 no-search 与 search 两组推理；
6. 汇总指标并更新结果报告。

它只负责编排，不实现 Agent 或指标算法。新增正式实验时应创建新的
`run*.sh`，不要不断向 Smoke 入口叠加逻辑。

### `experiment/scripts/build_index.py`

读取 `experiment/data/corpus.jsonl`，使用 E5 模型生成向量并建立
FAISS 索引，同时输出与向量顺序对应的 metadata。它回答的是“如何把
实验语料变成可检索环境”。

### `experiment/scripts/run_eval.py`

加载模型、Retriever 和 QA 数据，分别执行 no-search/search 轨迹，
并将每个问题的动作、观察、答案、延迟和命中情况写入 JSONL。它负责
评测运行，不负责聚合最终指标。

### `experiment/scripts/summarize.py`

读取轨迹 JSONL，调用统一指标函数生成 `metrics.json` 和 Markdown
报告。任何新指标都应先在 `metrics.py` 中实现和测试，再由这里展示。

### `experiment/search_r1_lab/agent.py`

实验版 Search-R1 Agent 控制器。负责模型生成、识别搜索动作、调用
Retriever、追加观察、控制最大搜索轮数并抽取最终答案。这里决定一条
Agent 轨迹如何推进。

### `experiment/search_r1_lab/protocol.py`

定义并解析 `<search>`、`<information>` 和 `<answer>` 协议，
包括截断到第一个有效动作、提取搜索 Query 和最终答案。修改标签格式
会影响所有轨迹，必须同步更新测试。

### `experiment/search_r1_lab/retrieval.py`

封装 E5 Embedding 与 FAISS 检索，负责构建索引、加载索引、执行
Top-k 查询并返回文档。这里衡量的是环境检索能力，不包含 RL 更新。

### `experiment/search_r1_lab/metrics.py`

实现答案归一化、EM、Contains、Token F1、格式正确率、搜索调用率和
Hit@k 等统计。该模块定义实验的统一评价口径，不能在不同脚本中复制
另一套实现。

### `experiment/search_r1_lab/io.py`

集中处理 JSONL 读写，保证轨迹与小型数据文件的格式一致，避免业务
逻辑重复处理序列化细节。

### `experiment/tests/`

- `test_protocol.py`：保护搜索和答案标签解析行为；
- `test_metrics.py`：保护答案归一化与聚合指标行为。

测试失败时不能继续生成新实验结论。

## 数据与产物边界

Git 中只保存：

- 实验源码与 Shell 入口；
- 小型确定性数据；
- 配置和训练计划；
- 聚合指标和代表性失败案例；
- 可以代码审查的结果报告。

以下内容不提交 Git：

- 模型权重和 Checkpoint；
- Wikipedia 等大型语料；
- FAISS 大型索引；
- 原始完整日志；
- 大规模 Rollout 轨迹；
- Token、密钥和账号配置。

这些大型产物统一放在 `/data/cache/search-r1/` 或
`SEARCH_R1_LAB_CACHE` 指定的位置。

## 修改边界

- 官方源码仍位于仓库根目录，实验代码不得复制或改写官方模块；
- 通用实验逻辑放入 `experiment/search_r1_lab/`，一次性编排放入 `experiment/runs/`；
- 一个实验对应一个独立的 `experiment/learning/experiments/<编号-名称>/`；
- 指标变化必须先补测试，再重新生成历史可比结果；
- 实验报告必须记录 Git commit、模型、数据、命令、GPU、日志和结论。

## 建议阅读顺序

```text
experiment/README.md
  -> experiment/TRAINING_PLAN.md
  -> experiment/runs/runsmoke.sh
  -> experiment/search_r1_lab/agent.py
  -> experiment/search_r1_lab/protocol.py
  -> experiment/search_r1_lab/retrieval.py
  -> experiment/search_r1_lab/metrics.py
  -> experiment/learning/experiments/
```

This is not yet a full paper reproduction. The full NQ/HotpotQA + Wikipedia
GRPO run is deliberately gated on the small inference and training smoke tests.

## References

- Search-R1 paper: https://arxiv.org/abs/2503.09516
- Official implementation: https://github.com/PeterGriffinJin/Search-R1
- Official checkpoints: https://huggingface.co/collections/PeterJinGo/search-r1
