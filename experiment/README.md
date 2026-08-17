# Search-R1 L20 Experiment

基于官方 Search-R1 的独立实验区，用于学习搜索 Agent、复现评测并逐步开展 GRPO 训练。官方源码保留在仓库根目录，所有自定义内容都在 `experiment/`。

## 快速开始

```bash
source experiment/env.sh
bash experiment/runs/runsmoke.sh
```

默认使用 1 张 GPU 运行 Search-R1 Qwen2.5-3B，E5 Retriever 在 CPU 上运行。

## 目录作用

| 路径 | 作用 |
| --- | --- |
| `README.md` | 实验入口和目录导航 |
| `TRAINING_PLAN.md` | Smoke、四象限评测、Tiny GRPO 和完整复现路线 |
| `FINAL_REPORT.md` | 最近一次已验证的实验结论 |
| `env.sh` | 统一模型、缓存和 Python 路径 |
| `data/` | 可提交 Git 的小型语料和 QA 数据 |
| `runs/` | 一键运行实验的 Shell 入口 |
| `scripts/` | 建索引、执行评测和汇总指标 |
| `search_r1_lab/` | Agent、协议、Retriever、指标和 I/O 实现 |
| `tests/` | 协议与指标回归测试 |
| `results/` | 分阶段实验结果摘要 |

## 关键模块

| 文件 | 作用 |
| --- | --- |
| `runs/runsmoke.sh` | 串联环境检查、索引、推理、评测和报告 |
| `scripts/build_index.py` | 使用 E5 + FAISS 构建检索索引 |
| `scripts/run_eval.py` | 运行 no-search/search 轨迹 |
| `scripts/summarize.py` | 生成 EM、F1、Hit@k 等指标 |
| `search_r1_lab/agent.py` | 控制搜索、观察和回答的 Agent 循环 |
| `search_r1_lab/protocol.py` | 解析 `<search>` 和 `<answer>` 协议 |
| `search_r1_lab/retrieval.py` | 构建和查询向量索引 |
| `search_r1_lab/metrics.py` | 统一答案与搜索指标 |
| `search_r1_lab/io.py` | 读写 JSONL 数据和轨迹 |

## 产物边界

Git 只保存源码、小型数据、配置和结果摘要。模型、Checkpoint、大型索引、完整日志和 Rollout 轨迹保存在 `/data/cache/search-r1/` 或 `SEARCH_R1_LAB_CACHE`。

## 阅读顺序

```text
README.md
  -> TRAINING_PLAN.md
  -> runs/runsmoke.sh
  -> search_r1_lab/agent.py
  -> search_r1_lab/protocol.py
  -> search_r1_lab/retrieval.py
  -> search_r1_lab/metrics.py
  -> results/
```

## References

- Search-R1 paper: https://arxiv.org/abs/2503.09516
- Official implementation: https://github.com/PeterGriffinJin/Search-R1
- Official checkpoints: https://huggingface.co/collections/PeterJinGo/search-r1
