# Experiment 02: Tiny GRPO

在 8 x NVIDIA L20 上，从 Qwen2.5-3B Base 使用 Search-R1 官方 veRL/GRPO 代码完成 1、5、20 个优化步。

## 冻结验证集结果

| Model | Mode | EM | Contains | F1 | Valid | Request | Hit@1 | Hit@3 | Turns | Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base (pre) | no-search | 0.0% | 0.0% | 4.0% | 81.2% | 0.0% | 0.0% | 0.0% | 0.75 | 1.70s |
| Base (pre) | search | 6.2% | 68.8% | 24.8% | 81.2% | 75.0% | 95.8% | 95.8% | 0.75 | 2.45s |
| Tiny GRPO (20-step) | no-search | 0.0% | 0.0% | 4.2% | 100.0% | 0.0% | 0.0% | 0.0% | 1.00 | 0.56s |
| Tiny GRPO (20-step) | search | 28.1% | 87.5% | 49.7% | 100.0% | 100.0% | 100.0% | 100.0% | 1.00 | 0.57s |

## Search 模式训练前后变化

- exact_match: +21.9 pp
- answer_contains: +18.8 pp
- token_f1: +24.9 pp
- retriever_request_rate: +25.0 pp

## 训练运行验收

| Updates | Status | Last reward | KL | PG loss |
| ---: | --- | ---: | ---: | ---: |
| 1 | PASS | 0.164 | -0.001 | 0.103 |
| 5 | PASS | 0.281 | -0.003 | 0.015 |
| 20 | PASS | 0.992 | -0.002 | -0.032 |

## SwanLab

- [1-step backfill](https://swanlab.cn/@richliu0153/Search-R1/runs/gn2jgh4o)
- [5-step backfill](https://swanlab.cn/@richliu0153/Search-R1/runs/op6kjbwj)
- [20-step backfill](https://swanlab.cn/@richliu0153/Search-R1/runs/w7da3cox)

这些 Run 从 console 日志按原始 step 回填，不包含训练时未采集的实时 GPU/系统遥测。

## 结论

- PASS 表示官方 GRPO 数据流、工具调用、梯度更新、checkpoint 保存和回载评测全部闭环；不以指标必须提升作为工程验收条件。
- 训练前后差异只适用于这 32 条训练集外合成问题；20 步单随机种子不能代表真实数据泛化结论。
- Retriever preflight 单独验证索引质量，最终指标还同时受模型是否调用搜索、是否利用证据和答案格式影响。

## 最终验收

- Status: PASS
- [x] 32_examples_per_quadrant
- [x] identical_question_ids
- [x] required_fields
- [x] finite_metrics
- [x] no_search_has_no_requests
- [x] retriever_preflight
- [x] all_training_runs_pass
- [x] checkpoint_reloaded

## 结果类型

| Model | Mode | Exact | Contains only | Incorrect | Invalid format | Retrieval miss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base (pre) | no-search | 0 | 0 | 26 | 6 | 0 |
| Base (pre) | search | 2 | 20 | 3 | 6 | 1 |
| Tiny GRPO (20-step) | no-search | 0 | 0 | 32 | 0 | 0 |
| Tiny GRPO (20-step) | search | 9 | 19 | 4 | 0 | 0 |
