# Experiment 04: Official Preliminary NQ + PPO

使用官方 NQ 数据格式、Wiki-18/E5 Retriever、Search-R1 Agent loop、EM reward 和 veRL GAE/PPO，在 8 x L20 上完成短程闭环。

## 训练过程

- 资产：官方 NQ 79,168 train / 3,610 test，冻结 train512 / val128；Wiki-18 共 21,015,324 篇，E5-base-v2 + 64.6GB FAISS index。
- 基线：Qwen2.5-3B Base 在冻结 val128 上启用 Search，max_turns=2、topk=3。
- PPO：1-step 和 5-step 都从同一 Base 独立启动；batch64，Actor lr=1e-6，Critic lr=1e-5，GAE，state_masking=true。
- 验证：每次训练都保存 Actor/Critic；再从 Actor checkpoint 独立回载，对同一 val128 生成 128 条轨迹。

## 同一冻结 NQ 验证集

| Model | EM | Token F1 | Search rate | Avg searches | Valid answer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 3.1% | 6.0% | 82.8% | 1.63 | 95.3% |
| PPO 1-step reload | 3.1% | 6.0% | 82.8% | 1.63 | 95.3% |
| PPO 5-step reload | 5.5% | 9.0% | 89.1% | 1.83 | 93.8% |

## Retriever

- Hit@1: 52.3%
- Hit@3: 64.1%
- Mean latency/query: 0.003s
- P95 latency/query: 0.012s

## PPO 训练

| Updates | Reward | PPO KL | Value loss | Step time | Peak GPU memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0470 | 0.000000 | 2.081000 | 173.3s | 43973 MiB |
| 5 | 0.0620 | -0.000000 | 1.245000 | 169.3s | 44145 MiB |

## 验收结论

- Status: PASS
- [x] official_nq_validation_128
- [x] same_validation_ids
- [x] retriever_preflight
- [x] ppo_1step
- [x] ppo_5step
- [x] checkpoint_reload
- PASS 表示官方真实数据检索、Actor/Critic/Reference/vLLM Rollout、梯度更新、双 checkpoint 保存和独立回载评测闭环全部成立。

## 实验结论

- 1-step 的 EM/F1 与 Base 相同（3.1%/6.0%），作用是验证参数更新与 checkpoint 闭环，不能证明效果提升。
- 5-step 相对 Base：EM 3.1% → 5.5%，Token F1 6.0% → 9.0%，搜索率 82.8% → 89.1%。
- 5-step 的平均搜索次数从 1.63 升到 1.83，短程 PPO 已改变工具调用行为；但 128 条单次评测不足以判断稳定泛化。

## 结论边界

- PASS 表示 Actor、Critic、Reference、vLLM Rollout、检索、梯度更新、双 checkpoint 保存和回载评测闭环成立。
- 5 个更新步只用于 Preliminary 工程验收；指标变化不代表完整 NQ 收敛或论文效果复现。
- Hit@k 是冻结 NQ 原问题直接检索的词面覆盖率，和模型生成 query 后的端到端 EM/F1 分开解释。
