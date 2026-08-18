# Experiment 05: Official v0.1 Multi-dataset PPO/GRPO

在官方 NQ + HotpotQA 训练格式和七数据集验证格式上，完成 PPO/GRPO 短程训练、checkpoint 与独立回载评测闭环。

## 总体结果

| Model | EM | Token F1 | Search rate | Avg searches | Valid answer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 6.2% | 12.3% | 85.7% | 2.77 | 99.1% |
| PPO 1-step | 6.2% | 12.3% | 85.7% | 2.77 | 99.1% |
| GRPO 1-step | 6.2% | 12.3% | 85.7% | 2.77 | 99.1% |

## 七数据集 EM

| Dataset | Base | PPO 1-step | GRPO 1-step |
| --- | ---: | ---: | ---: |
| nq | 0.0% | 0.0% | 0.0% |
| hotpotqa | 6.2% | 6.2% | 6.2% |
| triviaqa | 18.8% | 18.8% | 18.8% |
| popqa | 6.2% | 6.2% | 6.2% |
| 2wikimultihopqa | 0.0% | 0.0% | 0.0% |
| musique | 0.0% | 0.0% | 0.0% |
| bamboogle | 12.5% | 12.5% | 12.5% |

## 训练验收

- PPO: PASS，reward=0.0620，step=168.7s，peak=37469 MiB。
- GRPO: PASS，reward=0.1380，step=200.7s，peak=40334 MiB。

## 验收结论

- Status: PASS
- [x] seven_dataset_validation
- [x] same_validation_ids
- [x] ppo_training
- [x] grpo_training
- [x] checkpoint_reload

## 结论边界

- 1-step 结果用于验证官方 v0.1 多数据集 PPO/GRPO 工程闭环，不代表 305-step 收敛结果或论文效果复现。
- PPO 按官方脚本使用 n_agent=1；GRPO 使用 n_agent=5。二者采样成本不同，step 时间不能直接解释为算法本体开销。
- 每个验证集只冻结 16 条，分数据集指标方差很大，仅用于接口、奖励与回载一致性检查。
