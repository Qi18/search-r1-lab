# Stage 03-0：随机种子与可复现性门禁

状态：**PASS**（2026-08-17，L20 × 8）

## 配置

- 基座：Qwen2.5-3B，官方 Search-R1 FSDP + vLLM + GRPO 路径
- 数据：Stage02 固定 train64 / val32 / corpus96
- 每次 2 个更新，batch32，`n_agent=4`，`max_turns=2`，topk3
- 不保存 checkpoint；保留 console 日志、检索请求 JSONL 和验收 JSON
- seed 同时控制 controller 的 Python/NumPy/PyTorch、DataLoader shuffle、各 rank RNG、vLLM engine 和 rollout CUDA RNG

## 结果

| 运行 | seed | step1 reward | step2 reward | 最终 val EM | 训练检索轨迹哈希 |
| --- | ---: | ---: | ---: | ---: | --- |
| same-a | 3103 | 0.383 | 0.242 | 0.34375 | `3b1afccc...3578e` |
| same-b | 3103 | 0.383 | 0.242 | 0.37500 | `3b1afccc...3578e` |
| different | 3104 | 0.617 | 0.672 | 0.78125 | `f82d8515...5ecbc` |

验收条件及结果：

- 同 seed 的训练期检索 query + topk + result IDs 哈希一致：PASS。
- 异 seed 的训练期轨迹哈希发生变化：PASS。
- 同 seed 的全部训练指标和训练后验证轨迹位级一致：FAIL（非硬门禁）。

## 结论

1. Stage03 的 seed 已真正进入数据顺序和生成采样，不是空配置；3103 与 3104 得到明显不同的样本顺序、轨迹和 reward。
2. 相同 seed 可以复现前两个训练 step 的检索行为以及主要 reward/行为指标。
3. FSDP/CUDA 更新仍存在轻微数值非确定性：第二步 `actor/kl_loss`、`grad_norm` 有末位差异，并放大为训练后验证轨迹差异，val EM 为 0.34375 与 0.375。
4. 因此正式消融必须使用多个不同 seed 汇总均值/方差；同 seed replica 只用于复现性诊断，不能作为独立样本。
5. seed=3104 的 0.78125 只是 2-step、32 条合成验证集上的单次结果，不能据此宣称模型质量提升。

运行产物：`/data/cache/search-r1/experiments/03-ablations/seed-gate/`。

SwanLab 历史回放（console 指标上传，不含实时系统遥测）：

- same-a: `e717o5lv`
- same-b: `xt5m0nxa`
- different: `gfk3vl14`

下一步按 `PLAN.md` 进入 03-A 行为探针，再做 03-B 的 `state_masking` 和 reward 核心消融。
