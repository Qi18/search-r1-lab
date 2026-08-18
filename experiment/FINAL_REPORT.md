# Search-R1 当前实验报告

更新时间：2026-08-18

## 当前结论

Stage00–03 已完成，项目具备进入官方真实数据实验的条件：搜索闭环、四象限评测、Tiny GRPO、checkpoint 回载、seed 传递和实验验收均已打通。

下一阶段是官方 Preliminary 路线：NQ + Wikipedia/E5 Retriever + PPO 小步训练。

## 已完成阶段

| Stage | 关键结果 | 结论边界 |
| --- | --- | --- |
| 00 | search EM 62.5%，no-search EM 0%，Hit@3 100% | 只有 8 条合成问题 |
| 01 | Base+Search EM 4.7%，官方 GRPO+Search EM 78.1% | 64 条合成事实，Retriever 条件命中率 100% |
| 02 | 20-step 后冻结 val Search EM 28.125%，Base 为 6.25% | train64/val32，不能外推真实 QA |
| 03 | 同 seed 训练轨迹哈希一致，异 seed 哈希变化 | GPU 更新不保证位级确定 |

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

## 下一验收点

Stage04-0 只做真实数据 preflight：

1. 下载并校验官方 NQ/HotpotQA 数据；
2. 下载并校验 Wiki-18 corpus 与 E5 index；
3. 启动 Retriever，测 Hit@k、P50/P95 和资源占用；
4. 估算 NQ PPO 1-step 的显存、时间和存储；
5. preflight 通过后再启动训练。

详细结果位于 `experiment/results/`；大文件位于 `/data/cache/search-r1/`；训练指标同步到 SwanLab 项目 `Search-R1`。
