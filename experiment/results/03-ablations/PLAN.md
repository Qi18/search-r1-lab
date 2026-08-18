# Stage 03：可复现性门禁

状态：**PASS**。

Stage03 的作用是验证 seed 是否真正控制数据顺序和生成采样，并明确 FSDP/CUDA 的非位级确定边界。它不再提前执行官方 v0.3 消融。

## 已完成

1. 同 seed=3103 重复两次 2-step；
2. 异 seed=3104 运行一次 2-step；
3. 比较训练期 query、topk 和 result IDs 的轨迹哈希；
4. 记录 reward、训练指标与训练后验证差异；
5. 将 console 指标回放到 SwanLab。

固定配置：Qwen2.5-3B、train64/val32/corpus96、GRPO、batch32、`n_agent=4`、`max_turns=2`、topk3、lr 1e-6。

## 旧计划迁移

| 原 Stage03 项目 | 新位置 | 原因 |
| --- | --- | --- |
| Base/GRPO、Search on/off | Stage04 基线 | 与真实 NQ Preliminary 一起测 |
| `state_masking=true/false` | Stage06 | 对应官方 v0.2 masking 修复 |
| answer/format reward | Stage07 | 对应官方 v0.3 reward design |
| Retriever/backbone/data scaling | Stage07 | 对应官方 v0.3 系统消融 |
| 多跳、topk、噪声和故障 | Stage08 | 属于官方复现后的自定义扩展 |

结果保留在本目录；目录名 `03-ablations` 作为历史路径不再改名。运行产物位于 `/data/cache/search-r1/experiments/03-ablations/`。

下一步：Stage04-0，官方 NQ/Wiki-18/E5 数据和索引 preflight。
