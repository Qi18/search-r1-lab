# Search-R1 experiment report

## Current status

The repository baseline and Stage 01 pilot are initialized on L20-Server. The
reference experiments are `runs/runsmoke.sh` and `runs/run01_base_vs_rl.sh`,
following the same operational pattern as NanoChat: ordered shell pipelines,
unbuffered Python output, cache isolation, explicit
acceptance gates, and a checked-in result summary.

Environment:

- compute: 8x NVIDIA L20, using GPU0 for Qwen and CPU for the small retriever;
- language model: official Search-R1 Qwen2.5-3B GRPO checkpoint;
- retriever: E5-small-v2 + FAISS Flat inner-product index;
- corpus: eight deterministic synthetic documents;
- evaluation: eight exact-answer questions;
- large artifact root: `/data/cache/search-r1-lab`.

## Verified run

- five protocol and metric unit tests passed;
- the full `runs/runsmoke.sh` chain completed;
- all eight search queries retrieved the expected evidence in Top-3;
- all sixteen trajectories used valid `<answer>` formatting;
- every trajectory generated exactly one `<search>` action;
- no-search made zero Retriever requests, while search made eight.

## Result

| Mode | EM | Answer contains | F1 | Generated search | Retriever request | Hit@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Search disabled | 0.0% | 0.0% | 3.1% | 100.0% | 0.0% | 0.0% |
| Search enabled | 62.5% | 100.0% | 83.3% | 100.0% | 100.0% | 100.0% |

## 结论

- 搜索链路有效：search 模式真实调用 Retriever，8 条问题全部命中目标证据；
- 检索显著改善答案：EM 提升 62.5 个百分点，F1 提升 80.2 个百分点；
- 当前 Search-R1 checkpoint 有明确搜索倾向：两种模式都生成搜索动作，但 no-search 不执行工具调用；
- search 的 Contains 为 100%，高于 EM 的 62.5%，差异主要来自答案中的额外措辞。

## 结论边界

- 本阶段只比较同一个 Search-R1 GRPO checkpoint 是否启用 Retriever，不能单独证明 RL 的收益；
- 数据只有 8 条确定性合成问题，且 Retriever 命中率为 100%，不能代表真实任务泛化能力；
- no-search 是禁用工具的搜索模型，不等同于 Qwen Base；模型、Retriever 和 RL 的贡献需要 Stage 01 四象限实验拆分。

## 实现说明

The first implementation placed Qwen on GPU0 and E5 on GPU1 in one Python
process. After the no-search leg, the first E5 query failed with
`CUDA error: invalid resource handle`. The final design keeps Qwen isolated on
GPU0 and runs the small E5 retriever on CPU. This preserves the retrieval model,
index, questions, and metrics while avoiding cross-device CUDA state.

Detailed outcomes are in
`experiment/results/00-search-smoke/results.md`. Large artifacts are under
`/data/cache/search-r1-lab/experiments/00-search-smoke`.

## Stage 01 pilot

使用同一组 8 条问题、Prompt、Retriever、Top-3、greedy decoding、Token 上限和 GPU，完成 Base/GRPO 与 Retriever 开关四象限：

| Model | Retriever | EM | Contains | Valid | Request | Hit@3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B Base | disabled | 0.0% | 0.0% | 75.0% | 0.0% | 0.0% |
| Qwen2.5-3B Base | enabled | 12.5% | 62.5% | 75.0% | 75.0% | 83.3% |
| Search-R1 GRPO | disabled | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% |
| Search-R1 GRPO | enabled | 62.5% | 100.0% | 100.0% | 100.0% | 100.0% |

### 初步结论

- Base 可以使用搜索工具，但协议不稳定：2/8 没有合法答案，实际请求率为 75%；
- GRPO 的格式有效率、请求率和 Hit@3 都达到 100%；
- 开启搜索时，GRPO 相对 Base 的观察差异为 EM +50.0、F1 +53.9 个百分点；
- 关闭搜索时两者 EM 都为 0%，当前优势主要体现在搜索协议执行和证据利用，而不是合成事实的内部记忆。

### 状态边界

Stage 01 pilot 链路已完成，但 8 条合成数据不足以完成最终归因。下一步先扩展到 50-200 条固定 QA，再决定是否进入 Tiny GRPO。

详细结果位于 `experiment/results/01-base-vs-rl/results.md`，完整轨迹和日志位于 `/data/cache/search-r1-lab/experiments/01-base-vs-rl`。
