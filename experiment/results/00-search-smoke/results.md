# Experiment 00: Search-R1 inference smoke test

- Trajectories: `/data/cache/search-r1-lab/experiments/00-search-smoke/trajectories.jsonl`
- Generated: `2026-08-17T06:56:12.648637+00:00`

| Mode | EM | Contains | F1 | Valid answer | Generated search | Retriever request | Hit@k | Avg turns | Avg latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-search | 0.0% | 0.0% | 3.1% | 100.0% | 100.0% | 0.0% | 0.0% | 1.00 | 1.99s |
| search | 62.5% | 100.0% | 83.3% | 100.0% | 100.0% | 100.0% | 100.0% | 1.00 | 1.67s |

## Per-question outcomes

| Mode | ID | Expected | Prediction | Generated | Requests |
| --- | --- | --- | --- | ---: | ---: |
| no-search | q1 | Mira Voss | The Helix Gate at Orilon Station was designed by the architectural firm Gensler. | 1 | 0 |
| no-search | q2 | 2041 | The Helix Gate entered service in the year 2371. | 1 | 0 |
| no-search | q3 | Xenon-129 | Uranium-238 | 1 | 0 |
| no-search | q4 | 17 | No autonomous drones surveyed Arcturus Ridge. | 1 | 0 |
| no-search | q5 | Ivo Sen | first curator of the Ember Archive | 1 | 0 |
| no-search | q6 | Rhea Calder | final answer> The Violet-7 coolant was invented by the German company Volkswagen in the 1970s. | 1 | 0 |
| no-search | q7 | Pelion Island | Quartz Harbor Observatory stands on an island. | 1 | 0 |
| no-search | q8 | every 19 minutes | The Aurora Ledger writes an off-site backup once a week. | 1 | 0 |
| search | q1 | Mira Voss | Mira Voss | 1 | 1 |
| search | q2 | 2041 | 2041 | 1 | 1 |
| search | q3 | Xenon-129 | Xenon-129 | 1 | 1 |
| search | q4 | 17 | 17 | 1 | 1 |
| search | q5 | Ivo Sen | First curator of the Ember Archive was the historian Ivo Sen | 1 | 1 |
| search | q6 | Rhea Calder | Engineer Rhea Calder | 1 | 1 |
| search | q7 | Pelion Island | Pelion Island | 1 | 1 |
| search | q8 | every 19 minutes | The Aurora Ledger writes an encrypted off-site backup every 19 minutes. | 1 | 1 |

## 结论

- 搜索链路有效：search 模式 Retriever 请求率为 100.0%，Hit@k 为 100.0%。
- 检索显著改善答案：EM 提升 62.5 个百分点，F1 提升 80.2 个百分点。
- 模型搜索倾向明确：两种模式的搜索标签生成率均为 100.0%，但 no-search 的 Retriever 请求率为 0.0%。
- search 模式 Contains 为 100.0%，高于 EM 的 62.5%，差异主要来自额外措辞。

## 结论边界

- 本阶段只比较同一个 Search-R1 GRPO checkpoint 是否启用 Retriever，不能单独证明 RL 的收益。
- 数据只有 8 条确定性合成问题，且 Retriever 命中率为 100%，不能代表真实任务泛化能力。
- no-search 是禁用工具的搜索模型，不等同于 Qwen Base；模型、Retriever 和 RL 的贡献需要 Stage 01 四象限实验拆分。
