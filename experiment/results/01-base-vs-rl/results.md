# Experiment 01: Base vs Search-R1 GRPO

- Examples per quadrant: 8
- Generated: `2026-08-17T07:11:25.678846+00:00`
- Controls: same questions, prompt, retriever index, Top-3, greedy decoding, token limit and GPU

## 四象限结果

| Model | Retriever | EM | Contains | F1 | Valid | Search tag | Request | Hit@1 | Hit@3 | Turns | Input tok | Output tok | Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B Base | no-search | 0.0% | 0.0% | 4.2% | 75.0% | 75.0% | 0.0% | 0.0% | 0.0% | 0.75 | 190.2 | 159.8 | 2.12s |
| Qwen2.5-3B Base | search | 12.5% | 62.5% | 29.4% | 75.0% | 75.0% | 75.0% | 83.3% | 83.3% | 0.75 | 287.5 | 147.1 | 2.18s |
| Search-R1 GRPO | no-search | 0.0% | 0.0% | 3.1% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1.00 | 251.8 | 147.8 | 1.96s |
| Search-R1 GRPO | search | 62.5% | 100.0% | 83.3% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 1.00 | 380.4 | 121.1 | 1.67s |

## 观察差异

| Comparison | EM | F1 |
| --- | ---: | ---: |
| Base: search - no-search | +12.5 pp | +25.2 pp |
| GRPO: search - no-search | +62.5 pp | +80.2 pp |
| No-search: GRPO - Base | +0.0 pp | -1.1 pp |
| Search: GRPO - Base | +50.0 pp | +53.9 pp |
| Difference in retrieval gain | +50.0 pp | +55.0 pp |

## 结果类型

| Model | Retriever | Exact | Contains only | Incorrect | Invalid format | Retrieval miss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B Base | no-search | 0 | 0 | 6 | 2 | 0 |
| Qwen2.5-3B Base | search | 1 | 4 | 0 | 2 | 1 |
| Search-R1 GRPO | no-search | 0 | 0 | 8 | 0 | 0 |
| Search-R1 GRPO | search | 5 | 3 | 0 | 0 | 0 |

## 初步结论

- Base 在工具可用时的实际请求率为 75.0%；GRPO 为 100.0%。
- GRPO 开启搜索相对关闭搜索的观察差异：EM +62.5 pp，F1 +80.2 pp。
- 开启搜索时 GRPO 相对 Base 的观察差异：EM +50.0 pp，F1 +53.9 pp。

## 结论边界

- 当前是 8 条合成问题的 Stage 01 pilot，只能验证四象限链路和形成初步观察。
- 工具可用不等于模型实际调用工具；Retriever 收益必须结合请求率解释。
- 观察差异不是严格因果归因，下一步需扩展固定数据集并检查多随机种子或采样稳定性。

## Per-question outcomes

| Model | Mode | ID | Expected | Prediction | Outcome | Generated | Requests | Hit@3 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Qwen2.5-3B Base | no-search | q1 | Mira Voss | The Helix Gate at Orilon Station was designed by the architect and engineer, Dr. J. K. Rowling. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q2 | 2041 | The Helix Gate entered service in 1999. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q3 | Xenon-129 | final answer | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q4 | 17 | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q5 | Ivo Sen | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q6 | Rhea Calder | Dr. John Smith invented the Violet-7 coolant. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q7 | Pelion Island | Quartz Harbor Observatory is located on the island of Kauai in Hawaii. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q8 | every 19 minutes | The Aurora Ledger writes an off-site backup every 24 hours. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | search | q1 | Mira Voss | Systems architect Mira Voss designed the Helix Gate at Orilon Station. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q2 | 2041 | The Helix Gate entered service in 2041. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q3 | Xenon-129 | final answer | retrieval_miss | 1 | 1 | 0 |
| Qwen2.5-3B Base | search | q4 | 17 | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q5 | Ivo Sen | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q6 | Rhea Calder | Based on the information found, the inventor of the Violet-7 coolant is Engineer Rhea Calder. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q7 | Pelion Island | Quartz Harbor Observatory stands on Pelion Island. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q8 | every 19 minutes | Every 19 minutes. | exact | 1 | 1 | 1 |
| Search-R1 GRPO | no-search | q1 | Mira Voss | The Helix Gate at Orilon Station was designed by the architectural firm Gensler. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q2 | 2041 | The Helix Gate entered service in the year 2371. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q3 | Xenon-129 | Uranium-238 | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q4 | 17 | No autonomous drones surveyed Arcturus Ridge. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q5 | Ivo Sen | first curator of the Ember Archive | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q6 | Rhea Calder | final answer> The Violet-7 coolant was invented by the German company Volkswagen in the 1970s. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q7 | Pelion Island | Quartz Harbor Observatory stands on an island. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q8 | every 19 minutes | The Aurora Ledger writes an off-site backup once a week. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | search | q1 | Mira Voss | Mira Voss | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q2 | 2041 | 2041 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q3 | Xenon-129 | Xenon-129 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q4 | 17 | 17 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q5 | Ivo Sen | First curator of the Ember Archive was the historian Ivo Sen | contains_only | 1 | 1 | 1 |
| Search-R1 GRPO | search | q6 | Rhea Calder | Engineer Rhea Calder | contains_only | 1 | 1 | 1 |
| Search-R1 GRPO | search | q7 | Pelion Island | Pelion Island | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q8 | every 19 minutes | The Aurora Ledger writes an encrypted off-site backup every 19 minutes. | contains_only | 1 | 1 | 1 |
