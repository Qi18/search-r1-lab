# Experiment 01: Base vs Search-R1 GRPO

- Examples per quadrant: 64
- Generated: `2026-08-17T07:41:06.481305+00:00`
- Controls: same questions, prompt, retriever index, Top-3, greedy decoding, token limit and GPU

## 四象限结果

| Model | Retriever | EM | Contains | F1 | Valid | Search tag | Request | Hit@1 | Hit@3 | Turns | Input tok | Output tok | Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B Base | no-search | 0.0% | 0.0% | 1.8% | 71.9% | 59.4% | 0.0% | 0.0% | 0.0% | 0.62 | 176.6 | 144.9 | 1.90s |
| Qwen2.5-3B Base | search | 4.7% | 46.9% | 20.8% | 73.4% | 59.4% | 59.4% | 100.0% | 100.0% | 0.59 | 224.2 | 130.8 | 1.74s |
| Search-R1 GRPO | no-search | 0.0% | 0.0% | 2.6% | 98.4% | 100.0% | 0.0% | 0.0% | 0.0% | 1.03 | 249.8 | 100.2 | 1.32s |
| Search-R1 GRPO | search | 78.1% | 90.6% | 87.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 1.00 | 330.9 | 70.6 | 0.98s |

## 观察差异

| Comparison | EM | F1 |
| --- | ---: | ---: |
| Base: search - no-search | +4.7 pp | +19.1 pp |
| GRPO: search - no-search | +78.1 pp | +84.4 pp |
| No-search: GRPO - Base | +0.0 pp | +0.8 pp |
| Search: GRPO - Base | +73.4 pp | +66.1 pp |
| Difference in retrieval gain | +73.4 pp | +65.3 pp |

## 结果类型

| Model | Retriever | Exact | Contains only | Incorrect | Invalid format | Retrieval miss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B Base | no-search | 0 | 0 | 46 | 18 | 0 |
| Qwen2.5-3B Base | search | 3 | 27 | 17 | 17 | 0 |
| Search-R1 GRPO | no-search | 0 | 0 | 63 | 1 | 0 |
| Search-R1 GRPO | search | 50 | 8 | 6 | 0 | 0 |

## 结论

- Base 在工具可用时的实际请求率为 59.4%；GRPO 为 100.0%。
- 两个 search 组的条件 Hit@1/Hit@3 都是 100%，但 Base Contains 为 46.9%，GRPO 为 90.6%。
- 因此差异主要来自是否稳定调用工具以及能否利用返回证据，而不是底层索引本身。
- GRPO 开启搜索相对关闭搜索的观察差异：EM +78.1 pp，F1 +84.4 pp。
- 开启搜索时 GRPO 相对 Base 的观察差异：EM +73.4 pp，F1 +66.1 pp。

## 结论边界

- 当前使用 64 条固定合成问题，可完成 Stage 01 链路验收，但不能代表真实任务泛化能力。
- 工具可用不等于模型实际调用工具；Retriever 收益必须结合请求率解释。
- 观察差异不是严格因果归因，后续仍需真实数据集和多随机种子验证。

## 最终验收

- Status: PASS
- Retriever preflight: Hit@1 100.0%, Hit@3 100.0%

- [x] minimum_examples
- [x] balanced_quadrants
- [x] identical_question_ids
- [x] required_fields
- [x] nonempty_trajectories
- [x] finite_metrics
- [x] no_search_has_no_requests
- [x] failures_classified
- [x] retriever_preflight

## Per-question outcomes

| Model | Mode | ID | Expected | Prediction | Outcome | Generated | Requests | Hit@3 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Qwen2.5-3B Base | no-search | q-designer-01 | Mira Solen | It is not possible to determine who designed Avelin Observatory without further information or a search query. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-designer-02 | Tomas Vey | It is not possible to determine who designed Boreal Relay without further information or context. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-designer-03 | Lena Oris | Based on historical records, Cindar Gate was designed by the Ottoman architect Sinan. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-designer-04 | Soren Kale | It is not possible to determine who designed Dovren Array without further information or context. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-designer-05 | Nadia Voss | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-designer-06 | Ilan Meru | Shigeru Miyamoto designed Faron Spire. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-designer-07 | Rhea Nolin | Based on the information I found, I cannot provide a specific answer to the question. However, I can tell you that the design of Galen Vault was likely influenced by the needs and goals of the developers and designers who created it. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-designer-08 | Dara Quen | Helion Beacon was designed by a team of engineers and scientists at the Helion Corporation. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-service-01 | 2037 | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-service-02 | 2042 | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-service-03 | 2031 | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-service-04 | 2046 | Lumen Portal entered service in 2016. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-service-05 | 2039 | (missing) | invalid_format | 3 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-service-06 | 2044 | Neris Passage entered service in 1998. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-service-07 | 2034 | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-service-08 | 2048 | Pelar Arch entered service in 1943. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-isotope-01 | Xenon-129 | Quorin Sensor uses the isotope 131I. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-isotope-02 | Carbon-13 | Based on the information available, it appears that the Ravel Spectrometer does not use a specific isotope but rather measures the mass-to-charge ratio of ions to identify isotopes present in a sample. This means that the Ravel Spectrometer can detect and analyze various isotopes depending on the sample being analyzed. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-isotope-03 | Oxygen-18 | The isotope used by Sable Detector is Cobalt-60. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-isotope-04 | Helium-3 | There is no specific isotope used by Tarin Scanner. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-isotope-05 | Nitrogen-15 | Based on the information available, it appears that the Ulmar Imager does not have a specific isotope associated with it. However, given the context of imaging devices, it is likely that the Ulmar Imager uses a radioactive isotope similar to those commonly used in medical imaging, such as technetium-99m (Tc-99m), iodine-123 (I-123), or fluorine-18 (F-18). Without more specific information about the Ulmar Imager, it is challenging to pinpoint the exact isotope used. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-isotope-06 | Deuterium | There is no specific isotope mentioned for Vesper Counter in the provided information. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-isotope-07 | Silicon-29 | The isotope used by Weyland Analyzer is Cobalt-60. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-isotope-08 | Neon-22 | The isotope used by Xandor Monitor is Cobalt-60. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-drones-01 | 17 | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-drones-02 | 23 | 10 autonomous drones joined the Zephyr Basin Mission. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-drones-03 | 31 | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-drones-04 | 14 | 10 autonomous drones joined the Bracken Field Survey. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-drones-05 | 28 | 10 autonomous drones joined the Cobalt Dune Mission. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-drones-06 | 19 | 10 autonomous drones joined the Delta Crater Expedition. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-drones-07 | 26 | 10 autonomous drones joined the Ember Plain Survey. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-drones-08 | 12 | 10 autonomous drones have joined the Frost Vale Mission. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-curator-01 | Ivo Sen | There is no information available about the first curator of the Garnet Archive. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-curator-02 | Asha Pell | Based on the information available, the first curator of the Harbor Manuscript Room was not found. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-curator-03 | Niko Daren | There is no information available about the first curator of the Ivory Record Hall. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-curator-04 | Lea Morin | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-curator-05 | Oren Vale | There is no information available about the first curator of the Keystone Repository. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-curator-06 | Talia Sorn | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-curator-07 | Evan Kiro | There is no information available about the first curator of the Marble Registry. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-curator-08 | Sia Loren | There is no information available about the first curator of the Northwind Library. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-inventor-01 | Rhea Calder | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-inventor-02 | Marek Tovin | The Pulsar-8 lubricant was invented by a company called Pulsar-8. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-inventor-03 | Nina Sel | Based on the information available, the Quartz-2 catalyst was invented by a team of researchers led by Dr. John Smith. However, without further research, I cannot provide a definitive answer. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-inventor-04 | Jonas Mire | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-inventor-05 | Ari Venn | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-inventor-06 | Mila Koren | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-inventor-07 | Theo Rask | Dr. John Smith and Dr. Jane Doe invented the Umber-6 ceramic. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-inventor-08 | Lora Dey | Dr. John Smith invented the Violet-7 coolant. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-island-01 | Pelion Island | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-island-02 | Neris Island | Xeric Lighthouse stands on the island of St. John in the U.S. Virgin Islands. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-island-03 | Calder Island | Yonder Research Dome stands on the island of Kauai. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-island-04 | Orin Island | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-island-05 | Vela Island | Aster Radio Tower is located on the island of Sardinia. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-island-06 | Soren Island | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-island-07 | Tarin Island | Cedar Skywatch is located on an island. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-island-08 | Maren Island | Drift Signal Post stands on the island of Kauai, Hawaii. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-backup-01 | every 19 minutes | The Ember Ledger writes an off-site backup every 24 hours. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-backup-02 | every 23 minutes | The Fallow Registry does not write an off-site backup. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-backup-03 | every 31 minutes | Based on the information available, I couldn't find any specific details about the Granite Journal's backup practices. However, I can suggest that the newspaper may have a backup plan in place, but the frequency of the backups is not mentioned. It would be best to contact the newspaper directly for more information. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-backup-04 | every 17 minutes | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-backup-05 | every 29 minutes | The Indigo Record does not write an off-site backup. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-backup-06 | every 13 minutes | The Jasper Logbook does not write an off-site backup. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-backup-07 | every 37 minutes | The Kindle Archive does not write an off-site backup. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | no-search | q-backup-08 | every 41 minutes | The Lattice Ledger does not write an off-site backup. | incorrect | 1 | 0 | 0 |
| Qwen2.5-3B Base | search | q-designer-01 | Mira Solen | Ilana Meru designed Avelin Observatory. | incorrect | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-designer-02 | Tomas Vey | Doc 1 | incorrect | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-designer-03 | Lena Oris | Based on the search results, Cindar Gate was designed by Lena Oris. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-designer-04 | Soren Kale | Jonas Mire designed the Radian-5 alloy. | incorrect | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-designer-05 | Nadia Voss | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-designer-06 | Ilan Meru | Shigeru Miyamoto designed Faron Spire. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-designer-07 | Rhea Nolin | Based on the search results, I found that Galen Vault was designed by Rhea Nolin. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-designer-08 | Dara Quen | Helion Beacon was designed by Dara Quen. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-service-01 | 2037 | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-service-02 | 2042 | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-service-03 | 2031 | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-service-04 | 2046 | Lumen Portal entered service in 2046. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-service-05 | 2039 | Meridian Lock entered service in 2039. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-service-06 | 2044 | Neris Passage entered service in 1998. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-service-07 | 2034 | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-service-08 | 2048 | Pelar Arch entered service in 2048. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-isotope-01 | Xenon-129 | Quorin Sensor uses Xenon-129. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-isotope-02 | Carbon-13 | Based on the information from Doc 1, which is the most relevant to the question, the isotope used by the Ravel Spectrometer is Carbon-13. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-isotope-03 | Oxygen-18 | Deuterium | incorrect | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-isotope-04 | Helium-3 | Helium-3 | exact | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-isotope-05 | Nitrogen-15 | Based on the information from the search results, the isotope used by the Ulmar Imager is Nitrogen-15 (N-15). | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-isotope-06 | Deuterium | Deuterium | exact | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-isotope-07 | Silicon-29 | Weyland Analyzer uses Silicon-29. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-isotope-08 | Neon-22 | Neon-22 | exact | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-drones-01 | 17 | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-drones-02 | 23 | 10 autonomous drones joined the Zephyr Basin Mission. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-drones-03 | 31 | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-drones-04 | 14 | 10 autonomous drones joined the Bracken Field Survey. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-drones-05 | 28 | 10 autonomous drones joined the Cobalt Dune Mission. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-drones-06 | 19 | 10 autonomous drones joined the Delta Crater Expedition. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-drones-07 | 26 | 10 autonomous drones joined the Ember Plain Survey. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-drones-08 | 12 | 10 autonomous drones have joined the Frost Vale Mission. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-curator-01 | Ivo Sen | The first curator of the Garnet Archive was Ivo Sen. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-curator-02 | Asha Pell | Based on the search results, the first curator of the Harbor Manuscript Room was Asha Pell. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-curator-03 | Niko Daren | Based on the search results, the first curator of the Ivory Record Hall was Niko Daren. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-curator-04 | Lea Morin | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-curator-05 | Oren Vale | Doc 1 | incorrect | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-curator-06 | Talia Sorn | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-curator-07 | Evan Kiro | Doc 1 | incorrect | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-curator-08 | Sia Loren | Doc 1 | incorrect | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-inventor-01 | Rhea Calder | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-inventor-02 | Marek Tovin | Doc 1 | incorrect | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-inventor-03 | Nina Sel | Based on the search results, the Quartz-2 catalyst was invented by Nina Sel. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-inventor-04 | Jonas Mire | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-inventor-05 | Ari Venn | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-inventor-06 | Mila Koren | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-inventor-07 | Theo Rask | Based on the search results, the Umber-6 ceramic was invented by Theo Rask. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-inventor-08 | Lora Dey | Based on the search results, the inventor of the Violet-7 coolant is Lora Dey. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-island-01 | Pelion Island | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-island-02 | Neris Island | Xeric Lighthouse stands on the island of St. John in the U.S. Virgin Islands. | incorrect | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-island-03 | Calder Island | Yonder Research Dome stands on Calder Island. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-island-04 | Orin Island | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-island-05 | Vela Island | Aster Radio Tower is located on Vela Island. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-island-06 | Soren Island | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-island-07 | Tarin Island | Cedar Skywatch stands on Tarin Island. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-island-08 | Maren Island | Drift Signal Post stands on Maren Island. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-backup-01 | every 19 minutes | Ember Ledger writes an off-site backup every 19 minutes. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-backup-02 | every 23 minutes | The Fallow Registry writes an off-site backup every 23 minutes. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-backup-03 | every 31 minutes | Based on the information found in the search results, the Granite Journal writes an off-site backup every 31 minutes. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-backup-04 | every 17 minutes | (missing) | invalid_format | 0 | 0 | 0 |
| Qwen2.5-3B Base | search | q-backup-05 | every 29 minutes | Indigo Record writes an off-site backup every 29 minutes. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-backup-06 | every 13 minutes | It writes an off-site backup every 13 minutes. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-backup-07 | every 37 minutes | The Kindle Archive writes an off-site backup every 37 minutes. | contains_only | 1 | 1 | 1 |
| Qwen2.5-3B Base | search | q-backup-08 | every 41 minutes | The Lattice Ledger writes an off-site backup every 41 minutes. | contains_only | 1 | 1 | 1 |
| Search-R1 GRPO | no-search | q-designer-01 | Mira Solen | The Avelin Observatory was designed by the Swedish architect and engineer, Per Olof Strindberg. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-designer-02 | Tomas Vey | The Boreal Relay was designed by Bell Labs in the 1970s. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-designer-03 | Lena Oris | final answer> The design of Cindar Gate is not specified in the information returned by the search tool. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-designer-04 | Soren Kale | final answer> I couldn't find any information about who designed Dovren Array using the given query. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-designer-05 | Nadia Voss | The designer of Elaris Station is not specified in the information available. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-designer-06 | Ilan Meru | Final answer: The designer of Faron Spire is not specified in the given information. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-designer-07 | Rhea Nolin | Galen Vault designer is unknown. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-designer-08 | Dara Quen | Helion Beacon was designed by the company Helion Space Technologies. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-service-01 | 2037 | 2010 | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-service-02 | 2042 | 1962 | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-service-03 | 2031 | 1965 | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-service-04 | 2046 | 2015 | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-service-05 | 2039 | 1836 | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-service-06 | 2044 | Neris Passage entered service in 1937. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-service-07 | 2034 | Orilon Crossing entered service in 1997. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-service-08 | 2048 | Pelar Arch entered service in 1944. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-isotope-01 | Xenon-129 | Quorin Sensor is not a sensor that uses a specific isotope. It is a sensor that is used in various applications, such as in the field of nuclear physics. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-isotope-02 | Carbon-13 | Ravel Spectrometer | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-isotope-03 | Oxygen-18 | Uranium-235 | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-isotope-04 | Helium-3 | Carbon-14 | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-isotope-05 | Nitrogen-15 | Ulmar Imager uses the isotope Technetium-99m | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-isotope-06 | Deuterium | Uranium-235 | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-isotope-07 | Silicon-29 | Weyland Analyzer uses the isotope plutonium-239 | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-isotope-08 | Neon-22 | The Xandor Monitor uses the isotope cobalt-60. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-drones-01 | 17 | 0 | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-drones-02 | 23 | 10 autonomous drones joined the Zephyr Basin Mission. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-drones-03 | 31 | The number of autonomous drones that joined the Amber Shelf Expedition is not specified in the given information. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-drones-04 | 14 | 10 autonomous drones joined the Bracken Field Survey. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-drones-05 | 28 | The number of autonomous drones that joined the Cobalt Dune Mission is not specified in the given information. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-drones-06 | 19 | The Delta Crater Expedition was a space exploration mission that took place in the 21st century. Autonomous drones are robots that can operate without human intervention. The number of autonomous drones that joined the Delta Crater Expedition is not specified in the given information. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-drones-07 | 26 | The number of autonomous drones that joined the Ember Plain Survey is not specified in the given information. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-drones-08 | 12 | The number of autonomous drones that joined the Frost Vale Mission is not specified in the given information. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-curator-01 | Ivo Sen | first curator of the Garnet Archive | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-curator-02 | Asha Pell | First, I will search for the information about the first curator of the Harbor Manuscript Room. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-curator-03 | Niko Daren | First, I will search for the information about the first curator of the Ivory Record Hall. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-curator-04 | Lea Morin | First curator of the Juniper Collection | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-curator-05 | Oren Vale | First curator of the Keystone Repository | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-curator-06 | Talia Sorn | first curator of the Lantern Archive | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-curator-07 | Evan Kiro | first curator of the Marble Registry | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-curator-08 | Sia Loren | (missing) | invalid_format | 3 | 0 | 0 |
| Search-R1 GRPO | no-search | q-inventor-01 | Rhea Calder | John D. Anderson | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-inventor-02 | Marek Tovin | final answer> The Pulsar-8 lubricant was invented by the company Pulsar-8, which is a subsidiary of the company Shell. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-inventor-03 | Nina Sel | John B. Goodenough | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-inventor-04 | Jonas Mire | final answer>The Radian-5 alloy was invented by the German company Daimler-Benz in the 1950s. It was used in the construction of the company's cars and other products. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-inventor-05 | Ari Venn | final answer> The Solace-9 membrane was invented by a team of researchers at the University of California, Berkeley. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-inventor-06 | Mila Koren | final answer>The Tundra-3 sealant was invented by the American company 3M in 1957. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-inventor-07 | Theo Rask | John W. Goodenough | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-inventor-08 | Lora Dey | final answer> The Violet-7 coolant was invented by the German company Volkswagen in the 1970s. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-island-01 | Pelion Island | Westhaven Observatory stands on the island of Martha's Vineyard in Massachusetts, United States. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-island-02 | Neris Island | Xeric Lighthouse stands on the island of Saint John, in the U.S. state of Maine. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-island-03 | Calder Island | Yonder Research Dome stands on the island of Oahu, Hawaii. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-island-04 | Orin Island | Zeal Weather Station stands on the island of Zeeland in the Netherlands. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-island-05 | Vela Island | Aster Radio Tower stands on the island of Svalbard in Norway. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-island-06 | Soren Island | Beryl Marine Lab stands on the island of Saint Lucia. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-island-07 | Tarin Island | Cedar Island | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-island-08 | Maren Island | Drift Signal Post stands on an island in the United States. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-backup-01 | every 19 minutes | The Ember Ledger writes an off-site backup once a week. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-backup-02 | every 23 minutes | The Fallow Registry does not write an off-site backup. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-backup-03 | every 31 minutes | The Granite Journal does not provide information on how often they write an off-site backup. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-backup-04 | every 17 minutes | The Horizon Catalog does not write an off-site backup. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-backup-05 | every 29 minutes | The Indigo Record does not provide information on how often it writes an off-site backup. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-backup-06 | every 13 minutes | The Jasper Logbook does not provide information about how often it writes an off-site backup. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-backup-07 | every 37 minutes | The Kindle Archive writes an off-site backup once a week. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | no-search | q-backup-08 | every 41 minutes | The Lattice Ledger writes an off-site backup once a week. | incorrect | 1 | 0 | 0 |
| Search-R1 GRPO | search | q-designer-01 | Mira Solen | Mira Solen | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-designer-02 | Tomas Vey | Tomás Vey | incorrect | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-designer-03 | Lena Oris | Lena Oris | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-designer-04 | Soren Kale | Jonas Mire | incorrect | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-designer-05 | Nadia Voss | Nadia Voss | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-designer-06 | Ilan Meru | Ilana Meru | incorrect | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-designer-07 | Rhea Nolin | Nolin | incorrect | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-designer-08 | Dara Quen | Ilana Meru | incorrect | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-service-01 | 2037 | 2037 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-service-02 | 2042 | 2042 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-service-03 | 2031 | 2031 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-service-04 | 2046 | Lumen Portal entered service in 2046. | contains_only | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-service-05 | 2039 | 2039 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-service-06 | 2044 | 2044 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-service-07 | 2034 | 2034 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-service-08 | 2048 | 2048 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-isotope-01 | Xenon-129 | Xenon-129 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-isotope-02 | Carbon-13 | Carbon-13 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-isotope-03 | Oxygen-18 | Oxygen-18 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-isotope-04 | Helium-3 | Helium-3 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-isotope-05 | Nitrogen-15 | Nitrogen-15 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-isotope-06 | Deuterium | Deuterium | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-isotope-07 | Silicon-29 | Silicon-29 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-isotope-08 | Neon-22 | Neon-22 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-drones-01 | 17 | 17 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-drones-02 | 23 | 23 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-drones-03 | 31 | 31 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-drones-04 | 14 | 14 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-drones-05 | 28 | 28 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-drones-06 | 19 | 19 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-drones-07 | 26 | 26 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-drones-08 | 12 | 12 | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-curator-01 | Ivo Sen | Ivo Sen | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-curator-02 | Asha Pell | Asha Pell | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-curator-03 | Niko Daren | the first curator of the Ivory Record Hall was Niko Daren | contains_only | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-curator-04 | Lea Morin | Lea Morin | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-curator-05 | Oren Vale | Oren Vale | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-curator-06 | Talia Sorn | First curator of the Lantern Archive was Talia Sorn | contains_only | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-curator-07 | Evan Kiro | Evans Kiro | incorrect | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-curator-08 | Sia Loren | Sia Loren | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-inventor-01 | Rhea Calder | Rhea Calder | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-inventor-02 | Marek Tovin | Marek Tovin | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-inventor-03 | Nina Sel | Nina Sel | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-inventor-04 | Jonas Mire | Jonas Mire | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-inventor-05 | Ari Venn | Ari Venn | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-inventor-06 | Mila Koren | Mila Koren | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-inventor-07 | Theo Rask | Theo Rask | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-inventor-08 | Lora Dey | Lora Dey | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-island-01 | Pelion Island | Pelion Island | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-island-02 | Neris Island | Xeric Lighthouse stands on Neris Island. | contains_only | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-island-03 | Calder Island | Calder Island | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-island-04 | Orin Island | Orin Island | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-island-05 | Vela Island | Aster Radio Tower stands on Vela Island. | contains_only | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-island-06 | Soren Island | Beryl Marine Lab stands on Soren Island. | contains_only | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-island-07 | Tarin Island | Cedar Skywatch stands on Tarin Island. | contains_only | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-island-08 | Maren Island | Maren Island | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-backup-01 | every 19 minutes | every 19 minutes | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-backup-02 | every 23 minutes | every 23 minutes | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-backup-03 | every 31 minutes | every 31 minutes | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-backup-04 | every 17 minutes | every 17 minutes | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-backup-05 | every 29 minutes | every 29 minutes | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-backup-06 | every 13 minutes | every 13 minutes | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-backup-07 | every 37 minutes | every 37 minutes | exact | 1 | 1 | 1 |
| Search-R1 GRPO | search | q-backup-08 | every 41 minutes | The Lattice Ledger writes an off-site backup every 41 minutes. | contains_only | 1 | 1 | 1 |
