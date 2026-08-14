# Experiment 00: Search-R1 inference smoke test

- Trajectories: `/data/cache/search-r1-lab/experiments/00-search-smoke/trajectories.jsonl`
- Generated: `2026-08-14T04:01:14.677327+00:00`

| Mode | EM | Contains | F1 | Valid answer | Search calls | Hit@k | Avg searches | Avg latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-search | 0.0% | 0.0% | 3.1% | 100.0% | 100.0% | 0.0% | 1.00 | 1.97s |
| search | 62.5% | 100.0% | 83.3% | 100.0% | 100.0% | 100.0% | 1.00 | 1.74s |

## Per-question outcomes

| Mode | ID | Expected | Prediction | Searches |
| --- | --- | --- | --- | ---: |
| no-search | q1 | Mira Voss | The Helix Gate at Orilon Station was designed by the architectural firm Gensler. | 1 |
| no-search | q2 | 2041 | The Helix Gate entered service in the year 2371. | 1 |
| no-search | q3 | Xenon-129 | Uranium-238 | 1 |
| no-search | q4 | 17 | No autonomous drones surveyed Arcturus Ridge. | 1 |
| no-search | q5 | Ivo Sen | first curator of the Ember Archive | 1 |
| no-search | q6 | Rhea Calder | final answer> The Violet-7 coolant was invented by the German company Volkswagen in the 1970s. | 1 |
| no-search | q7 | Pelion Island | Quartz Harbor Observatory stands on an island. | 1 |
| no-search | q8 | every 19 minutes | The Aurora Ledger writes an off-site backup once a week. | 1 |
| search | q1 | Mira Voss | Mira Voss | 1 |
| search | q2 | 2041 | 2041 | 1 |
| search | q3 | Xenon-129 | Xenon-129 | 1 |
| search | q4 | 17 | 17 | 1 |
| search | q5 | Ivo Sen | First curator of the Ember Archive was the historian Ivo Sen | 1 |
| search | q6 | Rhea Calder | Engineer Rhea Calder | 1 |
| search | q7 | Pelion Island | Pelion Island | 1 |
| search | q8 | every 19 minutes | The Aurora Ledger writes an encrypted off-site backup every 19 minutes. | 1 |
