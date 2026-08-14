# Search-R1 experiment report

## Current status

The repository baseline is initialized on L20-Server. The reference experiment
is `runs/runsmoke.sh`, following the same operational pattern as NanoChat:
one ordered shell pipeline, unbuffered Python output, cache isolation, explicit
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
- every trajectory made exactly one `<search>` call.

## Result

| Mode | EM | Answer contains | F1 | Hit@3 |
| --- | ---: | ---: | ---: | ---: |
| Search disabled | 0.0% | 0.0% | 3.1% | 0.0% |
| Search enabled | 62.5% | 100.0% | 83.3% | 100.0% |

Enabling retrieval improved exact match by 62.5 percentage points and token F1
by 80.2 points. Three search-enabled answers contained the exact gold fact plus
extra wording, so answer containment reached 100% while strict EM was 62.5%.

The first implementation placed Qwen on GPU0 and E5 on GPU1 in one Python
process. After the no-search leg, the first E5 query failed with
`CUDA error: invalid resource handle`. The final design keeps Qwen isolated on
GPU0 and runs the small E5 retriever on CPU. This preserves the retrieval model,
index, questions, and metrics while avoiding cross-device CUDA state.

Detailed outcomes are in
`learning/experiments/00-search-smoke/results.md`. Large artifacts are under
`/data/cache/search-r1-lab/experiments/00-search-smoke`.
