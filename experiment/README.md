# Search-R1 L20 Experiment

A small, reproducible laboratory for understanding and reproducing Search-R1.
The repository follows NanoChat's experiment style: one readable shell entry
point, explicit stage ordering, unbuffered logs, cache isolation, and a checked
in result report.

The first experiment asks one concrete question:

> Does an official Search-R1 GRPO checkpoint answer synthetic facts more
> accurately when its search tool is enabled?

## Quick start

The default L20 setup expects:

- 1 CUDA GPU for the language model; the small E5 retriever runs on CPU
- the official Search-R1 Qwen2.5-3B GRPO checkpoint
- Python 3.10 with PyTorch, Transformers, NumPy, and FAISS

```bash
source experiment/env.sh
bash experiment/runs/runsmoke.sh
```

Override paths without editing scripts:

```bash
SEARCH_R1_MODEL_PATH=/path/to/model \
SEARCH_R1_LAB_CACHE=/path/to/cache \
bash experiment/runs/runsmoke.sh
```

## Reference pipeline

```text
environment check
  -> dense index build
  -> no-search baseline
  -> Search-R1 multi-turn search
  -> EM/F1/tool metrics
  -> experiment/learning/experiments/00-search-smoke/results.md
```

Large models, indexes, raw logs, and generated trajectories live under
`SEARCH_R1_LAB_CACHE`. Git tracks only source, small synthetic fixtures, plans,
and compact result summaries.

## Layout

```text
experiment/runs/runsmoke.sh                 reference smoke pipeline
experiment/scripts/build_index.py           build the E5 + FAISS index
experiment/scripts/run_eval.py              run no-search/search trajectories
experiment/scripts/summarize.py             calculate and render metrics
experiment/search_r1_lab/                   reusable protocol, retrieval, agent, metrics
experiment/data/                            deterministic synthetic corpus and QA set
experiment/learning/experiments/            checked-in experiment conclusions
experiment/TRAINING_PLAN.md                 staged path from smoke test to GRPO
experiment/FINAL_REPORT.md                  latest verified project state
```

This is not yet a full paper reproduction. The full NQ/HotpotQA + Wikipedia
GRPO run is deliberately gated on the small inference and training smoke tests.

## References

- Search-R1 paper: https://arxiv.org/abs/2503.09516
- Official implementation: https://github.com/PeterGriffinJin/Search-R1
- Official checkpoints: https://huggingface.co/collections/PeterJinGo/search-r1
