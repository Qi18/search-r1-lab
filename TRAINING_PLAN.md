# Search-R1 experiment plan

## Goal

Reproduce Search-R1 in progressively more expensive stages while keeping every
stage independently runnable and measurable.

```text
00 inference smoke
  -> 01 base-vs-RL ablation
  -> 02 tiny GRPO training
  -> 03 NQ/HotpotQA reproduction
```

## Stage 00: inference smoke

Reference command:

```bash
bash runs/runsmoke.sh
```

The experiment compares the same official Qwen2.5-3B GRPO checkpoint with the
search tool disabled and enabled on synthetic facts that are absent from the
base model's pretraining data.

Acceptance gates:

- deterministic fixture and index build succeed;
- every sample produces a recorded trajectory;
- format-valid rate, search-call rate, Hit@3, EM, and F1 are reported;
- no CUDA OOM, NaN, or malformed JSONL artifacts;
- source, command, logs, metrics, and conclusion are all recoverable.

## Stage 01: base-vs-RL ablation

Add Qwen2.5-3B Base and evaluate four cells:

| Model | Search |
| --- | --- |
| Qwen2.5-3B Base | disabled |
| Qwen2.5-3B Base | enabled |
| Search-R1 GRPO | disabled |
| Search-R1 GRPO | enabled |

This separates the value of retrieval from the value of RL-trained tool use.

## Stage 02: tiny GRPO training

Build an isolated veRL/vLLM environment rather than modifying the existing
PyTorch 2.6 container environment. Start with 50-200 QA items, four rollouts per
question, at most two search turns, and 5-20 optimizer steps.

Training acceptance gates:

- rollout, reward, and optimizer steps all execute;
- answer and format rewards remain finite;
- search-call and valid-format metrics are visible;
- a checkpoint can be loaded by the Stage 00 evaluator;
- the pre/post-training comparison is recorded.

## Stage 03: paper-scale reproduction

Only after Stage 02 passes:

- process NQ/HotpotQA;
- build or download the Wikipedia corpus and index;
- run the official 3B GRPO configuration on 8xL20;
- evaluate EM/F1 and search behavior against frozen baselines.

Full-run time and storage are estimated from the measured smoke throughput, not
from H100 numbers in external reports.
