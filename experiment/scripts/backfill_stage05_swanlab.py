from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import swanlab


METRIC_LINE = re.compile(r"step:(?P<step>\d+)(?P<body>.*)")
METRIC = re.compile(r" - (?P<key>[^:]+):(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)", re.I)


def training_row(path: Path) -> dict:
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        match = METRIC_LINE.search(line)
        if match and "actor/pg_loss" in line:
            row = {"step": int(match.group("step"))}
            row.update({item.group("key"): float(item.group("value")) for item in METRIC.finditer(match.group("body"))})
            return row
    raise RuntimeError(f"no training metrics in {path}")


def finite(row: dict) -> dict[str, float]:
    return {key: float(value) for key, value in row.items() if key != "step" and math.isfinite(float(value))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run", choices=("ppo", "grpo", "summary"), required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    metrics = json.loads(Path(args.metrics).read_text())
    common = {
        "project": "Search-R1",
        "group": "stage05-official-v01-short-gate",
        "tags": ["Search-R1", "Stage05", "v0.1", "backfill"],
        "logdir": args.logdir,
        "mode": "online",
        "reinit": True,
    }
    if args.run in ("ppo", "grpo"):
        acceptance = metrics["training"][args.run]
        if acceptance["status"] != "PASS":
            raise RuntimeError(f"{args.run} is not accepted")
        row = training_row(root / f"{args.run}-1step" / "train.log")
        run = swanlab.init(
            **common,
            experiment_name=f"stage05-{args.run}-1step-backfill",
            description="Historical replay of Stage05 short-gate training metrics; no live telemetry.",
            config={
                "backfill": True,
                "algorithm": args.run.upper(),
                "train_examples": 512,
                "validation_examples": 112,
                "train_batch_size": 32,
                "n_agent": 1 if args.run == "ppo" else 5,
                "max_turns": 4,
                "retriever": "Wiki-18/E5-base-v2 topk=3",
                "updates": 1,
                "gpus": 8,
            },
        )
        run.log(finite(row), step=1)
        run.log({"acceptance/pass": 1.0}, step=1)
    else:
        if metrics["status"] != "PASS":
            raise RuntimeError("Stage05 summary is not accepted")
        run = swanlab.init(
            **common,
            experiment_name="stage05-seven-dataset-acceptance-backfill",
            description="Frozen seven-dataset baseline and PPO/GRPO checkpoint-reload summary.",
            config={"backfill": True, "validation_examples": 112, "datasets": 7, "retriever_topk": 3},
        )
        for step, name in enumerate(("baseline", "ppo-1step", "grpo-1step")):
            item = metrics["evaluations"][name]["overall"]
            run.log({f"eval/{key}": value for key, value in item.items() if key != "examples"}, step=step)
        run.log({"acceptance/pass": 1.0}, step=3)
    swanlab.finish()
    Path(args.manifest).write_text(json.dumps({"run": args.run, "swanlab_run_id": getattr(run, "id", None)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
