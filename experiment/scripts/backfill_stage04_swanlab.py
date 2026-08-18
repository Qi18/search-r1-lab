from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import swanlab


METRIC_LINE = re.compile(r"step:(?P<step>\d+)(?P<body>.*)")
METRIC = re.compile(r" - (?P<key>[^:]+):(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)", re.I)


def training_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = METRIC_LINE.search(line)
        if not match:
            continue
        row = {"step": int(match.group("step"))}
        row.update({m.group("key"): float(m.group("value")) for m in METRIC.finditer(match.group("body"))})
        if "actor/pg_loss" in row:
            rows.append(row)
    return rows


def finite(row: dict) -> dict[str, float]:
    result = {}
    for key, value in row.items():
        if key == "step":
            continue
        number = float(value)
        if math.isfinite(number):
            result[key] = number
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run", choices=("train-1", "train-5", "summary"), required=True)
    parser.add_argument("--project", default="Search-R1")
    parser.add_argument("--workspace")
    parser.add_argument("--mode", choices=("offline", "online"), default="online")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    common = {
        "project": args.project,
        "job_type": "historical-backfill",
        "group": "stage04-official-preliminary-nq-ppo",
        "tags": ["Search-R1", "Stage04", "NQ", "PPO", "backfill"],
        "logdir": str(logdir),
        "mode": args.mode,
        "reinit": True,
    }
    if args.workspace:
        common["workspace"] = args.workspace

    if args.run.startswith("train-"):
        step = int(args.run.split("-")[1])
        run_dir = run_root / f"stage04-nq-ppo-{step}step"
        acceptance = json.loads((run_dir / "acceptance.json").read_text())
        rows = training_rows(run_dir / "train.log")
        if acceptance["status"] != "PASS" or len(rows) != step:
            raise RuntimeError(f"Stage04 {step}-step run is not accepted")
        run = swanlab.init(
            **common,
            experiment_name=f"stage04-nq-ppo-{step}step-backfill",
            description="Historical replay of official Preliminary NQ PPO console metrics; no live telemetry.",
            config={
                "backfill": True,
                "algorithm": "PPO/GAE",
                "base_model": "Qwen2.5-3B",
                "updates": step,
                "train_examples": 512,
                "validation_examples": 128,
                "train_batch_size": 64,
                "learning_rate": 1e-6,
                "critic_learning_rate": 1e-5,
                "gpus": 8,
                "max_turns": 2,
                "retriever": "Wiki-18/E5-base-v2 topk=3",
            },
        )
        for row in rows:
            run.log(finite(row), step=int(row["step"]))
        run.log({"acceptance/pass": 1.0}, step=step)
    else:
        metrics = json.loads(Path(args.metrics).read_text())
        if metrics["status"] != "PASS":
            raise RuntimeError("Stage04 summary is not accepted")
        run = swanlab.init(
            **common,
            experiment_name="stage04-nq-ppo-acceptance-backfill",
            description="Frozen NQ baseline and checkpoint-reload acceptance summary.",
            config={"backfill": True, "validation_examples": 128, "retriever_topk": 3},
        )
        for step, name in enumerate(("baseline", "ppo-1step", "ppo-5step")):
            item = metrics["evaluations"][name]
            run.log({f"eval/{key}": value for key, value in item.items() if key != "examples"}, step=step)
        run.log({
            "retriever/hit_at_1": metrics["retriever"]["hit_at_1"],
            "retriever/hit_at_3": metrics["retriever"]["hit_at_3"],
            "acceptance/pass": 1.0,
        }, step=3)
    swanlab.finish()
    payload = {"run": args.run, "swanlab_run_id": getattr(run, "id", None), "mode": args.mode}
    Path(args.manifest).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
