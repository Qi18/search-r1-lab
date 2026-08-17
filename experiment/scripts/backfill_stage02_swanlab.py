from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import swanlab

from check_stage02_run import parse_metrics


DEFAULT_STEPS = (1, 5, 20)


def finite_metrics(row: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in row.items():
        if key == "step":
            continue
        number = float(value)
        if math.isfinite(number):
            metrics[key] = number
    return metrics


def run_paths(run_root: Path, expected_step: int) -> tuple[Path, Path, Path]:
    run_dir = run_root / f"02-tiny-grpo-{expected_step}step"
    return (
        run_dir / "train.log",
        run_dir / "acceptance.json",
        run_dir / "checkpoints" / "actor" / f"global_step_{expected_step}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill Stage02 console metrics into SwanLab experiments."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project", default="Search-R1")
    parser.add_argument("--workspace")
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    logdir = Path(args.logdir).resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "kind": "historical-backfill",
        "mode": args.mode,
        "project": args.project,
        "workspace": args.workspace,
        "runs": [],
    }

    for expected_step in args.steps:
        train_log, acceptance_path, checkpoint = run_paths(run_root, expected_step)
        if not train_log.is_file():
            raise FileNotFoundError(train_log)
        if not acceptance_path.is_file():
            raise FileNotFoundError(acceptance_path)
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(checkpoint / "config.json")

        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        rows = parse_metrics(train_log.read_text(encoding="utf-8", errors="replace"))
        training_rows = [row for row in rows if "actor/pg_loss" in row]
        if len(training_rows) != expected_step:
            raise RuntimeError(
                f"{train_log}: expected {expected_step} training rows, got {len(training_rows)}"
            )
        if acceptance.get("status") != "PASS":
            raise RuntimeError(f"{acceptance_path}: acceptance is not PASS")

        experiment_name = f"stage02-tiny-grpo-{expected_step}step-backfill"
        before = {path.name for path in logdir.glob("run-*")}
        init_kwargs = {
            "project": args.project,
            "experiment_name": experiment_name,
            "description": (
                "Historical replay of Search-R1 Stage02 console metrics. "
                "This run was created after training and does not contain live system telemetry."
            ),
            "job_type": "historical-backfill",
            "group": "stage02-tiny-grpo",
            "tags": ["Search-R1", "Stage02", "GRPO", "backfill"],
            "config": {
                "backfill": True,
                "algorithm": "GRPO",
                "base_model": "Qwen2.5-3B",
                "expected_steps": expected_step,
                "train_examples": 64,
                "validation_examples": 32,
                "train_batch_size": 32,
                "learning_rate": 1e-6,
                "gpus": 8,
                "max_turns": 2,
                "retriever_topk": 3,
                "checkpoint": str(checkpoint),
                "source_log": str(train_log),
                "acceptance": acceptance["status"],
            },
            "logdir": str(logdir),
            "mode": args.mode,
            "reinit": True,
        }
        if args.workspace:
            init_kwargs["workspace"] = args.workspace
        run = swanlab.init(**init_kwargs)
        for row in training_rows:
            run.log(finite_metrics(row), step=int(row["step"]))
        run.log({"acceptance/pass": 1.0}, step=expected_step)
        swanlab.finish()

        after = {path.name for path in logdir.glob("run-*")}
        manifest["runs"].append(
            {
                "experiment_name": experiment_name,
                "expected_steps": expected_step,
                "logged_rows": len(training_rows),
                "acceptance": acceptance["status"],
                "checkpoint": str(checkpoint),
                "swanlab_run_id": getattr(run, "id", None),
                "new_log_directories": sorted(after - before),
            }
        )

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
