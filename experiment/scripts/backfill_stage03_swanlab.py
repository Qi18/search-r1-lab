from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import swanlab


FINAL_VAL = re.compile(r"Final validation metrics:.*?'val/test_score/nq': ([0-9.]+)")


def finite_metrics(row: dict) -> dict[str, float]:
    result = {}
    for key, value in row.items():
        number = float(value)
        if math.isfinite(number):
            result[key] = number
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project", default="Search-R1")
    parser.add_argument("--workspace")
    parser.add_argument("--mode", choices=("offline", "online"), default="online")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    root = Path(args.run_root)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    manifest = {"kind": "historical-backfill", "stage": "03-0", "runs": []}

    # SwanLab 0.9.4 clears online auth state after finish(), so upload one run
    # per process. This also makes retries idempotent at the command level.
    for run_name in (args.run,):
        run_dir = root / run_name
        acceptance = json.loads((run_dir / "acceptance.json").read_text())
        if not acceptance.get("accepted"):
            raise RuntimeError(f"{run_name}: acceptance failed")
        text = (run_dir / "train.log").read_text(errors="replace")
        final_match = FINAL_VAL.search(text)
        final_val = float(final_match.group(1)) if final_match else None

        init_kwargs = {
            "project": args.project,
            "experiment_name": f"stage03-{run_name}-backfill",
            "description": "Historical replay of Stage03-0 seed-gate console metrics; no live system telemetry.",
            "job_type": "historical-backfill",
            "group": "stage03-seed-gate",
            "tags": ["Search-R1", "Stage03", "GRPO", "seed-gate", "backfill"],
            "config": {
                "backfill": True,
                "seed": acceptance["seed"],
                "updates": acceptance["max_training_step"],
                "base_model": "Qwen2.5-3B",
                "train_examples": 64,
                "validation_examples": 32,
                "gpus": 8,
                "training_trajectory_proxy_sha256": acceptance[
                    "training_trajectory_proxy_sha256"
                ],
                "source_log": str(run_dir / "train.log"),
            },
            "logdir": str(logdir),
            "mode": args.mode,
            "reinit": True,
        }
        if args.workspace:
            init_kwargs["workspace"] = args.workspace
        run = swanlab.init(**init_kwargs)
        for step, metrics in sorted(
            acceptance["training_metrics"].items(), key=lambda item: int(item[0])
        ):
            run.log(finite_metrics(metrics), step=int(step))
        final_step = acceptance["max_training_step"] + 1
        final_metrics = {"acceptance/pass": 1.0}
        if final_val is not None:
            final_metrics["val/test_score/nq"] = final_val
        run.log(final_metrics, step=final_step)
        swanlab.finish()
        manifest["runs"].append(
            {
                "run_name": run_name,
                "seed": acceptance["seed"],
                "swanlab_run_id": getattr(run, "id", None),
                "final_val_em": final_val,
            }
        )

    Path(args.manifest).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
