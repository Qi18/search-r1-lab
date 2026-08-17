from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--same-a", required=True)
    parser.add_argument("--same-b", required=True)
    parser.add_argument("--different", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    a, b, different = map(load, (args.same_a, args.same_b, args.different))
    if a["seed"] != b["seed"] or a["seed"] == different["seed"]:
        raise SystemExit("invalid seed grouping")

    key = "training_trajectory_proxy_sha256"
    same_trajectory = a[key] == b[key]
    different_trajectory = a[key] != different[key]
    same_metrics = a["training_metrics"] == b["training_metrics"]
    same_validation = (
        a["post_training_validation_trajectory_proxy_sha256"]
        == b["post_training_validation_trajectory_proxy_sha256"]
    )
    accepted = same_trajectory and different_trajectory
    result = {
        "accepted": accepted,
        "same_seed": a["seed"],
        "different_seed": different["seed"],
        "same_seed_trajectory_match": same_trajectory,
        "same_seed_metrics_exact_match": same_metrics,
        "same_seed_post_training_validation_match": same_validation,
        "different_seed_trajectory_changed": different_trajectory,
        "runs": {"same_a": a, "same_b": b, "different": different},
        "metric_note": "The hard gate covers training-time retrieval trajectories. GPU update kernels may prevent bitwise metric and post-training validation identity.",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not accepted:
        raise SystemExit("Stage03 seed gate failed")


if __name__ == "__main__":
    main()
