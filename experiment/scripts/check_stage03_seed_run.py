from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"step:(\d+) - (.*)")
METRIC = re.compile(r"([\w/]+):(-?\d+(?:\.\d+)?)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--training-request-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    text = ANSI.sub("", Path(args.log).read_text(errors="replace"))
    if f"stage03_controller_seed={args.seed}" not in text:
        raise SystemExit("seed was not observed by the controller")
    if f"stage03_seed={args.seed}" not in text:
        raise SystemExit("seed was not observed by rollout workers")

    steps = {}
    for match in STEP.finditer(text):
        step = int(match.group(1))
        metrics = {key: float(value) for key, value in METRIC.findall(match.group(2))}
        if "critic/score/mean" in metrics:
            steps[str(step)] = metrics
    if str(args.expected_step) not in steps:
        raise SystemExit(f"missing training step {args.expected_step}")

    records = [json.loads(line) for line in Path(args.requests).read_text().splitlines() if line]
    canonical = [
        {"queries": row["queries"], "topk": row["topk"], "result_ids": row["result_ids"]}
        for row in records
    ]
    if len(canonical) < args.training_request_count:
        raise SystemExit("retrieval log is shorter than the training request boundary")
    training_records = canonical[: args.training_request_count]
    validation_records = canonical[args.training_request_count :]

    def digest(rows: list[dict]) -> str:
        return hashlib.sha256(
            json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    output = {
        "accepted": True,
        "seed": args.seed,
        "max_training_step": max(map(int, steps)),
        "training_metrics": steps,
        "retrieval_request_count": len(records),
        "training_request_count": len(training_records),
        "training_trajectory_proxy_sha256": digest(training_records),
        "post_training_validation_request_count": len(validation_records),
        "post_training_validation_trajectory_proxy_sha256": digest(validation_records),
        "full_trajectory_proxy_sha256": digest(canonical),
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
