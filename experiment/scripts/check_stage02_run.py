from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


METRIC_LINE = re.compile(r"step:(?P<step>\d+)(?P<body>.*)")
METRIC = re.compile(r" - (?P<key>[^:]+):(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)", re.IGNORECASE)
FATAL_PATTERNS = (
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"raytaskerror", re.IGNORECASE),
    re.compile(r"(?<![a-z0-9_])(?:nan|inf)(?![a-z0-9_])", re.IGNORECASE),
)
REQUIRED_METRICS = (
    "actor/pg_loss",
    "actor/grad_norm",
    "actor/kl_loss",
    "actor/ppo_kl",
    "critic/score/mean",
    "critic/advantages/max",
    "critic/advantages/min",
)


def parse_metrics(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        match = METRIC_LINE.search(line)
        if not match:
            continue
        row: dict[str, float | int] = {"step": int(match.group("step"))}
        for metric in METRIC.finditer(match.group("body")):
            row[metric.group("key")] = float(metric.group("value"))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    checkpoint = Path(args.checkpoint)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metric_rows = parse_metrics(text)
    training_rows = [row for row in metric_rows if "actor/pg_loss" in row]
    last = training_rows[-1] if training_rows else {}
    lower = text.lower()

    checks = {
        "log_exists": log_path.is_file(),
        "optimizer_step_logged": bool(training_rows),
        "expected_step_logged": any(row.get("step") == args.expected_step for row in training_rows),
        "required_metrics_present": all(key in last for key in REQUIRED_METRICS),
        "finite_metrics": all(math.isfinite(float(last[key])) for key in REQUIRED_METRICS if key in last),
        "reward_variation": any(
            float(row.get("critic/score/max", 0.0)) > float(row.get("critic/score/min", 0.0))
            for row in training_rows
        ),
        "advantage_variation": any(
            float(row.get("critic/advantages/max", 0.0)) > 0.0
            and float(row.get("critic/advantages/min", 0.0)) < 0.0
            for row in training_rows
        ),
        "search_observed": "<search>" in lower,
        "checkpoint_config": (checkpoint / "config.json").is_file(),
        "checkpoint_weights": any(checkpoint.glob("*.safetensors")),
        "checkpoint_tokenizer": (checkpoint / "tokenizer_config.json").is_file(),
        "no_fatal_pattern": not any(pattern.search(text) for pattern in FATAL_PATTERNS),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "expected_step": args.expected_step,
        "checks": checks,
        "last_training_metrics": last,
        "metric_rows": len(metric_rows),
        "training_rows": len(training_rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
