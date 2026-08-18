from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


METRIC_LINE = re.compile(r"step:(?P<step>\d+)(?P<body>.*)")
METRIC = re.compile(r" - (?P<key>[^:]+):(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)", re.I)
FATAL = ("out of memory", "traceback (most recent call last)", "raytaskerror")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("ppo", "grpo"), required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--critic")
    parser.add_argument("--gpu-csv", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        match = METRIC_LINE.search(line)
        if not match:
            continue
        row = {"step": int(match.group("step"))}
        row.update({item.group("key"): float(item.group("value")) for item in METRIC.finditer(match.group("body"))})
        if "actor/pg_loss" in row:
            rows.append(row)
    last = next((row for row in reversed(rows) if row["step"] == args.step), {})
    common = ("actor/pg_loss", "actor/ppo_kl", "actor/grad_norm", "critic/score/mean", "timing_s/step")
    algorithm_metrics = ("critic/vf_loss", "critic/grad_norm") if args.algorithm == "ppo" else ("actor/kl_loss",)
    required = common + algorithm_metrics
    actor = Path(args.actor)
    peak = 0
    with Path(args.gpu_csv).open(encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            values = [item.strip() for item in row]
            if len(values) >= 3 and values[2].isdigit():
                peak = max(peak, int(values[2]))
    checks = {
        "expected_step_logged": bool(last),
        "algorithm_metrics": all(key in last for key in required),
        "finite_metrics": all(math.isfinite(float(last[key])) for key in required if key in last),
        "search_observed": float(last.get("env/number_of_valid_search", 0)) > 0,
        "state_masking_observed": float(last.get("state_tokens/coverage", 0)) > 0,
        "actor_checkpoint": (actor / "config.json").is_file() and any(actor.glob("*.safetensors")),
        "no_fatal_pattern": not any(pattern in text.lower() for pattern in FATAL),
    }
    if args.algorithm == "ppo":
        critic = Path(args.critic or "")
        checks["critic_checkpoint"] = any(critic.rglob("*.pt")) or any(critic.rglob("*.safetensors"))
    else:
        checks["critic_not_required"] = args.critic is None
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "algorithm": args.algorithm,
        "step": args.step,
        "checks": checks,
        "last_training_metrics": last,
        "peak_gpu_memory_mib": peak or None,
        "training_rows": len(rows),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
