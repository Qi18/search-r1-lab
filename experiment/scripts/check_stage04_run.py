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
    parser.add_argument("--log", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--critic", required=True)
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
        row.update({m.group("key"): float(m.group("value")) for m in METRIC.finditer(match.group("body"))})
        if "actor/pg_loss" in row:
            rows.append(row)
    last = next((row for row in reversed(rows) if row["step"] == args.step), {})
    required = ("actor/pg_loss", "actor/ppo_kl", "critic/vf_loss", "critic/grad_norm", "critic/score/mean", "timing_s/step")
    actor = Path(args.actor)
    critic = Path(args.critic)
    peaks = []
    with Path(args.gpu_csv).open(encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            for value in row:
                value = value.strip()
                if value.isdigit():
                    peaks.append(int(value))
    checks = {
        "expected_step_logged": bool(last),
        "all_required_metrics": all(key in last for key in required),
        "ppo_actor_metrics": all(key in last for key in ("actor/pg_loss", "actor/ppo_kl")),
        "ppo_critic_metrics": all(key in last for key in ("critic/vf_loss", "critic/grad_norm")),
        "finite_metrics": all(math.isfinite(float(last[key])) for key in required if key in last),
        "search_observed": float(last.get("env/number_of_valid_search", 0)) > 0,
        "actor_checkpoint": (actor / "config.json").is_file() and any(actor.glob("*.safetensors")),
        "critic_checkpoint": any(critic.rglob("*.pt")) or any(critic.rglob("*.safetensors")),
        "no_fatal_pattern": not any(pattern in text.lower() for pattern in FATAL),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "step": args.step,
        "checks": checks,
        "last_training_metrics": last,
        "peak_gpu_memory_mib": max(peaks) if peaks else None,
        "training_rows": len(rows),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
