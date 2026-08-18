from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import swanlab


METRIC_LINE = re.compile(r"step:(?P<step>\d+)(?P<body>.*)")
METRIC = re.compile(r" - (?P<key>[^:]+):(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)", re.I)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def read_pid(path: Path) -> int:
    while not path.is_file():
        time.sleep(2)
    return int(path.read_text().strip())


def parse(line: str) -> tuple[int, dict[str, float]] | None:
    match = METRIC_LINE.search(line)
    if not match:
        return None
    metrics = {}
    for item in METRIC.finditer(match.group("body")):
        value = float(item.group("value"))
        if math.isfinite(value):
            metrics[item.group("key")] = value
    return int(match.group("step")), metrics


def follow_algorithm(root: Path, algorithm: str, supervisor_pid: int, logdir: Path) -> str:
    run_dir = root / f"{algorithm}-305step"
    log_path = run_dir / "train.log"
    acceptance = run_dir / "acceptance.json"
    while not log_path.is_file():
        if not alive(supervisor_pid):
            raise RuntimeError(f"supervisor exited before {algorithm} started")
        time.sleep(5)
    run = swanlab.init(
        project="Search-R1",
        experiment_name=f"stage05-{algorithm}-305step-live",
        description="Live sidecar metrics parsed from the formal Stage05 veRL console log.",
        group="stage05-official-v01-formal",
        tags=["Search-R1", "Stage05", algorithm.upper(), "live", "sidecar"],
        logdir=str(logdir / algorithm),
        mode="online",
        reinit=True,
        config={
            "algorithm": algorithm.upper(),
            "updates": 305,
            "train_batch_size": 32,
            "n_agent": 1 if algorithm == "ppo" else 5,
            "max_turns": 4,
            "topk": 3,
            "learning_rate": 1e-6,
            "warmup_ratio": 0.95,
            "state_masking": True,
            "gpus": 8,
            "telemetry": "live console-log sidecar",
        },
    )
    seen: set[int] = set()
    position = 0
    idle_after_acceptance = 0
    while True:
        with log_path.open(encoding="utf-8", errors="replace") as handle:
            handle.seek(position)
            for line in handle:
                parsed = parse(line)
                if parsed is None:
                    continue
                step, metrics = parsed
                if "actor/pg_loss" in metrics and step not in seen:
                    run.log(metrics, step=step)
                    seen.add(step)
            position = handle.tell()
        if acceptance.is_file():
            payload = json.loads(acceptance.read_text())
            run.log({"acceptance/pass": float(payload.get("status") == "PASS")}, step=305)
            idle_after_acceptance += 1
            if idle_after_acceptance >= 2:
                break
        elif not alive(supervisor_pid):
            raise RuntimeError(f"supervisor exited before {algorithm} acceptance")
        time.sleep(5)
    swanlab.finish()
    return str(getattr(run, "id", ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    pid = read_pid(Path(args.pid_file))
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    runs = {algorithm: follow_algorithm(root, algorithm, pid, logdir) for algorithm in ("ppo", "grpo")}
    Path(args.manifest).write_text(json.dumps(runs, indent=2) + "\n")


if __name__ == "__main__":
    main()
