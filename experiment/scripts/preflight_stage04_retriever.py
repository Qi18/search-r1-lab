from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text).split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.validation)
    hit1 = hit3 = 0
    latencies = []
    examples = []
    for start in range(0, len(frame), args.batch_size):
        batch = frame.iloc[start : start + args.batch_size]
        began = time.perf_counter()
        response = requests.post(
            args.url,
            json={"queries": batch["question"].tolist(), "topk": 3, "return_scores": True},
            timeout=600,
        )
        response.raise_for_status()
        rows = response.json()["result"]
        elapsed = time.perf_counter() - began
        latencies.extend([elapsed / len(batch)] * len(batch))
        for (_, source), retrieved in zip(batch.iterrows(), rows):
            answers = [normalize(str(answer)) for answer in source["golden_answers"]]
            docs = [normalize(json.dumps(item["document"], ensure_ascii=False)) for item in retrieved]
            flags = [any(answer and answer in doc for answer in answers) for doc in docs]
            hit1 += int(bool(flags and flags[0]))
            hit3 += int(any(flags[:3]))
            examples.append({"id": source["id"], "hit1": bool(flags and flags[0]), "hit3": any(flags[:3])})

    count = len(frame)
    payload = {
        "status": "PASS" if count > 0 and hit3 > 0 else "FAIL",
        "queries": count,
        "hit_at_1": hit1 / count,
        "hit_at_3": hit3 / count,
        "latency_seconds": {
            "mean": statistics.mean(latencies),
            "p95": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
        },
        "examples": examples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "examples"}, indent=2))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
