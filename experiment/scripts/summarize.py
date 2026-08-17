from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from search_r1_lab.io import read_jsonl
from search_r1_lab.metrics import compute_metrics


def percentage(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.results)
    metrics = compute_metrics(rows)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": str(Path(args.results).resolve()),
        "metrics": metrics,
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Experiment 00: Search-R1 inference smoke test",
        "",
        f"- Trajectories: `{Path(args.results).resolve()}`",
        f"- Generated: `{payload['generated_at']}`",
        "",
        "| Mode | EM | Contains | F1 | Valid answer | Generated search | Retriever request | Hit@k | Avg turns | Avg latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("no-search", "search"):
        if mode not in metrics:
            continue
        item = metrics[mode]
        lines.append(
            f"| {mode} | {percentage(item['exact_match'])} | "
            f"{percentage(item['answer_contains'])} | "
            f"{percentage(item['token_f1'])} | {percentage(item['valid_answer_rate'])} | "
            f"{percentage(item['generated_search_tag_rate'])} | "
            f"{percentage(item['retriever_request_rate'])} | "
            f"{percentage(item['retrieval_hit_rate'])} | "
            f"{item['avg_search_turns']:.2f} | {item['avg_latency_seconds']:.2f}s |"
        )

    lines.extend(
        [
            "",
            "## Per-question outcomes",
            "",
            "| Mode | ID | Expected | Prediction | Generated | Requests |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        prediction = row["prediction"].replace("|", "\\|") or "(missing)"
        lines.append(
            f"| {row['mode']} | {row['id']} | {row['answer']} | "
            f"{prediction} | {row['generated_search_count']} | "
            f"{row['retriever_request_count']} |"
        )
    lines.append("")
    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote report: {markdown_path}")


if __name__ == "__main__":
    main()
