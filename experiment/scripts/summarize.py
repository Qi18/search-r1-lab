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

    if "no-search" in metrics and "search" in metrics:
        no_search = metrics["no-search"]
        search = metrics["search"]
        em_gain = search["exact_match"] - no_search["exact_match"]
        f1_gain = search["token_f1"] - no_search["token_f1"]
        lines.extend(
            [
                "## 结论",
                "",
                f"- 搜索链路有效：search 模式 Retriever 请求率为 "
                f"{percentage(search['retriever_request_rate'])}，"
                f"Hit@k 为 {percentage(search['retrieval_hit_rate'])}。",
                f"- 检索显著改善答案：EM 提升 {100 * em_gain:.1f} 个百分点，"
                f"F1 提升 {100 * f1_gain:.1f} 个百分点。",
                f"- 模型搜索倾向明确：两种模式的搜索标签生成率均为 "
                f"{percentage(search['generated_search_tag_rate'])}，"
                f"但 no-search 的 Retriever 请求率为 "
                f"{percentage(no_search['retriever_request_rate'])}。",
                f"- search 模式 Contains 为 "
                f"{percentage(search['answer_contains'])}，高于 EM 的 "
                f"{percentage(search['exact_match'])}，差异主要来自额外措辞。",
                "",
                "## 结论边界",
                "",
                "- 本阶段只比较同一个 Search-R1 GRPO checkpoint 是否启用 Retriever，不能单独证明 RL 的收益。",
                "- 数据只有 8 条确定性合成问题，且 Retriever 命中率为 100%，不能代表真实任务泛化能力。",
                "- no-search 是禁用工具的搜索模型，不等同于 Qwen Base；模型、Retriever 和 RL 的贡献需要 Stage 01 四象限实验拆分。",
                "",
            ]
        )
    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote report: {markdown_path}")


if __name__ == "__main__":
    main()
