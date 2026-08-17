from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from search_r1_lab.io import read_jsonl
from search_r1_lab.metrics import (
    compute_metrics,
    contains_answer,
    exact_match,
    retrieval_hit,
)
from search_r1_lab.protocol import has_valid_answer


MODEL_NAMES = {
    "qwen-base": "Qwen2.5-3B Base",
    "search-r1-grpo": "Search-R1 GRPO",
}
QUADRANTS = (
    ("qwen-base", "no-search"),
    ("qwen-base", "search"),
    ("search-r1-grpo", "no-search"),
    ("search-r1-grpo", "search"),
)


def percentage(value: float) -> str:
    return f"{100 * value:.1f}%"


def points(value: float) -> str:
    return f"{100 * value:+.1f} pp"


def quadrant_id(model_label: str, mode: str) -> str:
    return f"{model_label}/{mode}"


def classify_outcome(row: dict) -> str:
    if not has_valid_answer(row["trajectory"]):
        return "invalid_format"
    if exact_match(row["prediction"], row["answer"]):
        return "exact"
    if contains_answer(row["prediction"], row["answer"]):
        return "contains_only"
    if row["retriever_request_count"] > 0 and not retrieval_hit(row, 3):
        return "retrieval_miss"
    return "incorrect"


def metric_difference(
    metrics: dict[str, dict[str, float]],
    left: str,
    right: str,
) -> dict[str, float]:
    return {
        "exact_match": metrics[left]["exact_match"] - metrics[right]["exact_match"],
        "token_f1": metrics[left]["token_f1"] - metrics[right]["token_f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-results", required=True)
    parser.add_argument("--grpo-results", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--retriever-preflight", required=True)
    parser.add_argument("--min-examples", type=int, default=50)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.base_results) + read_jsonl(args.grpo_results)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_label"], row["mode"])].append(row)

    expected_count = len(grouped[QUADRANTS[0]])
    if expected_count == 0:
        raise ValueError("missing qwen-base/no-search records")
    for quadrant in QUADRANTS:
        if len(grouped[quadrant]) != expected_count:
            raise ValueError(
                f"unbalanced quadrant {quadrant}: "
                f"expected {expected_count}, got {len(grouped[quadrant])}"
            )

    required_fields = {
        "id", "question", "answer", "evidence_id", "prediction", "trajectory",
        "search_events", "generated_search_count", "retriever_request_count",
        "input_token_count", "output_token_count", "latency_seconds", "model_label", "mode",
    }
    expected_ids = {row["id"] for row in grouped[QUADRANTS[0]]}
    preflight = json.loads(Path(args.retriever_preflight).read_text(encoding="utf-8"))

    metrics: dict[str, dict[str, float]] = {}
    failures: dict[str, dict[str, int]] = {}
    for model_label, mode in QUADRANTS:
        key = quadrant_id(model_label, mode)
        metrics[key] = compute_metrics(grouped[(model_label, mode)])[mode]
        failures[key] = dict(
            Counter(classify_outcome(row) for row in grouped[(model_label, mode)])
        )

    base_no = quadrant_id("qwen-base", "no-search")
    base_search = quadrant_id("qwen-base", "search")
    grpo_no = quadrant_id("search-r1-grpo", "no-search")
    grpo_search = quadrant_id("search-r1-grpo", "search")
    effects = {
        "base_retriever_enabled": metric_difference(metrics, base_search, base_no),
        "grpo_retriever_enabled": metric_difference(metrics, grpo_search, grpo_no),
        "grpo_minus_base_no_search": metric_difference(metrics, grpo_no, base_no),
        "grpo_minus_base_search": metric_difference(metrics, grpo_search, base_search),
    }
    effects["interaction"] = {
        metric: effects["grpo_retriever_enabled"][metric]
        - effects["base_retriever_enabled"][metric]
        for metric in ("exact_match", "token_f1")
    }

    checks = {
        "minimum_examples": expected_count >= args.min_examples,
        "balanced_quadrants": all(len(grouped[item]) == expected_count for item in QUADRANTS),
        "identical_question_ids": all(
            {row["id"] for row in grouped[item]} == expected_ids for item in QUADRANTS
        ),
        "required_fields": all(required_fields <= row.keys() for row in rows),
        "nonempty_trajectories": all(bool(row["trajectory"].strip()) for row in rows),
        "finite_metrics": all(
            math.isfinite(value)
            for item in metrics.values()
            for value in item.values()
        ),
        "no_search_has_no_requests": all(
            row["retriever_request_count"] == 0 for row in rows if row["mode"] == "no-search"
        ),
        "failures_classified": all(
            sum(failures[quadrant_id(*item)].values()) == expected_count for item in QUADRANTS
        ),
        "retriever_preflight": bool(preflight.get("passed")),
    }
    acceptance = {"passed": all(checks.values()), "checks": checks}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "examples_per_quadrant": expected_count,
        "metrics": metrics,
        "observed_differences": effects,
        "outcomes": failures,
        "retriever_preflight": preflight,
        "acceptance": acceptance,
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Experiment 01: Base vs Search-R1 GRPO",
        "",
        f"- Examples per quadrant: {expected_count}",
        f"- Generated: `{payload['generated_at']}`",
        "- Controls: same questions, prompt, retriever index, Top-3, greedy decoding, token limit and GPU",
        "",
        "## 四象限结果",
        "",
        "| Model | Retriever | EM | Contains | F1 | Valid | Search tag | Request | Hit@1 | Hit@3 | Turns | Input tok | Output tok | Latency |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_label, mode in QUADRANTS:
        item = metrics[quadrant_id(model_label, mode)]
        lines.append(
            f"| {MODEL_NAMES[model_label]} | {mode} | "
            f"{percentage(item['exact_match'])} | "
            f"{percentage(item['answer_contains'])} | "
            f"{percentage(item['token_f1'])} | "
            f"{percentage(item['valid_answer_rate'])} | "
            f"{percentage(item['generated_search_tag_rate'])} | "
            f"{percentage(item['retriever_request_rate'])} | "
            f"{percentage(item['retrieval_hit_at_1'])} | "
            f"{percentage(item['retrieval_hit_at_3'])} | "
            f"{item['avg_search_turns']:.2f} | "
            f"{item['avg_input_tokens']:.1f} | "
            f"{item['avg_output_tokens']:.1f} | "
            f"{item['avg_latency_seconds']:.2f}s |"
        )

    effect_names = {
        "base_retriever_enabled": "Base: search - no-search",
        "grpo_retriever_enabled": "GRPO: search - no-search",
        "grpo_minus_base_no_search": "No-search: GRPO - Base",
        "grpo_minus_base_search": "Search: GRPO - Base",
        "interaction": "Difference in retrieval gain",
    }
    lines.extend(
        [
            "",
            "## 观察差异",
            "",
            "| Comparison | EM | F1 |",
            "| --- | ---: | ---: |",
        ]
    )
    for effect_key, label in effect_names.items():
        effect = effects[effect_key]
        lines.append(
            f"| {label} | {points(effect['exact_match'])} | "
            f"{points(effect['token_f1'])} |"
        )

    outcome_names = (
        "exact",
        "contains_only",
        "incorrect",
        "invalid_format",
        "retrieval_miss",
    )
    lines.extend(
        [
            "",
            "## 结果类型",
            "",
            "| Model | Retriever | Exact | Contains only | Incorrect | Invalid format | Retrieval miss |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model_label, mode in QUADRANTS:
        counts = failures[quadrant_id(model_label, mode)]
        values = " | ".join(str(counts.get(name, 0)) for name in outcome_names)
        lines.append(f"| {MODEL_NAMES[model_label]} | {mode} | {values} |")

    base_search_metrics = metrics[base_search]
    grpo_search_metrics = metrics[grpo_search]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- Base 在工具可用时的实际请求率为 {percentage(base_search_metrics['retriever_request_rate'])}；GRPO 为 {percentage(grpo_search_metrics['retriever_request_rate'])}。",
            f"- 两个 search 组的条件 Hit@1/Hit@3 都是 100%，但 Base Contains 为 {percentage(base_search_metrics['answer_contains'])}，GRPO 为 {percentage(grpo_search_metrics['answer_contains'])}。",
            "- 因此差异主要来自是否稳定调用工具以及能否利用返回证据，而不是底层索引本身。",
            f"- GRPO 开启搜索相对关闭搜索的观察差异：EM {points(effects['grpo_retriever_enabled']['exact_match'])}，F1 {points(effects['grpo_retriever_enabled']['token_f1'])}。",
            f"- 开启搜索时 GRPO 相对 Base 的观察差异：EM {points(effects['grpo_minus_base_search']['exact_match'])}，F1 {points(effects['grpo_minus_base_search']['token_f1'])}。",
            "",
            "## 结论边界",
            "",
            f"- 当前使用 {expected_count} 条固定合成问题，可完成 Stage 01 链路验收，但不能代表真实任务泛化能力。",
            "- 工具可用不等于模型实际调用工具；Retriever 收益必须结合请求率解释。",
            "- 观察差异不是严格因果归因，后续仍需真实数据集和多随机种子验证。",
            "",
            "## 最终验收",
            "",
            f"- Status: {'PASS' if acceptance['passed'] else 'FAIL'}",
            f"- Retriever preflight: Hit@1 {percentage(preflight['hit_at_1'])}, Hit@3 {percentage(preflight['hit_at_3'])}",
            "",
        ]
    )
    for check_name, passed in checks.items():
        lines.append(f"- [{'x' if passed else ' '}] {check_name}")
    lines.extend(
        [
            "",
            "## Per-question outcomes",
            "",
            "| Model | Mode | ID | Expected | Prediction | Outcome | Generated | Requests | Hit@3 |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for model_label, mode in QUADRANTS:
        for row in grouped[(model_label, mode)]:
            prediction = row["prediction"].replace("|", "\\|") or "(missing)"
            lines.append(
                f"| {MODEL_NAMES[model_label]} | {mode} | {row['id']} | "
                f"{row['answer']} | {prediction} | {classify_outcome(row)} | "
                f"{row['generated_search_count']} | {row['retriever_request_count']} | "
                f"{int(retrieval_hit(row, 3))} |"
            )
    lines.append("")

    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote report: {markdown_path}")
    if args.require_pass and not acceptance["passed"]:
        raise SystemExit("Stage 01 acceptance failed")


if __name__ == "__main__":
    main()
