from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from search_r1_lab.io import read_jsonl
from search_r1_lab.metrics import compute_metrics, contains_answer, exact_match, retrieval_hit
from search_r1_lab.protocol import has_valid_answer

QUADRANTS = (
    ("qwen-base-pre", "no-search"),
    ("qwen-base-pre", "search"),
    ("tiny-grpo-20step-post", "no-search"),
    ("tiny-grpo-20step-post", "search"),
)
NAMES = {"qwen-base-pre": "Base (pre)", "tiny-grpo-20step-post": "Tiny GRPO (20-step)"}


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def points(value: float) -> str:
    return f"{100 * value:+.1f} pp"


def outcome(row: dict) -> str:
    if not has_valid_answer(row["trajectory"]):
        return "invalid_format"
    if exact_match(row["prediction"], row["answer"]):
        return "exact"
    if contains_answer(row["prediction"], row["answer"]):
        return "contains_only"
    if row["retriever_request_count"] and not retrieval_hit(row, 3):
        return "retrieval_miss"
    return "incorrect"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-results", required=True)
    parser.add_argument("--post-results", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--retriever-preflight", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.pre_results) + read_jsonl(args.post_results)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_label"], row["mode"])].append(row)

    count = len(grouped[QUADRANTS[0]])
    expected_ids = {row["id"] for row in grouped[QUADRANTS[0]]}
    metrics = {}
    outcomes = {}
    for quadrant in QUADRANTS:
        label, mode = quadrant
        metrics[f"{label}/{mode}"] = compute_metrics(grouped[quadrant])[mode]
        outcomes[f"{label}/{mode}"] = dict(Counter(outcome(row) for row in grouped[quadrant]))

    run_root = Path(args.run_root)
    run_acceptance = {}
    for step in (1, 5, 20):
        path = run_root / f"02-tiny-grpo-{step}step" / "acceptance.json"
        run_acceptance[str(step)] = json.loads(path.read_text()) if path.is_file() else {"status": "MISSING"}
    preflight = json.loads(Path(args.retriever_preflight).read_text())
    before = metrics["qwen-base-pre/search"]
    after = metrics["tiny-grpo-20step-post/search"]
    changes = {
        key: after[key] - before[key]
        for key in ("exact_match", "answer_contains", "token_f1", "retriever_request_rate")
    }
    required_fields = {
        "id", "question", "answer", "prediction", "trajectory", "search_events",
        "retriever_request_count", "generated_search_count", "latency_seconds",
    }
    checks = {
        "32_examples_per_quadrant": count == 32 and all(len(grouped[q]) == count for q in QUADRANTS),
        "identical_question_ids": all({row["id"] for row in grouped[q]} == expected_ids for q in QUADRANTS),
        "required_fields": all(required_fields <= row.keys() for row in rows),
        "finite_metrics": all(math.isfinite(v) for item in metrics.values() for v in item.values()),
        "no_search_has_no_requests": all(row["retriever_request_count"] == 0 for row in rows if row["mode"] == "no-search"),
        "retriever_preflight": bool(preflight.get("passed")),
        "all_training_runs_pass": all(item.get("status") == "PASS" for item in run_acceptance.values()),
        "checkpoint_reloaded": len(grouped[("tiny-grpo-20step-post", "search")]) == 32,
    }
    accepted = all(checks.values())
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "search_mode_changes": changes,
        "outcomes": outcomes,
        "training_acceptance": run_acceptance,
        "retriever_preflight": preflight,
        "acceptance": {"passed": accepted, "checks": checks},
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Experiment 02: Tiny GRPO",
        "",
        "在 8 x NVIDIA L20 上，从 Qwen2.5-3B Base 使用 Search-R1 官方 veRL/GRPO 代码完成 1、5、20 个优化步。",
        "",
        "## 冻结验证集结果",
        "",
        "| Model | Mode | EM | Contains | F1 | Valid | Request | Hit@1 | Hit@3 | Turns | Latency |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, mode in QUADRANTS:
        item = metrics[f"{label}/{mode}"]
        lines.append(
            f"| {NAMES[label]} | {mode} | {pct(item['exact_match'])} | "
            f"{pct(item['answer_contains'])} | {pct(item['token_f1'])} | "
            f"{pct(item['valid_answer_rate'])} | {pct(item['retriever_request_rate'])} | "
            f"{pct(item['retrieval_hit_at_1'])} | {pct(item['retrieval_hit_at_3'])} | "
            f"{item['avg_search_turns']:.2f} | {item['avg_latency_seconds']:.2f}s |"
        )

    lines.extend(["", "## Search 模式训练前后变化", ""])
    for key, value in changes.items():
        lines.append(f"- {key}: {points(value)}")
    lines.extend(["", "## 训练运行验收", "", "| Updates | Status | Last reward | KL | PG loss |", "| ---: | --- | ---: | ---: | ---: |"])
    for step in (1, 5, 20):
        acceptance = run_acceptance[str(step)]
        last = acceptance.get("last_training_metrics", {})
        lines.append(
            f"| {step} | {acceptance.get('status')} | {last.get('critic/score/mean', 'n/a')} | "
            f"{last.get('actor/ppo_kl', 'n/a')} | {last.get('actor/pg_loss', 'n/a')} |"
        )

    lines.extend([
        "", "## 结论", "",
        "- PASS 表示官方 GRPO 数据流、工具调用、梯度更新、checkpoint 保存和回载评测全部闭环；不以指标必须提升作为工程验收条件。",
        "- 训练前后差异只适用于这 32 条训练集外合成问题；20 步单随机种子不能代表真实数据泛化结论。",
        "- Retriever preflight 单独验证索引质量，最终指标还同时受模型是否调用搜索、是否利用证据和答案格式影响。",
        "", "## 最终验收", "",
        f"- Status: {'PASS' if accepted else 'FAIL'}",
    ])
    for key, value in checks.items():
        lines.append(f"- [{'x' if value else ' '}] {key}")
    lines.extend(["", "## 结果类型", "", "| Model | Mode | Exact | Contains only | Incorrect | Invalid format | Retrieval miss |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for label, mode in QUADRANTS:
        item = outcomes[f"{label}/{mode}"]
        lines.append(
            f"| {NAMES[label]} | {mode} | {item.get('exact', 0)} | {item.get('contains_only', 0)} | "
            f"{item.get('incorrect', 0)} | {item.get('invalid_format', 0)} | {item.get('retrieval_miss', 0)} |"
        )
    lines.append("")
    report = Path(args.markdown)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print(json.dumps(payload, indent=2))
    print(f"wrote report: {report}")
    if args.require_pass and not accepted:
        raise SystemExit("Stage 02 acceptance failed")


if __name__ == "__main__":
    main()
