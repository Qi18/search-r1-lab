from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SOURCES = ("nq", "hotpotqa", "triviaqa", "popqa", "2wikimultihopqa", "musique", "bamboogle")
LABELS = {"baseline": "Base", "ppo-1step": "PPO 1-step", "grpo-1step": "GRPO 1-step"}


def normalize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def f1(prediction: str, answers: list[str]) -> float:
    pred = normalize(prediction)
    best = 0.0
    for answer in answers:
        gold = normalize(str(answer))
        common = sum((Counter(pred) & Counter(gold)).values())
        if not pred or not gold or not common:
            score = float(pred == gold)
        else:
            precision, recall = common / len(pred), common / len(gold)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def aggregate(rows: list[dict]) -> dict:
    return {
        "examples": len(rows),
        "em": sum(row["reward"] for row in rows) / len(rows),
        "token_f1": sum(f1(row["prediction"], row["golden_answers"]) for row in rows) / len(rows),
        "search_rate": sum(row["search_count"] > 0 for row in rows) / len(rows),
        "avg_searches": sum(row["search_count"] for row in rows) / len(rows),
        "valid_answer_rate": sum(bool(row["prediction"]) for row in rows) / len(rows),
    }


def evaluate(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["data_source"]].append(row)
    return {"overall": aggregate(rows), "by_source": {source: aggregate(grouped[source]) for source in SOURCES}}


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    evaluations = {name: evaluate(root / "eval" / name / "trajectories.jsonl") for name in LABELS}
    training = {name: json.loads((root / f"{name}-1step" / "acceptance.json").read_text()) for name in ("ppo", "grpo")}
    id_sets = [{json.loads(line)["id"] for line in (root / "eval" / name / "trajectories.jsonl").read_text().splitlines() if line} for name in LABELS]
    checks = {
        "seven_dataset_validation": all(set(item["by_source"]) == set(SOURCES) and item["overall"]["examples"] == 112 for item in evaluations.values()),
        "same_validation_ids": len(id_sets[0]) == 112 and all(ids == id_sets[0] for ids in id_sets[1:]),
        "ppo_training": training["ppo"]["status"] == "PASS",
        "grpo_training": training["grpo"]["status"] == "PASS",
        "checkpoint_reload": all(item["overall"]["examples"] == 112 for item in evaluations.values()),
    }
    accepted = all(checks.values())
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if accepted else "FAIL",
        "evaluations": evaluations,
        "training": training,
        "checks": checks,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Experiment 05: Official v0.1 Multi-dataset PPO/GRPO",
        "",
        "在官方 NQ + HotpotQA 训练格式和七数据集验证格式上，完成 PPO/GRPO 短程训练、checkpoint 与独立回载评测闭环。",
        "",
        "## 总体结果",
        "",
        "| Model | EM | Token F1 | Search rate | Avg searches | Valid answer |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in LABELS.items():
        item = evaluations[name]["overall"]
        lines.append(f"| {label} | {pct(item['em'])} | {pct(item['token_f1'])} | {pct(item['search_rate'])} | {item['avg_searches']:.2f} | {pct(item['valid_answer_rate'])} |")
    lines.extend(["", "## 七数据集 EM", "", "| Dataset | Base | PPO 1-step | GRPO 1-step |", "| --- | ---: | ---: | ---: |"])
    for source in SOURCES:
        values = [pct(evaluations[name]["by_source"][source]["em"]) for name in LABELS]
        lines.append(f"| {source} | {' | '.join(values)} |")
    lines.extend(["", "## 训练验收", ""])
    for name in ("ppo", "grpo"):
        item = training[name]
        metric = item["last_training_metrics"]
        lines.append(f"- {name.upper()}: {item['status']}，reward={metric.get('critic/score/mean', 0):.4f}，step={metric.get('timing_s/step', 0):.1f}s，peak={item.get('peak_gpu_memory_mib')} MiB。")
    lines.extend(["", "## 验收结论", "", f"- Status: {'PASS' if accepted else 'FAIL'}"])
    for key, value in checks.items():
        lines.append(f"- [{'x' if value else ' '}] {key}")
    lines.extend([
        "", "## 结论边界", "",
        "- 1-step 结果用于验证官方 v0.1 多数据集 PPO/GRPO 工程闭环，不代表 305-step 收敛结果或论文效果复现。",
        "- PPO 按官方脚本使用 n_agent=1；GRPO 使用 n_agent=5。二者采样成本不同，step 时间不能直接解释为算法本体开销。",
        "- 每个验证集只冻结 16 条，分数据集指标方差很大，仅用于接口、奖励与回载一致性检查。",
        "",
    ])
    output_md.write_text("\n".join(lines))
    print(json.dumps(payload, indent=2))
    if args.require_pass and not accepted:
        raise SystemExit("Stage05 acceptance failed")


if __name__ == "__main__":
    main()
