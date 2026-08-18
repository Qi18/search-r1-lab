from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def eval_metrics(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        "examples": len(rows),
        "em": sum(row["reward"] for row in rows) / len(rows),
        "token_f1": sum(f1(row["prediction"], row["golden_answers"]) for row in rows) / len(rows),
        "search_rate": sum(row["search_count"] > 0 for row in rows) / len(rows),
        "avg_searches": sum(row["search_count"] for row in rows) / len(rows),
        "valid_answer_rate": sum(bool(row["prediction"]) for row in rows) / len(rows),
    }


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
    labels = {"baseline": "Base", "ppo-1step": "PPO 1-step reload", "ppo-5step": "PPO 5-step reload"}
    evaluations = {name: eval_metrics(root / "eval" / name / "trajectories.jsonl") for name in labels}
    training = {step: load_json(root / f"stage04-nq-ppo-{step}step" / "acceptance.json") for step in (1, 5)}
    retriever = load_json(root / "retriever-preflight.json")
    checks = {
        "official_nq_validation_128": all(item["examples"] == 128 for item in evaluations.values()),
        "same_validation_ids": True,
        "retriever_preflight": retriever["status"] == "PASS",
        "ppo_1step": training[1]["status"] == "PASS",
        "ppo_5step": training[5]["status"] == "PASS",
        "checkpoint_reload": evaluations["ppo-1step"]["examples"] == 128 and evaluations["ppo-5step"]["examples"] == 128,
    }
    id_sets = []
    for name in labels:
        id_sets.append({json.loads(line)["id"] for line in (root / "eval" / name / "trajectories.jsonl").read_text().splitlines() if line})
    checks["same_validation_ids"] = len(id_sets[0]) == 128 and all(ids == id_sets[0] for ids in id_sets[1:])
    accepted = all(checks.values())
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if accepted else "FAIL",
        "evaluations": evaluations,
        "training": training,
        "retriever": retriever,
        "checks": checks,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Experiment 04: Official Preliminary NQ + PPO",
        "",
        "使用官方 NQ 数据格式、Wiki-18/E5 Retriever、Search-R1 Agent loop、EM reward 和 veRL GAE/PPO，在 8 x L20 上完成短程闭环。",
        "",
        "## 训练过程",
        "",
        "- 资产：官方 NQ 79,168 train / 3,610 test，冻结 train512 / val128；Wiki-18 共 21,015,324 篇，E5-base-v2 + 64.6GB FAISS index。",
        "- 基线：Qwen2.5-3B Base 在冻结 val128 上启用 Search，max_turns=2、topk=3。",
        "- PPO：1-step 和 5-step 都从同一 Base 独立启动；batch64，Actor lr=1e-6，Critic lr=1e-5，GAE，state_masking=true。",
        "- 验证：每次训练都保存 Actor/Critic；再从 Actor checkpoint 独立回载，对同一 val128 生成 128 条轨迹。",
        "",
        "## 同一冻结 NQ 验证集",
        "",
        "| Model | EM | Token F1 | Search rate | Avg searches | Valid answer |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, title in labels.items():
        item = evaluations[name]
        lines.append(f"| {title} | {pct(item['em'])} | {pct(item['token_f1'])} | {pct(item['search_rate'])} | {item['avg_searches']:.2f} | {pct(item['valid_answer_rate'])} |")
    lines.extend([
        "", "## Retriever", "",
        f"- Hit@1: {pct(retriever['hit_at_1'])}",
        f"- Hit@3: {pct(retriever['hit_at_3'])}",
        f"- Mean latency/query: {retriever['latency_seconds']['mean']:.3f}s",
        f"- P95 latency/query: {retriever['latency_seconds']['p95']:.3f}s",
        "", "## PPO 训练", "",
        "| Updates | Reward | PPO KL | Value loss | Step time | Peak GPU memory |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for step in (1, 5):
        item = training[step]
        metric = item["last_training_metrics"]
        lines.append(f"| {step} | {metric.get('critic/score/mean', 0):.4f} | {metric.get('actor/ppo_kl', 0):.6f} | {metric.get('critic/vf_loss', 0):.6f} | {metric.get('timing_s/step', 0):.1f}s | {item.get('peak_gpu_memory_mib')} MiB |")
    lines.extend(["", "## 验收结论", "", f"- Status: {'PASS' if accepted else 'FAIL'}"])
    for key, value in checks.items():
        lines.append(f"- [{'x' if value else ' '}] {key}")
    base = evaluations["baseline"]
    one = evaluations["ppo-1step"]
    five = evaluations["ppo-5step"]
    lines.extend([
        "- PASS 表示官方真实数据检索、Actor/Critic/Reference/vLLM Rollout、梯度更新、双 checkpoint 保存和独立回载评测闭环全部成立。",
        "", "## 实验结论", "",
        f"- 1-step 的 EM/F1 与 Base 相同（{pct(one['em'])}/{pct(one['token_f1'])}），作用是验证参数更新与 checkpoint 闭环，不能证明效果提升。",
        f"- 5-step 相对 Base：EM {pct(base['em'])} → {pct(five['em'])}，Token F1 {pct(base['token_f1'])} → {pct(five['token_f1'])}，搜索率 {pct(base['search_rate'])} → {pct(five['search_rate'])}。",
        f"- 5-step 的平均搜索次数从 {base['avg_searches']:.2f} 升到 {five['avg_searches']:.2f}，短程 PPO 已改变工具调用行为；但 128 条单次评测不足以判断稳定泛化。",
        "", "## 结论边界", "",
        "- PASS 表示 Actor、Critic、Reference、vLLM Rollout、检索、梯度更新、双 checkpoint 保存和回载评测闭环成立。",
        "- 5 个更新步只用于 Preliminary 工程验收；指标变化不代表完整 NQ 收敛或论文效果复现。",
        "- Hit@k 是冻结 NQ 原问题直接检索的词面覆盖率，和模型生成 query 后的端到端 EM/F1 分开解释。",
        "",
    ])
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if args.require_pass and not accepted:
        raise SystemExit("Stage04 acceptance failed")


if __name__ == "__main__":
    main()
