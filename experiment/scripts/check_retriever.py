from __future__ import annotations

import argparse
import json
from pathlib import Path

from search_r1_lab.io import read_jsonl
from search_r1_lab.retrieval import DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--model", default="intfloat/e5-small-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--min-hit-at-3", type=float, default=0.95)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.eval)
    retriever = DenseRetriever(
        corpus_path=args.corpus,
        index_path=args.index,
        model_name=args.model,
        device=args.device,
    )
    outcomes = []
    for row in rows:
        results = retriever.search(row["question"], args.topk)
        retrieved_ids = [result["id"] for result in results]
        outcomes.append(
            {
                "id": row["id"],
                "evidence_id": row["evidence_id"],
                "retrieved_ids": retrieved_ids,
                "hit_at_1": row["evidence_id"] in retrieved_ids[:1],
                "hit_at_3": row["evidence_id"] in retrieved_ids[:3],
            }
        )

    count = len(outcomes)
    hit_at_1 = sum(item["hit_at_1"] for item in outcomes) / count
    hit_at_3 = sum(item["hit_at_3"] for item in outcomes) / count
    payload = {
        "examples": count,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "minimum_hit_at_3": args.min_hit_at_3,
        "passed": hit_at_3 >= args.min_hit_at_3,
        "outcomes": outcomes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"retriever preflight: examples={count} hit@1={hit_at_1:.1%} hit@3={hit_at_3:.1%}")
    if not payload["passed"]:
        raise SystemExit("retriever preflight failed")


if __name__ == "__main__":
    main()
