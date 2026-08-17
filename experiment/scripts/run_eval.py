from __future__ import annotations

import argparse

from search_r1_lab.agent import AgentConfig, SearchR1Agent
from search_r1_lab.io import read_jsonl, write_jsonl
from search_r1_lab.retrieval import DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--retriever-model", default="intfloat/e5-small-v2")
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--retriever-device", default="cpu")
    parser.add_argument("--mode", choices=("both", "search", "no-search"), default="both")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-search-turns", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    examples = read_jsonl(args.eval)
    if args.limit:
        examples = examples[: args.limit]
    retriever = DenseRetriever(
        corpus_path=args.corpus,
        index_path=args.index,
        model_name=args.retriever_model,
        device=args.retriever_device,
    )
    agent = SearchR1Agent(
        model_path=args.model,
        retriever=retriever,
        device=args.model_device,
        config=AgentConfig(
            max_search_turns=args.max_search_turns,
            max_new_tokens=args.max_new_tokens,
            topk=args.topk,
        ),
    )

    modes = ("no-search", "search") if args.mode == "both" else (args.mode,)
    rows = []
    for mode in modes:
        for position, example in enumerate(examples, start=1):
            print(f"[{mode}] {position}/{len(examples)} {example['id']}", flush=True)
            result = agent.answer(example["question"], search_enabled=mode == "search")
            rows.append(
                {
                    **example,
                    **result,
                    "mode": mode,
                    "model": args.model,
                    "topk": args.topk,
                }
            )
            print(
                f"prediction={result['prediction']!r} "
                f"generated={result['generated_search_count']} requests={result['retriever_request_count']} "
                f"latency={result['latency_seconds']:.2f}s",
                flush=True,
            )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
