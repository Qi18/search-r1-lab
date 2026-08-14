from __future__ import annotations

import argparse
from pathlib import Path

from search_r1_lab.retrieval import DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--model", default="intfloat/e5-small-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if Path(args.index).exists() and Path(args.metadata).exists() and not args.force:
        print(f"reuse index: {args.index}")
        return

    print(f"building index from {args.corpus}")
    DenseRetriever.build(
        corpus_path=args.corpus,
        index_path=args.index,
        metadata_path=args.metadata,
        model_name=args.model,
        device=args.device,
    )
    print(f"index ready: {args.index}")


if __name__ == "__main__":
    main()
