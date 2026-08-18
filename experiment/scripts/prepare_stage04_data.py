from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=404)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    train_source = source / "train.parquet"
    test_source = source / "test.parquet"
    if not train_source.is_file() or not test_source.is_file():
        raise SystemExit(f"missing official parquet files under {source}")

    train_all = pd.read_parquet(train_source)
    test_all = pd.read_parquet(test_source)
    train_nq = train_all[train_all["data_source"] == "nq"]
    test_nq = test_all[test_all["data_source"] == "nq"]
    if len(train_nq) < args.train_size or len(test_nq) < args.val_size:
        raise SystemExit("not enough NQ rows for the frozen subsets")

    train = train_nq.sample(n=args.train_size, random_state=args.seed).reset_index(drop=True)
    val = test_nq.iloc[: args.val_size].reset_index(drop=True)
    if set(train["id"]) & set(val["id"]):
        raise SystemExit("train/validation ids overlap")

    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.parquet"
    val_path = output / "val.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "PeterJinGo/nq_hotpotqa_train",
        "selection": {
            "data_source": "nq",
            "train": f"random sample seed={args.seed}",
            "validation": "first rows in the official test ordering",
        },
        "source_counts": {"train_nq": len(train_nq), "test_nq": len(test_nq)},
        "frozen_counts": {"train": len(train), "validation": len(val)},
        "sha256": {
            "official_train": sha256(train_source),
            "official_test": sha256(test_source),
            "frozen_train": sha256(train_path),
            "frozen_validation": sha256(val_path),
        },
        "ids": {"train": train["id"].tolist(), "validation": val["id"].tolist()},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("source_counts", "frozen_counts", "sha256")}, indent=2))


if __name__ == "__main__":
    main()
