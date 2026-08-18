from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


TRAIN_SOURCES = ("nq", "hotpotqa")
EVAL_SOURCES = ("nq", "hotpotqa", "triviaqa", "popqa", "2wikimultihopqa", "musique", "bamboogle")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_sample(frame: pd.DataFrame, sources: tuple[str, ...], per_source: int, seed: int) -> pd.DataFrame:
    parts = []
    for offset, source in enumerate(sources):
        rows = frame[frame["data_source"] == source]
        if len(rows) < per_source:
            raise SystemExit(f"not enough {source} rows: need {per_source}, found {len(rows)}")
        parts.append(rows.sample(n=per_source, random_state=seed + offset))
    return pd.concat(parts, ignore_index=True)


def counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame["data_source"].value_counts().sort_index().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-per-source", type=int, default=256)
    parser.add_argument("--val-per-source", type=int, default=16)
    parser.add_argument("--seed", type=int, default=505)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    train_source = source / "train.parquet"
    test_source = source / "test.parquet"
    train_all = pd.read_parquet(train_source)
    test_all = pd.read_parquet(test_source)
    train = balanced_sample(train_all, TRAIN_SOURCES, args.train_per_source, args.seed)
    val = balanced_sample(test_all, EVAL_SOURCES, args.val_per_source, args.seed + 100)
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
            "seed": args.seed,
            "train": f"balanced random sample, {args.train_per_source} per source",
            "validation": f"balanced random sample, {args.val_per_source} per source",
        },
        "official_counts": {"train": counts(train_all), "test": counts(test_all)},
        "frozen_counts": {"train": counts(train), "validation": counts(val)},
        "sha256": {
            "official_train": sha256(train_source),
            "official_test": sha256(test_source),
            "frozen_train": sha256(train_path),
            "frozen_validation": sha256(val_path),
        },
        "ids": {"train": train["id"].tolist(), "validation": val["id"].tolist()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: manifest[key] for key in ("official_counts", "frozen_counts", "sha256")}, indent=2))


if __name__ == "__main__":
    main()
