from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from search_r1_lab.io import read_jsonl, write_jsonl


VALIDATION_FACTS = (
    ("material", "Alder Pressure Shell", "titanium carbide", "What material lines the Alder Pressure Shell?", "The Alder Pressure Shell is lined with titanium carbide."),
    ("material", "Birch Thermal Chamber", "boron nitride", "What material lines the Birch Thermal Chamber?", "The Birch Thermal Chamber is lined with boron nitride."),
    ("material", "Cairn Vacuum Vessel", "silicon carbide", "What material lines the Cairn Vacuum Vessel?", "The Cairn Vacuum Vessel is lined with silicon carbide."),
    ("material", "Dawn Reaction Cell", "zirconium oxide", "What material lines the Dawn Reaction Cell?", "The Dawn Reaction Cell is lined with zirconium oxide."),
    ("material", "Elm Cryogenic Tank", "aluminum nitride", "What material lines the Elm Cryogenic Tank?", "The Elm Cryogenic Tank is lined with aluminum nitride."),
    ("material", "Flint Containment Drum", "tantalum carbide", "What material lines the Flint Containment Drum?", "The Flint Containment Drum is lined with tantalum carbide."),
    ("material", "Grove Test Capsule", "magnesium oxide", "What material lines the Grove Test Capsule?", "The Grove Test Capsule is lined with magnesium oxide."),
    ("material", "Hearth Plasma Tube", "yttrium oxide", "What material lines the Hearth Plasma Tube?", "The Hearth Plasma Tube is lined with yttrium oxide."),
    ("depth", "Iris Ocean Node", "740 meters", "At what depth is the Iris Ocean Node anchored?", "The Iris Ocean Node is anchored at a depth of 740 meters."),
    ("depth", "Jade Current Monitor", "915 meters", "At what depth is the Jade Current Monitor anchored?", "The Jade Current Monitor is anchored at a depth of 915 meters."),
    ("depth", "Kelp Seismic Pod", "680 meters", "At what depth is the Kelp Seismic Pod anchored?", "The Kelp Seismic Pod is anchored at a depth of 680 meters."),
    ("depth", "Lotus Abyss Station", "1,120 meters", "At what depth is the Lotus Abyss Station anchored?", "The Lotus Abyss Station is anchored at a depth of 1,120 meters."),
    ("depth", "Mica Salinity Array", "835 meters", "At what depth is the Mica Salinity Array anchored?", "The Mica Salinity Array is anchored at a depth of 835 meters."),
    ("depth", "Nacre Hydrophone", "960 meters", "At what depth is the Nacre Hydrophone anchored?", "The Nacre Hydrophone is anchored at a depth of 960 meters."),
    ("depth", "Opal Trench Relay", "1,275 meters", "At what depth is the Opal Trench Relay anchored?", "The Opal Trench Relay is anchored at a depth of 1,275 meters."),
    ("depth", "Pearl Benthic Lab", "705 meters", "At what depth is the Pearl Benthic Lab anchored?", "The Pearl Benthic Lab is anchored at a depth of 705 meters."),
    ("commander", "Quartz Horizon Mission", "Elara Niven", "Who commanded the Quartz Horizon Mission?", "The Quartz Horizon Mission was commanded by Elara Niven."),
    ("commander", "Riverglass Expedition", "Pavel Orin", "Who commanded the Riverglass Expedition?", "The Riverglass Expedition was commanded by Pavel Orin."),
    ("commander", "Silver Comet Survey", "Mina Tarek", "Who commanded the Silver Comet Survey?", "The Silver Comet Survey was commanded by Mina Tarek."),
    ("commander", "Timberline Transit", "Joren Vale", "Who commanded the Timberline Transit?", "The Timberline Transit was commanded by Joren Vale."),
    ("commander", "Umbral Coast Mission", "Sela Rinn", "Who commanded the Umbral Coast Mission?", "The Umbral Coast Mission was commanded by Sela Rinn."),
    ("commander", "Verdant Orbit Survey", "Kian Dorel", "Who commanded the Verdant Orbit Survey?", "The Verdant Orbit Survey was commanded by Kian Dorel."),
    ("commander", "Willow Star Expedition", "Anya Vesk", "Who commanded the Willow Star Expedition?", "The Willow Star Expedition was commanded by Anya Vesk."),
    ("commander", "Xylem Frontier Transit", "Ravi Sorn", "Who commanded the Xylem Frontier Transit?", "The Xylem Frontier Transit was commanded by Ravi Sorn."),
    ("signal", "Yarrow Dock Beacon", "amber", "What color does the Yarrow Dock Beacon flash during calibration?", "The Yarrow Dock Beacon flashes amber during calibration."),
    ("signal", "Zenith Mooring Light", "violet", "What color does the Zenith Mooring Light flash during calibration?", "The Zenith Mooring Light flashes violet during calibration."),
    ("signal", "Auburn Range Marker", "cyan", "What color does the Auburn Range Marker flash during calibration?", "The Auburn Range Marker flashes cyan during calibration."),
    ("signal", "Bramble Safety Lamp", "indigo", "What color does the Bramble Safety Lamp flash during calibration?", "The Bramble Safety Lamp flashes indigo during calibration."),
    ("signal", "Copper Channel Buoy", "white", "What color does the Copper Channel Buoy flash during calibration?", "The Copper Channel Buoy flashes white during calibration."),
    ("signal", "Dune Approach Signal", "green", "What color does the Dune Approach Signal flash during calibration?", "The Dune Approach Signal flashes green during calibration."),
    ("signal", "Estuary Guide Light", "crimson", "What color does the Estuary Guide Light flash during calibration?", "The Estuary Guide Light flashes crimson during calibration."),
    ("signal", "Fjord Navigation Lamp", "blue", "What color does the Fjord Navigation Lamp flash during calibration?", "The Fjord Navigation Lamp flashes blue during calibration."),
)

PROMPT = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as you want. If you find no further external knowledge needed, provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_rl_record(row: dict, split: str, index: int) -> dict:
    return {
        "data_source": "nq",
        "prompt": [{"role": "user", "content": PROMPT.format(question=row["question"])}],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": {"target": [row["answer"]]}},
        "extra_info": {"split": split, "index": index, "question_id": row["id"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage01-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    stage01_dir = Path(args.stage01_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_corpus = read_jsonl(stage01_dir / "corpus.jsonl")
    train_eval = read_jsonl(stage01_dir / "eval.jsonl")
    val_corpus = []
    val_eval = []
    group_positions: dict[str, int] = {}
    for group, entity, answer, question, fact in VALIDATION_FACTS:
        group_positions[group] = group_positions.get(group, 0) + 1
        evidence_id = f"stage02-{group}-{group_positions[group]:02d}"
        val_corpus.append({"id": evidence_id, "contents": f"{entity}\n{fact}"})
        val_eval.append({"id": f"q-{evidence_id}", "question": question, "answer": answer, "evidence_id": evidence_id})

    corpus = train_corpus + val_corpus
    train_records = [to_rl_record(row, "train", index) for index, row in enumerate(train_eval)]
    val_records = [to_rl_record(row, "validation", 10_000 + index) for index, row in enumerate(val_eval)]

    assert len(train_records) == 64
    assert len(val_records) == 32
    assert len(corpus) == 96
    assert {row["id"] for row in train_eval}.isdisjoint({row["id"] for row in val_eval})
    corpus_by_id = {row["id"]: row["contents"] for row in corpus}
    for row in train_eval + val_eval:
        assert row["answer"] in corpus_by_id[row["evidence_id"]]

    paths = {
        "corpus": output_dir / "corpus.jsonl",
        "train_eval": output_dir / "train_eval.jsonl",
        "val_eval": output_dir / "val_eval.jsonl",
        "train": output_dir / "train.parquet",
        "val": output_dir / "val.parquet",
    }
    write_jsonl(paths["corpus"], corpus)
    write_jsonl(paths["train_eval"], train_eval)
    write_jsonl(paths["val_eval"], val_eval)
    pd.DataFrame(train_records).to_parquet(paths["train"], index=False)
    pd.DataFrame(val_records).to_parquet(paths["val"], index=False)

    manifest = {
        "documents": len(corpus),
        "train_examples": len(train_records),
        "validation_examples": len(val_records),
        "validation_is_disjoint": True,
        "files": {name: {"path": path.name, "sha256": sha256(path)} for name, path in paths.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
