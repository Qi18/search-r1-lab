from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from search_r1_lab.io import write_jsonl


DESIGNERS = (
    ("Avelin Observatory", "Mira Solen"),
    ("Boreal Relay", "Tomas Vey"),
    ("Cindar Gate", "Lena Oris"),
    ("Dovren Array", "Soren Kale"),
    ("Elaris Station", "Nadia Voss"),
    ("Faron Spire", "Ilan Meru"),
    ("Galen Vault", "Rhea Nolin"),
    ("Helion Beacon", "Dara Quen"),
)
SERVICE_YEARS = (
    ("Istral Gate", "2037"),
    ("Jovian Causeway", "2042"),
    ("Kestrel Bridge", "2031"),
    ("Lumen Portal", "2046"),
    ("Meridian Lock", "2039"),
    ("Neris Passage", "2044"),
    ("Orilon Crossing", "2034"),
    ("Pelar Arch", "2048"),
)
ISOTOPES = (
    ("Quorin Sensor", "Xenon-129"),
    ("Ravel Spectrometer", "Carbon-13"),
    ("Sable Detector", "Oxygen-18"),
    ("Tarin Scanner", "Helium-3"),
    ("Ulmar Imager", "Nitrogen-15"),
    ("Vesper Counter", "Deuterium"),
    ("Weyland Analyzer", "Silicon-29"),
    ("Xandor Monitor", "Neon-22"),
)
DRONE_COUNTS = (
    ("Yarrow Ridge Survey", "17"),
    ("Zephyr Basin Mission", "23"),
    ("Amber Shelf Expedition", "31"),
    ("Bracken Field Survey", "14"),
    ("Cobalt Dune Mission", "28"),
    ("Delta Crater Expedition", "19"),
    ("Ember Plain Survey", "26"),
    ("Frost Vale Mission", "12"),
)
CURATORS = (
    ("Garnet Archive", "Ivo Sen"),
    ("Harbor Manuscript Room", "Asha Pell"),
    ("Ivory Record Hall", "Niko Daren"),
    ("Juniper Collection", "Lea Morin"),
    ("Keystone Repository", "Oren Vale"),
    ("Lantern Archive", "Talia Sorn"),
    ("Marble Registry", "Evan Kiro"),
    ("Northwind Library", "Sia Loren"),
)
INVENTORS = (
    ("Orchid-4 coolant", "Rhea Calder"),
    ("Pulsar-8 lubricant", "Marek Tovin"),
    ("Quartz-2 catalyst", "Nina Sel"),
    ("Radian-5 alloy", "Jonas Mire"),
    ("Solace-9 membrane", "Ari Venn"),
    ("Tundra-3 sealant", "Mila Koren"),
    ("Umber-6 ceramic", "Theo Rask"),
    ("Violet-7 coolant", "Lora Dey"),
)
ISLANDS = (
    ("Westhaven Observatory", "Pelion Island"),
    ("Xeric Lighthouse", "Neris Island"),
    ("Yonder Research Dome", "Calder Island"),
    ("Zeal Weather Station", "Orin Island"),
    ("Aster Radio Tower", "Vela Island"),
    ("Beryl Marine Lab", "Soren Island"),
    ("Cedar Skywatch", "Tarin Island"),
    ("Drift Signal Post", "Maren Island"),
)
BACKUP_INTERVALS = (
    ("Ember Ledger", "every 19 minutes"),
    ("Fallow Registry", "every 23 minutes"),
    ("Granite Journal", "every 31 minutes"),
    ("Horizon Catalog", "every 17 minutes"),
    ("Indigo Record", "every 29 minutes"),
    ("Jasper Logbook", "every 13 minutes"),
    ("Kindle Archive", "every 37 minutes"),
    ("Lattice Ledger", "every 41 minutes"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    corpus: list[dict] = []
    evaluation: list[dict] = []

    def add(group: str, position: int, entity: str, answer: str, question: str, fact: str) -> None:
        record_id = f"{group}-{position:02d}"
        corpus.append({"id": record_id, "contents": f"{entity}\n{fact}"})
        evaluation.append(
            {
                "id": f"q-{record_id}",
                "question": question,
                "answer": answer,
                "evidence_id": record_id,
            }
        )

    for position, (entity, answer) in enumerate(DESIGNERS, 1):
        add("designer", position, entity, answer, f"Who designed {entity}?", f"{entity} was designed by {answer}.")
    for position, (entity, answer) in enumerate(SERVICE_YEARS, 1):
        add("service", position, entity, answer, f"In what year did {entity} enter service?", f"{entity} entered service in {answer}.")
    for position, (entity, answer) in enumerate(ISOTOPES, 1):
        add("isotope", position, entity, answer, f"Which isotope is used by {entity}?", f"The calibration cells in {entity} use {answer}.")
    for position, (entity, answer) in enumerate(DRONE_COUNTS, 1):
        add("drones", position, entity, answer, f"How many autonomous drones joined the {entity}?", f"The {entity} used {answer} autonomous drones.")
    for position, (entity, answer) in enumerate(CURATORS, 1):
        add("curator", position, entity, answer, f"Who was the first curator of the {entity}?", f"The first curator of the {entity} was {answer}.")
    for position, (entity, answer) in enumerate(INVENTORS, 1):
        add("inventor", position, entity, answer, f"Who invented the {entity}?", f"The {entity} was invented by {answer}.")
    for position, (entity, answer) in enumerate(ISLANDS, 1):
        add("island", position, entity, answer, f"On which island does {entity} stand?", f"{entity} stands on {answer}.")
    for position, (entity, answer) in enumerate(BACKUP_INTERVALS, 1):
        add("backup", position, entity, answer, f"How often does the {entity} write an off-site backup?", f"The {entity} writes an off-site backup {answer}.")

    assert len(corpus) == len(evaluation) == 64
    assert len({row["id"] for row in corpus}) == 64
    assert len({row["id"] for row in evaluation}) == 64
    assert all(row["answer"] in corpus[index]["contents"] for index, row in enumerate(evaluation))

    output_dir = Path(args.output_dir)
    corpus_path = output_dir / "corpus.jsonl"
    eval_path = output_dir / "eval.jsonl"
    write_jsonl(corpus_path, corpus)
    write_jsonl(eval_path, evaluation)
    print(f"wrote {len(corpus)} documents: {corpus_path} sha256={sha256(corpus_path)}")
    print(f"wrote {len(evaluation)} questions: {eval_path} sha256={sha256(eval_path)}")


if __name__ == "__main__":
    main()
