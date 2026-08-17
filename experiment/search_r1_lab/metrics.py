from __future__ import annotations

import re
import string
from collections import Counter, defaultdict

from .protocol import has_valid_answer


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def contains_answer(prediction: str, answer: str) -> float:
    expected = normalize_answer(answer)
    predicted = normalize_answer(prediction)
    return float(bool(expected) and expected in predicted)


def token_f1(prediction: str, answer: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(answer).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def retrieval_hit(record: dict) -> float:
    expected = record.get("evidence_id")
    if not expected:
        return 0.0
    retrieved = {
        result.get("id")
        for event in record.get("search_events", [])
        for result in event.get("results", [])
    }
    return float(expected in retrieved)


def compute_metrics(records: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["mode"]].append(record)

    summary: dict[str, dict[str, float]] = {}
    for mode, rows in grouped.items():
        count = len(rows)
        summary[mode] = {
            "examples": count,
            "exact_match": sum(exact_match(row["prediction"], row["answer"]) for row in rows) / count,
            "token_f1": sum(token_f1(row["prediction"], row["answer"]) for row in rows) / count,
            "answer_contains": sum(contains_answer(row["prediction"], row["answer"]) for row in rows) / count,
            "valid_answer_rate": sum(has_valid_answer(row["trajectory"]) for row in rows) / count,
            "generated_search_tag_rate": sum(row["generated_search_count"] > 0 for row in rows) / count,
            "retriever_request_rate": sum(row["retriever_request_count"] > 0 for row in rows) / count,
            "retrieval_hit_rate": sum(retrieval_hit(row) for row in rows) / count,
            "avg_search_turns": sum(row["generated_search_count"] for row in rows) / count,
            "avg_latency_seconds": sum(row["latency_seconds"] for row in rows) / count,
        }
    return summary
