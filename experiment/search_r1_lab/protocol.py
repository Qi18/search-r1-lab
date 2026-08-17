from __future__ import annotations

import re


SEARCH_RE = re.compile(r"<search>\s*(.*?)\s*</search>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_search_query(text: str) -> str | None:
    matches = SEARCH_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else ""


def has_valid_answer(text: str) -> bool:
    return bool(ANSWER_RE.search(text))


def trim_to_first_action(text: str) -> str:
    endings: list[tuple[int, str]] = []
    for closing_tag in ("</search>", "</answer>"):
        index = text.lower().find(closing_tag)
        if index >= 0:
            endings.append((index + len(closing_tag), text[: index + len(closing_tag)]))
    return min(endings, key=lambda item: item[0])[1] if endings else text
