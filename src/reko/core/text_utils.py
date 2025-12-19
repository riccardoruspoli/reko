import re
from collections.abc import Sequence


def normalize_sequence(candidate: Sequence[str] | str | None) -> list[str]:
    if candidate is None:
        return []
    if isinstance(candidate, str):
        items = (candidate,)
    else:
        items = candidate

    values: list[str] = []
    for item in items:
        if not isinstance(item, str):
            item = str(item)
        cleaned = re.sub(r"\s+", " ", item).strip()
        if cleaned:
            values.append(cleaned)
    return values


def is_valid_tldr(tl_dr: str, min_words: int = 8) -> bool:
    text = tl_dr.strip()
    if not text:
        return False
    words = text.split()
    return len(words) >= min_words


def normalize_key_points(candidate: Sequence[str] | str | None) -> list[str]:
    """Normalize a key points response (string or sequence) into a clean list."""

    raw_items = normalize_sequence(candidate)
    key_points: list[str] = []

    for item in raw_items:
        fragments = re.split(r"[\n\r]+", item)
        for fragment in fragments:
            cleaned = re.sub(r"^[\s\-•\d\.)]+", "", fragment).strip()
            if cleaned:
                key_points.append(cleaned)

    return key_points
