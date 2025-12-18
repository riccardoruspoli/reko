import re
from typing import Sequence

from .models import SummaryDocument


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


def build_markdown(
    title: str, tl_dr: str | None = None, key_points: Sequence[str] | None = None
) -> str:
    points = [str(point).strip() for point in key_points or [] if str(point).strip()]
    return SummaryDocument(
        title=title,
        summary=tl_dr.strip() if tl_dr and tl_dr.strip() else None,
        key_points=points or None,
    ).to_markdown()
