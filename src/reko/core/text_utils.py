import re
from collections.abc import Sequence

_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+")
_INLINE_BULLET_RE = re.compile(r"\s[-•*]\s+")
_INLINE_NUMBERED_BULLET_RE = re.compile(r"\s\d+[.)]\s+")
_WHITESPACE_RE = re.compile(r"\s+")


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


def _key_point_items(candidate: Sequence[str] | str | None) -> list[str]:
    if candidate is None:
        return []
    if isinstance(candidate, str):
        return [candidate]
    return [str(item) for item in candidate]


def _normalize_line_breaks(text: str) -> str:
    return (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def _strip_bullet_prefix(text: str) -> tuple[str, bool]:
    cleaned, count = _BULLET_PREFIX_RE.subn("", text.strip(), count=1)
    return cleaned, count > 0


def _split_compact_bullets(text: str, had_bullet_prefix: bool) -> list[str]:
    unordered_count = len(_INLINE_BULLET_RE.findall(text))
    numbered_count = len(_INLINE_NUMBERED_BULLET_RE.findall(text))

    if unordered_count > 1 or (had_bullet_prefix and unordered_count):
        return _INLINE_BULLET_RE.split(text)
    if numbered_count > 1 or (had_bullet_prefix and numbered_count):
        return _INLINE_NUMBERED_BULLET_RE.split(text)
    return [text]


def normalize_key_points(candidate: Sequence[str] | str | None) -> list[str]:
    """Normalize a key points response (string or sequence) into a clean list."""

    key_points: list[str] = []
    for item in _key_point_items(candidate):
        for line in _normalize_line_breaks(item).splitlines():
            line = line.strip()
            if not line:
                continue

            line, had_bullet_prefix = _strip_bullet_prefix(line)
            for value in _split_compact_bullets(line, had_bullet_prefix):
                cleaned = _WHITESPACE_RE.sub(" ", value).strip()
                if cleaned:
                    key_points.append(cleaned)

    return key_points
