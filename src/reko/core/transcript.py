import json
import logging
from typing import Any

from iso639 import Lang

logger = logging.getLogger(__name__)


# TODO: what to do in case of failure? Stop the process or return the code itself?
def resolve_language(language_code: str) -> tuple[str, str]:
    """Resolve a language code to its iso639 Lang object."""
    try:
        return Lang(language_code).pt1, Lang(language_code).name
    except Exception:
        return language_code, language_code


def get_transcript_segments(serialized_transcript: str) -> list[dict[str, Any]]:
    """Load a serialized transcript into a list of segment dicts.

    Returns only entries that look like transcript segments with a `text` field.
    """
    try:
        data = json.loads(serialized_transcript)
    except json.JSONDecodeError:
        return []

    if isinstance(data, list):
        return [
            segment
            for segment in data
            if isinstance(segment, dict) and segment.get("text")
        ]

    return []


def get_transcript_words_count(serialized_transcript: str) -> int:
    """Count the total number of words in the transcript segments."""
    segments = get_transcript_segments(serialized_transcript)
    total_words = 0
    for segment in segments:
        text = segment.get("text", "").strip()
        total_words += len(text.split())
    return total_words
