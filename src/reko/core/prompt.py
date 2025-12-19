import logging
from collections.abc import Sequence
from typing import TypedDict

from reko.core.models import SummaryChunk, TranscriptChunk

logger = logging.getLogger(__name__)


class LengthProfile(TypedDict):
    """Configuration controlling summarization length and key-point detail.

    Fields:
        length_guidance: Instruction appended to the reduce prompt.
        bullet_ranges: Inclusive min/max bullet count for key points.
        min_words_ratio: Minimum final-summary words as a ratio of mapped summary words.
    """

    length_guidance: str
    bullet_ranges: tuple[int, int]
    min_words_ratio: float


LENGTH_PROFILES: dict[str, LengthProfile] = {
    "short": {
        "length_guidance": "Make this concise: keep only the most important facts and outcomes.",
        "bullet_ranges": (1, 3),
        "min_words_ratio": 0.2,
    },
    "medium": {
        "length_guidance": "Balance concision and coverage: include key details without being exhaustive.",
        "bullet_ranges": (3, 5),
        "min_words_ratio": 0.4,
    },
    "long": {
        "length_guidance": "Be detailed and thorough: preserve most concrete details from the chunk summaries.",
        "bullet_ranges": (5, 7),
        "min_words_ratio": 0.6,
    },
}

DEFAULT_TRANSLATION_GUIDANCE = "Translate while preserving meaning and formatting."
KEY_POINTS_TRANSLATION_GUIDANCE = (
    "Translate each bullet, keep the same number of bullets and bullet structure."
)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""

    if seconds <= 0:
        return "00:00"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def build_chunk_context(
    chunk: TranscriptChunk, total_chunks: int, language: str
) -> str:
    """Describe a chunk for prompting (position, timestamps, word count)."""

    start_ts = _format_timestamp(chunk.start)
    end_ts = _format_timestamp(chunk.end)
    context = (
        f"Chunk {chunk.index + 1} of {total_chunks}. "
        f"Coverage {start_ts} to {end_ts} with {chunk.word_count} words. "
        "Summarize faithfully and avoid duplication with other chunks."
    )
    if language:
        context += f" Write in {language}."
    return context


def format_mapped_chunks(mapped: Sequence[SummaryChunk]) -> str:
    """Format mapped chunk summaries into a reduce-ready string prompt."""

    lines: list[str] = [
        "You are given chunk-level summaries. Merge them sequentially, improving only the transitions.",
        "Do not drop facts or shorten the content. Preserve the substance of each summary.",
        "",
    ]
    for entry in mapped:
        idx = entry.index + 1
        start = _format_timestamp(entry.start)
        end = _format_timestamp(entry.end)
        words = entry.word_count
        lines.append(f"[Chunk {idx}] {start}-{end} ({words} words)")
        lines.append(entry.summary.strip())
        lines.append("---")
    return "\n".join(lines).strip()


def build_reduce_context(
    *, chunk_count: int, length_guidance: str, min_summary_words: int, language: str
) -> str:
    """Build reduce-step instructions for combining mapped chunk summaries."""

    reduce_context = (
        f"The transcript was processed in {chunk_count} chunks."
        " Merge these chunk summaries into a coherent, polished narrative in chronological order. "
        f"{length_guidance}"
        f" The summary should be at least {min_summary_words} words long."
        " You may lightly rewrite opening/closing clauses to create smooth transitions between chunks."
    )
    if language:
        reduce_context += f" Respond in {language}."
    return reduce_context


def build_key_points_guidance(
    *, min_bullets: int, max_bullets: int, language: str
) -> str:
    """Build instructions for producing bullet-style key points."""

    guidance = (
        f"Create between {min_bullets} and {max_bullets} bullet-style key points in chronological order."
        " Each bullet must be a single, concrete sentence that preserves names, numbers, and outcomes."
        " Cover the full scope of the mapped chunks without adding new information."
    )
    if language:
        guidance += f" Respond in {language}."
    return guidance
