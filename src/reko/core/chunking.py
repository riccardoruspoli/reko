"""Utilities to split raw YouTube transcripts into timestamped chunks."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from ..errors import ProcessingError

logger = logging.getLogger(__name__)


@dataclass
class TranscriptChunk:
    """Represents a contiguous transcript span with timing and word count."""

    index: int
    text: str
    start: float
    end: float
    word_count: int


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


def prepare_transcript_text(serialized_transcript: str) -> str:
    """Return a whitespace-normalized transcript string."""
    segments = get_transcript_segments(serialized_transcript)
    parts = [segment.get("text", "").strip() for segment in segments]
    joined = " ".join(filter(None, parts))
    return re.sub(r"\s+", " ", joined).strip()


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    if seconds <= 0:
        return "00:00"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def chunk_transcript(
    serialized_transcript: str,
    target_chunk_words: int,
) -> list[TranscriptChunk]:
    """Split into chunks aiming for `target_chunk_words` words; segments are kept whole, so chunks may exceed the target if a single segment is long."""

    # extract the segments that make up the transcript, as YouTube returns them
    segments = get_transcript_segments(serialized_transcript)
    if not segments:
        raise ProcessingError("No valid transcript segments found.")

    chunks: list[TranscriptChunk] = []
    current_text_parts: list[str] = []
    current_words = 0
    chunk_start: float | None = None
    chunk_end: float | None = None

    # segments are split into chunks according to the maximum number of words per chunk
    for segment in segments:
        current_text_parts, current_words, chunk_start, chunk_end = _process_segment(
            segment=segment,
            target_chunk_words=target_chunk_words,
            chunks=chunks,
            current_text_parts=current_text_parts,
            current_words=current_words,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )

    # flush any remaining text as a final chunk
    if current_text_parts:
        chunk_text = re.sub(r"\s+", " ", " ".join(current_text_parts)).strip()
        chunks.append(
            TranscriptChunk(
                index=len(chunks),
                text=chunk_text,
                start=chunk_start or 0.0,
                end=chunk_end or chunk_start or 0.0,
                word_count=len(chunk_text.split()),
            )
        )

    return chunks


def _process_segment(
    segment: dict[str, Any],
    target_chunk_words: int,
    chunks: list[TranscriptChunk],
    current_text_parts: list[str],
    current_words: int,
    chunk_start: float | None,
    chunk_end: float | None,
) -> tuple[list[str], int, float | None, float | None]:
    """Accumulate a single segment into chunks, flushing when word target is hit."""
    text = segment.get("text", "").strip()
    if not text:
        return current_text_parts, current_words, chunk_start, chunk_end

    start = float(segment.get("start", chunk_end or 0.0))
    duration = float(segment.get("duration", 0.0))
    end = start + duration
    word_count = len(text.split())

    if chunk_start is None:
        chunk_start = start

    prospective_words = current_words + word_count

    # flush the current chunk if adding this segment would exceed the target
    if (
        target_chunk_words
        and prospective_words > target_chunk_words
        and current_text_parts
    ):
        chunk_text = re.sub(r"\s+", " ", " ".join(current_text_parts)).strip()
        chunks.append(
            TranscriptChunk(
                index=len(chunks),
                text=chunk_text,
                start=chunk_start or 0.0,
                end=chunk_end or chunk_start or 0.0,
                word_count=len(chunk_text.split()),
            )
        )
        current_text_parts = []
        current_words = 0
        chunk_start = start
        chunk_end = None

    current_text_parts.append(text)
    current_words += word_count
    chunk_end = end if chunk_end is None else max(chunk_end, end)

    return current_text_parts, current_words, chunk_start, chunk_end


def build_chunk_context(
    chunk: TranscriptChunk, total_chunks: int, language: str
) -> str:
    """Describe a chunk for prompting (position, timestamps, word count)."""
    start_ts = format_timestamp(chunk.start)
    end_ts = format_timestamp(chunk.end)
    context = (
        f"Chunk {chunk.index + 1} of {total_chunks}. "
        f"Coverage {start_ts} to {end_ts} with {chunk.word_count} words. "
        "Summarize faithfully and avoid duplication with other chunks."
    )
    if language:
        context += f" Write in {language}."
    return context


def format_mapped_chunks(mapped: Sequence[dict[str, Any]]) -> str:
    """Format mapped chunk summaries into a reduce-ready string prompt."""
    lines: list[str] = [
        "You are given chunk-level summaries. Merge them sequentially, improving only the transitions.",
        "Do not drop facts or shorten the content. Preserve the substance of each summary.",
        "",
    ]
    for entry in mapped:
        idx = entry.get("index", 0) + 1
        start = format_timestamp(float(entry.get("start", 0.0)))
        end = format_timestamp(float(entry.get("end", 0.0)))
        words = entry.get("word_count", 0)
        lines.append(f"[Chunk {idx}] {start}-{end} ({words} words)")
        lines.append(entry.get("summary", "").strip())
        lines.append("---")
    return "\n".join(lines).strip()
