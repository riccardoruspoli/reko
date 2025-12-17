import logging
import re
from typing import Any

from .errors import ProcessingError
from .models import TranscriptChunk
from .transcript import get_transcript_segments

logger = logging.getLogger(__name__)


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
