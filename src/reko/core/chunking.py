import logging
import re

from .errors import ProcessingError
from .models import Transcript, TranscriptChunk, TranscriptSegment

logger = logging.getLogger(__name__)


def chunk_transcript(
    transcript: Transcript,
    target_chunk_words: int,
) -> list[TranscriptChunk]:
    """Split into chunks aiming for `target_chunk_words` words; segments are kept whole, so chunks may exceed the target if a single segment is long."""

    if not transcript.segments:
        raise ProcessingError("No valid transcript segments found.")

    chunks: list[TranscriptChunk] = []
    current_text_parts: list[str] = []
    current_words = 0
    chunk_start: float | None = None
    chunk_end: float | None = None

    # segments are split into chunks according to the maximum number of words per chunk
    for segment in transcript.segments:
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
    segment: TranscriptSegment,
    target_chunk_words: int,
    chunks: list[TranscriptChunk],
    current_text_parts: list[str],
    current_words: int,
    chunk_start: float | None,
    chunk_end: float | None,
) -> tuple[list[str], int, float | None, float | None]:
    """Accumulate a single segment into chunks, flushing when word target is hit."""
    text = segment.text.strip()
    if not text:
        return current_text_parts, current_words, chunk_start, chunk_end

    start = float(segment.start)
    end = float(segment.end)
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
