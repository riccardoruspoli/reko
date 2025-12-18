import logging
from typing import Sequence

from tqdm import tqdm

from ..adapters.dspy.modules import (
    AggregateSummarizer,
    ChunkSummarizer,
    KeyPointsGenerator,
)
from .chunking import chunk_transcript
from .errors import ProcessingError
from .models import SummaryChunk, SummaryOutput, Transcript
from .prompt import (
    LENGTH_PROFILES,
    build_chunk_context,
    build_key_points_guidance,
    build_reduce_context,
    format_mapped_chunks,
)
from .text_utils import is_valid_tldr, normalize_key_points, normalize_sequence

logger = logging.getLogger(__name__)


def _get_length_profile(summary_length: str) -> dict[str, object]:
    profile = LENGTH_PROFILES.get(summary_length)
    if not profile:
        raise ProcessingError(f"Unknown summary length profile: {summary_length}")
    return profile


def _summarize_chunks(
    transcript: Transcript, target_chunk_words: int, max_retries: int, language: str
) -> list[SummaryChunk]:
    chunks = chunk_transcript(transcript, target_chunk_words=target_chunk_words)

    if not chunks:
        raise ProcessingError("No transcript chunks available for summarization.")

    summarizer = ChunkSummarizer()
    total_chunks = len(chunks)
    mapped: list[SummaryChunk] = []

    for chunk in tqdm(chunks, desc="Summarizing chunks", unit="chunk"):
        context = build_chunk_context(chunk, total_chunks, language=language)

        # 8 words minimum, or 8 words + 1 per 30 words of source
        min_summary_words = max(8, chunk.word_count // 30 + 8)

        summary: str | None = None
        for attempt in range(1 + max_retries):
            prediction = summarizer(
                chunk_text=chunk.text,
                chunk_context=context,
            )

            summary_parts = normalize_sequence(getattr(prediction, "summary", ""))
            summary = " ".join(summary_parts).strip()

            if is_valid_tldr(summary, min_summary_words):
                break

            logger.warning(
                "Chunk %d summary failed validation (attempt %d/%d).",
                chunk.index,
                attempt + 1,
                max_retries + 1,
            )
            summary = None

        if summary is None:
            raise ProcessingError(
                f"Chunk {chunk.index} summary failed validation after {max_retries} attempts."
            )

        mapped.append(
            SummaryChunk(
                index=chunk.index,
                start=chunk.start,
                end=chunk.end,
                word_count=chunk.word_count,
                summary=summary,
            )
        )

    return mapped


def _aggregate_chunk_results(
    mapped_results: Sequence[SummaryChunk],
    max_retries: int,
    language: str,
    summary_length: str,
) -> str:
    if not mapped_results:
        raise ProcessingError(
            "Aggregation failed because no chunk summaries were provided."
        )

    logger.info("Starting reduce step for %d chunks.", len(mapped_results))
    aggregator = AggregateSummarizer()
    chunk_count = len(mapped_results)
    length_profile = _get_length_profile(summary_length)
    source_words = sum(len(entry.summary.split()) for entry in mapped_results)
    min_words_ratio = float(length_profile.get("min_words_ratio", 0.0))
    min_summary_words = int(max(40, source_words * min_words_ratio))

    reduce_context = build_reduce_context(
        chunk_count=chunk_count,
        length_guidance=str(length_profile.get("length_guidance", "")),
        min_summary_words=min_summary_words,
        language=language,
    )

    formatted_chunks = format_mapped_chunks(mapped_results)

    for attempt in range(1 + max_retries):
        prediction = aggregator(
            mapped_chunks=formatted_chunks,
            reduce_context=reduce_context,
        )
        summary_parts = normalize_sequence(getattr(prediction, "final_summary", ""))
        summary = " ".join(summary_parts).strip()

        if is_valid_tldr(summary, min_summary_words):
            return summary

        logger.warning(
            "Aggregate summary failed validation (attempt %d/%d). %d words (target min %d words).",
            attempt + 1,
            max_retries + 1,
            len(summary.split()),
            min_summary_words,
        )

    raise ProcessingError(
        f"Aggregate summary failed validation after {max_retries + 1} attempts."
    )


def _generate_key_points(
    mapped_results: Sequence[SummaryChunk],
    final_summary: str,
    max_retries: int,
    language: str,
    summary_length: str,
) -> list[str]:
    if not mapped_results and not final_summary:
        raise ProcessingError(
            "Cannot generate key points without mapped chunks or summary."
        )
    if not final_summary.strip():
        raise ProcessingError("Cannot generate key points from an empty summary.")

    generator = KeyPointsGenerator()
    formatted_chunks = format_mapped_chunks(mapped_results) if mapped_results else ""

    min_bullets, max_bullets = _get_length_profile(summary_length).get(
        "bullet_ranges", (3, 5)
    )
    guidance = build_key_points_guidance(
        min_bullets=int(min_bullets),
        max_bullets=int(max_bullets),
        language=language,
    )

    for attempt in range(1 + max_retries):
        prediction = generator(
            mapped_chunks=formatted_chunks,
            final_summary=final_summary,
            guidance=guidance,
        )
        key_points = normalize_key_points(getattr(prediction, "key_points", []))
        if key_points:
            return key_points

        logger.warning(
            "Key point generation returned empty output (attempt %d/%d).",
            attempt + 1,
            max_retries + 1,
        )

    raise ProcessingError(
        f"Key point generation failed to produce output after {max_retries} attempts."
    )


def generate_summary_outputs(
    transcript: Transcript,
    target_chunk_words: int,
    include_summary: bool,
    include_key_points: bool,
    max_retries: int,
    summary_length: str,
) -> SummaryOutput:
    language = transcript.language.name
    mapped_results = _summarize_chunks(
        transcript=transcript,
        target_chunk_words=target_chunk_words,
        max_retries=max_retries,
        language=language,
    )
    final_summary = _aggregate_chunk_results(
        mapped_results=mapped_results,
        max_retries=max_retries,
        language=language,
        summary_length=summary_length,
    )
    key_points: list[str] | None = None
    if include_key_points:
        key_points = _generate_key_points(
            mapped_results=mapped_results,
            final_summary=final_summary,
            max_retries=max_retries,
            language=language,
            summary_length=summary_length,
        )

    return SummaryOutput(
        summary=final_summary if include_summary else None,
        key_points=key_points if include_key_points else None,
    )
