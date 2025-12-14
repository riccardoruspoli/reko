import logging
from typing import Any, Sequence

from tqdm import tqdm

from ..errors import ProcessingError
from .chunking import build_chunk_context, chunk_transcript, format_mapped_chunks
from .modules import (
    AggregateSummarizer,
    ChunkSummarizer,
    KeyPointsGenerator,
    Translator,
)
from .text_utils import is_valid_tldr, normalize_key_points, normalize_sequence

LENGTH_PROFILES: dict[str, dict[str, object]] = {
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

logger = logging.getLogger(__name__)


def summarize_chunks(
    serialized_transcript: str, target_chunk_words: int, max_retries: int, language: str
) -> list[dict[str, Any]]:
    chunks = chunk_transcript(
        serialized_transcript, target_chunk_words=target_chunk_words
    )

    if not chunks:
        raise ProcessingError("No transcript chunks available for summarization.")

    summarizer = ChunkSummarizer()
    total_chunks = len(chunks)
    mapped: list[dict[str, Any]] = []

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
            {
                "index": chunk.index,
                "start": chunk.start,
                "end": chunk.end,
                "word_count": chunk.word_count,
                "summary": summary,
            }
        )

    return mapped


def aggregate_chunk_results(
    mapped_results: Sequence[dict[str, Any]],
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
    source_words = sum(
        len(str(entry.get("summary", "")).split()) for entry in mapped_results
    )
    min_summary_words = max(
        40,
        source_words * LENGTH_PROFILES.get(summary_length).get("min_words_ratio"),
    )

    reduce_context = (
        f"The transcript was processed in {chunk_count} chunks."
        " Merge these chunk summaries into a coherent, polished narrative in chronological order. "
        f"{LENGTH_PROFILES.get(summary_length).get('length_guidance')}"
        f" The summary should be at least {min_summary_words} words long."
        " You may lightly rewrite opening/closing clauses to create smooth transitions between chunks."
    )
    if language:
        reduce_context += f" Respond in {language}."

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


def generate_key_points(
    mapped_results: Sequence[dict[str, Any]],
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

    min_bullets, max_bullets = LENGTH_PROFILES.get(summary_length).get("bullet_ranges")
    guidance = (
        f"Create between {min_bullets} and {max_bullets} bullet-style key points in chronological order."
        " Each bullet must be a single, concrete sentence that preserves names, numbers, and outcomes."
        " Cover the full scope of the mapped chunks without adding new information."
    )
    if language:
        guidance += f" Respond in {language}."

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
    serialized_transcript: str,
    target_chunk_words: int,
    include_summary: bool,
    include_key_points: bool,
    max_retries: int,
    output_language: str,
    summary_length: str,
) -> tuple[str, list[str]]:
    mapped_results = summarize_chunks(
        serialized_transcript=serialized_transcript,
        target_chunk_words=target_chunk_words,
        max_retries=max_retries,
        language=output_language,
    )
    final_summary = aggregate_chunk_results(
        mapped_results=mapped_results,
        max_retries=max_retries,
        language=output_language,
        summary_length=summary_length,
    )
    key_points: list[str] = []
    if include_key_points:
        key_points = generate_key_points(
            mapped_results=mapped_results,
            final_summary=final_summary,
            max_retries=max_retries,
            language=output_language,
            summary_length=summary_length,
        )
    return final_summary if include_summary else "", key_points


def translate_text(
    text: str,
    target_language: str,
    max_retries: int,
    guidance: str | None = None,
) -> str:
    logger.debug("Translating text to %s", target_language)
    if not text.strip():
        return text

    translator = Translator()
    guidance = guidance or "Translate while preserving meaning and formatting."

    for attempt in range(1 + max_retries):
        prediction = translator(
            source_text=text,
            target_language=target_language,
            guidance=guidance,
        )
        translated = getattr(prediction, "translated_text", "").strip()
        if translated:
            return translated
        logger.warning(
            "Translation returned empty output (attempt %d/%d).",
            attempt + 1,
            max_retries + 1,
        )

    raise ProcessingError(
        f"Translation failed to produce output after {max_retries + 1} attempts."
    )


def translate_key_points(
    points: list[str],
    target_language: str,
    max_retries: int,
) -> list[str]:
    if not points:
        return points

    source = "\n".join(f"- {point}" for point in points)
    translated = translate_text(
        source,
        target_language=target_language,
        max_retries=max_retries,
        guidance="Translate each bullet, keep the same number of bullets and bullet structure.",
    )
    return normalize_key_points(translated)
