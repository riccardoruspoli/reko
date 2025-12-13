import logging
from typing import Any, Sequence

from tqdm import tqdm

from .chunking import build_chunk_context, chunk_transcript, format_mapped_chunks
from .modules import (
    AggregateSummarizer,
    ChunkSummarizer,
    KeyPointsGenerator,
    Translator,
)
from .text_utils import is_valid_tldr, normalize_key_points, normalize_sequence

logger = logging.getLogger(__name__)


def summarize_chunks(
    serialized_transcript: str, target_chunk_words: int, max_retries: int, language: str
) -> list[dict[str, Any]]:
    chunks = chunk_transcript(
        serialized_transcript, target_chunk_words=target_chunk_words
    )

    if not chunks:
        raise RuntimeError("No transcript chunks available for summarization.")

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
            raise RuntimeError(
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
) -> str:
    if not mapped_results:
        raise RuntimeError(
            "Aggregation failed because no chunk summaries were provided."
        )

    logger.info("Starting reduce step for %d chunks.", len(mapped_results))
    aggregator = AggregateSummarizer()
    chunk_count = len(mapped_results)
    source_word_estimate = sum(
        len(str(entry.get("summary", "")).split()) for entry in mapped_results
    )
    min_summary_words = max(40, int(source_word_estimate * 0.6))

    reduce_context = (
        f"The transcript was processed in {chunk_count} chunks."
        " Merge these chunk summaries into a coherent, polished narrative."
        " Do not shorten, compress, or generalize any content."
        " Preserve every concrete detail and factual element from the chunk summaries."
        " You may lightly rewrite opening/closing clauses to create smooth transitions between chunks."
        " Maintain chronological order, keep the length comparable to the combined input, and avoid adding new information."
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

    raise RuntimeError(
        f"Aggregate summary failed validation after {max_retries + 1} attempts."
    )


def generate_key_points(
    mapped_results: Sequence[dict[str, Any]],
    final_summary: str,
    max_retries: int,
    language: str,
) -> list[str]:
    if not mapped_results and not final_summary:
        raise RuntimeError(
            "Cannot generate key points without mapped chunks or summary."
        )
    if not final_summary.strip():
        raise RuntimeError("Cannot generate key points from an empty summary.")

    generator = KeyPointsGenerator()
    formatted_chunks = format_mapped_chunks(mapped_results) if mapped_results else ""
    chunk_count = len(mapped_results)

    # minimum 5, maximum 12, +/-2
    bullet_target = max(5, min(12, chunk_count + 3))
    guidance = (
        f"Create {bullet_target} +/-2 bullet-style key points in chronological order."
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

    raise RuntimeError(
        f"Key point generation failed to produce output after {max_retries} attempts."
    )


def generate_summary_outputs(
    serialized_transcript: str,
    target_chunk_words: int,
    include_summary: bool,
    include_key_points: bool,
    max_retries: int,
    output_language: str,
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
    )
    key_points: list[str] = []
    if include_key_points:
        key_points = generate_key_points(
            mapped_results=mapped_results,
            final_summary=final_summary,
            max_retries=max_retries,
            language=output_language,
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

    raise RuntimeError(
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
