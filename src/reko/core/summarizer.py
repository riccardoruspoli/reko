import logging
from collections.abc import Mapping, Sequence

import dspy
from tqdm import tqdm

from reko.adapters.dspy.modules import (
    AggregateSummarizer,
    ChunkSummarizer,
    KeyPointsGenerator,
)
from reko.core.chunking import chunk_transcript
from reko.core.errors import JobCancelledError, ProcessingError
from reko.core.models import SummaryChunk, SummaryOutput, Transcript
from reko.core.progress import CancelCheck, ProgressEvent, ProgressReporter
from reko.core.prompt import (
    LENGTH_PROFILES,
    LengthProfile,
    build_chunk_context,
    build_key_points_guidance,
    build_reduce_context,
    format_mapped_chunks,
)
from reko.core.text_utils import is_valid_tldr, normalize_key_points, normalize_sequence

logger = logging.getLogger(__name__)


def _get_length_profile(summary_length: str) -> LengthProfile:
    """Return the configured length profile for the requested summary length."""

    profile = LENGTH_PROFILES.get(summary_length)
    if not profile:
        raise ProcessingError(f"Unknown summary length profile: {summary_length}")
    return profile


def _ensure_not_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check and cancel_check():
        raise JobCancelledError("Job cancelled.")


def format_direct_transcript(transcript: Transcript) -> str:
    """Format a transcript as chronological timestamped text for a direct request."""

    lines = []
    for segment in transcript.segments:
        minutes, seconds = divmod(int(segment.start), 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] {segment.text.strip()}")
    return "\n".join(line for line in lines if line.strip())


def generate_direct_summary_outputs(
    *,
    transcript: Transcript,
    prompt: str,
    include_summary: bool,
    include_key_points: bool,
    max_retries: int,
    summary_length: str,
    progress: ProgressReporter | None = None,
    cancel_check: CancelCheck | None = None,
) -> SummaryOutput:
    """Generate requested outputs from a complete transcript in one validated call."""

    transcript_text = format_direct_transcript(transcript)
    if not transcript_text:
        raise ProcessingError("Cannot summarize an empty transcript.")
    minimum_points, maximum_points = _get_length_profile(summary_length)[
        "bullet_ranges"
    ]
    last_error: ProcessingError | None = None
    for attempt in range(max_retries + 1):
        _ensure_not_cancelled(cancel_check)
        if progress:
            progress(
                ProgressEvent(
                    phase="direct",
                    message="Generating full-context output",
                    completed=attempt,
                    total=max_retries + 1,
                )
            )
        output = _direct_model_output(prompt, transcript_text)
        try:
            result = _parse_direct_output(
                output,
                include_summary=include_summary,
                include_key_points=include_key_points,
                minimum_points=minimum_points,
                maximum_points=maximum_points,
            )
        except ProcessingError as error:
            last_error = error
            logger.warning(
                "Direct output failed validation (attempt %d/%d): %s",
                attempt + 1,
                max_retries + 1,
                error,
            )
            continue
        if progress:
            progress(
                ProgressEvent(
                    phase="direct",
                    message="Generated full-context output",
                    completed=attempt + 1,
                    total=max_retries + 1,
                    metrics={"direct_attempts": attempt + 1},
                )
            )
        return result
    raise ProcessingError(
        "Direct output failed validation after "
        f"{max_retries + 1} attempts: {last_error}"
    )


def _direct_model_output(prompt: str, transcript_text: str) -> str:
    response = dspy.settings.lm(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript_text},
        ]
    )
    if not isinstance(response, list) or not response:
        raise ProcessingError("Direct model returned no text output.")
    output = response[0]
    if isinstance(output, Mapping):
        output = output.get("text")
    if not isinstance(output, str):
        raise ProcessingError("Direct model returned a non-text output.")
    return output


def _parse_direct_output(
    output: str,
    *,
    include_summary: bool,
    include_key_points: bool,
    minimum_points: int,
    maximum_points: int,
) -> SummaryOutput:
    from reko.core.models import SummaryDocument

    if include_summary and "## Summary" not in output:
        raise ProcessingError("Direct output is missing the Summary heading.")
    if include_key_points and "## Key Points" not in output:
        raise ProcessingError("Direct output is missing the Key Points heading.")
    document = SummaryDocument.from_markdown(output)
    if include_summary and not is_valid_tldr(document.summary or "", 40):
        raise ProcessingError("Direct output is missing a valid Summary section.")
    points = document.key_points or []
    if include_key_points and not minimum_points <= len(points) <= maximum_points:
        raise ProcessingError("Direct output has an invalid Key Points section.")
    return SummaryOutput(
        summary=document.summary if include_summary else None,
        key_points=points if include_key_points else None,
    )


def _summarize_chunks(
    transcript: Transcript,
    target_chunk_words: int,
    max_retries: int,
    language: str,
    progress: ProgressReporter | None = None,
    cancel_check: CancelCheck | None = None,
) -> list[SummaryChunk]:
    """Map step: chunk the transcript and produce a validated summary per chunk.

    Retries each chunk summary up to `1 + max_retries` times until it passes a
    simple minimum-length validation heuristic.

    Raises `ProcessingError` if chunking yields no chunks or a chunk summary fails
    validation after all retries.
    """

    if progress:
        progress(
            ProgressEvent(phase="chunking", message="Splitting transcript into chunks")
        )
    _ensure_not_cancelled(cancel_check)
    chunks = chunk_transcript(transcript, target_chunk_words=target_chunk_words)

    if not chunks:
        raise ProcessingError("No transcript chunks available for summarization.")

    summarizer = ChunkSummarizer()
    total_chunks = len(chunks)
    mapped: list[SummaryChunk] = []
    retry_count = 0
    if progress:
        progress(
            ProgressEvent(
                phase="chunking",
                message=f"Split transcript into {total_chunks} chunks",
                completed=total_chunks,
                total=total_chunks,
                metrics={"chunk_count": total_chunks},
            )
        )

    for chunk in tqdm(chunks, desc="Summarizing chunks", unit="chunk"):
        _ensure_not_cancelled(cancel_check)
        if progress:
            progress(
                ProgressEvent(
                    phase="summarizing",
                    message=f"Summarizing chunk {chunk.index + 1} of {total_chunks}",
                    completed=chunk.index,
                    total=total_chunks,
                )
            )
        context = build_chunk_context(chunk, total_chunks, language=language)

        # 8 words minimum, or 8 words + 1 per 30 words of source
        min_summary_words = max(8, chunk.word_count // 30 + 8)

        summary: str | None = None
        for attempt in range(1 + max_retries):
            _ensure_not_cancelled(cancel_check)
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
            retry_count += 1
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
        if progress:
            progress(
                ProgressEvent(
                    phase="summarizing",
                    message=f"Summarized chunk {chunk.index + 1} of {total_chunks}",
                    completed=chunk.index + 1,
                    total=total_chunks,
                    metrics={"retry_count": retry_count},
                )
            )

    return mapped


def _aggregate_chunk_results(
    mapped_results: Sequence[SummaryChunk],
    max_retries: int,
    language: str,
    summary_length: str,
    progress: ProgressReporter | None = None,
    cancel_check: CancelCheck | None = None,
) -> str:
    """Reduce step: merge chunk summaries into a single validated final summary.

    The minimum word target is derived from the mapped chunk summaries (not the
    original transcript text) using the chosen `summary_length` profile.

    Raises `ProcessingError` if no chunk summaries are provided or the reduce
    summary fails validation after all retries.
    """

    if not mapped_results:
        raise ProcessingError(
            "Aggregation failed because no chunk summaries were provided."
        )

    logger.info("Starting reduce step for %d chunks.", len(mapped_results))
    aggregator = AggregateSummarizer()
    chunk_count = len(mapped_results)
    length_profile = _get_length_profile(summary_length)
    source_words = sum(len(entry.summary.split()) for entry in mapped_results)
    min_summary_words = int(max(40, source_words * length_profile["min_words_ratio"]))

    reduce_context = build_reduce_context(
        chunk_count=chunk_count,
        length_guidance=length_profile["length_guidance"],
        min_summary_words=min_summary_words,
        language=language,
    )

    formatted_chunks = format_mapped_chunks(mapped_results)

    for attempt in range(1 + max_retries):
        _ensure_not_cancelled(cancel_check)
        if progress:
            progress(
                ProgressEvent(
                    phase="reducing",
                    message="Combining chunk summaries",
                    completed=attempt,
                    total=max_retries + 1,
                )
            )
        prediction = aggregator(
            mapped_chunks=formatted_chunks,
            reduce_context=reduce_context,
        )
        summary_parts = normalize_sequence(getattr(prediction, "final_summary", ""))
        summary = " ".join(summary_parts).strip()

        if is_valid_tldr(summary, min_summary_words):
            if progress:
                progress(
                    ProgressEvent(
                        phase="reducing",
                        message="Combined chunk summaries",
                        completed=attempt + 1,
                        total=max_retries + 1,
                    )
                )
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
    progress: ProgressReporter | None = None,
    cancel_check: CancelCheck | None = None,
) -> list[str]:
    """Generate bullet-style key points for the transcript.

    Uses the `summary_length` profile to determine an acceptable bullet count
    range and retries up to `1 + max_retries` times until at least one bullet is
    produced.

    Raises `ProcessingError` if the input summary is empty or generation fails
    after all retries.
    """

    if not mapped_results and not final_summary:
        raise ProcessingError(
            "Cannot generate key points without mapped chunks or summary."
        )
    if not final_summary.strip():
        raise ProcessingError("Cannot generate key points from an empty summary.")

    generator = KeyPointsGenerator()
    formatted_chunks = format_mapped_chunks(mapped_results) if mapped_results else ""

    min_bullets, max_bullets = _get_length_profile(summary_length)["bullet_ranges"]
    guidance = build_key_points_guidance(
        min_bullets=min_bullets,
        max_bullets=max_bullets,
        language=language,
    )

    for attempt in range(1 + max_retries):
        _ensure_not_cancelled(cancel_check)
        if progress:
            progress(
                ProgressEvent(
                    phase="key_points",
                    message="Generating key points",
                    completed=attempt,
                    total=max_retries + 1,
                )
            )
        prediction = generator(
            mapped_chunks=formatted_chunks,
            final_summary=final_summary,
            guidance=guidance,
        )
        key_points = normalize_key_points(getattr(prediction, "key_points", []))
        if key_points:
            if progress:
                progress(
                    ProgressEvent(
                        phase="key_points",
                        message="Generated key points",
                        completed=attempt + 1,
                        total=max_retries + 1,
                    )
                )
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
    progress: ProgressReporter | None = None,
    cancel_check: CancelCheck | None = None,
) -> SummaryOutput:
    """Generate transcript summary and optional key points.

    Returns a `SummaryOutput` where `summary` and/or `key_points` may be `None`
    based on the `include_summary` / `include_key_points` flags.

    Raises `ProcessingError` when chunking fails or model outputs cannot be
    validated after retries.
    """

    language = transcript.language.name
    mapped_results = _summarize_chunks(
        transcript=transcript,
        target_chunk_words=target_chunk_words,
        max_retries=max_retries,
        language=language,
        progress=progress,
        cancel_check=cancel_check,
    )
    final_summary = _aggregate_chunk_results(
        mapped_results=mapped_results,
        max_retries=max_retries,
        language=language,
        summary_length=summary_length,
        progress=progress,
        cancel_check=cancel_check,
    )
    key_points: list[str] | None = None
    if include_key_points:
        key_points = _generate_key_points(
            mapped_results=mapped_results,
            final_summary=final_summary,
            max_retries=max_retries,
            language=language,
            summary_length=summary_length,
            progress=progress,
            cancel_check=cancel_check,
        )

    return SummaryOutput(
        summary=final_summary if include_summary else None,
        key_points=key_points if include_key_points else None,
    )
