from __future__ import annotations

import logging
import os

from iso639 import Lang

from ..adapters.dspy import configure_dspy
from ..adapters.storage import save_summary
from .chunking import get_transcript_words_count
from .errors import InputError
from .models import SummaryConfig
from .summarizer import generate_summary_outputs, translate_key_points, translate_text
from .text_utils import build_markdown
from .youtube_client import (
    get_playlist_videos,
    get_transcription,
    get_video_data,
    is_playlist,
)

logger = logging.getLogger(__name__)


def _summary_has_section(section: str, content: str) -> bool:
    """Check if the summary content has a specific section."""
    return f"## {section}" in content


def _exist_summary(summary_path: str) -> bool:
    """Check if the summary file exists."""
    return os.path.exists(summary_path)


def _is_summary_complete(summary_path: str, config: SummaryConfig) -> bool:
    """Check if the existing summary file contains the requested sections."""
    with open(summary_path, "r", encoding="utf-8") as f:
        existing = f.read()

    return (
        not config.include_summary or _summary_has_section("Summary", existing)
    ) and (
        not config.include_key_points or _summary_has_section("Key Points", existing)
    )


def _summarize_video_url(url: str, config: SummaryConfig) -> None:
    video_id, video_title = get_video_data(url)
    logger.info("Processing video %s", video_id)

    summary_path = os.path.join("summary", f"{video_id}.md")
    logger.debug("Summary output path: %s", summary_path)

    if not config.force and _exist_summary(summary_path):
        if _is_summary_complete(summary_path, config):
            logger.info(
                "Summary with requested sections already exists. Use --force to regenerate."
            )
            with open(summary_path, "r", encoding="utf-8") as f:
                existing = f.read()
            if config.print_output:
                print(existing)
            return

        logger.debug("Existing summary missing requested sections; regenerating.")

    transcript, transcript_language = get_transcription(
        video_id, config.target_language.pt1
    )
    logger.debug(
        "Transcript contains %d words.", get_transcript_words_count(transcript)
    )

    try:
        resolved_transcript_lang = Lang(transcript_language).pt1
        transcript_lang_name = Lang(transcript_language).name
    except Exception:
        resolved_transcript_lang = transcript_language
        transcript_lang_name = transcript_language

    logger.info(
        "Transcript language resolved to %s (target %s).",
        transcript_lang_name,
        config.target_language.name,
    )

    configure_dspy(config)

    final_summary, key_points = generate_summary_outputs(
        serialized_transcript=transcript,
        target_chunk_words=config.target_chunk_words,
        include_summary=config.include_summary,
        include_key_points=config.include_key_points,
        max_retries=config.max_retries,
        output_language=transcript_lang_name,
        summary_length=config.length,
    )

    if config.target_language.pt1 != resolved_transcript_lang:
        logger.info(
            "Translating outputs from %s to %s",
            transcript_lang_name,
            config.target_language.name,
        )
        if config.include_summary:
            final_summary = translate_text(
                final_summary,
                target_language=config.target_language.name,
                max_retries=config.max_retries,
            )
        if config.include_key_points:
            key_points = translate_key_points(
                key_points,
                target_language=config.target_language.name,
                max_retries=config.max_retries,
            )

    markdown_summary = build_markdown(
        video_title,
        final_summary if config.include_summary else None,
        key_points if config.include_key_points else None,
    )

    logger.debug("Output generated with %d characters", len(markdown_summary))
    if config.print_output:
        print(markdown_summary)
    if config.save_output:
        save_summary(video_id, markdown_summary)


def summarize(input_value: str, config: SummaryConfig) -> None:
    """Summarize either a single URL or a text file containing one URL per line."""
    if os.path.isfile(input_value):
        logger.info("Input is a batch file; processing multiple URLs.")
        try:
            with open(input_value, "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]
        except OSError as e:
            raise InputError(f"Failed to read batch file: {input_value}") from e
        if not urls:
            raise InputError(f"No URLs found in batch file: {input_value}")
        for url in urls:
            _summarize_video_url(url, config)
        return

    if is_playlist(input_value):
        logger.info("Input is a playlist; processing all videos in the playlist.")
        urls = get_playlist_videos(input_value)
        for url in urls:
            _summarize_video_url(url, config)
        return

    _summarize_video_url(input_value, config)
