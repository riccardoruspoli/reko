import logging
import os
import re
import time

from pytubefix import YouTube

from reko.adapters.dspy.config import dspy_context
from reko.adapters.storage import is_summary_complete, save_summary
from reko.adapters.youtube import (
    get_playlist_videos,
    get_transcription,
    get_video,
    is_playlist,
)
from reko.core.errors import InputError
from reko.core.models import SummaryConfig, SummaryDocument
from reko.core.summarizer import generate_summary_outputs
from reko.core.translation import translate_key_points, translate_text

logger = logging.getLogger(__name__)
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _summarize_video_to_markdown(video: YouTube, config: SummaryConfig) -> str:
    logger.info("Processing video %s", video.video_id)

    summary_path = os.path.join("summary", f"{video.video_id}.md")
    logger.debug("Summary output path: %s", summary_path)

    if is_summary_complete(summary_path, config) and not config.force:
        logger.info(
            "Summary with requested sections already exists. Use --force to regenerate."
        )
        with open(summary_path, encoding="utf-8") as f:
            return f.read()

    logger.debug("Existing summary missing requested sections; regenerating.")

    transcript = get_transcription(video, config.target_language)
    logger.debug("Transcript contains %d words.", transcript.word_count)

    logger.info(
        "Transcript language resolved to %s (target %s).",
        transcript.language.name,
        config.target_language.name,
    )

    with dspy_context(config):
        output = generate_summary_outputs(
            transcript=transcript,
            target_chunk_words=config.target_chunk_words,
            include_summary=config.include_summary,
            include_key_points=config.include_key_points,
            max_retries=config.max_retries,
            summary_length=config.length,
        )

        if config.target_language.pt1 != transcript.language.pt1:
            logger.info(
                "Translating outputs from %s to %s",
                transcript.language.name,
                config.target_language.name,
            )
            if output.summary is not None:
                output.summary = translate_text(
                    output.summary,
                    target_language=config.target_language.name,
                    max_retries=config.max_retries,
                )
            if output.key_points is not None:
                output.key_points = translate_key_points(
                    output.key_points,
                    target_language=config.target_language.name,
                    max_retries=config.max_retries,
                )

    markdown_summary = SummaryDocument(
        title=video.title,
        summary=output.summary,
        key_points=output.key_points,
    ).to_markdown()

    logger.debug("Output generated with %d characters", len(markdown_summary))
    if config.save_output:
        save_summary(video.video_id, markdown_summary)
    return markdown_summary


def _summarize_video(video: YouTube, config: SummaryConfig) -> None:
    markdown_summary = _summarize_video_to_markdown(video, config)
    if config.print_output:
        print(markdown_summary)


def summarize(input_value: str, config: SummaryConfig) -> None:
    """Summarize either a single URL or a text file containing one URL per line."""

    if os.path.isfile(input_value):
        logger.info("Input is a batch file; processing multiple URLs.")
        try:
            with open(input_value, encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]
        except OSError as e:
            raise InputError(f"Failed to read batch file: {input_value}") from e
        if not urls:
            raise InputError(f"No URLs found in batch file: {input_value}")
        for url in urls:
            _summarize_video(get_video(url), config)
        return

    if is_playlist(input_value):
        logger.info("Input is a playlist; processing all videos in the playlist.")
        videos = get_playlist_videos(input_value)
        for video in videos:
            _summarize_video(video, config)
        return

    _summarize_video(get_video(input_value), config)


def summarize_one_to_markdown(url: str, config: SummaryConfig) -> str:
    """Summarize a single YouTube video URL and return markdown."""

    if os.path.isfile(url):
        raise InputError("Expected a URL, got a file path.")
    if is_playlist(url):
        raise InputError("Playlists are not supported by the web API.")
    return _summarize_video_to_markdown(get_video(url), config)


def summarize_one_with_stats(
    url: str, config: SummaryConfig
) -> tuple[str, int, int, float, str]:
    """Summarize a single YouTube video URL and return (markdown, input_words, output_words, elapsed_seconds, video_id)."""

    if os.path.isfile(url):
        raise InputError("Expected a URL, got a file path.")
    if is_playlist(url):
        raise InputError("Playlists are not supported by the web API.")

    video = get_video(url)
    started_at = time.perf_counter()

    transcript = get_transcription(video, config.target_language)
    with dspy_context(config):
        output = generate_summary_outputs(
            transcript=transcript,
            target_chunk_words=config.target_chunk_words,
            include_summary=config.include_summary,
            include_key_points=config.include_key_points,
            max_retries=config.max_retries,
            summary_length=config.length,
        )

        if config.target_language.pt1 != transcript.language.pt1:
            if output.summary is not None:
                output.summary = translate_text(
                    output.summary,
                    target_language=config.target_language.name,
                    max_retries=config.max_retries,
                )
            if output.key_points is not None:
                output.key_points = translate_key_points(
                    output.key_points,
                    target_language=config.target_language.name,
                    max_retries=config.max_retries,
                )

    markdown_summary = SummaryDocument(
        title=video.title,
        summary=output.summary,
        key_points=output.key_points,
    ).to_markdown()

    elapsed_seconds = time.perf_counter() - started_at
    input_words = int(transcript.word_count)
    output_words = _count_words(markdown_summary)
    return markdown_summary, input_words, output_words, elapsed_seconds, video.video_id
