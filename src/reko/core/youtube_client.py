import logging
import os

from pytubefix import Playlist, YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter

logger = logging.getLogger(__name__)


def is_playlist(url: str) -> bool:
    try:
        Playlist(url)
        return True
    except Exception:
        logger.exception("Error checking if URL (%s) is a playlist.", url)
        exit(1)


def get_playlist_videos(url: str) -> list[tuple[str, str]]:
    try:
        playlist = Playlist(url)
        return [video.watch_url for video in playlist.videos]
    except Exception:
        logger.exception("Error fetching videos from playlist %s", url)
        exit(1)


def get_video_data(url: str) -> tuple[str, str]:
    try:
        video = YouTube(url)
        return video.video_id, video.title
    except Exception:
        logger.exception("Error fetching video metadata for %s", url)
        exit(1)


def get_transcription(video_id: str, target_language: str) -> tuple[str, str]:
    """Fetch a transcript in the requested language, falling back to English.

    Returns the serialized transcript and the language code it was fetched in.
    """
    ytt_api = YouTubeTranscriptApi()
    try:
        language_priority = (
            [target_language, "en"] if target_language != "en" else ["en"]
        )
        transcript = ytt_api.fetch(video_id, languages=language_priority)
        formatter = JSONFormatter()
        return formatter.format_transcript(transcript), transcript.language_code
    except Exception:
        logger.exception("Error fetching transcript for video %s", video_id)
        exit(1)


def save_summary(id: str, summary: str) -> None:
    os.makedirs("summary", exist_ok=True)
    with open(f"summary/{id}.md", "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info("Saved summary as %s.md", id)
