import logging
import os

from pytubefix import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter

logger = logging.getLogger(__name__)


def get_video_data(url: str) -> tuple[str, str]:
    try:
        video = YouTube(url)
        return video.video_id, video.title
    except Exception:
        logger.exception("Error fetching video metadata for %s", url)
        exit(1)


def get_transcription(video_id: str) -> str:
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript = ytt_api.fetch(video_id)
        formatter = JSONFormatter()
        return formatter.format_transcript(transcript)
    except Exception:
        logger.exception("Error fetching transcript for video %s", video_id)
        exit(1)


def save_summary(title: str, summary: str) -> None:
    os.makedirs("summary", exist_ok=True)
    with open(f"summary/{title}.md", "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info("Saved summary for %s", title)
