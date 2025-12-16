import logging
from urllib.parse import parse_qs, urlparse

from pytubefix import Playlist, YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter

from .errors import TranscriptError, YouTubeError

logger = logging.getLogger(__name__)


def is_playlist(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    query = parse_qs(parsed.query)
    if query.get("list", [""])[0]:
        return True

    path = parsed.path.rstrip("/").lower()
    return path.endswith("/playlist")


def get_playlist_videos(url: str) -> list[str]:
    try:
        playlist = Playlist(url)
        return [video.watch_url for video in playlist.videos]
    except Exception as e:
        raise YouTubeError(f"Failed to fetch videos for playlist: {url}") from e


def get_video_data(url: str) -> tuple[str, str]:
    try:
        video = YouTube(url)
        return video.video_id, video.title
    except Exception as e:
        raise YouTubeError(f"Failed to fetch video metadata: {url}") from e


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
    except Exception as e:
        raise TranscriptError(
            f"Failed to fetch transcript for video {video_id} (tried: {', '.join(language_priority)})."
        ) from e
