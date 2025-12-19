import logging
from urllib.parse import parse_qs, urlparse

from iso639 import Lang
from pytubefix import Playlist, YouTube
from youtube_transcript_api import YouTubeTranscriptApi

from reko.core.errors import TranscriptError, YouTubeError
from reko.core.models import Transcript, TranscriptSegment
from reko.core.transcript import resolve_language

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


def get_playlist_videos(url: str) -> list[YouTube]:
    try:
        playlist = Playlist(url)
        return list(playlist.videos)
    except Exception as e:
        raise YouTubeError(f"Failed to fetch videos for playlist: {url}") from e


def get_video(url: str) -> YouTube:
    try:
        return YouTube(url)
    except Exception as e:
        raise YouTubeError(f"Failed to fetch video metadata: {url}") from e


def get_transcription(video: YouTube, target_language: Lang) -> Transcript:
    """Fetch a transcript in the requested language, falling back to English."""

    ytt_api = YouTubeTranscriptApi()
    try:
        if not target_language.pt1:
            raise TranscriptError(
                f"Target language {target_language!r} does not have an ISO 639-1 code."
            )
        language_priority = (
            [target_language.pt1, "en"] if target_language.pt1 != "en" else ["en"]
        )
        transcript = ytt_api.fetch(video.video_id, languages=language_priority)
        segments = [
            TranscriptSegment(
                text=snippet.text.strip(),
                start=float(snippet.start),
                duration=float(snippet.duration),
            )
            for snippet in transcript
            if snippet.text and snippet.text.strip()
        ]
        return Transcript(
            segments=segments, language=resolve_language(transcript.language_code)
        )
    except Exception as e:
        raise TranscriptError(
            f"Failed to fetch transcript for video {video.video_id} (tried: {', '.join(language_priority)})."
        ) from e
