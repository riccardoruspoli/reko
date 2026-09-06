import json
import logging
from collections.abc import Callable
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

from iso639 import Lang
from pytubefix import Playlist, YouTube
from pytubefix.extract import video_id as extract_video_id
from youtube_transcript_api import YouTubeTranscriptApi

from reko.adapters.transcript_cache import TranscriptCache
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
    path = parsed.path.rstrip("/").lower()

    # If the URL is the watch endpoint, treat it as a single video even if "list" is present.
    if path.endswith("/watch") or path == "watch":
        return False

    # Explicit playlist endpoint
    if path.endswith("/playlist") or path == "playlist":
        return True

    # Fallback: if there's a list query param and it's not a watch URL, consider it a playlist.
    return bool(query.get("list", [""])[0])


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


def get_video_id(url: str) -> str | None:
    try:
        return extract_video_id(url)
    except Exception:
        return None


def get_video_title(video: YouTube, url: str) -> str | None:
    try:
        title = str(video.title).strip()
    except Exception as error:
        logger.debug(
            "Pytubefix could not load the title for video %s: %s",
            video.video_id,
            error,
        )
    else:
        return title or None

    oembed_url = "https://www.youtube.com/oembed?" + urlencode(
        {"url": url, "format": "json"}
    )
    try:
        with urlopen(oembed_url, timeout=5) as response:  # noqa: S310
            payload = json.load(response)
        title = payload.get("title")
    except (OSError, ValueError, TypeError) as error:
        logger.warning(
            "Could not load the title for video %s: %s",
            video.video_id,
            error,
        )
        return None
    return title.strip() if isinstance(title, str) and title.strip() else None


def get_transcription(
    video: YouTube,
    target_language: Lang,
    *,
    refresh: bool = False,
    cache: TranscriptCache | None = None,
    cache_status: Callable[[bool], None] | None = None,
    title: str | None = None,
) -> Transcript:
    """Fetch a transcript in the requested language, falling back to English."""

    cache = cache or TranscriptCache()
    if not refresh:
        cached_record = cache.load(video.video_id, target_language)
        if cached_record is not None:
            if cache_status:
                cache_status(True)
            logger.info(
                "Using cached %s transcript for video %s.",
                cached_record.transcript.language.name,
                video.video_id,
            )
            return cached_record.transcript

    if cache_status:
        cache_status(False)

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
        result = Transcript(
            segments=segments, language=resolve_language(transcript.language_code)
        )
    except Exception as e:
        raise TranscriptError(
            f"Failed to fetch transcript for video {video.video_id} (tried: {', '.join(language_priority)})."
        ) from e

    try:
        cache.save(video.video_id, target_language, result, title=title)
    except OSError as error:
        logger.warning(
            "Could not save transcript cache for %s: %s", video.video_id, error
        )
    return result
