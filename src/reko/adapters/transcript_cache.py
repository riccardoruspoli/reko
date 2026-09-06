"""Persistent, language-aware cache for raw YouTube transcripts."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from iso639 import Lang

from reko.core.models import Transcript, TranscriptSegment
from reko.core.transcript import resolve_language

logger = logging.getLogger(__name__)
_CACHE_VERSION = 1


def default_data_dir() -> Path:
    """Return the application data directory, overridable for containers/tests."""

    return Path(os.environ.get("REKO_DATA_DIR", "data"))


class TranscriptCache:
    """Store raw transcript segments by video and requested language."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.root = (data_dir or default_data_dir()) / "transcripts"

    def load(self, video_id: str, requested_language: Lang) -> Transcript | None:
        path = self._path(video_id, requested_language)
        try:
            with path.open(encoding="utf-8") as cache_file:
                payload = json.load(cache_file)
            return self._decode(payload, video_id, requested_language)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.warning("Ignoring invalid transcript cache file %s: %s", path, error)
            return None

    def save(
        self, video_id: str, requested_language: Lang, transcript: Transcript
    ) -> None:
        path = self._path(video_id, requested_language)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _CACHE_VERSION,
            "video_id": video_id,
            "requested_language": requested_language.pt1,
            "resolved_language": transcript.language.pt1,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "segments": [asdict(segment) for segment in transcript.segments],
        }

        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary_file:
            json.dump(
                payload, temporary_file, ensure_ascii=False, separators=(",", ":")
            )
            temporary_path = Path(temporary_file.name)
        try:
            temporary_path.replace(path)
        except OSError:
            temporary_path.unlink(missing_ok=True)
            raise

    def _path(self, video_id: str, requested_language: Lang) -> Path:
        language_code = requested_language.pt1
        if not language_code:
            raise ValueError("Requested language must have an ISO 639-1 code.")
        return self.root / video_id / f"{language_code}.json"

    @staticmethod
    def _decode(payload: object, video_id: str, requested_language: Lang) -> Transcript:
        if not isinstance(payload, dict):
            raise ValueError("Cache payload is not an object.")
        if payload.get("version") != _CACHE_VERSION:
            raise ValueError("Unsupported cache version.")
        if payload.get("video_id") != video_id:
            raise ValueError("Cache video ID does not match.")
        if payload.get("requested_language") != requested_language.pt1:
            raise ValueError("Cache language does not match.")

        resolved_language = resolve_language(str(payload["resolved_language"]))
        raw_segments = payload["segments"]
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError("Cache contains no transcript segments.")
        segments = [
            TranscriptSegment(
                text=str(segment["text"]),
                start=float(segment["start"]),
                duration=float(segment["duration"]),
            )
            for segment in raw_segments
            if isinstance(segment, dict) and str(segment.get("text", "")).strip()
        ]
        if not segments:
            raise ValueError("Cache contains no valid transcript segments.")
        return Transcript(segments=segments, language=resolved_language)
