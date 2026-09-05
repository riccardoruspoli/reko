from __future__ import annotations

from pathlib import Path

from iso639 import Lang

from reko.adapters.transcript_cache import TranscriptCache
from reko.adapters.youtube import get_transcription
from reko.core.models import Transcript, TranscriptSegment


def make_transcript(language: str = "en") -> Transcript:
    return Transcript(
        language=Lang(language),
        segments=[
            TranscriptSegment(text="First segment.", start=1.25, duration=2.5),
            TranscriptSegment(text="Second segment.", start=3.75, duration=1.0),
        ],
    )


def test_cache_round_trip_preserves_raw_segments(tmp_path: Path) -> None:
    cache = TranscriptCache(tmp_path)
    transcript = make_transcript()

    cache.save("video-123", Lang("it"), transcript)

    restored = cache.load("video-123", Lang("it"))

    assert restored is not None
    assert restored.language.pt1 == "en"
    assert restored.segments == transcript.segments
    assert (tmp_path / "transcripts" / "video-123" / "it.json").is_file()


def test_cache_is_scoped_to_requested_language(tmp_path: Path) -> None:
    cache = TranscriptCache(tmp_path)
    cache.save("video-123", Lang("it"), make_transcript("en"))

    assert cache.load("video-123", Lang("it")) is not None
    assert cache.load("video-123", Lang("en")) is None


def test_invalid_cache_is_treated_as_a_cache_miss(tmp_path: Path) -> None:
    path = tmp_path / "transcripts" / "video-123" / "en.json"
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")

    assert TranscriptCache(tmp_path).load("video-123", Lang("en")) is None


def test_get_transcription_uses_cached_value_without_creating_api_client(
    tmp_path: Path, monkeypatch
) -> None:
    cache = TranscriptCache(tmp_path)
    cached = make_transcript()
    cache.save("video-123", Lang("en"), cached)

    def fail_if_called() -> None:
        raise AssertionError("YouTubeTranscriptApi must not be created on a cache hit")

    monkeypatch.setattr("reko.adapters.youtube.YouTubeTranscriptApi", fail_if_called)
    video = type("Video", (), {"video_id": "video-123"})()

    transcript = get_transcription(video, Lang("en"), cache=cache)

    assert transcript == cached
