from __future__ import annotations

from pathlib import Path

import pytest
from iso639 import Lang

from reko.adapters import transcript_cache
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


def test_refresh_transcript_bypasses_cache_and_replaces_it(
    tmp_path: Path, monkeypatch
) -> None:
    cache = TranscriptCache(tmp_path)
    cache.save("video-123", Lang("en"), make_transcript())

    class FetchedTranscript(list):
        language_code = "en"

    class FakeYouTubeTranscriptApi:
        def fetch(self, video_id: str, languages: list[str]) -> FetchedTranscript:
            assert video_id == "video-123"
            assert languages == ["en"]
            return FetchedTranscript(
                [
                    type(
                        "Snippet",
                        (),
                        {"text": "Fresh transcript.", "start": 0, "duration": 4},
                    )()
                ]
            )

    monkeypatch.setattr(
        "reko.adapters.youtube.YouTubeTranscriptApi", FakeYouTubeTranscriptApi
    )
    video = type("Video", (), {"video_id": "video-123"})()

    transcript = get_transcription(video, Lang("en"), refresh=True, cache=cache)

    assert [segment.text for segment in transcript.segments] == ["Fresh transcript."]
    assert cache.load("video-123", Lang("en")) == transcript


def test_cache_validates_payloads_paths_and_data_directory(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("REKO_DATA_DIR", str(tmp_path))
    assert transcript_cache.default_data_dir() == tmp_path
    cache = TranscriptCache(tmp_path)

    with pytest.raises(ValueError, match="ISO 639-1"):
        cache._path("video", type("Language", (), {"pt1": None})())
    with pytest.raises(ValueError, match="not an object"):
        cache._decode([], "video", Lang("en"))
    with pytest.raises(ValueError, match="Unsupported cache version"):
        cache._decode({}, "video", Lang("en"))
    with pytest.raises(ValueError, match="video ID"):
        cache._decode(
            {
                "version": 1,
                "video_id": "other",
                "requested_language": "en",
                "resolved_language": "en",
                "segments": [],
            },
            "video",
            Lang("en"),
        )
    with pytest.raises(ValueError, match="language"):
        cache._decode(
            {
                "version": 1,
                "video_id": "video",
                "requested_language": "it",
                "resolved_language": "en",
                "segments": [],
            },
            "video",
            Lang("en"),
        )
    valid_prefix = {
        "version": 1,
        "video_id": "video",
        "requested_language": "en",
        "resolved_language": "en",
    }
    with pytest.raises(ValueError, match="no transcript segments"):
        cache._decode({**valid_prefix, "segments": []}, "video", Lang("en"))
    with pytest.raises(ValueError, match="no valid transcript segments"):
        cache._decode(
            {**valid_prefix, "segments": [{"text": " "}]}, "video", Lang("en")
        )


def test_cache_save_removes_temporary_file_when_replacement_fails(
    monkeypatch, tmp_path: Path
) -> None:
    cache = TranscriptCache(tmp_path)
    monkeypatch.setattr(
        Path, "replace", lambda *_args: (_ for _ in ()).throw(OSError())
    )

    with pytest.raises(OSError):
        cache.save("video", Lang("en"), make_transcript())
    assert not list((tmp_path / "transcripts" / "video").glob("tmp*"))
