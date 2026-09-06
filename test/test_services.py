from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from iso639 import Lang

from reko.adapters.transcript_cache import CachedTranscript
from reko.core import services
from reko.core.errors import InputError, JobCancelledError
from reko.core.models import SummaryConfig, SummaryOutput, Transcript, TranscriptSegment


def config(**changes: object) -> SummaryConfig:
    values: dict[str, object] = {
        "host": None,
        "model": "ollama/test",
        "target_chunk_words": 100,
        "max_tokens": 100,
        "temperature": 1,
        "force": False,
        "include_summary": True,
        "include_key_points": True,
        "max_retries": 0,
        "print_output": False,
        "save_output": False,
        "target_language": Lang("en"),
        "length": "medium",
        "think": False,
    }
    values.update(changes)
    return SummaryConfig(**values)  # type: ignore[arg-type]


def fake_transcript(language: str = "en") -> Transcript:
    return Transcript([TranscriptSegment("one two three", 0, 1)], Lang(language))


def test_summarize_one_with_stats_reports_phases_and_translates(monkeypatch) -> None:
    translated = []
    monkeypatch.setattr(
        services, "get_video", lambda url: SimpleNamespace(video_id="id", title="Title")
    )
    monkeypatch.setattr(
        services, "get_transcription", lambda *args, **kwargs: fake_transcript("en")
    )
    monkeypatch.setattr(services, "dspy_context", lambda _: nullcontext())
    monkeypatch.setattr(
        services,
        "generate_summary_outputs",
        lambda **kwargs: SummaryOutput("summary", ["point"]),
    )
    monkeypatch.setattr(
        services,
        "translate_text",
        lambda text, **kwargs: translated.append(text) or "translated summary",
    )
    monkeypatch.setattr(
        services,
        "translate_key_points",
        lambda points, **kwargs: translated.append("points") or ["translated point"],
    )
    events = []
    markdown, input_words, output_words, elapsed, video_id = (
        services.summarize_one_with_stats(
            "https://example.test", config(target_language=Lang("it")), events.append
        )
    )

    assert "translated summary" in markdown
    assert input_words == 3
    assert output_words > 0
    assert elapsed >= 0
    assert video_id == "id"
    assert translated == ["summary", "points"]
    assert {event.phase for event in events} >= {
        "video",
        "transcript",
        "rendering",
        "translating",
    }


def test_summarize_one_with_stats_uses_cached_title_without_youtube_metadata(
    monkeypatch,
) -> None:
    cached = CachedTranscript(fake_transcript(), "Cached title")
    cache = type("Cache", (), {"load": lambda *_args: cached})()
    monkeypatch.setattr(services, "TranscriptCache", lambda: cache)
    monkeypatch.setattr(
        services,
        "get_video",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("metadata must stay local")
        ),
    )
    monkeypatch.setattr(services, "dspy_context", lambda _: nullcontext())
    monkeypatch.setattr(
        services,
        "generate_summary_outputs",
        lambda **_kwargs: SummaryOutput("summary", None),
    )

    markdown, *_ = services.summarize_one_with_stats(
        "https://www.youtube.com/watch?v=A4Ncs9gXBAI", config()
    )

    assert markdown.startswith("# Cached title")


def test_summarize_one_with_stats_backfills_missing_cached_title(monkeypatch) -> None:
    cached = CachedTranscript(fake_transcript(), None)
    saved = []
    cache = SimpleNamespace(
        load=lambda *_args: cached,
        save=lambda *args, **kwargs: saved.append((args, kwargs)),
    )
    monkeypatch.setattr(services, "TranscriptCache", lambda: cache)
    monkeypatch.setattr(
        services,
        "get_video",
        lambda *_args: SimpleNamespace(video_id="A4Ncs9gXBAI", title="Recovered title"),
    )
    monkeypatch.setattr(services, "dspy_context", lambda _: nullcontext())
    monkeypatch.setattr(
        services,
        "generate_summary_outputs",
        lambda **_kwargs: SummaryOutput("summary", None),
    )

    markdown, *_ = services.summarize_one_with_stats(
        "https://www.youtube.com/watch?v=A4Ncs9gXBAI", config()
    )

    assert markdown.startswith("# Recovered title")
    assert saved == [
        (
            ("A4Ncs9gXBAI", Lang("en"), cached.transcript),
            {"title": "Recovered title"},
        )
    ]


def test_service_guards_and_cancellation(monkeypatch, tmp_path) -> None:
    source = tmp_path / "urls.txt"
    source.write_text("url", encoding="utf-8")
    with pytest.raises(InputError, match="file path"):
        services.summarize_one_with_stats(str(source), config())
    monkeypatch.setattr(services, "is_playlist", lambda _: True)
    with pytest.raises(InputError, match="Playlists"):
        services.summarize_one_with_stats("https://example.test", config())

    monkeypatch.setattr(services, "is_playlist", lambda _: False)
    with pytest.raises(JobCancelledError):
        services.summarize_one_with_stats(
            "https://example.test", config(), cancel_check=lambda: True
        )


def test_summarize_dispatches_file_playlist_and_single_video(
    monkeypatch, tmp_path
) -> None:
    calls = []
    monkeypatch.setattr(
        services, "_summarize_video", lambda video, cfg: calls.append(video)
    )
    monkeypatch.setattr(services, "get_video", lambda url: f"video:{url}")
    monkeypatch.setattr(services, "get_playlist_videos", lambda url: ["one", "two"])

    source = tmp_path / "urls.txt"
    source.write_text("first\n\nsecond\n", encoding="utf-8")
    services.summarize(str(source), config())
    assert calls == ["video:first", "video:second"]
    calls.clear()
    monkeypatch.setattr(services, "is_playlist", lambda _: True)
    services.summarize("playlist", config())
    assert calls == ["one", "two"]


def test_markdown_service_reuses_existing_output_and_saves_new_output(
    monkeypatch, capsys
) -> None:
    video = SimpleNamespace(
        video_id="id", title="Title", watch_url="https://example.test"
    )
    monkeypatch.setattr(services, "is_summary_complete", lambda *_: True)

    class ExistingFile:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return "# Existing"

    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: ExistingFile())
    assert services._summarize_video_to_markdown(video, config()) == "# Existing"

    saved = []
    monkeypatch.setattr(services, "is_summary_complete", lambda *_: False)
    monkeypatch.setattr(
        services, "get_transcription", lambda *_args, **_kwargs: fake_transcript()
    )
    monkeypatch.setattr(services, "dspy_context", lambda _: nullcontext())
    monkeypatch.setattr(
        services,
        "generate_summary_outputs",
        lambda **_kwargs: SummaryOutput("summary", ["point"]),
    )
    monkeypatch.setattr(services, "save_summary", lambda *args: saved.append(args))
    result = services._summarize_video_to_markdown(video, config(save_output=True))
    services._summarize_video(video, config(print_output=True))

    assert "## Summary" in result
    assert saved == [("id", result)]
    assert "## Summary" in capsys.readouterr().out


def test_service_uses_a_fallback_title_when_metadata_is_unavailable(
    monkeypatch,
) -> None:
    class Video:
        video_id = "id"
        watch_url = "https://example.test"

        @property
        def title(self) -> str:
            raise RuntimeError("metadata request rejected")

    monkeypatch.setattr(
        services,
        "get_transcription",
        lambda *_args, **_kwargs: fake_transcript(),
    )
    monkeypatch.setattr(services, "dspy_context", lambda _: nullcontext())
    monkeypatch.setattr(
        services,
        "generate_summary_outputs",
        lambda **_kwargs: SummaryOutput("summary", ["point"]),
    )
    monkeypatch.setattr(services, "get_video_title", lambda *_args: None)

    result = services._summarize_video_to_markdown(Video(), config())

    assert result.startswith("# YouTube video id")
