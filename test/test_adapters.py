from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from iso639 import Lang

from reko.adapters import youtube
from reko.adapters.dspy import config as dspy_config
from reko.adapters.dspy import modules
from reko.adapters.transcript_cache import CachedTranscript
from reko.core.errors import TranscriptError, YouTubeError
from reko.core.models import SummaryConfig, Transcript, TranscriptSegment


def summary_config(model: str = "ollama/test", **changes: object) -> SummaryConfig:
    values: dict[str, object] = {
        "host": None,
        "model": model,
        "target_chunk_words": 100,
        "max_tokens": 321,
        "temperature": 1.0,
        "force": False,
        "include_summary": True,
        "include_key_points": True,
        "max_retries": 0,
        "print_output": False,
        "save_output": False,
        "target_language": Lang("en"),
        "length": "medium",
        "think": True,
        "reasoning_effort": None,
    }
    values.update(changes)
    return SummaryConfig(**values)  # type: ignore[arg-type]


def test_dspy_model_configuration_and_context_creation(monkeypatch) -> None:
    assert dspy_config._model_name("openai/GPT-5") == "gpt-5"
    assert dspy_config._model_name("MODEL") == "model"
    assert dspy_config._is_openai_gpt5_model("openai/gpt-5.2")
    assert not dspy_config._is_openai_gpt5_model("openai/gpt-5-chat")
    assert dspy_config._is_legacy_gpt5_model("openai/gpt-5-turbo")

    legacy = dspy_config._gpt5_lm_kwargs(
        summary_config("openai/gpt-5-turbo", reasoning_effort="medium")
    )
    assert legacy["max_tokens"] == 321
    assert (
        dspy_config._gpt5_lm_kwargs(
            summary_config("openai/gpt-5-turbo", temperature=0.2)
        )["temperature"]
        is None
    )
    modern = dspy_config._gpt5_lm_kwargs(summary_config("openai/gpt-5.2"))
    assert modern["max_completion_tokens"] == 321
    with pytest.raises(ValueError, match="reasoning_effort"):
        dspy_config._gpt5_lm_kwargs(
            summary_config("openai/gpt-5.2", reasoning_effort="invalid")
        )
    with pytest.raises(ValueError, match="does not support"):
        dspy_config._gpt5_lm_kwargs(
            summary_config("openai/gpt-5-turbo", reasoning_effort="xhigh")
        )

    created: dict[str, object] = {}

    def fake_lm(**kwargs):
        created["lm"] = kwargs
        return "lm"

    def fake_context(**kwargs):
        created["context"] = kwargs
        return nullcontext()

    monkeypatch.setattr(dspy_config.dspy, "LM", fake_lm)
    monkeypatch.setattr(dspy_config, "JSONAdapter", lambda: "adapter")
    monkeypatch.setattr(dspy_config.dspy, "context", fake_context)
    result = dspy_config.dspy_context(summary_config(host="http://localhost"))

    assert created["lm"]["api_base"] == "http://localhost"
    assert created["lm"]["think"] is True
    assert created["context"] == {"lm": "lm", "adapter": "adapter"}
    assert result is not None


def test_dspy_modules_forward_the_expected_signature_arguments(monkeypatch) -> None:
    monkeypatch.setattr(
        modules.dspy,
        "Predict",
        lambda signature: lambda **kwargs: {"signature": signature, **kwargs},
    )

    assert modules.ChunkSummarizer()("text", "context")["chunk_text"] == "text"
    assert (
        modules.AggregateSummarizer()("chunks", "context")["mapped_chunks"] == "chunks"
    )
    assert (
        modules.KeyPointsGenerator()("chunks", "summary", "guide")["guidance"]
        == "guide"
    )
    assert (
        modules.Translator()("source", "Italian", "guide")["target_language"]
        == "Italian"
    )


def test_youtube_url_detection_and_wrapped_provider_errors(monkeypatch) -> None:
    assert youtube.is_playlist("https://youtube.test/playlist?list=123")
    assert not youtube.is_playlist("https://youtube.test/watch?v=1&list=123")
    assert not youtube.is_playlist("not a url")
    assert youtube.get_video_id("https://www.youtube.com/watch?v=abc") == "abc"
    assert youtube.get_video_id("https://youtu.be/abc") == "abc"
    assert youtube.get_video_id("https://www.youtube.com/shorts/abc") == "abc"
    assert youtube.get_video_id("https://example.test/video") is None

    monkeypatch.setattr(youtube, "YouTube", lambda url: {"url": url})
    monkeypatch.setattr(
        youtube, "Playlist", lambda url: SimpleNamespace(videos=["one", "two"])
    )
    assert youtube.get_video("https://example.test") == {"url": "https://example.test"}
    assert youtube.get_playlist_videos("https://example.test") == ["one", "two"]

    monkeypatch.setattr(
        youtube, "YouTube", lambda url: (_ for _ in ()).throw(RuntimeError())
    )
    monkeypatch.setattr(
        youtube, "Playlist", lambda url: (_ for _ in ()).throw(RuntimeError())
    )
    with pytest.raises(YouTubeError):
        youtube.get_video("url")
    with pytest.raises(YouTubeError):
        youtube.get_playlist_videos("url")


def test_get_transcription_uses_cache_fetches_and_handles_provider_errors(
    monkeypatch,
) -> None:
    cached = Transcript([TranscriptSegment("cached", 0, 1)], Lang("en"))
    cache = SimpleNamespace(load=lambda *_: CachedTranscript(cached, "Cached title"))
    video = SimpleNamespace(video_id="abc")
    cache_status: list[bool] = []
    assert (
        youtube.get_transcription(
            video, Lang("en"), cache=cache, cache_status=cache_status.append
        )
        is cached
    )
    assert cache_status == [True]

    saved: list[Transcript] = []
    cache = SimpleNamespace(
        load=lambda *_: None,
        save=lambda *_args, **_kwargs: saved.append(_args[-1]),
    )
    snippets = [
        SimpleNamespace(text=" first ", start=1, duration=2),
        SimpleNamespace(text="", start=3, duration=2),
    ]

    class FakeTranscript:
        language_code = "en"

        def __iter__(self):
            return iter(snippets)

    monkeypatch.setattr(
        youtube,
        "YouTubeTranscriptApi",
        lambda: SimpleNamespace(fetch=lambda *_args, **_kwargs: FakeTranscript()),
    )
    result = youtube.get_transcription(
        video, Lang("it"), cache=cache, cache_status=cache_status.append
    )
    assert result.segments[0].text == "first"
    assert result.language.pt1 == "en"
    assert saved == [result]
    assert cache_status[-1] is False

    monkeypatch.setattr(
        youtube,
        "YouTubeTranscriptApi",
        lambda: SimpleNamespace(
            fetch=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("no transcript")
            )
        ),
    )
    with pytest.raises(TranscriptError, match="abc"):
        youtube.get_transcription(video, Lang("en"), refresh=True, cache=cache)
