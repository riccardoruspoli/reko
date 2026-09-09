from __future__ import annotations

from types import SimpleNamespace

import pytest
from iso639 import Lang

from reko.core import summarizer, translation
from reko.core.errors import JobCancelledError, ProcessingError
from reko.core.models import SummaryOutput, Transcript, TranscriptSegment

VALID_SUMMARY = " ".join(f"word{index}" for index in range(40))


def transcript() -> Transcript:
    return Transcript(
        [TranscriptSegment("one two three four five six seven eight nine ten", 0, 2)],
        Lang("en"),
    )


def test_chunk_summarization_reports_retries_and_cancellation(monkeypatch) -> None:
    calls = 0

    class FakeSummarizer:
        def __call__(self, **kwargs):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                summary="too short"
                if calls == 1
                else "one two three four five six seven eight"
            )

    monkeypatch.setattr(summarizer, "ChunkSummarizer", FakeSummarizer)
    events = []
    result = summarizer._summarize_chunks(
        transcript(), 100, 1, "English", progress=events.append
    )
    assert result[0].summary.startswith("one two")
    assert events[-1].completed == 1

    with pytest.raises(JobCancelledError):
        summarizer._summarize_chunks(
            transcript(), 100, 0, "English", cancel_check=lambda: True
        )


def test_reduce_and_key_points_cover_success_failure_and_profiles(monkeypatch) -> None:
    mapped = summarizer._summarize_chunks
    summaries = [
        SimpleNamespace(
            index=0,
            start=0,
            end=1,
            word_count=10,
            summary="one two three four five six seven eight nine ten",
        )
    ]

    class Aggregate:
        def __call__(self, **kwargs):
            return SimpleNamespace(final_summary=VALID_SUMMARY)

    class KeyPoints:
        def __call__(self, **kwargs):
            return SimpleNamespace(key_points="- First\n- Second")

    monkeypatch.setattr(summarizer, "AggregateSummarizer", Aggregate)
    monkeypatch.setattr(summarizer, "KeyPointsGenerator", KeyPoints)
    assert "word0" in summarizer._aggregate_chunk_results(
        summaries, 0, "English", "short"
    )
    assert summarizer._generate_key_points(
        summaries, "summary", 0, "English", "medium"
    ) == ["First", "Second"]
    with pytest.raises(ProcessingError, match="Unknown summary length"):
        summarizer._get_length_profile("unknown")
    with pytest.raises(ProcessingError, match="no chunk summaries"):
        summarizer._aggregate_chunk_results([], 0, "English", "short")
    with pytest.raises(ProcessingError, match="empty summary"):
        summarizer._generate_key_points(summaries, "", 0, "English", "short")
    assert mapped is summarizer._summarize_chunks


def test_generate_summary_and_translation_with_mocked_models(monkeypatch) -> None:
    class Chunk:
        def __call__(self, **kwargs):
            return SimpleNamespace(
                summary="one two three four five six seven eight nine ten"
            )

    class Aggregate:
        def __call__(self, **kwargs):
            return SimpleNamespace(final_summary=VALID_SUMMARY)

    class KeyPoints:
        def __call__(self, **kwargs):
            return SimpleNamespace(key_points=["point"])

    class Translator:
        def __call__(self, **kwargs):
            return SimpleNamespace(translated_text="translated")

    monkeypatch.setattr(summarizer, "ChunkSummarizer", Chunk)
    monkeypatch.setattr(summarizer, "AggregateSummarizer", Aggregate)
    monkeypatch.setattr(summarizer, "KeyPointsGenerator", KeyPoints)
    monkeypatch.setattr(translation, "Translator", Translator)
    output = summarizer.generate_summary_outputs(
        transcript(), 100, False, True, 0, "medium"
    )
    assert output == SummaryOutput(summary=None, key_points=["point"])
    assert translation.translate_text("source", "Italian", 0) == "translated"
    assert translation.translate_key_points(["one"], "Italian", 0) == ["translated"]
    assert translation.translate_text(" ", "Italian", 0) == " "
    with pytest.raises(JobCancelledError):
        translation.translate_text("source", "Italian", 0, cancel_check=lambda: True)


def test_generation_errors_when_models_never_produce_valid_output(monkeypatch) -> None:
    class Empty:
        def __call__(self, **kwargs):
            return SimpleNamespace(
                summary="", final_summary="", key_points=[], translated_text=""
            )

    monkeypatch.setattr(summarizer, "ChunkSummarizer", Empty)
    monkeypatch.setattr(summarizer, "AggregateSummarizer", Empty)
    monkeypatch.setattr(summarizer, "KeyPointsGenerator", Empty)
    monkeypatch.setattr(translation, "Translator", Empty)
    with pytest.raises(ProcessingError, match="failed validation"):
        summarizer._summarize_chunks(transcript(), 100, 0, "English")
    with pytest.raises(ProcessingError, match="failed validation"):
        summarizer._aggregate_chunk_results(
            [
                SimpleNamespace(
                    index=0,
                    start=0,
                    end=1,
                    word_count=10,
                    summary="one two three four five six seven eight nine ten",
                )
            ],
            0,
            "English",
            "short",
        )
    with pytest.raises(ProcessingError, match="Key point"):
        summarizer._generate_key_points(
            [SimpleNamespace(index=0, start=0, end=1, word_count=1, summary="source")],
            "summary",
            0,
            "English",
            "short",
        )
    with pytest.raises(ProcessingError, match="Translation failed"):
        translation.translate_text("source", "Italian", 0)


def test_direct_generation_validates_and_retries(monkeypatch) -> None:
    responses = iter(
        [
            "## Summary\n\ntoo short\n\n## Key Points\n\n- Point",
            "## Summary\n\n"
            + VALID_SUMMARY
            + "\n\n## Key Points\n\n- First\n- Second\n- Third",
        ]
    )
    monkeypatch.setattr(
        summarizer, "_direct_model_output", lambda *_args: next(responses)
    )
    events = []

    output = summarizer.generate_direct_summary_outputs(
        transcript=transcript(),
        prompt="prompt",
        include_summary=True,
        include_key_points=True,
        max_retries=1,
        summary_length="medium",
        progress=events.append,
    )

    assert output == SummaryOutput(VALID_SUMMARY, ["First", "Second", "Third"])
    assert events[-1].metrics == {"direct_attempts": 2}


def test_direct_generation_handles_requested_sections_and_exhaustion(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        summarizer,
        "_direct_model_output",
        lambda *_args: "## Key Points\n\n- First\n- Second\n- Third",
    )
    key_points = summarizer.generate_direct_summary_outputs(
        transcript=transcript(),
        prompt="prompt",
        include_summary=False,
        include_key_points=True,
        max_retries=0,
        summary_length="medium",
    )
    assert key_points == SummaryOutput(None, ["First", "Second", "Third"])

    with pytest.raises(ProcessingError, match="Direct output failed validation"):
        summarizer.generate_direct_summary_outputs(
            transcript=transcript(),
            prompt="prompt",
            include_summary=True,
            include_key_points=False,
            max_retries=0,
            summary_length="short",
        )


def test_format_direct_transcript_uses_timestamps() -> None:
    source = Transcript(
        [TranscriptSegment("First", 4, 1), TranscriptSegment("Second", 65, 1)],
        Lang("en"),
    )
    assert (
        summarizer.format_direct_transcript(source) == "[00:04] First\n[01:05] Second"
    )
