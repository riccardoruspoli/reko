from __future__ import annotations

import pytest
from iso639 import Lang

from reko.adapters.storage import is_summary_complete, save_summary
from reko.core.chunking import chunk_transcript
from reko.core.errors import OutputError, ProcessingError
from reko.core.models import (
    SummaryChunk,
    SummaryConfig,
    SummaryDocument,
    Transcript,
    TranscriptChunk,
    TranscriptSegment,
)
from reko.core.prompt import (
    _format_timestamp,
    build_chunk_context,
    build_key_points_guidance,
    build_reduce_context,
    format_mapped_chunks,
)
from reko.core.text_utils import is_valid_tldr, normalize_key_points, normalize_sequence
from reko.core.transcript import resolve_language


def config(*, include_summary: bool = True, include_key_points: bool = True) -> SummaryConfig:
    return SummaryConfig(
        host=None,
        model="ollama/test",
        target_chunk_words=3,
        max_tokens=100,
        temperature=1,
        force=False,
        include_summary=include_summary,
        include_key_points=include_key_points,
        max_retries=0,
        print_output=False,
        save_output=False,
        target_language=Lang("en"),
        length="medium",
        think=False,
    )


def test_text_normalizers_cover_sequences_validation_and_compact_bullets() -> None:
    assert normalize_sequence(None) == []
    assert normalize_sequence([" a\n b ", 4]) == ["a b", "4"]
    assert is_valid_tldr("one two", 2)
    assert not is_valid_tldr(" ")
    assert normalize_key_points("- One\\n- Two\r\n3. Three - Four - Five") == [
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
    ]
    assert normalize_key_points(["* First", "plain text"]) == ["First", "plain text"]


def test_chunking_preserves_segments_and_rejects_empty_transcripts() -> None:
    transcript = Transcript(
        language=Lang("en"),
        segments=[
            TranscriptSegment("one two", 0, 2),
            TranscriptSegment("three four", 2, 3),
            TranscriptSegment("five", 5, 1),
        ],
    )
    chunks = chunk_transcript(transcript, target_chunk_words=3)

    assert [(chunk.index, chunk.text, chunk.start, chunk.end) for chunk in chunks] == [
        (0, "one two", 0, 2),
        (1, "three four five", 2, 6),
    ]
    assert chunk_transcript(transcript, target_chunk_words=0)[0].word_count == 5
    with pytest.raises(ProcessingError, match="No valid transcript"):
        chunk_transcript(Transcript([], Lang("en")), target_chunk_words=1)


def test_prompt_builders_include_expected_context() -> None:
    chunk = TranscriptChunk(index=0, text="source", start=-2, end=125.9, word_count=1)
    mapped = [SummaryChunk(0, 0, 65, 12, "summary")]

    assert _format_timestamp(-1) == "00:00"
    assert _format_timestamp(65) == "01:05"
    assert "Chunk 1 of 2" in build_chunk_context(chunk, 2, "Italian")
    assert "Write in Italian" in build_chunk_context(chunk, 2, "Italian")
    assert "[Chunk 1] 00:00-01:05 (12 words)" in format_mapped_chunks(mapped)
    assert "at least 10 words" in build_reduce_context(
        chunk_count=1, length_guidance="Be brief.", min_summary_words=10, language=""
    )
    assert "Respond in Italian" in build_key_points_guidance(
        min_bullets=2, max_bullets=4, language="Italian"
    )


def test_markdown_round_trip_and_summary_storage(tmp_path, monkeypatch) -> None:
    document = SummaryDocument(" Title ", "Body", ["First", "Second"])
    markdown = document.to_markdown()
    assert SummaryDocument.from_markdown(markdown) == SummaryDocument(
        "Title", "Body", ["First", "Second"]
    )
    assert SummaryDocument.from_markdown("# Only title") == SummaryDocument("Only title")

    monkeypatch.chdir(tmp_path)
    save_summary("video", markdown)
    assert is_summary_complete("summary/video.md", config())
    assert not is_summary_complete("missing.md", config())
    assert is_summary_complete("summary/video.md", config(include_key_points=False))


def test_storage_wraps_write_errors(monkeypatch) -> None:
    monkeypatch.setattr("reko.adapters.storage.os.makedirs", lambda *args, **kwargs: None)

    def fail_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", fail_open)
    with pytest.raises(OutputError, match="Failed to write"):
        save_summary("video", "text")


def test_resolve_language_rejects_unknown_codes() -> None:
    assert resolve_language("en").pt1 == "en"
    with pytest.raises(ProcessingError, match="Unsupported language"):
        resolve_language("not-a-language")
