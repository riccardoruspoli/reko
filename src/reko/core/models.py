from __future__ import annotations

from dataclasses import dataclass

from iso639 import Lang

from reko.core.markdown import (
    _summary_document_from_markdown,
    _summary_document_to_markdown,
)


@dataclass
class SummaryConfig:
    host: str | None
    model: str
    target_chunk_words: int
    max_tokens: int
    temperature: float
    force: bool
    include_summary: bool
    include_key_points: bool
    max_retries: int
    print_output: bool
    save_output: bool
    target_language: Lang
    length: str
    think: bool


@dataclass
class TranscriptChunk:
    index: int
    text: str
    start: float
    end: float
    word_count: int


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Transcript:
    segments: list[TranscriptSegment]
    language: Lang

    @property
    def word_count(self) -> int:
        return sum(segment.word_count for segment in self.segments)


@dataclass(frozen=True)
class SummaryChunk:
    index: int
    start: float
    end: float
    word_count: int
    summary: str


@dataclass
class SummaryOutput:
    """Model output payload."""

    summary: str | None
    key_points: list[str] | None


@dataclass
class SummaryDocument:
    """Document-facing summary."""

    title: str
    summary: str | None = None
    key_points: list[str] | None = None

    def to_markdown(self) -> str:
        return _summary_document_to_markdown(self)

    @classmethod
    def from_markdown(cls, markdown: str) -> SummaryDocument:
        return _summary_document_from_markdown(markdown)
