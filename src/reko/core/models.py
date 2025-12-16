from dataclasses import dataclass

from iso639 import Lang


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
