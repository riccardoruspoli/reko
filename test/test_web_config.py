from __future__ import annotations

from reko.api import _build_summary_config
from reko.core.models import DEFAULT_MAX_TOKENS, DEFAULT_TARGET_CHUNK_WORDS


def test_web_config_uses_internal_generation_defaults() -> None:
    config = _build_summary_config(
        {
            "provider": "ollama",
            "modelName": "llama3",
            "targetLanguage": "en",
            "temperature": 1,
            "maxRetries": 3,
            "think": False,
            "includeSummary": True,
            "includeKeyPoints": True,
            "length": "medium",
        }
    )

    assert config.target_chunk_words == DEFAULT_TARGET_CHUNK_WORDS
    assert DEFAULT_TARGET_CHUNK_WORDS == 1200
    assert config.max_tokens == DEFAULT_MAX_TOKENS
    assert config.refresh_transcript is False


def test_web_config_can_request_a_transcript_refresh() -> None:
    config = _build_summary_config(
        {
            "provider": "ollama",
            "modelName": "llama3",
            "targetLanguage": "en",
            "temperature": 1,
            "maxRetries": 3,
            "think": False,
            "refreshTranscript": True,
            "includeSummary": True,
            "includeKeyPoints": True,
            "length": "medium",
        }
    )

    assert config.refresh_transcript is True
