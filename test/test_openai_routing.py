from __future__ import annotations

import pytest

from reko.core import openai_routing


def route(**changes: object) -> openai_routing.DirectRouteDecision:
    values: dict[str, object] = {
        "model": "openai/gpt-5-nano",
        "host": None,
        "prompt": "Summarize in Italian.",
        "transcript": "source transcript",
        "max_completion_tokens": 16_384,
        "safety_margin_tokens": 1_000,
    }
    values.update(changes)
    return openai_routing.select_direct_route(**values)  # type: ignore[arg-type]


def test_rejects_non_official_openai_models() -> None:
    assert route(model="ollama/test").reason == "not_official_openai"
    assert route(host="http://localhost:1234").reason == "not_official_openai"


def test_uses_litellm_metadata_and_token_count(monkeypatch) -> None:
    monkeypatch.setattr(
        openai_routing.litellm,
        "get_model_info",
        lambda **_kwargs: {"max_input_tokens": 100_000},
    )
    monkeypatch.setattr(openai_routing, "_count_tokens", lambda *_args: 123)

    decision = route()

    assert decision.enabled is True
    assert decision.metadata_source == "litellm"
    assert decision.input_tokens == 123
    assert decision.hard_input_ceiling_tokens == 100_000
    assert decision.direct_input_ceiling_tokens == 82_616


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/gpt-5.6-luna", 1_050_000),
        ("openai/gpt-5-nano", 400_000),
        ("openai/gpt-5-mini", 400_000),
        ("openai/other", openai_routing.FALLBACK_CONTEXT_TOKENS),
    ],
)
def test_uses_conservative_family_fallbacks(monkeypatch, model, expected) -> None:
    monkeypatch.setattr(
        openai_routing.litellm,
        "get_model_info",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("missing")),
    )
    monkeypatch.setattr(openai_routing, "_count_tokens", lambda *_args: 1)

    decision = route(model=model)

    assert decision.enabled is True
    assert decision.metadata_source == "family_fallback"
    assert decision.hard_input_ceiling_tokens == expected


def test_uses_price_breakpoint_as_direct_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(
        openai_routing.litellm,
        "get_model_info",
        lambda **_kwargs: {
            "max_input_tokens": 1_000_000,
            "input_price_breakpoints_tokens": [250_000, 500_000],
        },
    )
    monkeypatch.setattr(openai_routing, "_count_tokens", lambda *_args: 250_001)

    decision = route()

    assert decision.enabled is False
    assert decision.reason == "input_exceeds_direct_ceiling"
    assert decision.price_breakpoint_tokens == 250_000
    assert decision.direct_input_ceiling_tokens == 250_000


def test_rejects_input_at_the_hard_limit_and_missing_tokenizer(monkeypatch) -> None:
    monkeypatch.setattr(
        openai_routing.litellm,
        "get_model_info",
        lambda **_kwargs: {"max_input_tokens": 20_000},
    )
    monkeypatch.setattr(openai_routing, "_count_tokens", lambda *_args: 2_617)
    assert route().reason == "input_exceeds_direct_ceiling"

    monkeypatch.setattr(openai_routing, "_count_tokens", lambda *_args: None)
    assert route().reason == "tokenizer_unavailable"


def test_rejects_invalid_allowances() -> None:
    with pytest.raises(ValueError, match="Token allowances"):
        route(max_completion_tokens=0)
    with pytest.raises(ValueError, match="Token allowances"):
        route(safety_margin_tokens=-1)
