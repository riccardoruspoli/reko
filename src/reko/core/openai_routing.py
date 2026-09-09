from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import litellm
import tiktoken

DEFAULT_SAFETY_MARGIN_TOKENS = 4_096
FALLBACK_CONTEXT_TOKENS = 128_000


@dataclass(frozen=True)
class DirectRouteDecision:
    enabled: bool
    reason: str
    input_tokens: int | None = None
    direct_input_ceiling_tokens: int | None = None
    hard_input_ceiling_tokens: int | None = None
    price_breakpoint_tokens: int | None = None
    metadata_source: str | None = None


def is_official_openai_model(model: str, host: str | None) -> bool:
    return host is None and model.lower().startswith("openai/")


def select_direct_route(
    *,
    model: str,
    host: str | None,
    prompt: str,
    transcript: str,
    max_completion_tokens: int,
    safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS,
) -> DirectRouteDecision:
    if not is_official_openai_model(model, host):
        return DirectRouteDecision(False, "not_official_openai")
    if max_completion_tokens < 1 or safety_margin_tokens < 0:
        raise ValueError("Token allowances must be positive.")

    model_name = model.split("/", 1)[1]
    capabilities, metadata_source = _capabilities(model_name)
    hard_ceiling = _hard_input_ceiling(model_name, capabilities)
    price_breakpoint = _price_breakpoint(capabilities)
    direct_ceiling = hard_ceiling - max_completion_tokens - safety_margin_tokens
    if price_breakpoint is not None:
        direct_ceiling = min(direct_ceiling, price_breakpoint)
    if direct_ceiling < 1:
        return DirectRouteDecision(
            False,
            "invalid_direct_ceiling",
            direct_input_ceiling_tokens=direct_ceiling,
            hard_input_ceiling_tokens=hard_ceiling,
            price_breakpoint_tokens=price_breakpoint,
            metadata_source=metadata_source,
        )

    input_tokens = _count_tokens(model_name, prompt, transcript)
    if input_tokens is None:
        return DirectRouteDecision(
            False,
            "tokenizer_unavailable",
            direct_input_ceiling_tokens=direct_ceiling,
            hard_input_ceiling_tokens=hard_ceiling,
            price_breakpoint_tokens=price_breakpoint,
            metadata_source=metadata_source,
        )
    if input_tokens > direct_ceiling:
        return DirectRouteDecision(
            False,
            "input_exceeds_direct_ceiling",
            input_tokens=input_tokens,
            direct_input_ceiling_tokens=direct_ceiling,
            hard_input_ceiling_tokens=hard_ceiling,
            price_breakpoint_tokens=price_breakpoint,
            metadata_source=metadata_source,
        )
    return DirectRouteDecision(
        True,
        "eligible",
        input_tokens=input_tokens,
        direct_input_ceiling_tokens=direct_ceiling,
        hard_input_ceiling_tokens=hard_ceiling,
        price_breakpoint_tokens=price_breakpoint,
        metadata_source=metadata_source,
    )


def _capabilities(model_name: str) -> tuple[dict[str, Any], str]:
    try:
        info = litellm.get_model_info(model=model_name)
    except Exception:
        return {}, "family_fallback"
    return dict(info) if isinstance(info, dict) else {}, "litellm"


def _hard_input_ceiling(model_name: str, capabilities: dict[str, Any]) -> int:
    metadata_limit = capabilities.get("max_input_tokens")
    if isinstance(metadata_limit, int) and metadata_limit > 0:
        return metadata_limit
    lowered = model_name.lower()
    if lowered.startswith("gpt-5.6"):
        return 1_050_000
    if lowered.startswith(("gpt-5-nano", "gpt-5-mini")):
        return 400_000
    return FALLBACK_CONTEXT_TOKENS


def _price_breakpoint(capabilities: dict[str, Any]) -> int | None:
    known = (
        "input_price_breakpoint_tokens",
        "input_price_breakpoints_tokens",
        "input_cost_per_token_above_272k_tokens",
    )
    for key in known:
        value = capabilities.get(key)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, list):
            thresholds = [item for item in value if isinstance(item, int) and item > 0]
            if thresholds:
                return min(thresholds)
        if key == "input_cost_per_token_above_272k_tokens" and isinstance(value, float):
            return 272_000
    return None


def _count_tokens(model_name: str, prompt: str, transcript: str) -> int | None:
    try:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(f"{prompt}\n{transcript}"))
    except Exception:
        return None
