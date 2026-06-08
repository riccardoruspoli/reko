import logging
import math

import dspy
from dspy import JSONAdapter

from reko.core.models import SummaryConfig

logger = logging.getLogger(__name__)

SUPPORTED_GPT5_REASONING_EFFORTS = {
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
}


def _model_name(model: str) -> str:
    return model.split("/", 1)[1].lower() if "/" in model else model.lower()


def _is_openai_gpt5_model(model: str) -> bool:
    provider, _, name = model.lower().partition("/")
    if provider != "openai":
        return False
    return name.startswith("gpt-5") and not name.startswith("gpt-5-chat")


def _is_legacy_gpt5_model(model: str) -> bool:
    name = _model_name(model)
    return name.startswith("gpt-5") and not name.startswith(("gpt-5.", "gpt-5-chat"))


def _gpt5_lm_kwargs(config: SummaryConfig) -> dict:
    reasoning_effort = config.reasoning_effort or "low"
    if reasoning_effort not in SUPPORTED_GPT5_REASONING_EFFORTS:
        raise ValueError(
            "reasoning_effort must be one of: "
            + ", ".join(sorted(SUPPORTED_GPT5_REASONING_EFFORTS))
        )
    if _is_legacy_gpt5_model(config.model) and reasoning_effort in {"none", "xhigh"}:
        raise ValueError(
            f"{config.model} does not support reasoning_effort={reasoning_effort!r}; "
            "use minimal, low, medium, or high."
        )
    if not math.isclose(config.temperature, 1.0):
        logger.warning(
            "Ignoring temperature %.2f for %s; OpenAI reasoning models only support the default temperature.",
            config.temperature,
            config.model,
        )

    lm_kwargs = {
        "model_type": "responses",
        "temperature": None,
        "reasoning": {"effort": reasoning_effort},
    }
    if _is_legacy_gpt5_model(config.model):
        lm_kwargs["max_tokens"] = config.max_tokens
    else:
        lm_kwargs["max_tokens"] = None
        lm_kwargs["max_completion_tokens"] = config.max_tokens
    return lm_kwargs


def dspy_context(config: SummaryConfig):
    logger.debug(
        "Creating DSPy context with model=%s host=%s max_tokens=%d temperature=%.2f reasoning_effort=%s",
        config.model,
        config.host,
        config.max_tokens,
        config.temperature,
        config.reasoning_effort,
    )

    lm_kwargs = {
        "model_type": "chat",
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if _is_openai_gpt5_model(config.model):
        lm_kwargs = _gpt5_lm_kwargs(config)

    constructor_kwargs = {
        "model": config.model,
        **lm_kwargs,
        "cache": False,
    }
    if config.host:
        constructor_kwargs["api_base"] = config.host
    if config.model.startswith("ollama/"):
        constructor_kwargs["think"] = config.think

    lm = dspy.LM(**constructor_kwargs)
    return dspy.context(lm=lm, adapter=JSONAdapter())
