import logging

import dspy
from dspy import JSONAdapter

from reko.core.models import SummaryConfig

logger = logging.getLogger(__name__)


def dspy_context(config: SummaryConfig):
    logger.debug(
        "Creating DSPy context with model=%s host=%s max_tokens=%d temperature=%.2f",
        config.model,
        config.host,
        config.max_tokens,
        config.temperature,
    )

    lm = dspy.LM(
        model=config.model,
        model_type="chat",
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_base=config.host,
        cache=False,
        think=config.think if config.model.startswith("ollama/") else None,
    )
    return dspy.context(lm=lm, adapter=JSONAdapter())
