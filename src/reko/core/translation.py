import logging

from reko.adapters.dspy.modules import Translator
from reko.core.errors import ProcessingError
from reko.core.prompt import (
    DEFAULT_TRANSLATION_GUIDANCE,
    KEY_POINTS_TRANSLATION_GUIDANCE,
)
from reko.core.text_utils import normalize_key_points

logger = logging.getLogger(__name__)


def translate_text(
    text: str,
    target_language: str,
    max_retries: int,
    guidance: str | None = None,
) -> str:
    logger.debug("Translating text to %s", target_language)
    if not text.strip():
        return text

    translator = Translator()
    guidance = guidance or DEFAULT_TRANSLATION_GUIDANCE

    for attempt in range(1 + max_retries):
        prediction = translator(
            source_text=text,
            target_language=target_language,
            guidance=guidance,
        )
        translated = getattr(prediction, "translated_text", "").strip()
        if translated:
            return translated
        logger.warning(
            "Translation returned empty output (attempt %d/%d).",
            attempt + 1,
            max_retries + 1,
        )

    raise ProcessingError(
        f"Translation failed to produce output after {max_retries + 1} attempts."
    )


def translate_key_points(
    points: list[str],
    target_language: str,
    max_retries: int,
) -> list[str]:
    if not points:
        return points

    source = "\n".join(f"- {point}" for point in points)
    translated = translate_text(
        source,
        target_language=target_language,
        max_retries=max_retries,
        guidance=KEY_POINTS_TRANSLATION_GUIDANCE,
    )
    return normalize_key_points(translated)
