from iso639 import Lang

from reko.core.errors import ProcessingError


def resolve_language(language_code: str) -> Lang:
    """Resolve a language code into an `iso639.Lang` instance."""

    try:
        return Lang(language_code)
    except Exception:
        pass

    raise ProcessingError(f"Unsupported language code: {language_code!r}")
