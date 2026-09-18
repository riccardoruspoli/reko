import logging
import os

from reko.core.errors import OutputError
from reko.core.models import SummaryConfig, SummaryDocument

logger = logging.getLogger(__name__)


def save_summary(id: str, summary: str) -> None:
    try:
        os.makedirs("summary", exist_ok=True)
        with open(f"summary/{id}.md", "w", encoding="utf-8") as f:
            f.write(summary)
    except OSError as e:
        raise OutputError(f"Failed to write summary file for video {id}.") from e
    else:
        logger.info("Saved summary as %s.md", id)


def is_summary_complete(summary_path: str, config: SummaryConfig) -> bool:
    """Check if the summary file contains the requested sections. If the file doesn't exist, or is missing the requested sections, return False."""

    if not os.path.exists(summary_path):
        return False

    with open(summary_path, encoding="utf-8") as f:
        existing = f.read()

    document = SummaryDocument.from_markdown(existing)

    has_summary = bool(document.summary and document.summary.strip())
    has_key_points = bool(document.key_points)
    has_brief = document.brief is not None
    return (
        (not config.include_summary or has_summary)
        and (not config.include_key_points or has_key_points)
        and (not config.include_brief or has_brief)
    )
