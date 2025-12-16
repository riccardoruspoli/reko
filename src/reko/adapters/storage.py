import logging
import os

from ..core.errors import OutputError

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
