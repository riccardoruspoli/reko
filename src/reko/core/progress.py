"""Typed progress events emitted by long-running summarization work."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProgressEvent:
    """A serializable update about a summarization phase."""

    phase: str
    message: str
    completed: int | None = None
    total: int | None = None
    metrics: dict[str, int | float | str | bool] = field(default_factory=dict)


ProgressReporter = Callable[[ProgressEvent], None]
CancelCheck = Callable[[], bool]
