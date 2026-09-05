"""In-process job execution, state snapshots, and SSE event streaming."""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from reko.core.models import SummaryConfig
from reko.core.progress import CancelCheck, ProgressEvent, ProgressReporter


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


@dataclass(frozen=True)
class JobEvent:
    sequence: int
    name: str
    payload: dict[str, Any]


@dataclass
class SummaryJob:
    job_id: str
    submitted_at: datetime
    state: JobState = JobState.QUEUED
    phase: str = "queued"
    message: str = "Waiting to start"
    completed: int | None = None
    total: int | None = None
    metrics: dict[str, int | float | str | bool] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = False
    finished_at: datetime | None = None
    events: list[JobEvent] = field(default_factory=list)
    condition: threading.Condition = field(
        default_factory=threading.Condition, repr=False
    )
    future: Future[None] | None = field(default=None, repr=False)


JobRunner = Callable[
    [str, SummaryConfig, ProgressReporter, CancelCheck], dict[str, Any]
]


def _max_concurrent_jobs() -> int:
    value = os.environ.get("REKO_MAX_CONCURRENT_JOBS", "1")
    try:
        return max(1, int(value))
    except ValueError:
        return 1


class JobManager:
    """Run a bounded number of jobs and expose reconnect-safe progress events."""

    def __init__(
        self,
        runner: JobRunner,
        *,
        max_concurrent_jobs: int | None = None,
        retention: timedelta = timedelta(hours=1),
    ) -> None:
        self._runner = runner
        self._retention = retention
        self._jobs: dict[str, SummaryJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_jobs or _max_concurrent_jobs(),
            thread_name_prefix="reko-job",
        )

    def submit(self, url: str, config: SummaryConfig) -> dict[str, Any]:
        self._prune()
        job = SummaryJob(job_id=str(uuid.uuid4()), submitted_at=datetime.now(UTC))
        with self._lock:
            self._jobs[job.job_id] = job
        self._emit(job, "state")
        job.future = self._executor.submit(self._run, job, url, config)
        return self.snapshot(job.job_id)

    def snapshot(self, job_id: str) -> dict[str, Any]:
        job = self._get(job_id)
        with job.condition:
            return self._snapshot(job)

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self._get(job_id)
        with job.condition:
            if job.state.terminal:
                return self._snapshot(job)
            job.cancel_requested = True
            if job.future and job.future.cancel():
                job.state = JobState.CANCELLED
                job.phase = "cancelled"
                job.message = "Cancelled before execution"
                job.finished_at = datetime.now(UTC)
            else:
                job.state = JobState.CANCELLING
                job.message = "Cancelling after the current step"
            self._append_event(job, "state")
            return self._snapshot(job)

    def iter_events(
        self, job_id: str, after_sequence: int = 0
    ) -> Generator[str, None, None]:
        job = self._get(job_id)
        sequence = max(0, after_sequence)
        while True:
            keep_alive = False
            with job.condition:
                events = [event for event in job.events if event.sequence > sequence]
                terminal = job.state.terminal
                if not events and not terminal:
                    job.condition.wait(timeout=15)
                    events = [
                        event for event in job.events if event.sequence > sequence
                    ]
                    terminal = job.state.terminal
                if not events and not terminal:
                    keep_alive = True

            if keep_alive:
                yield ": keep-alive\n\n"
                continue
            for event in events:
                sequence = event.sequence
                payload = json.dumps(event.payload, separators=(",", ":"))
                yield f"id: {event.sequence}\nevent: {event.name}\ndata: {payload}\n\n"
            if terminal:
                return

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job: SummaryJob, url: str, config: SummaryConfig) -> None:
        with job.condition:
            if job.cancel_requested:
                job.state = JobState.CANCELLED
                job.phase = "cancelled"
                job.message = "Cancelled before execution"
                job.finished_at = datetime.now(UTC)
                self._append_event(job, "state")
                return
            job.state = JobState.RUNNING
            job.phase = "starting"
            job.message = "Starting job"
            self._append_event(job, "state")

        def report(event: ProgressEvent) -> None:
            with job.condition:
                if job.cancel_requested:
                    raise JobCancelledError()
                job.phase = event.phase
                job.message = event.message
                job.completed = event.completed
                job.total = event.total
                job.metrics.update(event.metrics)
                self._append_event(job, "progress")

        def is_cancelled() -> bool:
            with job.condition:
                return job.cancel_requested

        try:
            result = self._runner(url, config, report, is_cancelled)
        except JobCancelledError:
            self._finish_cancelled(job)
        except Exception as error:
            with job.condition:
                job.state = JobState.FAILED
                job.phase = "failed"
                job.message = "Job failed"
                job.error = str(error)
                job.finished_at = datetime.now(UTC)
                self._append_event(job, "terminal")
        else:
            with job.condition:
                if job.cancel_requested:
                    self._finish_cancelled(job)
                    return
                job.state = JobState.SUCCEEDED
                job.phase = "completed"
                job.message = "Job completed"
                job.result = result
                job.finished_at = datetime.now(UTC)
                self._append_event(job, "terminal")

    def _finish_cancelled(self, job: SummaryJob) -> None:
        with job.condition:
            job.state = JobState.CANCELLED
            job.phase = "cancelled"
            job.message = "Job cancelled"
            job.finished_at = datetime.now(UTC)
            self._append_event(job, "terminal")

    def _emit(self, job: SummaryJob, name: str) -> None:
        with job.condition:
            self._append_event(job, name)

    def _append_event(self, job: SummaryJob, name: str) -> None:
        sequence = len(job.events) + 1
        job.events.append(
            JobEvent(sequence=sequence, name=name, payload=self._snapshot(job))
        )
        job.condition.notify_all()

    @staticmethod
    def _snapshot(job: SummaryJob) -> dict[str, Any]:
        elapsed_end = job.finished_at or datetime.now(UTC)
        return {
            "job_id": job.job_id,
            "state": job.state.value,
            "phase": job.phase,
            "message": job.message,
            "completed": job.completed,
            "total": job.total,
            "metrics": dict(job.metrics),
            "result": job.result,
            "error": job.error,
            "cancel_requested": job.cancel_requested,
            "submitted_at": job.submitted_at.isoformat(),
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            "elapsed_seconds": round(
                (elapsed_end - job.submitted_at).total_seconds(), 3
            ),
            "event_sequence": len(job.events),
        }

    def _get(self, job_id: str) -> SummaryJob:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as error:
                raise KeyError(f"Unknown job: {job_id}") from error

    def _prune(self) -> None:
        cutoff = datetime.now(UTC) - self._retention
        with self._lock:
            expired = [
                job_id
                for job_id, job in self._jobs.items()
                if job.finished_at and job.finished_at < cutoff
            ]
            for job_id in expired:
                del self._jobs[job_id]


class JobCancelledError(Exception):
    """Raised by a cooperative progress callback after a cancellation request."""
