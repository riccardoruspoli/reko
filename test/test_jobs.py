from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest
from iso639 import Lang

from reko.core.models import SummaryConfig
from reko.core.progress import CancelCheck, ProgressEvent, ProgressReporter
from reko.web.jobs import JobManager, JobState


def make_config() -> SummaryConfig:
    return SummaryConfig(
        host=None,
        model="ollama/test",
        target_chunk_words=800,
        max_tokens=1024,
        temperature=0,
        force=True,
        include_summary=True,
        include_key_points=True,
        max_retries=0,
        print_output=False,
        save_output=False,
        target_language=Lang("en"),
        length="medium",
        think=False,
    )


def wait_for(
    manager: JobManager, job_id: str, predicate: Callable[[dict[str, object]], bool]
) -> dict[str, object]:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = manager.snapshot(job_id)
        if predicate(snapshot):
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"Job {job_id} did not reach the expected state")


def test_job_records_progress_result_and_sse_events() -> None:
    def runner(
        url: str,
        config: SummaryConfig,
        report: ProgressReporter,
        is_cancelled: CancelCheck,
    ) -> dict[str, object]:
        assert url == "https://example.test/video"
        assert config.model == "ollama/test"
        assert not is_cancelled()
        report(
            ProgressEvent(
                phase="summarizing",
                message="Summarizing chunks",
                completed=1,
                total=2,
                metrics={"chunks": 2},
            )
        )
        return {"markdown": "# Done"}

    manager = JobManager(runner, max_concurrent_jobs=1)
    try:
        job = manager.submit("https://example.test/video", make_config())
        completed = wait_for(
            manager, job["job_id"], lambda snapshot: snapshot["state"] == "succeeded"
        )

        assert completed["phase"] == "completed"
        assert completed["metrics"] == {"chunks": 2}
        assert completed["result"] == {"markdown": "# Done"}
        events = "".join(manager.iter_events(job["job_id"]))
        assert "event: state" in events
        assert "event: progress" in events
        assert "event: terminal" in events
        assert '"state":"succeeded"' in events
    finally:
        manager.shutdown()


def test_job_cancellation_stops_queued_work() -> None:
    started = threading.Event()
    release = threading.Event()

    def runner(
        url: str,
        config: SummaryConfig,
        report: ProgressReporter,
        is_cancelled: CancelCheck,
    ) -> dict[str, object]:
        started.set()
        release.wait(timeout=2)
        if is_cancelled():
            report(ProgressEvent(phase="unexpected", message="Unexpected work"))
        return {"url": url}

    manager = JobManager(runner, max_concurrent_jobs=1)
    try:
        first = manager.submit("https://example.test/first", make_config())
        assert started.wait(timeout=2)
        second = manager.submit("https://example.test/second", make_config())

        cancelled = manager.cancel(second["job_id"])
        assert cancelled["state"] == JobState.CANCELLED.value
        release.set()
        wait_for(
            manager, first["job_id"], lambda snapshot: snapshot["state"] == "succeeded"
        )
        assert manager.snapshot(second["job_id"])["state"] == "cancelled"
    finally:
        release.set()
        manager.shutdown()


def test_job_failure_and_unknown_job_are_reported() -> None:
    def runner(
        url: str,
        config: SummaryConfig,
        report: ProgressReporter,
        is_cancelled: CancelCheck,
    ) -> dict[str, object]:
        raise RuntimeError("simulated failure")

    manager = JobManager(runner, max_concurrent_jobs=1)
    try:
        job = manager.submit("https://example.test/video", make_config())
        failed = wait_for(
            manager, job["job_id"], lambda snapshot: snapshot["state"] == "failed"
        )

        assert failed["error"] == "simulated failure"
        with pytest.raises(KeyError, match="Unknown job"):
            manager.snapshot("missing")
    finally:
        manager.shutdown()
