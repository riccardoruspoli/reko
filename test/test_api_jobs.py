from __future__ import annotations

import time

from fastapi.testclient import TestClient

from reko.api import create_app
from reko.core.models import SummaryConfig
from reko.core.progress import CancelCheck, ProgressEvent, ProgressReporter


def payload() -> dict[str, object]:
    return {
        "url": "https://example.test/video",
        "config": {
            "provider": "ollama",
            "modelName": "test",
            "targetLanguage": "en",
            "temperature": 0,
            "maxRetries": 0,
            "think": False,
            "includeSummary": True,
            "includeKeyPoints": False,
            "length": "medium",
        },
    }


def wait_for_completed(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(50):
        response = client.get(f"/api/jobs/{job_id}")
        job = response.json()["job"]
        if job["state"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError("Job did not finish")


def test_job_api_submits_reports_and_streams_events() -> None:
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
            ProgressEvent(phase="summarizing", message="Working", completed=1, total=1)
        )
        return {"markdown": "# Result", "html": "<h1>Result</h1>"}

    app = create_app(runner)
    try:
        with TestClient(app) as client:
            response = client.post("/api/jobs", json=payload())
            assert response.status_code == 202
            job_id = response.json()["job"]["job_id"]

            completed = wait_for_completed(client, job_id)
            assert completed["state"] == "succeeded"
            assert completed["result"]["markdown"] == "# Result"

            events = client.get(f"/api/jobs/{job_id}/events")
            assert events.status_code == 200
            assert events.headers["content-type"].startswith("text/event-stream")
            assert "event: progress" in events.text
            assert "event: terminal" in events.text
    finally:
        app.state.job_manager.shutdown()


def test_job_api_validates_requests_and_missing_jobs() -> None:
    app = create_app(lambda *_: {})
    try:
        with TestClient(app) as client:
            assert client.post("/api/jobs", content="bad").json() == {
                "ok": False,
                "error": "Invalid JSON body.",
            }
            assert client.post("/api/jobs", json={}).json() == {
                "ok": False,
                "error": "Missing 'url'.",
            }
            assert client.get("/api/jobs/missing").status_code == 404
            assert client.delete("/api/jobs/missing").status_code == 404
            assert client.get("/api/jobs/missing/events").status_code == 404
    finally:
        app.state.job_manager.shutdown()


def test_job_events_reject_invalid_last_event_id() -> None:
    app = create_app(lambda *_: {})
    try:
        with TestClient(app) as client:
            response = client.post("/api/jobs", json=payload())
            job_id = response.json()["job"]["job_id"]
            invalid = client.get(
                f"/api/jobs/{job_id}/events", headers={"Last-Event-ID": "invalid"}
            )
            assert invalid.status_code == 400
    finally:
        app.state.job_manager.shutdown()


def test_job_events_resume_after_last_event_id() -> None:
    def runner(
        url: str,
        config: SummaryConfig,
        report: ProgressReporter,
        is_cancelled: CancelCheck,
    ) -> dict[str, object]:
        report(
            ProgressEvent(phase="summarizing", message="Working", completed=1, total=1)
        )
        return {"markdown": "# Result", "html": "<h1>Result</h1>"}

    app = create_app(runner)
    try:
        with TestClient(app) as client:
            job_id = client.post("/api/jobs", json=payload()).json()["job"]["job_id"]
            wait_for_completed(client, job_id)

            complete_stream = client.get(f"/api/jobs/{job_id}/events").text
            event_ids = [
                int(line.removeprefix("id: "))
                for line in complete_stream.splitlines()
                if line.startswith("id: ")
            ]
            assert len(event_ids) >= 3

            resumed_stream = client.get(
                f"/api/jobs/{job_id}/events",
                headers={"Last-Event-ID": str(event_ids[-2])},
            ).text
            resumed_ids = [
                int(line.removeprefix("id: "))
                for line in resumed_stream.splitlines()
                if line.startswith("id: ")
            ]
            assert resumed_ids == [event_ids[-1]]
    finally:
        app.state.job_manager.shutdown()


def test_health_remains_available_while_a_job_runs() -> None:
    import threading

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
        return {"markdown": "# Done", "html": "<h1>Done</h1>"}

    app = create_app(runner)
    try:
        with TestClient(app) as client:
            job = client.post("/api/jobs", json=payload()).json()["job"]
            assert started.wait(timeout=2)
            assert client.get("/health").json() == {"ok": True}
            release.set()
            assert wait_for_completed(client, job["job_id"])["state"] == "succeeded"
    finally:
        release.set()
        app.state.job_manager.shutdown()
