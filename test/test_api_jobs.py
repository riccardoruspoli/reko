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
        report(ProgressEvent(phase="summarizing", message="Working", completed=1, total=1))
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
