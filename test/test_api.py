from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from reko import api
from reko.api import _build_summary_config, create_app


def test_healthcheck_is_available_without_external_services() -> None:
    app = create_app()
    route = next(route for route in app.routes if route.path == "/health")

    assert route.endpoint() == {"ok": True}


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("provider", "", "Missing provider"),
        ("modelName", "", "Missing model name"),
        ("targetLanguage", "invalid", "Invalid language code"),
        ("length", "invalid", "length must be one"),
    ],
)
def test_web_config_rejects_invalid_values(
    field: str, value: object, message: str
) -> None:
    data = payload()["config"]
    assert isinstance(data, dict)
    data[field] = value
    with pytest.raises(ValueError, match=message):
        _build_summary_config(data)


def test_synchronous_api_keeps_compatibility_without_external_calls(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        api,
        "summarize_one_with_stats",
        lambda *_args: ("# Result", 4, 2, 0.1, "video"),
    )
    app = create_app()
    try:
        with TestClient(app) as client:
            response = client.post("/api/summarize", json=payload())
            assert response.status_code == 200
            assert response.json()["markdown"] == "# Result"
            assert client.post("/api/summarize", json={}).status_code == 400
            assert client.post("/api/summarize", content="invalid").status_code == 400
    finally:
        app.state.job_manager.shutdown()
