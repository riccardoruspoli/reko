from __future__ import annotations

from reko.api import create_app


def test_healthcheck_is_available_without_external_services() -> None:
    app = create_app()
    route = next(route for route in app.routes if route.path == "/health")

    assert route.endpoint() == {"ok": True}
