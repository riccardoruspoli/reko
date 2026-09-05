from __future__ import annotations

import argparse

import pytest

from reko import cli
from reko.core.errors import InputError


def test_parse_and_run_summarize_command(monkeypatch) -> None:
    received = {}
    monkeypatch.setattr(
        cli,
        "summarize",
        lambda target, config: received.update(target=target, config=config),
    )
    status = cli.main(
        [
            "summarize",
            "https://example.test/video",
            "ollama/test",
            "--summary-only",
            "--print-only",
            "--refresh-transcript",
            "--language",
            "it",
        ]
    )

    assert status == 0
    assert received["target"] == "https://example.test/video"
    assert received["config"].include_summary is True
    assert received["config"].include_key_points is False
    assert received["config"].print_output is True
    assert received["config"].save_output is False
    assert received["config"].refresh_transcript is True


def test_cli_validation_and_error_exit_codes(monkeypatch) -> None:
    with pytest.raises(SystemExit):
        cli._parse_args(["summarize", "url", "model", "--max-retries", "-1"])
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_language("invalid-language")

    args = argparse.Namespace(
        func=lambda _: (_ for _ in ()).throw(InputError("bad")),
        log_level=20,
        prog="reko",
        verbose=False,
    )
    monkeypatch.setattr(cli, "_parse_args", lambda _: args)
    assert cli.main([]) == 2

    args.func = lambda _: (_ for _ in ()).throw(KeyboardInterrupt())
    assert cli.main([]) == 130
    args.func = lambda _: (_ for _ in ()).throw(RuntimeError("bad"))
    assert cli.main([]) == 1


def test_serve_starts_uvicorn_with_parsed_options(monkeypatch) -> None:
    received = {}
    monkeypatch.setattr(cli, "create_app", lambda: "app")
    monkeypatch.setattr(
        cli.uvicorn,
        "run",
        lambda app, **kwargs: received.update(app=app, **kwargs),
    )

    assert cli.main(["serve", "--host", "0.0.0.0", "--port", "9999", "--verbose"]) == 0
    assert received == {"app": "app", "host": "0.0.0.0", "port": 9999, "log_level": 10}
