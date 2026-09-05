from __future__ import annotations

from pathlib import Path

import markdown as md
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from iso639 import Lang

from reko.core.errors import RekoError
from reko.core.models import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TARGET_CHUNK_WORDS,
    SummaryConfig,
)
from reko.core.progress import CancelCheck, ProgressReporter
from reko.core.services import summarize_one_with_stats
from reko.web.jobs import JobManager, JobRunner


def _build_summary_config(config: dict) -> SummaryConfig:
    provider = str(config["provider"]).strip().lower()
    model_name = str(config["modelName"]).strip()
    if not provider:
        raise ValueError("Missing provider.")
    if not model_name:
        raise ValueError("Missing model name.")

    provider_prefix = "openai" if provider == "lmstudio" else provider
    model = f"{provider_prefix}/{model_name}"

    host = config.get("host")
    if host is not None:
        host = str(host).strip() or None

    target_language_value = str(config["targetLanguage"]).strip()
    try:
        target_language = Lang(target_language_value)
    except Exception as e:
        raise ValueError(
            f"Invalid language code: {target_language_value!r} (expected an ISO 639 code like 'en')."
        ) from e

    length = str(config["length"]).strip().lower()
    if length not in {"short", "medium", "long"}:
        raise ValueError("length must be one of: short, medium, long.")

    include_summary = bool(config["includeSummary"])
    include_key_points = bool(config["includeKeyPoints"])
    if not (include_summary or include_key_points):
        raise ValueError("At least one of summary/key points must be enabled.")

    temperature = float(config["temperature"])
    max_retries = int(config["maxRetries"])

    think = bool(config["think"])
    refresh_transcript = bool(config.get("refreshTranscript", False))
    reasoning_effort = config.get("reasoningEffort")
    if reasoning_effort is not None:
        reasoning_effort = str(reasoning_effort).strip().lower() or None

    return SummaryConfig(
        host=host,
        model=model,
        target_chunk_words=DEFAULT_TARGET_CHUNK_WORDS,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=temperature,
        force=True,
        include_summary=include_summary,
        include_key_points=include_key_points,
        max_retries=max_retries,
        print_output=False,
        save_output=False,
        target_language=target_language,
        length=length,
        think=think,
        reasoning_effort=reasoning_effort,
        refresh_transcript=refresh_transcript,
    )


def _parse_summary_request(payload: object) -> tuple[str, SummaryConfig]:
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")

    url = payload.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Missing 'url'.")

    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Missing or invalid 'config' object.")
    return url.strip(), _build_summary_config(config_payload)


def _result_payload(
    markdown_text: str,
    input_words: int,
    output_words: int,
    elapsed_seconds: float,
    video_id: str,
) -> dict:
    html = md.markdown(
        markdown_text,
        extensions=["fenced_code", "tables"],
        output_format="html5",
    )
    return {
        "ok": True,
        "video_id": video_id,
        "markdown": markdown_text,
        "html": html,
        "stats": {
            "input_words": input_words,
            "output_words": output_words,
            "elapsed_seconds": elapsed_seconds,
        },
    }


def _default_job_runner(
    url: str,
    config: SummaryConfig,
    progress: ProgressReporter,
    cancel_check: CancelCheck,
) -> dict:
    return _result_payload(
        *summarize_one_with_stats(
            url,
            config,
            progress=progress,
            cancel_check=cancel_check,
        )
    )


def _request_error(error: Exception) -> JSONResponse:
    if isinstance(error, KeyError):
        message = f"Missing config field: {error.args[0]}"
    else:
        message = str(error)
    return JSONResponse(status_code=400, content={"ok": False, "error": message})


async def _read_summary_request(request: Request) -> tuple[str, SummaryConfig]:
    try:
        payload = await request.json()
    except Exception as error:
        raise ValueError("Invalid JSON body.") from error
    return _parse_summary_request(payload)


def create_app(job_runner: JobRunner | None = None) -> FastAPI:
    web_dir = Path(__file__).resolve().parent / "web"
    templates = Jinja2Templates(directory=str(web_dir / "templates"))

    app = FastAPI(title="reko", version="0.2.0")
    app.state.job_manager = JobManager(job_runner or _default_job_runner)

    app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")

    @app.get("/health")
    def healthcheck() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {},
        )

    @app.post("/api/summarize")
    async def api_summarize(request: Request):
        try:
            url, cfg = await _read_summary_request(request)
            return _result_payload(*summarize_one_with_stats(url, cfg))
        except (KeyError, TypeError, ValueError, RekoError) as error:
            return _request_error(error)

    @app.post("/api/jobs", status_code=202)
    async def create_job(request: Request):
        try:
            url, cfg = await _read_summary_request(request)
            return JSONResponse(
                status_code=202,
                content={"ok": True, "job": app.state.job_manager.submit(url, cfg)},
            )
        except (KeyError, TypeError, ValueError, RekoError) as error:
            return _request_error(error)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return {"ok": True, "job": app.state.job_manager.snapshot(job_id)}
        except KeyError as error:
            return JSONResponse(
                status_code=404, content={"ok": False, "error": str(error)}
            )

    @app.get("/api/jobs/{job_id}/events")
    def get_job_events(job_id: str, request: Request):
        try:
            after_sequence = int(request.headers.get("last-event-id", "0"))
            if after_sequence < 0:
                raise ValueError
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "Invalid Last-Event-ID header."},
            )
        try:
            app.state.job_manager.snapshot(job_id)
            event_stream = app.state.job_manager.iter_events(job_id, after_sequence)
        except KeyError as error:
            return JSONResponse(
                status_code=404, content={"ok": False, "error": str(error)}
            )
        return StreamingResponse(
            event_stream,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: str):
        try:
            return {"ok": True, "job": app.state.job_manager.cancel(job_id)}
        except KeyError as error:
            return JSONResponse(
                status_code=404, content={"ok": False, "error": str(error)}
            )

    return app
