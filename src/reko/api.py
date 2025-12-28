from __future__ import annotations

from pathlib import Path

import markdown as md
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from iso639 import Lang

from reko.core.errors import RekoError
from reko.core.models import SummaryConfig
from reko.core.services import summarize_one_with_stats


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
    target_chunk_words = int(config["targetChunkWords"])
    max_tokens = int(config["maxTokens"])
    max_retries = int(config["maxRetries"])

    think = bool(config["think"])

    return SummaryConfig(
        host=host,
        model=model,
        target_chunk_words=target_chunk_words,
        max_tokens=max_tokens,
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
    )


def create_app() -> FastAPI:
    web_dir = Path(__file__).resolve().parent / "web"
    templates = Jinja2Templates(directory=str(web_dir / "templates"))

    app = FastAPI(title="reko", version="0.1.1")

    app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {"request": request},
        )

    @app.post("/api/summarize")
    async def api_summarize(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "Invalid JSON body."},
            )

        if not isinstance(payload, dict):
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "Expected a JSON object."},
            )

        url = payload.get("url")
        if not isinstance(url, str) or not url.strip():
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "Missing 'url'."},
            )

        config_payload = payload.get("config")
        if not isinstance(config_payload, dict):
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "Missing or invalid 'config' object."},
            )

        try:
            cfg = _build_summary_config(config_payload)
            markdown_text, input_words, output_words, elapsed_seconds, video_id = (
                summarize_one_with_stats(url.strip(), cfg)
            )
        except KeyError as e:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": f"Missing config field: {e.args[0]}"},
            )
        except (TypeError, ValueError, RekoError) as e:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": str(e)},
            )

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

    return app
