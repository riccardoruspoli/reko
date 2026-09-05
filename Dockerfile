FROM ghcr.io/astral-sh/uv:0.12.10 AS uv

FROM python:3.13-slim AS builder

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN uv sync --locked --no-dev --no-editable

FROM python:3.13-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    REKO_DATA_DIR=/data

RUN groupadd --gid 10001 reko \
    && useradd --uid 10001 --gid reko --create-home reko \
    && mkdir /data \
    && chown reko:reko /data

COPY --from=builder /app/.venv /app/.venv

USER reko
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health')"

CMD ["reko", "serve", "--host", "0.0.0.0", "--port", "8000"]
