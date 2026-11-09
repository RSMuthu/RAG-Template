# syntax=docker/dockerfile:1
# The RAG service image. EXTRAS picks the optional backends: qdrant, pgvector, docx.
#   docker build -t rag-service .
#   docker build --build-arg EXTRAS="pgvector docx" -t rag-service .

FROM ghcr.io/astral-sh/uv:0.12.3-python3.12-trixie-slim AS build
ARG EXTRAS="qdrant pgvector"
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project $(printf -- '--extra %s ' $EXTRAS)
COPY app ./app
# --reinstall-package: uv caches the project's own wheel by pyproject.toml alone, so without it
# a code change could ship the previous build's code.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --reinstall-package rag-template \
    $(printf -- '--extra %s ' $EXTRAS)

FROM python:3.12-slim-trixie
RUN useradd --create-home --uid 10001 rag && mkdir /data && chown rag /data
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY harness.yaml harness.*.yaml ingest.yaml ./
COPY prompts ./prompts
COPY evals ./evals
COPY data ./data
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 HARNESSWEAVER_ENV=prod
USER rag
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --start-interval=2s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
