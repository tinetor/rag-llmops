# syntax=docker/dockerfile:1

# ---------- Build stage: install dependencies into a virtualenv ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Dependencies first, with a placeholder package, so this layer is reused
# when only the application code changes.
COPY pyproject.toml ./
RUN mkdir -p src/rag_llmops \
    && touch src/rag_llmops/__init__.py \
    && pip install . \
    && pip uninstall -y rag-llmops

# Then the application code (base dependencies only: no torch).
COPY src ./src
RUN pip install --no-deps .

# ---------- Runtime stage: no build files, non-root user ----------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --uid 1000 app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY configs ./configs
COPY run_ingestion.py run_rag.py ./

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"

CMD ["uvicorn", "rag_llmops.api:app", "--host", "0.0.0.0", "--port", "8080"]
