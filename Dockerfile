FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/app/data

# Run as an unprivileged user, never root.
RUN useradd --create-home --uid 10001 app
WORKDIR /app

# 1. Dependencies (cached layer: only rebuilt when requirements.txt changes).
COPY requirements.txt .
RUN pip install -r requirements.txt \
 && mkdir -p /app/data && chown -R app:app /app

USER app

# 2. Bake the embedding model (nomic-embed-text-v1.5, ~0.5 GB) into the image at BUILD
#    time, so `docker compose up` never waits on a download. After this step the
#    container runs fully offline for embeddings (HF_HUB_OFFLINE=1).
RUN python -c "from fastembed import TextEmbedding as T; from pathlib import Path; \
list(T('nomic-ai/nomic-embed-text-v1.5', cache_dir=str(Path.home()/'.cache'/'fastembed')).embed(['warm up']))"
ENV HF_HUB_OFFLINE=1

# 3. Application code (changes most often, so copied last).
COPY --chown=app:app . .

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

# Startup ingests policies/ into Chroma, then serves. No key in .env -> offline FakeLLM mode.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
