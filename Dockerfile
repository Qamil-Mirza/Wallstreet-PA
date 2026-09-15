# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Native libraries required by WeasyPrint's Cairo/Pango rendering path.
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    libcairo2 \
    libffi-dev \
    libgdk-pixbuf-2.0-0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash appuser

WORKDIR /app

# Research images intentionally install from pyproject.toml. That dependency
# set excludes the legacy TTS/Torch stack and keeps scheduled research small.
FROM base AS research-dependencies

COPY pyproject.toml README.md ./
COPY news_bot/ ./news_bot/
RUN pip install --no-cache-dir .

FROM research-dependencies AS production

COPY --chown=appuser:appuser scripts/ ./scripts/

# Keep both the explicit operational mount points and the paths derived from
# RESEARCH_DATA_DIR available to the fixed unprivileged runtime identity.
RUN mkdir -p \
    /app/data/cache \
    /app/data/source_cache \
    /app/data/reports \
    /app/data/backups \
    /app/reports \
    /app/cache \
    /app/backups \
    /app/logs \
    /app/audio_output \
    && chown -R appuser:appuser /app

USER appuser
CMD ["python", "-m", "news_bot.research.scheduler"]

FROM production AS development

USER root
RUN pip install --no-cache-dir '.[dev]' 'PyYAML>=6,<7' \
    'pydub>=0.25,<1' \
    'trafilatura>=2,<3'
COPY --chown=appuser:appuser tests/ ./tests/
USER appuser
CMD ["python", "-m", "pytest", "-q"]

# The original audio newsletter remains an explicit opt-in image. Research and
# test targets never inherit this stage or resolve its TTS/Torch dependencies.
FROM base AS newsletter-dependencies

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

FROM newsletter-dependencies AS newsletter-production

COPY --chown=appuser:appuser news_bot/ ./news_bot/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser pyproject.toml ./
RUN mkdir -p /app/logs /app/audio_output \
    && chown -R appuser:appuser /app
USER appuser
CMD ["python", "-m", "news_bot.main"]
