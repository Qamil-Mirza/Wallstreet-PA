# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Newsletter audio dependencies plus the native libraries required by
# WeasyPrint's Cairo/Pango rendering path.
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    fonts-dejavu-core \
    git \
    libcairo2 \
    libffi-dev \
    libgdk-pixbuf-2.0-0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libsndfile1 \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash appuser

WORKDIR /app

FROM base AS dependencies

COPY requirements.txt ./
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

FROM dependencies AS production

COPY --chown=appuser:appuser news_bot/ ./news_bot/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser pyproject.toml ./

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

FROM dependencies AS development

COPY --chown=appuser:appuser . .
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
CMD ["python", "-m", "pytest", "-q"]
