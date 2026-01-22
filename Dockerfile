# =============================================================================
# Newsletter Bot with Coqui TTS
# =============================================================================
# Multi-stage build for optimal image size
# Requires Python 3.11 for Coqui TTS compatibility

FROM python:3.11-slim as base

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# =============================================================================
# Dependencies stage
# =============================================================================
FROM base as dependencies

# Install Python dependencies
COPY requirements.txt .

# Install PyTorch CPU-only for smaller image (remove --index-url for GPU)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Production stage
# =============================================================================
FROM dependencies as production

# Copy application code
COPY news_bot/ ./news_bot/
COPY pyproject.toml .

# Create directories for outputs and logs
RUN mkdir -p /app/audio_output /app/logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Default command
CMD ["python", "-m", "news_bot.main"]

# =============================================================================
# Development stage (includes tests)
# =============================================================================
FROM dependencies as development

# Copy everything including tests
COPY . .

# Create directories
RUN mkdir -p /app/audio_output /app/logs

# Default command for dev runs tests
CMD ["python", "-m", "pytest", "tests/", "-v"]
