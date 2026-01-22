# 📊 The Daily Brief By The Journey

A fully automated daily finance/economics email bot that runs locally and:

- **Pulls** recent financial news from MarketAux (multi-feed, sectioned)
- **Extracts** full article text using `trafilatura` (with BeautifulSoup fallback)
- **Summarizes** each story using a local Ollama LLM in an analyst-style voice (3-sentence brief)
- **Generates** an audio podcast using Coqui TTS with a "Wall Street radio host" persona
- **Sends** a beautifully formatted HTML email every morning, organized by enabled sections

All components run locally on modest hardware using only free services.

## Features

- 🆓 **Free tier news**: MarketAux (recommended)
- 🤖 **Local LLM**: Ollama (llama3, llama3.1, etc.)
- 🎙️ **Text-to-Speech**: Coqui TTS generates audio podcasts from summaries (non-commercial use only)
- 📧 **SMTP email**: Works with Gmail, Outlook, or any SMTP provider
- 🧩 **Section feature flags**: Toggle which sections appear in the final email
- 🧾 **LLM trace logs**: Full prompt + raw model output saved to `logs/`
- 🧱 **Block-page detection**: Detects paywalls/adblock/captcha pages and skips LLM calls
- 🧪 **Fully tested**: pytest test suite with mocks (200+ tests)
- 📦 **Modular design**: Each component in its own module

## Quick Start

### 1. Clone and Install

> ⚠️ **Python 3.11 Required**: Coqui TTS requires Python 3.9-3.11. Python 3.12+ is not supported.

```bash
cd newsletter

# Create venv with Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Install System Dependencies

For TTS audio generation, you need `ffmpeg` installed:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg
```

### 3. Configure Environment

Copy the example config and fill in your values:

```bash
cp env.example .env
```

Edit `.env` with your credentials:

```bash
# Get a free API key from https://www.marketaux.com/
NEWS_API_KEY=your_marketaux_api_key

# MarketAux base URL (default)
NEWS_API_BASE_URL=https://api.marketaux.com/v1

# Gmail with App Password (recommended)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
RECIPIENT_EMAIL=recipient@example.com

# Ollama settings (make sure Ollama is running)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# TTS Settings (Coqui TTS)
TTS_ENABLED=true
TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC
TTS_OUTPUT_DIR=audio_output
TTS_USE_CUDA=false
TTS_DURATION_MINUTES=2.0

# Section Feature Flags (true/false)
SECTION_WORLD_ENABLED=true
SECTION_US_TECH_ENABLED=true
SECTION_US_INDUSTRY_ENABLED=true
SECTION_MALAYSIA_TECH_ENABLED=true
SECTION_MALAYSIA_INDUSTRY_ENABLED=true
```

### 4. Start Ollama

Make sure Ollama is running with your chosen model:

```bash
ollama pull llama3
ollama serve
```

### 5. Run the Bot

```bash
python -m news_bot.main
```

## Deployment

### Option 1: Docker (Recommended)

The easiest way to deploy is with Docker, which handles all dependencies automatically.

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/newsletter.git
cd newsletter

# 2. Create your configuration
cp env.example .env
# Edit .env with your API keys and settings

# 3. Start services (Ollama + Bot)
docker compose up -d ollama

# 4. Pull the LLM model (first time only)
docker compose exec ollama ollama pull llama3

# 5. Run the newsletter
docker compose run bot
```

#### Scheduling with Docker

For daily runs, add to crontab:

```bash
0 7 * * * cd /path/to/newsletter && docker compose run --rm bot >> logs/cron.log 2>&1
```

#### GPU Support

For NVIDIA GPU acceleration, uncomment the GPU section in `docker-compose.yml`.

### Option 2: Setup Script (Server Deployment)

For non-Docker deployments, use the automated setup script:

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/newsletter.git
cd newsletter

# 2. Run setup (installs dependencies, creates venv)
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Configure
# Edit .env with your settings (created by setup script)

# 4. Start Ollama
ollama serve &

# 5. Run
source venv/bin/activate
python -m news_bot.main
```

#### Scheduling (Server)

Use the run script with cron:

```bash
crontab -e
```

Add:
```
0 7 * * * /path/to/newsletter/scripts/run.sh >> /path/to/newsletter/logs/cron.log 2>&1
```

### Server Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.11 |
| RAM | 4GB | 8GB+ |
| Disk | 5GB | 10GB+ |
| OS | Ubuntu 20.04+ | Ubuntu 22.04+ |

#### System Dependencies

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y ffmpeg espeak-ng python3.11 python3.11-venv

# macOS
brew install ffmpeg espeak python@3.11

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3
```

### Known Compatibility Issues

| Issue | Solution |
|-------|----------|
| `BeamSearchScorer` import error | Pin `transformers==4.40.0` (included in requirements.txt) |
| PyTorch `weights_only` error | Use `torch<2.6` (included in setup) |
| TTS not installing | Use Python 3.9-3.11 (not 3.12+) |

## TTS Audio Generation

The bot generates an audio podcast from article summaries using a two-stage process:

1. **Script Generation**: Summaries are passed to Ollama with a "Wall Street radio host" persona prompt, creating an engaging broadcast script
2. **Speech Synthesis**: The script is converted to audio using Coqui TTS

### Audio Output

- Audio files are saved to `audio_output/` (configurable via `TTS_OUTPUT_DIR`)
- Filename format: `broadcast_YYYYMMDD.mp3`
- Default duration target: ~2 minutes (configurable via `TTS_DURATION_MINUTES`)

### Available TTS Models

The default model (`tts_models/en/ljspeech/tacotron2-DDC`) provides good quality with reasonable speed. Other options:

```bash
# List all available models
tts --list_models

# High-quality alternatives:
# - tts_models/en/ljspeech/glow-tts
# - tts_models/en/ljspeech/tacotron2-DCA
# - tts_models/en/vctk/vits (multi-speaker)
```

### Adjusting Speech Speed

Control the speech speed in your `.env`:

```bash
TTS_SPEED=1.0   # Normal speed
TTS_SPEED=1.2   # 20% faster
TTS_SPEED=1.5   # 50% faster
TTS_SPEED=0.8   # 20% slower
```

### Disabling TTS

To disable audio generation, set in `.env`:

```bash
TTS_ENABLED=false
```

## Scheduling

### macOS/Linux (cron)

Run daily at 7:00 AM:

```bash
crontab -e
```

Add:

```
0 7 * * * /path/to/venv/bin/python -m news_bot.main >> /path/to/newsletter.log 2>&1
```

### Windows (Task Scheduler)

1. Open Task Scheduler
2. Create Basic Task → "Daily Newsletter"
3. Trigger: Daily at your preferred time
4. Action: Start a Program
   - Program: `C:\path\to\venv\Scripts\python.exe`
   - Arguments: `-m news_bot.main`
   - Start in: `C:\path\to\newsletter`

## Project Structure

```
newsletter/
├── news_bot/
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── news_client.py        # News API client
│   ├── article_extractor.py  # Content extraction (trafilatura + fallback + block detection)
│   ├── classifier.py         # Keyword-based categorization helpers
│   ├── selection.py          # Category labels (legacy)
│   ├── summarizer.py         # Ollama LLM integration (smart chunking + trace logs)
│   ├── script_generator.py   # Radio host script generation from summaries
│   ├── tts_engine.py         # Coqui TTS audio synthesis
│   ├── email_client.py       # HTML email rendering & SMTP
│   └── main.py               # Orchestration & entry point
├── scripts/
│   ├── setup.sh              # Automated setup for server deployment
│   └── run.sh                # Run script for cron jobs
├── tests/
│   ├── test_*.py             # Comprehensive test suite (230+ tests)
├── audio_output/             # Generated audio files (git-ignored)
├── logs/                     # LLM trace logs (git-ignored)
├── Dockerfile                # Container build configuration
├── docker-compose.yml        # Multi-service orchestration
├── requirements.txt          # Pinned Python dependencies
├── env.example
└── README.md
```

## Running Tests

```bash
pytest -v
```

With coverage:

```bash
pip install pytest-cov
pytest --cov=news_bot --cov-report=term-missing
```

## Configuration Options

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEWS_API_KEY` | Yes | - | API key for news service |
| `NEWS_API_BASE_URL` | No | `https://api.marketaux.com/v1` | News API base URL |
| `SMTP_HOST` | Yes | - | SMTP server hostname |
| `SMTP_PORT` | No | `587` | SMTP server port |
| `SMTP_USER` | Yes | - | SMTP username (email) |
| `SMTP_PASSWORD` | Yes | - | SMTP password or app password |
| `RECIPIENT_EMAIL` | Yes | - | Email recipient |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | No | `llama3` | Ollama model name |
| `TTS_ENABLED` | No | `true` | Enable/disable audio generation |
| `TTS_MODEL` | No | `tts_models/en/ljspeech/tacotron2-DDC` | Coqui TTS model |
| `TTS_LANGUAGE` | No | `en` | Language code for multilingual models |
| `TTS_SPEAKER` | No | `Claribel Dervla` | Speaker name for XTTS models |
| `TTS_SPEED` | No | `1.0` | Speech speed (1.0=normal, 1.2=faster, 0.8=slower) |
| `TTS_OUTPUT_DIR` | No | `audio_output` | Directory for audio files |
| `TTS_USE_CUDA` | No | `false` | Use GPU for TTS (requires CUDA) |
| `TTS_DURATION_MINUTES` | No | `2.0` | Target audio duration in minutes |
| `SECTION_WORLD_ENABLED` | No | `true` | Include **World News** section |
| `SECTION_US_TECH_ENABLED` | No | `true` | Include **US Tech** section |
| `SECTION_US_INDUSTRY_ENABLED` | No | `true` | Include **US Industry** section |
| `SECTION_MALAYSIA_TECH_ENABLED` | No | `true` | Include **Malaysia Tech** section |
| `SECTION_MALAYSIA_INDUSTRY_ENABLED` | No | `true` | Include **Malaysia Industry** section |

## Supported News APIs

### MarketAux (Recommended)
- **Free tier**: available, but results may vary by endpoint/plan
- **Sign up**: `https://www.marketaux.com/`
- This project fetches multiple feeds (World/US/Malaysia + Tech/Industry) and deduplicates URLs.

## LLM Trace Logs (Visibility)

Every LLM summarization call writes a full trace (prompt + raw output) to:

- `logs/{OLLAMA_MODEL}_trace_{YYYYMMDD}_{HHMMSS}.log`

The `logs/` directory is git-ignored by default.

## Blocked/Paywalled Pages

Some sites return block pages (paywalls, adblock warnings, captcha). After extraction, the bot detects common block phrases (e.g., "Please enable Javascript and cookies", "If you have an ad-blocker enabled…").

- Blocked articles are marked as **blocked** in the pipeline
- The LLM is **not called** for blocked content
- The email shows a fallback line: **"Summary unavailable due to site access restrictions."**

## Gmail Setup

1. Enable 2-Factor Authentication on your Google account
2. Go to https://myaccount.google.com/apppasswords
3. Generate an App Password for "Mail"
4. Use the 16-character password as `SMTP_PASSWORD`

## Article Categories

The current email format is **sectioned** by region/industry (World/US/Malaysia + Tech/Industry).  
Keyword-based categories still exist in code for legacy labeling, but the default pipeline focuses on **sectioned coverage**.

## Troubleshooting

### "Connection refused" from Ollama
Make sure Ollama is running: `ollama serve`

### SMTP Authentication Failed
- For Gmail: Use an App Password, not your regular password
- Check that 2FA is enabled on your Google account

### No articles fetched
- Verify your NEWS_API_KEY is valid
- Check you haven't exceeded the free tier limits
- Try enabling fewer sections to reduce requests and increase hit rate

### TTS: "Coqui TTS is not installed"
- Ensure you're using Python 3.11 (not 3.12+)
- Run `pip install TTS pydub`

### TTS: "pydub is required for MP3 conversion"
- Run `pip install pydub`
- Ensure `ffmpeg` is installed on your system

### TTS: Model download is slow
- First run downloads the TTS model (~100MB)
- Subsequent runs use the cached model

### TTS: CUDA/GPU errors
- Set `TTS_USE_CUDA=false` in `.env` to use CPU
- GPU acceleration requires CUDA-compatible NVIDIA GPU and proper drivers

## License

This project is licensed under **MIT**.

### Coqui TTS License Notice

⚠️ **The Coqui TTS library and XTTS models are licensed for non-commercial use only.**

If you intend to use this project commercially, you must either:
- Obtain a commercial license from Coqui AI
- Replace the TTS component with a commercially-licensed alternative
- Disable TTS generation (`TTS_ENABLED=false`)

For more information, see the [Coqui TTS License](https://github.com/coqui-ai/TTS/blob/dev/LICENSE.txt).
