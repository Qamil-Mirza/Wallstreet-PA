# 📊 3 Things Newsletter Bot

A fully automated daily finance/economics email bot that:

- **Pulls** recent financial news from free APIs (MarketAux or FMP)
- **Classifies** articles into Macro, Deal/Corporate, and Feature/Trend categories
- **Summarizes** each story using a local Ollama LLM in analyst-style voice
- **Sends** a beautifully formatted HTML email every morning

All components run locally on modest hardware using only free services.

## Features

- 🆓 **Free tier APIs**: MarketAux or Financial Modeling Prep
- 🤖 **Local LLM**: Ollama (llama3, llama3.1, etc.)
- 📧 **SMTP email**: Works with Gmail, Outlook, or any SMTP provider
- 🧪 **Fully tested**: pytest test suite with mocks
- 📦 **Modular design**: Each component in its own module

## Quick Start

### 1. Clone and Install

```bash
cd newsletter
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example config and fill in your values:

```bash
cp env.example .env
```

Edit `.env` with your credentials:

```bash
# Get a free API key from https://www.marketaux.com/
NEWS_API_KEY=your_marketaux_api_key

# Gmail with App Password (recommended)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
RECIPIENT_EMAIL=recipient@example.com

# Ollama settings (make sure Ollama is running)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

### 3. Start Ollama

Make sure Ollama is running with your chosen model:

```bash
ollama pull llama3
ollama serve
```

### 4. Run the Bot

```bash
python -m news_bot.main
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
│   ├── config.py           # Configuration management
│   ├── news_client.py      # News API client
│   ├── article_extractor.py # Content scraping
│   ├── classifier.py       # Article categorization
│   ├── selection.py        # Top 3 selection logic
│   ├── summarizer.py       # Ollama LLM integration
│   ├── email_client.py     # HTML email rendering & SMTP
│   └── main.py             # Orchestration & entry point
├── tests/
│   ├── test_config.py
│   ├── test_news_client.py
│   ├── test_article_extractor.py
│   ├── test_classifier.py
│   ├── test_selection.py
│   ├── test_summarizer.py
│   └── test_email_client.py
├── requirements.txt
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

## Supported News APIs

### MarketAux (Recommended)
- Free tier: 100 requests/day
- Sign up: https://www.marketaux.com/

### Financial Modeling Prep
- Free tier: 250 requests/day
- Sign up: https://financialmodelingprep.com/

To use FMP, set:
```
NEWS_API_BASE_URL=https://financialmodelingprep.com
```

## Gmail Setup

1. Enable 2-Factor Authentication on your Google account
2. Go to https://myaccount.google.com/apppasswords
3. Generate an App Password for "Mail"
4. Use the 16-character password as `SMTP_PASSWORD`

## Article Categories

- **Macro & Economics**: Fed, ECB, inflation, GDP, jobs, bonds, rates
- **Deals & Corporate**: Mergers, acquisitions, IPOs, buyouts
- **Feature & Trends**: Everything else (industry trends, earnings, etc.)

## Troubleshooting

### "Connection refused" from Ollama
Make sure Ollama is running: `ollama serve`

### SMTP Authentication Failed
- For Gmail: Use an App Password, not your regular password
- Check that 2FA is enabled on your Google account

### No articles fetched
- Verify your NEWS_API_KEY is valid
- Check you haven't exceeded the free tier limits

## License

MIT
