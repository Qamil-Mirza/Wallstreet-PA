#!/bin/bash
# =============================================================================
# Newsletter Bot Run Script
# =============================================================================
# Runs the newsletter bot with proper environment setup.
# Suitable for cron jobs.
#
# Usage:
#   ./scripts/run.sh
#
# Crontab example (run daily at 7 AM):
#   0 7 * * * /path/to/newsletter/scripts/run.sh >> /path/to/newsletter/logs/cron.log 2>&1
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Activate virtual environment
source venv/bin/activate

# Run the newsletter bot
python -m news_bot.main

# Capture exit code
EXIT_CODE=$?

# Log completion
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Newsletter run completed successfully"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Newsletter run failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
