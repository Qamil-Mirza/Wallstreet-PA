#!/usr/bin/env python3
"""
Main entry point for the 3 Things newsletter bot.

Orchestrates the daily run: fetch → classify → select → summarize → email.
"""

import logging
import sys
from datetime import date

from .article_extractor import ensure_batch_content
from .classifier import bucket_articles
from .config import ConfigError, load_config
from .email_client import EmailError, build_email_html, send_email
from .news_client import NewsClientError, fetch_recent_articles
from .selection import select_three_articles
from .summarizer import summarize_articles


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_daily() -> None:
    """
    Execute the daily newsletter pipeline.
    
    Steps:
    1. Load configuration
    2. Fetch recent articles from news API
    3. Ensure articles have content (scrape if needed)
    4. Classify and bucket articles
    5. Select top 3 articles
    6. Summarize using Ollama
    7. Build and send email
    """
    logger.info("Starting daily newsletter run...")
    
    # Step 1: Load configuration
    try:
        config = load_config()
        logger.info("Configuration loaded successfully")
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Step 2: Fetch articles
    try:
        articles = fetch_recent_articles(config, limit=30)
        logger.info(f"Fetched {len(articles)} articles")
    except NewsClientError as e:
        logger.error(f"Failed to fetch news: {e}")
        sys.exit(1)
    
    if not articles:
        logger.error("No articles found - nothing to send")
        sys.exit(1)
    
    # Step 3: Ensure content is available
    articles_with_content = ensure_batch_content(articles)
    logger.info("Content extraction complete")
    
    # Step 4: Classify and bucket
    buckets = bucket_articles(articles_with_content)
    logger.info(
        f"Classification: {len(buckets['macro'])} macro, "
        f"{len(buckets['deal'])} deal, {len(buckets['feature'])} feature"
    )
    
    # Step 5: Select top 3
    selected = select_three_articles(buckets)
    logger.info(f"Selected {len(selected)} articles for newsletter")
    
    if not selected:
        logger.error("No articles selected - nothing to send")
        sys.exit(1)
    
    for i, article in enumerate(selected, 1):
        logger.info(f"  {i}. {article.title[:60]}...")
    
    # Step 6: Summarize
    logger.info("Generating summaries with Ollama...")
    summaries = summarize_articles(selected, config)
    logger.info(f"Generated {len(summaries)} summaries")
    
    # Step 7: Build and send email
    today = date.today()
    subject = f"Your 3 Things – Markets & Economy – {today.strftime('%b %d, %Y')}"
    
    html = build_email_html(today, selected, summaries, buckets)
    logger.info("Email HTML generated")
    
    try:
        send_email(config, subject, html)
        logger.info("Newsletter sent successfully!")
    except EmailError as e:
        logger.error(f"Failed to send email: {e}")
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    try:
        run_daily()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
