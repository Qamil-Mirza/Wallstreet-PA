#!/usr/bin/env python3
"""
Main entry point for the 3 Things newsletter bot.

Orchestrates the daily run: fetch → summarize → email (organized by region/industry sections).
"""

import logging
import sys
from datetime import date

from .article_extractor import ensure_batch_content
from .config import ConfigError, load_config
from .email_client import EmailError, build_sectioned_email_html, send_email
from .news_client import NewsClientError, fetch_articles_by_section
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
    2. Fetch articles by section (World, US Tech/Industry, Malaysia Tech/Industry)
    3. Ensure articles have content (scrape if needed)
    4. Summarize using Ollama
    5. Build and send sectioned email
    """
    logger.info("Starting daily newsletter run...")
    
    # Step 1: Load configuration
    try:
        config = load_config()
        logger.info("Configuration loaded successfully")
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Step 2: Fetch articles by section
    try:
        sections = fetch_articles_by_section(config, per_section_limit=5)
        total_articles = sum(len(arts) for arts in sections.values())
        logger.info(f"Fetched {total_articles} articles across {len(sections)} sections")
    except NewsClientError as e:
        logger.error(f"Failed to fetch news: {e}")
        sys.exit(1)
    
    if total_articles == 0:
        logger.error("No articles found - nothing to send")
        sys.exit(1)
    
    # Log section distribution
    for section_name, articles in sections.items():
        logger.info(f"  {section_name}: {len(articles)} articles")
    
    # Step 3: Ensure content is available for all articles
    logger.info("Extracting article content...")
    for section_name in sections:
        sections[section_name] = ensure_batch_content(sections[section_name])
    logger.info("Content extraction complete")
    
    # Step 4: Summarize all articles across all sections
    all_articles = []
    for articles in sections.values():
        all_articles.extend(articles)
    
    logger.info(f"Generating summaries for {len(all_articles)} articles with Ollama...")
    summaries = summarize_articles(all_articles, config)
    logger.info(f"Generated {len(summaries)} summaries")
    
    # Step 5: Build and send sectioned email
    today = date.today()
    subject = f"The Daily Briefing – {today.strftime('%b %d, %Y')}"
    
    html = build_sectioned_email_html(today, sections, summaries)
    logger.info("Sectioned email HTML generated")
    
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
