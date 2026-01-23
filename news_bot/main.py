#!/usr/bin/env python3
"""
Main entry point for the 3 Things newsletter bot.

Orchestrates the daily run: fetch → summarize → email (organized by region/industry sections).
Optionally generates TTS audio broadcast from summaries.
"""

import logging
import sys
from datetime import date, datetime
from pathlib import Path

from .article_extractor import ensure_batch_content
from .config import Config, ConfigError, load_config
from .email_client import EmailError, build_sectioned_email_html, send_email
from .news_client import NewsClientError, fetch_articles_by_section
from .script_generator import ScriptGeneratorError, generate_broadcast_script
from .news_client import ArticleMeta
from .summarizer import summarize_articles
from .tts_engine import TTSConfig, TTSEngine, TTSEngineError


def _setup_logging() -> Path:
    """
    Configure logging to both console and file.
    
    Creates a timestamped log file in the logs/ directory.
    
    Returns:
        Path to the log file.
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"run_{timestamp}.log"
    
    # Log format
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_logger.addHandler(file_handler)
    
    return log_file


# Setup logging and get log file path
_log_file = _setup_logging()
logger = logging.getLogger(__name__)


def generate_tts_broadcast(summaries: dict[str, str], config: Config) -> Path | None:
    """
    Generate TTS audio broadcast from article summaries.
    
    Steps:
    1. Generate radio host script from summaries
    2. Convert script to audio using Coqui TTS
    3. Save MP3 to configured output directory
    
    Args:
        summaries: Dictionary mapping article IDs to summary text.
        config: Application configuration.
    
    Returns:
        Path to generated MP3 file, or None if TTS is disabled or fails.
    """
    if not config.tts_enabled:
        logger.info("TTS is disabled, skipping audio generation")
        return None
    
    if not summaries:
        logger.warning("No summaries available for TTS generation")
        return None
    
    # Generate radio script
    logger.info("Generating radio broadcast script...")
    try:
        script = generate_broadcast_script(
            summaries=summaries,
            ollama_model=config.ollama_model,
            ollama_base_url=config.ollama_base_url,
            duration_minutes=config.tts_duration_minutes,
        )
        logger.info(f"Script generated: {len(script)} characters, ~{len(script.split())} words")
    except ScriptGeneratorError as e:
        logger.error(f"Script generation failed: {e}")
        return None
    
    # Initialize TTS engine
    tts_config = TTSConfig(
        model_name=config.tts_model,
        language=config.tts_language,
        speaker=config.tts_speaker,
        speed=config.tts_speed,
        output_dir=Path(config.tts_output_dir),
        use_cuda=config.tts_use_cuda,
    )
    
    try:
        engine = TTSEngine(tts_config)
    except TTSEngineError as e:
        logger.error(f"TTS engine initialization failed: {e}")
        return None
    
    # Generate audio
    logger.info("Generating TTS audio...")
    today = date.today()
    filename = f"broadcast_{today.strftime('%Y%m%d')}"
    
    try:
        audio_path = engine.synthesize_to_mp3(script, filename)
        logger.info(f"Audio broadcast saved: {audio_path}")
        return audio_path
    except TTSEngineError as e:
        logger.error(f"TTS synthesis failed: {e}")
        return None


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
    logger.info(f"Log file: {_log_file.absolute()}")
    
    # Load config
    try:
        config = load_config()
        logger.info("Configuration loaded successfully")
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Fetch articles by section
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
    
    # Ensure content is available for all articles
    logger.info("Extracting article content...")
    for section_name in sections:
        sections[section_name] = ensure_batch_content(sections[section_name])
    logger.info("Content extraction complete")
    
    # Summarize all articles across all sections
    all_articles = []
    for articles in sections.values():
        all_articles.extend(articles)
    
    logger.info(f"Generating summaries for {len(all_articles)} articles with Ollama...")
    summaries = summarize_articles(all_articles, config)
    
    # Filter out dropped articles (None summaries = invalid/refusal)
    valid_summaries = {k: v for k, v in summaries.items() if v is not None}
    dropped_count = len(summaries) - len(valid_summaries)
    
    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} article(s) with invalid summaries")
    
    logger.info(f"Generated {len(valid_summaries)} valid summaries")
    
    # Filter sections to only include articles with valid summaries
    for section_name in sections:
        sections[section_name] = [
            article for article in sections[section_name]
            if article.id in valid_summaries
        ]
    
    # Check if we have any articles left after filtering
    remaining_articles = sum(len(arts) for arts in sections.values())
    if remaining_articles == 0:
        logger.error("No articles with valid summaries - nothing to send")
        sys.exit(1)
    
    logger.info(f"Final article count after filtering: {remaining_articles}")
    
    # Generate TTS audio broadcast if enabled (use only valid summaries)
    audio_path = generate_tts_broadcast(valid_summaries, config)
    if audio_path:
        logger.info(f"TTS broadcast ready: {audio_path}")
    
    # Build and send sectioned email
    today = date.today()
    subject = f"The Daily Briefing – {today.strftime('%b %d, %Y')}"
    
    html = build_sectioned_email_html(today, sections, valid_summaries)
    logger.info("Sectioned email HTML generated")
    
    try:
        send_email(config, subject, html, attachment_path=audio_path)
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
