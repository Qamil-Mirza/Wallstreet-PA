"""
Configuration management for the newsletter bot.

Loads settings from environment variables / .env file and provides
typed accessors with validation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass


@dataclass
class RSSFeedEntry:
    """Configuration for a single RSS feed from environment."""
    
    name: str
    url: str
    section: str = "RSS Feeds"
    enabled: bool = True
    limit: int = 5


@dataclass
class Config:
    """Application configuration container."""
    
    # News API settings
    news_api_key: str
    news_api_base_url: str
    
    # SMTP settings
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    recipient_email: str
    
    # Ollama settings
    ollama_base_url: str
    ollama_model: str
    
    # TTS settings (using Coqui TTS)
    tts_enabled: bool = True
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    tts_language: str = "en"  # Language code for multilingual models (XTTS)
    tts_speaker: str = "Claribel Dervla"  # Speaker name for multi-speaker models (XTTS)
    tts_speed: float = 1.0  # Speech speed (1.0 = normal, 1.2 = faster, 0.8 = slower)
    tts_output_dir: str = "audio_output"
    tts_use_cuda: bool = False
    tts_duration_minutes: float = 2.0
    
    # Section feature flags (all enabled by default)
    section_world_enabled: bool = True
    section_us_tech_enabled: bool = True
    section_us_industry_enabled: bool = True
    section_malaysia_tech_enabled: bool = True
    section_malaysia_industry_enabled: bool = True
    
    # RSS feed settings
    rss_enabled: bool = False
    rss_feeds: list[RSSFeedEntry] = field(default_factory=list)


def _get_required_env(key: str) -> str:
    """Get a required environment variable or raise ConfigError."""
    value = os.environ.get(key)
    if not value:
        raise ConfigError(f"Missing required environment variable: {key}")
    return value


def _get_optional_env(key: str, default: str) -> str:
    """Get an optional environment variable with a default."""
    return os.environ.get(key, default)


def _get_bool_env(key: str, default: bool = True) -> bool:
    """Get a boolean environment variable. Accepts true/false/1/0/yes/no."""
    value = os.environ.get(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def _parse_rss_feeds() -> list[RSSFeedEntry]:
    """
    Parse RSS feed configuration from environment variables.
    
    Format: RSS_FEEDS=url1,url2,url3 (comma-separated URLs)
    
    The feed name is automatically derived from the URL domain.
    
    Returns:
        List of RSSFeedEntry objects.
    """
    feeds = []
    
    simple_feeds = os.environ.get("RSS_FEEDS", "")
    if simple_feeds:
        for url in simple_feeds.split(","):
            url = url.strip()
            if url:
                # Extract name from URL domain
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    name = domain.replace("www.", "").split(".")[0].title()
                except Exception:
                    name = "RSS Feed"
                
                feeds.append(RSSFeedEntry(
                    name=name,
                    url=url,
                    section="RSS Feeds",
                    enabled=True,
                    limit=5,
                ))
    
    return feeds


def load_config(env_path: Optional[Path] = None) -> Config:
    """
    Load configuration from environment variables.
    
    Args:
        env_path: Optional path to .env file. If not provided,
                  searches for .env in current and parent directories.
    
    Returns:
        Config object with all settings populated.
    
    Raises:
        ConfigError: If required settings are missing.
    """
    # Load .env file if it exists
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()
    
    # Parse SMTP port with validation
    smtp_port_str = _get_optional_env("SMTP_PORT", "587")
    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        raise ConfigError(f"SMTP_PORT must be an integer, got: {smtp_port_str}")
    
    # Parse TTS duration
    tts_duration_str = _get_optional_env("TTS_DURATION_MINUTES", "2.0")
    try:
        tts_duration = float(tts_duration_str)
    except ValueError:
        raise ConfigError(f"TTS_DURATION_MINUTES must be a number, got: {tts_duration_str}")
    
    # Parse RSS feeds from environment
    rss_feeds = _parse_rss_feeds()
    
    return Config(
        news_api_key=_get_required_env("NEWS_API_KEY"),
        news_api_base_url=_get_optional_env(
            "NEWS_API_BASE_URL", 
            "https://api.marketaux.com/v1"
        ),
        smtp_host=_get_required_env("SMTP_HOST"),
        smtp_port=smtp_port,
        smtp_user=_get_required_env("SMTP_USER"),
        smtp_password=_get_required_env("SMTP_PASSWORD"),
        recipient_email=_get_required_env("RECIPIENT_EMAIL"),
        ollama_base_url=_get_optional_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=_get_optional_env("OLLAMA_MODEL", "llama3"),
        # TTS settings
        tts_enabled=_get_bool_env("TTS_ENABLED", True),
        tts_model=_get_optional_env("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC"),
        tts_language=_get_optional_env("TTS_LANGUAGE", "en"),
        tts_speaker=_get_optional_env("TTS_SPEAKER", "Claribel Dervla"),
        tts_speed=float(_get_optional_env("TTS_SPEED", "1.0")),
        tts_output_dir=_get_optional_env("TTS_OUTPUT_DIR", "audio_output"),
        tts_use_cuda=_get_bool_env("TTS_USE_CUDA", False),
        tts_duration_minutes=tts_duration,
        # Section feature flags
        section_world_enabled=_get_bool_env("SECTION_WORLD_ENABLED", True),
        section_us_tech_enabled=_get_bool_env("SECTION_US_TECH_ENABLED", True),
        section_us_industry_enabled=_get_bool_env("SECTION_US_INDUSTRY_ENABLED", True),
        section_malaysia_tech_enabled=_get_bool_env("SECTION_MALAYSIA_TECH_ENABLED", True),
        section_malaysia_industry_enabled=_get_bool_env("SECTION_MALAYSIA_INDUSTRY_ENABLED", True),
        # RSS settings
        rss_enabled=_get_bool_env("RSS_ENABLED", False),
        rss_feeds=rss_feeds,
    )
