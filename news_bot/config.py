"""
Configuration management for the newsletter bot.

Loads settings from environment variables / .env file and provides
typed accessors with validation.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass


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


def _get_required_env(key: str) -> str:
    """Get a required environment variable or raise ConfigError."""
    value = os.environ.get(key)
    if not value:
        raise ConfigError(f"Missing required environment variable: {key}")
    return value


def _get_optional_env(key: str, default: str) -> str:
    """Get an optional environment variable with a default."""
    return os.environ.get(key, default)


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
    )
