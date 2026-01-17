"""Tests for configuration management."""

from unittest.mock import patch

import pytest

from news_bot.config import Config, ConfigError, load_config


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_success(self, monkeypatch):
        """Test successful config loading with all required vars."""
        monkeypatch.setenv("NEWS_API_KEY", "test_key")
        monkeypatch.setenv("NEWS_API_BASE_URL", "https://api.test.com")
        monkeypatch.setenv("SMTP_HOST", "smtp.test.com")
        monkeypatch.setenv("SMTP_PORT", "465")
        monkeypatch.setenv("SMTP_USER", "user@test.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret123")
        monkeypatch.setenv("RECIPIENT_EMAIL", "recipient@test.com")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.1")

        config = load_config()

        assert config.news_api_key == "test_key"
        assert config.news_api_base_url == "https://api.test.com"
        assert config.smtp_host == "smtp.test.com"
        assert config.smtp_port == 465
        assert config.smtp_user == "user@test.com"
        assert config.smtp_password == "secret123"
        assert config.recipient_email == "recipient@test.com"
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.ollama_model == "llama3.1"

    def test_load_config_uses_defaults(self, monkeypatch):
        """Test that optional values use defaults when not set."""
        monkeypatch.setenv("NEWS_API_KEY", "test_key")
        monkeypatch.setenv("SMTP_HOST", "smtp.test.com")
        monkeypatch.setenv("SMTP_USER", "user@test.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret123")
        monkeypatch.setenv("RECIPIENT_EMAIL", "recipient@test.com")
        # Clear optional vars
        monkeypatch.delenv("NEWS_API_BASE_URL", raising=False)
        monkeypatch.delenv("SMTP_PORT", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)

        config = load_config()

        assert config.news_api_base_url == "https://api.marketaux.com/v1"
        assert config.smtp_port == 587
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.ollama_model == "llama3"

    def test_load_config_missing_news_api_key(self, monkeypatch):
        """Test that missing NEWS_API_KEY raises ConfigError."""
        # Clear all potentially conflicting env vars first
        for key in ["NEWS_API_KEY", "NEWS_API_BASE_URL", "SMTP_HOST", "SMTP_PORT",
                    "SMTP_USER", "SMTP_PASSWORD", "RECIPIENT_EMAIL",
                    "OLLAMA_BASE_URL", "OLLAMA_MODEL"]:
            monkeypatch.delenv(key, raising=False)
        
        monkeypatch.setenv("SMTP_HOST", "smtp.test.com")
        monkeypatch.setenv("SMTP_USER", "user@test.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret123")
        monkeypatch.setenv("RECIPIENT_EMAIL", "recipient@test.com")

        # Patch load_dotenv to prevent loading from .env file
        with patch("news_bot.config.load_dotenv"):
            with pytest.raises(ConfigError) as exc_info:
                load_config()

        assert "NEWS_API_KEY" in str(exc_info.value)

    def test_load_config_missing_smtp_host(self, monkeypatch):
        """Test that missing SMTP_HOST raises ConfigError."""
        # Clear all potentially conflicting env vars first
        for key in ["NEWS_API_KEY", "NEWS_API_BASE_URL", "SMTP_HOST", "SMTP_PORT",
                    "SMTP_USER", "SMTP_PASSWORD", "RECIPIENT_EMAIL",
                    "OLLAMA_BASE_URL", "OLLAMA_MODEL"]:
            monkeypatch.delenv(key, raising=False)
        
        monkeypatch.setenv("NEWS_API_KEY", "test_key")
        monkeypatch.setenv("SMTP_USER", "user@test.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret123")
        monkeypatch.setenv("RECIPIENT_EMAIL", "recipient@test.com")

        # Patch load_dotenv to prevent loading from .env file
        with patch("news_bot.config.load_dotenv"):
            with pytest.raises(ConfigError) as exc_info:
                load_config()

        assert "SMTP_HOST" in str(exc_info.value)

    def test_load_config_invalid_smtp_port(self, monkeypatch):
        """Test that non-integer SMTP_PORT raises ConfigError."""
        monkeypatch.setenv("NEWS_API_KEY", "test_key")
        monkeypatch.setenv("SMTP_HOST", "smtp.test.com")
        monkeypatch.setenv("SMTP_PORT", "not_a_number")
        monkeypatch.setenv("SMTP_USER", "user@test.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret123")
        monkeypatch.setenv("RECIPIENT_EMAIL", "recipient@test.com")

        with pytest.raises(ConfigError) as exc_info:
            load_config()

        assert "SMTP_PORT" in str(exc_info.value)
        assert "integer" in str(exc_info.value).lower()


class TestConfig:
    """Tests for Config dataclass."""

    def test_config_is_dataclass(self):
        """Test that Config is a proper dataclass."""
        config = Config(
            news_api_key="key",
            news_api_base_url="https://api.test.com",
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_user="user@test.com",
            smtp_password="secret",
            recipient_email="recipient@test.com",
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3",
        )

        assert config.news_api_key == "key"
        assert config.smtp_port == 587

    def test_config_section_flags_default_to_true(self):
        """Test that section flags default to True."""
        config = Config(
            news_api_key="key",
            news_api_base_url="https://api.test.com",
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_user="user@test.com",
            smtp_password="secret",
            recipient_email="recipient@test.com",
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3",
        )

        assert config.section_world_enabled is True
        assert config.section_us_tech_enabled is True
        assert config.section_us_industry_enabled is True
        assert config.section_malaysia_tech_enabled is True
        assert config.section_malaysia_industry_enabled is True


class TestSectionFeatureFlags:
    """Tests for section feature flag loading."""

    def test_load_config_section_flags_from_env(self, monkeypatch):
        """Test that section flags are loaded from environment."""
        monkeypatch.setenv("NEWS_API_KEY", "test_key")
        monkeypatch.setenv("SMTP_HOST", "smtp.test.com")
        monkeypatch.setenv("SMTP_USER", "user@test.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret123")
        monkeypatch.setenv("RECIPIENT_EMAIL", "recipient@test.com")
        
        # Set section flags
        monkeypatch.setenv("SECTION_WORLD_ENABLED", "true")
        monkeypatch.setenv("SECTION_US_TECH_ENABLED", "false")
        monkeypatch.setenv("SECTION_US_INDUSTRY_ENABLED", "1")
        monkeypatch.setenv("SECTION_MALAYSIA_TECH_ENABLED", "0")
        monkeypatch.setenv("SECTION_MALAYSIA_INDUSTRY_ENABLED", "yes")

        config = load_config()

        assert config.section_world_enabled is True
        assert config.section_us_tech_enabled is False
        assert config.section_us_industry_enabled is True
        assert config.section_malaysia_tech_enabled is False
        assert config.section_malaysia_industry_enabled is True

    def test_load_config_section_flags_default_true(self, monkeypatch):
        """Test that section flags default to True when not set."""
        monkeypatch.setenv("NEWS_API_KEY", "test_key")
        monkeypatch.setenv("SMTP_HOST", "smtp.test.com")
        monkeypatch.setenv("SMTP_USER", "user@test.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret123")
        monkeypatch.setenv("RECIPIENT_EMAIL", "recipient@test.com")
        
        # Clear section flags to ensure defaults are used
        for key in ["SECTION_WORLD_ENABLED", "SECTION_US_TECH_ENABLED",
                    "SECTION_US_INDUSTRY_ENABLED", "SECTION_MALAYSIA_TECH_ENABLED",
                    "SECTION_MALAYSIA_INDUSTRY_ENABLED"]:
            monkeypatch.delenv(key, raising=False)

        # Patch load_dotenv to prevent loading from .env file
        with patch("news_bot.config.load_dotenv"):
            config = load_config()

        assert config.section_world_enabled is True
        assert config.section_us_tech_enabled is True
        assert config.section_us_industry_enabled is True
        assert config.section_malaysia_tech_enabled is True
        assert config.section_malaysia_industry_enabled is True
