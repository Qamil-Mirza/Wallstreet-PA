"""Tests for news client module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from news_bot.config import Config
from news_bot.news_client import (
    ArticleMeta,
    NewsClientError,
    _normalize_marketaux_article,
    _normalize_fmp_article,
    _parse_datetime,
    fetch_recent_articles,
)


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    return Config(
        news_api_key="test_api_key",
        news_api_base_url="https://api.marketaux.com/v1",
        smtp_host="smtp.test.com",
        smtp_port=587,
        smtp_user="user@test.com",
        smtp_password="secret",
        recipient_email="recipient@test.com",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3",
    )


class TestParseDatetime:
    """Tests for datetime parsing."""

    def test_parse_iso_format_with_milliseconds(self):
        """Test parsing ISO format with milliseconds."""
        result = _parse_datetime("2024-01-15T10:30:00.123Z")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_parse_iso_format_without_milliseconds(self):
        """Test parsing ISO format without milliseconds."""
        result = _parse_datetime("2024-01-15T10:30:00Z")
        assert result.year == 2024
        assert result.month == 1

    def test_parse_date_only(self):
        """Test parsing date-only format."""
        result = _parse_datetime("2024-01-15")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_invalid_returns_now(self):
        """Test that invalid format returns current time."""
        result = _parse_datetime("not-a-date")
        # Should be close to now
        assert abs((datetime.now() - result).total_seconds()) < 5


class TestNormalizeMarketauxArticle:
    """Tests for MarketAux article normalization."""

    def test_normalize_complete_article(self):
        """Test normalizing a complete MarketAux article."""
        raw = {
            "uuid": "abc123",
            "title": "Fed Raises Interest Rates",
            "url": "https://example.com/article",
            "description": "The Federal Reserve raised rates today.",
            "snippet": "Full article content here...",
            "published_at": "2024-01-15T10:30:00Z",
            "source": "Reuters",
        }

        result = _normalize_marketaux_article(raw)

        assert result is not None
        assert result.id == "abc123"
        assert result.title == "Fed Raises Interest Rates"
        assert result.url == "https://example.com/article"
        assert result.summary == "The Federal Reserve raised rates today."
        assert result.content == "Full article content here..."
        assert result.source == "Reuters"

    def test_normalize_missing_title_returns_none(self):
        """Test that missing title returns None."""
        raw = {
            "uuid": "abc123",
            "url": "https://example.com/article",
        }

        result = _normalize_marketaux_article(raw)
        assert result is None

    def test_normalize_missing_url_returns_none(self):
        """Test that missing URL returns None."""
        raw = {
            "uuid": "abc123",
            "title": "Some Title",
        }

        result = _normalize_marketaux_article(raw)
        assert result is None

    def test_normalize_uses_url_as_id_fallback(self):
        """Test that URL is used as ID when uuid is missing."""
        raw = {
            "title": "Some Title",
            "url": "https://example.com/article",
        }

        result = _normalize_marketaux_article(raw)

        assert result is not None
        assert result.id == "https://example.com/article"


class TestNormalizeFmpArticle:
    """Tests for FMP article normalization."""

    def test_normalize_complete_fmp_article(self):
        """Test normalizing a complete FMP article."""
        raw = {
            "title": "Apple Acquires Startup",
            "url": "https://example.com/apple",
            "text": "Apple Inc announced acquisition of a small startup for $1B.",
            "publishedDate": "2024-01-15T10:30:00Z",
            "site": "Bloomberg",
        }

        result = _normalize_fmp_article(raw)

        assert result is not None
        assert result.title == "Apple Acquires Startup"
        assert result.url == "https://example.com/apple"
        assert result.source == "Bloomberg"
        assert "Apple Inc" in result.content


class TestFetchRecentArticles:
    """Tests for fetching articles from APIs."""

    def test_fetch_recent_articles_marketaux_normalization(self, mock_config):
        """Test that MarketAux response is normalized correctly."""
        mock_response = {
            "data": [
                {
                    "uuid": "article-1",
                    "title": "First Article",
                    "url": "https://example.com/1",
                    "description": "Description 1",
                    "published_at": "2024-01-15T10:00:00Z",
                },
                {
                    "uuid": "article-2",
                    "title": "Second Article",
                    "url": "https://example.com/2",
                    "description": "Description 2",
                    "published_at": "2024-01-15T11:00:00Z",
                },
            ]
        }

        with patch("news_bot.news_client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = MagicMock()

            result = fetch_recent_articles(mock_config, limit=10)

        assert len(result) == 2
        assert result[0].id == "article-1"
        assert result[0].title == "First Article"
        assert result[1].id == "article-2"

    def test_fetch_recent_articles_handles_empty(self, mock_config):
        """Test handling of empty API response."""
        mock_response = {"data": []}

        with patch("news_bot.news_client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = MagicMock()

            result = fetch_recent_articles(mock_config, limit=10)

        assert result == []

    def test_fetch_recent_articles_skips_invalid(self, mock_config):
        """Test that invalid articles are skipped."""
        mock_response = {
            "data": [
                {
                    "uuid": "valid",
                    "title": "Valid Article",
                    "url": "https://example.com/valid",
                    "published_at": "2024-01-15T10:00:00Z",
                },
                {
                    "uuid": "missing-url",
                    "title": "No URL Article",
                    # Missing url field
                },
                {
                    "uuid": "missing-title",
                    # Missing title field
                    "url": "https://example.com/notitle",
                },
            ]
        }

        with patch("news_bot.news_client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = MagicMock()

            result = fetch_recent_articles(mock_config, limit=10)

        assert len(result) == 1
        assert result[0].id == "valid"

    def test_fetch_recent_articles_api_error(self, mock_config):
        """Test that API errors raise NewsClientError."""
        import requests as req

        with patch("news_bot.news_client.requests.get") as mock_get:
            mock_get.side_effect = req.RequestException("Connection failed")

            with pytest.raises(NewsClientError) as exc_info:
                fetch_recent_articles(mock_config)

            assert "Connection failed" in str(exc_info.value)

    def test_fetch_fmp_articles(self):
        """Test fetching from FMP API."""
        config = Config(
            news_api_key="test_key",
            news_api_base_url="https://financialmodelingprep.com",
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_user="user@test.com",
            smtp_password="secret",
            recipient_email="recipient@test.com",
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3",
        )

        mock_response = [
            {
                "title": "FMP Article",
                "url": "https://example.com/fmp",
                "text": "Article content",
                "publishedDate": "2024-01-15T10:00:00Z",
                "site": "FMP News",
            }
        ]

        with patch("news_bot.news_client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = MagicMock()

            result = fetch_recent_articles(config, limit=10)

        assert len(result) == 1
        assert result[0].title == "FMP Article"
