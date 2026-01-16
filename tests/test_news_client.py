"""Tests for news client module."""

from datetime import datetime
from unittest.mock import MagicMock, patch, call

import pytest

from news_bot.config import Config
from news_bot.news_client import (
    ArticleMeta,
    NewsClientError,
    NewsFeedConfig,
    ALL_FEEDS,
    _get_enabled_feeds,
    _normalize_marketaux_article,
    _normalize_fmp_article,
    _parse_datetime,
    _fetch_marketaux_single,
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
        section_world_enabled=True,
        section_us_tech_enabled=True,
        section_us_industry_enabled=True,
        section_malaysia_tech_enabled=True,
        section_malaysia_industry_enabled=True,
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


class TestFetchMarketauxSingle:
    """Tests for single MarketAux feed fetching."""

    def test_fetch_single_with_filters(self, mock_config):
        """Test that filters are passed correctly to API."""
        mock_response = {
            "meta": {"returned": 2, "found": 10},
            "data": [
                {
                    "uuid": "article-1",
                    "title": "Tech Article",
                    "url": "https://example.com/tech1",
                    "description": "Tech news",
                    "published_at": "2024-01-15T10:00:00Z",
                },
            ]
        }

        with patch("news_bot.news_client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = MagicMock()

            result = _fetch_marketaux_single(
                mock_config,
                limit=10,
                countries="us",
                industries="Technology",
                feed_name="US Tech",
            )

            # Verify the API was called with correct params
            call_args = mock_get.call_args
            params = call_args[1]["params"]
            assert params["countries"] == "us"
            assert params["industries"] == "Technology"
            assert params["language"] == "en"

        assert len(result) == 1
        assert result[0].title == "Tech Article"

    def test_fetch_single_no_filters(self, mock_config):
        """Test fetch without country/industry filters."""
        mock_response = {
            "meta": {"returned": 1, "found": 100},
            "data": [
                {
                    "uuid": "world-1",
                    "title": "World News",
                    "url": "https://example.com/world",
                    "published_at": "2024-01-15T10:00:00Z",
                },
            ]
        }

        with patch("news_bot.news_client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = MagicMock()

            result = _fetch_marketaux_single(mock_config, limit=10)

            # Verify no country/industry params when not specified
            call_args = mock_get.call_args
            params = call_args[1]["params"]
            assert "countries" not in params
            assert "industries" not in params

        assert len(result) == 1


class TestFetchRecentArticles:
    """Tests for fetching articles from APIs."""

    def test_fetch_recent_articles_multi_feed(self, mock_config):
        """Test that multiple feeds are fetched and deduplicated."""
        # Create different responses for each feed
        def create_response(feed_index):
            return {
                "meta": {"returned": 2, "found": 10},
                "data": [
                    {
                        "uuid": f"article-{feed_index}-1",
                        "title": f"Article from feed {feed_index}",
                        "url": f"https://example.com/feed{feed_index}/1",
                        "description": "Description",
                        "published_at": "2024-01-15T10:00:00Z",
                    },
                    {
                        "uuid": f"article-{feed_index}-2",
                        "title": f"Second article from feed {feed_index}",
                        "url": f"https://example.com/feed{feed_index}/2",
                        "description": "Description 2",
                        "published_at": "2024-01-15T11:00:00Z",
                    },
                ]
            }

        with patch("news_bot.news_client.requests.get") as mock_get:
            # Return different response for each call (5 feeds)
            mock_get.return_value.raise_for_status = MagicMock()
            mock_get.return_value.json.side_effect = [
                create_response(i) for i in range(len(ALL_FEEDS))
            ]

            result = fetch_recent_articles(mock_config, limit=30)

        # Should have called API once per feed (5 times)
        assert mock_get.call_count == len(ALL_FEEDS)
        
        # Should have 2 articles per feed * 5 feeds = 10 unique articles
        assert len(result) == 10

    def test_fetch_recent_articles_deduplicates_by_url(self, mock_config):
        """Test that duplicate URLs across feeds are removed."""
        # All feeds return the same article (same URL)
        mock_response = {
            "meta": {"returned": 1, "found": 10},
            "data": [
                {
                    "uuid": "duplicate-article",
                    "title": "Same Article",
                    "url": "https://example.com/same-url",
                    "description": "Description",
                    "published_at": "2024-01-15T10:00:00Z",
                },
            ]
        }

        with patch("news_bot.news_client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = MagicMock()

            result = fetch_recent_articles(mock_config, limit=30)

        # Should only have 1 article despite 5 feeds returning the same URL
        assert len(result) == 1
        assert result[0].url == "https://example.com/same-url"

    def test_fetch_recent_articles_handles_empty(self, mock_config):
        """Test handling of empty API responses from all feeds."""
        mock_response = {"meta": {"returned": 0, "found": 0}, "data": []}

        with patch("news_bot.news_client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = MagicMock()

            result = fetch_recent_articles(mock_config, limit=10)

        assert result == []

    def test_fetch_recent_articles_skips_invalid(self, mock_config):
        """Test that invalid articles are skipped."""
        mock_response = {
            "meta": {"returned": 3, "found": 3},
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

        # Only 1 valid article per feed, but same URL so deduplicated to 1
        assert len(result) == 1
        assert result[0].id == "valid"

    def test_fetch_recent_articles_partial_failure(self, mock_config):
        """Test that partial feed failures still return results from successful feeds."""
        import requests as req

        call_count = [0]
        
        def mock_get_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_response = MagicMock()
            if call_count[0] <= 2:
                # First 2 feeds fail
                raise req.RequestException("Connection failed")
            else:
                # Remaining feeds succeed
                mock_response.json.return_value = {
                    "meta": {"returned": 1, "found": 1},
                    "data": [
                        {
                            "uuid": f"article-{call_count[0]}",
                            "title": f"Article {call_count[0]}",
                            "url": f"https://example.com/{call_count[0]}",
                            "published_at": "2024-01-15T10:00:00Z",
                        },
                    ]
                }
                mock_response.raise_for_status = MagicMock()
                return mock_response

        with patch("news_bot.news_client.requests.get", side_effect=mock_get_side_effect):
            result = fetch_recent_articles(mock_config, limit=30)

        # Should have results from the 3 successful feeds
        assert len(result) == 3

    def test_fetch_recent_articles_all_feeds_fail(self, mock_config):
        """Test that all feeds failing raises NewsClientError."""
        import requests as req

        with patch("news_bot.news_client.requests.get") as mock_get:
            mock_get.side_effect = req.RequestException("Connection failed")

            with pytest.raises(NewsClientError) as exc_info:
                fetch_recent_articles(mock_config)

            assert "All feeds failed" in str(exc_info.value)

    def test_fetch_fmp_articles(self):
        """Test fetching from FMP API (uses single endpoint, not multi-feed)."""
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
            section_world_enabled=True,
            section_us_tech_enabled=True,
            section_us_industry_enabled=True,
            section_malaysia_tech_enabled=True,
            section_malaysia_industry_enabled=True,
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


class TestNewsFeedConfig:
    """Tests for NewsFeedConfig dataclass."""

    def test_all_feeds_configured(self):
        """Test that all feeds are properly configured."""
        assert len(ALL_FEEDS) == 5
        
        feed_names = [f.name for f in ALL_FEEDS]
        assert "World News" in feed_names
        assert "US Tech" in feed_names
        assert "US Industry" in feed_names
        assert "Malaysia Tech" in feed_names
        assert "Malaysia Industry" in feed_names

    def test_feeds_have_config_keys(self):
        """Test that all feeds have config keys for feature flags."""
        expected_keys = {"world", "us_tech", "us_industry", "malaysia_tech", "malaysia_industry"}
        actual_keys = {f.config_key for f in ALL_FEEDS}
        assert actual_keys == expected_keys


class TestFeatureFlags:
    """Tests for section feature flags."""

    def test_all_sections_enabled_returns_all_feeds(self, mock_config):
        """Test that all feeds are returned when all sections enabled."""
        enabled = _get_enabled_feeds(mock_config)
        assert len(enabled) == 5

    def test_disable_malaysia_sections(self, mock_config):
        """Test that disabling Malaysia sections filters them out."""
        mock_config.section_malaysia_tech_enabled = False
        mock_config.section_malaysia_industry_enabled = False
        
        enabled = _get_enabled_feeds(mock_config)
        
        assert len(enabled) == 3
        feed_names = [f.name for f in enabled]
        assert "World News" in feed_names
        assert "US Tech" in feed_names
        assert "US Industry" in feed_names
        assert "Malaysia Tech" not in feed_names
        assert "Malaysia Industry" not in feed_names

    def test_only_world_enabled(self, mock_config):
        """Test enabling only World News section."""
        mock_config.section_us_tech_enabled = False
        mock_config.section_us_industry_enabled = False
        mock_config.section_malaysia_tech_enabled = False
        mock_config.section_malaysia_industry_enabled = False
        
        enabled = _get_enabled_feeds(mock_config)
        
        assert len(enabled) == 1
        assert enabled[0].name == "World News"

    def test_no_sections_enabled_falls_back_to_world(self, mock_config):
        """Test that disabling all sections falls back to World News."""
        mock_config.section_world_enabled = False
        mock_config.section_us_tech_enabled = False
        mock_config.section_us_industry_enabled = False
        mock_config.section_malaysia_tech_enabled = False
        mock_config.section_malaysia_industry_enabled = False
        
        enabled = _get_enabled_feeds(mock_config)
        
        # Should fallback to World News
        assert len(enabled) == 1
        assert enabled[0].name == "World News"

    def test_us_feeds_have_country_filter(self):
        """Test that US feeds have correct country filter."""
        us_feeds = [f for f in ALL_FEEDS if "US" in f.name]
        assert len(us_feeds) == 2
        for feed in us_feeds:
            assert feed.countries == "us"

    def test_malaysia_feeds_have_country_filter(self):
        """Test that Malaysia feeds have correct country filter."""
        my_feeds = [f for f in ALL_FEEDS if "Malaysia" in f.name]
        assert len(my_feeds) == 2
        for feed in my_feeds:
            assert feed.countries == "my"

    def test_world_news_has_no_filters(self):
        """Test that World News feed has no country/industry filters."""
        world_feed = next(f for f in ALL_FEEDS if f.name == "World News")
        assert world_feed.countries is None
        assert world_feed.industries is None


class TestFetchArticlesBySection:
    """Tests for sectioned article fetching."""

    def test_fetch_articles_by_section_returns_dict(self, mock_config):
        """Test that fetch_articles_by_section returns a dictionary."""
        from news_bot.news_client import fetch_articles_by_section
        
        mock_response = {
            "meta": {"returned": 2, "found": 10},
            "data": [
                {
                    "uuid": "article-1",
                    "title": "Test Article",
                    "url": "https://example.com/1",
                    "published_at": "2024-01-15T10:00:00Z",
                },
            ]
        }

        with patch("news_bot.news_client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = MagicMock()

            result = fetch_articles_by_section(mock_config, per_section_limit=5)

        assert isinstance(result, dict)
        assert "World News" in result
        assert "US Tech" in result
        assert "US Industry" in result
        assert "Malaysia Tech" in result
        assert "Malaysia Industry" in result

    def test_fetch_articles_by_section_deduplicates_across_sections(self, mock_config):
        """Test that same URL appearing in multiple feeds is only in first section."""
        from news_bot.news_client import fetch_articles_by_section
        
        # All feeds return same article
        mock_response = {
            "meta": {"returned": 1, "found": 10},
            "data": [
                {
                    "uuid": "same-article",
                    "title": "Same Article",
                    "url": "https://example.com/same",
                    "published_at": "2024-01-15T10:00:00Z",
                },
            ]
        }

        with patch("news_bot.news_client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = MagicMock()

            result = fetch_articles_by_section(mock_config, per_section_limit=5)

        # Count total articles across all sections
        total = sum(len(arts) for arts in result.values())
        # Should only be 1 article total (deduplicated)
        assert total == 1
        # First section (World News) should have it
        assert len(result["World News"]) == 1

    def test_fetch_articles_by_section_handles_partial_failure(self, mock_config):
        """Test that partial feed failures still return sections."""
        from news_bot.news_client import fetch_articles_by_section
        import requests as req

        call_count = [0]
        
        def mock_get_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_response = MagicMock()
            if call_count[0] == 2:  # Second feed fails
                raise req.RequestException("Connection failed")
            else:
                mock_response.json.return_value = {
                    "meta": {"returned": 1, "found": 1},
                    "data": [
                        {
                            "uuid": f"article-{call_count[0]}",
                            "title": f"Article {call_count[0]}",
                            "url": f"https://example.com/{call_count[0]}",
                            "published_at": "2024-01-15T10:00:00Z",
                        },
                    ]
                }
                mock_response.raise_for_status = MagicMock()
                return mock_response

        with patch("news_bot.news_client.requests.get", side_effect=mock_get_side_effect):
            result = fetch_articles_by_section(mock_config, per_section_limit=5)

        # Should have all 5 sections, but one is empty
        assert len(result) == 5
        # US Tech (second feed) should be empty
        assert len(result["US Tech"]) == 0
        # Others should have content
        assert len(result["World News"]) == 1
