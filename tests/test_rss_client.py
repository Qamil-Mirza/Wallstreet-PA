"""Tests for RSS client module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from news_bot.rss_client import (
    RSSClientError,
    RSSFeedConfig,
    _clean_html,
    _detect_feed_type,
    _generate_article_id,
    _parse_rss_date,
    fetch_multiple_feeds,
    fetch_rss_feed,
    parse_feed,
)


# =============================================================================
# Sample RSS/Atom Feed Content for Testing
# =============================================================================

SAMPLE_RSS_2_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:content="http://purl.org/rss/1.0/modules/content/">
    <channel>
        <title>Test Finance News</title>
        <link>https://example.com</link>
        <description>Latest finance news</description>
        <item>
            <title>Fed Raises Interest Rates by 25bps</title>
            <link>https://example.com/fed-rates</link>
            <description>The Federal Reserve raised rates today in response to inflation concerns.</description>
            <content:encoded><![CDATA[<p>The Federal Reserve raised interest rates by 25 basis points today, citing persistent inflation concerns. This marks the 10th consecutive rate hike.</p>]]></content:encoded>
            <pubDate>Mon, 15 Jan 2024 10:30:00 GMT</pubDate>
            <dc:creator>John Smith</dc:creator>
            <guid>https://example.com/fed-rates</guid>
        </item>
        <item>
            <title>Apple Reports Record Q4 Earnings</title>
            <link>https://example.com/apple-earnings</link>
            <description>Apple Inc reported record earnings for Q4 2023.</description>
            <pubDate>Tue, 16 Jan 2024 14:00:00 GMT</pubDate>
            <guid>apple-q4-2023</guid>
        </item>
    </channel>
</rss>"""

SAMPLE_ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
    <title>Tech News Atom Feed</title>
    <link href="https://tech.example.com" />
    <updated>2024-01-15T10:30:00Z</updated>
    <entry>
        <title>NVIDIA Unveils New AI Chip</title>
        <link href="https://tech.example.com/nvidia-ai-chip" />
        <id>urn:uuid:nvidia-ai-2024</id>
        <published>2024-01-15T09:00:00Z</published>
        <updated>2024-01-15T09:00:00Z</updated>
        <summary>NVIDIA announced their next-generation AI accelerator.</summary>
        <content type="html"><![CDATA[<p>NVIDIA unveiled the H200 chip with 2x performance over H100.</p>]]></content>
        <author>
            <name>Tech Reporter</name>
        </author>
    </entry>
    <entry>
        <title>Microsoft Cloud Revenue Surges</title>
        <link href="https://tech.example.com/msft-cloud" />
        <id>urn:uuid:msft-cloud-2024</id>
        <updated>2024-01-14T15:00:00Z</updated>
        <summary>Azure revenue grew 29% year-over-year.</summary>
    </entry>
</feed>"""

SAMPLE_RSS_MINIMAL = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Minimal Feed</title>
        <item>
            <title>Article Title</title>
            <link>https://example.com/article</link>
        </item>
    </channel>
</rss>"""

SAMPLE_RSS_EMPTY = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Empty Feed</title>
    </channel>
</rss>"""


# =============================================================================
# Test Date Parsing
# =============================================================================

class TestParsRssDate:
    """Tests for RSS date parsing."""

    def test_parse_rfc822_format(self):
        """Test parsing RFC 822 date format (common in RSS 2.0)."""
        result = _parse_rss_date("Mon, 15 Jan 2024 10:30:00 GMT")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_parse_rfc822_with_timezone(self):
        """Test parsing RFC 822 with timezone offset."""
        result = _parse_rss_date("Mon, 15 Jan 2024 10:30:00 +0000")
        assert result.year == 2024
        assert result.month == 1

    def test_parse_iso8601_with_z(self):
        """Test parsing ISO 8601 format with Z suffix."""
        result = _parse_rss_date("2024-01-15T10:30:00Z")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_iso8601_with_milliseconds(self):
        """Test parsing ISO 8601 with milliseconds."""
        result = _parse_rss_date("2024-01-15T10:30:00.123Z")
        assert result.year == 2024
        assert result.month == 1

    def test_parse_date_only(self):
        """Test parsing date-only format."""
        result = _parse_rss_date("2024-01-15")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_invalid_returns_now(self):
        """Test that invalid format returns current time."""
        result = _parse_rss_date("not-a-date")
        assert abs((datetime.now() - result).total_seconds()) < 5

    def test_parse_empty_returns_now(self):
        """Test that empty string returns current time."""
        result = _parse_rss_date("")
        assert abs((datetime.now() - result).total_seconds()) < 5

    def test_parse_none_returns_now(self):
        """Test that None returns current time."""
        result = _parse_rss_date(None)
        assert abs((datetime.now() - result).total_seconds()) < 5


# =============================================================================
# Test HTML Cleaning
# =============================================================================

class TestCleanHtml:
    """Tests for HTML content cleaning."""

    def test_clean_simple_tags(self):
        """Test removing simple HTML tags."""
        result = _clean_html("<p>Hello <b>World</b></p>")
        assert result == "Hello World"

    def test_clean_cdata(self):
        """Test removing CDATA markers."""
        result = _clean_html("<![CDATA[Some content]]>")
        assert result == "Some content"

    def test_decode_html_entities(self):
        """Test decoding common HTML entities."""
        result = _clean_html("&amp; &lt; &gt; &quot;")
        assert result == "& < > \""

    def test_clean_empty(self):
        """Test cleaning empty string."""
        assert _clean_html("") == ""
        assert _clean_html(None) == ""

    def test_clean_whitespace_normalization(self):
        """Test that excessive whitespace is normalized."""
        result = _clean_html("Hello    World\n\nTest")
        assert "  " not in result  # No double spaces


# =============================================================================
# Test Article ID Generation
# =============================================================================

class TestGenerateArticleId:
    """Tests for article ID generation."""

    def test_generate_id_from_url(self):
        """Test that ID is generated from URL hash."""
        url = "https://example.com/article/12345"
        result = _generate_article_id(url)
        assert len(result) == 16
        assert result.isalnum()

    def test_same_url_same_id(self):
        """Test that same URL generates same ID."""
        url = "https://example.com/article"
        id1 = _generate_article_id(url)
        id2 = _generate_article_id(url)
        assert id1 == id2

    def test_different_urls_different_ids(self):
        """Test that different URLs generate different IDs."""
        id1 = _generate_article_id("https://example.com/article1")
        id2 = _generate_article_id("https://example.com/article2")
        assert id1 != id2


# =============================================================================
# Test Feed Type Detection
# =============================================================================

class TestDetectFeedType:
    """Tests for feed type detection."""

    def test_detect_rss2(self):
        """Test detecting RSS 2.0 feed."""
        from xml.etree import ElementTree
        root = ElementTree.fromstring(SAMPLE_RSS_2_FEED)
        assert _detect_feed_type(root) == "rss2"

    def test_detect_atom(self):
        """Test detecting Atom feed."""
        from xml.etree import ElementTree
        root = ElementTree.fromstring(SAMPLE_ATOM_FEED)
        assert _detect_feed_type(root) == "atom"


# =============================================================================
# Test RSS 2.0 Feed Parsing
# =============================================================================

class TestParseRss2Feed:
    """Tests for RSS 2.0 feed parsing."""

    def test_parse_rss2_feed(self):
        """Test parsing a complete RSS 2.0 feed."""
        articles = parse_feed(SAMPLE_RSS_2_FEED, "Test Feed")
        
        assert len(articles) == 2
        
        # First article
        assert articles[0].title == "Fed Raises Interest Rates by 25bps"
        assert articles[0].url == "https://example.com/fed-rates"
        assert "Federal Reserve" in articles[0].summary
        assert articles[0].source == "John Smith"
        assert articles[0].published_at.year == 2024
        
        # Second article (no author)
        assert articles[1].title == "Apple Reports Record Q4 Earnings"
        assert articles[1].source == "Test Finance News"  # Falls back to feed title

    def test_parse_rss2_extracts_content_encoded(self):
        """Test that content:encoded is extracted for full content."""
        articles = parse_feed(SAMPLE_RSS_2_FEED, "Test Feed")
        
        # First article has content:encoded
        assert "10th consecutive rate hike" in articles[0].content

    def test_parse_minimal_rss(self):
        """Test parsing RSS with minimal fields."""
        articles = parse_feed(SAMPLE_RSS_MINIMAL, "Minimal")
        
        assert len(articles) == 1
        assert articles[0].title == "Article Title"
        assert articles[0].url == "https://example.com/article"
        assert articles[0].source == "Minimal Feed"

    def test_parse_empty_rss(self):
        """Test parsing RSS with no items."""
        articles = parse_feed(SAMPLE_RSS_EMPTY, "Empty")
        assert len(articles) == 0


# =============================================================================
# Test Atom Feed Parsing
# =============================================================================

class TestParseAtomFeed:
    """Tests for Atom feed parsing."""

    def test_parse_atom_feed(self):
        """Test parsing a complete Atom feed."""
        articles = parse_feed(SAMPLE_ATOM_FEED, "Tech Feed")
        
        assert len(articles) == 2
        
        # First entry
        assert articles[0].title == "NVIDIA Unveils New AI Chip"
        assert articles[0].url == "https://tech.example.com/nvidia-ai-chip"
        assert "next-generation AI accelerator" in articles[0].summary
        assert articles[0].source == "Tech Reporter"
        
        # Second entry (no author)
        assert articles[1].title == "Microsoft Cloud Revenue Surges"
        assert articles[1].source == "Tech News Atom Feed"  # Falls back to feed title

    def test_parse_atom_uses_published_date(self):
        """Test that Atom parsing prefers published over updated."""
        articles = parse_feed(SAMPLE_ATOM_FEED, "Tech Feed")
        
        # First entry has both published and updated
        assert articles[0].published_at.hour == 9  # From <published>


# =============================================================================
# Test RSS Feed Fetching
# =============================================================================

class TestFetchRssFeed:
    """Tests for RSS feed fetching."""

    def test_fetch_rss_feed_success(self):
        """Test successful RSS feed fetch."""
        with patch("news_bot.rss_client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = SAMPLE_RSS_2_FEED
            mock_response.headers = {"Content-Type": "application/rss+xml"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            articles = fetch_rss_feed(
                url="https://example.com/feed.xml",
                feed_name="Test Feed",
                limit=10,
            )
            
            assert len(articles) == 2
            # Articles are sorted by date (newest first), so Apple (Jan 16) comes before Fed (Jan 15)
            assert articles[0].title == "Apple Reports Record Q4 Earnings"
            assert articles[1].title == "Fed Raises Interest Rates by 25bps"

    def test_fetch_rss_feed_respects_limit(self):
        """Test that limit parameter is respected."""
        with patch("news_bot.rss_client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = SAMPLE_RSS_2_FEED
            mock_response.headers = {"Content-Type": "application/rss+xml"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            articles = fetch_rss_feed(
                url="https://example.com/feed.xml",
                feed_name="Test Feed",
                limit=1,
            )
            
            assert len(articles) == 1

    def test_fetch_rss_feed_timeout(self):
        """Test handling of request timeout."""
        import requests
        
        with patch("news_bot.rss_client.requests.get") as mock_get:
            mock_get.side_effect = requests.Timeout("Connection timed out")
            
            with pytest.raises(RSSClientError) as exc_info:
                fetch_rss_feed("https://example.com/feed.xml", "Test")
            
            assert "timed out" in str(exc_info.value)

    def test_fetch_rss_feed_connection_error(self):
        """Test handling of connection errors."""
        import requests
        
        with patch("news_bot.rss_client.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("Failed to connect")
            
            with pytest.raises(RSSClientError) as exc_info:
                fetch_rss_feed("https://example.com/feed.xml", "Test")
            
            assert "Failed to fetch RSS feed" in str(exc_info.value)

    def test_fetch_rss_feed_invalid_xml(self):
        """Test handling of invalid XML response."""
        with patch("news_bot.rss_client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = "This is not XML"
            mock_response.headers = {"Content-Type": "text/plain"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            with pytest.raises(RSSClientError) as exc_info:
                fetch_rss_feed("https://example.com/feed.xml", "Test")
            
            assert "Failed to parse RSS feed XML" in str(exc_info.value)


# =============================================================================
# Test Multiple Feed Fetching
# =============================================================================

class TestFetchMultipleFeeds:
    """Tests for fetching multiple RSS feeds."""

    def test_fetch_multiple_feeds_success(self):
        """Test fetching multiple feeds successfully."""
        feeds = [
            RSSFeedConfig(name="Feed 1", url="https://example1.com/feed.xml", section="News"),
            RSSFeedConfig(name="Feed 2", url="https://example2.com/feed.xml", section="News"),
        ]
        
        with patch("news_bot.rss_client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = SAMPLE_RSS_2_FEED
            mock_response.headers = {"Content-Type": "application/rss+xml"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            sections = fetch_multiple_feeds(feeds, deduplicate=False)
            
            assert "News" in sections
            # 2 articles from each feed = 4 total (dedup disabled)
            assert len(sections["News"]) == 4

    def test_fetch_multiple_feeds_deduplication(self):
        """Test that duplicate URLs are removed across feeds."""
        feeds = [
            RSSFeedConfig(name="Feed 1", url="https://example1.com/feed.xml", section="News"),
            RSSFeedConfig(name="Feed 2", url="https://example2.com/feed.xml", section="News"),
        ]
        
        with patch("news_bot.rss_client.requests.get") as mock_get:
            # Both feeds return same content
            mock_response = MagicMock()
            mock_response.text = SAMPLE_RSS_2_FEED
            mock_response.headers = {"Content-Type": "application/rss+xml"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            sections = fetch_multiple_feeds(feeds, deduplicate=True)
            
            # Should only have 2 unique articles (duplicates removed)
            assert len(sections["News"]) == 2

    def test_fetch_multiple_feeds_different_sections(self):
        """Test that feeds are organized by section."""
        feeds = [
            RSSFeedConfig(name="Finance Feed", url="https://finance.com/feed.xml", section="Finance"),
            RSSFeedConfig(name="Tech Feed", url="https://tech.com/feed.xml", section="Tech"),
        ]
        
        with patch("news_bot.rss_client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = SAMPLE_RSS_2_FEED
            mock_response.headers = {"Content-Type": "application/rss+xml"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            # Disable deduplication so each section gets its own articles
            sections = fetch_multiple_feeds(feeds, deduplicate=False)
            
            assert "Finance" in sections
            assert "Tech" in sections
            assert len(sections["Finance"]) == 2
            assert len(sections["Tech"]) == 2

    def test_fetch_multiple_feeds_partial_failure(self):
        """Test that partial failures don't stop other feeds."""
        import requests
        
        feeds = [
            RSSFeedConfig(name="Good Feed", url="https://good.com/feed.xml", section="News"),
            RSSFeedConfig(name="Bad Feed", url="https://bad.com/feed.xml", section="News"),
        ]
        
        call_count = [0]
        
        def mock_get_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First feed succeeds
                mock_response = MagicMock()
                mock_response.text = SAMPLE_RSS_2_FEED
                mock_response.headers = {"Content-Type": "application/rss+xml"}
                mock_response.raise_for_status = MagicMock()
                return mock_response
            else:
                # Second feed fails
                raise requests.ConnectionError("Failed")
        
        with patch("news_bot.rss_client.requests.get", side_effect=mock_get_side_effect):
            sections = fetch_multiple_feeds(feeds)
            
            # Should still have results from the first feed
            assert "News" in sections
            assert len(sections["News"]) == 2

    def test_fetch_multiple_feeds_disabled_feed(self):
        """Test that disabled feeds are skipped."""
        feeds = [
            RSSFeedConfig(name="Enabled Feed", url="https://enabled.com/feed.xml", enabled=True),
            RSSFeedConfig(name="Disabled Feed", url="https://disabled.com/feed.xml", enabled=False),
        ]
        
        with patch("news_bot.rss_client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = SAMPLE_RSS_2_FEED
            mock_response.headers = {"Content-Type": "application/rss+xml"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            fetch_multiple_feeds(feeds)
            
            # Should only be called once (for the enabled feed)
            assert mock_get.call_count == 1


# =============================================================================
# Test RSSFeedConfig
# =============================================================================

class TestRSSFeedConfig:
    """Tests for RSSFeedConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RSSFeedConfig(name="Test", url="https://example.com/feed.xml")
        
        assert config.section == "RSS Feeds"
        assert config.enabled is True
        assert config.limit == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RSSFeedConfig(
            name="Custom Feed",
            url="https://custom.com/feed.xml",
            section="Custom Section",
            enabled=False,
            limit=5,
        )
        
        assert config.name == "Custom Feed"
        assert config.section == "Custom Section"
        assert config.enabled is False
        assert config.limit == 5


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_parse_feed_missing_required_fields(self):
        """Test that articles missing title or link are skipped."""
        incomplete_feed = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Test</title>
                <item>
                    <title>Has Title Only</title>
                </item>
                <item>
                    <link>https://example.com/no-title</link>
                </item>
                <item>
                    <title>Complete Article</title>
                    <link>https://example.com/complete</link>
                </item>
            </channel>
        </rss>"""
        
        articles = parse_feed(incomplete_feed, "Test")
        
        # Only the complete article should be parsed
        assert len(articles) == 1
        assert articles[0].title == "Complete Article"

    def test_parse_feed_with_special_characters(self):
        """Test parsing feed with special characters in content."""
        # Note: XML only supports 5 predefined entities: &amp; &lt; &gt; &quot; &apos;
        # Other HTML entities like &mdash; are NOT valid XML and will cause parse errors
        # Real RSS feeds must use numeric entities or CDATA for special characters
        special_chars_feed = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Test</title>
                <item>
                    <title>Stock &amp; Bond Markets Rally</title>
                    <link>https://example.com/markets</link>
                    <description>The S&amp;P 500 rose 2% &#8212; best day in weeks.</description>
                </item>
            </channel>
        </rss>"""
        
        articles = parse_feed(special_chars_feed, "Test")
        
        assert len(articles) == 1
        assert "Stock & Bond" in articles[0].title
        assert "S&P 500" in articles[0].summary

    def test_articles_sorted_by_date(self):
        """Test that fetched articles are sorted by date (newest first)."""
        with patch("news_bot.rss_client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = SAMPLE_RSS_2_FEED
            mock_response.headers = {"Content-Type": "application/rss+xml"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            articles = fetch_rss_feed("https://example.com/feed.xml", "Test", limit=10)
            
            # Articles should be sorted by date descending
            for i in range(len(articles) - 1):
                assert articles[i].published_at >= articles[i + 1].published_at
