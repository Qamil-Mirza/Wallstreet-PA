"""Tests for article content extraction."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from news_bot.article_extractor import (
    BLOCKED_CONTENT_MARKER,
    _extract_text_from_html,
    _fetch_and_extract,
    ensure_article_content,
    ensure_batch_content,
    is_blocked_content,
)
from news_bot.news_client import ArticleMeta


def make_article(
    article_id: str = "test-id",
    title: str = "Test Article",
    content: str = None,
    summary: str = None,
) -> ArticleMeta:
    """Helper to create test articles."""
    return ArticleMeta(
        id=article_id,
        title=title,
        url=f"https://example.com/{article_id}",
        summary=summary,
        content=content,
        published_at=datetime.now(),
        source="Test Source",
    )


class TestExtractTextFromHtml:
    """Tests for _extract_text_from_html function."""

    def test_extract_paragraphs(self):
        """Test extracting text from paragraph tags."""
        html = """
        <html>
        <body>
            <article>
                <p>This is the first paragraph with enough content to pass the filter.</p>
                <p>This is the second paragraph with more meaningful text.</p>
            </article>
        </body>
        </html>
        """

        result = _extract_text_from_html(html)

        assert "first paragraph" in result
        assert "second paragraph" in result

    def test_extract_removes_scripts(self):
        """Test that script tags are removed."""
        html = """
        <html>
        <body>
            <script>alert('bad');</script>
            <p>This is legitimate content that should be extracted properly.</p>
        </body>
        </html>
        """

        result = _extract_text_from_html(html)

        assert "alert" not in result
        assert "legitimate content" in result

    def test_extract_removes_nav_and_footer(self):
        """Test that nav and footer elements are removed."""
        html = """
        <html>
        <body>
            <nav><a href="/">Home</a></nav>
            <article>
                <p>Main article content that we want to extract and keep.</p>
            </article>
            <footer>Copyright 2024</footer>
        </body>
        </html>
        """

        result = _extract_text_from_html(html)

        assert "Home" not in result
        assert "Copyright" not in result
        assert "Main article content" in result

    def test_extract_prefers_article_tag(self):
        """Test that article tag is preferred for content."""
        html = """
        <html>
        <body>
            <div>Sidebar content that should be ignored by extractor.</div>
            <article>
                <p>This is the real article content we want to capture.</p>
            </article>
        </body>
        </html>
        """

        result = _extract_text_from_html(html)

        assert "real article content" in result

    def test_extract_filters_short_paragraphs(self):
        """Test that very short paragraphs are filtered out."""
        html = """
        <html>
        <body>
            <p>Short</p>
            <p>This is a much longer paragraph that should pass the length filter.</p>
        </body>
        </html>
        """

        result = _extract_text_from_html(html)

        # Short paragraph should be filtered
        assert result.strip() != "Short"
        assert "longer paragraph" in result


class TestFetchAndExtract:
    """Tests for _fetch_and_extract function."""

    def test_fetch_and_extract_success(self):
        """Test successful fetch and extraction."""
        html = """
        <html>
        <body>
            <article>
                <p>This is article content from the fetched webpage that we need.</p>
            </article>
        </body>
        </html>
        """

        with patch("news_bot.article_extractor.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            content, is_blocked = _fetch_and_extract("https://example.com/article")

        assert content is not None
        assert "article content" in content
        assert is_blocked is False

    def test_fetch_and_extract_non_html(self):
        """Test handling of non-HTML content types."""
        with patch("news_bot.article_extractor.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            content, is_blocked = _fetch_and_extract("https://example.com/api")

        assert content is None
        assert is_blocked is False

    def test_fetch_and_extract_request_error(self):
        """Test handling of request errors."""
        import requests as req

        with patch("news_bot.article_extractor.requests.get") as mock_get:
            mock_get.side_effect = req.RequestException("Connection failed")

            content, is_blocked = _fetch_and_extract("https://example.com/article")

        assert content is None
        assert is_blocked is False


class TestEnsureArticleContent:
    """Tests for ensure_article_content function."""

    def test_ensure_article_content_uses_existing_content(self):
        """Test that existing content is preserved without HTTP call."""
        # Content must be >100 chars to pass threshold
        long_content = "This is existing content that is definitely long enough to pass the 100 character threshold for the article extractor module test."
        article = make_article(content=long_content)

        with patch("news_bot.article_extractor.requests.get") as mock_get:
            result = ensure_article_content(article)

        # No HTTP call should be made
        mock_get.assert_not_called()

        # Content should be preserved
        assert result.content == article.content

    def test_ensure_article_content_fetches_when_empty(self):
        """Test that URL is fetched when content is empty."""
        article = make_article(content=None)

        # Paragraphs must be >30 chars each, and total extracted must be >100 chars
        html = """
        <html><body>
            <p>This is the first paragraph with extracted content from the webpage that has enough length to pass the filter.</p>
            <p>This is the second paragraph with more content to ensure we pass the 100 character minimum threshold.</p>
        </body></html>
        """

        with patch("news_bot.article_extractor.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = ensure_article_content(article)

        mock_get.assert_called_once()
        assert "first paragraph" in result.content

    def test_ensure_article_content_fetches_when_too_short(self):
        """Test that URL is fetched when content is too short."""
        article = make_article(content="Short")

        # Paragraphs must be >30 chars each, and total extracted must be >100 chars
        html = """
        <html><body>
            <p>This is longer extracted content that passes the minimum thirty character paragraph filter requirement.</p>
            <p>Additional paragraph content to ensure we exceed the one hundred character total minimum threshold.</p>
        </body></html>
        """

        with patch("news_bot.article_extractor.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = html
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = ensure_article_content(article)

        mock_get.assert_called_once()
        assert "longer extracted content" in result.content

    def test_ensure_article_content_uses_summary_fallback(self):
        """Test that summary is used as fallback when extraction fails."""
        article = make_article(
            content=None,
            summary="This is a summary that is long enough to serve as content fallback."
        )

        with patch("news_bot.article_extractor.requests.get") as mock_get:
            import requests as req
            mock_get.side_effect = req.RequestException("Failed")

            result = ensure_article_content(article)

        assert result.content == article.summary

    def test_ensure_article_content_preserves_other_fields(self):
        """Test that other article fields are preserved."""
        # Content must be >100 chars to pass threshold
        long_content = "Existing content that is definitely long enough to pass the one hundred character minimum threshold test for the article."
        article = make_article(
            article_id="test-123",
            title="Test Title",
            content=long_content
        )

        result = ensure_article_content(article)

        assert result.id == article.id
        assert result.title == article.title
        assert result.url == article.url
        assert result.published_at == article.published_at
        assert result.source == article.source


class TestEnsureBatchContent:
    """Tests for ensure_batch_content function."""

    def test_ensure_batch_content_processes_all(self):
        """Test that all articles in batch are processed."""
        # Content must be >100 chars to pass threshold
        articles = [
            make_article(
                "1",
                content="Content one that is sufficiently long for the test. Adding more text to ensure we pass the 100 character threshold."
            ),
            make_article(
                "2",
                content="Content two that is also long enough for testing. Adding more text here as well to exceed the minimum threshold."
            ),
        ]

        result = ensure_batch_content(articles)

        assert len(result) == 2
        assert result[0].id == "1"
        assert result[1].id == "2"

    def test_ensure_batch_content_handles_errors(self):
        """Test that errors don't block other articles."""
        # Content must be >100 chars to pass threshold
        articles = [
            make_article("1", content=None),
            make_article(
                "2",
                content="Valid content that is long enough to pass threshold. Adding more text to ensure we exceed the one hundred character minimum."
            ),
        ]

        with patch("news_bot.article_extractor.ensure_article_content") as mock_ensure:
            def side_effect(article):
                if article.id == "1":
                    raise Exception("Processing failed")
                return article

            mock_ensure.side_effect = side_effect

            result = ensure_batch_content(articles)

        # Should still have both articles
        assert len(result) == 2

    def test_ensure_batch_content_empty_list(self):
        """Test handling of empty list."""
        result = ensure_batch_content([])
        assert result == []


class TestBlockDetection:
    """Tests for blocked/paywalled content detection."""

    def test_is_blocked_content_javascript_required(self):
        """Test detection of JavaScript requirement message."""
        text = "Please enable Javascript and cookies to continue. If you have an ad-blocker enabled you may be blocked from proceeding."
        
        assert is_blocked_content(text) is True

    def test_is_blocked_content_ad_blocker(self):
        """Test detection of ad-blocker message."""
        text = "We've detected that you have an ad-blocker enabled. Please disable your ad blocker to access this content."
        
        assert is_blocked_content(text) is True

    def test_is_blocked_content_paywall(self):
        """Test detection of paywall message."""
        text = "Subscribe to continue reading. This is premium content available to subscribers only."
        
        assert is_blocked_content(text) is True

    def test_is_blocked_content_captcha(self):
        """Test detection of CAPTCHA/bot check."""
        text = "Please verify you are human by completing the CAPTCHA below."
        
        assert is_blocked_content(text) is True

    def test_is_blocked_content_access_denied(self):
        """Test detection of access denied message."""
        text = "Access to this page has been denied. Please contact support if you believe this is an error."
        
        assert is_blocked_content(text) is True

    def test_is_blocked_content_legitimate_article(self):
        """Test that legitimate article content is not flagged as blocked."""
        text = """
        Apple Inc. reported quarterly earnings that beat analyst expectations.
        Revenue reached $89.5 billion, up 8% from the same quarter last year.
        The company's services segment continued to show strong growth, with
        revenue reaching an all-time high. CEO Tim Cook said the company is
        optimistic about the upcoming product cycle. Analysts expect continued
        growth in the wearables and services segments.
        """
        
        assert is_blocked_content(text) is False

    def test_is_blocked_content_empty(self):
        """Test that empty content is not flagged as blocked."""
        assert is_blocked_content("") is False
        assert is_blocked_content(None) is False

    def test_is_blocked_content_short_with_indicators(self):
        """Test detection of short content with multiple block indicators."""
        text = "Please sign in or subscribe to continue"
        
        assert is_blocked_content(text) is True

    def test_fetch_and_extract_returns_blocked_flag(self):
        """Test that _fetch_and_extract returns blocked flag for block pages."""
        block_html = """
        <html><body>
            <p>Please enable Javascript and cookies to continue browsing.</p>
            <p>If you have an ad-blocker enabled you may be blocked from proceeding.</p>
        </body></html>
        """

        with patch("news_bot.article_extractor.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = block_html
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            content, is_blocked = _fetch_and_extract("https://example.com/article")

        assert is_blocked is True
        assert content is None

    def test_ensure_article_content_marks_blocked(self):
        """Test that blocked content is marked with BLOCKED_CONTENT_MARKER."""
        article = make_article(content=None)

        block_html = """
        <html><body>
            <p>Please enable Javascript to view this page. Cookies are required.</p>
        </body></html>
        """

        with patch("news_bot.article_extractor.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = block_html
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = ensure_article_content(article)

        assert result.content == BLOCKED_CONTENT_MARKER

    def test_ensure_article_content_checks_existing_content(self):
        """Test that existing blocked content is detected and marked."""
        # Content must be > 100 chars to be checked without fetching
        blocked_content = (
            "Please subscribe to continue reading this premium content. "
            "This article is available to subscribers only. Sign in to access "
            "the full story and exclusive member benefits."
        )
        article = make_article(content=blocked_content)

        result = ensure_article_content(article)

        assert result.content == BLOCKED_CONTENT_MARKER
