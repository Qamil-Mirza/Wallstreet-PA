"""Tests for article content extraction."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from news_bot.article_extractor import (
    _extract_text_from_html,
    _fetch_and_extract,
    ensure_article_content,
    ensure_batch_content,
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

            result = _fetch_and_extract("https://example.com/article")

        assert result is not None
        assert "article content" in result

    def test_fetch_and_extract_non_html(self):
        """Test handling of non-HTML content types."""
        with patch("news_bot.article_extractor.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = _fetch_and_extract("https://example.com/api")

        assert result is None

    def test_fetch_and_extract_request_error(self):
        """Test handling of request errors."""
        import requests as req

        with patch("news_bot.article_extractor.requests.get") as mock_get:
            mock_get.side_effect = req.RequestException("Connection failed")

            result = _fetch_and_extract("https://example.com/article")

        assert result is None


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
