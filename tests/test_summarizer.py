"""Tests for article summarization with Ollama."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from news_bot.config import Config
from news_bot.news_client import ArticleMeta
from news_bot.summarizer import (
    SummarizerError,
    summarize_article,
    summarize_articles,
)


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    return Config(
        news_api_key="test_key",
        news_api_base_url="https://api.test.com",
        smtp_host="smtp.test.com",
        smtp_port=587,
        smtp_user="user@test.com",
        smtp_password="secret",
        recipient_email="recipient@test.com",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3",
    )


def make_article(article_id: str, title: str, content: str = "") -> ArticleMeta:
    """Helper to create test articles."""
    return ArticleMeta(
        id=article_id,
        title=title,
        url=f"https://example.com/{article_id}",
        summary=content[:100] if content else None,
        content=content or "Default content for testing.",
        published_at=datetime.now(),
        source="Test Source",
    )


class TestSummarizeArticle:
    """Tests for summarize_article function."""

    def test_summarize_article_calls_ollama(self):
        """Test that summarize_article calls Ollama API correctly."""
        mock_response = {
            "response": "• Key point 1\n• Key point 2\n• Key point 3"
        }

        with patch("news_bot.summarizer.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()

            result = summarize_article(
                content="Article content here",
                title="Test Article Title",
                model="llama3",
                base_url="http://localhost:11434",
            )

        assert "Key point 1" in result
        assert "Key point 2" in result

        # Verify the API was called
        mock_post.assert_called_once()
        call_args = mock_post.call_args

        # Check URL
        assert call_args[0][0] == "http://localhost:11434/api/generate"

        # Check payload
        payload = call_args[1]["json"]
        assert payload["model"] == "llama3"
        assert "Article content here" in payload["prompt"]
        assert payload["stream"] is False

    def test_summarize_article_timeout(self):
        """Test handling of Ollama timeout."""
        import requests as req

        with patch("news_bot.summarizer.requests.post") as mock_post:
            mock_post.side_effect = req.Timeout()

            with pytest.raises(SummarizerError) as exc_info:
                summarize_article(
                    content="Content",
                    title="Title",
                    model="llama3",
                    base_url="http://localhost:11434",
                )

            assert "timed out" in str(exc_info.value).lower()

    def test_summarize_article_connection_error(self):
        """Test handling of connection errors."""
        import requests as req

        with patch("news_bot.summarizer.requests.post") as mock_post:
            mock_post.side_effect = req.ConnectionError("Connection refused")

            with pytest.raises(SummarizerError) as exc_info:
                summarize_article(
                    content="Content",
                    title="Title",
                    model="llama3",
                    base_url="http://localhost:11434",
                )

            assert "Failed to connect" in str(exc_info.value)

    def test_summarize_article_empty_response(self):
        """Test handling of empty Ollama response."""
        mock_response = {"response": ""}

        with patch("news_bot.summarizer.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()

            with pytest.raises(SummarizerError) as exc_info:
                summarize_article(
                    content="Content",
                    title="Title",
                    model="llama3",
                    base_url="http://localhost:11434",
                )

            assert "empty response" in str(exc_info.value).lower()

    def test_summarize_article_strips_whitespace(self):
        """Test that response is stripped of extra whitespace."""
        mock_response = {"response": "  \n• Bullet point\n  "}

        with patch("news_bot.summarizer.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()

            result = summarize_article(
                content="Content",
                title="Title",
                model="llama3",
                base_url="http://localhost:11434",
            )

        assert result == "• Bullet point"


class TestSummarizeArticles:
    """Tests for summarize_articles function."""

    def test_summarize_articles_maps_ids(self, mock_config):
        """Test that summaries are mapped correctly to article IDs."""
        articles = [
            make_article("article-1", "First Article", "Content 1"),
            make_article("article-2", "Second Article", "Content 2"),
        ]

        responses = [
            {"response": "• Summary for article 1"},
            {"response": "• Summary for article 2"},
        ]

        with patch("news_bot.summarizer.requests.post") as mock_post:
            mock_post.return_value.json.side_effect = responses
            mock_post.return_value.raise_for_status = MagicMock()

            result = summarize_articles(articles, mock_config)

        assert len(result) == 2
        assert "article-1" in result
        assert "article-2" in result
        assert "Summary for article 1" in result["article-1"]
        assert "Summary for article 2" in result["article-2"]

    def test_summarize_articles_handles_failures(self, mock_config):
        """Test that failures for one article don't block others."""
        articles = [
            make_article("article-1", "First Article", "Content 1"),
            make_article("article-2", "Second Article", "Content 2"),
        ]

        call_count = 0

        def mock_post_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()

            if call_count == 1:
                # First call fails
                import requests as req
                raise req.Timeout()
            else:
                # Second call succeeds
                mock_response.json.return_value = {"response": "• Success"}
                mock_response.raise_for_status = MagicMock()
                return mock_response

        with patch("news_bot.summarizer.requests.post") as mock_post:
            mock_post.side_effect = mock_post_side_effect

            result = summarize_articles(articles, mock_config)

        assert len(result) == 2
        # First article should have fallback summary
        assert "article-1" in result
        assert "First Article" in result["article-1"]  # Title as fallback
        # Second should have actual summary
        assert "Success" in result["article-2"]

    def test_summarize_articles_empty_list(self, mock_config):
        """Test summarizing empty list returns empty dict."""
        result = summarize_articles([], mock_config)
        assert result == {}

    def test_summarize_articles_uses_config_values(self, mock_config):
        """Test that config model and base_url are used."""
        articles = [make_article("test", "Test Article")]

        with patch("news_bot.summarizer.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "• Summary"}
            mock_post.return_value.raise_for_status = MagicMock()

            summarize_articles(articles, mock_config)

        call_args = mock_post.call_args
        assert "http://localhost:11434/api/generate" in call_args[0][0]
        assert call_args[1]["json"]["model"] == "llama3"
