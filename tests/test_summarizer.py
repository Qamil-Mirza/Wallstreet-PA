"""Tests for article summarization with Ollama."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from news_bot.config import Config
from news_bot.news_client import ArticleMeta
from news_bot.article_extractor import BLOCKED_CONTENT_MARKER
from news_bot.summarizer import (
    BLOCKED_SUMMARY_FALLBACK,
    SummarizerError,
    is_article_blocked,
    summarize_article,
    summarize_articles,
    smart_chunk_content,
    _split_into_paragraphs,
    _score_paragraph,
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


class TestSmartChunking:
    """Tests for smart paragraph chunking logic."""

    def test_split_into_paragraphs_basic(self):
        """Test basic paragraph splitting."""
        text = """First paragraph with enough content to pass the minimum length threshold.

Second paragraph also with sufficient content to be included in the result.

Third paragraph meeting the minimum character requirement for inclusion."""
        
        paragraphs = _split_into_paragraphs(text)
        
        assert len(paragraphs) == 3
        assert "First paragraph" in paragraphs[0]
        assert "Second paragraph" in paragraphs[1]
        assert "Third paragraph" in paragraphs[2]

    def test_split_into_paragraphs_filters_short(self):
        """Test that very short paragraphs are filtered out."""
        text = """First paragraph with enough content to pass the minimum length threshold.

Too short.

Third paragraph also with sufficient content to be included in the result."""
        
        paragraphs = _split_into_paragraphs(text)
        
        assert len(paragraphs) == 2
        assert "Too short" not in str(paragraphs)

    def test_score_paragraph_financial_terms(self):
        """Test that financial terms increase score."""
        financial_para = "Revenue increased 25% to $1.2 billion, with EPS beating consensus estimates."
        generic_para = "The company announced plans to expand its operations into new markets."
        
        financial_score = _score_paragraph(financial_para, 2, 5, "")
        generic_score = _score_paragraph(generic_para, 2, 5, "")
        
        assert financial_score > generic_score

    def test_score_paragraph_position_bonus(self):
        """Test that first and last paragraphs get position bonus."""
        para = "A standard paragraph with some content for testing purposes here."
        
        first_score = _score_paragraph(para, 0, 5, "")  # First
        middle_score = _score_paragraph(para, 2, 5, "")  # Middle
        last_score = _score_paragraph(para, 4, 5, "")  # Last
        
        assert first_score > middle_score
        assert last_score > middle_score

    def test_score_paragraph_title_match(self):
        """Test that matching title words increases score."""
        para = "Apple announced new iPhone sales exceeded expectations this quarter."
        
        with_title_match = _score_paragraph(para, 2, 5, "Apple iPhone Revenue")
        without_title_match = _score_paragraph(para, 2, 5, "Microsoft Azure Cloud")
        
        assert with_title_match > without_title_match

    def test_score_paragraph_thesis_phrases(self):
        """Test that thesis/conclusion phrases increase score."""
        thesis_para = "Bottom line: We believe the outlook remains positive despite near-term risks."
        regular_para = "The company held its annual meeting last week in San Francisco."
        
        thesis_score = _score_paragraph(thesis_para, 2, 5, "")
        regular_score = _score_paragraph(regular_para, 2, 5, "")
        
        assert thesis_score > regular_score

    def test_smart_chunk_content_short_content(self):
        """Test that short content is returned as-is."""
        short_content = "This is short content under the budget limit."
        
        result = smart_chunk_content(short_content, char_budget=4000)
        
        assert result == short_content

    def test_smart_chunk_content_preserves_order(self):
        """Test that selected paragraphs maintain original order."""
        content = """First paragraph introduces the topic with enough detail to be meaningful.

Second paragraph provides additional context and background information here.

Third paragraph contains key financial data: revenue up 30% to $500 million.

Fourth paragraph discusses market implications and competitive positioning.

Fifth paragraph concludes with outlook: We believe growth will continue."""
        
        result = smart_chunk_content(content, title="Revenue Report", char_budget=500)
        
        # First paragraph should come before third (even if third scores higher)
        first_pos = result.find("First paragraph") if "First paragraph" in result else -1
        third_pos = result.find("Third paragraph") if "Third paragraph" in result else -1
        
        if first_pos >= 0 and third_pos >= 0:
            assert first_pos < third_pos

    def test_smart_chunk_content_includes_first_and_last(self):
        """Test that first and last paragraphs are always included when available."""
        content = """Opening paragraph that sets up the story with enough content here.

Middle paragraph one with some filler content that is not very important.

Middle paragraph two also not particularly noteworthy or relevant to include.

Middle paragraph three continues with generic content we might skip.

Closing paragraph that wraps up the article with conclusions and outlook."""
        
        result = smart_chunk_content(content, char_budget=400)
        
        # First and last should be present
        assert "Opening paragraph" in result
        assert "Closing paragraph" in result

    def test_smart_chunk_content_respects_budget(self):
        """Test that output respects character budget."""
        # Create content that's definitely over budget
        long_content = "\n\n".join([
            f"Paragraph {i} with enough content to be considered a real paragraph."
            for i in range(20)
        ])
        
        budget = 500
        result = smart_chunk_content(long_content, char_budget=budget)
        
        # Allow small overage for clean truncation
        assert len(result) <= budget + 50


class TestBlockedArticleHandling:
    """Tests for blocked article detection and handling in summarizer."""

    def test_is_article_blocked_true(self):
        """Test that articles with BLOCKED_CONTENT_MARKER are detected."""
        article = make_article("blocked-1", "Blocked Article", BLOCKED_CONTENT_MARKER)
        
        assert is_article_blocked(article) is True

    def test_is_article_blocked_false(self):
        """Test that normal articles are not flagged as blocked."""
        article = make_article("normal-1", "Normal Article", "This is valid content.")
        
        assert is_article_blocked(article) is False

    def test_summarize_articles_skips_blocked_no_llm_call(self, mock_config):
        """Test that blocked articles don't trigger LLM calls."""
        articles = [
            make_article("blocked-1", "Blocked Article", BLOCKED_CONTENT_MARKER),
            make_article("normal-1", "Normal Article", "Valid content for summarization."),
        ]

        with patch("news_bot.summarizer.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "• Summary"}
            mock_post.return_value.raise_for_status = MagicMock()

            result = summarize_articles(articles, mock_config)

        # Should only call LLM once (for the normal article)
        assert mock_post.call_count == 1
        
        # Blocked article gets fallback
        assert result["blocked-1"] == BLOCKED_SUMMARY_FALLBACK
        
        # Normal article gets actual summary
        assert "Summary" in result["normal-1"]

    def test_summarize_articles_all_blocked(self, mock_config):
        """Test handling when all articles are blocked."""
        articles = [
            make_article("blocked-1", "Blocked One", BLOCKED_CONTENT_MARKER),
            make_article("blocked-2", "Blocked Two", BLOCKED_CONTENT_MARKER),
        ]

        with patch("news_bot.summarizer.requests.post") as mock_post:
            result = summarize_articles(articles, mock_config)

        # No LLM calls should be made
        mock_post.assert_not_called()
        
        # Both should have fallback
        assert result["blocked-1"] == BLOCKED_SUMMARY_FALLBACK
        assert result["blocked-2"] == BLOCKED_SUMMARY_FALLBACK
