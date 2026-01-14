"""Tests for email client module."""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from news_bot.classifier import ArticleCategory
from news_bot.config import Config
from news_bot.email_client import (
    EmailError,
    _format_summary_html,
    build_email_html,
    send_email,
)
from news_bot.news_client import ArticleMeta


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    return Config(
        news_api_key="test_key",
        news_api_base_url="https://api.test.com",
        smtp_host="smtp.test.com",
        smtp_port=587,
        smtp_user="sender@test.com",
        smtp_password="secret123",
        recipient_email="recipient@test.com",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3",
    )


def make_article(
    article_id: str,
    title: str,
    source: str = "Test Source",
) -> ArticleMeta:
    """Helper to create test articles."""
    return ArticleMeta(
        id=article_id,
        title=title,
        url=f"https://example.com/{article_id}",
        summary="Test summary",
        content="Test content",
        published_at=datetime(2024, 1, 15, 10, 30),
        source=source,
    )


class TestFormatSummaryHtml:
    """Tests for _format_summary_html function."""

    def test_format_narrative_text(self):
        """Test formatting narrative text to HTML paragraph."""
        summary = "The Fed raised rates by 25 basis points. Markets reacted positively to the news."

        result = _format_summary_html(summary)

        assert "<p>" in result
        assert "Fed raised rates" in result
        assert "Markets reacted" in result
        assert "</p>" in result

    def test_format_multiline_to_single_paragraph(self):
        """Test that multiline text is joined into flowing paragraph."""
        summary = "First sentence here.\nSecond sentence continues.\nThird wraps it up."

        result = _format_summary_html(summary)

        assert "<p>" in result
        # Should be joined with spaces
        assert "First sentence here. Second sentence continues. Third wraps it up." in result

    def test_format_strips_stray_bullets(self):
        """Test that stray bullet characters are removed."""
        summary = "• This starts with a bullet but shouldn't"

        result = _format_summary_html(summary)

        assert "<p>" in result
        assert "•" not in result
        assert "This starts with" in result

    def test_format_handles_empty_lines(self):
        """Test that empty lines are handled gracefully."""
        summary = "First part.\n\nSecond part.\n\n"

        result = _format_summary_html(summary)

        assert "<p>" in result
        assert "First part. Second part." in result

    def test_format_plain_text(self):
        """Test formatting plain text."""
        summary = "Just regular text without any formatting"

        result = _format_summary_html(summary)

        assert "<p>Just regular text without any formatting</p>" in result


class TestBuildEmailHtml:
    """Tests for build_email_html function."""

    def test_build_email_html_contains_all_articles(self):
        """Test that HTML contains all article information."""
        articles = [
            make_article("1", "Fed Raises Rates", "Reuters"),
            make_article("2", "Big Merger Announced", "Bloomberg"),
            make_article("3", "Tech Trends 2024", "WSJ"),
        ]

        summaries = {
            "1": "• Rate hike details\n• Market impact",
            "2": "• Deal value $10B\n• Expected close Q2",
            "3": "• AI adoption rising\n• Cloud growth",
        }

        buckets: dict[ArticleCategory, list[ArticleMeta]] = {
            "macro": [articles[0]],
            "deal": [articles[1]],
            "feature": [articles[2]],
        }

        result = build_email_html(
            date(2024, 1, 15),
            articles,
            summaries,
            buckets,
        )

        # Check date
        assert "January 15, 2024" in result

        # Check all headlines present
        assert "Fed Raises Rates" in result
        assert "Big Merger Announced" in result
        assert "Tech Trends 2024" in result

        # Check sources
        assert "Reuters" in result
        assert "Bloomberg" in result
        assert "WSJ" in result

        # Check summary content
        assert "Rate hike details" in result
        assert "Deal value $10B" in result
        assert "AI adoption rising" in result

        # Check URLs
        assert "https://example.com/1" in result
        assert "https://example.com/2" in result
        assert "https://example.com/3" in result

    def test_build_email_html_contains_category_labels(self):
        """Test that category labels are present."""
        articles = [make_article("1", "Fed News")]
        summaries = {"1": "• Summary"}
        buckets: dict[ArticleCategory, list[ArticleMeta]] = {
            "macro": [articles[0]],
            "deal": [],
            "feature": [],
        }

        result = build_email_html(date(2024, 1, 15), articles, summaries, buckets)

        assert "Macro" in result

    def test_build_email_html_valid_structure(self):
        """Test that HTML has valid structure."""
        articles = [make_article("1", "Test Article")]
        summaries = {"1": "• Point"}

        result = build_email_html(date(2024, 1, 15), articles, summaries)

        assert "<!DOCTYPE html>" in result
        assert "<html>" in result
        assert "</html>" in result
        assert "<head>" in result
        assert "<body>" in result
        assert "The Daily Briefing" in result

    def test_build_email_html_empty_articles(self):
        """Test building HTML with no articles."""
        result = build_email_html(date(2024, 1, 15), [], {})

        # Should still produce valid HTML
        assert "<!DOCTYPE html>" in result
        assert "The Daily Briefing" in result


class TestSendEmail:
    """Tests for send_email function."""

    def test_send_email_uses_smtp(self, mock_config):
        """Test that email is sent via SMTP correctly."""
        html_body = "<html><body>Test email</body></html>"

        with patch("news_bot.email_client.smtplib.SMTP") as mock_smtp_class:
            mock_smtp = MagicMock()
            mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

            send_email(mock_config, "Test Subject", html_body)

        # Check SMTP was created with correct host/port
        mock_smtp_class.assert_called_once_with("smtp.test.com", 587)

        # Check starttls was called
        mock_smtp.starttls.assert_called_once()

        # Check login was called with correct credentials
        mock_smtp.login.assert_called_once_with("sender@test.com", "secret123")

        # Check sendmail was called
        mock_smtp.sendmail.assert_called_once()
        call_args = mock_smtp.sendmail.call_args[0]
        assert call_args[0] == "sender@test.com"  # From
        assert call_args[1] == "recipient@test.com"  # To

    def test_send_email_auth_failure(self, mock_config):
        """Test handling of SMTP authentication failure."""
        import smtplib

        with patch("news_bot.email_client.smtplib.SMTP") as mock_smtp_class:
            mock_smtp = MagicMock()
            mock_smtp.login.side_effect = smtplib.SMTPAuthenticationError(
                535, "Authentication failed"
            )
            mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(EmailError) as exc_info:
                send_email(mock_config, "Subject", "<html></html>")

            assert "authentication" in str(exc_info.value).lower()

    def test_send_email_smtp_error(self, mock_config):
        """Test handling of general SMTP errors."""
        import smtplib

        with patch("news_bot.email_client.smtplib.SMTP") as mock_smtp_class:
            mock_smtp = MagicMock()
            mock_smtp.sendmail.side_effect = smtplib.SMTPException("Send failed")
            mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(EmailError) as exc_info:
                send_email(mock_config, "Subject", "<html></html>")

            assert "Failed to send" in str(exc_info.value)

    def test_send_email_connection_error(self, mock_config):
        """Test handling of connection errors."""
        with patch("news_bot.email_client.smtplib.SMTP") as mock_smtp_class:
            mock_smtp_class.side_effect = ConnectionRefusedError()

            with pytest.raises(EmailError):
                send_email(mock_config, "Subject", "<html></html>")
