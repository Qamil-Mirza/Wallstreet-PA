"""Tests for email client module."""

import os
from datetime import date, datetime
from email import message_from_string
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from news_bot.classifier import ArticleCategory
from news_bot.config import Config
from news_bot import email_client
from news_bot.email_client import (
    EmailError,
    SECTION_CONFIG,
    _format_summary_html,
    build_email_html,
    build_sectioned_email_html,
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
        """Test that double newlines create separate paragraphs."""
        summary = "First part.\n\nSecond part.\n\n"

        result = _format_summary_html(summary)

        # Double newlines should create separate paragraphs
        assert "<p>First part.</p>" in result
        assert "<p>Second part.</p>" in result

    def test_format_so_what_styling(self):
        """Test that 'So what?' gets special styling."""
        summary = "Company announced X.\n\nSo what? This matters because Y."

        result = _format_summary_html(summary)

        # "So what?" should be bolded
        assert "<strong>So what?</strong>" in result
        assert "This matters because Y." in result

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

    def test_send_email_attaches_pdf_and_audio_with_correct_mime_types(
        self, mock_config, tmp_path: Path
    ):
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-test")
        audio = tmp_path / "brief.mp3"
        audio.write_bytes(b"ID3-test")

        with patch("news_bot.email_client.smtplib.SMTP") as smtp_class:
            smtp = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp
            send_email(mock_config, "Research", "<p>Body</p>", attachment_paths=(pdf, audio))

        message = message_from_string(smtp.sendmail.call_args.args[2])
        assert message.get_content_subtype() == "mixed"
        assert [part.get_content_type() for part in message.walk()] == [
            "multipart/mixed", "text/html", "application/pdf", "audio/mpeg"
        ]

    def test_send_email_combines_legacy_attachment_and_deduplicates(
        self, mock_config, tmp_path: Path
    ):
        wav = tmp_path / "brief.wav"
        wav.write_bytes(b"RIFF\x04\x00\x00\x00WAVE")
        with patch("news_bot.email_client.smtplib.SMTP") as smtp_class:
            smtp = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp
            send_email(mock_config, "Research", "<p>Body</p>", wav, attachment_paths=(wav,))
        message = message_from_string(smtp.sendmail.call_args.args[2])
        assert [part.get_content_type() for part in message.walk()].count("audio/wav") == 1

    def test_send_email_skips_unknown_and_missing_attachments_but_sends(
        self, mock_config, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        unknown = tmp_path / "payload.bin"
        unknown.write_bytes(b"opaque")
        missing = tmp_path / "missing.pdf"
        with patch("news_bot.email_client.smtplib.SMTP") as smtp_class:
            smtp = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp
            with caplog.at_level("WARNING"):
                send_email(mock_config, "Research", "<p>Body</p>", attachment_paths=(unknown, missing))
        message = message_from_string(smtp.sendmail.call_args.args[2])
        assert message.get_content_subtype() == "alternative"
        assert "unsupported attachment" in caplog.text.lower()
        assert "missing or unreadable attachment" in caplog.text.lower()

    @pytest.mark.parametrize(
        ("filename", "payload"),
        (
            ("fake.pdf", b"not-a-pdf"),
            ("fake.wav", b"RIFFxxxx-not-wave"),
            ("fake.mp3", b"not-an-mp3"),
        ),
    )
    def test_send_email_rejects_invalid_attachment_signatures(
        self,
        mock_config,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        filename: str,
        payload: bytes,
    ):
        attachment = tmp_path / filename
        attachment.write_bytes(payload)

        with patch("news_bot.email_client.smtplib.SMTP") as smtp_class:
            smtp = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp
            with caplog.at_level("WARNING"):
                send_email(
                    mock_config,
                    "Research",
                    "<p>Body</p>",
                    attachment_paths=(attachment,),
                )

        message = message_from_string(smtp.sendmail.call_args.args[2])
        assert message.get_content_subtype() == "alternative"
        assert "invalid attachment" in caplog.text.lower()
        assert filename in caplog.text

    @pytest.mark.parametrize(
        ("payload", "expected_subtype"),
        (
            (b"\xff\xfb\x90\x64frame", "mixed"),
            (b"\xff\xe0\x00\x00invalid", "alternative"),
        ),
    )
    def test_send_email_accepts_only_valid_mpeg_frame_headers(
        self,
        mock_config,
        tmp_path: Path,
        payload: bytes,
        expected_subtype: str,
    ):
        attachment = tmp_path / "frame.mp3"
        attachment.write_bytes(payload)
        with patch("news_bot.email_client.smtplib.SMTP") as smtp_class:
            smtp = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp
            send_email(
                mock_config,
                "Research",
                "<p>Body</p>",
                attachment_paths=(attachment,),
            )

        message = message_from_string(smtp.sendmail.call_args.args[2])
        assert message.get_content_subtype() == expected_subtype

    def test_send_email_rejects_symlink_directory_and_fifo_without_path_leak(
        self,
        mock_config,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ):
        private_dir = tmp_path / "sensitive-parent"
        private_dir.mkdir()
        target = private_dir / "target.pdf"
        target.write_bytes(b"%PDF-secret")
        symlink = tmp_path / "linked.pdf"
        symlink.symlink_to(target)
        directory = tmp_path / "folder.pdf"
        directory.mkdir()
        fifo = tmp_path / "pipe.wav"
        os.mkfifo(fifo)
        real_read_bytes = Path.read_bytes

        def never_block_on_fifo(path: Path) -> bytes:
            if path == fifo:
                raise OSError("test prevented a blocking FIFO read")
            return real_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", never_block_on_fifo)

        with patch("news_bot.email_client.smtplib.SMTP") as smtp_class:
            smtp = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp
            with caplog.at_level("WARNING"):
                send_email(
                    mock_config,
                    "Research",
                    "<p>Body</p>",
                    attachment_paths=(symlink, directory, fifo),
                )

        message = message_from_string(smtp.sendmail.call_args.args[2])
        assert message.get_content_subtype() == "alternative"
        assert "linked.pdf" in caplog.text
        assert "folder.pdf" in caplog.text
        assert "pipe.wav" in caplog.text
        assert "sensitive-parent" not in caplog.text
        assert str(tmp_path) not in caplog.text

    def test_send_email_skips_per_file_oversize_without_reading_payload(
        self,
        mock_config,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        oversized = tmp_path / "oversized.pdf"
        with oversized.open("wb") as handle:
            handle.write(b"%PDF-")
            handle.seek(email_client.MAX_ATTACHMENT_BYTES)
            handle.write(b"x")

        with patch("news_bot.email_client.os.read", wraps=os.read) as read:
            with patch("news_bot.email_client.smtplib.SMTP") as smtp_class:
                smtp = MagicMock()
                smtp_class.return_value.__enter__.return_value = smtp
                with caplog.at_level("WARNING"):
                    send_email(
                        mock_config,
                        "Research",
                        "<p>Body</p>",
                        attachment_paths=(oversized,),
                    )

        assert read.call_count == 0
        assert "oversized.pdf" in caplog.text
        assert "oversized" in caplog.text.lower()

    def test_send_email_enforces_attachment_count_in_original_order(
        self,
        mock_config,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        attachments = []
        max_attachments = email_client.MAX_ATTACHMENTS
        for index in range(max_attachments + 2):
            attachment = tmp_path / f"report-{index:02d}.pdf"
            attachment.write_bytes(b"%PDF-valid")
            attachments.append(attachment)

        with patch("news_bot.email_client.smtplib.SMTP") as smtp_class:
            smtp = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp
            with caplog.at_level("WARNING"):
                send_email(
                    mock_config,
                    "Research",
                    "<p>Body</p>",
                    attachment_paths=attachments,
                )

        message = message_from_string(smtp.sendmail.call_args.args[2])
        parts = [part for part in message.walk() if part.get_content_type() == "application/pdf"]
        assert len(parts) == max_attachments
        assert [part.get_filename() for part in parts] == [
            path.name for path in attachments[:max_attachments]
        ]
        assert attachments[max_attachments].name in caplog.text
        assert attachments[max_attachments + 1].name in caplog.text

    def test_send_email_count_limit_caps_supported_files_even_when_invalid(
        self,
        mock_config,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        attachments = []
        for index in range(email_client.MAX_ATTACHMENTS):
            attachment = tmp_path / f"invalid-{index:02d}.pdf"
            attachment.write_bytes(b"invalid")
            attachments.append(attachment)
        final_valid = tmp_path / "valid-after-limit.pdf"
        final_valid.write_bytes(b"%PDF-valid")
        attachments.append(final_valid)

        with patch("news_bot.email_client.smtplib.SMTP") as smtp_class:
            smtp = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp
            with caplog.at_level("WARNING"):
                send_email(
                    mock_config,
                    "Research",
                    "<p>Body</p>",
                    attachment_paths=attachments,
                )

        message = message_from_string(smtp.sendmail.call_args.args[2])
        assert message.get_content_subtype() == "alternative"
        assert final_valid.name in caplog.text
        assert "count" in caplog.text.lower()

    def test_send_email_enforces_cumulative_attachment_limit(
        self,
        mock_config,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        payload_size = (email_client.MAX_TOTAL_ATTACHMENT_BYTES // 3) + 1
        attachments = []
        for index in range(3):
            attachment = tmp_path / f"part-{index}.pdf"
            attachment.write_bytes(b"%PDF-" + b"x" * (payload_size - 5))
            attachments.append(attachment)

        assert payload_size < email_client.MAX_ATTACHMENT_BYTES
        with patch("news_bot.email_client.smtplib.SMTP") as smtp_class:
            smtp = MagicMock()
            smtp_class.return_value.__enter__.return_value = smtp
            with caplog.at_level("WARNING"):
                send_email(
                    mock_config,
                    "Research",
                    "<p>Body</p>",
                    attachment_paths=attachments,
                )

        message = message_from_string(smtp.sendmail.call_args.args[2])
        parts = [part for part in message.walk() if part.get_content_type() == "application/pdf"]
        assert len(parts) == 2
        assert attachments[2].name in caplog.text
        assert "total" in caplog.text.lower()

    def test_send_email_does_not_open_unsupported_attachment(
        self, mock_config, tmp_path: Path
    ):
        unsupported = tmp_path / "payload.bin"
        unsupported.write_bytes(b"opaque")

        with patch("news_bot.email_client.os.open", wraps=os.open) as safe_open:
            with patch("news_bot.email_client.smtplib.SMTP") as smtp_class:
                smtp = MagicMock()
                smtp_class.return_value.__enter__.return_value = smtp
                send_email(
                    mock_config,
                    "Research",
                    "<p>Body</p>",
                    attachment_paths=(unsupported,),
                )

        safe_open.assert_not_called()
        message = message_from_string(smtp.sendmail.call_args.args[2])
        assert message.get_content_subtype() == "alternative"

    def test_send_email_rejects_file_replaced_between_lstat_and_open(
        self,
        mock_config,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        attachment = tmp_path / "raced.pdf"
        attachment.write_bytes(b"%PDF-valid")
        real_fstat = os.fstat

        def mismatched_fstat(fd: int):
            result = real_fstat(fd)
            values = list(result)
            values[1] += 1
            return os.stat_result(values)

        with patch("news_bot.email_client.os.fstat", side_effect=mismatched_fstat):
            with patch("news_bot.email_client.smtplib.SMTP") as smtp_class:
                smtp = MagicMock()
                smtp_class.return_value.__enter__.return_value = smtp
                with caplog.at_level("WARNING"):
                    send_email(
                        mock_config,
                        "Research",
                        "<p>Body</p>",
                        attachment_paths=(attachment,),
                    )

        message = message_from_string(smtp.sendmail.call_args.args[2])
        assert message.get_content_subtype() == "alternative"
        assert "raced.pdf" in caplog.text

    def test_send_email_rejects_same_inode_size_mutation_during_read(
        self,
        mock_config,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        attachment = tmp_path / "mutated.pdf"
        original = b"%PDF-original"
        attacker = b"%PDF-attacker"
        assert len(original) == len(attacker)
        attachment.write_bytes(original)
        original_stat = attachment.stat()
        real_read = os.read
        mutated = False

        def mutate_after_read(fd: int, size: int) -> bytes:
            nonlocal mutated
            payload = real_read(fd, size)
            if not mutated:
                mutated = True
                attachment.write_bytes(attacker)
                os.utime(
                    attachment,
                    ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000),
                )
            return payload

        with patch("news_bot.email_client.os.read", side_effect=mutate_after_read):
            with patch("news_bot.email_client.smtplib.SMTP") as smtp_class:
                smtp = MagicMock()
                smtp_class.return_value.__enter__.return_value = smtp
                with caplog.at_level("WARNING"):
                    send_email(
                        mock_config,
                        "Research",
                        "<p>Body</p>",
                        attachment_paths=(attachment,),
                    )

        message = message_from_string(smtp.sendmail.call_args.args[2])
        assert message.get_content_subtype() == "alternative"
        assert "mutated.pdf" in caplog.text
        assert str(tmp_path) not in caplog.text

    def test_send_email_documents_finalized_atomic_attachment_contract(self):
        documentation = send_email.__doc__ or ""
        assert "finalized" in documentation.lower()
        assert "atomic" in documentation.lower()


class TestBuildSectionedEmailHtml:
    """Tests for sectioned email HTML builder."""

    def test_build_sectioned_email_contains_all_sections(self):
        """Test that sectioned email contains all 5 section headers."""
        sections = {
            "World News": [make_article("world-1", "World Article")],
            "US Tech": [make_article("us-tech-1", "US Tech Article")],
            "US Industry": [make_article("us-ind-1", "US Industry Article")],
            "Malaysia Tech": [make_article("my-tech-1", "Malaysia Tech Article")],
            "Malaysia Industry": [make_article("my-ind-1", "Malaysia Industry Article")],
        }
        summaries = {
            "world-1": "World summary",
            "us-tech-1": "US Tech summary",
            "us-ind-1": "US Industry summary",
            "my-tech-1": "Malaysia Tech summary",
            "my-ind-1": "Malaysia Industry summary",
        }

        html = build_sectioned_email_html(date(2024, 1, 15), sections, summaries)

        # Check all section names appear
        assert "World News" in html
        assert "US Tech" in html
        assert "US Industry" in html
        assert "Malaysia Tech" in html
        assert "Malaysia Industry" in html

    def test_build_sectioned_email_contains_articles(self):
        """Test that articles appear in their sections."""
        sections = {
            "World News": [make_article("world-1", "Global Market Update")],
            "US Tech": [make_article("us-tech-1", "Apple Announcement")],
            "US Industry": [],
            "Malaysia Tech": [make_article("my-tech-1", "Malaysian Startup")],
            "Malaysia Industry": [],
        }
        summaries = {
            "world-1": "Markets are moving.",
            "us-tech-1": "Apple news here.",
            "my-tech-1": "Startup raises funds.",
        }

        html = build_sectioned_email_html(date(2024, 1, 15), sections, summaries)

        # Check article titles appear
        assert "Global Market Update" in html
        assert "Apple Announcement" in html
        assert "Malaysian Startup" in html

        # Check summaries appear
        assert "Markets are moving." in html
        assert "Apple news here." in html

    def test_build_sectioned_email_skips_empty_sections(self):
        """Test that empty sections are skipped entirely."""
        sections = {
            "World News": [make_article("world-1", "World Article")],
            "US Tech": [],  # Empty - should be skipped
            "US Industry": [],  # Empty - should be skipped
            "Malaysia Tech": [],  # Empty - should be skipped
            "Malaysia Industry": [],  # Empty - should be skipped
        }
        summaries = {"world-1": "Summary"}

        html = build_sectioned_email_html(date(2024, 1, 15), sections, summaries)

        # Empty sections should be skipped entirely (not shown)
        assert "US Tech" not in html
        assert "US Industry" not in html
        assert "Malaysia Tech" not in html
        assert "Malaysia Industry" not in html
        # World News should have its article
        assert "World News" in html
        assert "World Article" in html

    def test_build_sectioned_email_includes_section_emojis(self):
        """Test that section emojis are included."""
        sections = {
            "World News": [make_article("world-1", "Article")],
            "US Tech": [],
            "US Industry": [],
            "Malaysia Tech": [],
            "Malaysia Industry": [],
        }
        summaries = {"world-1": "Summary"}

        html = build_sectioned_email_html(date(2024, 1, 15), sections, summaries)

        # Check emojis from SECTION_CONFIG
        assert SECTION_CONFIG["World News"]["emoji"] in html

    def test_build_sectioned_email_has_date(self):
        """Test that email includes formatted date."""
        sections = {"World News": [], "US Tech": [], "US Industry": [], 
                    "Malaysia Tech": [], "Malaysia Industry": []}
        summaries = {}

        html = build_sectioned_email_html(date(2024, 1, 15), sections, summaries)

        assert "January 15, 2024" in html

    def test_build_sectioned_email_preserves_section_order(self):
        """Test that sections appear in correct order."""
        sections = {
            "Malaysia Industry": [make_article("mi-1", "MI Article")],
            "World News": [make_article("wn-1", "WN Article")],
            "US Tech": [make_article("ut-1", "UT Article")],
            "Malaysia Tech": [make_article("mt-1", "MT Article")],
            "US Industry": [make_article("ui-1", "UI Article")],
        }
        summaries = {k: "Summary" for k in ["mi-1", "wn-1", "ut-1", "mt-1", "ui-1"]}

        html = build_sectioned_email_html(date(2024, 1, 15), sections, summaries)

        # Check order by finding positions
        world_pos = html.find("World News")
        us_tech_pos = html.find("US Tech")
        us_ind_pos = html.find("US Industry")
        my_tech_pos = html.find("Malaysia Tech")
        my_ind_pos = html.find("Malaysia Industry")

        assert world_pos < us_tech_pos < us_ind_pos < my_tech_pos < my_ind_pos
