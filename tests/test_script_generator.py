"""Tests for radio script generation."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from news_bot.script_generator import (
    ScriptConfig,
    ScriptGeneratorError,
    _clean_script,
    _format_summaries_for_script,
    generate_script,
    generate_broadcast_script,
)


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_SUMMARIES = {
    "article-1": (
        "Nvidia reported Q4 earnings significantly above consensus, driven by "
        "unprecedented demand for H100 accelerators. Revenue surged 265% YoY to $22.1B. "
        "So what? The AI infrastructure supercycle supports continued margin expansion."
    ),
    "article-2": (
        "Apple announced a $110B stock buyback program, the largest in corporate history. "
        "The board also raised the quarterly dividend by 4% to $0.25 per share. "
        "So what? Strong cash generation validates the services pivot thesis."
    ),
    "article-3": (
        "Tesla delivered 387,000 vehicles in Q1, missing analyst expectations of 449,000. "
        "Production was impacted by factory retooling in Austin and Berlin. "
        "So what? Near-term volume headwinds may pressure margins through H1 2024."
    ),
}


# =============================================================================
# ScriptConfig Tests
# =============================================================================

class TestScriptConfig:
    """Tests for ScriptConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ScriptConfig()
        
        assert config.duration_minutes == 2.0
        assert config.words_per_minute == 150

    def test_target_word_count_calculation(self):
        """Test word count calculation based on duration."""
        config = ScriptConfig(duration_minutes=2.0, words_per_minute=150)
        assert config.target_word_count == 300
        
        config = ScriptConfig(duration_minutes=3.0, words_per_minute=150)
        assert config.target_word_count == 450
        
        config = ScriptConfig(duration_minutes=1.5, words_per_minute=140)
        assert config.target_word_count == 210

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ScriptConfig(duration_minutes=5.0, words_per_minute=120)
        
        assert config.duration_minutes == 5.0
        assert config.words_per_minute == 120
        assert config.target_word_count == 600


# =============================================================================
# Clean Script Tests
# =============================================================================

class TestCleanScript:
    """Tests for _clean_script function."""

    def test_removes_markdown_bold(self):
        """Test removal of markdown bold formatting."""
        text = "This is **important** news from the **market**."
        result = _clean_script(text)
        assert "**" not in result
        assert "important" in result
        assert "market" in result

    def test_removes_markdown_italic(self):
        """Test removal of markdown italic formatting."""
        text = "The *key takeaway* is that *growth* continues."
        result = _clean_script(text)
        assert "*" not in result
        assert "key takeaway" in result

    def test_removes_markdown_headers(self):
        """Test removal of markdown headers."""
        text = "## Main Story\nThe market rallied today."
        result = _clean_script(text)
        assert "##" not in result
        assert "Main Story" in result

    def test_removes_stage_directions_brackets(self):
        """Test removal of bracketed stage directions."""
        text = "Good morning [pause] and welcome [transition] to the show."
        result = _clean_script(text)
        assert "[" not in result
        assert "]" not in result
        assert "Good morning" in result
        assert "and welcome" in result

    def test_removes_parenthetical_directions(self):
        """Test removal of common parenthetical stage directions."""
        text = "Breaking news (pause) from Wall Street (laughs) today."
        result = _clean_script(text)
        assert "(pause)" not in result
        assert "(laughs)" not in result

    def test_normalizes_ellipses(self):
        """Test normalization of ellipses."""
        text = "Markets are moving.... and moving fast......"
        result = _clean_script(text)
        assert "...." not in result
        assert "......" not in result

    def test_removes_preambles(self):
        """Test removal of common LLM preambles."""
        preambles = [
            "Here is the script:\nGood morning markets...",
            "Here's your script:\nGood morning markets...",
            "Radio script:\nGood morning markets...",
        ]
        
        for text in preambles:
            result = _clean_script(text)
            assert result.startswith("Good morning")

    def test_preserves_valid_content(self):
        """Test that valid script content is preserved."""
        text = "Good morning, Wall Street! The bulls are running today."
        result = _clean_script(text)
        assert result == text

    def test_cleans_excessive_whitespace(self):
        """Test cleanup of excessive whitespace."""
        text = "Line one.\n\n\n\nLine two.\n\n\n\n\nLine three."
        result = _clean_script(text)
        assert "\n\n\n" not in result


# =============================================================================
# Format Summaries Tests
# =============================================================================

class TestFormatSummariesForScript:
    """Tests for _format_summaries_for_script function."""

    def test_formats_single_summary(self):
        """Test formatting of a single summary."""
        summaries = {"article-1": "This is the summary content."}
        result = _format_summaries_for_script(summaries)
        
        assert "Story 1:" in result
        assert "This is the summary content." in result

    def test_formats_multiple_summaries(self):
        """Test formatting of multiple summaries."""
        summaries = {
            "a": "First summary.",
            "b": "Second summary.",
            "c": "Third summary.",
        }
        result = _format_summaries_for_script(summaries)
        
        assert "Story 1:" in result
        assert "Story 2:" in result
        assert "Story 3:" in result
        assert "First summary." in result
        assert "Second summary." in result
        assert "Third summary." in result

    def test_separates_with_double_newlines(self):
        """Test that summaries are separated by double newlines."""
        summaries = {"a": "First", "b": "Second"}
        result = _format_summaries_for_script(summaries)
        
        assert "\n\n" in result

    def test_empty_dict_returns_empty_string(self):
        """Test that empty input returns empty output."""
        result = _format_summaries_for_script({})
        assert result == ""


# =============================================================================
# Generate Script Tests
# =============================================================================

class TestGenerateScript:
    """Tests for generate_script function."""

    def test_generate_script_calls_ollama(self):
        """Test that generate_script calls Ollama API correctly."""
        mock_response = {
            "response": "Good morning, Wall Street! Here's what's moving markets today..."
        }

        with patch("news_bot.script_generator.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()

            result = generate_script(
                summaries=SAMPLE_SUMMARIES,
                model="llama3",
                base_url="http://localhost:11434",
            )

        # Verify API was called
        mock_post.assert_called_once()
        call_args = mock_post.call_args

        # Check URL
        assert call_args[0][0] == "http://localhost:11434/api/generate"

        # Check payload structure
        payload = call_args[1]["json"]
        assert payload["model"] == "llama3"
        assert payload["stream"] is False
        assert "temperature" in payload["options"]

    def test_generate_script_includes_summaries_in_prompt(self):
        """Test that summaries are included in the prompt."""
        with patch("news_bot.script_generator.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "Script content"}
            mock_post.return_value.raise_for_status = MagicMock()

            generate_script(
                summaries=SAMPLE_SUMMARIES,
                model="llama3",
                base_url="http://localhost:11434",
            )

        payload = mock_post.call_args[1]["json"]
        prompt = payload["prompt"]

        # Check that summary content is in the prompt
        assert "Nvidia" in prompt
        assert "Apple" in prompt
        assert "Tesla" in prompt

    def test_generate_script_uses_custom_config(self):
        """Test that custom ScriptConfig is respected."""
        config = ScriptConfig(duration_minutes=5.0, words_per_minute=130)

        with patch("news_bot.script_generator.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "Script"}
            mock_post.return_value.raise_for_status = MagicMock()

            generate_script(
                summaries=SAMPLE_SUMMARIES,
                model="llama3",
                base_url="http://localhost:11434",
                script_config=config,
            )

        payload = mock_post.call_args[1]["json"]
        prompt = payload["prompt"]

        # Check that duration is reflected in prompt
        assert "5.0" in prompt or "5-minute" in prompt

    def test_generate_script_uses_broadcast_date(self):
        """Test that broadcast date is included in prompt."""
        test_date = date(2026, 1, 15)

        with patch("news_bot.script_generator.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "Script"}
            mock_post.return_value.raise_for_status = MagicMock()

            generate_script(
                summaries=SAMPLE_SUMMARIES,
                model="llama3",
                base_url="http://localhost:11434",
                broadcast_date=test_date,
            )

        payload = mock_post.call_args[1]["json"]
        prompt = payload["prompt"]

        assert "January 15, 2026" in prompt

    def test_generate_script_empty_summaries_raises_error(self):
        """Test that empty summaries raise an error."""
        with pytest.raises(ScriptGeneratorError) as exc_info:
            generate_script(
                summaries={},
                model="llama3",
                base_url="http://localhost:11434",
            )

        assert "No summaries provided" in str(exc_info.value)

    def test_generate_script_timeout_error(self):
        """Test handling of timeout errors."""
        import requests as req

        with patch("news_bot.script_generator.requests.post") as mock_post:
            mock_post.side_effect = req.Timeout()

            with pytest.raises(ScriptGeneratorError) as exc_info:
                generate_script(
                    summaries=SAMPLE_SUMMARIES,
                    model="llama3",
                    base_url="http://localhost:11434",
                )

            assert "timed out" in str(exc_info.value).lower()

    def test_generate_script_connection_error(self):
        """Test handling of connection errors."""
        import requests as req

        with patch("news_bot.script_generator.requests.post") as mock_post:
            mock_post.side_effect = req.ConnectionError("Connection refused")

            with pytest.raises(ScriptGeneratorError) as exc_info:
                generate_script(
                    summaries=SAMPLE_SUMMARIES,
                    model="llama3",
                    base_url="http://localhost:11434",
                )

            assert "Failed to connect" in str(exc_info.value)

    def test_generate_script_empty_response_error(self):
        """Test handling of empty Ollama response."""
        with patch("news_bot.script_generator.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": ""}
            mock_post.return_value.raise_for_status = MagicMock()

            with pytest.raises(ScriptGeneratorError) as exc_info:
                generate_script(
                    summaries=SAMPLE_SUMMARIES,
                    model="llama3",
                    base_url="http://localhost:11434",
                )

            assert "empty response" in str(exc_info.value).lower()

    def test_generate_script_cleans_output(self):
        """Test that script output is cleaned."""
        mock_response = {
            "response": "Here is the script:\n\n**Good morning**, Wall Street!"
        }

        with patch("news_bot.script_generator.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()

            result = generate_script(
                summaries=SAMPLE_SUMMARIES,
                model="llama3",
                base_url="http://localhost:11434",
            )

        # Preamble should be removed
        assert not result.startswith("Here is")
        # Markdown should be removed
        assert "**" not in result


# =============================================================================
# Generate Broadcast Script Tests
# =============================================================================

class TestGenerateBroadcastScript:
    """Tests for generate_broadcast_script convenience function."""

    def test_generate_broadcast_script_basic(self):
        """Test basic broadcast script generation."""
        with patch("news_bot.script_generator.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {
                "response": "Good morning, here's your market update!"
            }
            mock_post.return_value.raise_for_status = MagicMock()

            result = generate_broadcast_script(
                summaries=SAMPLE_SUMMARIES,
                ollama_model="llama3",
                ollama_base_url="http://localhost:11434",
            )

        assert "Good morning" in result

    def test_generate_broadcast_script_custom_duration(self):
        """Test broadcast script with custom duration."""
        with patch("news_bot.script_generator.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "Script"}
            mock_post.return_value.raise_for_status = MagicMock()

            generate_broadcast_script(
                summaries=SAMPLE_SUMMARIES,
                ollama_model="llama3",
                ollama_base_url="http://localhost:11434",
                duration_minutes=3.5,
            )

        payload = mock_post.call_args[1]["json"]
        prompt = payload["prompt"]

        # Duration should be reflected
        assert "3.5" in prompt


# =============================================================================
# Integration-Style Tests
# =============================================================================

class TestScriptGeneratorIntegration:
    """Integration-style tests for complete script generation flow."""

    def test_full_script_generation_flow(self):
        """Test the complete flow from summaries to script."""
        realistic_response = """
Good morning, Wall Street! It's January 17th, and the market's already showing us what kind of day it's going to be.

Let me cut through the noise with three stories that'll shape your portfolio before the closing bell.

First up... Nvidia's earnings absolutely crushed it. We're talking two hundred sixty-five percent year-over-year revenue growth to twenty-two point one billion dollars. The H100 accelerators are flying off the shelves faster than traders can say "artificial intelligence". The smart money? They're doubling down on the AI infrastructure play.

Shifting gears to Cupertino... Apple just dropped the largest corporate buyback in history. One hundred ten billion dollars. Let that sink in. Plus they're bumping the dividend four percent. What does this tell us? Their services pivot is generating serious cash, and management wants you to know it.

But here's where it gets interesting... Tesla delivered three hundred eighty-seven thousand vehicles, missing expectations by a mile. Factory retooling in Austin and Berlin hit production hard. If you're long Tesla, buckle up for a bumpy first half.

That's your market brief. Stay sharp out there... the Street never sleeps.
"""

        with patch("news_bot.script_generator.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": realistic_response}
            mock_post.return_value.raise_for_status = MagicMock()

            result = generate_script(
                summaries=SAMPLE_SUMMARIES,
                model="llama3",
                base_url="http://localhost:11434",
            )

        # Script should contain key elements
        assert "Wall Street" in result
        assert "Nvidia" in result
        assert "Apple" in result
        assert "Tesla" in result

    def test_script_maintains_story_count(self):
        """Test that the prompt correctly indicates story count."""
        with patch("news_bot.script_generator.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "Script"}
            mock_post.return_value.raise_for_status = MagicMock()

            generate_script(
                summaries=SAMPLE_SUMMARIES,  # 3 stories
                model="llama3",
                base_url="http://localhost:11434",
            )

        payload = mock_post.call_args[1]["json"]
        prompt = payload["prompt"]

        # Should mention 3 stories
        assert "3 stories" in prompt or "3 story" in prompt.lower()
