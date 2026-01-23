"""Tests for summary validation logic."""

from unittest.mock import MagicMock, patch

import pytest

from news_bot.summary_validator import (
    BORDERLINE_THRESHOLD,
    MIN_SUMMARY_LENGTH,
    ValidationResult,
    validate_summary,
    validate_summary_llm,
    validate_summary_rules,
    _check_length,
    _check_refusal_patterns,
    _check_structure,
)


class TestRefusalPatternDetection:
    """Tests for refusal pattern detection."""

    def test_detects_direct_refusal_i_cannot(self):
        """Test detection of 'I cannot' refusals."""
        summary = "I cannot summarize this article due to lack of information."
        result = _check_refusal_patterns(summary)
        
        assert result.is_valid is False
        assert "refusal pattern" in result.reason.lower()

    def test_detects_direct_refusal_i_cant(self):
        """Test detection of 'I can't' refusals."""
        summary = "I can't provide a summary for this content."
        result = _check_refusal_patterns(summary)
        
        assert result.is_valid is False

    def test_detects_unable_to_provide(self):
        """Test detection of 'unable to provide' refusals."""
        summary = "Unable to provide a summary as the article content is restricted."
        result = _check_refusal_patterns(summary)
        
        assert result.is_valid is False

    def test_detects_content_policy_refusal(self):
        """Test detection of content policy refusals."""
        summary = "Due to content policy restrictions, I cannot generate this summary."
        result = _check_refusal_patterns(summary)
        
        assert result.is_valid is False

    def test_detects_apology_refusal(self):
        """Test detection of apologetic refusals."""
        summary = "I apologize, but I cannot summarize this article as it lacks details."
        result = _check_refusal_patterns(summary)
        
        assert result.is_valid is False

    def test_detects_sorry_refusal(self):
        """Test detection of 'sorry' refusals."""
        summary = "I'm sorry, but I cannot provide a summary for this content."
        result = _check_refusal_patterns(summary)
        
        assert result.is_valid is False

    def test_detects_insufficient_information(self):
        """Test detection of 'insufficient information' refusals."""
        summary = "There is insufficient information in the article to create a summary."
        result = _check_refusal_patterns(summary)
        
        assert result.is_valid is False

    def test_detects_meta_commentary(self):
        """Test detection of meta-commentary instead of actual summary."""
        summary = "This article discusses the latest developments in AI technology."
        result = _check_refusal_patterns(summary)
        
        assert result.is_valid is False
        
    def test_detects_article_talks_about(self):
        """Test detection of 'the article talks about' meta-commentary."""
        summary = "The article talks about Apple's quarterly earnings report."
        result = _check_refusal_patterns(summary)
        
        assert result.is_valid is False

    def test_valid_summary_passes(self):
        """Test that a valid summary passes pattern check."""
        summary = (
            "Apple reported Q4 earnings of $1.50 per share, beating estimates by 5%. "
            "Revenue reached $89.5 billion, up 8% year-over-year with iPhone sales driving growth. "
            "So what? Strong iPhone demand validates the premium pricing strategy, but supply chain "
            "constraints in China remain a risk for Q1 guidance (theme: Consumer Tech, Hardware)."
        )
        result = _check_refusal_patterns(summary)
        
        assert result.is_valid is True

    def test_medium_refusal_has_lower_confidence(self):
        """Test that medium refusal patterns return lower confidence."""
        summary = "Based on the provided content, limited information is available to summarize."
        result = _check_refusal_patterns(summary)
        
        # Medium patterns should still mark as invalid but with lower confidence
        assert result.is_valid is False
        assert result.confidence < 0.8


class TestLengthCheck:
    """Tests for summary length validation."""

    def test_rejects_very_short_summary(self):
        """Test that very short summaries are rejected."""
        summary = "Too short."
        result = _check_length(summary)
        
        assert result.is_valid is False
        assert "too short" in result.reason.lower()

    def test_rejects_empty_summary(self):
        """Test that empty summaries are rejected."""
        summary = ""
        result = _check_length(summary)
        
        assert result.is_valid is False

    def test_rejects_whitespace_only(self):
        """Test that whitespace-only summaries are rejected."""
        summary = "   \n\t   "
        result = _check_length(summary)
        
        assert result.is_valid is False

    def test_accepts_minimum_length_summary(self):
        """Test that summaries at minimum length pass."""
        summary = "A" * MIN_SUMMARY_LENGTH
        result = _check_length(summary)
        
        assert result.is_valid is True

    def test_accepts_long_summary(self):
        """Test that long summaries pass length check."""
        summary = "A" * 500
        result = _check_length(summary)
        
        assert result.is_valid is True


class TestStructureCheck:
    """Tests for summary structure validation."""

    def test_valid_with_so_what(self):
        """Test that summaries with 'So what?' pass."""
        summary = (
            "Company X announced earnings. Revenue was $10B. "
            "So what? This indicates strong growth potential."
        )
        result = _check_structure(summary)
        
        assert result.is_valid is True
        assert result.confidence == 1.0

    def test_borderline_without_so_what(self):
        """Test that summaries without 'So what?' are borderline."""
        summary = (
            "Company X announced earnings. Revenue was $10B. "
            "This indicates strong growth potential for the sector."
        )
        result = _check_structure(summary)
        
        # Should be valid but with lower confidence (borderline)
        assert result.is_valid is True
        assert result.confidence < 1.0
        assert "so what?" in result.reason.lower()


class TestValidateSummaryRules:
    """Tests for combined rule-based validation."""

    def test_valid_summary_passes_all_rules(self):
        """Test that a well-formed summary passes all rules."""
        summary = (
            "Apple reported Q4 earnings of $1.50 per share, beating estimates by 5%. "
            "Revenue reached $89.5 billion, up 8% year-over-year with iPhone sales driving growth. "
            "So what? Strong iPhone demand validates the premium pricing strategy, but supply chain "
            "constraints in China remain a risk for Q1 guidance (theme: Consumer Tech, Hardware)."
        )
        result = validate_summary_rules(summary)
        
        assert result.is_valid is True
        assert result.confidence >= BORDERLINE_THRESHOLD

    def test_refusal_fails_fast(self):
        """Test that refusals are caught even if length/structure is ok."""
        summary = (
            "I cannot summarize this article because it contains restricted content. "
            "Please provide a different article for summarization. "
            "So what? N/A"
        )
        result = validate_summary_rules(summary)
        
        assert result.is_valid is False

    def test_short_summary_fails(self):
        """Test that short summaries fail validation."""
        summary = "Short."
        result = validate_summary_rules(summary)
        
        assert result.is_valid is False


class TestLLMValidation:
    """Tests for LLM-based validation."""

    def test_llm_validation_valid_response(self):
        """Test LLM validation with VALID response."""
        mock_response = {"response": "VALID"}
        
        with patch("news_bot.summary_validator.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()
            
            result = validate_summary_llm(
                summary="Some summary text",
                model="llama3",
                base_url="http://localhost:11434",
            )
        
        assert result.is_valid is True
        assert "llm classified as valid" in result.reason.lower()

    def test_llm_validation_invalid_response(self):
        """Test LLM validation with INVALID response."""
        mock_response = {"response": "INVALID"}
        
        with patch("news_bot.summary_validator.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()
            
            result = validate_summary_llm(
                summary="I cannot summarize this",
                model="llama3",
                base_url="http://localhost:11434",
            )
        
        assert result.is_valid is False
        assert "llm classified as invalid" in result.reason.lower()

    def test_llm_validation_handles_network_error(self):
        """Test LLM validation gracefully handles network errors."""
        import requests as req
        
        with patch("news_bot.summary_validator.requests.post") as mock_post:
            mock_post.side_effect = req.RequestException("Connection failed")
            
            result = validate_summary_llm(
                summary="Some summary",
                model="llama3",
                base_url="http://localhost:11434",
            )
        
        # Should default to accepting on error
        assert result.is_valid is True
        assert "failed" in result.reason.lower()

    def test_llm_validation_handles_ambiguous_response(self):
        """Test LLM validation handles ambiguous responses."""
        mock_response = {"response": "Maybe it's valid?"}
        
        with patch("news_bot.summary_validator.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()
            
            result = validate_summary_llm(
                summary="Some summary",
                model="llama3",
                base_url="http://localhost:11434",
            )
        
        # Should default to accepting on ambiguous
        assert result.is_valid is True


class TestHybridValidation:
    """Tests for the hybrid validation approach."""

    def test_clear_refusal_skips_llm(self):
        """Test that clear refusals don't trigger LLM validation."""
        summary = "I cannot summarize this article."
        
        with patch("news_bot.summary_validator.validate_summary_llm") as mock_llm:
            result = validate_summary(
                summary=summary,
                model="llama3",
                base_url="http://localhost:11434",
                use_llm_fallback=True,
            )
        
        # Should NOT call LLM for clear refusals
        mock_llm.assert_not_called()
        assert result.is_valid is False

    def test_clear_valid_skips_llm(self):
        """Test that clearly valid summaries don't trigger LLM validation."""
        summary = (
            "Apple reported Q4 earnings of $1.50 per share, beating estimates by 5%. "
            "Revenue reached $89.5 billion, up 8% year-over-year. "
            "So what? Strong demand validates pricing strategy (theme: Tech)."
        )
        
        with patch("news_bot.summary_validator.validate_summary_llm") as mock_llm:
            result = validate_summary(
                summary=summary,
                model="llama3",
                base_url="http://localhost:11434",
                use_llm_fallback=True,
            )
        
        # Should NOT call LLM for clearly valid summaries
        mock_llm.assert_not_called()
        assert result.is_valid is True

    def test_borderline_triggers_llm(self):
        """Test that borderline cases trigger LLM validation."""
        # Summary without "So what?" - borderline
        summary = (
            "Apple reported Q4 earnings of $1.50 per share, beating estimates. "
            "Revenue reached $89.5 billion with strong iPhone sales. "
            "This validates the premium pricing strategy for the company."
        )
        
        mock_llm_result = ValidationResult(is_valid=True, reason="LLM approved", confidence=0.85)
        
        with patch("news_bot.summary_validator.validate_summary_llm", return_value=mock_llm_result) as mock_llm:
            result = validate_summary(
                summary=summary,
                model="llama3",
                base_url="http://localhost:11434",
                use_llm_fallback=True,
            )
        
        # Should call LLM for borderline cases
        mock_llm.assert_called_once()
        assert result.is_valid is True

    def test_no_llm_fallback_uses_rules_only(self):
        """Test that disabling LLM fallback uses rules only."""
        summary = (
            "Some borderline summary without the expected structure. "
            "It has enough content but missing So what section entirely."
        )
        
        with patch("news_bot.summary_validator.validate_summary_llm") as mock_llm:
            result = validate_summary(
                summary=summary,
                model="llama3",
                base_url="http://localhost:11434",
                use_llm_fallback=False,  # Disabled
            )
        
        # Should NOT call LLM when disabled
        mock_llm.assert_not_called()

    def test_missing_model_skips_llm(self):
        """Test that missing model/base_url skips LLM validation."""
        summary = "Some borderline summary text here."
        
        with patch("news_bot.summary_validator.validate_summary_llm") as mock_llm:
            result = validate_summary(
                summary=summary,
                model=None,  # No model
                base_url=None,  # No base URL
                use_llm_fallback=True,
            )
        
        # Should NOT call LLM without credentials
        mock_llm.assert_not_called()
