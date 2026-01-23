"""
Summary validation to detect LLM refusals and low-quality outputs.

Uses a hybrid approach: fast rule-based checks first, with optional
LLM validation for borderline cases.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import requests


logger = logging.getLogger(__name__)


# =============================================================================
# REFUSAL PATTERNS - Common LLM refusal phrases
# =============================================================================

# High-confidence refusal patterns (clear rejections)
REFUSAL_PATTERNS_STRONG = [
    # Direct refusals
    r"\bi cannot\b",
    r"\bi can't\b",
    r"\bi am unable to\b",
    r"\bi'm unable to\b",
    r"\bunable to (provide|generate|create|summarize|write)\b",
    r"\bcannot (provide|generate|create|summarize|write)\b",
    # Content policy
    r"\bcontent policy\b",
    r"\bguidelines\b.*\b(prevent|restrict|prohibit)\b",
    r"\binappropriate\b",
    r"\bharmful content\b",
    # Apologies indicating refusal
    r"\bi apologize\b.*\b(cannot|can't|unable)\b",
    r"\bi'm sorry\b.*\b(cannot|can't|unable)\b",
    r"\bsorry, but i\b",
    # Explicit inability
    r"\bno information (available|provided|found)\b",
    r"\binsufficient information\b",
    r"\bcannot be summarized\b",
    r"\bnot enough (content|information|details)\b",
    # Meta-commentary instead of summary
    r"^(this|the) article (discusses|talks about|is about|covers)\b",
    r"^(this|the) (text|content|passage) (discusses|talks about|is about)\b",
]

# Medium-confidence patterns (might be borderline)
REFUSAL_PATTERNS_MEDIUM = [
    r"\bi don't have\b.*\b(access|information)\b",
    r"\bbased on the (provided|given|available)\b.*\blimited\b",
    r"\bthe article (does not|doesn't) (provide|contain|include)\b",
    r"\bno specific (details|data|numbers|figures)\b",
    r"\bgeneric\b.*\b(response|summary|content)\b",
]

# Minimum summary length (characters)
MIN_SUMMARY_LENGTH = 80

# Expected structure marker
EXPECTED_STRUCTURE_MARKER = "so what?"


@dataclass
class ValidationResult:
    """Result of summary validation."""
    
    is_valid: bool
    reason: Optional[str] = None
    confidence: float = 1.0  # 1.0 = certain, <0.7 = borderline (needs LLM check)


def _check_refusal_patterns(summary: str) -> ValidationResult:
    """
    Check for refusal patterns in the summary.
    
    Returns:
        ValidationResult with is_valid=False if refusal detected.
    """
    summary_lower = summary.lower()
    
    # Check strong refusal patterns (high confidence rejection)
    for pattern in REFUSAL_PATTERNS_STRONG:
        if re.search(pattern, summary_lower):
            return ValidationResult(
                is_valid=False,
                reason=f"Refusal pattern detected: {pattern}",
                confidence=0.95,
            )
    
    # Check medium refusal patterns (borderline)
    for pattern in REFUSAL_PATTERNS_MEDIUM:
        if re.search(pattern, summary_lower):
            return ValidationResult(
                is_valid=False,
                reason=f"Possible refusal pattern: {pattern}",
                confidence=0.6,  # Borderline - may need LLM validation
            )
    
    return ValidationResult(is_valid=True, confidence=1.0)


def _check_length(summary: str) -> ValidationResult:
    """
    Check if summary meets minimum length requirements.
    """
    if len(summary.strip()) < MIN_SUMMARY_LENGTH:
        return ValidationResult(
            is_valid=False,
            reason=f"Summary too short ({len(summary)} chars, minimum {MIN_SUMMARY_LENGTH})",
            confidence=0.9,
        )
    return ValidationResult(is_valid=True, confidence=1.0)


def _check_structure(summary: str) -> ValidationResult:
    """
    Check if summary has expected structure (contains "So what?").
    
    This is a soft check - missing structure results in borderline confidence.
    """
    summary_lower = summary.lower()
    
    if EXPECTED_STRUCTURE_MARKER not in summary_lower:
        return ValidationResult(
            is_valid=True,  # Don't reject, but flag as borderline
            reason="Missing expected 'So what?' structure",
            confidence=0.7,  # Borderline
        )
    
    return ValidationResult(is_valid=True, confidence=1.0)


def validate_summary_rules(summary: str) -> ValidationResult:
    """
    Perform rule-based validation on a summary.
    
    Checks:
    1. Refusal patterns (strong and medium)
    2. Minimum length
    3. Expected structure
    
    Args:
        summary: The generated summary text.
    
    Returns:
        ValidationResult indicating validity and confidence.
    """
    # Check for refusal patterns first (most important)
    refusal_result = _check_refusal_patterns(summary)
    if not refusal_result.is_valid:
        return refusal_result
    
    # Check length
    length_result = _check_length(summary)
    if not length_result.is_valid:
        return length_result
    
    # Check structure (soft check)
    structure_result = _check_structure(summary)
    
    # If structure check passed but with low confidence, propagate that
    if structure_result.confidence < 1.0:
        return ValidationResult(
            is_valid=True,
            reason=structure_result.reason,
            confidence=structure_result.confidence,
        )
    
    return ValidationResult(is_valid=True, confidence=1.0)


# =============================================================================
# LLM VALIDATION - For borderline cases
# =============================================================================

LLM_VALIDATION_PROMPT = """You are a quality control reviewer. Analyze the following text and determine if it is a valid news summary or if it's a refusal/error message.

A VALID summary:
- Contains specific facts, numbers, or events from a news article
- Discusses a company, market, or economic topic
- Provides analysis or implications

An INVALID response (refusal/error):
- Says "I cannot", "I'm unable to", or similar refusals
- Apologizes for not being able to help
- Says there's not enough information
- Is generic filler without specific content
- Describes what the article is about instead of summarizing it

Text to analyze:
\"\"\"
{summary}
\"\"\"

Respond with ONLY one word: VALID or INVALID"""


def validate_summary_llm(
    summary: str,
    model: str,
    base_url: str,
    timeout: int = 30,
) -> ValidationResult:
    """
    Use LLM to validate if a summary is a genuine summary or a refusal.
    
    This is used for borderline cases where rule-based checks are uncertain.
    
    Args:
        summary: The summary text to validate.
        model: Ollama model name.
        base_url: Ollama API base URL.
        timeout: Request timeout in seconds.
    
    Returns:
        ValidationResult from LLM analysis.
    """
    prompt = LLM_VALIDATION_PROMPT.format(summary=summary)
    
    url = f"{base_url.rstrip('/')}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for consistent classification
            "num_predict": 10,   # Only need one word
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"LLM validation request failed: {e}")
        # On failure, default to accepting the summary
        return ValidationResult(
            is_valid=True,
            reason="LLM validation failed, defaulting to accept",
            confidence=0.5,
        )
    
    data = response.json()
    llm_response = data.get("response", "").strip().upper()
    
    if "INVALID" in llm_response:
        return ValidationResult(
            is_valid=False,
            reason="LLM classified as invalid/refusal",
            confidence=0.85,
        )
    elif "VALID" in llm_response:
        return ValidationResult(
            is_valid=True,
            reason="LLM classified as valid",
            confidence=0.85,
        )
    else:
        # Ambiguous response - default to accept
        logger.warning(f"Ambiguous LLM validation response: {llm_response}")
        return ValidationResult(
            is_valid=True,
            reason=f"Ambiguous LLM response: {llm_response}",
            confidence=0.5,
        )


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

# Confidence threshold below which we call LLM validation
BORDERLINE_THRESHOLD = 0.75


def validate_summary(
    summary: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    use_llm_fallback: bool = True,
) -> ValidationResult:
    """
    Validate a summary using hybrid approach.
    
    1. Run fast rule-based checks
    2. If result is borderline (confidence < 0.75), use LLM validation
    3. Return final validation result
    
    Args:
        summary: The summary text to validate.
        model: Ollama model name (required if use_llm_fallback=True).
        base_url: Ollama API base URL (required if use_llm_fallback=True).
        use_llm_fallback: Whether to use LLM for borderline cases.
    
    Returns:
        ValidationResult indicating if summary should be kept or dropped.
    """
    # Step 1: Rule-based validation
    rule_result = validate_summary_rules(summary)
    
    # If high confidence (either valid or invalid), return immediately
    if rule_result.confidence >= BORDERLINE_THRESHOLD:
        if not rule_result.is_valid:
            logger.debug(f"Summary rejected by rules: {rule_result.reason}")
        return rule_result
    
    # Step 2: Borderline case - use LLM if available
    if use_llm_fallback and model and base_url:
        logger.debug(f"Borderline summary (confidence={rule_result.confidence}), using LLM validation")
        llm_result = validate_summary_llm(summary, model, base_url)
        
        if not llm_result.is_valid:
            logger.debug(f"Summary rejected by LLM: {llm_result.reason}")
        
        return llm_result
    
    # No LLM fallback available - use rule result as-is
    return rule_result
