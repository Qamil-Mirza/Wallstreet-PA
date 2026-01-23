"""
Article summarization using Ollama (local LLM).

Generates analyst-style narrative summaries of articles.
Includes smart chunking to select the most relevant paragraphs.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from .article_extractor import BLOCKED_CONTENT_MARKER
from .config import Config
from .news_client import ArticleMeta
from .summary_validator import validate_summary


# Fallback summary for blocked articles
BLOCKED_SUMMARY_FALLBACK = "Summary unavailable due to site access restrictions."

# Marker for dropped articles (invalid summaries)
DROPPED_ARTICLE_MARKER = None


logger = logging.getLogger(__name__)


# =============================================================================
# SMART CHUNKING - Paragraph Scoring & Selection
# =============================================================================

# Financial terms that indicate important content
FINANCIAL_TERMS = [
    # Metrics
    "eps", "revenue", "earnings", "profit", "margin", "ebitda", "ebit",
    "guidance", "forecast", "outlook", "estimate", "consensus",
    "beat", "miss", "exceeded", "fell short",
    # Money terms
    "million", "billion", "trillion", "mn", "bn", "tn",
    "$", "usd", "eur", "gbp",
    # Percentages and changes
    "%", "percent", "basis points", "bps",
    "increased", "decreased", "grew", "declined", "rose", "fell",
    "up", "down", "gain", "loss", "growth",
    # Valuation
    "valuation", "multiple", "p/e", "pe ratio", "price-to",
    "market cap", "enterprise value",
    # Corporate actions
    "dividend", "buyback", "repurchase", "acquisition", "merger",
    "ipo", "offering", "stake",
    # Rates
    "interest rate", "yield", "coupon", "spread",
]

# Thesis/conclusion phrases
THESIS_PHRASES = [
    "bottom line", "the bottom line", "in summary", "to summarize",
    "we believe", "we expect", "we think", "our view",
    "outlook", "going forward", "looking ahead",
    "risk", "risks include", "key risk", "downside",
    "upside", "catalyst", "driver", "tailwind", "headwind",
    "implies", "suggests", "indicates", "signals",
    "conclusion", "in conclusion", "key takeaway", "takeaway",
    "importantly", "notably", "significantly", "critically",
]

# Common ticker patterns (1-5 uppercase letters, optionally in parentheses)
TICKER_PATTERN = re.compile(r'\b[A-Z]{1,5}\b|\([A-Z]{1,5}\)')

# Number patterns (including decimals, percentages, currency)
NUMBER_PATTERN = re.compile(r'\$?\d+[\d,]*\.?\d*[%]?|\d+\.\d+[%]?')


@dataclass
class ScoredParagraph:
    """A paragraph with its relevance score and original position."""
    index: int
    text: str
    score: float
    char_count: int


def _clean_text(text: str) -> str:
    """Clean article text by normalizing whitespace and removing artifacts."""
    # Normalize various whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\t', ' ', text)
    # Remove excessive whitespace
    text = re.sub(r' +', ' ', text)
    # Remove common web artifacts
    text = re.sub(r'(Advertisement|ADVERTISEMENT|Share this article|Read more:?)[\s]*', '', text)
    return text.strip()


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, filtering out very short ones."""
    cleaned = _clean_text(text)
    
    # Split on double newlines or single newlines followed by uppercase (new paragraph)
    paragraphs = re.split(r'\n\n+|\n(?=[A-Z])', cleaned)
    
    # Filter and clean paragraphs
    result = []
    for p in paragraphs:
        p = p.strip()
        # Keep paragraphs with at least 50 characters (roughly a sentence)
        if len(p) >= 50:
            result.append(p)
    
    return result


def _score_paragraph(
    paragraph: str,
    index: int,
    total_paragraphs: int,
    title: str = ""
) -> float:
    """
    Score a paragraph based on financial relevance.
    
    Scoring factors:
    - Presence of ticker symbols or company names from title
    - Presence of numbers, percentages, financial terms
    - Presence of thesis/conclusion phrases
    - Position bonus for first/last paragraphs
    """
    text_lower = paragraph.lower()
    score = 0.0
    
    # 1. Ticker/Company name presence (from title)
    if title:
        # Check if words from title appear in paragraph
        title_words = set(w.lower() for w in title.split() if len(w) > 3)
        for word in title_words:
            if word in text_lower:
                score += 2.0
                break  # Cap at one match
    
    # Check for ticker patterns
    tickers = TICKER_PATTERN.findall(paragraph)
    if tickers:
        score += min(len(tickers) * 1.5, 4.5)  # Cap at 3 tickers worth
    
    # 2. Numbers and percentages (key financial data)
    numbers = NUMBER_PATTERN.findall(paragraph)
    if numbers:
        score += min(len(numbers) * 1.0, 5.0)  # Cap at 5 numbers worth
    
    # 3. Financial terms
    term_matches = sum(1 for term in FINANCIAL_TERMS if term in text_lower)
    score += min(term_matches * 0.5, 4.0)  # Cap contribution
    
    # 4. Thesis/conclusion phrases
    thesis_matches = sum(1 for phrase in THESIS_PHRASES if phrase in text_lower)
    score += min(thesis_matches * 1.5, 4.5)  # Higher weight for thesis statements
    
    # 5. Position bonus
    if index == 0:  # First paragraph (usually has the lede)
        score += 3.0
    elif index == 1:  # Second paragraph (often key details)
        score += 1.5
    elif index == total_paragraphs - 1:  # Last paragraph (often conclusion)
        score += 2.5
    elif index == total_paragraphs - 2:  # Second to last
        score += 1.0
    
    # Penalty for very short paragraphs
    if len(paragraph) < 100:
        score *= 0.7
    
    return score


def smart_chunk_content(
    content: str,
    title: str = "",
    char_budget: int = 4000,
    min_paragraphs: int = 3,
) -> str:
    """
    Intelligently select the most relevant paragraphs from article content.
    
    Strategy:
    1. Split into paragraphs and score each
    2. Always include first 1-2 and last 1-2 paragraphs
    3. Fill remaining budget with highest-scoring middle paragraphs
    4. Concatenate in original order
    
    Args:
        content: Raw article content.
        title: Article title for context matching.
        char_budget: Maximum characters for output.
        min_paragraphs: Minimum paragraphs to include if available.
    
    Returns:
        Concatenated content string optimized for LLM summarization.
    """
    paragraphs = _split_into_paragraphs(content)
    
    # If content is already short enough, return cleaned version
    if len(content) <= char_budget:
        return _clean_text(content)
    
    # If too few paragraphs, just truncate
    if len(paragraphs) <= min_paragraphs:
        return _clean_text(content)[:char_budget]
    
    total = len(paragraphs)
    
    # Score all paragraphs
    scored: list[ScoredParagraph] = []
    for i, p in enumerate(paragraphs):
        score = _score_paragraph(p, i, total, title)
        scored.append(ScoredParagraph(
            index=i,
            text=p,
            score=score,
            char_count=len(p),
        ))
    
    # Always include first 2 and last 2 paragraphs (if they exist)
    must_include_indices = set()
    if total >= 1:
        must_include_indices.add(0)
    if total >= 2:
        must_include_indices.add(1)
    if total >= 3:
        must_include_indices.add(total - 1)
    if total >= 4:
        must_include_indices.add(total - 2)
    
    # Start with must-include paragraphs
    selected_indices = set(must_include_indices)
    current_chars = sum(scored[i].char_count for i in selected_indices)
    
    # Get remaining paragraphs sorted by score (descending)
    remaining = [s for s in scored if s.index not in selected_indices]
    remaining.sort(key=lambda x: x.score, reverse=True)
    
    # Add highest-scoring paragraphs until budget is reached
    separator_chars = 4  # "\n\n" between paragraphs
    
    for sp in remaining:
        needed = sp.char_count + separator_chars
        if current_chars + needed <= char_budget:
            selected_indices.add(sp.index)
            current_chars += needed
    
    # Sort selected by original index to maintain order
    selected_sorted = sorted(selected_indices)
    
    # Concatenate selected paragraphs
    result_parts = [scored[i].text for i in selected_sorted]
    result = "\n\n".join(result_parts)
    
    # Final trim if slightly over budget
    if len(result) > char_budget:
        result = result[:char_budget].rsplit(' ', 1)[0] + "..."
    
    logger.debug(
        f"Smart chunking: {len(paragraphs)} paragraphs → {len(selected_sorted)} selected "
        f"({len(result)}/{char_budget} chars)"
    )
    
    return result

# Trace logger for detailed LLM I/O logging
_trace_logger: Optional[logging.Logger] = None


def _get_trace_logger(model_name: str) -> logging.Logger:
    """
    Get or create a trace logger that writes to a timestamped file.
    
    Creates a file: logs/{model_name}_trace_{date}_{time}.log
    """
    global _trace_logger
    
    if _trace_logger is not None:
        return _trace_logger
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate timestamped filename
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    # Sanitize model name for filename (replace special chars)
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    log_filename = f"{safe_model_name}_trace_{timestamp}.log"
    log_path = logs_dir / log_filename
    
    # Create dedicated trace logger
    _trace_logger = logging.getLogger(f"llm_trace.{safe_model_name}")
    _trace_logger.setLevel(logging.DEBUG)
    _trace_logger.propagate = False  # Don't propagate to root logger
    
    # File handler with detailed format
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s]\n%(message)s\n",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    _trace_logger.addHandler(file_handler)
    
    logger.info(f"LLM trace logging to: {log_path}")
    
    return _trace_logger


def _log_llm_call(
    trace_logger: logging.Logger,
    article_title: str,
    prompt: str,
    response: str,
    model: str,
    duration_ms: float,
    tokens_eval: Optional[int] = None,
    original_chars: Optional[int] = None,
    chunked_chars: Optional[int] = None,
) -> None:
    """Log a complete LLM call with input and output."""
    separator = "=" * 80
    
    chunking_info = ""
    if original_chars and chunked_chars:
        reduction = ((original_chars - chunked_chars) / original_chars * 100) if original_chars > 0 else 0
        chunking_info = f"CHUNKING: {original_chars} → {chunked_chars} chars ({reduction:.1f}% reduction)"
    
    log_message = f"""
{separator}
ARTICLE: {article_title}
MODEL: {model}
DURATION: {duration_ms:.0f}ms
{f"TOKENS: {tokens_eval}" if tokens_eval else ""}
{chunking_info}
{separator}

>>> INPUT PROMPT >>>
{prompt}

<<< OUTPUT RESPONSE <<<
{response}

{separator}
"""
    trace_logger.debug(log_message)


PROMPT_TEMPLATE = """<role>
You are a Senior Equity Research Analyst for a top-tier investment bank. Your task is to write a high-conviction news brief for institutional clients based on the provided article.
</role>

<instructions>
You must output a SINGLE paragraph containing EXACTLY three sentences. Follow this structure strictly:

1.  **Sentence 1 (The Event):** State clearly who did what, where, and when. Focus on the core transactional or operational update.
2.  **Sentence 2 (The Numbers):** Extract specific metrics (CAPEX, revenue, margins, bps, dates). Do not generalize; use the exact figures from the text.
3.  **Sentence 3 (The Implication):** You MUST start this sentence with "So what?". Connect the metrics to the P&L or stock valuation. Mention one upside driver and one specific risk. End with a theme tag in parentheses.
</instructions>

<constraints>
* **Length:** EXACTLY 3 sentences. No more, no less.
* **Formatting:** Single paragraph. No line breaks between sentences.
* **Tone:** Clinical, skeptical, and professional.
* **Banned Words:** "Exciting," "game-changer," "massive," "whopping," "revolution," "skyrocket," "transformative."
* **Requirement:** Do not output introductory filler (e.g., "Here is the brief"). Start directly with the first sentence.
</constraints>

<example>
Input Article Snippet: Nvidia reported Q4 earnings today. Revenue hit $22.1B, up 265% year-over-year, driven by H100 chip demand. Data Center revenue was $18.4B. However, supply chain constraints remain a bottleneck for H2 delivery.

Output:
Nvidia reported Q4 earnings significantly above consensus, driven by unprecedented demand for H100 accelerators in the data center segment. Revenue surged 265% YoY to $22.1B, with Data Center contributing $18.4B of the total top line. So what? While this confirms the AI infrastructure supercycle supports continued margin expansion, persistent supply chain bottlenecks in H2 could cap near-term volume upside (theme: AI Hardware, Semi-Cap).
</example>

<article>
{content}
</article>

Based on the article above, write the 3-sentence brief.
Remember: Start directly with the news. Ensure the third sentence starts with "So what?".

Brief:"""


class SummarizerError(Exception):
    """Raised when summarization fails."""
    pass


def _clean_summary(text: str) -> str:
    """Remove common LLM preambles and clean up the summary."""
    # Common preambles to strip
    preambles = [
        "here is a concise summary",
        "here's a concise summary", 
        "here is the summary",
        "here's the summary",
        "here is a summary",
        "here's a summary",
        "here is my summary",
        "here's my summary",
        "summary:",
        "here are the key points",
        "here's what you need to know",
        "here is the 3-sentence brief",
        "here's the 3-sentence brief",
        "here is my 3-sentence brief",
        "3-sentence brief:",
    ]
    
    lines = text.strip().split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_lower = line.lower().strip()
        # Skip lines that are just preambles
        skip = False
        for preamble in preambles:
            if line_lower.startswith(preamble) or line_lower == preamble.rstrip(':'):
                skip = True
                break
        if not skip and line.strip():
            cleaned_lines.append(line.strip())
    
    # Join with space, preserving "So what?" as a natural break
    result = ' '.join(cleaned_lines)
    
    # Ensure "So what?" starts on visual break if present
    result = result.replace('. So what?', '.\n\nSo what?')
    
    return result


def summarize_article(
    content: str,
    title: str,
    model: str,
    base_url: str,
    timeout: int = 120,
    char_budget: int = 4000,
) -> str:
    """
    Summarize article content using Ollama.
    
    Uses smart chunking to select the most relevant paragraphs
    rather than simple truncation.
    
    Args:
        content: Article text content to summarize.
        title: Article title for context.
        model: Ollama model name (e.g., "llama3").
        base_url: Ollama API base URL.
        timeout: Request timeout in seconds.
        char_budget: Maximum characters to send to LLM.
    
    Returns:
        Narrative summary text.
    
    Raises:
        SummarizerError: If the Ollama request fails.
    """
    # Get trace logger for this model
    trace_logger = _get_trace_logger(model)
    
    # Track original content size for logging
    original_chars = len(content)
    
    # Use smart chunking to select best paragraphs
    chunked_content = smart_chunk_content(content, title=title, char_budget=char_budget)
    chunked_chars = len(chunked_content)
    
    prompt = PROMPT_TEMPLATE.format(title=title, content=chunked_content)
    
    url = f"{base_url.rstrip('/')}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,  # Lower for consistent, factual output
            "num_predict": 400,  # Allow room for structured 3-sentence format
        }
    }
    
    start_time = datetime.now()
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.Timeout:
        raise SummarizerError(f"Ollama request timed out after {timeout}s")
    except requests.RequestException as e:
        raise SummarizerError(f"Failed to connect to Ollama: {e}")
    
    end_time = datetime.now()
    duration_ms = (end_time - start_time).total_seconds() * 1000
    
    data = response.json()
    
    # Ollama returns response in "response" field
    raw_response = data.get("response", "")
    
    # Get token count if available
    tokens_eval = data.get("eval_count")
    
    # Log the complete LLM call trace with chunking info
    _log_llm_call(
        trace_logger=trace_logger,
        article_title=title,
        prompt=prompt,
        response=raw_response,
        model=model,
        duration_ms=duration_ms,
        tokens_eval=tokens_eval,
        original_chars=original_chars,
        chunked_chars=chunked_chars,
    )
    
    if not raw_response:
        raise SummarizerError("Ollama returned empty response")
    
    # Clean up any preambles the LLM might have added
    return _clean_summary(raw_response)


def is_article_blocked(article: ArticleMeta) -> bool:
    """Check if an article's content is marked as blocked."""
    return article.content == BLOCKED_CONTENT_MARKER


def summarize_articles(
    articles: list[ArticleMeta],
    config: Config
) -> dict[str, Optional[str]]:
    """
    Summarize multiple articles with quality validation.
    
    Skips articles with blocked content. Validates generated summaries
    and drops articles with invalid summaries (e.g., LLM refusals).
    
    Args:
        articles: List of articles to summarize.
        config: Application configuration.
    
    Returns:
        Dictionary mapping article IDs to summary text.
        - Blocked articles get BLOCKED_SUMMARY_FALLBACK.
        - Articles with invalid summaries get None (DROPPED_ARTICLE_MARKER).
    """
    summaries: dict[str, Optional[str]] = {}
    blocked_count = 0
    dropped_count = 0
    
    for article in articles:
        # Skip blocked content - don't call LLM
        if is_article_blocked(article):
            logger.warning(f"Skipping blocked article (no LLM call): {article.title[:50]}...")
            summaries[article.id] = BLOCKED_SUMMARY_FALLBACK
            blocked_count += 1
            continue
        
        content = article.content or article.summary or article.title
        
        try:
            summary = summarize_article(
                content=content,
                title=article.title,
                model=config.ollama_model,
                base_url=config.ollama_base_url,
            )
            
            # Validate the generated summary
            validation = validate_summary(
                summary=summary,
                model=config.ollama_model,
                base_url=config.ollama_base_url,
                use_llm_fallback=True,
            )
            
            if not validation.is_valid:
                logger.warning(
                    f"Dropped article (invalid summary): {article.title[:50]}... "
                    f"Reason: {validation.reason}"
                )
                summaries[article.id] = DROPPED_ARTICLE_MARKER
                dropped_count += 1
                continue
            
            summaries[article.id] = summary
            logger.info(f"Summarized: {article.title[:50]}...")
            
        except SummarizerError as e:
            logger.error(f"Failed to summarize '{article.title}': {e}")
            # Mark as dropped instead of providing fallback
            summaries[article.id] = DROPPED_ARTICLE_MARKER
            dropped_count += 1
    
    if blocked_count > 0:
        logger.info(f"Skipped {blocked_count} blocked article(s)")
    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} article(s) due to invalid summaries")
    
    return summaries
