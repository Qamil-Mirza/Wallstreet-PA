"""
Article summarization using Ollama (local LLM).

Generates analyst-style narrative summaries of articles.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from .config import Config
from .news_client import ArticleMeta


logger = logging.getLogger(__name__)

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
) -> None:
    """Log a complete LLM call with input and output."""
    separator = "=" * 80
    
    log_message = f"""
{separator}
ARTICLE: {article_title}
MODEL: {model}
DURATION: {duration_ms:.0f}ms
{f"TOKENS: {tokens_eval}" if tokens_eval else ""}
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
    timeout: int = 120
) -> str:
    """
    Summarize article content using Ollama.
    
    Args:
        content: Article text content to summarize.
        title: Article title for context.
        model: Ollama model name (e.g., "llama3").
        base_url: Ollama API base URL.
        timeout: Request timeout in seconds.
    
    Returns:
        Narrative summary text.
    
    Raises:
        SummarizerError: If the Ollama request fails.
    """
    # Get trace logger for this model
    trace_logger = _get_trace_logger(model)
    
    prompt = PROMPT_TEMPLATE.format(title=title, content=content[:4000])
    
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
    
    # Log the complete LLM call trace
    _log_llm_call(
        trace_logger=trace_logger,
        article_title=title,
        prompt=prompt,
        response=raw_response,
        model=model,
        duration_ms=duration_ms,
        tokens_eval=tokens_eval,
    )
    
    if not raw_response:
        raise SummarizerError("Ollama returned empty response")
    
    # Clean up any preambles the LLM might have added
    return _clean_summary(raw_response)


def summarize_articles(
    articles: list[ArticleMeta],
    config: Config
) -> dict[str, str]:
    """
    Summarize multiple articles.
    
    Args:
        articles: List of articles to summarize.
        config: Application configuration.
    
    Returns:
        Dictionary mapping article IDs to summary text.
    """
    summaries: dict[str, str] = {}
    
    for article in articles:
        content = article.content or article.summary or article.title
        
        try:
            summary = summarize_article(
                content=content,
                title=article.title,
                model=config.ollama_model,
                base_url=config.ollama_base_url,
            )
            summaries[article.id] = summary
            logger.info(f"Summarized: {article.title[:50]}...")
            
        except SummarizerError as e:
            logger.error(f"Failed to summarize '{article.title}': {e}")
            # Provide fallback summary as narrative
            fallback = article.title
            if article.summary:
                fallback += f" {article.summary[:200]}"
            summaries[article.id] = fallback
    
    return summaries
