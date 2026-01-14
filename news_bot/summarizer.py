"""
Article summarization using Ollama (local LLM).

Generates analyst-style narrative summaries of articles.
"""

import logging
from typing import Optional

import requests

from .config import Config
from .news_client import ArticleMeta


logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """You are a senior Wall Street research analyst briefing a client over the phone. Summarize this news in 2-3 flowing sentences as if you're speaking directly to them.

Rules:
- Start immediately with the key news. No preamble like "Here's a summary" or "This article discusses"
- Write in a conversational but professional tone, like you're explaining to a smart colleague
- Include specific numbers (percentages, dollar amounts, dates) when available
- End with the "so what" - why this matters for markets or investors
- Keep it tight: 2-3 sentences max, no fluff

Article: {title}

{content}

Your briefing (start directly with the news):"""


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
    
    return ' '.join(cleaned_lines)


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
    prompt = PROMPT_TEMPLATE.format(title=title, content=content[:4000])
    
    url = f"{base_url.rstrip('/')}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,  # Slightly higher for natural flow
            "num_predict": 300,  # Shorter for concise narrative
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.Timeout:
        raise SummarizerError(f"Ollama request timed out after {timeout}s")
    except requests.RequestException as e:
        raise SummarizerError(f"Failed to connect to Ollama: {e}")
    
    data = response.json()
    
    # Ollama returns response in "response" field
    summary = data.get("response", "")
    
    if not summary:
        raise SummarizerError("Ollama returned empty response")
    
    # Clean up any preambles the LLM might have added
    return _clean_summary(summary)


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
