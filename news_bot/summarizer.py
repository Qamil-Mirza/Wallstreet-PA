"""
Article summarization using Ollama (local LLM).

Generates analyst-style bullet summaries of articles.
"""

import logging
from typing import Optional

import requests

from .config import Config
from .news_client import ArticleMeta


logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """You are a senior Wall Street research analyst writing a morning briefing. 
Summarize the following news article in 4-6 concise bullet points.

Guidelines:
- Be direct and factual, like a Bloomberg terminal headline
- Lead with the key news, then supporting details
- Include specific numbers (percentages, dollar amounts) when available
- Note market implications if relevant
- Skip generic filler; every bullet should add value

Article Title: {title}

Article Content:
{content}

Provide your summary as bullet points (use • for bullets):"""


class SummarizerError(Exception):
    """Raised when summarization fails."""
    pass


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
        Bullet-point summary text.
    
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
            "temperature": 0.3,  # Lower temp for more focused output
            "num_predict": 500,  # Limit response length
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
    
    return summary.strip()


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
            # Provide fallback summary
            summaries[article.id] = f"• {article.title}"
            if article.summary:
                summaries[article.id] += f"\n• {article.summary[:200]}"
    
    return summaries
