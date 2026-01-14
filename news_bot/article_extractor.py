"""
Article content extraction.

Ensures articles have usable content by scraping URLs when needed.
"""

import logging
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .news_client import ArticleMeta


logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when article content extraction fails."""
    pass


def _extract_text_from_html(html: str) -> str:
    """
    Extract readable text from HTML content.
    
    Focuses on main content areas and paragraph text.
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script, style, nav, header, footer elements
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()
    
    # Try to find main content area
    main_content = (
        soup.find("article") or
        soup.find("main") or
        soup.find(class_=lambda x: x and ("article" in x.lower() or "content" in x.lower())) or
        soup.find("div", class_="post") or
        soup.body
    )
    
    if not main_content:
        return ""
    
    # Extract text from paragraphs
    paragraphs = main_content.find_all("p")
    
    if paragraphs:
        text_parts = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 30:  # Filter out very short paragraphs
                text_parts.append(text)
        return "\n\n".join(text_parts)
    
    # Fallback: get all text
    return main_content.get_text(separator="\n", strip=True)


def _fetch_and_extract(url: str, timeout: int = 15) -> Optional[str]:
    """
    Fetch a URL and extract its text content.
    
    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.
    
    Returns:
        Extracted text content or None if extraction fails.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type.lower():
            return None
        
        text = _extract_text_from_html(response.text)
        return text if text else None
        
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to extract content from {url}: {e}")
        return None


def ensure_article_content(article: ArticleMeta) -> ArticleMeta:
    """
    Ensure an article has usable content.
    
    If the article already has content, returns it unchanged.
    Otherwise, attempts to scrape content from the article URL.
    
    Args:
        article: The article to process.
    
    Returns:
        ArticleMeta with content field populated (if possible).
    """
    # Check if we already have sufficient content
    if article.content and len(article.content.strip()) > 100:
        return article
    
    # Try to extract content from URL
    extracted = _fetch_and_extract(article.url)
    
    if extracted and len(extracted) > 100:
        return ArticleMeta(
            id=article.id,
            title=article.title,
            url=article.url,
            summary=article.summary,
            content=extracted,
            published_at=article.published_at,
            source=article.source,
        )
    
    # If extraction fails but we have a summary, use that as content
    if article.summary and len(article.summary.strip()) > 50:
        return ArticleMeta(
            id=article.id,
            title=article.title,
            url=article.url,
            summary=article.summary,
            content=article.summary,
            published_at=article.published_at,
            source=article.source,
        )
    
    # Return original if all else fails
    return article


def ensure_batch_content(articles: list[ArticleMeta]) -> list[ArticleMeta]:
    """
    Ensure all articles in a batch have content.
    
    Applies ensure_article_content to each article with error handling.
    
    Args:
        articles: List of articles to process.
    
    Returns:
        List of articles with content fields populated where possible.
    """
    results = []
    
    for article in articles:
        try:
            processed = ensure_article_content(article)
            results.append(processed)
        except Exception as e:
            logger.error(f"Failed to process article {article.id}: {e}")
            results.append(article)
    
    return results
