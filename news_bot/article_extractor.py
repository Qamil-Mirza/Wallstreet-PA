"""
Article content extraction.

Ensures articles have usable content by scraping URLs when needed.
Uses trafilatura for robust extraction with BeautifulSoup as fallback.
Detects blocked/paywalled content and marks it appropriately.
"""

import logging
import re
from typing import Optional

import requests
import trafilatura
from bs4 import BeautifulSoup

from .news_client import ArticleMeta


logger = logging.getLogger(__name__)


# Browser-like headers to avoid bot detection
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}


# Marker for blocked content - used to signal that LLM should not be called
BLOCKED_CONTENT_MARKER = "[[BLOCKED_CONTENT]]"

# Phrases that indicate a block/paywall page
BLOCK_PHRASES = [
    # JavaScript/Cookie blocks
    "please enable javascript",
    "javascript is required",
    "javascript is disabled",
    "please enable cookies",
    "cookies are required",
    "enable javascript and cookies",
    # Ad blocker detection
    "ad-blocker enabled",
    "ad blocker enabled",
    "adblocker enabled",
    "ad-blocker detected",
    "ad blocker detected",
    "adblocker detected",
    "disable your ad blocker",
    "disable your adblocker",
    "you may be blocked from",
    "please disable your ad",
    "please whitelist",
    # Paywalls
    "subscribe to continue",
    "subscribe to read",
    "subscription required",
    "subscribers only",
    "member-only content",
    "premium content",
    "unlock this article",
    "to continue reading",
    "sign in to read",
    "log in to continue",
    "create a free account to",
    # Bot detection
    "are you a robot",
    "are you human",
    "captcha",
    "verify you are human",
    "access denied",
    "access to this page has been denied",
    "forbidden",
    # Region blocks
    "not available in your region",
    "not available in your country",
    "content is not available",
    # Rate limiting
    "too many requests",
    "rate limit exceeded",
]


class ExtractionError(Exception):
    """Raised when article content extraction fails."""
    pass


def is_blocked_content(text: str) -> bool:
    """
    Detect if the extracted text is from a block/paywall page.
    
    Args:
        text: The extracted text content.
    
    Returns:
        True if the content appears to be from a blocked page.
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Check for block phrases
    for phrase in BLOCK_PHRASES:
        if phrase in text_lower:
            logger.debug(f"Detected block phrase: '{phrase}'")
            return True
    
    # Additional heuristics:
    # Very short content with common block indicators
    if len(text) < 500:
        short_indicators = [
            "javascript", "cookies", "subscribe", "sign in",
            "log in", "blocked", "denied", "robot"
        ]
        matches = sum(1 for ind in short_indicators if ind in text_lower)
        if matches >= 2:
            logger.debug(f"Short content with {matches} block indicators")
            return True
    
    return False


def _extract_with_trafilatura(html: str, url: str = None) -> Optional[str]:
    """
    Extract article text using trafilatura (primary method).
    
    Trafilatura is specifically designed for news/article extraction
    and handles boilerplate removal, ads, and navigation automatically.
    
    Args:
        html: Raw HTML content.
        url: Optional URL for better extraction context.
    
    Returns:
        Extracted article text or None if extraction fails.
    """
    try:
        # Use trafilatura's extract function with optimal settings for news
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            no_fallback=False,  # Allow fallback methods
            favor_precision=True,  # Prefer quality over quantity
            deduplicate=True,
        )
        
        if text and len(text.strip()) > 50:
            logger.debug(f"Trafilatura extracted {len(text)} chars")
            return text.strip()
        
        return None
        
    except Exception as e:
        logger.debug(f"Trafilatura extraction failed: {e}")
        return None


def _extract_with_beautifulsoup(html: str) -> str:
    """
    Extract readable text from HTML using BeautifulSoup (fallback method).
    
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


def _extract_text_from_html(html: str, url: str = None) -> str:
    """
    Extract readable text from HTML content.
    
    Uses trafilatura as the primary extraction method with
    BeautifulSoup as a fallback.
    
    Args:
        html: Raw HTML content.
        url: Optional URL for better extraction context.
    
    Returns:
        Extracted article text.
    """
    # Try trafilatura first (best for news articles)
    text = _extract_with_trafilatura(html, url)
    
    if text:
        return text
    
    # Fallback to BeautifulSoup
    logger.debug("Falling back to BeautifulSoup extraction")
    return _extract_with_beautifulsoup(html)


def _fetch_and_extract(url: str, timeout: int = 20) -> tuple[Optional[str], bool]:
    """
    Fetch a URL and extract its text content.
    
    Uses browser-like headers to avoid bot detection and trafilatura
    for robust article extraction.
    
    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.
    
    Returns:
        Tuple of (extracted text content, is_blocked).
        - If content is blocked, returns (None, True).
        - If extraction fails, returns (None, False).
        - If successful, returns (content, False).
    """
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()
        
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type.lower():
            return None, False
        
        html = response.text
        
        # Check for block indicators in raw HTML first (before extraction)
        # Some sites return a block page with very little extractable content
        html_lower = html.lower()
        if "captcha" in html_lower or "robot" in html_lower:
            if is_blocked_content(html[:2000]):  # Check first 2KB
                logger.warning(f"Detected block page in raw HTML at {url}")
                return None, True
        
        # Extract text using trafilatura (with BeautifulSoup fallback)
        text = _extract_text_from_html(html, url)
        
        if not text:
            logger.debug(f"No content extracted from {url}")
            return None, False
        
        # Check if extracted content is a block page
        if is_blocked_content(text):
            logger.warning(f"Detected blocked content at {url}")
            return None, True
        
        logger.info(f"Extracted {len(text)} chars from {url[:50]}...")
        return text, False
        
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None, False
    except Exception as e:
        logger.warning(f"Failed to extract content from {url}: {e}")
        return None, False


def ensure_article_content(article: ArticleMeta) -> ArticleMeta:
    """
    Ensure an article has usable content.
    
    If the article already has content, checks it for block pages.
    Otherwise, attempts to scrape content from the article URL.
    
    Blocked content is marked with BLOCKED_CONTENT_MARKER so downstream
    processing (e.g., summarizer) can skip it.
    
    Args:
        article: The article to process.
    
    Returns:
        ArticleMeta with content field populated (if possible).
        Content will be BLOCKED_CONTENT_MARKER if site is blocked.
    """
    # Check if we already have sufficient content
    if article.content and len(article.content.strip()) > 100:
        # Still check existing content for block pages
        if is_blocked_content(article.content):
            logger.warning(f"Existing content for '{article.title[:50]}' appears blocked")
            return ArticleMeta(
                id=article.id,
                title=article.title,
                url=article.url,
                summary=article.summary,
                content=BLOCKED_CONTENT_MARKER,
                published_at=article.published_at,
                source=article.source,
            )
        return article
    
    # Try to extract content from URL
    extracted, is_blocked = _fetch_and_extract(article.url)
    
    if is_blocked:
        logger.info(f"Article blocked: {article.title[:50]}...")
        return ArticleMeta(
            id=article.id,
            title=article.title,
            url=article.url,
            summary=article.summary,
            content=BLOCKED_CONTENT_MARKER,
            published_at=article.published_at,
            source=article.source,
        )
    
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
        # Check summary for block content too
        if is_blocked_content(article.summary):
            return ArticleMeta(
                id=article.id,
                title=article.title,
                url=article.url,
                summary=article.summary,
                content=BLOCKED_CONTENT_MARKER,
                published_at=article.published_at,
                source=article.source,
            )
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
