"""
News API client for fetching financial/economic articles.

Supports multiple free news APIs with normalization to a common schema.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin

import requests

from .config import Config


logger = logging.getLogger(__name__)


def _sanitize_error(error: Exception) -> str:
    """Remove API keys from error messages to prevent logging secrets."""
    error_str = str(error)
    error_str = re.sub(r'(apikey|api_key|api_token|token|key)=[^&\s]+', r'\1=***', error_str, flags=re.IGNORECASE)
    return error_str


@dataclass
class ArticleMeta:
    """Normalized article metadata."""
    
    id: str
    title: str
    url: str
    summary: Optional[str]
    content: Optional[str]
    published_at: datetime
    source: Optional[str]


class NewsClientError(Exception):
    """Raised when news API requests fail."""
    pass


def _parse_datetime(dt_string: str) -> datetime:
    """Parse datetime string from various API formats."""
    # Try common formats
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(dt_string, fmt)
        except ValueError:
            continue
    
    # Fallback: return current time if parsing fails
    return datetime.now()


def _normalize_marketaux_article(raw: dict) -> Optional[ArticleMeta]:
    """Normalize MarketAux API response to ArticleMeta."""
    title = raw.get("title")
    url = raw.get("url")
    
    if not title or not url:
        return None
    
    return ArticleMeta(
        id=raw.get("uuid", url),
        title=title,
        url=url,
        summary=raw.get("description"),
        content=raw.get("snippet") or raw.get("description"),
        published_at=_parse_datetime(raw.get("published_at", "")),
        source=raw.get("source"),
    )


def _normalize_fmp_article(raw: dict) -> Optional[ArticleMeta]:
    """Normalize Financial Modeling Prep API response to ArticleMeta."""
    title = raw.get("title")
    url = raw.get("url")
    
    if not title or not url:
        return None
    
    return ArticleMeta(
        id=url,  # FMP doesn't provide unique IDs
        title=title,
        url=url,
        summary=raw.get("text", "")[:500] if raw.get("text") else None,
        content=raw.get("text"),
        published_at=_parse_datetime(raw.get("publishedDate", "")),
        source=raw.get("site"),
    )


def fetch_recent_articles(config: Config, limit: int = 30) -> list[ArticleMeta]:
    """
    Fetch recent financial news articles from the configured API.
    
    Args:
        config: Application configuration with API credentials.
        limit: Maximum number of articles to fetch.
    
    Returns:
        List of normalized ArticleMeta objects.
    
    Raises:
        NewsClientError: If the API request fails.
    """
    base_url = config.news_api_base_url.rstrip("/")
    
    # Detect API type based on base URL
    if "marketaux" in base_url:
        return _fetch_marketaux(config, limit)
    elif "financialmodelingprep" in base_url:
        return _fetch_fmp(config, limit)
    else:
        # Default to MarketAux-style API
        return _fetch_marketaux(config, limit)


def _fetch_marketaux(config: Config, limit: int) -> list[ArticleMeta]:
    """Fetch articles from MarketAux API."""
    url = f"{config.news_api_base_url.rstrip('/')}/news/all"
    
    params = {
        "api_token": config.news_api_key,
        "limit": limit,
        "language": "en",
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise NewsClientError(f"Failed to fetch from MarketAux: {_sanitize_error(e)}")
    
    data = response.json()
    
    # Log API metadata
    meta = data.get("meta", {})
    returned = meta.get("returned", 0)
    found = meta.get("found", 0)
    logger.info(f"MarketAux API: returned {returned} of {found} articles")
    
    if returned < limit and found > returned:
        logger.warning(f"⚠️  Free tier limited to {returned} articles per request")
    
    raw_articles = data.get("data", [])
    
    articles = []
    for raw in raw_articles:
        article = _normalize_marketaux_article(raw)
        if article:
            articles.append(article)
    
    return articles


def _fetch_fmp(config: Config, limit: int) -> list[ArticleMeta]:
    """Fetch articles from Financial Modeling Prep API."""
    url = f"{config.news_api_base_url.rstrip('/')}/api/v3/stock_news"
    
    params = {
        "apikey": config.news_api_key,
        "limit": limit,
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise NewsClientError(f"Failed to fetch from FMP: {_sanitize_error(e)}")
    
    raw_articles = response.json()
    
    if not isinstance(raw_articles, list):
        raw_articles = []
    
    articles = []
    for raw in raw_articles:
        article = _normalize_fmp_article(raw)
        if article:
            articles.append(article)
    
    return articles
