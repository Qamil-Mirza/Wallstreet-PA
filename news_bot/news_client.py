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


@dataclass
class NewsFeedConfig:
    """Configuration for a specific news feed query."""
    
    name: str
    config_key: str  # Maps to Config.section_{key}_enabled
    countries: Optional[str] = None  # Comma-separated country codes (e.g., "us", "my")
    industries: Optional[str] = None  # Comma-separated industries (e.g., "Technology", "Industrials")
    limit: int = 10


# Default feed configurations for multi-region news fetching
ALL_FEEDS = [
    NewsFeedConfig(name="World News", config_key="world", limit=10),
    NewsFeedConfig(name="US Tech", config_key="us_tech", countries="us", industries="Technology", limit=10),
    NewsFeedConfig(name="US Industry", config_key="us_industry", countries="us", industries="Industrials", limit=10),
    NewsFeedConfig(name="Malaysia Tech", config_key="malaysia_tech", countries="my", industries="Technology", limit=10),
    NewsFeedConfig(name="Malaysia Industry", config_key="malaysia_industry", countries="my", industries="Industrials", limit=10),
]


def _get_enabled_feeds(config: Config) -> list[NewsFeedConfig]:
    """Filter feeds based on feature flags in config."""
    feed_flags = {
        "world": config.section_world_enabled,
        "us_tech": config.section_us_tech_enabled,
        "us_industry": config.section_us_industry_enabled,
        "malaysia_tech": config.section_malaysia_tech_enabled,
        "malaysia_industry": config.section_malaysia_industry_enabled,
    }
    
    enabled = [feed for feed in ALL_FEEDS if feed_flags.get(feed.config_key, True)]
    
    if not enabled:
        logger.warning("No sections enabled! Falling back to World News.")
        return [ALL_FEEDS[0]]  # Fallback to World News
    
    logger.info(f"Enabled sections: {[f.name for f in enabled]}")
    return enabled


def fetch_recent_articles(config: Config, limit: int = 30) -> list[ArticleMeta]:
    """
    Fetch recent financial news articles from multiple feeds.
    
    Makes multiple API calls to get diverse news coverage:
    - Broad world news
    - US Tech and Industry news
    - Malaysia Tech and Industry news
    
    Args:
        config: Application configuration with API credentials.
        limit: Maximum total articles to return (articles per feed = limit // num_feeds).
    
    Returns:
        List of normalized ArticleMeta objects, deduplicated by URL.
    
    Raises:
        NewsClientError: If all API requests fail.
    """
    base_url = config.news_api_base_url.rstrip("/")
    
    # Detect API type based on base URL
    if "marketaux" in base_url:
        return _fetch_marketaux_multi(config, limit)
    elif "financialmodelingprep" in base_url:
        return _fetch_fmp(config, limit)
    else:
        # Default to MarketAux-style API
        return _fetch_marketaux_multi(config, limit)


def _fetch_marketaux_multi(config: Config, total_limit: int) -> list[ArticleMeta]:
    """Fetch articles from MarketAux API using multiple feed configurations."""
    all_articles: list[ArticleMeta] = []
    seen_urls: set[str] = set()
    errors: list[str] = []
    
    enabled_feeds = _get_enabled_feeds(config)
    
    # Calculate per-feed limit (distribute evenly, minimum 3 per feed)
    per_feed_limit = max(3, total_limit // len(enabled_feeds))
    
    for feed in enabled_feeds:
        try:
            articles = _fetch_marketaux_single(
                config=config,
                limit=per_feed_limit,
                countries=feed.countries,
                industries=feed.industries,
                feed_name=feed.name,
            )
            
            # Deduplicate by URL
            for article in articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    all_articles.append(article)
                    
        except NewsClientError as e:
            errors.append(f"{feed.name}: {e}")
            logger.warning(f"Failed to fetch {feed.name}: {e}")
            continue
    
    if not all_articles and errors:
        raise NewsClientError(f"All feeds failed: {'; '.join(errors)}")
    
    logger.info(f"Total unique articles fetched: {len(all_articles)} from {len(enabled_feeds)} feeds")
    return all_articles


def fetch_articles_by_section(config: Config, per_section_limit: int = 5) -> dict[str, list[ArticleMeta]]:
    """
    Fetch articles organized by feed section.
    
    Makes multiple API calls to get diverse news coverage, returning articles
    grouped by their feed section for sectioned email rendering.
    
    Args:
        config: Application configuration with API credentials.
        per_section_limit: Maximum articles per section.
    
    Returns:
        Dictionary mapping section names to lists of ArticleMeta objects.
        Sections: "World News", "US Tech", "US Industry", "Malaysia Tech", "Malaysia Industry"
    
    Raises:
        NewsClientError: If all API requests fail.
    """
    base_url = config.news_api_base_url.rstrip("/")
    
    # Only MarketAux supports sectioned fetching
    if "marketaux" not in base_url:
        # Fallback: put all articles in "World News" section
        articles = fetch_recent_articles(config, limit=per_section_limit * 5)
        return {"World News": articles}
    
    sections: dict[str, list[ArticleMeta]] = {}
    seen_urls: set[str] = set()
    errors: list[str] = []
    
    enabled_feeds = _get_enabled_feeds(config)
    
    for feed in enabled_feeds:
        try:
            articles = _fetch_marketaux_single(
                config=config,
                limit=per_section_limit,
                countries=feed.countries,
                industries=feed.industries,
                feed_name=feed.name,
            )
            
            # Deduplicate by URL within and across sections
            unique_articles = []
            for article in articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    unique_articles.append(article)
            
            sections[feed.name] = unique_articles
            logger.info(f"Section '{feed.name}': {len(unique_articles)} articles")
                    
        except NewsClientError as e:
            errors.append(f"{feed.name}: {e}")
            logger.warning(f"Failed to fetch {feed.name}: {e}")
            sections[feed.name] = []  # Empty section on failure
            continue
    
    # Check if we got any articles at all
    total_articles = sum(len(arts) for arts in sections.values())
    if total_articles == 0 and errors:
        raise NewsClientError(f"All feeds failed: {'; '.join(errors)}")
    
    logger.info(f"Total articles fetched: {total_articles} across {len(sections)} sections")
    return sections


def _fetch_marketaux_single(
    config: Config,
    limit: int,
    countries: Optional[str] = None,
    industries: Optional[str] = None,
    feed_name: str = "default",
) -> list[ArticleMeta]:
    """Fetch articles from MarketAux API with specific filters."""
    url = f"{config.news_api_base_url.rstrip('/')}/news/all"
    
    params = {
        "api_token": config.news_api_key,
        "limit": limit,
        "language": "en",
    }
    
    # Add optional filters
    if countries:
        params["countries"] = countries
    if industries:
        params["industries"] = industries
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise NewsClientError(f"Failed to fetch from MarketAux ({feed_name}): {_sanitize_error(e)}")
    
    data = response.json()
    
    # Check for API errors
    if "error" in data:
        error_msg = data["error"].get("message", "Unknown error")
        raise NewsClientError(f"MarketAux API error ({feed_name}): {error_msg}")
    
    # Log API metadata
    meta = data.get("meta", {})
    returned = meta.get("returned", 0)
    found = meta.get("found", 0)
    logger.info(f"MarketAux [{feed_name}]: returned {returned} of {found} articles")
    
    if returned < limit and found > returned:
        logger.warning(f"⚠️  {feed_name}: Free tier limited to {returned} articles per request")
    
    raw_articles = data.get("data", [])
    
    articles = []
    for raw in raw_articles:
        article = _normalize_marketaux_article(raw)
        if article:
            articles.append(article)
    
    return articles


def _fetch_marketaux(config: Config, limit: int) -> list[ArticleMeta]:
    """Fetch articles from MarketAux API (single call, for backwards compatibility)."""
    return _fetch_marketaux_single(config, limit, feed_name="default")


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
