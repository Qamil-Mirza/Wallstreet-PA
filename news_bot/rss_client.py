"""
RSS feed client for fetching articles from any RSS/Atom feed.

Provides a flexible way to add custom news sources beyond MarketAux.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional
from xml.etree import ElementTree

import requests

from .news_client import ArticleMeta


logger = logging.getLogger(__name__)


# Common RSS namespaces
NAMESPACES = {
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "atom": "http://www.w3.org/2005/Atom",
    "media": "http://search.yahoo.com/mrss/",
}

# Browser-like headers to avoid bot detection
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


class RSSClientError(Exception):
    """Raised when RSS feed fetching or parsing fails."""
    pass


@dataclass
class RSSFeedConfig:
    """Configuration for a single RSS feed."""
    
    name: str  # Display name for the feed (e.g., "Reuters Finance")
    url: str  # RSS feed URL
    section: str = "RSS Feeds"  # Section name for email grouping
    enabled: bool = True
    limit: int = 10  # Max articles to fetch from this feed


def _generate_article_id(url: str) -> str:
    """Generate a unique ID from URL using hash."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _parse_rss_date(date_string: Optional[str]) -> datetime:
    """
    Parse various RSS date formats.
    
    Handles:
    - RFC 822 (common in RSS 2.0): "Mon, 15 Jan 2024 10:30:00 GMT"
    - ISO 8601 (common in Atom): "2024-01-15T10:30:00Z"
    - Custom formats from various feeds
    """
    if not date_string:
        return datetime.now()
    
    date_string = date_string.strip()
    
    # Try RFC 822 format (email.utils handles this well)
    try:
        return parsedate_to_datetime(date_string)
    except (TypeError, ValueError):
        pass
    
    # Try various ISO 8601 formats
    iso_formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    
    for fmt in iso_formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    # Fallback to current time
    logger.debug(f"Could not parse RSS date: {date_string}")
    return datetime.now()


def _clean_html(text: Optional[str]) -> str:
    """Remove HTML tags from text content."""
    if not text:
        return ""
    
    # Remove CDATA markers
    text = re.sub(r'<!\[CDATA\[|\]\]>', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Decode common HTML entities
    entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&nbsp;': ' ',
        '&mdash;': '—',
        '&ndash;': '–',
        '&ldquo;': '"',
        '&rdquo;': '"',
        '&lsquo;': "'",
        '&rsquo;': "'",
    }
    for entity, char in entities.items():
        text = text.replace(entity, char)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def _get_element_text(element: Optional[ElementTree.Element], default: str = "") -> str:
    """Safely get text content from an XML element."""
    if element is None:
        return default
    return element.text or default


def _parse_rss_item(item: ElementTree.Element) -> Optional[ArticleMeta]:
    """
    Parse an RSS 2.0 <item> element into ArticleMeta.
    
    RSS 2.0 structure:
    <item>
        <title>Article Title</title>
        <link>https://example.com/article</link>
        <description>Article summary...</description>
        <pubDate>Mon, 15 Jan 2024 10:30:00 GMT</pubDate>
        <dc:creator>Author Name</dc:creator>
        <content:encoded><![CDATA[Full content...]]></content:encoded>
    </item>
    """
    title = _get_element_text(item.find("title"))
    link = _get_element_text(item.find("link"))
    
    if not title or not link:
        return None
    
    # Get description/summary
    description = _get_element_text(item.find("description"))
    
    # Try to get full content from content:encoded
    content = None
    content_encoded = item.find("content:encoded", NAMESPACES)
    if content_encoded is not None and content_encoded.text:
        content = _clean_html(content_encoded.text)
    
    # Fall back to description for content
    if not content:
        content = _clean_html(description)
    
    # Get publication date
    pub_date = _get_element_text(item.find("pubDate"))
    if not pub_date:
        # Try dc:date
        dc_date = item.find("dc:date", NAMESPACES)
        pub_date = _get_element_text(dc_date)
    
    # Get source/author
    source = None
    dc_creator = item.find("dc:creator", NAMESPACES)
    if dc_creator is not None:
        source = _get_element_text(dc_creator)
    if not source:
        source = _get_element_text(item.find("source"))
    
    # Generate ID from link (GUID not always present)
    guid_elem = item.find("guid")
    article_id = _get_element_text(guid_elem) if guid_elem is not None else _generate_article_id(link)
    
    return ArticleMeta(
        id=article_id,
        title=_clean_html(title),
        url=link,
        summary=_clean_html(description),
        content=content,
        published_at=_parse_rss_date(pub_date),
        source=source,
    )


def _find_atom_element(entry: ElementTree.Element, tag_name: str, ns: dict) -> Optional[ElementTree.Element]:
    """Find an Atom element, trying both namespaced and non-namespaced versions."""
    # Try with explicit namespace prefix
    elem = entry.find(f"atom:{tag_name}", ns)
    if elem is not None:
        return elem
    
    # Try with full namespace URI
    elem = entry.find(f"{{http://www.w3.org/2005/Atom}}{tag_name}")
    if elem is not None:
        return elem
    
    # Try without namespace
    return entry.find(tag_name)


def _parse_atom_entry(entry: ElementTree.Element, ns: dict) -> Optional[ArticleMeta]:
    """
    Parse an Atom <entry> element into ArticleMeta.
    
    Atom structure:
    <entry>
        <title>Article Title</title>
        <link href="https://example.com/article" />
        <summary>Article summary...</summary>
        <content type="html">Full content...</content>
        <published>2024-01-15T10:30:00Z</published>
        <author><name>Author Name</name></author>
    </entry>
    """
    # Get title
    title_elem = _find_atom_element(entry, "title", ns)
    title = _get_element_text(title_elem)
    
    # Get link (Atom uses href attribute)
    link = None
    link_elem = _find_atom_element(entry, "link", ns)
    if link_elem is not None:
        link = link_elem.get("href", "")
        if not link:
            link = _get_element_text(link_elem)
    
    if not title or not link:
        return None
    
    # Get summary
    summary_elem = _find_atom_element(entry, "summary", ns)
    summary = _clean_html(_get_element_text(summary_elem))
    
    # Get content
    content_elem = _find_atom_element(entry, "content", ns)
    content = _clean_html(_get_element_text(content_elem)) if content_elem is not None else summary
    
    # Get publication date (prefer published, fall back to updated)
    pub_elem = _find_atom_element(entry, "published", ns)
    if pub_elem is None:
        pub_elem = _find_atom_element(entry, "updated", ns)
    pub_date = _get_element_text(pub_elem)
    
    # Get author
    source = None
    author_elem = _find_atom_element(entry, "author", ns)
    if author_elem is not None:
        name_elem = _find_atom_element(author_elem, "name", ns)
        source = _get_element_text(name_elem)
    
    # Get ID
    id_elem = _find_atom_element(entry, "id", ns)
    article_id = _get_element_text(id_elem) if id_elem is not None else _generate_article_id(link)
    
    return ArticleMeta(
        id=article_id,
        title=_clean_html(title),
        url=link,
        summary=summary,
        content=content or summary,
        published_at=_parse_rss_date(pub_date),
        source=source,
    )


def _detect_feed_type(root: ElementTree.Element) -> str:
    """Detect whether the feed is RSS 2.0, RSS 1.0, or Atom."""
    tag = root.tag.lower()
    
    # Remove namespace prefix if present
    if "}" in tag:
        tag = tag.split("}")[-1]
    
    if tag == "rss":
        return "rss2"
    elif tag == "rdf" or "rdf" in root.tag.lower():
        return "rss1"
    elif tag == "feed":
        return "atom"
    
    # Check for channel element (RSS indicator)
    if root.find("channel") is not None:
        return "rss2"
    
    # Check for entry element (Atom indicator)
    if root.find("entry") is not None or root.find("{http://www.w3.org/2005/Atom}entry") is not None:
        return "atom"
    
    return "unknown"


def _get_feed_source(root: ElementTree.Element, feed_type: str) -> Optional[str]:
    """Extract the feed source/title from the feed metadata."""
    if feed_type in ("rss2", "rss1"):
        channel = root.find("channel") or root.find(".//channel")
        if channel is not None:
            title_elem = channel.find("title")
            if title_elem is not None and title_elem.text:
                return title_elem.text.strip()
    elif feed_type == "atom":
        # Try with namespace prefix
        title_elem = root.find("atom:title", NAMESPACES)
        if title_elem is None:
            # Try with full namespace URI
            title_elem = root.find("{http://www.w3.org/2005/Atom}title")
        if title_elem is None:
            # Try without namespace
            title_elem = root.find("title")
        if title_elem is not None and title_elem.text:
            return title_elem.text.strip()
    return None


def parse_feed(xml_content: str, feed_name: str = "RSS Feed") -> list[ArticleMeta]:
    """
    Parse RSS/Atom feed XML content into ArticleMeta objects.
    
    Automatically detects feed type (RSS 2.0, RSS 1.0, Atom) and parses accordingly.
    
    Args:
        xml_content: Raw XML string of the feed.
        feed_name: Name to use as source if feed doesn't specify.
    
    Returns:
        List of ArticleMeta objects parsed from the feed.
    
    Raises:
        RSSClientError: If XML parsing fails.
    """
    try:
        root = ElementTree.fromstring(xml_content)
    except ElementTree.ParseError as e:
        raise RSSClientError(f"Failed to parse RSS feed XML: {e}")
    
    feed_type = _detect_feed_type(root)
    logger.debug(f"Detected feed type: {feed_type}")
    
    # Get feed source name
    feed_source = _get_feed_source(root, feed_type) or feed_name
    
    articles = []
    
    if feed_type == "rss2":
        # RSS 2.0: items are in <channel><item>
        channel = root.find("channel")
        if channel is None:
            logger.warning("RSS 2.0 feed has no channel element")
            return []
        
        for item in channel.findall("item"):
            article = _parse_rss_item(item)
            if article:
                # Set source to feed title if article doesn't have one
                if not article.source:
                    article = ArticleMeta(
                        id=article.id,
                        title=article.title,
                        url=article.url,
                        summary=article.summary,
                        content=article.content,
                        published_at=article.published_at,
                        source=feed_source,
                    )
                articles.append(article)
    
    elif feed_type == "rss1":
        # RSS 1.0: items are at the root level (RDF)
        # Try both namespaced and non-namespaced
        items = root.findall("item") or root.findall("{http://purl.org/rss/1.0/}item")
        for item in items:
            article = _parse_rss_item(item)
            if article:
                if not article.source:
                    article = ArticleMeta(
                        id=article.id,
                        title=article.title,
                        url=article.url,
                        summary=article.summary,
                        content=article.content,
                        published_at=article.published_at,
                        source=feed_source,
                    )
                articles.append(article)
    
    elif feed_type == "atom":
        # Atom: entries are direct children of feed
        # Try with namespace first, then without
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            # Try without namespace (some Atom feeds don't use namespace prefix)
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
        if not entries:
            entries = root.findall("entry")
        
        for entry in entries:
            article = _parse_atom_entry(entry, ns)
            if article:
                if not article.source:
                    article = ArticleMeta(
                        id=article.id,
                        title=article.title,
                        url=article.url,
                        summary=article.summary,
                        content=article.content,
                        published_at=article.published_at,
                        source=feed_source,
                    )
                articles.append(article)
    
    else:
        logger.warning(f"Unknown feed type for root tag: {root.tag}")
    
    logger.info(f"Parsed {len(articles)} articles from {feed_name}")
    return articles


def fetch_rss_feed(
    url: str,
    feed_name: str = "RSS Feed",
    limit: int = 10,
    timeout: int = 30,
) -> list[ArticleMeta]:
    """
    Fetch and parse an RSS/Atom feed from a URL.
    
    Args:
        url: URL of the RSS/Atom feed.
        feed_name: Display name for the feed source.
        limit: Maximum number of articles to return.
        timeout: Request timeout in seconds.
    
    Returns:
        List of ArticleMeta objects from the feed.
    
    Raises:
        RSSClientError: If fetching or parsing fails.
    """
    logger.info(f"Fetching RSS feed: {feed_name} ({url})")
    
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()
    except requests.Timeout:
        raise RSSClientError(f"RSS feed request timed out: {url}")
    except requests.RequestException as e:
        raise RSSClientError(f"Failed to fetch RSS feed {url}: {e}")
    
    # Check content type
    content_type = response.headers.get("Content-Type", "")
    if not any(t in content_type.lower() for t in ["xml", "rss", "atom", "text"]):
        logger.warning(f"Unexpected content type for RSS feed: {content_type}")
    
    articles = parse_feed(response.text, feed_name)
    
    # Sort by date (newest first) and limit
    articles.sort(key=lambda a: a.published_at, reverse=True)
    
    return articles[:limit]


def fetch_multiple_feeds(
    feeds: list[RSSFeedConfig],
    deduplicate: bool = True,
) -> dict[str, list[ArticleMeta]]:
    """
    Fetch articles from multiple RSS feeds, organized by section.
    
    Args:
        feeds: List of RSS feed configurations.
        deduplicate: If True, remove duplicate articles (by URL) across feeds.
    
    Returns:
        Dictionary mapping section names to lists of articles.
    """
    sections: dict[str, list[ArticleMeta]] = {}
    seen_urls: set[str] = set()
    errors: list[str] = []
    
    for feed in feeds:
        if not feed.enabled:
            continue
        
        try:
            articles = fetch_rss_feed(
                url=feed.url,
                feed_name=feed.name,
                limit=feed.limit,
            )
            
            # Initialize section if needed
            if feed.section not in sections:
                sections[feed.section] = []
            
            # Deduplicate by URL if enabled
            for article in articles:
                if deduplicate and article.url in seen_urls:
                    continue
                seen_urls.add(article.url)
                sections[feed.section].append(article)
            
            logger.info(f"Fetched {len(articles)} articles from {feed.name}")
            
        except RSSClientError as e:
            errors.append(f"{feed.name}: {e}")
            logger.warning(f"Failed to fetch {feed.name}: {e}")
            # Initialize empty section on failure
            if feed.section not in sections:
                sections[feed.section] = []
            continue
    
    if not sections and errors:
        raise RSSClientError(f"All RSS feeds failed: {'; '.join(errors)}")
    
    return sections


# =============================================================================
# Default RSS Feeds (Finance/Economics focused)
# =============================================================================

DEFAULT_RSS_FEEDS = [
    RSSFeedConfig(
        name="Reuters Business",
        url="https://feeds.reuters.com/reuters/businessNews",
        section="RSS Feeds",
        limit=5,
    ),
    RSSFeedConfig(
        name="BBC Business",
        url="https://feeds.bbci.co.uk/news/business/rss.xml",
        section="RSS Feeds",
        limit=5,
    ),
]
