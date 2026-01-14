"""
Article classification based on content keywords.

Classifies articles into Macro, Deal, or Feature categories.
"""

from typing import Literal

from .news_client import ArticleMeta


ArticleCategory = Literal["macro", "deal", "feature"]


# Keywords for classification (lowercase)
MACRO_KEYWORDS = {
    # Central banks
    "fed", "federal reserve", "ecb", "boe", "boj", "rba", "pboc",
    "central bank", "fomc", "monetary policy",
    # Economic indicators
    "inflation", "cpi", "pce", "deflation", "stagflation",
    "jobs report", "payrolls", "unemployment", "nonfarm",
    "gdp", "gross domestic product",
    "recession", "economic growth", "economic contraction",
    # Rates and yields
    "rate hike", "rate cut", "interest rate", "basis points",
    "yield", "yields", "treasury", "treasuries", "bonds", "bond market",
    # Other macro
    "trade deficit", "trade surplus", "current account",
    "fiscal policy", "government spending", "stimulus",
    "quantitative easing", "qe", "tightening",
}

DEAL_KEYWORDS = {
    # M&A
    "merger", "mergers", "acquisition", "acquisitions",
    "acquires", "acquired", "acquiring",
    "takeover", "takeout", "buyout", "buy out",
    "lbo", "leveraged buyout",
    # IPO and offerings
    "ipo", "initial public offering", "goes public", "going public",
    "secondary offering", "stock offering", "share sale",
    "spac", "direct listing",
    # Deal terms
    "deal value", "billion deal", "million deal",
    "all-cash", "all-stock", "cash and stock",
    "antitrust", "ftc", "doj review",
}


def classify_article(article: ArticleMeta) -> ArticleCategory:
    """
    Classify an article into a category based on keywords.
    
    Args:
        article: The article to classify.
    
    Returns:
        Category: "macro", "deal", or "feature" (default).
    """
    # Combine title and content for matching
    text_parts = [article.title.lower()]
    
    if article.content:
        text_parts.append(article.content.lower())
    elif article.summary:
        text_parts.append(article.summary.lower())
    
    text = " ".join(text_parts)
    
    # Check for macro keywords
    for keyword in MACRO_KEYWORDS:
        if keyword in text:
            return "macro"
    
    # Check for deal keywords
    for keyword in DEAL_KEYWORDS:
        if keyword in text:
            return "deal"
    
    # Default to feature
    return "feature"


def bucket_articles(
    articles: list[ArticleMeta]
) -> dict[ArticleCategory, list[ArticleMeta]]:
    """
    Bucket articles by category.
    
    Args:
        articles: List of articles to classify and bucket.
    
    Returns:
        Dictionary mapping categories to lists of articles.
        Each list is sorted by published_at descending (most recent first).
    """
    buckets: dict[ArticleCategory, list[ArticleMeta]] = {
        "macro": [],
        "deal": [],
        "feature": [],
    }
    
    for article in articles:
        category = classify_article(article)
        buckets[category].append(article)
    
    # Sort each bucket by published_at descending
    for category in buckets:
        buckets[category].sort(key=lambda a: a.published_at, reverse=True)
    
    return buckets
