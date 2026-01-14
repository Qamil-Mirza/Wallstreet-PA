"""
Article classification based on content keywords.

Classifies articles into Macro, Deal, or Feature categories.
"""

from typing import Literal

from .news_client import ArticleMeta


ArticleCategory = Literal["macro", "deal", "feature"]


# Keywords for classification (lowercase)
# Ordered by specificity - more specific phrases first
MACRO_KEYWORDS = [
    # Central banks (high priority)
    "federal reserve", "central bank", "fomc", "monetary policy",
    "fed chair", "fed meeting", "fed decision", "fed minutes",
    "the fed", " fed ", "ecb", "boe", "boj", "rba", "pboc",
    # Economic indicators
    "inflation", "consumer price", "cpi", "pce", "deflation", "stagflation",
    "jobs report", "payrolls", "unemployment", "nonfarm", "jobless claims",
    "gdp", "gross domestic product", "economic growth", "economic data",
    "recession", "soft landing", "hard landing", "economic contraction",
    "retail sales", "consumer spending", "consumer confidence",
    "housing starts", "home sales", "housing market",
    "manufacturing", "industrial production", "factory orders",
    "ism", "pmi", "purchasing managers",
    # Rates and yields
    "rate hike", "rate cut", "interest rate", "basis points", "bps",
    "yield", "yields", "treasury", "treasuries", "bond market",
    "10-year", "2-year", "yield curve", "inverted curve",
    "dovish", "hawkish", "pivot", "pause",
    # Markets and indices
    "s&p 500", "dow jones", "nasdaq", "stock market", "wall street",
    "bull market", "bear market", "market rally", "market selloff",
    "volatility", "vix",
    # Currency and commodities
    "dollar index", "forex", "currency", "exchange rate",
    "oil price", "crude oil", "brent", "wti", "gold price",
    "commodities",
    # Other macro
    "trade deficit", "trade surplus", "current account", "tariff",
    "fiscal policy", "government spending", "stimulus", "debt ceiling",
    "quantitative easing", "qe", "tightening", "balance sheet",
]

DEAL_KEYWORDS = [
    # M&A (high priority)
    "merger", "mergers", "acquisition", "acquisitions",
    "acquires", "acquired", "acquiring", "to acquire", "to buy",
    "takeover", "takeout", "buyout", "buy out", "bought out",
    "lbo", "leveraged buyout", "management buyout",
    "bid for", "bidding war", "hostile bid", "friendly deal",
    # IPO and offerings
    "ipo", "initial public offering", "goes public", "going public",
    "public offering", "secondary offering", "stock offering", "share sale",
    "spac", "direct listing", "market debut", "stock debut",
    "files to go public", "ipo filing", "s-1 filing",
    # Corporate actions
    "spin-off", "spinoff", "spin off", "divestiture", "divest",
    "stake sale", "sells stake", "buys stake", "takes stake",
    "strategic review", "strategic alternatives", "explores sale",
    "private equity", " pe firm", "buyout firm",
    "venture capital", " vc ", "funding round", "series a", "series b", "series c",
    # Deal terms
    "deal value", "billion deal", "million deal", "purchase price",
    "all-cash", "all-stock", "cash and stock", "deal terms",
    "antitrust", "ftc", "doj review", "regulatory approval",
    # Restructuring
    "bankruptcy", "chapter 11", "restructuring", "creditors",
    "debt restructuring", "refinancing",
]


def _score_category(text: str, keywords: list[str]) -> int:
    """
    Score how well text matches a category's keywords.
    
    Returns the number of keyword matches found.
    """
    score = 0
    for keyword in keywords:
        if keyword in text:
            score += 1
    return score


def classify_article(article: ArticleMeta) -> ArticleCategory:
    """
    Classify an article into a category based on keywords.
    
    Uses scoring to handle articles that match multiple categories -
    picks the category with the most keyword matches.
    
    Args:
        article: The article to classify.
    
    Returns:
        Category: "macro", "deal", or "feature" (default).
    """
    # Combine title and content for matching
    # Title gets extra weight by including it twice
    text_parts = [article.title.lower(), article.title.lower()]
    
    if article.content:
        text_parts.append(article.content.lower())
    elif article.summary:
        text_parts.append(article.summary.lower())
    
    text = " ".join(text_parts)
    
    # Score each category
    macro_score = _score_category(text, MACRO_KEYWORDS)
    deal_score = _score_category(text, DEAL_KEYWORDS)
    
    # Return category with highest score, defaulting to feature
    if macro_score > 0 and macro_score >= deal_score:
        return "macro"
    elif deal_score > 0:
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
