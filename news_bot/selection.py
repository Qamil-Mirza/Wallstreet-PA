"""
Article selection logic.

Selects the top 3 articles across categories with backfill logic.
"""

from .classifier import ArticleCategory
from .news_client import ArticleMeta


def select_three_articles(
    buckets: dict[ArticleCategory, list[ArticleMeta]]
) -> list[ArticleMeta]:
    """
    Select up to 3 articles, one from each category with backfill.
    
    Selection strategy:
    1. Pick the newest article from each bucket (macro, deal, feature).
    2. If fewer than 3 articles selected, backfill from remaining articles
       sorted by recency.
    
    Args:
        buckets: Dictionary mapping categories to sorted article lists
                (most recent first).
    
    Returns:
        List of up to 3 selected articles, tagged with their categories.
    """
    selected: list[ArticleMeta] = []
    used_ids: set[str] = set()
    
    # Priority order for categories
    category_order: list[ArticleCategory] = ["macro", "deal", "feature"]
    
    # Phase 1: Pick one from each category
    for category in category_order:
        articles = buckets.get(category, [])
        for article in articles:
            if article.id not in used_ids:
                selected.append(article)
                used_ids.add(article.id)
                break
    
    # Phase 2: Backfill if we have fewer than 3
    if len(selected) < 3:
        # Gather all remaining articles
        remaining: list[ArticleMeta] = []
        for category in category_order:
            for article in buckets.get(category, []):
                if article.id not in used_ids:
                    remaining.append(article)
        
        # Sort by recency
        remaining.sort(key=lambda a: a.published_at, reverse=True)
        
        # Fill remaining slots
        for article in remaining:
            if len(selected) >= 3:
                break
            selected.append(article)
            used_ids.add(article.id)
    
    return selected


def get_article_category_label(
    article: ArticleMeta,
    buckets: dict[ArticleCategory, list[ArticleMeta]]
) -> str:
    """
    Get a display label for an article's category.
    
    Args:
        article: The article to label.
        buckets: The category buckets used for selection.
    
    Returns:
        Human-readable category label.
    """
    category_labels = {
        "macro": "Macro & Economics",
        "deal": "Deals & Corporate",
        "feature": "Feature & Trends",
    }
    
    for category, articles in buckets.items():
        if any(a.id == article.id for a in articles):
            return category_labels.get(category, "Feature & Trends")
    
    return "Feature & Trends"
