"""Tests for article selection logic."""

from datetime import datetime, timedelta

import pytest

from news_bot.classifier import ArticleCategory
from news_bot.news_client import ArticleMeta
from news_bot.selection import get_article_category_label, select_three_articles


def make_article(
    article_id: str,
    title: str = "Test Article",
    days_ago: int = 0,
) -> ArticleMeta:
    """Helper to create test articles."""
    return ArticleMeta(
        id=article_id,
        title=title,
        url=f"https://example.com/{article_id}",
        summary="Test summary",
        content="Test content",
        published_at=datetime.now() - timedelta(days=days_ago),
        source="Test Source",
    )


class TestSelectThreeArticles:
    """Tests for select_three_articles function."""

    def test_select_three_articles_all_buckets_present(self):
        """Test selection when all three buckets have articles."""
        buckets: dict[ArticleCategory, list[ArticleMeta]] = {
            "macro": [make_article("macro-1", "Fed article")],
            "deal": [make_article("deal-1", "Merger article")],
            "feature": [make_article("feature-1", "Trend article")],
        }

        result = select_three_articles(buckets)

        assert len(result) == 3
        ids = {a.id for a in result}
        assert ids == {"macro-1", "deal-1", "feature-1"}

    def test_select_three_articles_picks_newest_from_each(self):
        """Test that newest article from each bucket is selected."""
        buckets: dict[ArticleCategory, list[ArticleMeta]] = {
            "macro": [
                make_article("macro-new", "New Fed", days_ago=0),
                make_article("macro-old", "Old Fed", days_ago=5),
            ],
            "deal": [
                make_article("deal-new", "New Merger", days_ago=1),
                make_article("deal-old", "Old Merger", days_ago=3),
            ],
            "feature": [
                make_article("feature-new", "New Trend", days_ago=2),
            ],
        }

        result = select_three_articles(buckets)

        assert len(result) == 3
        ids = {a.id for a in result}
        assert ids == {"macro-new", "deal-new", "feature-new"}

    def test_select_three_articles_backfill_no_deal(self):
        """Test backfill when deal bucket is empty."""
        buckets: dict[ArticleCategory, list[ArticleMeta]] = {
            "macro": [
                make_article("macro-1", days_ago=0),
                make_article("macro-2", days_ago=2),
            ],
            "deal": [],
            "feature": [
                make_article("feature-1", days_ago=1),
                make_article("feature-2", days_ago=3),
            ],
        }

        result = select_three_articles(buckets)

        assert len(result) == 3
        ids = {a.id for a in result}
        # Should pick macro-1, feature-1, then backfill with next most recent
        assert "macro-1" in ids
        assert "feature-1" in ids
        # Backfill should be feature-2 (day 3) or macro-2 (day 2), macro-2 is newer
        assert "macro-2" in ids or "feature-2" in ids

    def test_select_three_articles_backfill_no_macro(self):
        """Test backfill when macro bucket is empty."""
        buckets: dict[ArticleCategory, list[ArticleMeta]] = {
            "macro": [],
            "deal": [make_article("deal-1", days_ago=0)],
            "feature": [
                make_article("feature-1", days_ago=1),
                make_article("feature-2", days_ago=2),
            ],
        }

        result = select_three_articles(buckets)

        assert len(result) == 3
        assert "deal-1" in {a.id for a in result}
        assert "feature-1" in {a.id for a in result}
        assert "feature-2" in {a.id for a in result}

    def test_select_three_articles_only_two_available(self):
        """Test when only 2 total articles are available."""
        buckets: dict[ArticleCategory, list[ArticleMeta]] = {
            "macro": [make_article("macro-1")],
            "deal": [],
            "feature": [make_article("feature-1")],
        }

        result = select_three_articles(buckets)

        assert len(result) == 2
        ids = {a.id for a in result}
        assert ids == {"macro-1", "feature-1"}

    def test_select_three_articles_only_one_available(self):
        """Test when only 1 article is available."""
        buckets: dict[ArticleCategory, list[ArticleMeta]] = {
            "macro": [],
            "deal": [make_article("deal-1")],
            "feature": [],
        }

        result = select_three_articles(buckets)

        assert len(result) == 1
        assert result[0].id == "deal-1"

    def test_select_three_articles_empty_buckets(self):
        """Test with all empty buckets."""
        buckets: dict[ArticleCategory, list[ArticleMeta]] = {
            "macro": [],
            "deal": [],
            "feature": [],
        }

        result = select_three_articles(buckets)

        assert result == []

    def test_select_three_articles_no_duplicates(self):
        """Test that no duplicate articles are selected."""
        buckets: dict[ArticleCategory, list[ArticleMeta]] = {
            "macro": [
                make_article("a1", days_ago=0),
                make_article("a2", days_ago=1),
                make_article("a3", days_ago=2),
            ],
            "deal": [],
            "feature": [],
        }

        result = select_three_articles(buckets)

        assert len(result) == 3
        ids = [a.id for a in result]
        assert len(set(ids)) == 3  # All unique


class TestGetArticleCategoryLabel:
    """Tests for get_article_category_label function."""

    def test_get_macro_label(self):
        """Test getting label for macro article."""
        article = make_article("macro-1")
        buckets: dict[ArticleCategory, list[ArticleMeta]] = {
            "macro": [article],
            "deal": [],
            "feature": [],
        }

        label = get_article_category_label(article, buckets)

        assert label == "Macro & Economics"

    def test_get_deal_label(self):
        """Test getting label for deal article."""
        article = make_article("deal-1")
        buckets: dict[ArticleCategory, list[ArticleMeta]] = {
            "macro": [],
            "deal": [article],
            "feature": [],
        }

        label = get_article_category_label(article, buckets)

        assert label == "Deals & Corporate"

    def test_get_feature_label(self):
        """Test getting label for feature article."""
        article = make_article("feature-1")
        buckets: dict[ArticleCategory, list[ArticleMeta]] = {
            "macro": [],
            "deal": [],
            "feature": [article],
        }

        label = get_article_category_label(article, buckets)

        assert label == "Feature & Trends"

    def test_get_label_article_not_in_buckets(self):
        """Test getting label for article not in any bucket."""
        article = make_article("unknown")
        buckets: dict[ArticleCategory, list[ArticleMeta]] = {
            "macro": [],
            "deal": [],
            "feature": [],
        }

        label = get_article_category_label(article, buckets)

        # Should default to feature
        assert label == "Feature & Trends"
