"""Tests for article classification."""

from datetime import datetime

import pytest

from news_bot.classifier import (
    ArticleCategory,
    _score_category,
    bucket_articles,
    classify_article,
    MACRO_KEYWORDS,
    DEAL_KEYWORDS,
)
from news_bot.news_client import ArticleMeta


def make_article(
    title: str,
    content: str = "",
    article_id: str = "test-id",
    published_at: datetime = None,
    summary: str = None,
) -> ArticleMeta:
    """Helper to create test articles."""
    return ArticleMeta(
        id=article_id,
        title=title,
        url=f"https://example.com/{article_id}",
        summary=summary or (content[:100] if content else None),
        content=content or None,
        published_at=published_at or datetime.now(),
        source="Test Source",
    )


class TestScoreCategory:
    """Tests for _score_category function."""

    def test_score_with_matching_keywords(self):
        """Test that matching keywords increase score."""
        text = "the federal reserve raised interest rates"
        score = _score_category(text, MACRO_KEYWORDS)
        assert score > 0

    def test_score_with_no_keywords(self):
        """Test that no keywords results in zero score."""
        text = "a completely unrelated topic about cats"
        score = _score_category(text, MACRO_KEYWORDS)
        assert score == 0

    def test_score_counts_multiple_matches(self):
        """Test that multiple keyword matches increase score."""
        text = "federal reserve inflation interest rate hike"
        score = _score_category(text, MACRO_KEYWORDS)
        assert score >= 3  # Multiple keywords should match


class TestClassifyArticle:
    """Tests for classify_article function."""

    def test_classify_macro_keywords(self):
        """Test that macro keywords classify as macro."""
        article = make_article(
            "Fed Raises Rates",
            "The Federal Reserve raised interest rates by 25 basis points"
        )
        assert classify_article(article) == "macro"

    def test_classify_deal_keywords(self):
        """Test that deal keywords classify as deal."""
        article = make_article(
            "Microsoft Acquires Startup",
            "Microsoft announces acquisition of AI startup for $2B in merger deal"
        )
        assert classify_article(article) == "deal"

    def test_classify_feature_default(self):
        """Test that articles without keywords default to feature."""
        article = make_article(
            "Tech Industry Overview",
            "A look at the latest technology trends"
        )
        assert classify_article(article) == "feature"

    def test_classify_prefers_higher_score(self):
        """Test that the category with more matches wins."""
        # Article with more macro keywords than deal keywords
        article = make_article(
            "Fed inflation rates",
            "The Federal Reserve, inflation, interest rate, monetary policy, yield curve"
        )
        result = classify_article(article)
        assert result == "macro"

    def test_classify_uses_title_weight(self):
        """Test that title has extra weight in classification."""
        # Title mentions Fed, content is generic
        article = make_article(
            "Fed Announces Decision",
            "Some generic content about markets"
        )
        result = classify_article(article)
        assert result == "macro"

    def test_classify_with_content_only(self):
        """Test classification when only content has keywords."""
        article = make_article(
            "News Update",
            "The company announced an acquisition and merger with a rival firm"
        )
        result = classify_article(article)
        assert result == "deal"


class TestBucketArticles:
    """Tests for bucket_articles function."""

    def test_bucket_articles_basic(self):
        """Test basic article bucketing."""
        articles = [
            make_article("Fed raises rates", article_id="1"),
            make_article("Company acquires rival", article_id="2"),
            make_article("Tech trends overview", article_id="3"),
        ]

        buckets = bucket_articles(articles)

        assert len(buckets["macro"]) == 1
        assert len(buckets["deal"]) == 1
        assert len(buckets["feature"]) == 1
        assert buckets["macro"][0].id == "1"
        assert buckets["deal"][0].id == "2"
        assert buckets["feature"][0].id == "3"

    def test_bucket_empty_list(self):
        """Test bucketing empty list returns empty buckets."""
        buckets = bucket_articles([])

        assert buckets["macro"] == []
        assert buckets["deal"] == []
        assert buckets["feature"] == []

    def test_bucket_sorted_by_date(self):
        """Test that buckets are sorted by date descending."""
        articles = [
            make_article("Old tech trends", article_id="1", published_at=datetime(2024, 1, 10)),
            make_article("New tech trends", article_id="2", published_at=datetime(2024, 1, 15)),
        ]

        buckets = bucket_articles(articles)

        # Both are feature articles, newer should be first
        assert buckets["feature"][0].id == "2"
        assert buckets["feature"][1].id == "1"

    def test_bucket_all_same_category(self):
        """Test bucketing when all articles are same category."""
        articles = [
            make_article("Fed news 1", article_id="1"),
            make_article("Central bank policy", article_id="2"),
            make_article("Interest rate decision", article_id="3"),
        ]

        buckets = bucket_articles(articles)

        assert len(buckets["macro"]) == 3
        assert len(buckets["deal"]) == 0
        assert len(buckets["feature"]) == 0

    def test_bucket_uses_summary_fallback(self):
        """Test that summary is used when content is None."""
        article = make_article(
            "News",
            content=None,
            summary="The Federal Reserve raised rates today"
        )

        buckets = bucket_articles([article])

        assert len(buckets["macro"]) == 1


class TestKeywords:
    """Tests for keyword lists."""

    def test_macro_keywords_not_empty(self):
        """Test that macro keywords list is populated."""
        assert len(MACRO_KEYWORDS) > 0

    def test_deal_keywords_not_empty(self):
        """Test that deal keywords list is populated."""
        assert len(DEAL_KEYWORDS) > 0

    def test_keywords_are_lowercase(self):
        """Test that all keywords are lowercase for matching."""
        for keyword in MACRO_KEYWORDS:
            assert keyword == keyword.lower(), f"Keyword '{keyword}' is not lowercase"
        for keyword in DEAL_KEYWORDS:
            assert keyword == keyword.lower(), f"Keyword '{keyword}' is not lowercase"
