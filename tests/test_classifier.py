"""Tests for article classification."""

from datetime import datetime

import pytest

from news_bot.classifier import (
    ArticleCategory,
    bucket_articles,
    classify_article,
)
from news_bot.news_client import ArticleMeta


def make_article(
    title: str,
    content: str = "",
    article_id: str = "test-id",
    published_at: datetime = None,
) -> ArticleMeta:
    """Helper to create test articles."""
    return ArticleMeta(
        id=article_id,
        title=title,
        url=f"https://example.com/{article_id}",
        summary=content[:100] if content else None,
        content=content or None,
        published_at=published_at or datetime.now(),
        source="Test Source",
    )


class TestClassifyArticle:
    """Tests for classify_article function."""

    # Macro classification tests
    def test_classify_article_macro_fed(self):
        """Test that Fed-related articles are classified as macro."""
        article = make_article("Fed hikes rates again by 25 basis points")
        assert classify_article(article) == "macro"

    def test_classify_article_macro_inflation(self):
        """Test that inflation articles are classified as macro."""
        article = make_article("CPI data shows inflation cooling to 3.2%")
        assert classify_article(article) == "macro"

    def test_classify_article_macro_gdp(self):
        """Test that GDP articles are classified as macro."""
        article = make_article("US GDP growth beats expectations at 2.5%")
        assert classify_article(article) == "macro"

    def test_classify_article_macro_jobs(self):
        """Test that jobs report articles are classified as macro."""
        article = make_article("Jobs report: 250K payrolls added in December")
        assert classify_article(article) == "macro"

    def test_classify_article_macro_ecb(self):
        """Test that ECB articles are classified as macro."""
        article = make_article("ECB holds rates steady amid uncertainty")
        assert classify_article(article) == "macro"

    def test_classify_article_macro_recession(self):
        """Test that recession articles are classified as macro."""
        article = make_article("Economists debate recession probability")
        assert classify_article(article) == "macro"

    def test_classify_article_macro_bonds(self):
        """Test that bond market articles are classified as macro."""
        article = make_article("Treasury yields surge on inflation fears")
        assert classify_article(article) == "macro"

    def test_classify_article_macro_in_content(self):
        """Test that macro keywords in content are detected."""
        article = make_article(
            "Markets React to Latest Data",
            content="The Federal Reserve signaled more rate hikes ahead."
        )
        assert classify_article(article) == "macro"

    # Deal classification tests
    def test_classify_article_deal_merger(self):
        """Test that merger articles are classified as deal."""
        article = make_article("Major tech merger announced today")
        assert classify_article(article) == "deal"

    def test_classify_article_deal_acquisition(self):
        """Test that acquisition articles are classified as deal."""
        article = make_article("Company A acquires Company B in $10B deal")
        assert classify_article(article) == "deal"

    def test_classify_article_deal_ipo(self):
        """Test that IPO articles are classified as deal."""
        article = make_article("Startup files for IPO valuing company at $5B")
        assert classify_article(article) == "deal"

    def test_classify_article_deal_takeover(self):
        """Test that takeover articles are classified as deal."""
        article = make_article("Hostile takeover bid rejected by board")
        assert classify_article(article) == "deal"

    def test_classify_article_deal_buyout(self):
        """Test that buyout articles are classified as deal."""
        article = make_article("Private equity firm completes leveraged buyout")
        assert classify_article(article) == "deal"

    def test_classify_article_deal_spac(self):
        """Test that SPAC articles are classified as deal."""
        article = make_article("Company to go public via SPAC merger")
        assert classify_article(article) == "deal"

    # Feature/default classification tests
    def test_classify_article_feature_default(self):
        """Test that generic articles default to feature."""
        article = make_article("Tech trends to watch in 2024")
        assert classify_article(article) == "feature"

    def test_classify_article_feature_industry(self):
        """Test that industry stories are classified as feature."""
        article = make_article("AI adoption accelerates in healthcare sector")
        assert classify_article(article) == "feature"

    def test_classify_article_feature_earnings(self):
        """Test that earnings articles are classified as feature."""
        article = make_article("Apple reports record quarterly earnings")
        assert classify_article(article) == "feature"

    def test_classify_article_empty_content(self):
        """Test classification with empty content."""
        article = make_article("Random headline with no keywords")
        assert classify_article(article) == "feature"


class TestBucketArticles:
    """Tests for bucket_articles function."""

    def test_bucket_articles_all_categories(self):
        """Test that articles are bucketed into correct categories."""
        articles = [
            make_article("Fed raises rates", article_id="1"),
            make_article("Company acquires rival", article_id="2"),
            make_article("Industry trends report", article_id="3"),
        ]

        buckets = bucket_articles(articles)

        assert len(buckets["macro"]) == 1
        assert len(buckets["deal"]) == 1
        assert len(buckets["feature"]) == 1
        assert buckets["macro"][0].id == "1"
        assert buckets["deal"][0].id == "2"
        assert buckets["feature"][0].id == "3"

    def test_bucket_articles_empty_list(self):
        """Test bucketing empty list returns empty buckets."""
        buckets = bucket_articles([])

        assert buckets["macro"] == []
        assert buckets["deal"] == []
        assert buckets["feature"] == []

    def test_bucket_articles_sorted_by_date(self):
        """Test that articles in buckets are sorted by date descending."""
        articles = [
            make_article(
                "Fed news old",
                article_id="1",
                published_at=datetime(2024, 1, 10)
            ),
            make_article(
                "Fed news new",
                article_id="2",
                published_at=datetime(2024, 1, 15)
            ),
            make_article(
                "Fed news middle",
                article_id="3",
                published_at=datetime(2024, 1, 12)
            ),
        ]

        buckets = bucket_articles(articles)

        macro_articles = buckets["macro"]
        assert len(macro_articles) == 3
        # Should be sorted newest first
        assert macro_articles[0].id == "2"  # Jan 15
        assert macro_articles[1].id == "3"  # Jan 12
        assert macro_articles[2].id == "1"  # Jan 10

    def test_bucket_articles_all_same_category(self):
        """Test when all articles fall into one category."""
        articles = [
            make_article("IPO filing", article_id="1"),
            make_article("Merger deal", article_id="2"),
            make_article("Acquisition news", article_id="3"),
        ]

        buckets = bucket_articles(articles)

        assert len(buckets["deal"]) == 3
        assert len(buckets["macro"]) == 0
        assert len(buckets["feature"]) == 0
