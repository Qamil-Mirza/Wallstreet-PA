"""Tests for article classification."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from news_bot.classifier import (
    ArticleCategory,
    _classify_by_keywords,
    bucket_articles,
    classify_article_llm,
    is_finance_related,
    select_three_articles,
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


class TestIsFinanceRelated:
    """Tests for is_finance_related function."""

    def test_finance_article_returns_true(self):
        """Test that finance article is identified as relevant."""
        article = make_article(
            "Fed Raises Rates",
            summary="Federal Reserve raises interest rates by 25 basis points"
        )

        with patch("news_bot.classifier._call_ollama") as mock_ollama:
            mock_ollama.return_value = "YES"

            result = is_finance_related(article, "llama3", "http://localhost:11434")

        assert result is True
        mock_ollama.assert_called_once()

    def test_non_finance_article_returns_false(self):
        """Test that non-finance article is filtered out."""
        article = make_article(
            "Celebrity Gossip Update",
            summary="Latest news about movie stars and celebrities"
        )

        with patch("news_bot.classifier._call_ollama") as mock_ollama:
            mock_ollama.return_value = "NO"

            result = is_finance_related(article, "llama3", "http://localhost:11434")

        assert result is False

    def test_defaults_to_true_on_error(self):
        """Test that errors default to True to avoid over-filtering."""
        article = make_article("Some Article")

        with patch("news_bot.classifier._call_ollama") as mock_ollama:
            from news_bot.classifier import ClassifierError
            mock_ollama.side_effect = ClassifierError("Connection failed")

            result = is_finance_related(article, "llama3", "http://localhost:11434")

        # Should default to True on error
        assert result is True


class TestClassifyArticleLlm:
    """Tests for classify_article_llm function."""

    def test_classify_macro(self):
        """Test that macro articles are classified correctly."""
        article = make_article(
            "Fed Signals Rate Pause",
            summary="Federal Reserve indicates pause in rate hikes"
        )

        with patch("news_bot.classifier._call_ollama") as mock_ollama:
            mock_ollama.return_value = "MACRO"

            result = classify_article_llm(article, "llama3", "http://localhost:11434")

        assert result == "macro"

    def test_classify_deal(self):
        """Test that deal articles are classified correctly."""
        article = make_article(
            "Microsoft Acquires Startup",
            summary="Microsoft announces acquisition of AI startup for $2B"
        )

        with patch("news_bot.classifier._call_ollama") as mock_ollama:
            mock_ollama.return_value = "DEAL"

            result = classify_article_llm(article, "llama3", "http://localhost:11434")

        assert result == "deal"

    def test_classify_feature(self):
        """Test that feature articles are classified correctly."""
        article = make_article(
            "Tech Industry Trends 2024",
            summary="Overview of emerging technology trends"
        )

        with patch("news_bot.classifier._call_ollama") as mock_ollama:
            mock_ollama.return_value = "FEATURE"

            result = classify_article_llm(article, "llama3", "http://localhost:11434")

        assert result == "feature"

    def test_defaults_to_feature_on_unknown(self):
        """Test that unknown responses default to feature."""
        article = make_article("Some Article")

        with patch("news_bot.classifier._call_ollama") as mock_ollama:
            mock_ollama.return_value = "UNKNOWN"

            result = classify_article_llm(article, "llama3", "http://localhost:11434")

        assert result == "feature"

    def test_falls_back_to_keywords_on_error(self):
        """Test that errors fall back to keyword classification."""
        article = make_article(
            "Fed Raises Interest Rates",
            summary="Federal Reserve raises rates"
        )

        with patch("news_bot.classifier._call_ollama") as mock_ollama:
            from news_bot.classifier import ClassifierError
            mock_ollama.side_effect = ClassifierError("Connection failed")

            result = classify_article_llm(article, "llama3", "http://localhost:11434")

        # Should fall back to keyword classification (finds "fed", "rates")
        assert result == "macro"


class TestClassifyByKeywords:
    """Tests for keyword-based fallback classification."""

    def test_classify_macro_keywords(self):
        """Test that macro keywords are detected."""
        article = make_article("Fed hikes rates again")
        assert _classify_by_keywords(article) == "macro"

    def test_classify_deal_keywords(self):
        """Test that deal keywords are detected."""
        article = make_article("Company acquires rival in $10B deal")
        assert _classify_by_keywords(article) == "deal"

    def test_classify_feature_default(self):
        """Test that articles without keywords default to feature."""
        article = make_article("Tech trends to watch")
        assert _classify_by_keywords(article) == "feature"


class TestSelectThreeArticles:
    """Tests for select_three_articles function."""

    def test_selects_one_per_category(self):
        """Test that one article is selected per category."""
        articles = [
            make_article("Fed News", article_id="1"),
            make_article("Merger News", article_id="2"),
            make_article("Tech Trends", article_id="3"),
        ]

        with patch("news_bot.classifier.is_finance_related") as mock_relevance:
            mock_relevance.return_value = True
            
            with patch("news_bot.classifier.classify_article_llm") as mock_classify:
                mock_classify.side_effect = ["macro", "deal", "feature"]

                selected, classifications = select_three_articles(
                    articles, "llama3", "http://localhost:11434"
                )

        assert len(selected) == 3
        # Articles should be in category order: macro, deal, feature
        assert selected[0].id == "1"  # macro
        assert selected[1].id == "2"  # deal
        assert selected[2].id == "3"  # feature

    def test_skips_non_finance_articles(self):
        """Test that non-finance articles are skipped."""
        articles = [
            make_article("Celebrity News", article_id="1"),
            make_article("Fed News", article_id="2"),
            make_article("Merger News", article_id="3"),
            make_article("Sports Update", article_id="4"),
            make_article("Tech Trends", article_id="5"),
        ]

        with patch("news_bot.classifier.is_finance_related") as mock_relevance:
            # Articles 1 and 4 are not finance-related
            mock_relevance.side_effect = [False, True, True, False, True]
            
            with patch("news_bot.classifier.classify_article_llm") as mock_classify:
                mock_classify.side_effect = ["macro", "deal", "feature"]

                selected, classifications = select_three_articles(
                    articles, "llama3", "http://localhost:11434"
                )

        assert len(selected) == 3
        # Should have selected articles 2, 3, 5 (skipped 1 and 4)
        selected_ids = [a.id for a in selected]
        assert "2" in selected_ids
        assert "3" in selected_ids
        assert "5" in selected_ids

    def test_stops_when_all_categories_filled(self):
        """Test that processing stops once all 3 categories are filled."""
        articles = [
            make_article("Macro Article", article_id="1"),
            make_article("Deal Article", article_id="2"),
            make_article("Feature Article", article_id="3"),
            make_article("Extra Article", article_id="4"),
            make_article("Another Extra", article_id="5"),
        ]

        with patch("news_bot.classifier.is_finance_related") as mock_relevance:
            mock_relevance.return_value = True
            
            with patch("news_bot.classifier.classify_article_llm") as mock_classify:
                mock_classify.side_effect = ["macro", "deal", "feature"]

                selected, classifications = select_three_articles(
                    articles, "llama3", "http://localhost:11434"
                )

        # Should only have called classify 3 times (stopped after getting all categories)
        assert mock_classify.call_count == 3
        assert len(selected) == 3

    def test_backfills_with_same_category(self):
        """Test that if we can't find all categories, we backfill with available articles."""
        articles = [
            make_article("Feature 1", article_id="1"),
            make_article("Feature 2", article_id="2"),
            make_article("Feature 3", article_id="3"),
        ]

        with patch("news_bot.classifier.is_finance_related") as mock_relevance:
            mock_relevance.return_value = True
            
            with patch("news_bot.classifier.classify_article_llm") as mock_classify:
                # All articles are feature
                mock_classify.side_effect = ["feature", "feature", "feature"]

                selected, classifications = select_three_articles(
                    articles, "llama3", "http://localhost:11434"
                )

        # Should have all 3 articles even though they're all feature
        assert len(selected) == 3
        selected_ids = [a.id for a in selected]
        assert "1" in selected_ids
        assert "2" in selected_ids
        assert "3" in selected_ids

    def test_partial_categories_with_backfill(self):
        """Test that partial categories are filled, then backfill happens."""
        articles = [
            make_article("Macro 1", article_id="1"),
            make_article("Macro 2", article_id="2"),
            make_article("Macro 3", article_id="3"),
        ]

        with patch("news_bot.classifier.is_finance_related") as mock_relevance:
            mock_relevance.return_value = True
            
            with patch("news_bot.classifier.classify_article_llm") as mock_classify:
                mock_classify.side_effect = ["macro", "macro", "macro"]

                selected, classifications = select_three_articles(
                    articles, "llama3", "http://localhost:11434"
                )

        # Should have 3 articles: 1 macro selected first, then 2 backfilled
        assert len(selected) == 3
        assert selected[0].id == "1"  # First macro selected
        # Other two are backfilled
        assert selected[1].id == "2"
        assert selected[2].id == "3"

    def test_handles_few_articles(self):
        """Test handling when fewer than 3 finance articles exist."""
        articles = [
            make_article("Non-finance", article_id="1"),
            make_article("Finance 1", article_id="2"),
        ]

        with patch("news_bot.classifier.is_finance_related") as mock_relevance:
            mock_relevance.side_effect = [False, True]
            
            with patch("news_bot.classifier.classify_article_llm") as mock_classify:
                mock_classify.side_effect = ["feature"]

                selected, classifications = select_three_articles(
                    articles, "llama3", "http://localhost:11434"
                )

        # Only 1 finance article available
        assert len(selected) == 1
        assert selected[0].id == "2"


class TestBucketArticles:
    """Tests for bucket_articles function."""

    def test_bucket_with_classifications(self):
        """Test bucketing with pre-computed classifications."""
        articles = [
            make_article("Article 1", article_id="1"),
            make_article("Article 2", article_id="2"),
            make_article("Article 3", article_id="3"),
        ]

        classifications = {
            "1": "macro",
            "2": "deal",
            "3": "feature",
        }

        buckets = bucket_articles(articles, classifications)

        assert len(buckets["macro"]) == 1
        assert len(buckets["deal"]) == 1
        assert len(buckets["feature"]) == 1
        assert buckets["macro"][0].id == "1"
        assert buckets["deal"][0].id == "2"
        assert buckets["feature"][0].id == "3"

    def test_bucket_without_classifications_uses_keywords(self):
        """Test that missing classifications fall back to keywords."""
        articles = [
            make_article("Fed raises rates", article_id="1"),
            make_article("Company acquires rival", article_id="2"),
        ]

        buckets = bucket_articles(articles)

        assert len(buckets["macro"]) == 1
        assert len(buckets["deal"]) == 1

    def test_bucket_empty_list(self):
        """Test bucketing empty list returns empty buckets."""
        buckets = bucket_articles([])

        assert buckets["macro"] == []
        assert buckets["deal"] == []
        assert buckets["feature"] == []

    def test_bucket_sorted_by_date(self):
        """Test that buckets are sorted by date descending."""
        articles = [
            make_article("Old article", article_id="1", published_at=datetime(2024, 1, 10)),
            make_article("New article", article_id="2", published_at=datetime(2024, 1, 15)),
        ]

        classifications = {"1": "feature", "2": "feature"}

        buckets = bucket_articles(articles, classifications)

        # Newer article should be first
        assert buckets["feature"][0].id == "2"
        assert buckets["feature"][1].id == "1"
