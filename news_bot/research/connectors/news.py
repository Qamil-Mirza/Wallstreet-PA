"""Adapters from the existing MarketAux and RSS clients to evidence inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone

from ...news_client import ArticleMeta, fetch_articles_by_section
from ..evidence import DocumentInput
from .base import (
    ConnectorBatch,
    ConnectorCheckpoint,
    ConnectorError,
    NormalizedResearchDocument,
)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class _ArticleConnector:
    name: str
    source_type: str

    def __init__(self, *, now: Callable[[], datetime] | None = None) -> None:
        self._now = now or (lambda: datetime.now(timezone.utc))

    def from_article(
        self, article: ArticleMeta, *, section: str
    ) -> NormalizedResearchDocument:
        if not isinstance(article, ArticleMeta):
            raise TypeError("article must be ArticleMeta")
        content = article.content or article.summary or article.title
        publisher = article.source or self.name
        return NormalizedResearchDocument(
            evidence=DocumentInput(
                source_type=self.source_type,
                url=article.url,
                publisher=publisher,
                published_at=_aware_utc(article.published_at),
                retrieved_at=_aware_utc(self._now()),
                content=content,
            ),
            tags=(section,),
        )

    def _batch(
        self,
        sections: Mapping[str, Sequence[ArticleMeta]],
    ) -> ConnectorBatch:
        documents = tuple(
            self.from_article(article, section=section)
            for section in sorted(sections)
            for article in sections[section]
        )
        cursor = max(
            (document.published_at.isoformat() for document in documents),
            default="empty",
        )
        return ConnectorBatch(
            connector=self.name,
            documents=documents,
            next_checkpoint=ConnectorCheckpoint(self.name, cursor=cursor),
        )


class MarketAuxResearchConnector(_ArticleConnector):
    """Reuse the production sectioned MarketAux client without duplicating it."""

    name = "marketaux"
    source_type = "marketaux_news"

    def __init__(
        self,
        news_config=None,
        *,
        section_fetcher: Callable[..., Mapping[str, Sequence[ArticleMeta]]] = fetch_articles_by_section,
        per_section_limit: int = 5,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(now=now)
        if not isinstance(per_section_limit, int) or per_section_limit <= 0:
            raise ValueError("per_section_limit must be positive")
        self.news_config = news_config
        self._section_fetcher = section_fetcher
        self.per_section_limit = per_section_limit

    def fetch(self, checkpoint: ConnectorCheckpoint) -> ConnectorBatch:
        if checkpoint.connector != self.name:
            raise ValueError("checkpoint belongs to another connector")
        if self.news_config is None:
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="configuration_missing"
            )
        try:
            sections = self._section_fetcher(
                self.news_config, per_section_limit=self.per_section_limit
            )
        except Exception:
            raise ConnectorError(
                self.name, retryable=True, diagnostic_code="upstream_fetch_failed"
            ) from None
        return self._batch(sections)


class RSSResearchConnector(_ArticleConnector):
    """Normalize already-parsed RSS ``ArticleMeta`` records."""

    name = "rss"
    source_type = "rss_news"

    def __init__(
        self,
        article_provider: Callable[[], Mapping[str, Sequence[ArticleMeta]]] | None = None,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(now=now)
        self._article_provider = article_provider

    def fetch(self, checkpoint: ConnectorCheckpoint) -> ConnectorBatch:
        if checkpoint.connector != self.name:
            raise ValueError("checkpoint belongs to another connector")
        if self._article_provider is None:
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="configuration_missing"
            )
        try:
            return self._batch(self._article_provider())
        except ConnectorError:
            raise
        except Exception:
            raise ConnectorError(
                self.name, retryable=True, diagnostic_code="upstream_fetch_failed"
            ) from None
