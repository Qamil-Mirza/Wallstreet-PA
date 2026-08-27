"""Adapters from the existing MarketAux and RSS clients to evidence inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone

from ...news_client import ArticleMeta, fetch_articles_by_section
from ..evidence import (
    DocumentInput,
    EvidenceIngestor,
    IngestedDocument,
    canonicalize_content,
    canonicalize_url,
)
from .base import (
    ConnectorBatch,
    ConnectorCheckpoint,
    ConnectorError,
    NormalizedResearchDocument,
    commit_connector_batch,
)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class _ArticleConnector:
    name: str
    source_type: str

    def __init__(
        self,
        *,
        now: Callable[[], datetime] | None = None,
        ingestor: EvidenceIngestor | None = None,
    ) -> None:
        if ingestor is not None and not isinstance(ingestor, EvidenceIngestor):
            raise TypeError("ingestor must be EvidenceIngestor")
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._ingestor = ingestor

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
                url=canonicalize_url(article.url),
                publisher=publisher,
                published_at=_aware_utc(article.published_at),
                retrieved_at=_aware_utc(self._now()),
                content=canonicalize_content(content),
            ),
            tags=(section,),
        )

    def _batch(
        self,
        sections: Mapping[str, Sequence[ArticleMeta]],
        checkpoint: ConnectorCheckpoint,
    ) -> ConnectorBatch:
        documents = tuple(
            self.from_article(article, section=section)
            for section in sorted(sections)
            for article in sections[section]
        )
        cursor_candidates = [
            document.published_at.isoformat() for document in documents
        ]
        if checkpoint.cursor is not None:
            cursor_candidates.append(checkpoint.cursor)
        cursor = max(cursor_candidates, default=None)
        return ConnectorBatch(
            connector=self.name,
            documents=documents,
            next_checkpoint=ConnectorCheckpoint(self.name, cursor=cursor),
        )

    def fetch_and_persist(
        self,
        checkpoint: ConnectorCheckpoint,
        *,
        persist_checkpoint: Callable[[ConnectorCheckpoint], None],
    ) -> tuple[IngestedDocument, ...]:
        """Fetch once, ingest canonical evidence, then advance the cursor."""
        if self._ingestor is None:
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="ingestor_missing"
            )
        return commit_connector_batch(
            self.fetch(checkpoint),
            ingestor=self._ingestor,
            persist_checkpoint=persist_checkpoint,
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
        ingestor: EvidenceIngestor | None = None,
    ) -> None:
        super().__init__(now=now, ingestor=ingestor)
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
        return self._batch(sections, checkpoint)


class RSSResearchConnector(_ArticleConnector):
    """Normalize already-parsed RSS ``ArticleMeta`` records."""

    name = "rss"
    source_type = "rss_news"

    def __init__(
        self,
        article_provider: Callable[[], Mapping[str, Sequence[ArticleMeta]]] | None = None,
        *,
        now: Callable[[], datetime] | None = None,
        ingestor: EvidenceIngestor | None = None,
    ) -> None:
        super().__init__(now=now, ingestor=ingestor)
        self._article_provider = article_provider

    def fetch(self, checkpoint: ConnectorCheckpoint) -> ConnectorBatch:
        if checkpoint.connector != self.name:
            raise ValueError("checkpoint belongs to another connector")
        if self._article_provider is None:
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="configuration_missing"
            )
        try:
            return self._batch(self._article_provider(), checkpoint)
        except ConnectorError:
            raise
        except Exception:
            raise ConnectorError(
                self.name, retryable=True, diagnostic_code="upstream_fetch_failed"
            ) from None
