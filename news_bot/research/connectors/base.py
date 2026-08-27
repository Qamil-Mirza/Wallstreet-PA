"""Shared immutable contracts for research-source connectors."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from ..evidence import (
    DocumentInput,
    EvidenceIngestor,
    IngestedDocument,
    canonicalize_url,
)


_DIAGNOSTIC_CODE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,63}")
_CONNECTOR_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}")


def _nonblank(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ValueError(f"{field_name} must be nonblank safe text")


def _connector_name(value: str) -> None:
    if not isinstance(value, str) or _CONNECTOR_NAME.fullmatch(value) is None:
        raise ValueError("connector must be a safe symbolic name")


@dataclass(frozen=True)
class ConnectorCheckpoint:
    """A connector cursor that becomes durable only after its batch does."""

    connector: str
    cursor: str | None = None
    etag: str | None = None
    last_modified: str | None = None

    def __post_init__(self) -> None:
        _connector_name(self.connector)
        for field_name in ("cursor", "etag", "last_modified"):
            value = getattr(self, field_name)
            if value is not None:
                _nonblank(value, field_name)
                if len(value) > 512:
                    raise ValueError(f"{field_name} is too long")


@dataclass(frozen=True)
class NormalizedResearchDocument:
    """Evidence input plus deterministic connector classification tags."""

    evidence: DocumentInput
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, DocumentInput):
            raise TypeError("evidence must be DocumentInput")
        if not isinstance(self.tags, tuple):
            raise TypeError("tags must be a tuple")
        normalized: list[str] = []
        for tag in self.tags:
            _nonblank(tag, "tag")
            if tag not in normalized:
                normalized.append(tag)
        if tuple(normalized) != self.tags:
            raise ValueError("tags must be unique and ordered")

    @property
    def canonical_url(self) -> str:
        return canonicalize_url(self.evidence.url)

    @property
    def publisher(self) -> str:
        return self.evidence.publisher

    @property
    def published_at(self) -> datetime:
        return self.evidence.published_at


@dataclass(frozen=True)
class ConnectorBatch:
    """A complete normalized fetch result and its proposed next cursor."""

    connector: str
    documents: tuple[NormalizedResearchDocument, ...]
    next_checkpoint: ConnectorCheckpoint

    def __post_init__(self) -> None:
        _connector_name(self.connector)
        if not isinstance(self.documents, tuple) or any(
            not isinstance(document, NormalizedResearchDocument)
            for document in self.documents
        ):
            raise TypeError("documents must be normalized document tuple")
        if self.next_checkpoint.connector != self.connector:
            raise ValueError("checkpoint must belong to connector batch")


@runtime_checkable
class ResearchConnector(Protocol):
    """Provider-neutral pull contract implemented by source adapters."""

    name: str

    def fetch(self, checkpoint: ConnectorCheckpoint) -> ConnectorBatch:
        """Return normalized documents and the proposed durable checkpoint."""


@dataclass(frozen=True, init=False)
class ConnectorError(RuntimeError):
    """A redacted connector failure suitable for persistence and retry policy."""

    connector: str
    status: int | None
    retryable: bool
    diagnostic_code: str

    def __init__(
        self,
        connector: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        diagnostic_code: str,
    ) -> None:
        _connector_name(connector)
        if status is not None and (
            not isinstance(status, int) or not 100 <= status <= 599
        ):
            raise ValueError("status must be an HTTP status or None")
        if not isinstance(retryable, bool):
            raise TypeError("retryable must be bool")
        if _DIAGNOSTIC_CODE.fullmatch(diagnostic_code) is None:
            raise ValueError("diagnostic_code must be a safe symbolic value")
        object.__setattr__(self, "connector", connector)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "retryable", retryable)
        object.__setattr__(self, "diagnostic_code", diagnostic_code)
        details = diagnostic_code
        if status is not None:
            details += f", status={status}"
        details += f", retryable={str(retryable).lower()}"
        super().__init__(f"{connector} connector failed ({details})")


def commit_connector_batch(
    batch: ConnectorBatch,
    *,
    persist_documents: Callable[[tuple[NormalizedResearchDocument, ...]], None]
    | None = None,
    ingestor: EvidenceIngestor | None = None,
    persist_checkpoint: Callable[[ConnectorCheckpoint], None],
) -> tuple[IngestedDocument, ...]:
    """Persist a complete batch before making its proposed checkpoint durable.

    ``persist_documents`` is the coordinator's atomic storage boundary. If it
    fails, this function deliberately never invokes ``persist_checkpoint``.
    """

    if not isinstance(batch, ConnectorBatch):
        raise TypeError("batch must be ConnectorBatch")
    if (persist_documents is None) == (ingestor is None):
        raise ValueError("provide exactly one document persistence boundary")
    if ingestor is not None:
        ingested = tuple(ingestor.ingest(document.evidence) for document in batch.documents)
    else:
        assert persist_documents is not None
        persist_documents(batch.documents)
        ingested = ()
    persist_checkpoint(batch.next_checkpoint)
    return ingested
