"""Shared immutable contracts for research-source connectors."""

from __future__ import annotations

import re
import hashlib
import json
import threading
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from ..evidence import (
    DocumentInput,
    EvidenceIngestor,
    IngestedDocument,
    canonicalize_url,
)


_DIAGNOSTIC_CODE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,63}")
_CONNECTOR_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_SIGNAL_TYPE = re.compile(r"[a-z][a-z0-9_]{0,63}")


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


def _normalized_text(value: str, field_name: str) -> str:
    _nonblank(value, field_name)
    return " ".join(unicodedata.normalize("NFC", value).split())


def signal_document_id(connector: str, source_locator: str) -> str:
    """Return a stable opaque source-document identity for one public record."""
    _connector_name(connector)
    locator = _normalized_text(source_locator, "source_locator")
    return "signal-" + hashlib.sha256(
        f"{connector}\0{locator}".encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class EmergingSignal:
    """A source-faithful, evidence-linked emerging-company observation."""

    source_document_id: str
    source_locator: str
    company: str
    signal_type: str
    amount: Decimal | None
    stage: str | None
    effective_date: date
    geography: str | None
    technology_terms: tuple[str, ...]
    evidence_passage_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "source_document_id",
            "source_locator",
            "company",
            "evidence_passage_id",
        ):
            object.__setattr__(
                self, field_name, _normalized_text(getattr(self, field_name), field_name)
            )
        if not isinstance(self.signal_type, str) or _SIGNAL_TYPE.fullmatch(
            self.signal_type
        ) is None:
            raise ValueError("signal_type must be a safe symbolic value")
        if self.amount is not None:
            if not isinstance(self.amount, Decimal):
                raise TypeError("amount must be Decimal or None")
            if not self.amount.is_finite():
                raise ValueError("amount must be finite")
        if self.stage is not None:
            object.__setattr__(self, "stage", _normalized_text(self.stage, "stage"))
        if not isinstance(self.effective_date, date) or isinstance(
            self.effective_date, datetime
        ):
            raise TypeError("effective_date must be date")
        if self.geography is not None:
            object.__setattr__(
                self,
                "geography",
                _normalized_text(self.geography, "geography").casefold(),
            )
        if not isinstance(self.technology_terms, tuple):
            raise TypeError("technology_terms must be a tuple")
        normalized_terms: list[str] = []
        for term in self.technology_terms:
            normalized = _normalized_text(term, "technology_term").casefold()
            if normalized not in normalized_terms:
                normalized_terms.append(normalized)
        object.__setattr__(self, "technology_terms", tuple(normalized_terms))


@dataclass(frozen=True)
class ConnectorStatus:
    """Persistable, redacted availability state for one connector batch."""

    available: bool = True
    retryable: bool = False
    diagnostic_code: str = "ok"

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool) or not isinstance(self.retryable, bool):
            raise TypeError("availability and retryability must be bool")
        if _DIAGNOSTIC_CODE.fullmatch(self.diagnostic_code) is None:
            raise ValueError("diagnostic_code must be a safe symbolic value")
        if self.available and (self.retryable or self.diagnostic_code != "ok"):
            raise ValueError("available status must use the ok diagnostic")


@dataclass(frozen=True)
class SignalConnectorBatch:
    """A signal-specific batch that never coerces records into documents."""

    connector: str
    signals: tuple[EmergingSignal, ...]
    next_checkpoint: ConnectorCheckpoint
    status: ConnectorStatus = ConnectorStatus()

    def __post_init__(self) -> None:
        _connector_name(self.connector)
        if not isinstance(self.signals, tuple) or any(
            not isinstance(signal, EmergingSignal) for signal in self.signals
        ):
            raise TypeError("signals must be an EmergingSignal tuple")
        if self.next_checkpoint.connector != self.connector:
            raise ValueError("checkpoint must belong to signal batch")
        if not isinstance(self.status, ConnectorStatus):
            raise TypeError("status must be ConnectorStatus")
        if not self.status.available and self.signals:
            raise ValueError("unavailable batches cannot contain signals")


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


@runtime_checkable
class SignalResearchConnector(Protocol):
    """Typed checkpoint contract for non-document public-signal sources."""

    name: str

    def fetch(self, checkpoint: ConnectorCheckpoint) -> SignalConnectorBatch:
        """Return emerging signals and the proposed durable checkpoint."""


class ConnectorError(RuntimeError):
    """A redacted connector failure suitable for persistence and retry policy."""

    connector: str
    status: int | None
    retryable: bool
    diagnostic_code: str

    _EXCEPTION_INTERNALS = frozenset(
        {"__traceback__", "__cause__", "__context__", "__suppress_context__"}
    )

    def __setattr__(self, name: str, value) -> None:
        if name not in self._EXCEPTION_INTERNALS:
            from dataclasses import FrozenInstanceError

            raise FrozenInstanceError(f"cannot assign to field {name!r}")
        RuntimeError.__setattr__(self, name, value)

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


def validated_api_base_url(
    value: str, *, host: str, path_prefix: str, field_name: str = "base_url"
) -> str:
    """Validate an immutable HTTPS API origin and its expected root path."""
    try:
        parsed = urlsplit(value)
    except (TypeError, ValueError):
        parsed = None
    if (
        parsed is None
        or parsed.scheme != "https"
        or parsed.hostname != host
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != path_prefix.rstrip("/")
    ):
        raise ValueError(f"{field_name} must be the allowlisted HTTPS API root")
    return value.rstrip("/")


class PacedJSONTransport:
    """Bounded, redirect-free JSON transport with per-instance pacing."""

    def __init__(
        self,
        *,
        connector: str,
        host: str,
        session,
        timeout_seconds: float,
        max_response_bytes: int,
        min_interval_seconds: float,
        clock: Callable[[], float],
        sleeper: Callable[[float], None],
    ) -> None:
        _connector_name(connector)
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not isinstance(max_response_bytes, int) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        if (
            not isinstance(min_interval_seconds, (int, float))
            or min_interval_seconds < 0
        ):
            raise ValueError("min_interval_seconds must be nonnegative")
        self.connector = connector
        self.host = host
        self.session = session
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = max_response_bytes
        self.min_interval_seconds = float(min_interval_seconds)
        self.clock = clock
        self.sleeper = sleeper
        self._pacing_lock = threading.Lock()
        self._last_request_at: float | None = None

    def _pace(self) -> None:
        with self._pacing_lock:
            now = self.clock()
            if self._last_request_at is not None:
                remaining = self.min_interval_seconds - (now - self._last_request_at)
                if remaining > 0:
                    self.sleeper(remaining)
                    now = self.clock()
            self._last_request_at = now

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, object] | None = None,
        json_body: dict[str, object] | None = None,
    ) -> object:
        try:
            requested = urlsplit(url)
        except ValueError:
            requested = None
        if (
            requested is None
            or requested.scheme != "https"
            or requested.hostname != self.host
            or requested.port not in (None, 443)
            or requested.username is not None
            or requested.password is not None
            or requested.fragment
        ):
            raise ConnectorError(
                self.connector,
                retryable=False,
                diagnostic_code="endpoint_disallowed",
            )
        normalized_method = method.upper()
        if normalized_method not in {"GET", "POST"}:
            raise ValueError("method must be GET or POST")
        self._pace()
        request_kwargs: dict[str, object] = {
            "headers": headers or {},
            "timeout": self.timeout_seconds,
            "allow_redirects": False,
            "stream": True,
        }
        if params is not None:
            request_kwargs["params"] = params
        if json_body is not None:
            request_kwargs["json"] = json_body
        try:
            request_method = getattr(self.session, normalized_method.lower())
            response = request_method(url, **request_kwargs)
        except Exception:
            raise ConnectorError(
                self.connector, retryable=True, diagnostic_code="transport_error"
            ) from None
        try:
            try:
                final = urlsplit(getattr(response, "url", ""))
            except ValueError:
                final = None
            if (
                getattr(response, "history", ())
                or final is None
                or final.scheme != "https"
                or final.hostname != self.host
                or final.port not in (None, 443)
            ):
                raise ConnectorError(
                    self.connector,
                    retryable=False,
                    diagnostic_code="redirect_rejected",
                )
            status = int(response.status_code)
            if not 200 <= status < 300:
                raise ConnectorError(
                    self.connector,
                    status=status,
                    retryable=status in {408, 425, 429} or 500 <= status <= 599,
                    diagnostic_code=f"http_{status}",
                )
            raw = bytearray()
            try:
                for chunk in response.iter_content(chunk_size=65_536):
                    raw.extend(chunk)
                    if len(raw) > self.max_response_bytes:
                        raise ConnectorError(
                            self.connector,
                            retryable=False,
                            diagnostic_code="response_too_large",
                        )
            except ConnectorError:
                raise
            except Exception:
                raise ConnectorError(
                    self.connector,
                    retryable=True,
                    diagnostic_code="response_read_failed",
                ) from None
            try:
                payload = json.loads(bytes(raw).decode("utf-8", errors="strict"))
            except (UnicodeError, json.JSONDecodeError):
                raise ConnectorError(
                    self.connector,
                    retryable=False,
                    diagnostic_code="invalid_json",
                ) from None
            if not isinstance(payload, (dict, list)):
                raise ConnectorError(
                    self.connector,
                    retryable=False,
                    diagnostic_code="invalid_payload",
                )
            return payload
        finally:
            try:
                response.close()
            except Exception:
                pass


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
