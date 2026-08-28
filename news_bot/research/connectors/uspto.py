"""Authenticated USPTO Open Data Portal patent-signal adapter."""

from __future__ import annotations

import json
import time
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import requests

from ..evidence import EvidenceIngestor

from .base import (
    ConnectorCheckpoint,
    ConnectorError,
    ConnectorStatus,
    EmergingSignal,
    PacedJSONTransport,
    SignalConnectorBatch,
    persist_signal_evidence_batch,
    signal_evidence_input,
    validated_api_base_url,
)
from .fmp import install_fmp_log_redaction


_HOST = "api.uspto.gov"
_ROOT = "/api/v1/patent/applications"


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return " ".join(unicodedata.normalize("NFC", value).split())


def _payload(value: str | bytes | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, (str, bytes)):
        raise TypeError("payload must be JSON text, bytes, or a mapping")
    try:
        decoded = json.loads(value)
    except (UnicodeError, json.JSONDecodeError, TypeError):
        raise ConnectorError(
            "uspto", retryable=False, diagnostic_code="invalid_json"
        ) from None
    if not isinstance(decoded, dict):
        raise ConnectorError(
            "uspto", retryable=False, diagnostic_code="invalid_payload"
        )
    return decoded


@dataclass(frozen=True)
class USPTOConfig:
    api_key: str | None = field(default=None, repr=False)
    base_url: str = "https://api.uspto.gov/api/v1/patent/applications"
    timeout_seconds: float = 15.0
    max_response_bytes: int = 5_000_000
    min_interval_seconds: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_url",
            validated_api_base_url(
                self.base_url, host=_HOST, path_prefix=_ROOT
            ),
        )
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or len(self.api_key.strip()) < 8
        ):
            raise ValueError("api_key must be a nonblank configured secret")
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not isinstance(self.max_response_bytes, int) or self.max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        if (
            not isinstance(self.min_interval_seconds, (int, float))
            or self.min_interval_seconds <= 0
        ):
            raise ValueError("min_interval_seconds must be positive")


class USPTOConnector:
    """Fetch and normalize patent applications without exposing the API key."""

    name = "uspto"

    def __init__(
        self,
        config: USPTOConfig | None = None,
        *,
        session=None,
        query: str | None = None,
        ingestor: EvidenceIngestor | None = None,
        retrieved_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or USPTOConfig()
        self.query = _text(query, "query") if query is not None else None
        self.ingestor = ingestor
        self._retrieved_clock = retrieved_clock
        install_fmp_log_redaction(self.config.api_key)
        self._owns_session = session is None
        self.session = session or requests.Session()
        if self._owns_session:
            self.session.trust_env = False
        self._transport = PacedJSONTransport(
            connector=self.name,
            host=_HOST,
            session=self.session,
            timeout_seconds=self.config.timeout_seconds,
            max_response_bytes=self.config.max_response_bytes,
            min_interval_seconds=self.config.min_interval_seconds,
            clock=clock,
            sleeper=sleeper,
        )

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def parse(
        self, payload: str | bytes | Mapping[str, object]
    ) -> tuple[EmergingSignal, ...]:
        data = _payload(payload)
        try:
            rows = data["patentFileWrapperDataBag"]
            if not isinstance(rows, list):
                raise ValueError("results must be a list")
            drafts = []
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError("result must be an object")
                application = _text(
                    row.get("applicationNumberText"), "applicationNumberText"
                )
                metadata = row.get("applicationMetaData")
                if not isinstance(metadata, Mapping):
                    raise ValueError("applicationMetaData must be an object")
                technology_center = metadata.get("technologyCenter")
                terms = (
                    (_text(technology_center, "technologyCenter"),)
                    if technology_center is not None
                    else ()
                )
                locator = f"uspto-application:{application}"
                effective_date = date.fromisoformat(
                    _text(metadata.get("filingDate"), "filingDate")
                )
                company = _text(
                    metadata.get("firstApplicantName"), "firstApplicantName"
                )
                source = signal_evidence_input(
                    connector=self.name,
                    source_locator=locator,
                    source_url=f"{self.config.base_url}/{application}",
                    publisher="United States Patent and Trademark Office",
                    effective_date=effective_date,
                    raw_record=row,
                    retrieved_at=self._retrieved_clock(),
                )
                drafts.append((locator, effective_date, company, terms, source))
            lineages = persist_signal_evidence_batch(
                self.ingestor,
                connector=self.name,
                sources=tuple(draft[4] for draft in drafts),
            )
            return tuple(
                    EmergingSignal(
                        source_document_id=document_id,
                        source_locator=locator,
                        company=company,
                        signal_type="patent",
                        amount=None,
                        stage=None,
                        effective_date=effective_date,
                        geography=None,
                        technology_terms=terms,
                        evidence_passage_id=passage_id,
                    )
                for (locator, effective_date, company, terms, _),
                (document_id, passage_id) in zip(drafts, lineages, strict=True)
            )
        except ConnectorError:
            raise
        except (KeyError, TypeError, ValueError):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_payload"
            ) from None

    def fetch(self, checkpoint: ConnectorCheckpoint) -> SignalConnectorBatch:
        if checkpoint.connector != self.name:
            raise ValueError("checkpoint belongs to another connector")
        if self.config.api_key is None:
            raise ConnectorError(
                self.name,
                retryable=False,
                diagnostic_code="configuration_missing",
            )
        if self.query is None:
            raise ConnectorError(
                self.name,
                retryable=False,
                diagnostic_code="request_context_missing",
            )
        try:
            start = int(checkpoint.cursor) if checkpoint.cursor is not None else 0
        except ValueError:
            raise ValueError("checkpoint cursor must be a numeric offset") from None
        if start < 0:
            raise ValueError("checkpoint cursor must be nonnegative")
        params: dict[str, object] = {"q": self.query, "start": start, "rows": 100}
        try:
            payload = self._transport.request(
                "GET",
                f"{self.config.base_url}/search",
                headers={"X-API-KEY": self.config.api_key, "Accept": "application/json"},
                params=params,
            )
        except ConnectorError as error:
            if error.status == 503:
                return SignalConnectorBatch(
                    self.name,
                    (),
                    checkpoint,
                    ConnectorStatus(
                        available=False,
                        retryable=True,
                        diagnostic_code="maintenance",
                    ),
                )
            raise
        if not isinstance(payload, Mapping):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_payload"
            )
        signals = self.parse(payload)
        if not signals:
            return SignalConnectorBatch(self.name, (), checkpoint)
        count = payload.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < len(signals):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_payload"
            )
        next_cursor = str(start + len(signals))
        return SignalConnectorBatch(
            self.name,
            signals,
            ConnectorCheckpoint(self.name, cursor=next_cursor),
        )
