"""SBIR/STTR and USAspending public award-signal adapters."""

from __future__ import annotations

import json
import time
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation

import requests

from urllib.parse import quote

from ..evidence import EvidenceIngestor

from .base import (
    ConnectorCheckpoint,
    ConnectorError,
    ConnectorStatus,
    EmergingSignal,
    IncrementalPageCursor,
    PacedJSONTransport,
    SignalConnectorBatch,
    persist_signal_evidence_batch,
    signal_evidence_input,
    validated_api_base_url,
)
from .fmp import install_fmp_log_redaction


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return " ".join(unicodedata.normalize("NFC", value).split())


def _json_value(
    value: str | bytes | Mapping[str, object] | list[object], connector: str
) -> Mapping[str, object] | list[object]:
    if isinstance(value, (Mapping, list)):
        return value
    if not isinstance(value, (str, bytes)):
        raise TypeError("payload must be JSON text, bytes, or a mapping")
    try:
        decoded = json.loads(value)
    except (UnicodeError, json.JSONDecodeError, TypeError):
        raise ConnectorError(
            connector, retryable=False, diagnostic_code="invalid_json"
        ) from None
    if not isinstance(decoded, (dict, list)):
        raise ConnectorError(
            connector, retryable=False, diagnostic_code="invalid_payload"
        )
    return decoded


def _amount(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("amount is invalid")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("amount is invalid") from None
    if not amount.is_finite():
        raise ValueError("amount is nonfinite")
    return amount


@dataclass(frozen=True)
class SBIRConfig:
    api_key: str | None = field(default=None, repr=False)
    base_url: str = "https://api.www.sbir.gov/public/api"
    timeout_seconds: float = 15.0
    max_response_bytes: int = 5_000_000
    min_interval_seconds: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_url",
            validated_api_base_url(
                self.base_url,
                host="api.www.sbir.gov",
                path_prefix="/public/api",
            ),
        )
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or len(self.api_key.strip()) < 8
        ):
            raise ValueError("api_key must be a nonblank configured secret")
        _validate_transport_values(self)


@dataclass(frozen=True)
class USASpendingConfig:
    base_url: str = "https://api.usaspending.gov/api/v2"
    timeout_seconds: float = 15.0
    max_response_bytes: int = 5_000_000
    min_interval_seconds: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_url",
            validated_api_base_url(
                self.base_url, host="api.usaspending.gov", path_prefix="/api/v2"
            ),
        )
        _validate_transport_values(self)


def _validate_transport_values(config: SBIRConfig | USASpendingConfig) -> None:
    if not isinstance(config.timeout_seconds, (int, float)) or config.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not isinstance(config.max_response_bytes, int) or config.max_response_bytes <= 0:
        raise ValueError("max_response_bytes must be positive")
    if (
        not isinstance(config.min_interval_seconds, (int, float))
        or config.min_interval_seconds <= 0
    ):
        raise ValueError("min_interval_seconds must be positive")


class _APIConnector:
    name: str
    host: str

    def _init_transport(
        self,
        *,
        config: SBIRConfig | USASpendingConfig,
        session,
        clock: Callable[[], float],
        sleeper: Callable[[float], None],
    ) -> None:
        self._owns_session = session is None
        self.session = session or requests.Session()
        if self._owns_session:
            self.session.trust_env = False
        self._transport = PacedJSONTransport(
            connector=self.name,
            host=self.host,
            session=self.session,
            timeout_seconds=config.timeout_seconds,
            max_response_bytes=config.max_response_bytes,
            min_interval_seconds=config.min_interval_seconds,
            clock=clock,
            sleeper=sleeper,
        )

    def close(self) -> None:
        if self._owns_session:
            self.session.close()


class SBIRConnector(_APIConnector):
    """SBIR award API with a health-gated public bulk-file fallback."""

    name = "sbir"
    host = "api.www.sbir.gov"

    def __init__(
        self,
        config: SBIRConfig | None = None,
        *,
        session=None,
        query: str | None = None,
        bulk_loader: Callable[[], str | bytes] | None = None,
        ingestor: EvidenceIngestor | None = None,
        retrieved_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or SBIRConfig()
        self.query = _text(query, "query") if query is not None else None
        self.bulk_loader = bulk_loader
        self.ingestor = ingestor
        self._retrieved_clock = retrieved_clock
        install_fmp_log_redaction(self.config.api_key)
        self._init_transport(
            config=self.config,
            session=session,
            clock=clock,
            sleeper=sleeper,
        )

    def parse(
        self, payload: str | bytes | Mapping[str, object]
    ) -> tuple[EmergingSignal, ...]:
        data = _json_value(payload, self.name)
        try:
            rows = data["awards"] if isinstance(data, Mapping) else data
            if not isinstance(rows, list):
                raise ValueError("awards must be a list")
            drafts = []
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError("award must be an object")
                keywords = row.get("research_area_keywords")
                terms = (
                    tuple(
                        item.strip()
                        for item in _text(
                            keywords, "research_area_keywords"
                        ).replace(",", ";").split(";")
                        if item.strip()
                    )
                    if keywords is not None
                    else ()
                )
                award_id = _text(row.get("contract"), "contract")
                locator = f"sbir-award:{award_id}"
                geography_value = row.get("state")
                geography = (
                    _text(geography_value, "state")
                    if geography_value is not None
                    else None
                )
                stage_value = row.get("phase")
                stage = (
                    _text(stage_value, "phase")
                    if stage_value is not None
                    else None
                )
                effective_date = date.fromisoformat(
                    _text(
                        row.get("proposal_award_date"),
                        "proposal_award_date",
                    )
                )
                company = _text(row.get("firm"), "firm")
                amount = _amount(row.get("award_amount"))
                source = signal_evidence_input(
                    connector=self.name,
                    source_locator=locator,
                    source_url=(
                        f"{self.config.base_url}/awards?contract={quote(award_id)}"
                    ),
                    publisher="Small Business Innovation Research",
                    effective_date=effective_date,
                    raw_record=row,
                    retrieved_at=self._retrieved_clock(),
                )
                drafts.append(
                    (
                        locator,
                        effective_date,
                        company,
                        amount,
                        stage,
                        geography,
                        terms,
                        source,
                    )
                )
            lineages = persist_signal_evidence_batch(
                self.ingestor,
                connector=self.name,
                sources=tuple(draft[7] for draft in drafts),
            )
            return tuple(
                    EmergingSignal(
                        source_document_id=document_id,
                        source_locator=locator,
                        company=company,
                        signal_type="grant",
                        amount=amount,
                        stage=stage,
                        effective_date=effective_date,
                        geography=geography,
                        technology_terms=tuple(terms),
                        evidence_passage_id=passage_id,
                    )
                for (
                    locator,
                    effective_date,
                    company,
                    amount,
                    stage,
                    geography,
                    terms,
                    _,
                ), (document_id, passage_id) in zip(drafts, lineages, strict=True)
            )
        except ConnectorError:
            raise
        except (KeyError, TypeError, ValueError):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_payload"
            ) from None

    def _bulk_batch(self, checkpoint: ConnectorCheckpoint) -> SignalConnectorBatch:
        if self.bulk_loader is None:
            return SignalConnectorBatch(
                self.name,
                (),
                checkpoint,
                ConnectorStatus(False, True, "maintenance"),
            )
        try:
            bulk = self.bulk_loader()
            if isinstance(bulk, str):
                try:
                    raw = bulk.encode("utf-8", errors="strict")
                except UnicodeError:
                    raise ConnectorError(
                        self.name,
                        retryable=False,
                        diagnostic_code="invalid_encoding",
                    ) from None
            elif isinstance(bulk, bytes):
                raw = bulk
            else:
                raise ConnectorError(
                    self.name,
                    retryable=False,
                    diagnostic_code="invalid_bulk_payload",
                )
            if len(raw) > self.config.max_response_bytes:
                raise ConnectorError(
                    self.name,
                    retryable=False,
                    diagnostic_code="bulk_too_large",
                )
            signals = self.parse(raw)
        except ConnectorError:
            raise
        except Exception:
            raise ConnectorError(
                self.name, retryable=True, diagnostic_code="bulk_read_failed"
            ) from None
        return SignalConnectorBatch(self.name, signals, checkpoint)

    def fetch(self, checkpoint: ConnectorCheckpoint) -> SignalConnectorBatch:
        if checkpoint.connector != self.name:
            raise ValueError("checkpoint belongs to another connector")
        headers = {"Accept": "application/json"}
        if self.config.api_key is not None:
            headers["X-API-KEY"] = self.config.api_key
        try:
            start = int(checkpoint.cursor) if checkpoint.cursor is not None else 0
        except ValueError:
            raise ValueError("checkpoint cursor must be a numeric offset") from None
        if start < 0:
            raise ValueError("checkpoint cursor must be nonnegative")
        params: dict[str, object] = {"rows": 100, "start": start}
        if self.query is not None:
            params["keyword"] = self.query
        if checkpoint.cursor is not None:
            params["cursor"] = checkpoint.cursor
        try:
            payload = self._transport.request(
                "GET",
                f"{self.config.base_url}/awards",
                headers=headers,
                params=params,
            )
        except ConnectorError as error:
            if error.status in {401, 403}:
                raise
            if error.status in {502, 503, 504} or (
                error.status is None and error.retryable
            ):
                return self._bulk_batch(checkpoint)
            raise
        if not isinstance(payload, (Mapping, list)):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_payload"
            )
        signals = self.parse(payload)
        if not signals:
            return SignalConnectorBatch(self.name, (), checkpoint)
        cursor = str(start + len(signals))
        return SignalConnectorBatch(
            self.name, signals, ConnectorCheckpoint(self.name, cursor=cursor)
        )


class USASpendingConnector(_APIConnector):
    """Normalize the documented USAspending spending-by-award response."""

    name = "usaspending"
    host = "api.usaspending.gov"

    def __init__(
        self,
        config: USASpendingConfig | None = None,
        *,
        session=None,
        query: str | None = None,
        ingestor: EvidenceIngestor | None = None,
        retrieved_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or USASpendingConfig()
        self.query = _text(query, "query") if query is not None else None
        self.ingestor = ingestor
        self._retrieved_clock = retrieved_clock
        self._init_transport(
            config=self.config,
            session=session,
            clock=clock,
            sleeper=sleeper,
        )

    def parse(
        self, payload: str | bytes | Mapping[str, object]
    ) -> tuple[EmergingSignal, ...]:
        data = _json_value(payload, self.name)
        try:
            if not isinstance(data, Mapping):
                raise ValueError("payload must be an object")
            rows = data["results"]
            if not isinstance(rows, list):
                raise ValueError("results must be a list")
            drafts = []
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError("award must be an object")
                terms = row.get("Technology Terms", [])
                if not isinstance(terms, list) or any(
                    not isinstance(term, str) or not term.strip() for term in terms
                ):
                    raise ValueError("Technology Terms must be a text list")
                location_value = row.get("Recipient Location")
                geography = (
                    _text(location_value, "Recipient Location")
                    if location_value is not None
                    else None
                )
                award_id = _text(
                    row.get("Award ID"),
                    "Award ID",
                )
                locator = f"usaspending-award:{award_id}"
                effective_date = date.fromisoformat(
                    _text(row.get("Start Date"), "Start Date")
                )
                company = _text(row.get("Recipient Name"), "Recipient Name")
                amount = _amount(row.get("Award Amount"))
                source = signal_evidence_input(
                    connector=self.name,
                    source_locator=locator,
                    source_url=(
                        f"{self.config.base_url}/awards/{quote(award_id, safe='')}/"
                    ),
                    publisher="USAspending.gov",
                    effective_date=effective_date,
                    raw_record=row,
                    retrieved_at=self._retrieved_clock(),
                )
                drafts.append(
                    (
                        locator,
                        effective_date,
                        company,
                        amount,
                        geography,
                        tuple(terms),
                        source,
                    )
                )
            lineages = persist_signal_evidence_batch(
                self.ingestor,
                connector=self.name,
                sources=tuple(draft[6] for draft in drafts),
            )
            return tuple(
                    EmergingSignal(
                        source_document_id=document_id,
                        source_locator=locator,
                        company=company,
                        signal_type="contract",
                        amount=amount,
                        stage=None,
                        effective_date=effective_date,
                        geography=geography,
                        technology_terms=terms,
                        evidence_passage_id=passage_id,
                    )
                for (
                    locator,
                    effective_date,
                    company,
                    amount,
                    geography,
                    terms,
                    _,
                ), (document_id, passage_id) in zip(drafts, lineages, strict=True)
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
        if self.query is None:
            raise ConnectorError(
                self.name,
                retryable=False,
                diagnostic_code="request_context_missing",
            )
        cursor = IncrementalPageCursor.parse(
            checkpoint.cursor, connector=self.name
        )
        page = 1
        if cursor.continuation is not None:
            try:
                page = int(cursor.continuation)
            except (TypeError, ValueError):
                raise ConnectorError(
                    self.name,
                    retryable=False,
                    diagnostic_code="invalid_checkpoint",
                ) from None
            if page < 1:
                raise ConnectorError(
                    self.name,
                    retryable=False,
                    diagnostic_code="invalid_checkpoint",
                )
        body: dict[str, object] = {
            "filters": {"keywords": [self.query]},
            "fields": [
                "Award ID",
                "Recipient Name",
                "Award Amount",
                "Start Date",
                "Recipient Location",
                "Description",
            ],
            "page": page,
            "limit": 100,
            "subawards": False,
            "sort": "Start Date",
            "order": "desc",
        }
        try:
            payload = self._transport.request(
                "POST",
                f"{self.config.base_url}/search/spending_by_award/",
                headers={"Accept": "application/json"},
                json_body=body,
            )
        except ConnectorError as error:
            if error.status == 503:
                return SignalConnectorBatch(
                    self.name,
                    (),
                    checkpoint,
                    ConnectorStatus(False, True, "maintenance"),
                )
            raise
        if not isinstance(payload, Mapping):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_payload"
            )
        try:
            rows = payload.get("results")
            metadata = payload.get("page_metadata")
            if not isinstance(rows, list) or not isinstance(metadata, Mapping):
                raise ValueError("pagination payload shape is invalid")
            if not rows:
                return SignalConnectorBatch(self.name, (), checkpoint)
            markers = tuple(
                (
                    date.fromisoformat(_text(row.get("Start Date"), "Start Date")),
                    _text(row.get("Award ID"), "Award ID"),
                )
                for row in rows
                if isinstance(row, Mapping)
            )
            if len(markers) != len(rows):
                raise ValueError("award must be an object")
            filtered = tuple(
                row
                for row, marker in zip(rows, markers, strict=True)
                if cursor.is_new(marker)
            )
            next_page = metadata.get("next")
            if next_page is None:
                continuation = None
            elif (
                isinstance(next_page, int)
                and not isinstance(next_page, bool)
                and next_page > 0
            ):
                continuation = str(next_page)
            else:
                raise ValueError("next page is invalid")
        except (TypeError, ValueError):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_payload"
            ) from None
        normalized_payload = dict(payload)
        normalized_payload["results"] = list(filtered)
        signals = self.parse(normalized_payload)
        next_cursor = cursor.advance(markers, continuation=continuation)
        return SignalConnectorBatch(
            self.name,
            signals,
            ConnectorCheckpoint(self.name, cursor=next_cursor.encode()),
        )
