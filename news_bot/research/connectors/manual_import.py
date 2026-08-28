"""Strict, source-faithful import for licensed private-market CSV exports."""

from __future__ import annotations

import csv
import hashlib
import io
import unicodedata
from collections.abc import Callable
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation

from ..evidence import EvidenceIngestor
from .base import (
    ConnectorCheckpoint,
    ConnectorError,
    EmergingSignal,
    SignalConnectorBatch,
    persist_signal_evidence_batch,
    signal_evidence_input,
)


_FIELDS = (
    "company",
    "profile_date",
    "geography",
    "technology_terms",
    "source_locator",
    "stage",
    "amount",
)
_FORMULA_PREFIXES = ("=", "+", "-", "@")


def _text(value: str | None, field_name: str, *, required: bool = True) -> str | None:
    if not isinstance(value, str):
        if required:
            raise ValueError(f"{field_name} is required")
        return None
    normalized = " ".join(unicodedata.normalize("NFC", value).split())
    if required and not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized or None


class ManualImportConnector:
    """Normalize licensed exports without binding to a proprietary API."""

    name = "manual_import"

    def __init__(
        self,
        *,
        max_bytes: int = 5_000_000,
        content: str | bytes | None = None,
        ingestor: EvidenceIngestor | None = None,
        retrieved_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.max_bytes = max_bytes
        self.content = content
        self.ingestor = ingestor
        self._retrieved_clock = retrieved_clock

    def parse(self, content: str | bytes) -> tuple[EmergingSignal, ...]:
        if isinstance(content, str):
            try:
                raw = content.encode("utf-8", errors="strict")
            except UnicodeError:
                raise ConnectorError(
                    self.name, retryable=False, diagnostic_code="invalid_encoding"
                ) from None
        elif isinstance(content, bytes):
            raw = content
        else:
            raise TypeError("content must be str or bytes")
        if len(raw) > self.max_bytes:
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="input_too_large"
            )
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeError:
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_encoding"
            ) from None
        try:
            reader = csv.DictReader(io.StringIO(text, newline=""), dialect="excel")
            if tuple(reader.fieldnames or ()) != _FIELDS:
                raise ConnectorError(
                    self.name, retryable=False, diagnostic_code="invalid_schema"
                )
            drafts = []
            for row in reader:
                if None in row or set(row) != set(_FIELDS):
                    raise ValueError("row shape does not match schema")
                for value in row.values():
                    if isinstance(value, str) and value.lstrip().startswith(
                        _FORMULA_PREFIXES
                    ):
                        raise ConnectorError(
                            self.name,
                            retryable=False,
                            diagnostic_code="unsafe_cell",
                        )
                locator = _text(row["source_locator"], "source_locator")
                assert locator is not None
                amount_text = _text(row["amount"], "amount", required=False)
                amount = Decimal(amount_text) if amount_text is not None else None
                if amount is not None and not amount.is_finite():
                    raise ValueError("amount is nonfinite")
                terms_text = _text(
                    row["technology_terms"], "technology_terms", required=False
                )
                terms = (
                    tuple(item for item in terms_text.split("|") if item.strip())
                    if terms_text
                    else ()
                )
                effective_date = date.fromisoformat(
                    _text(row["profile_date"], "profile_date") or ""
                )
                company = _text(row["company"], "company") or ""
                stage = _text(row["stage"], "stage", required=False)
                geography = _text(row["geography"], "geography", required=False)
                source = signal_evidence_input(
                    connector=self.name,
                    source_locator=locator,
                    source_url=(
                        "https://local.invalid/manual-import?source="
                        + hashlib.sha256(locator.encode("utf-8")).hexdigest()
                    ),
                    publisher="Licensed manual import",
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
                        signal_type="private_market_profile",
                        amount=amount,
                        stage=stage,
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
                    stage,
                    geography,
                    terms,
                    _,
                ), (document_id, passage_id) in zip(drafts, lineages, strict=True)
            )
        except ConnectorError:
            raise
        except (csv.Error, ValueError, InvalidOperation):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_payload"
            ) from None

    def fetch(self, checkpoint: ConnectorCheckpoint) -> SignalConnectorBatch:
        if checkpoint.connector != self.name:
            raise ValueError("checkpoint belongs to another connector")
        if self.content is None:
            raise ConnectorError(
                self.name,
                retryable=False,
                diagnostic_code="request_context_missing",
            )
        signals = self.parse(self.content)
        return SignalConnectorBatch(self.name, signals, checkpoint)
