"""Strict, source-faithful import for licensed private-market CSV exports."""

from __future__ import annotations

import csv
import io
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation

from .base import (
    ConnectorCheckpoint,
    ConnectorError,
    EmergingSignal,
    SignalConnectorBatch,
    signal_document_id,
)


_FIELDS = (
    "company",
    "profile_date",
    "geography",
    "technology_terms",
    "evidence_passage_id",
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
        self, *, max_bytes: int = 5_000_000, content: str | bytes | None = None
    ) -> None:
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.max_bytes = max_bytes
        self.content = content

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
            records: list[EmergingSignal] = []
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
                records.append(
                    EmergingSignal(
                        source_document_id=signal_document_id(self.name, locator),
                        source_locator=locator,
                        company=_text(row["company"], "company") or "",
                        signal_type="private_market_profile",
                        amount=amount,
                        stage=_text(row["stage"], "stage", required=False),
                        effective_date=date.fromisoformat(
                            _text(row["profile_date"], "profile_date") or ""
                        ),
                        geography=_text(
                            row["geography"], "geography", required=False
                        ),
                        technology_terms=terms,
                        evidence_passage_id=_text(
                            row["evidence_passage_id"], "evidence_passage_id"
                        )
                        or "",
                    )
                )
            return tuple(records)
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
