"""SEC Form D quarterly TSV and bounded ZIP normalization."""

from __future__ import annotations

import csv
import io
import unicodedata
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath

from .base import (
    ConnectorCheckpoint,
    ConnectorError,
    EmergingSignal,
    SignalConnectorBatch,
    signal_document_id,
)


_FIELDS = (
    "ACCESSIONNUMBER",
    "ENTITYNAME",
    "TOTALOFFERINGAMOUNT",
    "FILED",
    "STATEORCOUNTRY",
    "INDUSTRYGROUP",
    "EVIDENCEPASSAGEID",
)


@dataclass(frozen=True)
class FormDConfig:
    max_archive_bytes: int = 20_000_000
    max_member_bytes: int = 100_000_000
    max_compression_ratio: int = 100

    def __post_init__(self) -> None:
        for field_name in (
            "max_archive_bytes",
            "max_member_bytes",
            "max_compression_ratio",
        ):
            if not isinstance(getattr(self, field_name), int) or getattr(
                self, field_name
            ) <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


def _decode(value: str | bytes, *, max_bytes: int, connector: str) -> str:
    if isinstance(value, str):
        try:
            raw = value.encode("utf-8", errors="strict")
        except UnicodeError:
            raise ConnectorError(
                connector, retryable=False, diagnostic_code="invalid_encoding"
            ) from None
    elif isinstance(value, bytes):
        raw = value
    else:
        raise TypeError("content must be str or bytes")
    if len(raw) > max_bytes:
        raise ConnectorError(
            connector, retryable=False, diagnostic_code="input_too_large"
        )
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeError:
        raise ConnectorError(
            connector, retryable=False, diagnostic_code="invalid_encoding"
        ) from None


def _required(row: dict[str, str | None], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return " ".join(unicodedata.normalize("NFC", value).split())


class FormDConnector:
    """Normalize the SEC's published quarterly Form D tabular export."""

    name = "form_d"

    def __init__(
        self,
        config: FormDConfig | None = None,
        *,
        archive_loader: Callable[
            [ConnectorCheckpoint], tuple[bytes, str | None]
        ]
        | None = None,
    ) -> None:
        self.config = config or FormDConfig()
        self.archive_loader = archive_loader

    def parse(self, content: str | bytes) -> tuple[EmergingSignal, ...]:
        text = _decode(
            content, max_bytes=self.config.max_member_bytes, connector=self.name
        )
        try:
            reader = csv.DictReader(io.StringIO(text, newline=""), dialect="excel-tab")
            if tuple(reader.fieldnames or ()) != _FIELDS:
                raise ConnectorError(
                    self.name, retryable=False, diagnostic_code="invalid_schema"
                )
            records: list[EmergingSignal] = []
            for row in reader:
                if None in row or set(row) != set(_FIELDS):
                    raise ValueError("row shape does not match schema")
                accession = _required(row, "ACCESSIONNUMBER")
                locator = f"sec-form-d:{accession}"
                amount_text = (row.get("TOTALOFFERINGAMOUNT") or "").strip()
                amount = Decimal(amount_text) if amount_text else None
                if amount is not None and not amount.is_finite():
                    raise ValueError("amount is nonfinite")
                records.append(
                    EmergingSignal(
                        source_document_id=signal_document_id(self.name, locator),
                        source_locator=locator,
                        company=_required(row, "ENTITYNAME"),
                        signal_type="funding",
                        amount=amount,
                        stage=None,
                        effective_date=date.fromisoformat(_required(row, "FILED")),
                        geography=_required(row, "STATEORCOUNTRY"),
                        technology_terms=(_required(row, "INDUSTRYGROUP"),),
                        evidence_passage_id=_required(row, "EVIDENCEPASSAGEID"),
                    )
                )
            return tuple(records)
        except ConnectorError:
            raise
        except (csv.Error, ValueError, InvalidOperation):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_payload"
            ) from None

    def parse_zip(self, archive_bytes: bytes) -> tuple[EmergingSignal, ...]:
        if not isinstance(archive_bytes, bytes):
            raise TypeError("archive_bytes must be bytes")
        if len(archive_bytes) > self.config.max_archive_bytes:
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="archive_too_large"
            )
        try:
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                members = [item for item in archive.infolist() if not item.is_dir()]
                if len(members) != 1:
                    raise ConnectorError(
                        self.name, retryable=False, diagnostic_code="invalid_archive"
                    )
                member = members[0]
                path = PurePosixPath(member.filename)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or path.suffix.casefold() != ".tsv"
                ):
                    raise ConnectorError(
                        self.name, retryable=False, diagnostic_code="invalid_archive"
                    )
                if member.file_size > self.config.max_member_bytes:
                    raise ConnectorError(
                        self.name,
                        retryable=False,
                        diagnostic_code="archive_member_too_large",
                    )
                if member.compress_size == 0:
                    if member.file_size:
                        raise ConnectorError(
                            self.name,
                            retryable=False,
                            diagnostic_code="archive_ratio_exceeded",
                        )
                elif (
                    member.file_size / member.compress_size
                    > self.config.max_compression_ratio
                ):
                    raise ConnectorError(
                        self.name,
                        retryable=False,
                        diagnostic_code="archive_ratio_exceeded",
                    )
                raw = bytearray()
                with archive.open(member, "r") as stream:
                    while chunk := stream.read(65_536):
                        raw.extend(chunk)
                        if len(raw) > self.config.max_member_bytes:
                            raise ConnectorError(
                                self.name,
                                retryable=False,
                                diagnostic_code="archive_member_too_large",
                            )
                text = _decode(
                    bytes(raw),
                    max_bytes=self.config.max_member_bytes,
                    connector=self.name,
                )
        except ConnectorError:
            raise
        except (zipfile.BadZipFile, OSError, RuntimeError):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_archive"
            ) from None
        return self.parse(text)

    def fetch(self, checkpoint: ConnectorCheckpoint) -> SignalConnectorBatch:
        if checkpoint.connector != self.name:
            raise ValueError("checkpoint belongs to another connector")
        if self.archive_loader is None:
            raise ConnectorError(
                self.name,
                retryable=False,
                diagnostic_code="request_context_missing",
            )
        try:
            archive, proposed_cursor = self.archive_loader(checkpoint)
        except ConnectorError:
            raise
        except Exception:
            raise ConnectorError(
                self.name, retryable=True, diagnostic_code="archive_read_failed"
            ) from None
        if proposed_cursor is not None and (
            not isinstance(proposed_cursor, str) or not proposed_cursor.strip()
        ):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_checkpoint"
            )
        signals = self.parse_zip(archive)
        if not signals:
            return SignalConnectorBatch(self.name, (), checkpoint)
        return SignalConnectorBatch(
            self.name,
            signals,
            ConnectorCheckpoint(
                self.name,
                cursor=(
                    proposed_cursor.strip()
                    if proposed_cursor is not None
                    else checkpoint.cursor
                ),
            ),
        )
