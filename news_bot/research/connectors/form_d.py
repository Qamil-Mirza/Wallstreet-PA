"""SEC Form D quarterly six-table archive normalization."""

from __future__ import annotations

import csv
import io
import stat
import re
import unicodedata
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from urllib.parse import quote

from ..evidence import DocumentInput, EvidenceIngestor
from .base import (
    ConnectorCheckpoint,
    ConnectorError,
    EmergingSignal,
    SignalConnectorBatch,
    persist_signal_evidence_batch,
    signal_evidence_input,
)


_TABLES = frozenset(
    {"FORMDSUBMISSION", "ISSUERS", "OFFERING", "RECIPIENTS", "RELATEDPERSONS", "SIGNATURES"}
)
_REQUIRED_COLUMNS = {
    "FORMDSUBMISSION": frozenset({"ACCESSIONNUMBER", "FILING_DATE"}),
    "ISSUERS": frozenset({"ACCESSIONNUMBER", "ISSUER_SEQ_KEY", "CIK", "ENTITYNAME", "STATEORCOUNTRY"}),
    "OFFERING": frozenset({"ACCESSIONNUMBER", "TOTALOFFERINGAMOUNT", "INDUSTRYGROUPTYPE"}),
    "RECIPIENTS": frozenset({"ACCESSIONNUMBER", "RECIPIENT_SEQ_KEY"}),
    "RELATEDPERSONS": frozenset({"ACCESSIONNUMBER", "RELATEDPERSON_SEQ_KEY"}),
    "SIGNATURES": frozenset({"ACCESSIONNUMBER", "SIGNATURE_SEQ_KEY"}),
}
_DIRECT_FIELDS = (
    "ACCESSIONNUMBER", "ENTITYNAME", "TOTALOFFERINGAMOUNT", "FILED",
    "STATEORCOUNTRY", "INDUSTRYGROUP",
)


@dataclass(frozen=True)
class FormDConfig:
    max_archive_bytes: int = 20_000_000
    max_member_bytes: int = 100_000_000
    max_total_uncompressed_bytes: int = 300_000_000
    max_compression_ratio: int = 100

    def __post_init__(self) -> None:
        for field_name in (
            "max_archive_bytes", "max_member_bytes",
            "max_total_uncompressed_bytes", "max_compression_ratio",
        ):
            if not isinstance(getattr(self, field_name), int) or getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True)
class _FormDSignalDraft:
    locator: str
    company: str
    amount: Decimal | None
    effective_date: date
    geography: str
    industry: str
    source: DocumentInput


def _decode(value: str | bytes, *, max_bytes: int, connector: str) -> str:
    if isinstance(value, str):
        try:
            raw = value.encode("utf-8", errors="strict")
        except UnicodeError:
            raise ConnectorError(connector, retryable=False, diagnostic_code="invalid_encoding") from None
    elif isinstance(value, bytes):
        raw = value
    else:
        raise TypeError("content must be str or bytes")
    if len(raw) > max_bytes:
        raise ConnectorError(connector, retryable=False, diagnostic_code="input_too_large")
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeError:
        raise ConnectorError(connector, retryable=False, diagnostic_code="invalid_encoding") from None


def _required(row: Mapping[str, str | None], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return " ".join(unicodedata.normalize("NFC", value).split())


def _parse_table(text: str, *, table: str, required: frozenset[str]) -> tuple[dict[str, str], ...]:
    reader = csv.DictReader(io.StringIO(text, newline=""), dialect="excel-tab")
    fields = tuple(reader.fieldnames or ())
    if len(fields) != len(set(fields)) or not required.issubset(fields):
        raise ConnectorError("form_d", retryable=False, diagnostic_code="invalid_schema")
    rows = []
    for source_row in reader:
        if None in source_row or set(source_row) != set(fields):
            raise ValueError(f"{table} row shape does not match schema")
        row = {key: value or "" for key, value in source_row.items()}
        _required(row, "ACCESSIONNUMBER")
        rows.append(row)
    return tuple(rows)


_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _filing_date(submission: Mapping[str, str]) -> date:
    if "FILING_DATE" not in submission:
        return date.fromisoformat(_required(submission, "FILED"))
    value = _required(submission, "FILING_DATE").upper()
    match = re.fullmatch(r"(\d{2})-([A-Z]{3})-(\d{2})", value)
    if match is None or match.group(2) not in _MONTHS:
        raise ValueError("FILING_DATE is invalid")
    return date(2000 + int(match.group(3)), _MONTHS[match.group(2)], int(match.group(1)))


class FormDConnector:
    """Normalize official SEC quarterly Form D archives and licensed extracts."""

    name = "form_d"

    def __init__(
        self, config: FormDConfig | None = None, *,
        archive_loader: Callable[[ConnectorCheckpoint], tuple[bytes, str | None]] | None = None,
        ingestor: EvidenceIngestor | None = None,
        retrieved_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.config = config or FormDConfig()
        self.archive_loader = archive_loader
        self.ingestor = ingestor
        self._retrieved_clock = retrieved_clock

    def _draft(
        self, *, accession: str, issuer: Mapping[str, str], offering: Mapping[str, str],
        submission: Mapping[str, str], raw_record: Mapping[str, object],
    ) -> _FormDSignalDraft:
        sequence = issuer.get("ISSUER_SEQ_KEY", "").strip()
        locator = (
            f"sec-form-d:{accession}:issuer:{sequence}"
            if sequence
            else f"sec-form-d:{accession}"
        )
        amount_text = (offering.get("TOTALOFFERINGAMOUNT") or "").strip()
        amount = (
            None
            if not amount_text or amount_text.casefold() == "indefinite"
            else Decimal(amount_text)
        )
        if amount is not None and not amount.is_finite():
            raise ValueError("amount is nonfinite")
        effective_date = _filing_date(submission)
        company = _required(issuer, "ENTITYNAME")
        geography = _required(issuer, "STATEORCOUNTRY")
        industry_field = "INDUSTRYGROUPTYPE" if "INDUSTRYGROUPTYPE" in offering else "INDUSTRYGROUP"
        industry = _required(offering, industry_field)
        cik = issuer.get("CIK", "").strip()
        if cik:
            if not cik.isdigit() or len(cik) > 10:
                raise ValueError("CIK is invalid")
            source_url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{accession.replace('-', '')}/"
            )
        else:
            source_url = (
                "https://www.sec.gov/files/dera/data/form-d-data-sets"
                f"?accession={quote(accession)}"
            )
        source = signal_evidence_input(
            connector=self.name, source_locator=locator,
            source_url=source_url,
            publisher="U.S. Securities and Exchange Commission",
            effective_date=effective_date, raw_record=raw_record,
            retrieved_at=self._retrieved_clock(),
        )
        return _FormDSignalDraft(
            locator=locator,
            company=company,
            amount=amount,
            effective_date=effective_date,
            geography=geography,
            industry=industry,
            source=source,
        )

    def _persist_drafts(
        self, drafts: list[_FormDSignalDraft]
    ) -> tuple[EmergingSignal, ...]:
        lineages = persist_signal_evidence_batch(
            self.ingestor,
            connector=self.name,
            sources=tuple(draft.source for draft in drafts),
        )
        return tuple(
            EmergingSignal(
                source_document_id=document_id,
                source_locator=draft.locator,
                company=draft.company,
                signal_type="funding",
                amount=draft.amount,
                stage=None,
                effective_date=draft.effective_date,
                geography=draft.geography,
                technology_terms=(draft.industry,),
                evidence_passage_id=passage_id,
            )
            for draft, (document_id, passage_id) in zip(
                drafts, lineages, strict=True
            )
        )

    def parse(self, content: str | bytes) -> tuple[EmergingSignal, ...]:
        """Parse a strict licensed single-table extract, not an SEC ZIP."""
        text = _decode(content, max_bytes=self.config.max_member_bytes, connector=self.name)
        try:
            reader = csv.DictReader(io.StringIO(text, newline=""), dialect="excel-tab")
            if tuple(reader.fieldnames or ()) != _DIRECT_FIELDS:
                raise ConnectorError(self.name, retryable=False, diagnostic_code="invalid_schema")
            drafts = []
            for row in reader:
                if None in row or set(row) != set(_DIRECT_FIELDS):
                    raise ValueError("row shape does not match schema")
                accession = _required(row, "ACCESSIONNUMBER")
                drafts.append(self._draft(
                    accession=accession,
                    issuer={"ENTITYNAME": row["ENTITYNAME"] or "", "STATEORCOUNTRY": row["STATEORCOUNTRY"] or ""},
                    offering={"TOTALOFFERINGAMOUNT": row["TOTALOFFERINGAMOUNT"] or "", "INDUSTRYGROUP": row["INDUSTRYGROUP"] or ""},
                    submission={"FILED": row["FILED"] or ""},
                    raw_record={"LICENSED_EXTRACT": dict(row)},
                ))
            return self._persist_drafts(drafts)
        except ConnectorError:
            raise
        except (csv.Error, ValueError, InvalidOperation):
            raise ConnectorError(self.name, retryable=False, diagnostic_code="invalid_payload") from None

    def _read_archive(self, archive_bytes: bytes) -> dict[str, str]:
        if not isinstance(archive_bytes, bytes):
            raise TypeError("archive_bytes must be bytes")
        if len(archive_bytes) > self.config.max_archive_bytes:
            raise ConnectorError(self.name, retryable=False, diagnostic_code="archive_too_large")
        try:
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                members = [item for item in archive.infolist() if not item.is_dir()]
                named: dict[str, zipfile.ZipInfo] = {}
                total_size = total_compressed = 0
                for member in members:
                    path = PurePosixPath(member.filename)
                    table = path.stem.upper()
                    mode = member.external_attr >> 16
                    if (path.is_absolute() or ".." in path.parts or path.suffix.casefold() != ".txt"
                            or table not in _TABLES or table in named or stat.S_ISLNK(mode)):
                        raise ConnectorError(self.name, retryable=False, diagnostic_code="invalid_archive")
                    if member.file_size > self.config.max_member_bytes:
                        raise ConnectorError(self.name, retryable=False, diagnostic_code="archive_member_too_large")
                    named[table] = member
                    total_size += member.file_size
                    total_compressed += member.compress_size
                if set(named) != _TABLES:
                    raise ConnectorError(self.name, retryable=False, diagnostic_code="invalid_archive")
                if total_size > self.config.max_total_uncompressed_bytes:
                    raise ConnectorError(self.name, retryable=False, diagnostic_code="archive_total_too_large")
                if total_size and (total_compressed == 0 or total_size / total_compressed > self.config.max_compression_ratio):
                    raise ConnectorError(self.name, retryable=False, diagnostic_code="archive_ratio_exceeded")
                result = {}
                actual_total = 0
                for table in sorted(named):
                    raw = bytearray()
                    with archive.open(named[table], "r") as stream:
                        while chunk := stream.read(65_536):
                            raw.extend(chunk)
                            actual_total += len(chunk)
                            if len(raw) > self.config.max_member_bytes:
                                raise ConnectorError(self.name, retryable=False, diagnostic_code="archive_member_too_large")
                            if actual_total > self.config.max_total_uncompressed_bytes:
                                raise ConnectorError(self.name, retryable=False, diagnostic_code="archive_total_too_large")
                    result[table] = _decode(bytes(raw), max_bytes=self.config.max_member_bytes, connector=self.name)
                return result
        except ConnectorError:
            raise
        except (zipfile.BadZipFile, OSError, RuntimeError):
            raise ConnectorError(self.name, retryable=False, diagnostic_code="invalid_archive") from None

    def parse_zip(self, archive_bytes: bytes) -> tuple[EmergingSignal, ...]:
        try:
            texts = self._read_archive(archive_bytes)
            tables = {table: _parse_table(texts[table], table=table, required=_REQUIRED_COLUMNS[table]) for table in _TABLES}
            submissions = self._unique_by_accession(tables["FORMDSUBMISSION"])
            issuers = self._issuers_by_accession(tables["ISSUERS"])
            offerings = self._unique_by_accession(tables["OFFERING"])
            recipients = self._rows_by_accession(
                tables["RECIPIENTS"], "RECIPIENT_SEQ_KEY"
            )
            related_people = self._rows_by_accession(
                tables["RELATEDPERSONS"], "RELATEDPERSON_SEQ_KEY"
            )
            signatures = self._rows_by_accession(
                tables["SIGNATURES"], "SIGNATURE_SEQ_KEY"
            )
            drafts = []
            for accession, submission in submissions.items():
                if accession not in issuers or accession not in offerings:
                    raise ValueError("joined Form D row is incomplete")
                for issuer in issuers[accession]:
                    raw_record = {
                        "FORMDSUBMISSION": submission,
                        "ISSUERS": issuer,
                        "OFFERING": offerings[accession],
                        "RECIPIENTS": list(recipients.get(accession, ())),
                        "RELATEDPERSONS": list(related_people.get(accession, ())),
                        "SIGNATURES": list(signatures.get(accession, ())),
                    }
                    drafts.append(self._draft(
                        accession=accession, issuer=issuer, offering=offerings[accession],
                        submission=submission, raw_record=raw_record,
                    ))
            return self._persist_drafts(drafts)
        except ConnectorError:
            raise
        except (KeyError, ValueError, InvalidOperation):
            raise ConnectorError(self.name, retryable=False, diagnostic_code="invalid_payload") from None

    @staticmethod
    def _unique_by_accession(rows: tuple[dict[str, str], ...]) -> dict[str, dict[str, str]]:
        result = {}
        for row in rows:
            accession = _required(row, "ACCESSIONNUMBER")
            if accession in result:
                raise ValueError("duplicate accession row")
            result[accession] = row
        return result

    @staticmethod
    def _issuers_by_accession(
        rows: tuple[dict[str, str], ...]
    ) -> dict[str, tuple[dict[str, str], ...]]:
        grouped: dict[str, list[dict[str, str]]] = {}
        seen = set()
        for row in rows:
            accession = _required(row, "ACCESSIONNUMBER")
            sequence = _required(row, "ISSUER_SEQ_KEY")
            key = (accession, sequence)
            if key in seen:
                raise ValueError("duplicate issuer key")
            seen.add(key)
            grouped.setdefault(accession, []).append(row)
        return {
            accession: tuple(sorted(items, key=lambda item: item["ISSUER_SEQ_KEY"]))
            for accession, items in grouped.items()
        }

    @staticmethod
    def _rows_by_accession(
        rows: tuple[dict[str, str], ...], sequence_field: str
    ) -> dict[str, tuple[dict[str, str], ...]]:
        grouped: dict[str, list[dict[str, str]]] = {}
        seen = set()
        for row in rows:
            accession = _required(row, "ACCESSIONNUMBER")
            sequence = _required(row, sequence_field)
            key = (accession, sequence)
            if key in seen:
                raise ValueError("duplicate accession sequence key")
            seen.add(key)
            grouped.setdefault(accession, []).append(row)
        return {
            accession: tuple(sorted(items, key=lambda item: item[sequence_field]))
            for accession, items in grouped.items()
        }

    def fetch(self, checkpoint: ConnectorCheckpoint) -> SignalConnectorBatch:
        if checkpoint.connector != self.name:
            raise ValueError("checkpoint belongs to another connector")
        if self.archive_loader is None:
            raise ConnectorError(self.name, retryable=False, diagnostic_code="request_context_missing")
        try:
            archive, proposed_cursor = self.archive_loader(checkpoint)
        except ConnectorError:
            raise
        except Exception:
            raise ConnectorError(self.name, retryable=True, diagnostic_code="archive_read_failed") from None
        if proposed_cursor is not None and (not isinstance(proposed_cursor, str) or not proposed_cursor.strip()):
            raise ConnectorError(self.name, retryable=False, diagnostic_code="invalid_checkpoint")
        signals = self.parse_zip(archive)
        if not signals:
            return SignalConnectorBatch(self.name, (), checkpoint)
        return SignalConnectorBatch(
            self.name, signals,
            ConnectorCheckpoint(self.name, cursor=proposed_cursor.strip() if proposed_cursor is not None else checkpoint.cursor),
        )
