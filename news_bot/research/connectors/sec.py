"""Safe, paced SEC EDGAR submissions and Company Facts connector."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlsplit

import requests

from ..evidence import DocumentInput
from .base import ConnectorCheckpoint, ConnectorError, NormalizedResearchDocument


_SUPPORTED_FORMS = frozenset(
    {"10-K", "10-Q", "8-K", "20-F", "6-K", "S-1", "S-3", "F-1", "F-3"}
)
_CIK = re.compile(r"[0-9]{1,10}")
_EMAIL = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
_PRIMARY_DOCUMENT = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class SECConfig:
    """Transport and policy settings for SEC public-data requests."""

    user_agent: str
    min_interval_seconds: float = 0.11
    timeout_seconds: float = 15.0
    max_response_bytes: int = 5_000_000
    allowed_forms: tuple[str, ...] = (
        "10-K",
        "10-Q",
        "8-K",
        "20-F",
        "6-K",
        "S-1",
        "S-3",
        "F-1",
        "F-3",
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.user_agent, str)
            or not self.user_agent.strip()
            or _EMAIL.search(self.user_agent) is None
            or "\n" in self.user_agent
            or "\r" in self.user_agent
        ):
            raise ValueError("SEC user_agent must be identifying and include an email")
        if (
            not isinstance(self.min_interval_seconds, (int, float))
            or self.min_interval_seconds < 0.101
        ):
            raise ValueError("SEC pacing must remain below 10 requests per second")
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("SEC timeout_seconds must be positive")
        if (
            not isinstance(self.max_response_bytes, int)
            or self.max_response_bytes <= 0
        ):
            raise ValueError("SEC max_response_bytes must be positive")
        if (
            not isinstance(self.allowed_forms, tuple)
            or not self.allowed_forms
            or not set(self.allowed_forms) <= _SUPPORTED_FORMS
        ):
            raise ValueError("SEC allowed_forms contains an unsupported form")


@dataclass(frozen=True)
class SECResult:
    """One SEC JSON response with parsed documents and cache validators."""

    payload: Mapping[str, Any] | None
    documents: tuple[NormalizedResearchDocument, ...]
    checkpoint: ConnectorCheckpoint
    not_modified: bool = False


def _normalize_cik(value: str) -> str:
    if not isinstance(value, str) or _CIK.fullmatch(value) is None:
        raise ValueError("CIK must contain one to ten digits")
    return value.zfill(10)


def _utc_midnight(value: str, field_name: str) -> datetime:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ConnectorError(
            "sec", retryable=False, diagnostic_code=f"invalid_{field_name}"
        ) from None
    return datetime.combine(parsed, datetime.min.time(), tzinfo=timezone.utc)


class SECConnector:
    """Read-only connector for SEC JSON APIs on ``data.sec.gov``."""

    name = "sec"
    base_url = "https://data.sec.gov"

    def __init__(
        self,
        config: SECConfig,
        *,
        session=None,
        clock=time.monotonic,
        sleeper=time.sleep,
    ) -> None:
        if not isinstance(config, SECConfig):
            raise TypeError("config must be SECConfig")
        self.config = config
        self.session = session or requests.Session()
        self.session.trust_env = False
        self._clock = clock
        self._sleeper = sleeper
        self._pacing_lock = threading.Lock()
        self._last_request_at: float | None = None

    @staticmethod
    def submissions_url(cik: str) -> str:
        return f"https://data.sec.gov/submissions/CIK{_normalize_cik(cik)}.json"

    @staticmethod
    def companyfacts_url(cik: str) -> str:
        return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{_normalize_cik(cik)}.json"

    def _pace(self) -> None:
        with self._pacing_lock:
            current = float(self._clock())
            if self._last_request_at is not None:
                remaining = (
                    self.config.min_interval_seconds
                    - (current - self._last_request_at)
                )
                if remaining > 0:
                    self._sleeper(remaining)
                    current += remaining
            self._last_request_at = current

    def _headers(self, checkpoint: ConnectorCheckpoint | None) -> dict[str, str]:
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }
        if checkpoint is not None:
            if checkpoint.connector != self.name:
                raise ValueError("checkpoint belongs to another connector")
            if checkpoint.etag:
                headers["If-None-Match"] = checkpoint.etag
            if checkpoint.last_modified:
                headers["If-Modified-Since"] = checkpoint.last_modified
        return headers

    def _request_json(
        self, url: str, checkpoint: ConnectorCheckpoint | None
    ) -> tuple[Mapping[str, Any] | None, ConnectorCheckpoint, bool]:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "data.sec.gov"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
        ):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="endpoint_rejected"
            )
        self._pace()
        try:
            response = self.session.get(
                url,
                headers=self._headers(checkpoint),
                timeout=self.config.timeout_seconds,
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException:
            raise ConnectorError(
                self.name, retryable=True, diagnostic_code="transport_error"
            ) from None
        except Exception:
            raise ConnectorError(
                self.name, retryable=True, diagnostic_code="transport_error"
            ) from None

        final = urlsplit(getattr(response, "url", ""))
        if (
            getattr(response, "history", ())
            or final.scheme != "https"
            or final.hostname != "data.sec.gov"
            or final.port not in (None, 443)
        ):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="redirect_rejected"
            )
        status = int(response.status_code)
        next_checkpoint = ConnectorCheckpoint(
            self.name,
            cursor=checkpoint.cursor if checkpoint else None,
            etag=response.headers.get("ETag")
            or (checkpoint.etag if checkpoint else None),
            last_modified=response.headers.get("Last-Modified")
            or (checkpoint.last_modified if checkpoint else None),
        )
        if status == 304:
            return None, next_checkpoint, True
        if not 200 <= status < 300:
            raise ConnectorError(
                self.name,
                status=status,
                retryable=status in {408, 425, 429} or 500 <= status <= 599,
                diagnostic_code=f"http_{status}",
            )
        raw = bytearray()
        try:
            for chunk in response.iter_content(chunk_size=65_536):
                raw.extend(chunk)
                if len(raw) > self.config.max_response_bytes:
                    raise ConnectorError(
                        self.name,
                        retryable=False,
                        diagnostic_code="response_too_large",
                    )
            payload = json.loads(bytes(raw).decode("utf-8", errors="strict"))
        except ConnectorError:
            raise
        except (UnicodeError, json.JSONDecodeError):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_json"
            ) from None
        except Exception:
            raise ConnectorError(
                self.name, retryable=True, diagnostic_code="response_read_failed"
            ) from None
        if not isinstance(payload, dict):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_payload"
            )
        return payload, next_checkpoint, False

    def fetch_submissions(
        self, cik: str, *, checkpoint: ConnectorCheckpoint | None = None
    ) -> SECResult:
        payload, next_checkpoint, not_modified = self._request_json(
            self.submissions_url(cik), checkpoint
        )
        retrieved_at = datetime.now(timezone.utc)
        documents = (
            self.parse_submissions(payload, retrieved_at=retrieved_at)
            if payload is not None
            else ()
        )
        return SECResult(payload, documents, next_checkpoint, not_modified)

    def fetch_companyfacts(
        self, cik: str, *, checkpoint: ConnectorCheckpoint | None = None
    ) -> SECResult:
        normalized_cik = _normalize_cik(cik)
        payload, next_checkpoint, not_modified = self._request_json(
            self.companyfacts_url(cik), checkpoint
        )
        documents = (
            (self.parse_companyfacts(payload, normalized_cik, datetime.now(timezone.utc)),)
            if payload is not None
            else ()
        )
        return SECResult(payload, documents, next_checkpoint, not_modified)

    def parse_submissions(
        self, payload: Mapping[str, Any], *, retrieved_at: datetime
    ) -> tuple[NormalizedResearchDocument, ...]:
        try:
            cik = _normalize_cik(str(payload["cik"]))
            publisher = str(payload["name"]).strip()
            recent = payload["filings"]["recent"]
            accessions = recent["accessionNumber"]
            filing_dates = recent["filingDate"]
            report_dates = recent["reportDate"]
            forms = recent["form"]
            primary_documents = recent["primaryDocument"]
            lengths = {
                len(accessions),
                len(filing_dates),
                len(report_dates),
                len(forms),
                len(primary_documents),
            }
            if len(lengths) != 1 or not publisher:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_submissions"
            ) from None

        documents: list[NormalizedResearchDocument] = []
        for accession, filing_date, report_date, form, primary in zip(
            accessions,
            filing_dates,
            report_dates,
            forms,
            primary_documents,
            strict=True,
        ):
            if form not in self.config.allowed_forms:
                continue
            if (
                not isinstance(accession, str)
                or re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession) is None
                or not isinstance(primary, str)
                or _PRIMARY_DOCUMENT.fullmatch(primary) is None
            ):
                raise ConnectorError(
                    self.name, retryable=False, diagnostic_code="invalid_filing_metadata"
                )
            accession_path = accession.replace("-", "")
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{accession_path}/{primary}"
            )
            content = json.dumps(
                {
                    "accession_number": accession,
                    "cik": cik,
                    "filing_date": filing_date,
                    "form": form,
                    "primary_document": primary,
                    "report_date": report_date,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            documents.append(
                NormalizedResearchDocument(
                    evidence=DocumentInput(
                        source_type="sec_filing_metadata",
                        url=url,
                        publisher=publisher,
                        published_at=_utc_midnight(filing_date, "filing_date"),
                        retrieved_at=retrieved_at,
                        content=content,
                    ),
                    tags=("SEC", form),
                )
            )
        return tuple(documents)

    def parse_companyfacts(
        self,
        payload: Mapping[str, Any],
        normalized_cik: str,
        retrieved_at: datetime,
    ) -> NormalizedResearchDocument:
        try:
            publisher = str(payload["entityName"]).strip()
            if not publisher or not isinstance(payload["facts"], dict):
                raise ValueError
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        except (KeyError, TypeError, ValueError):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_companyfacts"
            ) from None
        return NormalizedResearchDocument(
            evidence=DocumentInput(
                source_type="sec_companyfacts",
                url=(
                    "https://data.sec.gov/api/xbrl/companyfacts/"
                    f"CIK{normalized_cik}.json"
                ),
                publisher=publisher,
                published_at=retrieved_at,
                retrieved_at=retrieved_at,
                content=encoded,
            ),
            tags=("SEC", "companyfacts"),
        )
