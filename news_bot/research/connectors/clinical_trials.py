"""ClinicalTrials.gov v2 emerging-company trial signals."""

from __future__ import annotations

import json
import time
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone

import requests

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


_HOST = "clinicaltrials.gov"
_ROOT = "/api/v2"


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return " ".join(unicodedata.normalize("NFC", value).split())


def _json_mapping(value: str | bytes | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, (str, bytes)):
        raise TypeError("payload must be JSON text, bytes, or a mapping")
    try:
        decoded = json.loads(value)
    except (UnicodeError, json.JSONDecodeError, TypeError):
        raise ConnectorError(
            "clinical_trials", retryable=False, diagnostic_code="invalid_json"
        ) from None
    if not isinstance(decoded, dict):
        raise ConnectorError(
            "clinical_trials", retryable=False, diagnostic_code="invalid_payload"
        )
    return decoded


def _study_marker(study: object) -> tuple[date, str]:
    if not isinstance(study, Mapping):
        raise ValueError("study must be an object")
    protocol = study.get("protocolSection")
    if not isinstance(protocol, Mapping):
        raise ValueError("protocolSection must be an object")
    identity = protocol.get("identificationModule")
    status = protocol.get("statusModule")
    if not isinstance(identity, Mapping) or not isinstance(status, Mapping):
        raise ValueError("protocol modules must be objects")
    return (
        date.fromisoformat(
            _text(status.get("studyFirstSubmitDate"), "studyFirstSubmitDate")
        ),
        _text(identity.get("nctId"), "nctId"),
    )


@dataclass(frozen=True)
class ClinicalTrialsConfig:
    base_url: str = "https://clinicaltrials.gov/api/v2"
    timeout_seconds: float = 15.0
    max_response_bytes: int = 5_000_000
    min_interval_seconds: float = 0.2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_url",
            validated_api_base_url(self.base_url, host=_HOST, path_prefix=_ROOT),
        )
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not isinstance(self.max_response_bytes, int) or self.max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        if (
            not isinstance(self.min_interval_seconds, (int, float))
            or self.min_interval_seconds <= 0
        ):
            raise ValueError("min_interval_seconds must be positive")


class ClinicalTrialsConnector:
    name = "clinical_trials"

    def __init__(
        self,
        config: ClinicalTrialsConfig | None = None,
        *,
        session=None,
        query: str | None = None,
        ingestor: EvidenceIngestor | None = None,
        retrieved_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or ClinicalTrialsConfig()
        self.query = _text(query, "query") if query is not None else None
        self.ingestor = ingestor
        self._retrieved_clock = retrieved_clock
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
        data = _json_mapping(payload)
        try:
            studies = data["studies"]
            if not isinstance(studies, list):
                raise ValueError("studies must be a list")
            drafts = []
            for study in studies:
                if not isinstance(study, Mapping):
                    raise ValueError("study must be an object")
                protocol = study["protocolSection"]
                if not isinstance(protocol, Mapping):
                    raise ValueError("protocolSection must be an object")
                identity = protocol["identificationModule"]
                status = protocol["statusModule"]
                contacts = protocol.get("contactsLocationsModule", {})
                conditions = protocol.get("conditionsModule", {})
                if not all(
                    isinstance(value, Mapping)
                    for value in (identity, status, contacts, conditions)
                ):
                    raise ValueError("protocol modules must be objects")
                organization = identity["organization"]
                if not isinstance(organization, Mapping):
                    raise ValueError("organization must be an object")
                locations = contacts.get("locations", [])
                if not isinstance(locations, list):
                    raise ValueError("locations must be a list")
                geography = None
                if locations:
                    location = locations[0]
                    if not isinstance(location, Mapping):
                        raise ValueError("location must be an object")
                    pieces = [
                        _text(location[key], key)
                        for key in ("state", "country")
                        if key in location and location[key] is not None
                    ]
                    geography = ", ".join(pieces) or None
                terms = conditions.get("conditions", [])
                if not isinstance(terms, list) or any(
                    not isinstance(term, str) or not term.strip() for term in terms
                ):
                    raise ValueError("conditions must be a text list")
                nct_id = _text(identity.get("nctId"), "nctId")
                locator = f"clinicaltrials:{nct_id}"
                effective_date = date.fromisoformat(
                    _text(
                        status.get("studyFirstSubmitDate"),
                        "studyFirstSubmitDate",
                    )
                )
                company = _text(
                    organization.get("fullName"), "organization.fullName"
                )
                stage = _text(status.get("overallStatus"), "overallStatus")
                source = signal_evidence_input(
                    connector=self.name,
                    source_locator=locator,
                    source_url=f"{self.config.base_url}/studies/{nct_id}",
                    publisher="ClinicalTrials.gov",
                    effective_date=effective_date,
                    raw_record=study,
                    retrieved_at=self._retrieved_clock(),
                )
                drafts.append(
                    (
                        locator,
                        effective_date,
                        company,
                        stage,
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
                        signal_type="clinical_trial",
                        amount=None,
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
        params: dict[str, object] = {"query.term": self.query, "format": "json"}
        if cursor.continuation is not None:
            params["pageToken"] = cursor.continuation
        try:
            payload = self._transport.request(
                "GET",
                f"{self.config.base_url}/studies",
                headers={"Accept": "application/json"},
                params=params,
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
            studies = payload.get("studies")
            if not isinstance(studies, list):
                raise ValueError("studies must be a list")
            if not studies:
                return SignalConnectorBatch(self.name, (), checkpoint)
            markers = tuple(_study_marker(study) for study in studies)
            filtered = tuple(
                study
                for study, marker in zip(studies, markers, strict=True)
                if cursor.is_new(marker)
            )
            token = payload.get("nextPageToken")
            continuation = (
                _text(token, "nextPageToken") if token is not None else None
            )
        except (TypeError, ValueError):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_payload"
            ) from None
        normalized_payload = dict(payload)
        normalized_payload["studies"] = list(filtered)
        signals = self.parse(normalized_payload)
        next_cursor = cursor.advance(markers, continuation=continuation)
        return SignalConnectorBatch(
            self.name,
            signals,
            ConnectorCheckpoint(self.name, cursor=next_cursor.encode()),
        )
