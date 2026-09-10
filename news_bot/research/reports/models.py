"""Strict presentation-only contracts for institutional research reports."""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..models import InferenceMode


ReportType = Literal[
    "event_update",
    "portfolio_brief",
    "industry_landscape",
    "emerging_monitor",
]
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("must be text")
    value = unicodedata.normalize("NFC", value).strip()
    if not value or any(
        unicodedata.category(char) == "Cc" and char not in {"\n", "\t"}
        for char in value
    ):
        raise ValueError("must be nonblank display text without control characters")
    return value


def _identifier(value: str) -> str:
    value = _text(value)
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError("must be a safe identifier")
    return value


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must be timezone-aware")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError("must be UTC")
    return value


class DisplayModel(BaseModel):
    """Base contract that prevents coercion, mutation, and surprise fields."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )


class Citation(DisplayModel):
    evidence_id: str
    source: str
    url: str
    source_date: date
    data_date: date
    content_hash: str

    _evidence_id = field_validator("evidence_id")(_identifier)
    _source = field_validator("source")(_text)

    @field_validator("url")
    @classmethod
    def _public_http_url(cls, value: str) -> str:
        value = _text(value)
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
            raise ValueError("must be an absolute public HTTP(S) URL")
        return value

    @field_validator("content_hash")
    @classmethod
    def _content_hash(cls, value: str) -> str:
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError("must be a SHA-256 hex digest")
        return value.lower()


class ReportSection(DisplayModel):
    title: str
    body: str
    evidence_ids: tuple[str, ...] = ()

    _title = field_validator("title")(_text)
    _body = field_validator("body")(_text)

    @field_validator("evidence_ids")
    @classmethod
    def _evidence_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_identifier(value) for value in values)
        if len(set(normalized)) != len(normalized):
            raise ValueError("evidence IDs must be unique")
        return normalized


class ReportMetadata(DisplayModel):
    report_id: str
    report_type: ReportType
    title: str
    as_of: datetime
    inference_mode: InferenceMode
    provider: str
    model: str
    freshness: str
    citations: tuple[Citation, ...] = Field(min_length=1)
    methodology: str
    omissions: tuple[str, ...] = ()
    disclosure: str

    _report_id = field_validator("report_id")(_identifier)
    _title = field_validator("title")(_text)
    _as_of = field_validator("as_of")(_utc)
    _provider = field_validator("provider")(_identifier)
    _model = field_validator("model")(_identifier)
    _freshness = field_validator("freshness")(_text)
    _methodology = field_validator("methodology")(_text)
    _disclosure = field_validator("disclosure")(_text)

    @field_validator("omissions")
    @classmethod
    def _omissions(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_text(value) for value in values)

    @field_validator("citations")
    @classmethod
    def _citations(cls, values: tuple[Citation, ...]) -> tuple[Citation, ...]:
        identifiers = tuple(value.evidence_id for value in values)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("citation evidence IDs must be unique")
        return tuple(sorted(values, key=lambda item: item.evidence_id))


class DisplayExhibit(DisplayModel):
    """Common, presentation-safe base for calculated exhibits."""

    title: str
    source_notes: tuple[str, ...]

    _title = field_validator("title")(_text)


class _Report(DisplayModel):
    metadata: ReportMetadata

    @classmethod
    def _require_type(cls, metadata: ReportMetadata, expected: str) -> None:
        if metadata.report_type != expected:
            raise ValueError(f"metadata.report_type must be {expected}")

    @model_validator(mode="after")
    def _require_declared_evidence(self) -> "_Report":
        declared = {citation.evidence_id for citation in self.metadata.citations}
        referenced: set[str] = set()
        for value in self.__dict__.values():
            if isinstance(value, ReportSection):
                referenced.update(value.evidence_ids)
            elif isinstance(value, tuple):
                for item in value:
                    if isinstance(item, ReportSection):
                        referenced.update(item.evidence_ids)
        missing = referenced - declared
        if missing:
            raise ValueError(
                "section evidence is absent from citations: " + ", ".join(sorted(missing))
            )
        return self


class EventUpdate(_Report):
    thesis: ReportSection
    event_decomposition: tuple[ReportSection, ...] = Field(min_length=1)
    causal_decomposition: tuple[ReportSection, ...] = Field(min_length=1)
    read_through: tuple[ReportSection, ...] = Field(min_length=1)
    exhibits: tuple[DisplayExhibit, ...] = ()
    thesis_changes: tuple[ReportSection, ...] = Field(min_length=1)
    unchanged_assumptions: tuple[ReportSection, ...] = Field(min_length=1)
    questions: tuple[ReportSection, ...] = Field(min_length=1)
    signposts: tuple[ReportSection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _type(self) -> "EventUpdate":
        self._require_type(self.metadata, "event_update")
        return self


class PortfolioBrief(_Report):
    exposure_summary: tuple[ReportSection, ...] = Field(min_length=1)
    relevant_news: tuple[ReportSection, ...] = Field(min_length=1)
    value_chain_developments: tuple[ReportSection, ...] = Field(min_length=1)
    research_views: tuple[ReportSection, ...] = Field(min_length=1)
    change_history: tuple[ReportSection, ...] = Field(min_length=1)
    concentration_and_correlation: tuple[ReportSection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _type(self) -> "PortfolioBrief":
        self._require_type(self.metadata, "portfolio_brief")
        return self


class IndustryLandscape(_Report):
    value_chain: tuple[ReportSection, ...] = Field(min_length=1)
    profit_pools: tuple[ReportSection, ...] = Field(min_length=1)
    bottlenecks: tuple[ReportSection, ...] = Field(min_length=1)
    emerging_technology_and_companies: tuple[ReportSection, ...] = Field(min_length=1)
    long_term_scenarios: tuple[ReportSection, ...] = Field(min_length=1)
    signposts: tuple[ReportSection, ...] = Field(min_length=1)
    invalidation: tuple[ReportSection, ...] = Field(min_length=1)
    public_beneficiaries: tuple[ReportSection, ...] = Field(min_length=1)
    threats: tuple[ReportSection, ...] = Field(min_length=1)
    portfolio_relevance: tuple[ReportSection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _type(self) -> "IndustryLandscape":
        self._require_type(self.metadata, "industry_landscape")
        return self


class EmergingCompanyMonitor(_Report):
    company_and_technology_map: tuple[ReportSection, ...] = Field(min_length=1)
    adoption_signals: tuple[ReportSection, ...] = Field(min_length=1)
    confidence_and_limits: tuple[ReportSection, ...] = Field(min_length=1)
    public_market_translation: tuple[ReportSection, ...] = Field(min_length=1)
    private_company_rating: Literal[None] = None

    @model_validator(mode="after")
    def _type(self) -> "EmergingCompanyMonitor":
        self._require_type(self.metadata, "emerging_monitor")
        return self


class RenderedReportArtifact(DisplayModel):
    report_id: str
    report_type: ReportType
    as_of: datetime
    html: str
    html_path: Path
    pdf_path: Path | None
    pdf_error: str | None = None


# Explicit long-form aliases make the view-model purpose discoverable.
EventUpdateReport = EventUpdate
PortfolioBriefReport = PortfolioBrief
IndustryLandscapeReport = IndustryLandscape
EmergingCompanyMonitorReport = EmergingCompanyMonitor
ReportArtifact = RenderedReportArtifact
