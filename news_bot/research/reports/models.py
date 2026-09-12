"""Strict presentation-only contracts for institutional research reports."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..models import InferenceMode, RecommendationRating
from ..quality import CalculatedExhibit


ReportType = Literal[
    "event_update",
    "portfolio_brief",
    "industry_landscape",
    "emerging_monitor",
]
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PROVIDER_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|token|auth(?:orization)?|bearer|password|passwd|secret|client[_-]?secret)\b\s*[:=]\s*\S+"
)
_RAW_SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{8,}|AKIA[A-Z0-9]{12,})\b")
_IBKR_ACCOUNT = re.compile(r"(?i)\b(?:DU|U|F|FA|I|M)\d{5,}\b")
_ACCOUNT_ASSIGNMENT = re.compile(
    r"(?i)\b(?:account|acct)[_\s-]?(?:id|identifier|number|no\.?)\b"
    r"\s*(?:is|of|[:=])\s*"
    r"(?=(?:[^\d]*\d){5,})[A-Za-z0-9](?:[A-Za-z0-9\s-]*[A-Za-z0-9])?"
)
_ACCOUNT_REFERENCE = re.compile(
    r"(?i)\b(?:brokerage\s+)?(?:account|acct)\b\s*(?:is|of|[:=#-])?\s*"
    r"(?=(?:[^\d]*\d){5,})[A-Za-z0-9](?:[A-Za-z0-9\s-]*[A-Za-z0-9])?"
)
_MONEY_NUMBER = r"(?:\d(?:[\d,]*\d)?(?:\.\d+)?|\.\d+)"
_EXACT_PORTFOLIO_VALUE = re.compile(
    r"(?ix)\b(?:net\s+asset\s+value|nav|cash(?:\s+balance)?|position(?:\s+value)?|market\s+value)\b"
    r"\s*(?:is|was|were|of|at|totals?|totaled|stood\s+at|[:=])?\s*[+−-]?\s*(?:\(\s*)?"
    r"[+−-]?\s*(?:USD|EUR|GBP|JPY|MYR|[$€£])?\s*[+−-]?\s*"
    + _MONEY_NUMBER
    + r"(?![\d,]|\.\d)\s*\)?(?!\s*(?:[%x×]|[–—-]\s*\d))"
)
_MONEY_AMOUNT = re.compile(
    r"(?ix)(?:\(\s*)?(?:"
    r"[+−-]?\s*(?:USD|EUR|GBP|JPY|MYR|[$€£])\s*[+−-]?\s*"
    + _MONEY_NUMBER
    + r"|[+−-]?\s*"
    + _MONEY_NUMBER
    + r"\s*(?:USD|EUR|GBP|JPY|MYR|[$€£])"
    r")(?![\d,]|\.\d)\s*\)?(?!\s*(?:[%x×]|[–—-]\s*\d))"
)
_PUBLICATION_NUMBER = re.compile(
    r"(?ix)(?<![\w.])(?P<paren>\()?\s*"
    r"(?:(?:USD|EUR|GBP|JPY|MYR|[$€£])\s*)?"
    r"(?P<sign>[+−-])?\s*(?P<number>"
    + _MONEY_NUMBER
    + r")\s*\)?(?![\d,]|\.\d)"
)
_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
_MAX_PERCENT_DECODE_ROUNDS = 4
class ReportPublicationError(RuntimeError):
    """Sanitized hard failure at the report publication boundary."""


def _contains_private_material_once(value: str) -> bool:
    return bool(
        _SECRET_ASSIGNMENT.search(value)
        or _RAW_SECRET.search(value)
        or _IBKR_ACCOUNT.search(value)
        or _ACCOUNT_ASSIGNMENT.search(value)
        or _ACCOUNT_REFERENCE.search(value)
        or _EXACT_PORTFOLIO_VALUE.search(value)
        or _MONEY_AMOUNT.search(value)
    )


def _privacy_analysis_copy(value: str) -> str:
    if any(unicodedata.category(char) == "Cf" for char in value):
        raise ValueError("display text contains invisible format characters")
    normalized = unicodedata.normalize("NFKC", value)
    if any(unicodedata.category(char) == "Cf" for char in normalized):
        raise ValueError("display text contains invisible format characters")
    return normalized


def _privacy_decoded_variants(value: str) -> tuple[str, ...]:
    current = _privacy_analysis_copy(value)
    variants = [current]
    for _ in range(_MAX_PERCENT_DECODE_ROUNDS):
        decoded = _privacy_analysis_copy(unquote(current))
        if decoded == current:
            return tuple(variants)
        variants.append(decoded)
        current = decoded
    if _PERCENT_ESCAPE.search(current):
        raise ValueError("display text is excessively percent encoded")
    return tuple(variants)


def _contains_private_material(value: str) -> bool:
    return any(
        _contains_private_material_once(candidate)
        for candidate in _privacy_decoded_variants(value)
    )


def _display_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("must be text")
    value = unicodedata.normalize("NFC", value).strip()
    if not value or any(
        unicodedata.category(char) == "Cc" and char not in {"\n", "\t"}
        for char in value
    ):
        raise ValueError("must be nonblank display text without control characters")
    if _contains_private_material(value):
        raise ValueError("display text contains private or sensitive material")
    return value


def _identity_label(value: str) -> str:
    value = _display_text(value)
    if any(_MONEY_AMOUNT.search(item) for item in _privacy_decoded_variants(value)):
        raise ValueError("identity label contains private money material")
    return value


# Kept as an internal compatibility name for exhibit validators.
_text = _display_text


def _identifier(value: str) -> str:
    value = _display_text(value)
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError("must be a safe identifier")
    return value


def _provider_model_name(value: str) -> str:
    value = _display_text(value)
    if not _PROVIDER_MODEL.fullmatch(value) or "//" in value or ".." in value:
        raise ValueError("must be a safe provider or model name")
    return value


def _evidence_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(_identifier(value) for value in values)
    if not normalized:
        raise ValueError("evidence IDs are required for material content")
    if len(set(normalized)) != len(normalized):
        raise ValueError("evidence IDs must be unique")
    return normalized


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


def _context_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("privacy context values must be text")
    normalized = _privacy_analysis_copy(value).strip()
    if not normalized:
        raise ValueError("privacy context values must be nonblank")
    return value


def _account_key(value: str) -> str:
    return "".join(
        character
        for character in _privacy_analysis_copy(value).casefold()
        if character.isalnum()
    )


def _publication_number_values(value: str) -> tuple[Decimal, ...]:
    normalized = _privacy_analysis_copy(value).replace("−", "-")
    matches: list[Decimal] = []
    for match in _PUBLICATION_NUMBER.finditer(normalized):
        before = normalized[: match.start()].rstrip()
        after = normalized[match.end() :].lstrip()
        if after.startswith(("%", "x", "×")):
            continue
        if after.startswith(("-", "–", "—")) or before.endswith(("-", "–", "—")):
            continue
        try:
            numeric = Decimal(match.group("number").replace(",", ""))
        except Exception:
            continue
        if match.group("sign") == "-" or match.group("paren"):
            numeric = -numeric
        matches.append(numeric)
    return tuple(matches)


class PublicationPrivacyContext(DisplayModel):
    """Caller-supplied local values that must never cross publication."""

    sensitive_literals: tuple[str, ...]
    account_identifiers: tuple[str, ...]
    portfolio_values: tuple[Decimal, ...]

    _literals = field_validator("sensitive_literals")(
        lambda values: tuple(_context_text(value) for value in values)
    )
    _accounts = field_validator("account_identifiers")(
        lambda values: tuple(_context_text(value) for value in values)
    )

    @field_validator("account_identifiers")
    @classmethod
    def _valid_accounts(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            key = _account_key(value)
            if len(key) < 5 or not any(character.isdigit() for character in key):
                raise ValueError("privacy account identifiers must be specific")
        return values

    @field_validator("portfolio_values")
    @classmethod
    def _finite_values(cls, values: tuple[Decimal, ...]) -> tuple[Decimal, ...]:
        if any(not value.is_finite() for value in values):
            raise ValueError("privacy portfolio values must be finite")
        return values

    def assert_safe(self, report: BaseModel) -> None:
        literal_keys = tuple(
            variant.casefold()
            for literal in self.sensitive_literals
            for variant in _privacy_decoded_variants(literal)
        )
        account_keys = tuple(_account_key(value) for value in self.account_identifiers)
        portfolio_values = tuple(abs(value) for value in self.portfolio_values)

        def inspect(value: object) -> None:
            if isinstance(value, BaseModel):
                for field_value in value.__dict__.values():
                    inspect(field_value)
                return
            if isinstance(value, Mapping):
                for key, item in value.items():
                    inspect(key)
                    inspect(item)
                return
            if isinstance(value, (tuple, list, set, frozenset)):
                for item in value:
                    inspect(item)
                return
            if isinstance(value, str):
                try:
                    variants = _privacy_decoded_variants(value)
                except ValueError:
                    raise ReportPublicationError(
                        "report publication blocked by privacy policy"
                    ) from None
                for variant in variants:
                    folded = variant.casefold()
                    if any(literal in folded for literal in literal_keys):
                        raise ReportPublicationError(
                            "report publication blocked by privacy policy"
                        )
                    account_text = _account_key(variant)
                    if any(account in account_text for account in account_keys):
                        raise ReportPublicationError(
                            "report publication blocked by privacy policy"
                        )
                    if any(
                        abs(number) in portfolio_values
                        for number in _publication_number_values(variant)
                    ):
                        raise ReportPublicationError(
                            "report publication blocked by privacy policy"
                        )
                return
            if isinstance(value, Decimal):
                if abs(value) in portfolio_values:
                    raise ReportPublicationError(
                        "report publication blocked by privacy policy"
                    )
                return
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    numeric = abs(Decimal(str(value)))
                except Exception:
                    return
                if numeric in portfolio_values:
                    raise ReportPublicationError(
                        "report publication blocked by privacy policy"
                    )

        inspect(report)


class Citation(DisplayModel):
    evidence_id: str
    source: str
    url: str
    source_date: date
    data_date: date
    content_hash: str

    _evidence_id = field_validator("evidence_id")(_identifier)
    _source = field_validator("source")(_display_text)

    @field_validator("url")
    @classmethod
    def _public_http_url(cls, value: str) -> str:
        value = _display_text(value)
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("must be an absolute public HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("citation URL must not contain credentials")
        if "?" in value or "#" in value:
            raise ValueError("citation URL must be canonical without query or fragment")
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
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    _title = field_validator("title")(_display_text)
    _body = field_validator("body")(_display_text)
    _evidence = field_validator("evidence_ids")(_evidence_ids)


class NonClaimSection(DisplayModel):
    """Administrative prose explicitly distinguished from factual claim sections."""

    title: str
    body: str

    _title = field_validator("title")(_display_text)
    _body = field_validator("body")(_display_text)


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
    _title = field_validator("title")(_display_text)
    _as_of = field_validator("as_of")(_utc)
    _provider = field_validator("provider")(_provider_model_name)
    _model = field_validator("model")(_provider_model_name)
    _freshness = field_validator("freshness")(_display_text)
    _methodology = field_validator("methodology")(_display_text)
    _disclosure = field_validator("disclosure")(_display_text)

    @field_validator("omissions")
    @classmethod
    def _omissions(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_display_text(value) for value in values)

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
    source_notes: tuple[str, ...] = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    calculation: CalculatedExhibit
    verified_reconciliation: Literal[True]

    _title = field_validator("title")(_display_text)
    _sources = field_validator("source_notes")(
        lambda values: tuple(_display_text(value) for value in values)
    )
    _evidence = field_validator("evidence_ids")(_evidence_ids)

    @model_validator(mode="after")
    def _verified(self) -> "DisplayExhibit":
        if not self.calculation.reconciles:
            raise ValueError("calculated exhibit does not reconcile")
        return self


class RoundedExposure(DisplayModel):
    symbol: str
    label: str
    weight_band: Literal["<5%", "5–10%", "10–20%", "20%+"]
    rounded_weight_percent: Decimal = Field(ge=Decimal("0"), le=Decimal("100"))
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    _symbol = field_validator("symbol")(_identifier)
    _label = field_validator("label")(_identity_label)
    _evidence = field_validator("evidence_ids")(_evidence_ids)

    @field_validator("rounded_weight_percent")
    @classmethod
    def _whole_percent(cls, value: Decimal) -> Decimal:
        if not value.is_finite() or value != value.to_integral_value():
            raise ValueError("portfolio exposure must be rounded to a whole percent")
        return value

    @model_validator(mode="after")
    def _matching_band(self) -> "RoundedExposure":
        percent = self.rounded_weight_percent
        expected = (
            "<5%" if percent < 5 else
            "5–10%" if percent < 10 else
            "10–20%" if percent < 20 else "20%+"
        )
        if self.weight_band != expected:
            raise ValueError("rounded exposure does not match its weight band")
        return self

    @property
    def display_weight(self) -> str:
        return f"{self.rounded_weight_percent:.0f}% ({self.weight_band})"


class PortfolioNewsItem(DisplayModel):
    headline: str
    thesis_impact: str
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    _headline = field_validator("headline")(_display_text)
    _impact = field_validator("thesis_impact")(_display_text)
    _evidence = field_validator("evidence_ids")(_evidence_ids)


class ResearchView(DisplayModel):
    symbol: str
    thesis: str
    rating: RecommendationRating
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    _symbol = field_validator("symbol")(_identifier)
    _thesis = field_validator("thesis")(_display_text)
    _evidence = field_validator("evidence_ids")(_evidence_ids)


class ResearchViewChange(DisplayModel):
    """Dated rating history, including explicit no-change entries."""

    symbol: str
    previous_rating: RecommendationRating
    new_rating: RecommendationRating
    rationale: str
    changed_on: date
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    _symbol = field_validator("symbol")(_identifier)
    _rationale = field_validator("rationale")(_display_text)
    _evidence = field_validator("evidence_ids")(_evidence_ids)

class ConcentrationCorrelation(DisplayModel):
    summary: str
    risk_level: Literal["low", "moderate", "high"]
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    _summary = field_validator("summary")(_display_text)
    _evidence = field_validator("evidence_ids")(_evidence_ids)


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

        def collect(value: object) -> None:
            if isinstance(value, DisplayModel):
                evidence = getattr(value, "evidence_ids", ())
                referenced.update(evidence)
            elif isinstance(value, tuple):
                for item in value:
                    collect(item)

        for value in self.__dict__.values():
            if value is not self.metadata:
                collect(value)
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
    exhibits: tuple["ExposureExhibit | ValuationExhibit | ScenarioMatrix", ...] = ()
    thesis_changes: tuple[ReportSection, ...] = Field(min_length=1)
    unchanged_assumptions: tuple[ReportSection, ...] = Field(min_length=1)
    questions: tuple[ReportSection, ...] = Field(min_length=1)
    signposts: tuple[ReportSection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _type(self) -> "EventUpdate":
        self._require_type(self.metadata, "event_update")
        return self


class PortfolioBrief(_Report):
    exposure_summary: tuple[RoundedExposure, ...] = Field(min_length=1)
    relevant_news: tuple[PortfolioNewsItem, ...] = Field(min_length=1)
    value_chain_developments: tuple[ReportSection, ...] = Field(min_length=1)
    research_views: tuple[ResearchView, ...] = Field(min_length=1)
    change_history: tuple[ResearchViewChange, ...] = Field(min_length=1)
    concentration_and_correlation: ConcentrationCorrelation

    @model_validator(mode="after")
    def _type(self) -> "PortfolioBrief":
        self._require_type(self.metadata, "portfolio_brief")
        current_by_symbol: dict[str, RecommendationRating] = {}
        for view in self.research_views:
            symbol = view.symbol.casefold()
            if symbol in current_by_symbol:
                raise ValueError("current research view symbols must be unique")
            current_by_symbol[symbol] = view.rating

        history_by_symbol: dict[str, ResearchViewChange] = {}
        for entry in self.change_history:
            symbol = entry.symbol.casefold()
            if symbol in history_by_symbol:
                raise ValueError("rating history symbols must be unique")
            history_by_symbol[symbol] = entry

        if set(history_by_symbol) != set(current_by_symbol):
            raise ValueError(
                "rating history must contain exactly one entry per current view symbol"
            )
        for symbol, rating in current_by_symbol.items():
            if history_by_symbol[symbol].new_rating is not rating:
                raise ValueError("rating history must match the current research view rating")
        return self


class IndustryLandscape(_Report):
    industry_definition: ReportSection
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
    pdf_error: Literal[
        "pdf_backend_unavailable", "pdf_render_failed", "pdf_cleanup_failed"
    ] | None = None


# Explicit long-form aliases make the view-model purpose discoverable.
EventUpdateReport = EventUpdate
PortfolioBriefReport = PortfolioBrief
IndustryLandscapeReport = IndustryLandscape
EmergingCompanyMonitorReport = EmergingCompanyMonitor
ReportArtifact = RenderedReportArtifact
