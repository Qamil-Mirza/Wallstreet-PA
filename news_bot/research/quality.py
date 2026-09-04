"""Deterministic, fail-closed publication gates for investment research."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .agents.contracts import SecurityEligibility
from .models import InferenceMode, RecommendationRating, ReviewVerdict


class PublicationKind(str, Enum):
    """Research products whose release is controlled by the gates."""

    RECOMMENDATION = "recommendation"
    FOUNDATIONAL_REPORT = "foundational_report"
    PORTFOLIO_BRIEF = "portfolio_brief"
    EVENT_REPORT = "event_report"


class PublicationVerdict(str, Enum):
    """Whether an artifact is ready for final publication or retained as draft."""

    FINAL = "final"
    DRAFT = "draft"


class GateReasonCode(str, Enum):
    """Stable machine-readable reasons emitted by deterministic gates."""

    PORTFOLIO_MISSING = "portfolio_missing"
    PORTFOLIO_STALE = "portfolio_stale"
    PRICE_MISSING = "price_missing"
    PRICE_SESSION_UNKNOWN = "price_session_unknown"
    PRICE_NOT_TRADING_SESSION = "price_not_trading_session"
    PRICE_NOT_LATEST_SESSION = "price_not_latest_session"
    PRICE_STALE = "price_stale"
    FILING_STATUS_UNKNOWN = "filing_status_unknown"
    REQUIRED_FILING_UNAVAILABLE = "required_filing_unavailable"
    SECURITY_INELIGIBLE = "security_ineligible"
    MISSING_CLAIM_LINEAGE = "missing_claim_lineage"
    SEARCH_SNIPPET_INADMISSIBLE = "search_snippet_inadmissible"
    INSUFFICIENT_CORROBORATION = "insufficient_corroboration"
    UNRESOLVED_CONTRADICTION = "unresolved_contradiction"
    REVIEWER_NOT_PASSED = "reviewer_not_passed"
    EDITOR_UNAPPROVED_CLAIM = "editor_unapproved_claim"
    EXHIBIT_MISMATCH = "exhibit_mismatch"
    INFERENCE_DISCLOSURE_MISSING = "inference_disclosure_missing"
    LOCAL_CONFIDENCE_WEAK = "local_confidence_weak"
    EVENT_OUTSIDE_MATERIALITY_WINDOW = "event_outside_materiality_window"


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(char.isspace() for char in value):
        raise ValueError("identifier must be nonblank and contain no whitespace")
    return value


def _text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("text must be nonblank")
    return value.strip()


def _sorted_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(_identifier(value) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError("identifiers must be unique")
    return tuple(sorted(normalized))


class FrozenQualityContract(BaseModel):
    """Strict immutable gate boundary that revalidates copied/nested instances."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, revalidate_instances="always"
    )


class EvidenceReference(FrozenQualityContract):
    """One stored passage and the canonical source from which it came."""

    evidence_id: str
    canonical_source_id: str
    source_type: str
    stored: bool
    primary: bool

    _evidence_id = field_validator("evidence_id")(_identifier)
    _source_id = field_validator("canonical_source_id")(_identifier)
    _source_type = field_validator("source_type")(_identifier)

    @property
    def admissible(self) -> bool:
        return self.stored and self.source_type != "search_snippet"


class ClaimQualityInput(FrozenQualityContract):
    """Publication materiality and exact stored lineage for one claim."""

    claim_id: str
    section: str
    material: bool = False
    quantitative: bool = False
    consequential: bool = False
    evidence: tuple[EvidenceReference, ...] = ()

    _claim_id = field_validator("claim_id")(_identifier)
    _section = field_validator("section")(_identifier)

    @field_validator("evidence")
    @classmethod
    def _unique_evidence(
        cls, values: tuple[EvidenceReference, ...]
    ) -> tuple[EvidenceReference, ...]:
        identifiers = tuple(value.evidence_id for value in values)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("evidence IDs must be unique")
        return tuple(sorted(values, key=lambda value: value.evidence_id))


class MarketPrice(FrozenQualityContract):
    """Normalized closing price for one exchange trading session."""

    exchange: str
    session_date: date
    observed_at: datetime
    value: Decimal = Field(gt=Decimal("0"), allow_inf_nan=False)
    currency: str = Field(pattern=r"^[A-Z]{3}$")

    _exchange = field_validator("exchange")(_identifier)
    _observed = field_validator("observed_at")(_aware)

    @field_validator("value", mode="before")
    @classmethod
    def _exact_value(cls, value: object) -> object:
        if not isinstance(value, Decimal):
            raise TypeError("price value must be Decimal")
        return value

    @field_validator("currency", mode="before")
    @classmethod
    def _currency(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


class FilingAvailability(FrozenQualityContract):
    """As-of availability of the latest periodic filing required for a rating."""

    filing_due: bool | None
    available_as_of: bool | None
    filed_at: datetime | None = None

    @field_validator("filed_at")
    @classmethod
    def _filed_at(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware(value)

    @model_validator(mode="after")
    def _coherent(self) -> "FilingAvailability":
        if self.available_as_of is True and self.filed_at is None:
            raise ValueError("available filing requires filed_at")
        return self


class InferenceDisclosure(FrozenQualityContract):
    """Provider disclosure retained for both external and local-only inference."""

    mode: InferenceMode
    provider: str
    model: str
    confidence: Decimal = Field(ge=Decimal("0"), le=Decimal("1"), allow_inf_nan=False)

    _provider = field_validator("provider")(_text)
    _model = field_validator("model")(_text)

    @field_validator("confidence", mode="before")
    @classmethod
    def _exact_confidence(cls, value: object) -> object:
        if not isinstance(value, Decimal):
            raise TypeError("confidence must be Decimal")
        return value


class CalculatedRow(FrozenQualityContract):
    """One normalized or rendered exact-number exhibit row."""

    row_id: str
    values: Mapping[str, Decimal]

    _row_id = field_validator("row_id")(_identifier)

    @field_validator("values", mode="before")
    @classmethod
    def _decimal_values(cls, value: object) -> object:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("calculated row values must be a non-empty mapping")
        normalized: dict[str, Decimal] = {}
        for key, amount in value.items():
            safe_key = _identifier(key)
            if not isinstance(amount, Decimal):
                raise ValueError("calculated row values must be Decimal")
            if not amount.is_finite():
                raise ValueError("calculated row values must be finite")
            normalized[safe_key] = amount
        return normalized

    @field_validator("values")
    @classmethod
    def _freeze_values(cls, value: Mapping[str, Decimal]) -> Mapping[str, Decimal]:
        return MappingProxyType(dict(sorted(value.items())))


class CalculatedExhibit(FrozenQualityContract):
    """Rendered exhibit paired with the authoritative normalized rows."""

    exhibit_id: str
    normalized_rows: tuple[CalculatedRow, ...]
    rendered_rows: tuple[CalculatedRow, ...]
    section: str = "calculated_exhibits"

    _exhibit_id = field_validator("exhibit_id")(_identifier)
    _section = field_validator("section")(_identifier)

    @field_validator("normalized_rows", "rendered_rows")
    @classmethod
    def _unique_rows(cls, rows: tuple[CalculatedRow, ...]) -> tuple[CalculatedRow, ...]:
        identifiers = tuple(row.row_id for row in rows)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("calculated row IDs must be unique")
        return tuple(sorted(rows, key=lambda row: row.row_id))

    @property
    def reconciles(self) -> bool:
        def canonical(rows: tuple[CalculatedRow, ...]) -> tuple[tuple[str, tuple[tuple[str, Decimal], ...]], ...]:
            return tuple(
                (row.row_id, tuple(sorted(row.values.items())))
                for row in rows
            )

        return canonical(self.normalized_rows) == canonical(self.rendered_rows)


class ExchangeCalendar(FrozenQualityContract):
    """Calendar-aware exchange override used to resolve completed sessions."""

    exchange: str
    timezone_name: str
    close_time: time
    weekend_days: tuple[int, ...] = (5, 6)
    holidays: tuple[date, ...] = ()

    _exchange = field_validator("exchange")(_identifier)
    _timezone = field_validator("timezone_name")(_identifier)

    @field_validator("weekend_days")
    @classmethod
    def _weekends(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if len(values) != len(set(values)) or any(value < 0 or value > 6 for value in values):
            raise ValueError("weekend days must be unique values from 0 through 6")
        return tuple(sorted(values))

    @field_validator("holidays")
    @classmethod
    def _holidays(cls, values: tuple[date, ...]) -> tuple[date, ...]:
        if len(values) != len(set(values)):
            raise ValueError("holidays must be unique")
        return tuple(sorted(values))

    @model_validator(mode="after")
    def _known_timezone(self) -> "ExchangeCalendar":
        try:
            ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError:
            raise ValueError("exchange timezone is unknown") from None
        return self

    def is_session(self, value: date) -> bool:
        return value.weekday() not in self.weekend_days and value not in self.holidays

    def latest_completed_session(self, as_of: datetime) -> date:
        local = _aware(as_of).astimezone(ZoneInfo(self.timezone_name))
        candidate = local.date()
        if not self.is_session(candidate) or local.timetz().replace(tzinfo=None) < self.close_time:
            candidate -= timedelta(days=1)
        for _ in range(15):
            if self.is_session(candidate):
                return candidate
            candidate -= timedelta(days=1)
        raise ValueError("no completed trading session is known")


_NASDAQ_2026_HOLIDAYS = (
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
)
DEFAULT_EXCHANGE_CALENDARS: Mapping[str, ExchangeCalendar] = MappingProxyType({
    "NASDAQ": ExchangeCalendar(
        exchange="NASDAQ", timezone_name="America/New_York",
        close_time=time(16, 0), holidays=_NASDAQ_2026_HOLIDAYS,
    )
})


class QualityGatePolicy(FrozenQualityContract):
    """Versionable deterministic thresholds and source-specific overrides."""

    portfolio_max_age_hours: Decimal = Decimal("36")
    price_max_age_days: int = Field(default=4, ge=0)
    event_window_days: int = Field(default=7, ge=0)
    event_window_days_by_source: Mapping[str, int] = Field(default_factory=dict)
    local_min_confidence: Decimal = Decimal("0.70")

    @field_validator("portfolio_max_age_hours", "local_min_confidence", mode="before")
    @classmethod
    def _decimal_policy(cls, value: object) -> object:
        if not isinstance(value, Decimal):
            raise TypeError("quality thresholds must be Decimal")
        return value

    @field_validator("event_window_days_by_source", mode="before")
    @classmethod
    def _source_windows(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            raise TypeError("event source windows must be a mapping")
        result: dict[str, int] = {}
        for key, days in value.items():
            result[_identifier(key)] = days
            if not isinstance(days, int) or isinstance(days, bool) or days < 0:
                raise ValueError("event source windows must be non-negative integers")
        return result

    @field_validator("event_window_days_by_source")
    @classmethod
    def _freeze_source_windows(cls, value: Mapping[str, int]) -> Mapping[str, int]:
        return MappingProxyType(dict(sorted(value.items())))


class QualityGateInput(FrozenQualityContract):
    """Complete deterministic input required before publishing research."""

    as_of: datetime
    publication_kind: PublicationKind
    requested_rating: RecommendationRating = RecommendationRating.NO_RATING
    portfolio_snapshot_at: datetime | None
    price: MarketPrice | None
    filing: FilingAvailability | None
    security: SecurityEligibility
    claims: tuple[ClaimQualityInput, ...]
    contradictory_claim_ids: tuple[str, ...] = ()
    reviewer_verdict: ReviewVerdict
    reviewer_approved_claim_ids: tuple[str, ...] = ()
    editor_claim_ids: tuple[str, ...] = ()
    exhibits: tuple[CalculatedExhibit, ...] = ()
    inference_disclosure: InferenceDisclosure | None
    event_source_type: str | None = None
    event_published_at: datetime | None = None

    _as_of = field_validator("as_of")(_aware)
    _contradictions = field_validator("contradictory_claim_ids")(_sorted_unique)
    _approved = field_validator("reviewer_approved_claim_ids")(_sorted_unique)
    _editor = field_validator("editor_claim_ids")(_sorted_unique)

    @field_validator("portfolio_snapshot_at", "event_published_at")
    @classmethod
    def _optional_aware(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware(value)

    @field_validator("event_source_type")
    @classmethod
    def _optional_source(cls, value: str | None) -> str | None:
        return None if value is None else _identifier(value)

    @field_validator("claims")
    @classmethod
    def _unique_claims(cls, values: tuple[ClaimQualityInput, ...]) -> tuple[ClaimQualityInput, ...]:
        identifiers = tuple(value.claim_id for value in values)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("claim IDs must be unique")
        return tuple(sorted(values, key=lambda value: value.claim_id))

    @field_validator("exhibits")
    @classmethod
    def _unique_exhibits(cls, values: tuple[CalculatedExhibit, ...]) -> tuple[CalculatedExhibit, ...]:
        identifiers = tuple(value.exhibit_id for value in values)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("exhibit IDs must be unique")
        return tuple(sorted(values, key=lambda value: value.exhibit_id))

    @model_validator(mode="after")
    def _as_of_boundaries(self) -> "QualityGateInput":
        for field_name, value in (
            ("portfolio_snapshot_at", self.portfolio_snapshot_at),
            ("price.observed_at", None if self.price is None else self.price.observed_at),
            ("filing.filed_at", None if self.filing is None else self.filing.filed_at),
            ("event_published_at", self.event_published_at),
        ):
            if value is not None and value > self.as_of:
                raise ValueError(f"{field_name} cannot be after as_of")
        if (self.event_source_type is None) != (self.event_published_at is None):
            raise ValueError("event source type and publication time must be supplied together")
        claim_ids = {claim.claim_id for claim in self.claims}
        if not set(self.contradictory_claim_ids) <= claim_ids:
            raise ValueError("contradictory claims must be included in claims")
        return self


class QualityGateResult(FrozenQualityContract):
    """Deterministic release decision consumed by orchestration and rendering."""

    reason_codes: tuple[GateReasonCode, ...]
    allowed_sections: tuple[str, ...]
    allow_event_report: bool
    allow_sizing: bool
    publication_verdict: PublicationVerdict
    review_verdict: ReviewVerdict
    effective_rating: RecommendationRating
    portfolio_age_hours: Decimal | None
    inference_mode: InferenceMode | None
    inference_provider: str | None
    inference_model: str | None


def _canonical_rows(exhibit: CalculatedExhibit) -> bool:
    return exhibit.reconciles


def evaluate_quality_gates(
    gate_input: QualityGateInput,
    *,
    policy: QualityGatePolicy | None = None,
    calendars: Mapping[str, ExchangeCalendar] | None = None,
) -> QualityGateResult:
    """Compose all publication gates; model output cannot relax any failure."""
    data = QualityGateInput.model_validate(gate_input)
    rules = QualityGatePolicy.model_validate(policy or QualityGatePolicy())
    exchange_calendars = DEFAULT_EXCHANGE_CALENDARS if calendars is None else calendars
    reasons: set[GateReasonCode] = set()
    blocked_sections: set[str] = set()
    rating_blocked = False
    allow_sizing = True

    portfolio_age: Decimal | None = None
    if data.portfolio_snapshot_at is None:
        reasons.add(GateReasonCode.PORTFOLIO_MISSING)
        allow_sizing = False
        rating_blocked = True
    else:
        delta = data.as_of - data.portfolio_snapshot_at
        exact_microseconds = (
            Decimal(delta.days * 86400 + delta.seconds) * Decimal("1000000")
            + Decimal(delta.microseconds)
        )
        portfolio_age = exact_microseconds / Decimal("3600000000")
        if portfolio_age > rules.portfolio_max_age_hours:
            reasons.add(GateReasonCode.PORTFOLIO_STALE)
            allow_sizing = False
            rating_blocked = True

    if not data.security.rating_eligible:
        reasons.add(GateReasonCode.SECURITY_INELIGIBLE)
        allow_sizing = False
        rating_blocked = True

    if data.price is None:
        reasons.add(GateReasonCode.PRICE_MISSING)
        rating_blocked = True
    else:
        supplied_calendar = exchange_calendars.get(data.price.exchange)
        try:
            calendar = (
                None if supplied_calendar is None
                else ExchangeCalendar.model_validate(supplied_calendar)
            )
        except (TypeError, ValueError):
            calendar = None
        if calendar is None or calendar.exchange != data.price.exchange:
            reasons.add(GateReasonCode.PRICE_SESSION_UNKNOWN)
            rating_blocked = True
        else:
            local_observed = data.price.observed_at.astimezone(
                ZoneInfo(calendar.timezone_name)
            )
            observed_before_close = (
                local_observed.date() < data.price.session_date
                or (
                    local_observed.date() == data.price.session_date
                    and local_observed.timetz().replace(tzinfo=None) < calendar.close_time
                )
            )
            if observed_before_close:
                reasons.add(GateReasonCode.PRICE_SESSION_UNKNOWN)
                rating_blocked = True
            if not calendar.is_session(data.price.session_date):
                reasons.add(GateReasonCode.PRICE_NOT_TRADING_SESSION)
                rating_blocked = True
            try:
                latest = calendar.latest_completed_session(data.as_of)
            except ValueError:
                reasons.add(GateReasonCode.PRICE_SESSION_UNKNOWN)
                rating_blocked = True
            else:
                if data.price.session_date != latest:
                    reasons.add(GateReasonCode.PRICE_NOT_LATEST_SESSION)
                    rating_blocked = True
            exchange_date = data.as_of.astimezone(
                ZoneInfo(calendar.timezone_name)
            ).date()
            calendar_age = (exchange_date - data.price.session_date).days
            if calendar_age < 0 or calendar_age > rules.price_max_age_days:
                reasons.add(GateReasonCode.PRICE_STALE)
                rating_blocked = True

    if (
        data.filing is None
        or data.filing.filing_due is None
        or data.filing.available_as_of is None
    ):
        reasons.add(GateReasonCode.FILING_STATUS_UNKNOWN)
        rating_blocked = True
    elif not data.filing.available_as_of:
        reasons.add(GateReasonCode.REQUIRED_FILING_UNAVAILABLE)
        rating_blocked = True

    hard_content_failure = False
    for item in data.claims:
        admissible = tuple(reference for reference in item.evidence if reference.admissible)
        if any(reference.source_type == "search_snippet" for reference in item.evidence):
            reasons.add(GateReasonCode.SEARCH_SNIPPET_INADMISSIBLE)
        if (item.material or item.quantitative) and not admissible:
            reasons.add(GateReasonCode.MISSING_CLAIM_LINEAGE)
            blocked_sections.add(item.section)
            hard_content_failure = True
            if item.section == "recommendation":
                rating_blocked = True
        if item.consequential:
            canonical = {reference.canonical_source_id for reference in admissible}
            has_primary = any(reference.primary for reference in admissible)
            if not has_primary and len(canonical) < 2:
                reasons.add(GateReasonCode.INSUFFICIENT_CORROBORATION)
                blocked_sections.add(item.section)
                hard_content_failure = True
                if item.section == "recommendation":
                    rating_blocked = True

    claim_sections = {item.claim_id: item.section for item in data.claims}
    for claim_id in data.contradictory_claim_ids:
        reasons.add(GateReasonCode.UNRESOLVED_CONTRADICTION)
        blocked_sections.add(claim_sections[claim_id])
        hard_content_failure = True
        if claim_sections[claim_id] == "recommendation":
            rating_blocked = True

    requires_pass = data.publication_kind in {
        PublicationKind.RECOMMENDATION,
        PublicationKind.FOUNDATIONAL_REPORT,
        PublicationKind.PORTFOLIO_BRIEF,
        PublicationKind.EVENT_REPORT,
    }
    if requires_pass and data.reviewer_verdict is not ReviewVerdict.PASS:
        reasons.add(GateReasonCode.REVIEWER_NOT_PASSED)
        rating_blocked = True
        blocked_sections.update(item.section for item in data.claims)

    if not set(data.editor_claim_ids) <= set(data.reviewer_approved_claim_ids):
        reasons.add(GateReasonCode.EDITOR_UNAPPROVED_CLAIM)
        hard_content_failure = True
        rating_blocked = True

    for exhibit in data.exhibits:
        if not _canonical_rows(exhibit):
            reasons.add(GateReasonCode.EXHIBIT_MISMATCH)
            blocked_sections.add(exhibit.section)
            hard_content_failure = True

    if data.inference_disclosure is None:
        reasons.add(GateReasonCode.INFERENCE_DISCLOSURE_MISSING)
        rating_blocked = True
    elif (
        data.inference_disclosure.mode is InferenceMode.LOCAL_ONLY
        and data.inference_disclosure.confidence < rules.local_min_confidence
    ):
        reasons.add(GateReasonCode.LOCAL_CONFIDENCE_WEAK)
        rating_blocked = True

    event_in_window = True
    if data.event_published_at is not None and data.event_source_type is not None:
        window = rules.event_window_days_by_source.get(
            data.event_source_type, rules.event_window_days
        )
        if data.as_of - data.event_published_at > timedelta(days=window):
            reasons.add(GateReasonCode.EVENT_OUTSIDE_MATERIALITY_WINDOW)
            event_in_window = False

    allowed_sections = tuple(sorted({item.section for item in data.claims} - blocked_sections))
    effective_review = ReviewVerdict.BLOCK if hard_content_failure else data.reviewer_verdict
    if hard_content_failure:
        rating_blocked = True
    effective_rating = (
        RecommendationRating.NO_RATING if rating_blocked
        else data.requested_rating
    )
    allow_sizing = (
        allow_sizing and effective_rating is not RecommendationRating.NO_RATING
    )
    event_release_blocked = (
        not event_in_window
        or hard_content_failure
        or data.reviewer_verdict is not ReviewVerdict.PASS
        or GateReasonCode.EDITOR_UNAPPROVED_CLAIM in reasons
        or GateReasonCode.INFERENCE_DISCLOSURE_MISSING in reasons
        or GateReasonCode.LOCAL_CONFIDENCE_WEAK in reasons
    )
    allow_event_report = event_in_window and not (
        data.publication_kind is PublicationKind.EVENT_REPORT and event_release_blocked
    )
    if data.publication_kind is PublicationKind.EVENT_REPORT:
        publication_blocked = event_release_blocked
    else:
        publication_blocked = bool(reasons)
    publication_verdict = (
        PublicationVerdict.DRAFT if publication_blocked else PublicationVerdict.FINAL
    )
    inference = data.inference_disclosure
    return QualityGateResult(
        reason_codes=tuple(sorted(reasons, key=lambda reason: reason.value)),
        allowed_sections=allowed_sections,
        allow_event_report=allow_event_report,
        allow_sizing=allow_sizing,
        publication_verdict=publication_verdict,
        review_verdict=effective_review,
        effective_rating=effective_rating,
        portfolio_age_hours=portfolio_age,
        inference_mode=None if inference is None else inference.mode,
        inference_provider=None if inference is None else inference.provider,
        inference_model=None if inference is None else inference.model,
    )


__all__ = [
    "CalculatedExhibit", "CalculatedRow", "ClaimQualityInput",
    "DEFAULT_EXCHANGE_CALENDARS", "EvidenceReference", "ExchangeCalendar",
    "FilingAvailability", "GateReasonCode", "InferenceDisclosure", "MarketPrice",
    "PublicationKind", "PublicationVerdict", "QualityGateInput",
    "QualityGatePolicy", "QualityGateResult", "evaluate_quality_gates",
]
