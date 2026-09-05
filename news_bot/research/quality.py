"""Deterministic, fail-closed publication gates for investment research."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol
import unicodedata
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

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
    PARTIAL = "partial"
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
    PRICE_CURRENCY_MISMATCH = "price_currency_mismatch"
    CALENDAR_COVERAGE_UNKNOWN = "calendar_coverage_unknown"
    FILING_STATUS_UNKNOWN = "filing_status_unknown"
    REQUIRED_FILING_UNAVAILABLE = "required_filing_unavailable"
    SECURITY_INELIGIBLE = "security_ineligible"
    MISSING_CLAIM_LINEAGE = "missing_claim_lineage"
    SEARCH_SNIPPET_INADMISSIBLE = "search_snippet_inadmissible"
    INSUFFICIENT_CORROBORATION = "insufficient_corroboration"
    UNRESOLVED_CONTRADICTION = "unresolved_contradiction"
    REVIEWER_NOT_PASSED = "reviewer_not_passed"
    REVIEWER_APPROVAL_MISSING = "reviewer_approval_missing"
    EDITOR_UNAPPROVED_CLAIM = "editor_unapproved_claim"
    EDITOR_SELECTION_MISSING = "editor_selection_missing"
    EXHIBIT_MISMATCH = "exhibit_mismatch"
    INFERENCE_DISCLOSURE_MISSING = "inference_disclosure_missing"
    LOCAL_CONFIDENCE_WEAK = "local_confidence_weak"
    EVENT_OUTSIDE_MATERIALITY_WINDOW = "event_outside_materiality_window"
    EVENT_FRESHNESS_UNKNOWN = "event_freshness_unknown"
    RESEARCH_CLAIMS_MISSING = "research_claims_missing"
    APPROVED_CLAIM_UNKNOWN = "approved_claim_unknown"
    EDITOR_CLAIM_UNKNOWN = "editor_claim_unknown"
    RECOMMENDATION_EVIDENCE_MISSING = "recommendation_evidence_missing"


def _aware(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise ValueError("datetime must be UTC")
    return value.astimezone(timezone.utc)


def _identifier(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("identifier must be nonblank and contain no whitespace")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized.strip() or any(char.isspace() for char in normalized):
        raise ValueError("identifier must be nonblank and contain no whitespace")
    return normalized


def _text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("text must be nonblank")
    return value.strip()


def _source_category(value: str) -> str:
    return _identifier(value).casefold()


def _comparison_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _sorted_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(_identifier(value) for value in values)
    if len(normalized) != len({_comparison_key(value) for value in normalized}):
        raise ValueError("identifiers must be unique")
    return tuple(sorted(normalized, key=_comparison_key))


def _canonical_values(values: tuple[str, ...]) -> tuple[str, ...]:
    by_key: dict[str, set[str]] = {}
    for value in values:
        normalized = _identifier(value)
        by_key.setdefault(_comparison_key(normalized), set()).add(normalized)
    return tuple(
        next(iter(by_key[key])) if len(by_key[key]) == 1 else key
        for key in sorted(by_key)
    )


def _canonical_categories(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({_source_category(value) for value in values}))


class FrozenQualityContract(BaseModel):
    """Strict immutable gate boundary that revalidates copied/nested instances."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        validate_default=True,
    )


class EvidenceReference(FrozenQualityContract):
    """One stored passage and the canonical source from which it came."""

    evidence_id: str
    canonical_source_id: str
    source_family: str
    source_type: str
    stored: StrictBool
    primary: StrictBool
    discovery_snippet: StrictBool = False
    published_at: datetime | None = None
    retrieved_at: datetime | None = None

    _evidence_id = field_validator("evidence_id")(_identifier)
    _source_id = field_validator("canonical_source_id")(_identifier)
    _source_family = field_validator("source_family")(_source_category)
    _source_type = field_validator("source_type")(_source_category)

    @field_validator("published_at", "retrieved_at")
    @classmethod
    def _source_dates(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware(value)

    @property
    def admissible(self) -> bool:
        return (
            self.stored
            and not self.discovery_snippet
            and self.source_type != "search_snippet"
        )


SourceReference = EvidenceReference


class ClaimQualityInput(FrozenQualityContract):
    """Publication materiality and exact stored lineage for one claim."""

    claim_id: str
    section: str
    material: StrictBool = False
    quantitative: StrictBool = False
    consequential: StrictBool = False
    evidence: tuple[EvidenceReference, ...] = ()

    _claim_id = field_validator("claim_id")(_identifier)
    _section = field_validator("section")(_source_category)

    @field_validator("evidence")
    @classmethod
    def _unique_evidence(
        cls, values: tuple[EvidenceReference, ...]
    ) -> tuple[EvidenceReference, ...]:
        identifiers = tuple(_comparison_key(value.evidence_id) for value in values)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("evidence IDs must be unique")
        return tuple(
            sorted(values, key=lambda value: _comparison_key(value.evidence_id))
        )


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

    filing_due: StrictBool | None
    available_as_of: StrictBool | None
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

    _row_id = field_validator("row_id")(_source_category)

    @field_validator("values", mode="before")
    @classmethod
    def _decimal_values(cls, value: object, info: ValidationInfo) -> object:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("calculated row values must be a non-empty mapping")
        normalized: dict[str, Decimal] = {}
        for key, amount in value.items():
            safe_key = _source_category(key)
            if safe_key in normalized:
                raise ValueError("calculated row keys contain a normalized duplicate")
            if not isinstance(amount, Decimal):
                if info.mode == "json" and isinstance(amount, str):
                    try:
                        amount = Decimal(amount)
                    except Exception:
                        raise ValueError(
                            "calculated row values must be Decimal"
                        ) from None
                else:
                    raise ValueError("calculated row values must be Decimal")
            if not amount.is_finite():
                raise ValueError("calculated row values must be finite")
            normalized[safe_key] = amount
        return normalized

    @field_validator("values")
    @classmethod
    def _freeze_values(cls, value: Mapping[str, Decimal]) -> Mapping[str, Decimal]:
        return MappingProxyType(dict(sorted(value.items())))

    @field_serializer("values")
    def _serialize_values(self, value: Mapping[str, Decimal]) -> dict[str, Decimal]:
        return dict(value)


class CalculatedExhibit(FrozenQualityContract):
    """Rendered exhibit paired with the authoritative normalized rows."""

    exhibit_id: str
    normalized_rows: tuple[CalculatedRow, ...]
    rendered_rows: tuple[CalculatedRow, ...]
    section: str = "calculated_exhibits"

    _exhibit_id = field_validator("exhibit_id")(_identifier)
    _section = field_validator("section")(_source_category)

    @field_validator("normalized_rows", "rendered_rows")
    @classmethod
    def _unique_rows(cls, rows: tuple[CalculatedRow, ...]) -> tuple[CalculatedRow, ...]:
        identifiers = tuple(_comparison_key(row.row_id) for row in rows)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("calculated row IDs must be unique")
        return tuple(sorted(rows, key=lambda row: row.row_id))

    @property
    def reconciles(self) -> bool:
        def canonical(
            rows: tuple[CalculatedRow, ...],
        ) -> tuple[tuple[str, tuple[tuple[str, Decimal], ...]], ...]:
            return tuple(
                (row.row_id, tuple(sorted(row.values.items()))) for row in rows
            )

        return canonical(self.normalized_rows) == canonical(self.rendered_rows)


class ExchangeCalendar(FrozenQualityContract):
    """Calendar-aware exchange override used to resolve completed sessions."""

    exchange: str
    timezone_name: str
    close_time: time
    coverage_start: date
    coverage_end: date
    weekend_days: tuple[int, ...] = (5, 6)
    holidays: tuple[date, ...] = ()
    session_overrides: Mapping[date, StrictBool] = Field(default_factory=dict)

    _exchange = field_validator("exchange")(_identifier)
    _timezone = field_validator("timezone_name")(_identifier)

    @field_validator("close_time")
    @classmethod
    def _local_close_time(cls, value: time) -> time:
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ValueError("exchange close must be a naive local wall time")
        return value

    @field_validator("weekend_days")
    @classmethod
    def _weekends(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if len(values) != len(set(values)) or any(
            value < 0 or value > 6 for value in values
        ):
            raise ValueError("weekend days must be unique values from 0 through 6")
        return tuple(sorted(values))

    @field_validator("holidays")
    @classmethod
    def _holidays(cls, values: tuple[date, ...]) -> tuple[date, ...]:
        if len(values) != len(set(values)):
            raise ValueError("holidays must be unique")
        return tuple(sorted(values))

    @field_validator("session_overrides", mode="before")
    @classmethod
    def _session_overrides(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            raise TypeError("session overrides must be a mapping")
        result: dict[date, bool] = {}
        for day, is_session in value.items():
            if not isinstance(day, date) or isinstance(day, datetime):
                raise ValueError("session override keys must be dates")
            if not isinstance(is_session, bool):
                raise ValueError("session override values must be booleans")
            if day in result:
                raise ValueError("session overrides contain a duplicate date")
            result[day] = is_session
        return result

    @field_validator("session_overrides")
    @classmethod
    def _freeze_session_overrides(
        cls, value: Mapping[date, bool]
    ) -> Mapping[date, bool]:
        return MappingProxyType(dict(sorted(value.items())))

    @field_serializer("session_overrides")
    def _serialize_session_overrides(
        self, value: Mapping[date, bool]
    ) -> dict[str, bool]:
        return {day.isoformat(): flag for day, flag in value.items()}

    @model_validator(mode="after")
    def _known_timezone(self) -> "ExchangeCalendar":
        if self.coverage_start > self.coverage_end:
            raise ValueError("calendar coverage start must not exceed coverage end")
        if any(
            day < self.coverage_start or day > self.coverage_end
            for day in (*self.holidays, *self.session_overrides)
        ):
            raise ValueError("calendar dates must be inside coverage")
        try:
            ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError:
            raise ValueError("exchange timezone is unknown") from None
        return self

    def is_session(self, value: date) -> bool:
        if not self.covers(value):
            raise ValueError("calendar coverage is unknown")
        override = self.session_overrides.get(value)
        if override is not None:
            return override
        return value.weekday() not in self.weekend_days and value not in self.holidays

    def covers(self, value: date) -> bool:
        return self.coverage_start <= value <= self.coverage_end

    def latest_completed_session(self, as_of: datetime) -> date:
        local = _aware(as_of).astimezone(ZoneInfo(self.timezone_name))
        candidate = local.date()
        if (
            not self.is_session(candidate)
            or local.timetz().replace(tzinfo=None) < self.close_time
        ):
            candidate -= timedelta(days=1)
        for _ in range(15):
            if self.is_session(candidate):
                return candidate
            candidate -= timedelta(days=1)
        raise ValueError("no completed trading session is known")


_NASDAQ_2026_HOLIDAYS = (
    date(2026, 1, 1),
    date(2026, 1, 19),
    date(2026, 2, 16),
    date(2026, 4, 3),
    date(2026, 5, 25),
    date(2026, 6, 19),
    date(2026, 7, 3),
    date(2026, 9, 7),
    date(2026, 11, 26),
    date(2026, 12, 25),
)
DEFAULT_EXCHANGE_CALENDARS: Mapping[str, ExchangeCalendar] = MappingProxyType(
    {
        "NASDAQ": ExchangeCalendar(
            exchange="NASDAQ",
            timezone_name="America/New_York",
            close_time=time(16, 0),
            coverage_start=date(2026, 1, 1),
            coverage_end=date(2026, 12, 31),
            holidays=_NASDAQ_2026_HOLIDAYS,
        )
    }
)


class QualityGatePolicy(FrozenQualityContract):
    """Versionable deterministic thresholds and source-specific overrides."""

    portfolio_max_age_hours: StrictInt = Field(default=36, gt=0)
    price_max_age_days: StrictInt = Field(default=4, gt=0)
    event_window_days: StrictInt = Field(default=7, gt=0)
    event_window_days_by_source: Mapping[str, StrictInt] = Field(default_factory=dict)
    local_min_confidence: Decimal = Field(
        default=Decimal("0.70"),
        ge=Decimal("0"),
        le=Decimal("1"),
        allow_inf_nan=False,
    )

    @field_validator("local_min_confidence", mode="before")
    @classmethod
    def _decimal_policy(cls, value: object, info: ValidationInfo) -> object:
        if not isinstance(value, Decimal):
            if info.mode == "json" and isinstance(value, str):
                try:
                    return Decimal(value)
                except Exception:
                    raise ValueError("quality thresholds must be Decimal") from None
            raise TypeError("quality thresholds must be Decimal")
        return value

    @field_validator("event_window_days_by_source", mode="before")
    @classmethod
    def _source_windows(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            raise TypeError("event source windows must be a mapping")
        result: dict[str, int] = {}
        for key, days in value.items():
            safe_key = _source_category(key)
            if safe_key in result:
                raise ValueError("event source windows contain a normalized duplicate")
            if not isinstance(days, int) or isinstance(days, bool) or days <= 0:
                raise ValueError("event source windows must be positive integers")
            result[safe_key] = days
        return result

    @field_validator("event_window_days_by_source")
    @classmethod
    def _freeze_source_windows(cls, value: Mapping[str, int]) -> Mapping[str, int]:
        return MappingProxyType(dict(sorted(value.items())))

    @field_serializer("event_window_days_by_source")
    def _serialize_source_windows(self, value: Mapping[str, int]) -> dict[str, int]:
        return dict(value)


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
        return None if value is None else _source_category(value)

    @field_validator("claims")
    @classmethod
    def _unique_claims(
        cls, values: tuple[ClaimQualityInput, ...]
    ) -> tuple[ClaimQualityInput, ...]:
        identifiers = tuple(_comparison_key(value.claim_id) for value in values)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("claim IDs must be unique")
        return tuple(sorted(values, key=lambda value: _comparison_key(value.claim_id)))

    @field_validator("exhibits")
    @classmethod
    def _unique_exhibits(
        cls, values: tuple[CalculatedExhibit, ...]
    ) -> tuple[CalculatedExhibit, ...]:
        identifiers = tuple(_comparison_key(value.exhibit_id) for value in values)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("exhibit IDs must be unique")
        return tuple(sorted(values, key=lambda value: value.exhibit_id))

    @model_validator(mode="after")
    def _as_of_boundaries(self) -> "QualityGateInput":
        for field_name, value in (
            ("portfolio_snapshot_at", self.portfolio_snapshot_at),
            (
                "price.observed_at",
                None if self.price is None else self.price.observed_at,
            ),
            ("filing.filed_at", None if self.filing is None else self.filing.filed_at),
            ("event_published_at", self.event_published_at),
        ):
            if value is not None and value > self.as_of:
                raise ValueError(f"{field_name} cannot be after as_of")
        claim_ids = {_comparison_key(claim.claim_id) for claim in self.claims}
        if (
            not {_comparison_key(value) for value in self.contradictory_claim_ids}
            <= claim_ids
        ):
            raise ValueError("contradictory claims must be included in claims")
        evidence_sources: dict[str, tuple[object, ...]] = {}
        canonical_sources: dict[str, tuple[object, ...]] = {}
        for item in self.claims:
            for reference in item.evidence:
                for field_name, value in (
                    ("evidence.published_at", reference.published_at),
                    ("evidence.retrieved_at", reference.retrieved_at),
                ):
                    if value is not None and value > self.as_of:
                        raise ValueError(f"{field_name} cannot be after as_of")
                evidence_key = _comparison_key(reference.evidence_id)
                canonical_key = _comparison_key(reference.canonical_source_id)
                source_identity = (
                    canonical_key,
                    reference.source_family,
                    reference.source_type,
                    reference.stored,
                    reference.primary,
                    reference.discovery_snippet,
                    reference.published_at,
                    reference.retrieved_at,
                )
                if (
                    evidence_key in evidence_sources
                    and evidence_sources[evidence_key] != source_identity
                ):
                    raise ValueError("evidence source mapping is inconsistent")
                evidence_sources[evidence_key] = source_identity
                canonical_identity = (
                    reference.source_family,
                    reference.source_type,
                    reference.stored,
                    reference.primary,
                    reference.discovery_snippet,
                    reference.published_at,
                    reference.retrieved_at,
                )
                if (
                    canonical_key in canonical_sources
                    and canonical_sources[canonical_key] != canonical_identity
                ):
                    raise ValueError("evidence source mapping is inconsistent")
                canonical_sources[canonical_key] = canonical_identity
        return self


class QualityGateResult(FrozenQualityContract):
    """Deterministic release decision consumed by orchestration and rendering."""

    reason_codes: tuple[GateReasonCode, ...]
    allowed_sections: tuple[str, ...]
    allowed_claim_ids: tuple[str, ...]
    allow_event_report: StrictBool
    allow_sizing: StrictBool
    publication_verdict: PublicationVerdict
    review_verdict: ReviewVerdict
    effective_rating: RecommendationRating
    portfolio_age_hours: Decimal | None = Field(
        default=None, ge=Decimal("0"), allow_inf_nan=False
    )
    inference_mode: InferenceMode | None
    inference_provider: str | None
    inference_model: str | None

    @field_validator("reason_codes")
    @classmethod
    def _canonical_reasons(
        cls, values: tuple[GateReasonCode, ...]
    ) -> tuple[GateReasonCode, ...]:
        return tuple(sorted(set(values), key=lambda value: value.value))

    _canonical_sections = field_validator("allowed_sections")(_canonical_categories)
    _canonical_claims = field_validator("allowed_claim_ids")(_canonical_values)

    @field_validator("inference_provider", "inference_model")
    @classmethod
    def _optional_inference_text(cls, value: str | None) -> str | None:
        return None if value is None else _text(value)

    @model_validator(mode="after")
    def _safety_invariants(self) -> "QualityGateResult":
        publishable = self.publication_verdict in {
            PublicationVerdict.FINAL,
            PublicationVerdict.PARTIAL,
        }
        if bool(self.allowed_claim_ids) != bool(self.allowed_sections):
            raise ValueError("allowed claims and sections must be coherent")
        if publishable and (
            self.review_verdict is not ReviewVerdict.PASS
            or not self.allowed_claim_ids
            or not self.allowed_sections
        ):
            raise ValueError("publishable result requires reviewed allowed content")
        metadata_present = (
            self.inference_provider is not None or self.inference_model is not None
        )
        if (self.inference_mode is None) != (not metadata_present):
            raise ValueError("inference metadata is inconsistent")
        if self.inference_mode is not None and (
            self.inference_provider is None or self.inference_model is None
        ):
            raise ValueError("inference metadata is inconsistent")
        if publishable and self.inference_mode is None:
            raise ValueError("publishable result requires inference disclosure")
        if (
            self.effective_rating is RecommendationRating.NO_RATING
            and self.allow_sizing
        ):
            raise ValueError("no-rating result cannot allow sizing")
        if self.publication_verdict is PublicationVerdict.DRAFT and (
            self.effective_rating is not RecommendationRating.NO_RATING
            or self.allow_sizing
        ):
            raise ValueError("draft result cannot retain a rating or sizing")
        return self

    @property
    def rating(self) -> RecommendationRating:
        """Compatibility name used by recommendation orchestration."""
        return self.effective_rating


class GateDecision(FrozenQualityContract):
    """One component's order-independent contribution to the release decision."""

    reason_codes: tuple[GateReasonCode, ...] = ()
    affected_sections: tuple[str, ...] = ()
    rejected_claim_ids: tuple[str, ...] = ()
    block_rating: StrictBool = False
    block_sizing: StrictBool = False
    block_publication: StrictBool = False
    block_event: StrictBool = False
    portfolio_age_hours: Decimal | None = Field(
        default=None, ge=Decimal("0"), allow_inf_nan=False
    )

    @field_validator("reason_codes")
    @classmethod
    def _reasons(cls, values: tuple[GateReasonCode, ...]) -> tuple[GateReasonCode, ...]:
        return tuple(sorted(set(values), key=lambda value: value.value))

    _sections = field_validator("affected_sections")(_canonical_categories)
    _claims = field_validator("rejected_claim_ids")(_canonical_values)


class ComponentGate(Protocol):
    def evaluate(self, context: QualityGateInput) -> GateDecision: ...


def _data(context: QualityGateInput) -> QualityGateInput:
    return QualityGateInput.model_validate(context)


def _claim_lookup(data: QualityGateInput) -> dict[str, ClaimQualityInput]:
    return {_comparison_key(item.claim_id): item for item in data.claims}


def _admissible(item: ClaimQualityInput) -> tuple[EvidenceReference, ...]:
    return tuple(reference for reference in item.evidence if reference.admissible)


class EvidenceLineageGate:
    """Reject claims that cannot be traced to admissible stored evidence."""

    def evaluate(self, context: QualityGateInput) -> GateDecision:
        data = _data(context)
        reasons: set[GateReasonCode] = set()
        sections: set[str] = set()
        rejected: set[str] = set()
        if not any(_admissible(item) for item in data.claims):
            reasons.add(GateReasonCode.RESEARCH_CLAIMS_MISSING)
        for item in data.claims:
            admissible = _admissible(item)
            has_snippet = any(
                reference.source_type == "search_snippet" for reference in item.evidence
            )
            if has_snippet:
                reasons.add(GateReasonCode.SEARCH_SNIPPET_INADMISSIBLE)
                sections.add(item.section)
                rejected.add(item.claim_id)
            if not admissible:
                rejected.add(item.claim_id)
                sections.add(item.section)
                if item.material or item.quantitative:
                    reasons.add(GateReasonCode.MISSING_CLAIM_LINEAGE)
        no_research = GateReasonCode.RESEARCH_CLAIMS_MISSING in reasons
        return GateDecision(
            reason_codes=tuple(reasons),
            affected_sections=tuple(sections),
            rejected_claim_ids=tuple(rejected),
            block_rating=no_research,
            block_publication=no_research,
        )


class CorroborationGate:
    """Require a primary source or two genuinely independent source families."""

    def evaluate(self, context: QualityGateInput) -> GateDecision:
        data = _data(context)
        rejected: set[str] = set()
        sections: set[str] = set()
        for item in data.claims:
            if not item.consequential:
                continue
            admissible = _admissible(item)
            if (
                not any(reference.primary for reference in admissible)
                and len({reference.source_family for reference in admissible}) < 2
            ):
                rejected.add(item.claim_id)
                sections.add(item.section)
        return GateDecision(
            reason_codes=(
                (GateReasonCode.INSUFFICIENT_CORROBORATION,) if rejected else ()
            ),
            affected_sections=tuple(sections),
            rejected_claim_ids=tuple(rejected),
        )


class PortfolioFreshnessGate:
    def __init__(self, policy: QualityGatePolicy | None = None) -> None:
        self._policy = QualityGatePolicy.model_validate(policy or QualityGatePolicy())

    def evaluate(self, context: QualityGateInput) -> GateDecision:
        data = _data(context)
        if data.portfolio_snapshot_at is None:
            return GateDecision(
                reason_codes=(GateReasonCode.PORTFOLIO_MISSING,),
                block_rating=True,
                block_sizing=True,
            )
        delta = data.as_of - data.portfolio_snapshot_at
        microseconds = Decimal(
            (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
        )
        age = microseconds / Decimal("3600000000")
        stale = age > Decimal(self._policy.portfolio_max_age_hours)
        return GateDecision(
            reason_codes=(GateReasonCode.PORTFOLIO_STALE,) if stale else (),
            block_rating=stale,
            block_sizing=stale,
            portfolio_age_hours=age,
        )


def _calendar_map(
    calendars: Mapping[str, ExchangeCalendar] | None,
) -> Mapping[str, ExchangeCalendar]:
    source = DEFAULT_EXCHANGE_CALENDARS if calendars is None else calendars
    if not isinstance(source, Mapping):
        raise TypeError("exchange calendars must be a mapping")
    result: dict[str, ExchangeCalendar] = {}
    for key, calendar in source.items():
        normalized = _comparison_key(_identifier(key))
        if normalized in result:
            raise ValueError("exchange calendars contain a normalized duplicate")
        result[normalized] = ExchangeCalendar.model_validate(calendar)
    return MappingProxyType(result)


class PriceFreshnessGate:
    def __init__(
        self,
        policy: QualityGatePolicy | None = None,
        calendars: Mapping[str, ExchangeCalendar] | None = None,
    ) -> None:
        self._policy = QualityGatePolicy.model_validate(policy or QualityGatePolicy())
        try:
            self._calendars: Mapping[str, ExchangeCalendar] | None = _calendar_map(
                calendars
            )
        except (TypeError, ValueError):
            self._calendars = None

    def evaluate(self, context: QualityGateInput) -> GateDecision:
        data = _data(context)
        if data.price is None:
            return GateDecision(
                reason_codes=(GateReasonCode.PRICE_MISSING,), block_rating=True
            )
        reasons: set[GateReasonCode] = set()
        price = data.price
        if price.currency != data.security.currency:
            reasons.add(GateReasonCode.PRICE_CURRENCY_MISMATCH)
        calendar = (
            None
            if self._calendars is None
            else self._calendars.get(_comparison_key(price.exchange))
        )
        if calendar is None or _comparison_key(calendar.exchange) != _comparison_key(
            price.exchange
        ):
            reasons.add(GateReasonCode.PRICE_SESSION_UNKNOWN)
        else:
            try:
                zone = ZoneInfo(calendar.timezone_name)
                local_as_of = data.as_of.astimezone(zone)
                local_observed = price.observed_at.astimezone(zone)
                if not calendar.covers(local_as_of.date()) or not calendar.covers(
                    price.session_date
                ):
                    reasons.add(GateReasonCode.CALENDAR_COVERAGE_UNKNOWN)
                else:
                    if local_observed.date() < price.session_date or (
                        local_observed.date() == price.session_date
                        and local_observed.timetz().replace(tzinfo=None)
                        < calendar.close_time
                    ):
                        reasons.add(GateReasonCode.PRICE_SESSION_UNKNOWN)
                    if not calendar.is_session(price.session_date):
                        reasons.add(GateReasonCode.PRICE_NOT_TRADING_SESSION)
                    if price.session_date != calendar.latest_completed_session(
                        data.as_of
                    ):
                        reasons.add(GateReasonCode.PRICE_NOT_LATEST_SESSION)
                    age = (local_as_of.date() - price.session_date).days
                    if age < 0 or age > self._policy.price_max_age_days:
                        reasons.add(GateReasonCode.PRICE_STALE)
            except (ValueError, ZoneInfoNotFoundError):
                reasons.add(GateReasonCode.CALENDAR_COVERAGE_UNKNOWN)
        return GateDecision(reason_codes=tuple(reasons), block_rating=bool(reasons))


class FilingFreshnessGate:
    def evaluate(self, context: QualityGateInput) -> GateDecision:
        data = _data(context)
        if (
            data.filing is None
            or data.filing.filing_due is None
            or data.filing.available_as_of is None
        ):
            return GateDecision(
                reason_codes=(GateReasonCode.FILING_STATUS_UNKNOWN,), block_rating=True
            )
        if not data.filing.available_as_of:
            return GateDecision(
                reason_codes=(GateReasonCode.REQUIRED_FILING_UNAVAILABLE,),
                block_rating=True,
            )
        return GateDecision()


class EligibilityGate:
    def evaluate(self, context: QualityGateInput) -> GateDecision:
        eligible = _data(context).security.rating_eligible
        return GateDecision(
            reason_codes=() if eligible else (GateReasonCode.SECURITY_INELIGIBLE,),
            block_rating=not eligible,
            block_sizing=not eligible,
        )


class ContradictionGate:
    def evaluate(self, context: QualityGateInput) -> GateDecision:
        data = _data(context)
        lookup = _claim_lookup(data)
        rejected = tuple(
            lookup[_comparison_key(claim_id)].claim_id
            for claim_id in data.contradictory_claim_ids
        )
        sections = tuple(lookup[_comparison_key(value)].section for value in rejected)
        return GateDecision(
            reason_codes=(GateReasonCode.UNRESOLVED_CONTRADICTION,) if rejected else (),
            affected_sections=sections,
            rejected_claim_ids=rejected,
        )


class ReviewerGate:
    def evaluate(self, context: QualityGateInput) -> GateDecision:
        data = _data(context)
        actual = set(_claim_lookup(data))
        approved = {
            _comparison_key(value) for value in data.reviewer_approved_claim_ids
        }
        reasons: set[GateReasonCode] = set()
        if data.reviewer_verdict is not ReviewVerdict.PASS:
            reasons.add(GateReasonCode.REVIEWER_NOT_PASSED)
        if not approved:
            reasons.add(GateReasonCode.REVIEWER_APPROVAL_MISSING)
        if not approved <= actual:
            reasons.add(GateReasonCode.APPROVED_CLAIM_UNKNOWN)
        failed = bool(reasons)
        return GateDecision(
            reason_codes=tuple(reasons),
            affected_sections=(
                tuple(item.section for item in data.claims) if failed else ()
            ),
            rejected_claim_ids=(
                tuple(item.claim_id for item in data.claims) if failed else ()
            ),
            block_rating=failed,
            block_publication=failed,
        )


class EditorGate:
    def evaluate(self, context: QualityGateInput) -> GateDecision:
        data = _data(context)
        actual = set(_claim_lookup(data))
        approved = {
            _comparison_key(value) for value in data.reviewer_approved_claim_ids
        }
        edited = {_comparison_key(value) for value in data.editor_claim_ids}
        reasons: set[GateReasonCode] = set()
        if not edited:
            reasons.add(GateReasonCode.EDITOR_SELECTION_MISSING)
        if not edited <= actual:
            reasons.add(GateReasonCode.EDITOR_CLAIM_UNKNOWN)
        if not edited <= approved:
            reasons.add(GateReasonCode.EDITOR_UNAPPROVED_CLAIM)
        failed = bool(reasons)
        return GateDecision(
            reason_codes=tuple(reasons),
            affected_sections=(
                tuple(item.section for item in data.claims) if failed else ()
            ),
            rejected_claim_ids=(
                tuple(item.claim_id for item in data.claims) if failed else ()
            ),
            block_rating=failed,
            block_publication=failed,
        )


class ExhibitReconciliationGate:
    def evaluate(self, context: QualityGateInput) -> GateDecision:
        data = _data(context)
        sections = tuple(
            exhibit.section for exhibit in data.exhibits if not exhibit.reconciles
        )
        failed = bool(sections)
        return GateDecision(
            reason_codes=(GateReasonCode.EXHIBIT_MISMATCH,) if failed else (),
            affected_sections=sections,
            block_rating=failed,
            block_publication=failed,
        )


class _InferenceDisclosureGate:
    def __init__(self, policy: QualityGatePolicy) -> None:
        self._policy = policy

    def evaluate(self, context: QualityGateInput) -> GateDecision:
        disclosure = _data(context).inference_disclosure
        if disclosure is None:
            return GateDecision(
                reason_codes=(GateReasonCode.INFERENCE_DISCLOSURE_MISSING,),
                block_rating=True,
                block_publication=True,
            )
        weak = (
            disclosure.mode is InferenceMode.LOCAL_ONLY
            and disclosure.confidence < self._policy.local_min_confidence
        )
        return GateDecision(
            reason_codes=(GateReasonCode.LOCAL_CONFIDENCE_WEAK,) if weak else (),
            block_rating=weak,
            block_publication=weak,
        )


class _EventFreshnessGate:
    def __init__(self, policy: QualityGatePolicy) -> None:
        self._policy = policy

    def evaluate(self, context: QualityGateInput) -> GateDecision:
        data = _data(context)
        required = data.publication_kind is PublicationKind.EVENT_REPORT
        if required and (
            data.event_published_at is None or data.event_source_type is None
        ):
            return GateDecision(
                reason_codes=(GateReasonCode.EVENT_FRESHNESS_UNKNOWN,),
                block_event=True,
            )
        if data.event_published_at is None or data.event_source_type is None:
            return GateDecision()
        window = self._policy.event_window_days_by_source.get(
            data.event_source_type, self._policy.event_window_days
        )
        expired = data.as_of - data.event_published_at > timedelta(days=window)
        return GateDecision(
            reason_codes=(
                (GateReasonCode.EVENT_OUTSIDE_MATERIALITY_WINDOW,) if expired else ()
            ),
            block_event=expired,
        )


class QualityGate:
    """Fixed-order composition of independent deterministic gate decisions."""

    __slots__ = ("_gates",)

    def __init__(
        self,
        *,
        policy: QualityGatePolicy | None = None,
        calendars: Mapping[str, ExchangeCalendar] | None = None,
    ) -> None:
        rules = QualityGatePolicy.model_validate(policy or QualityGatePolicy())
        self._gates: tuple[ComponentGate, ...] = (
            EvidenceLineageGate(),
            CorroborationGate(),
            PortfolioFreshnessGate(rules),
            PriceFreshnessGate(rules, calendars),
            FilingFreshnessGate(),
            EligibilityGate(),
            ContradictionGate(),
            ReviewerGate(),
            EditorGate(),
            ExhibitReconciliationGate(),
            _InferenceDisclosureGate(rules),
            _EventFreshnessGate(rules),
        )

    def evaluate(self, context: QualityGateInput) -> QualityGateResult:
        data = _data(context)
        return _merge_decisions(
            data, tuple(gate.evaluate(data) for gate in self._gates)
        )


class RecommendationGate(QualityGate):
    """Recommendation-facing facade over the same composed engine."""


def price_is_fresh(
    price_date: date,
    as_of_date: date,
    *,
    exchange: str = "NASDAQ",
    calendar: ExchangeCalendar | None = None,
) -> bool:
    """Check a dated close against the latest prior session and four-day limit."""
    if (
        not isinstance(price_date, date)
        or isinstance(price_date, datetime)
        or not isinstance(as_of_date, date)
        or isinstance(as_of_date, datetime)
    ):
        raise TypeError("price_date and as_of_date must be dates")
    exchange_id = _identifier(exchange)
    try:
        schedules = _calendar_map(None if calendar is None else {exchange_id: calendar})
        schedule = schedules.get(_comparison_key(exchange_id))
    except (TypeError, ValueError):
        return False
    if (
        schedule is None
        or _comparison_key(schedule.exchange) != _comparison_key(exchange_id)
        or not schedule.covers(price_date)
        or not schedule.covers(as_of_date)
    ):
        return False
    try:
        if not schedule.is_session(price_date):
            return False
        age = (as_of_date - price_date).days
        if age < 0 or age > 4:
            return False
        candidate = as_of_date - timedelta(days=1)
        for _ in range(15):
            if not schedule.covers(candidate):
                return False
            if schedule.is_session(candidate):
                return candidate == price_date
            candidate -= timedelta(days=1)
    except ValueError:
        return False
    return False


def _merge_decisions(
    data: QualityGateInput, decisions: tuple[GateDecision, ...]
) -> QualityGateResult:
    """Merge component outputs with set operations, independent of gate order."""
    reasons = {reason for decision in decisions for reason in decision.reason_codes}
    blocked_sections = {
        _comparison_key(section)
        for decision in decisions
        for section in decision.affected_sections
    }
    rejected = {
        _comparison_key(claim_id)
        for decision in decisions
        for claim_id in decision.rejected_claim_ids
    }
    lookup = _claim_lookup(data)
    approved = {_comparison_key(value) for value in data.reviewer_approved_claim_ids}
    edited = {_comparison_key(value) for value in data.editor_claim_ids}
    safe_keys = {
        key
        for key, item in lookup.items()
        if key in approved
        and key in edited
        and key not in rejected
        and _comparison_key(item.section) not in blocked_sections
    }

    block_rating = any(decision.block_rating for decision in decisions)
    if data.requested_rating is not RecommendationRating.NO_RATING:
        supported = any(
            key in safe_keys
            and item.section == "recommendation"
            and (item.material or item.consequential)
            for key, item in lookup.items()
        )
        if not supported:
            reasons.add(GateReasonCode.RECOMMENDATION_EVIDENCE_MISSING)
            block_rating = True

    selected_sections = {
        _comparison_key(item.section) for key, item in lookup.items() if key in edited
    }
    selected_failures = blocked_sections & selected_sections
    global_failure = any(decision.block_publication for decision in decisions)
    event_failure = any(decision.block_event for decision in decisions)
    critical_failure = False
    if data.publication_kind is PublicationKind.RECOMMENDATION:
        critical_failure = "recommendation" in blocked_sections or block_rating
    elif data.publication_kind is PublicationKind.EVENT_REPORT:
        critical_failure = "event" in blocked_sections or event_failure
    elif data.publication_kind in {
        PublicationKind.FOUNDATIONAL_REPORT,
        PublicationKind.PORTFOLIO_BRIEF,
    }:
        critical_failure = bool(blocked_sections) or block_rating

    reviewer_passed = data.reviewer_verdict is ReviewVerdict.PASS
    if global_failure or critical_failure or not safe_keys or not reviewer_passed:
        publication = PublicationVerdict.DRAFT
    elif selected_failures:
        publication = PublicationVerdict.PARTIAL
    else:
        publication = PublicationVerdict.FINAL

    if publication is PublicationVerdict.DRAFT:
        block_rating = True

    allowed_claim_ids = tuple(
        sorted((lookup[key].claim_id for key in safe_keys), key=_comparison_key)
    )
    allowed_sections = tuple(
        sorted({lookup[key].section for key in safe_keys}, key=_comparison_key)
    )
    effective_rating = (
        RecommendationRating.NO_RATING if block_rating else data.requested_rating
    )
    allow_sizing = effective_rating is not RecommendationRating.NO_RATING and not any(
        decision.block_sizing for decision in decisions
    )
    if publication in {PublicationVerdict.FINAL, PublicationVerdict.PARTIAL}:
        review = ReviewVerdict.PASS
    elif critical_failure or global_failure or selected_failures:
        review = ReviewVerdict.BLOCK
    else:
        review = data.reviewer_verdict
    if data.publication_kind is PublicationKind.EVENT_REPORT:
        allow_event_report = (
            publication in {PublicationVerdict.FINAL, PublicationVerdict.PARTIAL}
            and not event_failure
        )
    else:
        allow_event_report = not event_failure

    reported_ages = tuple(
        decision.portfolio_age_hours
        for decision in decisions
        if decision.portfolio_age_hours is not None
    )
    portfolio_age = max(reported_ages) if reported_ages else None
    inference = data.inference_disclosure
    return QualityGateResult(
        reason_codes=tuple(sorted(reasons, key=lambda reason: reason.value)),
        allowed_sections=allowed_sections,
        allowed_claim_ids=allowed_claim_ids,
        allow_event_report=allow_event_report,
        allow_sizing=allow_sizing,
        publication_verdict=publication,
        review_verdict=review,
        effective_rating=effective_rating,
        portfolio_age_hours=portfolio_age,
        inference_mode=None if inference is None else inference.mode,
        inference_provider=None if inference is None else inference.provider,
        inference_model=None if inference is None else inference.model,
    )


def evaluate_quality_gates(
    gate_input: QualityGateInput,
    *,
    policy: QualityGatePolicy | None = None,
    calendars: Mapping[str, ExchangeCalendar] | None = None,
) -> QualityGateResult:
    """Evaluate the reusable component composition through its public facade."""
    return QualityGate(policy=policy, calendars=calendars).evaluate(gate_input)


__all__ = [
    "CalculatedExhibit",
    "CalculatedRow",
    "ClaimQualityInput",
    "CorroborationGate",
    "ContradictionGate",
    "DEFAULT_EXCHANGE_CALENDARS",
    "EditorGate",
    "EligibilityGate",
    "EvidenceLineageGate",
    "EvidenceReference",
    "ExchangeCalendar",
    "ExhibitReconciliationGate",
    "FilingAvailability",
    "FilingFreshnessGate",
    "GateDecision",
    "GateReasonCode",
    "InferenceDisclosure",
    "MarketPrice",
    "PortfolioFreshnessGate",
    "PriceFreshnessGate",
    "ReviewerGate",
    "PublicationKind",
    "PublicationVerdict",
    "QualityGate",
    "QualityGateInput",
    "QualityGatePolicy",
    "QualityGateResult",
    "RecommendationGate",
    "SourceReference",
    "evaluate_quality_gates",
    "price_is_fresh",
]
