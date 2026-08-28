"""Frozen, versioned contracts for bounded qualitative research roles."""

from __future__ import annotations

import unicodedata
from datetime import datetime
from decimal import Decimal
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..models import ClaimKind, InferenceMode, RecommendationRating, ReviewVerdict


SCENARIO_PROBABILITY_TOLERANCE = Decimal("0.000001")


class AgentError(RuntimeError):
    """Base class for redacted agent boundary failures."""


class AgentContractError(AgentError, ValueError):
    """Raised when an agent handoff violates its declared contract."""


class EvidenceUnavailable(AgentContractError):
    """Raised when cited evidence is missing or quarantined."""


class IneligibleSecurity(AgentContractError):
    """Raised before inference when a security cannot receive a rating."""


def _safe_identifier(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
        or any(character.isspace() for character in value)
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ValueError("identifier is invalid")
    return value


def _safe_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("text is invalid")
    normalized = unicodedata.normalize("NFC", value).strip()
    if not normalized or any(
        unicodedata.category(character) == "Cc" and character not in {"\n", "\t"}
        for character in normalized
    ):
        raise ValueError("text is invalid")
    return normalized


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value


def _sorted_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(_safe_identifier(item) for item in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError("identifiers must be unique")
    return tuple(sorted(normalized))


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceInput(FrozenContract):
    schema_version: Literal["1"] = "1"
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    claim_ids: tuple[str, ...] = ()
    as_of: datetime

    _validate_evidence = field_validator("evidence_ids")(_sorted_ids)
    _validate_claims = field_validator("claim_ids")(_sorted_ids)
    _validate_as_of = field_validator("as_of")(_aware)


InputT = TypeVar("InputT", bound=EvidenceInput)


class AgentTask(FrozenContract, Generic[InputT]):
    schema_version: Literal["1"] = "1"
    task_id: str
    run_id: str
    input: InputT

    _validate_task_id = field_validator("task_id")(_safe_identifier)
    _validate_run_id = field_validator("run_id")(_safe_identifier)


class AnalyticalOutput(FrozenContract):
    schema_version: Literal["1"] = "1"
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    as_of: datetime
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    inference_mode: InferenceMode

    _validate_evidence = field_validator("evidence_ids")(_sorted_ids)
    _validate_as_of = field_validator("as_of")(_aware)


class DirectorInput(EvidenceInput):
    portfolio_exposures: tuple[str, ...]
    report_schedule: str
    material_event_ids: tuple[str, ...] = ()
    thesis_gap_ids: tuple[str, ...] = ()

    _portfolio = field_validator("portfolio_exposures")(_sorted_ids)
    _events = field_validator("material_event_ids")(_sorted_ids)
    _gaps = field_validator("thesis_gap_ids")(_sorted_ids)
    _schedule = field_validator("report_schedule")(_safe_text)


class DirectedResearchTask(FrozenContract):
    task_id: str
    kind: Literal["event_research", "thesis_gap", "fundamental_review", "landscape"]
    question: str
    priority: int = Field(ge=0, le=100)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    _task_id = field_validator("task_id")(_safe_identifier)
    _question = field_validator("question")(_safe_text)
    _evidence = field_validator("evidence_ids")(_sorted_ids)


class DirectorOutput(AnalyticalOutput):
    tasks: tuple[DirectedResearchTask, ...]

    @field_validator("tasks")
    @classmethod
    def _deterministic_tasks(
        cls, values: tuple[DirectedResearchTask, ...]
    ) -> tuple[DirectedResearchTask, ...]:
        identifiers = tuple(item.task_id for item in values)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("directed task IDs must be unique")
        return tuple(sorted(values, key=lambda item: item.task_id))


class EventCandidate(FrozenContract):
    event_id: str
    headline: str

    _event_id = field_validator("event_id")(_safe_identifier)
    _headline = field_validator("headline")(_safe_text)


class EventScoutInput(EvidenceInput):
    events: tuple[EventCandidate, ...] = Field(min_length=1)


class RankedEvent(FrozenContract):
    event_id: str
    rank: int = Field(gt=0)
    reason: str

    _event_id = field_validator("event_id")(_safe_identifier)
    _reason = field_validator("reason")(_safe_text)


class EventScoutOutput(AnalyticalOutput):
    ranked_events: tuple[RankedEvent, ...]


class EmergingSignal(FrozenContract):
    signal_id: str
    company: str

    _signal_id = field_validator("signal_id")(_safe_identifier)
    _company = field_validator("company")(_safe_text)


class EmergingScoutInput(EvidenceInput):
    signals: tuple[EmergingSignal, ...] = Field(min_length=1)


class RankedSignal(FrozenContract):
    signal_id: str
    rank: int = Field(gt=0)
    value_chain_role: str
    reason: str

    _signal_id = field_validator("signal_id")(_safe_identifier)
    _role = field_validator("value_chain_role")(_safe_text)
    _reason = field_validator("reason")(_safe_text)


class EmergingScoutOutput(AnalyticalOutput):
    ranked_signals: tuple[RankedSignal, ...]


class EvidenceAnalystInput(EvidenceInput):
    question: str

    _question = field_validator("question")(_safe_text)


class ClaimDraft(FrozenContract):
    claim_id: str
    text: str
    kind: ClaimKind
    evidence_ids: tuple[str, ...] = ()
    supporting_claim_ids: tuple[str, ...] = ()

    _claim_id = field_validator("claim_id")(_safe_identifier)
    _text = field_validator("text")(_safe_text)
    _evidence = field_validator("evidence_ids")(_sorted_ids)
    _claims = field_validator("supporting_claim_ids")(_sorted_ids)

    @model_validator(mode="after")
    def _citation_policy(self) -> "ClaimDraft":
        if self.kind in {ClaimKind.FACT, ClaimKind.GUIDANCE, ClaimKind.ESTIMATE}:
            if not self.evidence_ids:
                raise ValueError("reported claims require passage evidence")
        elif not self.evidence_ids and not self.supporting_claim_ids:
            raise ValueError("inference requires passage evidence or an approved claim")
        if self.claim_id in self.supporting_claim_ids:
            raise ValueError("claim cannot support itself")
        return self


class EvidenceAnalystOutput(AnalyticalOutput):
    claims: tuple[ClaimDraft, ...]


class IndustryStrategistInput(EvidenceInput):
    industry: str
    horizon_years: int = Field(ge=5, le=10)

    _industry = field_validator("industry")(_safe_text)


class Scenario(FrozenContract):
    name: Literal["base", "upside", "downside"]
    probability: Decimal = Field(ge=Decimal("0"), le=Decimal("1"), allow_inf_nan=False)
    horizon_years: int = Field(ge=5, le=10)
    description: str
    signposts: tuple[str, ...] = Field(min_length=1)

    _description = field_validator("description")(_safe_text)

    @field_validator("name", mode="before")
    @classmethod
    def _normalized_name(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("signposts")
    @classmethod
    def _signposts(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_safe_text(item) for item in values)


class IndustryStrategistOutput(AnalyticalOutput):
    scenarios: tuple[Scenario, ...] = Field(min_length=1)

    @field_validator("scenarios")
    @classmethod
    def _complete_ordered_set(
        cls, values: tuple[Scenario, ...]
    ) -> tuple[Scenario, ...]:
        required = ("base", "upside", "downside")
        if len(values) != 3 or {item.name for item in values} != set(required):
            raise ValueError("scenarios require exactly base, upside, and downside")
        order = {name: index for index, name in enumerate(required)}
        return tuple(sorted(values, key=lambda item: order[item.name]))

    @model_validator(mode="after")
    def _probabilities_sum_to_one(self) -> "IndustryStrategistOutput":
        total = sum((item.probability for item in self.scenarios), Decimal("0"))
        if abs(total - Decimal("1")) > SCENARIO_PROBABILITY_TOLERANCE:
            raise ValueError("scenario probabilities must sum to one")
        return self


class SecurityEligibility(FrozenContract):
    entity_id: str
    security_id: str | None = None
    symbol: str
    asset_kind: Literal["equity", "private", "etf", "cash", "nontradable"]
    resolved: bool
    public: bool
    tradable: bool

    _entity_id = field_validator("entity_id")(_safe_identifier)
    _security_id = field_validator("security_id")(
        lambda value: None if value is None else _safe_identifier(value)
    )
    _symbol = field_validator("symbol")(_safe_identifier)

    @property
    def rating_eligible(self) -> bool:
        return (
            self.asset_kind == "equity"
            and self.resolved
            and self.public
            and self.tradable
            and self.security_id is not None
        )


class FundamentalAnalystInput(EvidenceInput):
    security: SecurityEligibility
    horizon_months: int = Field(ge=6, le=24)


class ValuationRange(FrozenContract):
    low: Decimal = Field(allow_inf_nan=False)
    high: Decimal = Field(allow_inf_nan=False)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    as_of: datetime

    _as_of = field_validator("as_of")(_aware)

    @model_validator(mode="after")
    def _ordered(self) -> "ValuationRange":
        if self.low > self.high:
            raise ValueError("valuation low must not exceed high")
        return self


class FundamentalAnalystOutput(AnalyticalOutput):
    security_id: str
    thesis: str
    horizon_months: int = Field(ge=6, le=24)
    valuation: ValuationRange
    assumptions: tuple[str, ...] = Field(min_length=1)
    catalysts: tuple[str, ...] = Field(min_length=1)
    counter_thesis: str
    risks: tuple[str, ...] = Field(min_length=1)
    invalidation_conditions: tuple[str, ...] = Field(min_length=1)
    rating: RecommendationRating
    eligible: Literal[True]

    _security_id = field_validator("security_id")(_safe_identifier)
    _thesis = field_validator("thesis")(_safe_text)
    _counter = field_validator("counter_thesis")(_safe_text)

    @field_validator("assumptions", "catalysts", "risks", "invalidation_conditions")
    @classmethod
    def _text_items(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_safe_text(item) for item in values)


class ReviewerInput(EvidenceInput):
    target_claim_ids: tuple[str, ...] = Field(min_length=1)

    _targets = field_validator("target_claim_ids")(_sorted_ids)

    @model_validator(mode="after")
    def _targets_are_loaded(self) -> "ReviewerInput":
        if not set(self.target_claim_ids) <= set(self.claim_ids):
            raise ValueError("review targets must be included in claim_ids")
        return self


class ReviewIssue(FrozenContract):
    code: str = Field(pattern=r"^[a-z0-9_.-]+$")
    message: str
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    target_claim_ids: tuple[str, ...] = Field(min_length=1)

    _message = field_validator("message")(_safe_text)
    _evidence = field_validator("evidence_ids")(_sorted_ids)
    _targets = field_validator("target_claim_ids")(_sorted_ids)


class ReviewerOutput(AnalyticalOutput):
    verdict: ReviewVerdict
    issues: tuple[ReviewIssue, ...]

    @model_validator(mode="after")
    def _non_pass_needs_issues(self) -> "ReviewerOutput":
        if self.verdict is not ReviewVerdict.PASS and not self.issues:
            raise ValueError("revise and block verdicts require issues")
        return self


class EditorInput(EvidenceInput):
    approved_claim_ids: tuple[str, ...] = Field(min_length=1)

    _approved = field_validator("approved_claim_ids")(_sorted_ids)


class ReportSection(FrozenContract):
    heading: str
    approved_claim_ids: tuple[str, ...] = Field(min_length=1)

    _heading = field_validator("heading")(_safe_text)
    _claims = field_validator("approved_claim_ids")(_sorted_ids)


class ResearchEditorOutput(AnalyticalOutput):
    title: str
    sections: tuple[ReportSection, ...] = Field(min_length=1)

    _title = field_validator("title")(_safe_text)
