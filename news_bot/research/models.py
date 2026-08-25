"""Immutable domain records shared by the research pipeline."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


class InferenceMode(str, Enum):
    """Available inference-provider modes."""

    EXTERNAL = "external"
    LOCAL_ONLY = "local_only"


class AgentRole(str, Enum):
    """Specialized roles in the research workflow."""

    RESEARCH_DIRECTOR = "research_director"
    PORTFOLIO_MAPPER = "portfolio_mapper"
    EVENT_SCOUT = "event_scout"
    EMERGING_COMPANY_SCOUT = "emerging_company_scout"
    EVIDENCE_ANALYST = "evidence_analyst"
    INDUSTRY_STRATEGIST = "industry_strategist"
    FUNDAMENTAL_ANALYST = "fundamental_analyst"
    SKEPTICAL_REVIEWER = "skeptical_reviewer"
    RESEARCH_EDITOR = "research_editor"


class ClaimKind(str, Enum):
    """The epistemic category of an evidence claim."""

    FACT = "fact"
    GUIDANCE = "guidance"
    ESTIMATE = "estimate"
    INFERENCE = "inference"


class RecommendationRating(str, Enum):
    """Allowed investment-research ratings."""

    BUY = "buy"
    HOLD = "hold"
    SELL_REDUCE = "sell_reduce"
    NO_RATING = "no_rating"


class ReviewVerdict(str, Enum):
    """Possible outcomes from skeptical review."""

    PASS = "pass"
    REVISE = "revise"
    BLOCK = "block"


def _require_timezone(record_name: str, field_name: str, value: datetime) -> None:
    """Reject datetimes whose UTC offset cannot be determined."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{record_name}.{field_name} must be timezone-aware")


def _require_decimal(
    record_name: str, field_name: str, value: Decimal | None
) -> None:
    """Prevent implicit float or string values at exact-number boundaries."""
    if value is not None and not isinstance(value, Decimal):
        raise TypeError(f"{record_name}.{field_name} must be Decimal")


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Point-in-time portfolio value summary."""

    snapshot_id: str
    as_of: datetime
    base_currency: str
    nav: Decimal
    cash: Decimal
    is_stale: bool

    def __post_init__(self) -> None:
        _require_timezone(type(self).__name__, "as_of", self.as_of)
        _require_decimal(type(self).__name__, "nav", self.nav)
        _require_decimal(type(self).__name__, "cash", self.cash)


@dataclass(frozen=True)
class Position:
    """A normalized security position within a portfolio snapshot."""

    position_id: str
    snapshot_id: str
    symbol: str
    quantity: Decimal
    market_value: Decimal
    currency: str
    cost_basis: Decimal | None = None

    def __post_init__(self) -> None:
        _require_decimal(type(self).__name__, "quantity", self.quantity)
        _require_decimal(type(self).__name__, "market_value", self.market_value)
        _require_decimal(type(self).__name__, "cost_basis", self.cost_basis)


@dataclass(frozen=True)
class SourceDocument:
    """Metadata and lineage for an ingested source document."""

    document_id: str
    source_type: str
    canonical_url: str
    publisher: str
    published_at: datetime
    retrieved_at: datetime
    content_hash: str
    raw_content_path: str | None
    extraction_status: str

    def __post_init__(self) -> None:
        _require_timezone(type(self).__name__, "published_at", self.published_at)
        _require_timezone(type(self).__name__, "retrieved_at", self.retrieved_at)


@dataclass(frozen=True)
class EvidenceClaim:
    """An atomic research claim classified by evidence kind."""

    claim_id: str
    entity_id: str | None
    kind: ClaimKind
    text: str
    as_of: datetime
    confidence: Decimal
    status: str

    def __post_init__(self) -> None:
        _require_timezone(type(self).__name__, "as_of", self.as_of)
        _require_decimal(type(self).__name__, "confidence", self.confidence)


@dataclass(frozen=True)
class ModelUsage:
    """Provider-reported token usage and its computed monetary cost."""

    usage_id: str
    run_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    recorded_at: datetime

    def __post_init__(self) -> None:
        _require_timezone(type(self).__name__, "recorded_at", self.recorded_at)
        _require_decimal(type(self).__name__, "cost_usd", self.cost_usd)


@dataclass(frozen=True)
class AgentRunResult:
    """Auditable result metadata for one specialist-agent execution."""

    run_id: str
    role: AgentRole
    started_at: datetime
    completed_at: datetime
    status: str
    output: str
    usage: ModelUsage | None = None

    def __post_init__(self) -> None:
        _require_timezone(type(self).__name__, "started_at", self.started_at)
        _require_timezone(type(self).__name__, "completed_at", self.completed_at)
