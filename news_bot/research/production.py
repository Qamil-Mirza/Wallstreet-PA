"""Composable production adapters for the durable research orchestrator.

This module is deliberately a composition root.  It does not invent research
content: every stage either delegates to an existing typed component using
caller-supplied source inputs, or fails closed with ``RequiredStageUnavailable``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Protocol
from urllib.parse import urlparse

from pydantic import BaseModel

from .agents import (
    EmergingCompanyScout,
    EventScout,
    EvidenceAnalyst,
    FundamentalAnalyst,
    IndustryStrategist,
    ResearchDirector,
    ResearchEditor,
    SkepticalReviewer,
)
from .agents.base import BoundedAgent
from .agents.contracts import (
    AgentTask,
    AnalyticalOutput,
    EditorInput,
    EventCandidate,
    EventScoutInput,
    EventScoutOutput,
    EvidenceAnalystInput,
    EvidenceAnalystOutput,
    FundamentalAnalystInput,
    ReviewerInput,
    ReviewerOutput,
    SecurityEligibility,
)
from .budget import BudgetLedger, ModelPrice, PriceTable
from .config import ResearchConfig
from .connectors import (
    ConnectorBatch,
    ConnectorCheckpoint,
    ConnectorError,
    EmergingSignal,
    MarketAuxResearchConnector,
    ResearchConnector,
    SignalConnectorBatch,
    SignalResearchConnector,
)
from .connectors.sec import SECConfig, SECConnector
from .entities import (
    ETFHoldings,
    EntityResolver,
    ExposurePosition,
    PortfolioExposure,
    PortfolioExposureMapper,
    Relationship,
    ResolvedEntity,
    SecurityIdentity,
)
from .evidence import EvidenceIngestor, canonicalize_content
from .models import (
    AgentRole,
    EvidenceClaim,
    InferenceMode,
    RecommendationRating,
    ReviewVerdict,
)
from .orchestrator import (
    RequiredStageUnavailable,
    StageContext,
    StageExecutionControl,
    StageOutcome,
)
from .providers import (
    ModelProvider,
    ModelRoute,
    OllamaProvider,
    OpenAIProvider,
    ProviderRouter,
    ReasoningEffort,
)
from .quality import (
    ClaimQualityInput,
    EvidenceReference,
    InferenceDisclosure,
    PublicationKind,
    QualityGate,
    QualityGateInput,
    QualityGatePolicy,
)
from .reports import (
    Citation,
    EmergingCompanyMonitor,
    EventUpdate,
    IndustryLandscape,
    PortfolioBrief,
    PublicationPrivacyContext,
    RenderedReportArtifact,
    ReportMetadata,
    ReportRenderer,
    ReportSection,
)
from .store import ResearchStore, _canonical_json, _decimal_text, _utc_text


PRODUCTION_STAGE_NAMES = (
    "ingestion",
    "resolve",
    "materiality",
    "event_analysis",
    "event_update",
    "changed_theses",
    "selected_recommendations",
    "public_signals",
    "emerging_map",
    "industry_refresh",
    "historical_documents",
    "evidence",
    "analysis",
    "review",
    "revision",
    "publish",
)

_AGENT_ROLES = MappingProxyType(
    {
        "materiality": AgentRole.EVIDENCE_ANALYST,
        "event_analysis": AgentRole.FUNDAMENTAL_ANALYST,
        "event_update": AgentRole.EVENT_SCOUT,
        "changed_theses": AgentRole.RESEARCH_DIRECTOR,
        "selected_recommendations": AgentRole.FUNDAMENTAL_ANALYST,
        "emerging_map": AgentRole.INDUSTRY_STRATEGIST,
        "industry_refresh": AgentRole.INDUSTRY_STRATEGIST,
        "evidence": AgentRole.EVIDENCE_ANALYST,
        "analysis": AgentRole.FUNDAMENTAL_ANALYST,
        "review": AgentRole.SKEPTICAL_REVIEWER,
        "revision": AgentRole.RESEARCH_EDITOR,
    }
)
_REPORT_STAGE_TYPES = MappingProxyType(
    {
        "analysis": PortfolioBrief,
        "event_update": EventUpdate,
        "selected_recommendations": PortfolioBrief,
        "emerging_map": EmergingCompanyMonitor,
        "industry_refresh": IndustryLandscape,
    }
)
_REPORT_TYPES = {
    "event_update": EventUpdate,
    "portfolio_brief": PortfolioBrief,
    "industry_landscape": IndustryLandscape,
    "emerging_monitor": EmergingCompanyMonitor,
}


class ConnectorCheckpointStore(Protocol):
    """Durable cursor boundary owned by the embedding application."""

    def load(self, connector: str) -> ConnectorCheckpoint: ...

    def save(self, checkpoint: ConnectorCheckpoint) -> None: ...


class _FileConnectorCheckpointStore:
    """Atomic, restart-safe connector cursors in the research data volume."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, ConnectorCheckpoint]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError
            return {
                connector: ConnectorCheckpoint(
                    connector,
                    cursor=value.get("cursor"),
                    etag=value.get("etag"),
                    last_modified=value.get("last_modified"),
                )
                for connector, value in raw.items()
                if isinstance(connector, str)
                and isinstance(value, dict)
                and set(value) == {"cursor", "etag", "last_modified"}
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("connector checkpoint store is invalid") from exc

    def load(self, connector: str) -> ConnectorCheckpoint:
        with self._lock:
            return self._read().get(connector, ConnectorCheckpoint(connector))

    def save(self, checkpoint: ConnectorCheckpoint) -> None:
        if not isinstance(checkpoint, ConnectorCheckpoint):
            raise TypeError("checkpoint must be ConnectorCheckpoint")
        with self._lock:
            checkpoints = self._read()
            checkpoints[checkpoint.connector] = checkpoint
            payload = {
                name: {
                    "cursor": value.cursor,
                    "etag": value.etag,
                    "last_modified": value.last_modified,
                }
                for name, value in sorted(checkpoints.items())
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                dir=self.path.parent, prefix=f".{self.path.name}.", text=True
            )
            temporary_path = Path(temporary)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(
                        payload,
                        handle,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary_path.replace(self.path)
            finally:
                temporary_path.unlink(missing_ok=True)


def _environment_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _configured_news_sources(
    config: ResearchConfig,
) -> tuple[tuple[ResearchConnector, ...], ConnectorCheckpointStore | None]:
    """Compose the existing news adapter only when its key is configured.

    Docker Compose injects ``.env`` into the container environment.  Reading
    only that environment here keeps library/test construction independent of
    a repository-local dotenv file.
    """

    api_key = os.environ.get("NEWS_API_KEY", "").strip()
    if not api_key:
        return (), None

    from news_bot.config import Config as NewsletterConfig
    from news_bot.config import RSSFeedEntry

    rss_feeds = []
    for raw_url in os.environ.get("RSS_FEEDS", "").split(","):
        url = raw_url.strip()
        if not url:
            continue
        domain = urlparse(url).netloc.removeprefix("www.")
        name = domain.split(".", 1)[0].title() or "RSS Feed"
        rss_feeds.append(RSSFeedEntry(name=name, url=url))

    news_config = NewsletterConfig(
        news_api_key=api_key,
        news_api_base_url=os.environ.get(
            "NEWS_API_BASE_URL", "https://api.marketaux.com/v1"
        ),
        smtp_host="",
        smtp_port=587,
        smtp_user="",
        smtp_password="",
        recipient_email="",
        ollama_base_url=config.ollama_base_url,
        ollama_model=config.ollama_model,
        tts_enabled=False,
        section_world_enabled=_environment_flag("SECTION_WORLD_ENABLED", True),
        section_us_tech_enabled=_environment_flag(
            "SECTION_US_TECH_ENABLED", True
        ),
        section_us_industry_enabled=_environment_flag(
            "SECTION_US_INDUSTRY_ENABLED", True
        ),
        section_malaysia_tech_enabled=_environment_flag(
            "SECTION_MALAYSIA_TECH_ENABLED", True
        ),
        section_malaysia_industry_enabled=_environment_flag(
            "SECTION_MALAYSIA_INDUSTRY_ENABLED", True
        ),
        rss_enabled=_environment_flag("RSS_ENABLED", False),
        rss_feeds=rss_feeds,
    )
    return (
        (MarketAuxResearchConnector(news_config),),
        _FileConnectorCheckpointStore(
            config.data_dir / "source_checkpoints.json"
        ),
    )


class PublicationSink(Protocol):
    """Research-only delivery boundary; deliberately exposes no trade method."""

    def publish(
        self,
        report: EventUpdate | PortfolioBrief | IndustryLandscape | EmergingCompanyMonitor,
        artifact: RenderedReportArtifact,
        *,
        idempotency_key: str,
    ) -> str: ...


@dataclass(frozen=True)
class ExposurePlan:
    """Exact local inputs for deterministic portfolio exposure calculation."""

    positions: tuple[ExposurePosition, ...]
    nav: Decimal
    base_currency: str
    identities: tuple[SecurityIdentity, ...] = ()
    etf_holdings: tuple[ETFHoldings, ...] = ()
    fx_to_base: Mapping[str, Decimal] = field(default_factory=dict)
    relationship_exposures: Mapping[str, Sequence[Relationship]] = field(
        default_factory=dict
    )


class ProductionInputs:
    """Source-specific typed input hooks used by production stage adapters.

    The base implementation supplies nothing.  This is intentional: omitting a
    source hook must stop the stage instead of manufacturing a placeholder.
    """

    def agent_task(
        self, stage: str, context: StageContext, role: AgentRole
    ) -> AgentTask:
        raise NotImplementedError

    def is_material_event(
        self, context: StageContext, output: AnalyticalOutput
    ) -> bool:
        raise NotImplementedError

    def exposure_plan(self, context: StageContext) -> ExposurePlan:
        raise NotImplementedError

    def persist_resolution(
        self,
        context: StageContext,
        resolved: tuple[ResolvedEntity, ...],
        exposures: Mapping[str, PortfolioExposure],
    ) -> None:
        raise NotImplementedError

    def persist_signals(
        self, context: StageContext, signals: tuple[EmergingSignal, ...]
    ) -> None:
        raise NotImplementedError

    def quality_input(
        self, context: StageContext, review: ReviewerOutput
    ) -> QualityGateInput:
        raise NotImplementedError

    def report(
        self, context: StageContext, output: AnalyticalOutput
    ) -> EventUpdate | PortfolioBrief | IndustryLandscape | EmergingCompanyMonitor:
        raise NotImplementedError


@dataclass(frozen=True)
class ProductionComponents:
    """Existing production components consumed by the adapter factory."""

    config: ResearchConfig
    store: ResearchStore
    budget: BudgetLedger
    router: ProviderRouter
    evidence: EvidenceIngestor
    entities: EntityResolver
    exposures: PortfolioExposureMapper
    quality: QualityGate
    renderer: ReportRenderer | None
    inputs: ProductionInputs | object
    clock: Callable[[], datetime]
    document_connectors: tuple[ResearchConnector, ...] = ()
    signal_connectors: tuple[SignalResearchConnector, ...] = ()
    checkpoints: ConnectorCheckpointStore | None = None
    publication_sink: PublicationSink | None = None
    sec_config: SECConfig | None = None

    def __post_init__(self) -> None:
        expected = (
            (self.config, ResearchConfig, "config"),
            (self.store, ResearchStore, "store"),
            (self.budget, BudgetLedger, "budget"),
            (self.router, ProviderRouter, "router"),
            (self.evidence, EvidenceIngestor, "evidence"),
            (self.entities, EntityResolver, "entities"),
            (self.exposures, PortfolioExposureMapper, "exposures"),
            (self.quality, QualityGate, "quality"),
        )
        for value, kind, name in expected:
            if not isinstance(value, kind):
                raise TypeError(f"{name} must be {kind.__name__}")
        if self.budget.store.database_path != self.store.database_path:
            raise ValueError("budget and production store must share one database")
        if self.evidence.store.database_path != self.store.database_path:
            raise ValueError("evidence and production store must share one database")
        if self.renderer is not None and not isinstance(self.renderer, ReportRenderer):
            raise TypeError("renderer must be ReportRenderer or None")
        if not callable(self.clock):
            raise TypeError("clock must be callable")
        sec_config = self.sec_config or SECConfig(self.config.sec_user_agent)
        if sec_config.user_agent != self.config.sec_user_agent:
            raise ValueError("SEC user agent must come from research configuration")
        object.__setattr__(self, "sec_config", sec_config)
        if any(not isinstance(item, ResearchConnector) for item in self.document_connectors):
            raise TypeError("document_connectors must implement ResearchConnector")
        if any(
            isinstance(item, SECConnector)
            and item.config.user_agent != sec_config.user_agent
            for item in self.document_connectors
        ):
            raise ValueError("SEC connectors must use configured identifying user agent")
        if any(
            not isinstance(item, SignalResearchConnector)
            for item in self.signal_connectors
        ):
            raise TypeError("signal_connectors must implement SignalResearchConnector")


class _RuntimeProductionInputs(ProductionInputs):
    """Concrete daily-research inputs derived only from durable local state."""

    def __init__(self, store: ResearchStore) -> None:
        self.store = store

    def _documents(self, context: StageContext):
        documents = self.store.list_documents_as_of(context.as_of)
        if not documents:
            _unavailable("required production evidence is unavailable")
        return documents

    def _passage_ids(self, context: StageContext) -> tuple[str, ...]:
        identifiers = tuple(
            passage.passage_id
            for document in self._documents(context)
            for passage in self.store.list_document_passages(document.document_id)
        )
        if not identifiers:
            _unavailable("required production passages are unavailable")
        return tuple(sorted(set(identifiers)))

    def _claims(self, context: StageContext) -> tuple[EvidenceClaim, ...]:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT execution.output_json FROM agent_executions AS execution "
                "JOIN workflow_tasks AS task ON task.task_id = execution.task_id "
                "WHERE execution.workflow_run_id = ? AND task.stage = 'materiality' "
                "AND execution.state = 'succeeded'",
                (context.workflow_id,),
            ).fetchone()
        if row is None or row[0] is None:
            _unavailable("required production claims are unavailable")
        try:
            output = EvidenceAnalystOutput.model_validate_json(row[0])
        except Exception:
            _unavailable("required production claims are invalid")
        claims = tuple(
            claim
            for draft in output.claims
            if (claim := self.store.get_claim(draft.claim_id)) is not None
        )
        if len(claims) != len(output.claims) or not claims:
            _unavailable("required production claim lineage is unavailable")
        return tuple(sorted(claims, key=lambda claim: claim.claim_id))

    def _snapshot(self, context: StageContext):
        snapshot_id = _workflow_portfolio_ref(self.store, context.workflow_id)
        if snapshot_id is None:
            _unavailable("required production portfolio is unavailable")
        with self.store.connect() as connection:
            snapshot = connection.execute(
                "SELECT as_of, base_currency, nav FROM portfolio_snapshots "
                "WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
            positions = connection.execute(
                "SELECT position.symbol, position.market_value, position.currency, "
                "security.security_type, position.security_id "
                "FROM positions AS position LEFT JOIN securities AS security "
                "ON security.security_id = position.security_id "
                "WHERE position.snapshot_id = ? ORDER BY position.position_id",
                (snapshot_id,),
            ).fetchall()
        if snapshot is None or not positions:
            _unavailable("required production portfolio positions are unavailable")
        return snapshot_id, snapshot, tuple(positions)

    def _security(self, context: StageContext) -> SecurityEligibility:
        _, _, positions = self._snapshot(context)
        eligible = tuple(
            position
            for position in positions
            if position[4] is not None
            and str(position[3] or "").upper() in {"STK", "STOCK", "EQUITY"}
        )
        if not eligible:
            _unavailable("required production equity security is unavailable")
        symbol, _, currency, _, security_id = max(
            eligible, key=lambda item: abs(Decimal(item[1]))
        )
        entity_digest = hashlib.sha256(
            f"ibkr-security:{security_id}".encode("utf-8")
        ).hexdigest()
        return SecurityEligibility(
            entity_id=f"entity_{entity_digest}",
            security_id=security_id,
            symbol=symbol,
            currency=currency,
            asset_kind="equity",
            resolved=True,
            public=True,
            tradable=True,
        )

    def exposure_plan(self, context: StageContext) -> ExposurePlan:
        _, snapshot, rows = self._snapshot(context)
        positions = tuple(
            ExposurePosition(
                symbol=row[0],
                market_value=Decimal(row[1]),
                currency=row[2],
                asset_class=row[3] or "UNKNOWN",
            )
            for row in rows
        )
        return ExposurePlan(
            positions=positions,
            nav=Decimal(snapshot[2]),
            base_currency=snapshot[1],
            fx_to_base={
                currency: Decimal("1")
                for currency in {row[2] for row in rows}
                if currency == snapshot[1]
            },
        )

    def persist_resolution(
        self,
        context: StageContext,
        resolved: tuple[ResolvedEntity, ...],
        exposures: Mapping[str, PortfolioExposure],
    ) -> None:
        del context, resolved
        if not exposures:
            _unavailable("required production exposure result is unavailable")

    def _events(self, context: StageContext) -> tuple[EventCandidate, ...]:
        events = []
        for document in self._documents(context):
            passages = self.store.list_document_passages(document.document_id)
            if not passages:
                continue
            headline = " ".join(passages[0].text.split())[:240]
            events.append(
                EventCandidate(event_id=document.document_id, headline=headline)
            )
        if not events:
            _unavailable("required production events are unavailable")
        return tuple(events)

    def agent_task(
        self, stage: str, context: StageContext, role: AgentRole
    ) -> AgentTask:
        evidence_ids = self._passage_ids(context)
        claim_ids = (
            ()
            if stage == "materiality"
            else tuple(claim.claim_id for claim in self._claims(context))
        )
        if role is AgentRole.EVIDENCE_ANALYST:
            task_input = EvidenceAnalystInput(
                evidence_ids=evidence_ids,
                claim_ids=claim_ids,
                as_of=context.as_of,
                question=(
                    "Identify portfolio-relevant developments, their causal "
                    "implications, and the strongest counter-evidence."
                ),
            )
        elif role is AgentRole.FUNDAMENTAL_ANALYST:
            task_input = FundamentalAnalystInput(
                evidence_ids=evidence_ids,
                claim_ids=claim_ids,
                as_of=context.as_of,
                security=self._security(context),
                horizon_months=12,
            )
        elif role is AgentRole.EVENT_SCOUT:
            task_input = EventScoutInput(
                evidence_ids=evidence_ids,
                claim_ids=claim_ids,
                as_of=context.as_of,
                events=self._events(context),
            )
        elif role is AgentRole.SKEPTICAL_REVIEWER:
            task_input = ReviewerInput(
                evidence_ids=evidence_ids,
                claim_ids=claim_ids,
                target_claim_ids=claim_ids,
                as_of=context.as_of,
            )
        elif role is AgentRole.RESEARCH_EDITOR:
            task_input = EditorInput(
                evidence_ids=evidence_ids,
                claim_ids=claim_ids,
                approved_claim_ids=claim_ids,
                as_of=context.as_of,
            )
        else:
            _unavailable("required production agent role is unavailable")
        return AgentTask[type(task_input)](
            task_id=context.task_id,
            run_id=context.workflow_id,
            input=task_input,
        )

    def is_material_event(
        self, context: StageContext, output: AnalyticalOutput
    ) -> bool:
        if not isinstance(output, EvidenceAnalystOutput):
            _unavailable("required production materiality output is invalid")
        for draft in output.claims:
            claim = EvidenceClaim(
                claim_id=draft.claim_id,
                entity_id=None,
                kind=draft.kind,
                text=draft.text,
                as_of=output.as_of,
                confidence=Decimal(str(output.confidence)),
                status="active",
            )
            existing = self.store.get_claim(claim.claim_id)
            if existing is not None:
                if existing != claim:
                    _unavailable("required production claim conflicts with stored evidence")
                continue
            self.store.insert_claim_with_lineage(
                claim,
                passage_links=tuple(
                    (evidence_id, "supports") for evidence_id in draft.evidence_ids
                ),
                supporting_claim_ids=draft.supporting_claim_ids,
            )
        return bool(output.claims)

    def _citations(
        self, context: StageContext, claims: tuple[EvidenceClaim, ...]
    ) -> tuple[Citation, ...]:
        documents = {
            document.document_id: document for document in self._documents(context)
        }
        citations: dict[str, Citation] = {}
        for claim in claims:
            for lineage in self.store.list_claim_lineage(claim.claim_id):
                document = documents.get(lineage.passage.document_id)
                if document is None:
                    _unavailable("required production citation source is unavailable")
                citations[lineage.passage.passage_id] = Citation(
                    evidence_id=lineage.passage.passage_id,
                    source=document.publisher,
                    url=document.canonical_url,
                    source_date=document.published_at.date(),
                    data_date=document.published_at.date(),
                    content_hash=document.content_hash,
                )
        if not citations:
            _unavailable("required production citations are unavailable")
        return tuple(sorted(citations.values(), key=lambda item: item.evidence_id))

    def report(
        self, context: StageContext, output: AnalyticalOutput
    ) -> EventUpdate:
        if not isinstance(output, EventScoutOutput):
            _unavailable("required production event report output is invalid")
        claims = self._claims(context)
        citations = self._citations(context, claims)
        evidence_ids = tuple(item.evidence_id for item in citations)
        body = " ".join(claim.text for claim in claims)
        security = self._security(context)
        with self.store.connect() as connection:
            audit = connection.execute(
                "SELECT provider, model, inference_mode FROM agent_executions "
                "WHERE workflow_run_id = ? AND task_id = ? AND state = 'succeeded'",
                (context.workflow_id, context.task_id),
            ).fetchone()
            portfolio_outcome = connection.execute(
                "SELECT outcome_json FROM workflow_tasks WHERE workflow_id = ? "
                "AND stage = 'portfolio' AND state = 'completed'",
                (context.workflow_id,),
            ).fetchone()
        if audit is None or any(value is None for value in audit):
            _unavailable("required production inference provenance is unavailable")
        omissions = ()
        if portfolio_outcome is not None and portfolio_outcome[0] is not None:
            omissions = StageOutcome.model_validate_json(
                portfolio_outcome[0]
            ).omissions
        report_digest = hashlib.sha256(
            f"{context.workflow_id}:event-update".encode("utf-8")
        ).hexdigest()

        def section(title: str) -> ReportSection:
            return ReportSection(title=title, body=body, evidence_ids=evidence_ids)

        return EventUpdate(
            metadata=ReportMetadata(
                report_id=f"event-{report_digest[:24]}",
                report_type="event_update",
                title=f"Portfolio event update for {security.symbol}",
                as_of=context.as_of,
                inference_mode=InferenceMode(audit[2]),
                provider=audit[0],
                model=audit[1],
                freshness="Sources fall within the configured event window.",
                citations=citations,
                methodology=(
                    "Stored source passages were analyzed by bounded specialist "
                    "agents and checked against durable claim lineage."
                ),
                omissions=omissions,
                disclosure=(
                    "Research output only; no order execution is available and "
                    "portfolio sizing is suppressed when inputs are stale."
                ),
            ),
            thesis=section("Thesis"),
            event_decomposition=(section("Event decomposition"),),
            causal_decomposition=(section("Causal chain"),),
            read_through=(section("Portfolio read-through"),),
            thesis_changes=(section("Thesis changes"),),
            unchanged_assumptions=(section("Unchanged assumptions"),),
            questions=(section("Open questions"),),
            signposts=(section("Signposts"),),
        )

    def quality_input(
        self, context: StageContext, review: ReviewerOutput
    ) -> QualityGateInput:
        claims = self._claims(context)
        documents = {
            document.document_id: document for document in self._documents(context)
        }
        quality_claims = []
        for claim in claims:
            references = []
            for lineage in self.store.list_claim_lineage(claim.claim_id):
                document = documents[lineage.passage.document_id]
                references.append(
                    EvidenceReference(
                        evidence_id=lineage.passage.passage_id,
                        canonical_source_id=document.document_id,
                        source_family=document.document_id,
                        source_type=document.source_type,
                        stored=True,
                        primary=False,
                        published_at=document.published_at,
                        retrieved_at=document.retrieved_at,
                    )
                )
            quality_claims.append(
                ClaimQualityInput(
                    claim_id=claim.claim_id,
                    section="event",
                    material=True,
                    consequential=False,
                    evidence=tuple(references),
                )
            )
        _, snapshot, _ = self._snapshot(context)
        latest_document = max(
            documents.values(), key=lambda document: document.published_at
        )
        with self.store.connect() as connection:
            audit = connection.execute(
                "SELECT provider, model, inference_mode FROM agent_executions "
                "WHERE workflow_run_id = ? AND task_id = ? AND state = 'succeeded'",
                (context.workflow_id, context.task_id),
            ).fetchone()
        if audit is None or any(value is None for value in audit):
            _unavailable("required production review provenance is unavailable")
        approved = (
            tuple(claim.claim_id for claim in claims)
            if review.verdict is ReviewVerdict.PASS
            else ()
        )
        return QualityGateInput(
            as_of=context.as_of,
            publication_kind=PublicationKind.EVENT_REPORT,
            requested_rating=RecommendationRating.NO_RATING,
            portfolio_snapshot_at=datetime.fromisoformat(
                snapshot[0].replace("Z", "+00:00")
            ),
            price=None,
            filing=None,
            security=self._security(context),
            claims=tuple(quality_claims),
            reviewer_verdict=review.verdict,
            reviewer_approved_claim_ids=approved,
            editor_claim_ids=approved,
            inference_disclosure=InferenceDisclosure(
                mode=InferenceMode(audit[2]),
                provider=audit[0],
                model=audit[1],
                confidence=Decimal(str(review.confidence)),
            ),
            event_source_type=latest_document.source_type,
            event_published_at=latest_document.published_at,
        )


def _unavailable(message: str = "required production input is unavailable") -> None:
    raise RequiredStageUnavailable(message)


def _call_input(method: object, *args):
    if not callable(method):
        _unavailable()
    try:
        return method(*args)
    except NotImplementedError:
        _unavailable()


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _agent_attempt_id(task: AgentTask, role: AgentRole) -> str:
    identity = _canonical_json(
        {
            "role": role.value,
            "schema_version": task.schema_version,
            "task_id": task.task_id,
            "workflow_run_id": task.run_id,
        }
    )
    return "agent_attempt_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


class _DocumentIngestionAdapter:
    def __init__(self, components: ProductionComponents) -> None:
        self.components = components
        self._preview_lock = threading.Lock()
        self._previewed: dict[
            tuple[datetime, int | None], tuple[ConnectorBatch, ...]
        ] = {}

    def _fetch_batches(
        self, as_of: datetime, max_documents: int | None
    ) -> tuple[ConnectorBatch, ...]:
        connectors = self.components.document_connectors
        checkpoints = self.components.checkpoints
        if not connectors or checkpoints is None:
            _unavailable()
        source_cutoff = as_of
        batches: list[ConnectorBatch] = []
        count = 0
        for connector in connectors:
            try:
                batch = connector.fetch(checkpoints.load(connector.name))
            except ConnectorError:
                _unavailable("required production connector is unavailable")
            if not isinstance(batch, ConnectorBatch) or batch.connector != connector.name:
                _unavailable("required production connector returned an invalid batch")
            for document in batch.documents:
                age = source_cutoff - document.published_at.astimezone(timezone.utc)
                if age.total_seconds() < 0 or age.total_seconds() > (
                    self.components.config.source_max_staleness_hours * 3600
                ):
                    _unavailable(
                        "required production source is outside the freshness window"
                    )
                if (
                    document.evidence.retrieved_at.astimezone(timezone.utc)
                    > source_cutoff
                ):
                    _unavailable(
                        "required production source was retrieved after the cutoff"
                    )
            count += len(batch.documents)
            if max_documents is not None and count > max_documents:
                _unavailable("required production document bound was exceeded")
            batches.append(batch)
        return tuple(batches)

    def preview_source_hashes(
        self, as_of: datetime, max_documents: int | None = None
    ) -> tuple[str, ...]:
        """Fetch and bind current source content into workflow identity.

        Checkpoints are intentionally advanced only after the corresponding
        documents are durably ingested by ``run``.
        """
        if not self.components.document_connectors or self.components.checkpoints is None:
            return ()
        key = (as_of, max_documents)
        with self._preview_lock:
            batches = self._previewed.get(key)
            if batches is None:
                batches = self._fetch_batches(as_of, max_documents)
                self._previewed[key] = batches
        return tuple(
            sorted(
                {
                    hashlib.sha256(
                        canonicalize_content(document.evidence.content).encode("utf-8")
                    ).hexdigest()
                    for batch in batches
                    for document in batch.documents
                }
            )
        )

    def run(
        self, context: StageContext, control: StageExecutionControl
    ) -> StageOutcome:
        checkpoints = self.components.checkpoints
        if checkpoints is None:
            _unavailable()
        key = (context.as_of, context.max_documents)
        with self._preview_lock:
            batches = self._previewed.pop(key, None)
        if batches is None:
            batches = self._fetch_batches(context.as_of, context.max_documents)
        actual_hashes = {
            hashlib.sha256(
                canonicalize_content(document.evidence.content).encode("utf-8")
            ).hexdigest()
            for batch in batches
            for document in batch.documents
        }
        historical_hashes = {
            document.content_hash
            for document in self.components.store.list_documents_as_of(context.as_of)
        }
        if actual_hashes | historical_hashes != set(context.source_hashes):
            _unavailable(
                "required production sources do not match workflow identity"
            )
        ingested = []
        for batch in batches:
            control.checkpoint()
            documents = self.components.evidence.ingest_batch(
                tuple(item.evidence for item in batch.documents)
            )
            control.checkpoint()
            checkpoints.save(batch.next_checkpoint)
            ingested.extend(documents)
        payload = tuple((item.document_id, item.content_hash) for item in ingested)
        digest = _digest(payload)
        return StageOutcome(
            result_ref=f"ingestion-{digest[:24]}", result_hash=digest
        )


class _HistoricalDocumentsAdapter:
    def __init__(self, components: ProductionComponents) -> None:
        self.components = components

    def run(
        self, context: StageContext, control: StageExecutionControl
    ) -> StageOutcome:
        control.checkpoint()
        documents = self.components.store.list_documents_as_of(context.as_of)
        if context.max_documents is not None:
            documents = documents[: context.max_documents]
        if not documents:
            _unavailable()
        payload = tuple((item.document_id, item.content_hash) for item in documents)
        digest = _digest(payload)
        return StageOutcome(
            result_ref=f"historical-{digest[:24]}", result_hash=digest
        )


class _SignalIngestionAdapter:
    def __init__(self, components: ProductionComponents) -> None:
        self.components = components

    def run(
        self, context: StageContext, control: StageExecutionControl
    ) -> StageOutcome:
        connectors = self.components.signal_connectors
        checkpoints = self.components.checkpoints
        persist = getattr(self.components.inputs, "persist_signals", None)
        if not connectors or checkpoints is None or not callable(persist):
            _unavailable()
        signals: list[EmergingSignal] = []
        for connector in connectors:
            control.checkpoint()
            try:
                batch = connector.fetch(checkpoints.load(connector.name))
            except ConnectorError:
                _unavailable("required production signal source is unavailable")
            control.checkpoint()
            if (
                not isinstance(batch, SignalConnectorBatch)
                or batch.connector != connector.name
                or not batch.status.available
            ):
                _unavailable("required production signal source is unavailable")
            _call_input(persist, context, batch.signals)
            control.checkpoint()
            checkpoints.save(batch.next_checkpoint)
            signals.extend(batch.signals)
        payload = tuple(
            (item.source_document_id, item.evidence_passage_id) for item in signals
        )
        digest = _digest(payload)
        return StageOutcome(result_ref=f"signals-{digest[:24]}", result_hash=digest)


class _ResolutionAdapter:
    def __init__(self, components: ProductionComponents) -> None:
        self.components = components

    def run(
        self, context: StageContext, control: StageExecutionControl
    ) -> StageOutcome:
        plan = _call_input(
            getattr(self.components.inputs, "exposure_plan", None), context
        )
        if not isinstance(plan, ExposurePlan):
            _unavailable("required production exposure input is invalid")
        resolved = []
        for identity in plan.identities:
            entity = self.components.entities.resolve(identity)
            if entity is None:
                _unavailable("required production entity resolution is incomplete")
            resolved.append(entity)
        control.checkpoint()
        exposures = self.components.exposures.map_positions(
            plan.positions,
            nav=plan.nav,
            base_currency=plan.base_currency,
            etf_holdings=plan.etf_holdings,
            as_of=context.as_of,
            fx_to_base=plan.fx_to_base,
            relationship_exposures=plan.relationship_exposures,
        )
        persist = getattr(self.components.inputs, "persist_resolution", None)
        _call_input(persist, context, tuple(resolved), exposures)
        payload = {
            "entities": tuple(item.entity_id for item in resolved),
            "exposures": tuple(
                (
                    symbol,
                    _decimal_text(item.direct_weight),
                    _decimal_text(item.lookthrough_weight),
                    item.etf_holdings_status,
                )
                for symbol, item in sorted(exposures.items())
            ),
        }
        digest = _digest(payload)
        return StageOutcome(result_ref=f"resolution-{digest[:24]}", result_hash=digest)


class _AgentStageAdapter:
    def __init__(
        self,
        components: ProductionComponents,
        stage: str,
        role: AgentRole,
        agent: BoundedAgent,
    ) -> None:
        self.components = components
        self.stage = stage
        self.role = role
        self.agent = agent

    def _run_agent(
        self, context: StageContext, control: StageExecutionControl
    ) -> tuple[AgentTask, AnalyticalOutput, str, str]:
        task = _call_input(
            getattr(self.components.inputs, "agent_task", None),
            self.stage,
            context,
            self.role,
        )
        if not isinstance(task, AgentTask):
            _unavailable("required production agent task is invalid")
        if task.task_id != context.task_id or task.run_id != context.workflow_id:
            _unavailable("required production agent task identity is invalid")
        if self.components.budget.month_total(
            context.as_of.year, context.as_of.month
        ) > self.components.budget.hard_limit:
            _unavailable("required production budget state is unavailable")
        control.checkpoint()
        output = self.agent.run(task)
        control.checkpoint()
        canonical = output.model_dump(mode="json", warnings="error")
        return task, output, _agent_attempt_id(task, self.role), _digest(canonical)

    def run(
        self, context: StageContext, control: StageExecutionControl
    ) -> StageOutcome:
        task, output, attempt_id, output_hash = self._run_agent(context, control)
        del task
        if self.stage == "materiality":
            material = _call_input(
                getattr(self.components.inputs, "is_material_event", None),
                context,
                output,
            )
            if type(material) is not bool:
                _unavailable("required production materiality decision is invalid")
            return StageOutcome(
                result_ref=attempt_id,
                result_hash=output_hash,
                new_agent_runs=1,
                material_event=material,
            )
        if self.stage == "review":
            if not isinstance(output, ReviewerOutput):
                _unavailable("required production review output is invalid")
            gate_input = _call_input(
                getattr(self.components.inputs, "quality_input", None),
                context,
                output,
            )
            if not isinstance(gate_input, QualityGateInput):
                _unavailable("required production quality input is invalid")
            gate = self.components.quality.evaluate(gate_input)
            report_ids = _workflow_report_ids(
                self.components.store, context.workflow_id
            )
            result_ref = report_ids[0] if len(report_ids) == 1 else attempt_id
            if report_ids:
                with self.components.store.transaction() as connection:
                    connection.executemany(
                        "UPDATE reports SET status = 'reviewed' "
                        "WHERE report_id = ? AND status IN ('draft', 'reviewed')",
                        ((report_id,) for report_id in report_ids),
                    )
            return StageOutcome(
                result_ref=result_ref,
                result_hash=output_hash,
                new_agent_runs=1,
                reviewer_verdict=output.verdict,
                quality_gate=gate,
                published_claim_ids=gate.allowed_claim_ids,
            )
        expected_report = _REPORT_STAGE_TYPES.get(self.stage)
        if expected_report is not None:
            report = _call_input(
                getattr(self.components.inputs, "report", None), context, output
            )
            if not isinstance(report, expected_report):
                _unavailable("required production report input is invalid")
            return _persist_report(self.components, context, control, report).model_copy(
                update={"new_agent_runs": 1}
            )
        return StageOutcome(
            result_ref=attempt_id,
            result_hash=output_hash,
            new_agent_runs=1,
        )


class _RevisionAdapter:
    """Route revision back to the durable task's originating specialist."""

    def __init__(
        self,
        components: ProductionComponents,
        agents: Mapping[AgentRole, BoundedAgent],
    ) -> None:
        self.components = components
        self.agents = agents

    def run(
        self, context: StageContext, control: StageExecutionControl
    ) -> StageOutcome:
        role = context.assigned_role
        if role is None or role not in self.agents or role is AgentRole.SKEPTICAL_REVIEWER:
            _unavailable("required production revision role is unavailable")
        return _AgentStageAdapter(
            self.components, "revision", role, self.agents[role]
        ).run(context, control)


def _workflow_portfolio_ref(store: ResearchStore, workflow_id: str) -> str | None:
    with store.connect() as connection:
        row = connection.execute(
            "SELECT outcome_json FROM workflow_tasks WHERE workflow_id = ? "
            "AND stage = 'portfolio' AND state = 'completed'",
            (workflow_id,),
        ).fetchone()
    if row is None or row[0] is None:
        return None
    try:
        return StageOutcome.model_validate_json(row[0]).result_ref
    except Exception:
        _unavailable("required production portfolio provenance is unavailable")


def _privacy_context(store: ResearchStore) -> PublicationPrivacyContext:
    with store.connect() as connection:
        snapshots = connection.execute(
            "SELECT nav, cash, account_ref FROM portfolio_snapshots"
        ).fetchall()
        positions = connection.execute(
            "SELECT quantity, market_value, cost_basis FROM positions"
        ).fetchall()
    if not snapshots and not positions:
        return PublicationPrivacyContext(
            mode="no_sensitive_data",
            sensitive_literals=(),
            account_identifiers=(),
            portfolio_values=(),
        )
    identifiers = tuple(sorted({row[2] for row in snapshots}))
    values = {abs(Decimal(row[index])) for row in snapshots for index in (0, 1)}
    for quantity, market_value, cost_basis in positions:
        values.update((abs(Decimal(quantity)), abs(Decimal(market_value))))
        if cost_basis is not None:
            values.add(abs(Decimal(cost_basis)))
    return PublicationPrivacyContext(
        mode="enforced",
        sensitive_literals=(),
        account_identifiers=identifiers,
        portfolio_values=tuple(sorted(value for value in values if value != 0)),
    )


def _render(
    components: ProductionComponents,
    report: EventUpdate | PortfolioBrief | IndustryLandscape | EmergingCompanyMonitor,
) -> RenderedReportArtifact:
    renderer = components.renderer or ReportRenderer(
        components.config.report_dir,
        privacy_context=_privacy_context(components.store),
    )
    method = getattr(renderer, f"render_{report.metadata.report_type}", None)
    if not callable(method):
        _unavailable("required production report renderer is unavailable")
    return method(report)


def _persist_report(
    components: ProductionComponents,
    context: StageContext,
    control: StageExecutionControl,
    report: EventUpdate | PortfolioBrief | IndustryLandscape | EmergingCompanyMonitor,
) -> StageOutcome:
    if report.metadata.as_of != context.as_of:
        _unavailable("required production report date is invalid")
    body = json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    payload_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    control.checkpoint()
    artifact = _render(components, report)
    control.checkpoint()
    metadata = json.dumps(
        {
            "html_filename": artifact.html_path.name,
            "payload_sha256": payload_hash,
            "pdf_status": (
                "rendered" if artifact.pdf_path is not None else artifact.pdf_error
            ),
            "portfolio_snapshot_ref": _workflow_portfolio_ref(
                components.store, context.workflow_id
            ),
            "report_type": report.metadata.report_type,
            "schema": "trusted_report_payload/v1",
            "task_id": context.task_id,
            "workflow_id": context.workflow_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = (
        f"{report.metadata.report_type}-{context.workflow_id}",
        1,
        report.metadata.title,
        body,
        metadata,
    )
    with components.store.transaction() as connection:
        existing = connection.execute(
            "SELECT report_key, version, title, body, metadata_json "
            "FROM reports WHERE report_id = ?",
            (report.metadata.report_id,),
        ).fetchone()
        if existing is None:
            connection.execute(
                "INSERT INTO reports (report_id, report_key, version, title, body, "
                "status, created_by_run_id, created_at, published_at, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, 'draft', NULL, ?, NULL, ?)",
                (
                    report.metadata.report_id,
                    *expected[:4],
                    _utc_text(context.as_of),
                    expected[4],
                ),
            )
        elif tuple(existing) != expected:
            _unavailable("stored production report conflicts with deterministic output")
    return StageOutcome(
        result_ref=report.metadata.report_id,
        result_hash=payload_hash,
        report_id=report.metadata.report_id,
    )


def _workflow_report_ids(store: ResearchStore, workflow_id: str) -> tuple[str, ...]:
    result = []
    with store.connect() as connection:
        rows = connection.execute(
            "SELECT report_id, metadata_json FROM reports ORDER BY report_id"
        ).fetchall()
    for report_id, metadata_json in rows:
        try:
            metadata = json.loads(metadata_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if metadata.get("workflow_id") == workflow_id:
            result.append(report_id)
    return tuple(result)


class _PublicationAdapter:
    def __init__(self, components: ProductionComponents) -> None:
        self.components = components

    def run(
        self, context: StageContext, control: StageExecutionControl
    ) -> StageOutcome:
        sink = self.components.publication_sink
        if sink is None or not callable(getattr(sink, "publish", None)):
            _unavailable()
        if not context.allowed_claim_ids or context.publication_effect_key is None:
            _unavailable("required production publication authority is unavailable")
        report_ids = _workflow_report_ids(self.components.store, context.workflow_id)
        if len(report_ids) != 1:
            _unavailable("required production report is unavailable")
        report_id = report_ids[0]
        with self.components.store.connect() as connection:
            row = connection.execute(
                "SELECT body, status FROM reports WHERE report_id = ?",
                (report_id,),
            ).fetchone()
        if row is None or row[1] != "reviewed":
            _unavailable("required production reviewed report is unavailable")
        try:
            raw = json.loads(row[0])
            model = _REPORT_TYPES[raw["metadata"]["report_type"]]
            report = model.model_validate(raw, strict=True)
        except Exception:
            _unavailable("required production report is invalid")
        control.checkpoint()
        artifact = _render(self.components, report)
        control.checkpoint()
        receipt = sink.publish(
            report, artifact, idempotency_key=context.publication_effect_key
        )
        control.checkpoint()
        if not isinstance(receipt, str) or not receipt.strip():
            _unavailable("required production publication receipt is invalid")
        receipt_hash = hashlib.sha256(receipt.encode("utf-8")).hexdigest()
        body_hash = hashlib.sha256(row[0].encode("utf-8")).hexdigest()
        with self.components.store.transaction() as connection:
            cursor = connection.execute(
                "UPDATE reports SET status = 'published', published_at = ? "
                "WHERE report_id = ? AND status = 'reviewed'",
                (_utc_text(context.as_of), report_id),
            )
            if cursor.rowcount != 1:
                _unavailable("required production publication state changed")
        return StageOutcome(
            result_ref=report_id,
            result_hash=body_hash,
            report_id=report_id,
            published_claim_ids=context.allowed_claim_ids,
            publication_receipt_hash=receipt_hash,
        )


def _agents(components: ProductionComponents) -> Mapping[AgentRole, BoundedAgent]:
    constructors = {
        AgentRole.RESEARCH_DIRECTOR: ResearchDirector,
        AgentRole.EVENT_SCOUT: EventScout,
        AgentRole.EMERGING_COMPANY_SCOUT: EmergingCompanyScout,
        AgentRole.EVIDENCE_ANALYST: EvidenceAnalyst,
        AgentRole.INDUSTRY_STRATEGIST: IndustryStrategist,
        AgentRole.FUNDAMENTAL_ANALYST: FundamentalAnalyst,
        AgentRole.SKEPTICAL_REVIEWER: SkepticalReviewer,
        AgentRole.RESEARCH_EDITOR: ResearchEditor,
    }
    return MappingProxyType(
        {
            role: constructor(
                components.router, components.store, clock=components.clock
            )
            for role, constructor in constructors.items()
        }
    )


def build_production_stage_adapters(
    config: ResearchConfig | ProductionComponents,
    store: ResearchStore | None = None,
    *,
    budget: BudgetLedger | None = None,
    router: ProviderRouter | None = None,
    ollama_provider: ModelProvider | None = None,
    external_provider: ModelProvider | None = None,
    price_table: PriceTable | None = None,
    evidence: EvidenceIngestor | None = None,
    entities: EntityResolver | None = None,
    exposures: PortfolioExposureMapper | None = None,
    quality: QualityGate | None = None,
    renderer: ReportRenderer | None = None,
    inputs: ProductionInputs | object | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    document_connectors: Sequence[ResearchConnector] = (),
    signal_connectors: Sequence[SignalResearchConnector] = (),
    checkpoints: ConnectorCheckpointStore | None = None,
    publication_sink: PublicationSink | None = None,
) -> Mapping[str, object]:
    """Build an immutable complete stage registry for ``RuntimeFactories``.

    The registry is complete even when a deployment has not configured every
    source.  Such adapters raise ``RequiredStageUnavailable`` when invoked,
    preserving durable orchestration state without producing synthetic advice.
    """

    if isinstance(config, ProductionComponents):
        if store is not None or any(
            value is not None
            for value in (
                budget,
                router,
                ollama_provider,
                external_provider,
                price_table,
                evidence,
                entities,
                exposures,
                quality,
                renderer,
                inputs,
                checkpoints,
                publication_sink,
            )
        ) or document_connectors or signal_connectors:
            raise TypeError("component form does not accept composition overrides")
        components = config
    else:
        if not isinstance(config, ResearchConfig) or not isinstance(store, ResearchStore):
            raise TypeError("config and store must be ResearchConfig and ResearchStore")
        configured_document_connectors = tuple(document_connectors)
        configured_checkpoints = checkpoints
        if not configured_document_connectors and configured_checkpoints is None:
            (
                configured_document_connectors,
                configured_checkpoints,
            ) = _configured_news_sources(config)
        ledger = budget or BudgetLedger(
            store, config.budget_soft_usd, config.budget_hard_usd
        )
        if router is None:
            local = ollama_provider or OllamaProvider(
                config.ollama_base_url,
                config.ollama_model,
                timeout=(3.05, config.ollama_health_timeout_seconds),
                allowed_private_hosts={"ollama"},
            )
            external = external_provider
            if external is None and config.openai_api_key is not None:
                if price_table is None:
                    price_table = PriceTable(
                        effective_until=config.model_price_effective_until,
                        prices={
                            model: ModelPrice(
                                config.model_input_price_per_million_usd,
                                config.model_output_price_per_million_usd,
                            )
                            for model in set(config.model_routes.values())
                        },
                        clock=clock,
                    )
                if price_table.effective_until != config.model_price_effective_until:
                    raise ValueError(
                        "model price table date must match research configuration"
                    )
                high_effort = {
                    AgentRole.RESEARCH_DIRECTOR,
                    AgentRole.INDUSTRY_STRATEGIST,
                    AgentRole.FUNDAMENTAL_ANALYST,
                    AgentRole.SKEPTICAL_REVIEWER,
                    AgentRole.RESEARCH_EDITOR,
                }
                routes = {
                    role: ModelRoute(
                        model,
                        (
                            ReasoningEffort.LOW
                            if role is AgentRole.EVENT_SCOUT
                            else ReasoningEffort.HIGH
                            if role in high_effort
                            else ReasoningEffort.MEDIUM
                        ),
                    )
                    for role, model in config.model_routes.items()
                }
                external = OpenAIProvider(
                    client=None,
                    routes=routes,
                    budget=ledger,
                    prices=price_table,
                    api_key=config.openai_api_key,
                    clock=clock,
                )
            router = ProviderRouter(
                config,
                external=external,
                ollama=local,
                clock=clock,
            )
        components = ProductionComponents(
            config=config,
            store=store,
            budget=ledger,
            router=router,
            evidence=evidence or EvidenceIngestor(store, config.cache_dir),
            entities=entities
            or EntityResolver.from_sec_company_tickers({}, store=store),
            exposures=exposures or PortfolioExposureMapper(),
            quality=quality
            or QualityGate(
                policy=QualityGatePolicy(
                    portfolio_max_age_hours=math.ceil(
                        config.portfolio_max_staleness_hours
                    ),
                    event_window_days=math.ceil(
                        config.source_max_staleness_hours / 24
                    ),
                )
            ),
            renderer=renderer,
            inputs=inputs or _RuntimeProductionInputs(store),
            clock=clock,
            document_connectors=configured_document_connectors,
            signal_connectors=tuple(signal_connectors),
            checkpoints=configured_checkpoints,
            publication_sink=publication_sink,
            sec_config=SECConfig(config.sec_user_agent),
        )
    agents = _agents(components)
    adapters: dict[str, object] = {
        "ingestion": _DocumentIngestionAdapter(components),
        "resolve": _ResolutionAdapter(components),
        "public_signals": _SignalIngestionAdapter(components),
        "historical_documents": _HistoricalDocumentsAdapter(components),
        "publish": _PublicationAdapter(components),
    }
    adapters.update(
        {
            stage: _AgentStageAdapter(
                components, stage, role, agents[role]
            )
            for stage, role in _AGENT_ROLES.items()
        }
    )
    adapters["revision"] = _RevisionAdapter(components, agents)
    if set(adapters) != set(PRODUCTION_STAGE_NAMES):
        raise RuntimeError("production stage registry is incomplete")
    return MappingProxyType(adapters)


__all__ = [
    "ConnectorCheckpointStore",
    "ExposurePlan",
    "PRODUCTION_STAGE_NAMES",
    "ProductionComponents",
    "ProductionInputs",
    "PublicationSink",
    "build_production_stage_adapters",
]
