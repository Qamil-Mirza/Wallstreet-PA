"""Composable production adapters for the durable research orchestrator.

This module is deliberately a composition root.  It does not invent research
content: every stage either delegates to an existing typed component using
caller-supplied source inputs, or fails closed with ``RequiredStageUnavailable``.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Protocol

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
from .agents.contracts import AgentTask, AnalyticalOutput, ReviewerOutput
from .budget import BudgetLedger, ModelPrice, PriceTable
from .config import ResearchConfig
from .connectors import (
    ConnectorBatch,
    ConnectorCheckpoint,
    ConnectorError,
    EmergingSignal,
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
from .models import AgentRole, ReviewVerdict
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
from .quality import QualityGate, QualityGateInput, QualityGatePolicy
from .reports import (
    EmergingCompanyMonitor,
    EventUpdate,
    IndustryLandscape,
    PortfolioBrief,
    PublicationPrivacyContext,
    RenderedReportArtifact,
    ReportRenderer,
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
                age = as_of - document.published_at.astimezone(timezone.utc)
                if age.total_seconds() < 0 or age.total_seconds() > (
                    self.components.config.source_max_staleness_hours * 3600
                ):
                    _unavailable(
                        "required production source is outside the freshness window"
                    )
                if document.evidence.retrieved_at.astimezone(timezone.utc) > as_of:
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
        ledger = budget or BudgetLedger(
            store, config.budget_soft_usd, config.budget_hard_usd
        )
        if router is None:
            local = ollama_provider or OllamaProvider(
                config.ollama_base_url,
                config.ollama_model,
                timeout=(3.05, config.ollama_health_timeout_seconds),
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
            inputs=inputs or ProductionInputs(),
            clock=clock,
            document_connectors=tuple(document_connectors),
            signal_connectors=tuple(signal_connectors),
            checkpoints=checkpoints,
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
