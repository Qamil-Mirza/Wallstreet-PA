"""Production composition contracts for durable research stage adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from news_bot.research.agents.contracts import (
    AgentTask,
    ClaimDraft,
    EvidenceAnalystInput,
    EvidenceAnalystOutput,
    ReviewerInput,
    ReviewerOutput,
    SecurityEligibility,
)
from news_bot.research.budget import BudgetLedger, ModelPrice, PriceTable
from news_bot.research.config import ResearchConfig
from news_bot.research.connectors import (
    ConnectorBatch,
    ConnectorCheckpoint,
    ConnectorError,
    NormalizedResearchDocument,
)
from news_bot.research.entities import EntityResolver, PortfolioExposureMapper
from news_bot.research.evidence import DocumentInput, EvidenceIngestor
from news_bot.research.models import (
    AgentRole,
    ClaimKind,
    EvidenceClaim,
    InferenceMode,
    RecommendationRating,
    ReviewVerdict,
)
from news_bot.research.orchestrator import (
    RequiredStageUnavailable,
    StageContext,
    StageExecutionControl,
    WorkflowKind,
)
from news_bot.research.production import (
    PRODUCTION_STAGE_NAMES,
    ProductionComponents,
    ProductionInputs,
    build_production_stage_adapters,
)
from news_bot.research.providers import ProviderRouter
from news_bot.research.providers.base import ModelRequest, ModelResponse
from news_bot.research.quality import (
    ClaimQualityInput,
    EvidenceReference,
    FilingAvailability,
    InferenceDisclosure,
    PublicationKind,
    PublicationVerdict,
    QualityGate,
    QualityGateInput,
)
from news_bot.research.reports import PublicationPrivacyContext, ReportRenderer
from news_bot.research.store import ResearchStore
from news_bot.research.runtime import RuntimeFactories, RuntimeWorkflowService


NOW = datetime(2026, 8, 24, 12, tzinfo=timezone.utc)


def _config(tmp_path: Path) -> ResearchConfig:
    return ResearchConfig(
        enabled=True,
        data_dir=tmp_path,
        openai_api_key=None,
        inference_mode=InferenceMode.LOCAL_ONLY,
        ollama_base_url="http://localhost:11434",
        ollama_model="offline-test-model",
        budget_soft_usd=Decimal("4"),
        budget_hard_usd=Decimal("5"),
    )


def _context(stage: str, *, dependency_refs: tuple[str, ...] = ()) -> StageContext:
    return StageContext(
        workflow_id="workflow-production-test",
        workflow_kind=WorkflowKind.DAILY,
        task_id=f"task-{stage}",
        stage=stage,
        as_of=NOW,
        period_key="2026-08-24",
        source_hashes=(),
        dependency_result_refs=dependency_refs,
        task_lease_token="a" * 64,
        publication_effect_key="b" * 64 if stage == "publish" else None,
    )


class _CheckpointStore:
    def __init__(self) -> None:
        self.values: dict[str, ConnectorCheckpoint] = {}

    def load(self, connector: str) -> ConnectorCheckpoint:
        return self.values.get(connector, ConnectorCheckpoint(connector))

    def save(self, checkpoint: ConnectorCheckpoint) -> None:
        self.values[checkpoint.connector] = checkpoint


class _NoopInputs:
    """The registry is constructible before source-specific inputs are ready."""


class _OfflineEvidenceProvider:
    name = "offline-production-test"
    model = "offline-test-model"

    def generate(self, request: ModelRequest) -> ModelResponse:
        assert request.role is AgentRole.EVIDENCE_ANALYST
        payload = EvidenceAnalystOutput(
            schema_version="1",
            evidence_ids=("passage-event",),
            as_of=NOW,
            confidence=0.9,
            inference_mode=InferenceMode.LOCAL_ONLY,
            claims=(
                ClaimDraft(
                    claim_id="provider-placeholder",
                    text="Primary filing capacity changed.",
                    kind=ClaimKind.FACT,
                    evidence_ids=("passage-event",),
                    supporting_claim_ids=(),
                ),
            ),
        )
        return ModelResponse(
            data=payload,
            raw_response_hash=hashlib.sha256(b"event-output").hexdigest(),
            input_tokens=10,
            output_tokens=10,
            reasoning_tokens=0,
            model=self.model,
            latency_ms=1,
            provider=self.name,
            inference_mode=InferenceMode.LOCAL_ONLY,
            run_id=request.run_id,
        )


@dataclass
class _DocumentConnector:
    source: DocumentInput
    name: str = "fixture_news"

    def fetch(self, checkpoint: ConnectorCheckpoint) -> ConnectorBatch:
        assert checkpoint.connector == self.name
        return ConnectorBatch(
            connector=self.name,
            documents=(NormalizedResearchDocument(self.source, ("portfolio",)),),
            next_checkpoint=ConnectorCheckpoint(self.name, cursor="cursor-1"),
        )


def _components(
    tmp_path: Path,
    *,
    document_connectors=(),
    checkpoints=None,
    inputs=None,
    provider=None,
) -> tuple[ProductionComponents, ResearchStore]:
    config = _config(tmp_path)
    store = ResearchStore(config.database_path, clock=lambda: NOW)
    store.migrate()
    budget = BudgetLedger(store, Decimal("4"), Decimal("5"))
    router = ProviderRouter(
        config,
        external=None,
        ollama=provider or _OfflineEvidenceProvider(),
        clock=lambda: NOW,
        monotonic=lambda: 1.0,
    )
    components = ProductionComponents(
        config=config,
        store=store,
        budget=budget,
        router=router,
        evidence=EvidenceIngestor(store, tmp_path / "cache"),
        entities=EntityResolver.from_sec_company_tickers({}, store=store),
        exposures=PortfolioExposureMapper(),
        quality=QualityGate(),
        renderer=ReportRenderer(
            tmp_path / "reports",
            privacy_context=PublicationPrivacyContext(
                mode="no_sensitive_data",
                sensitive_literals=(),
                account_identifiers=(),
                portfolio_values=(),
            ),
        ),
        document_connectors=tuple(document_connectors),
        checkpoints=checkpoints,
        inputs=inputs or _NoopInputs(),
        clock=lambda: NOW,
    )
    return components, store


def test_factory_registers_every_runtime_stage_and_freezes_registry(tmp_path: Path):
    components, _ = _components(tmp_path)

    adapters = build_production_stage_adapters(components)

    assert set(adapters) == set(PRODUCTION_STAGE_NAMES)
    assert all(callable(getattr(adapter, "run", None)) for adapter in adapters.values())
    with pytest.raises(TypeError):
        adapters["ingestion"] = adapters["resolve"]


def test_factory_composes_registry_from_runtime_config_and_store(tmp_path: Path):
    components, store = _components(tmp_path)

    adapters = build_production_stage_adapters(
        components.config,
        store,
        budget=components.budget,
        router=components.router,
        evidence=components.evidence,
        entities=components.entities,
        exposures=components.exposures,
        quality=components.quality,
        renderer=components.renderer,
        inputs=components.inputs,
        clock=components.clock,
    )

    assert set(adapters) == set(PRODUCTION_STAGE_NAMES)


def test_runtime_default_composition_installs_complete_production_registry(
    tmp_path: Path,
):
    config = _config(tmp_path)
    store = ResearchStore(config.database_path, clock=lambda: NOW)
    store.migrate()

    service = RuntimeWorkflowService(
        config,
        store,
        RuntimeFactories(owner_id_factory=lambda: "production-runtime-owner"),
    )

    assert set(service.factories.stage_adapters) == set(PRODUCTION_STAGE_NAMES)


def test_default_composition_consumes_runtime_routing_budget_and_source_policy(
    tmp_path: Path,
):
    config = replace(
        _config(tmp_path),
        openai_api_key="test-key",
        inference_mode=InferenceMode.EXTERNAL,
        source_max_staleness_hours=72.0,
        sec_user_agent="Research Test research@example.com",
    )
    store = ResearchStore(config.database_path, clock=lambda: NOW)
    store.migrate()
    adapters = build_production_stage_adapters(config, store, clock=lambda: NOW)
    components = adapters["ingestion"].components

    assert components.budget.soft_limit == config.budget_soft_usd
    assert components.budget.hard_limit == config.budget_hard_usd
    assert components.router.ollama.base_url == config.ollama_base_url
    assert components.router.ollama.model == config.ollama_model
    assert {
        role: route.model for role, route in components.router.external.routes.items()
    } == dict(config.model_routes)
    assert (
        components.router.external.prices.effective_until
        == config.model_price_effective_until
    )
    assert {
        price.input_per_million
        for price in components.router.external.prices.prices.values()
    } == {config.model_input_price_per_million_usd}
    assert {
        price.output_per_million
        for price in components.router.external.prices.prices.values()
    } == {config.model_output_price_per_million_usd}
    assert components.sec_config.user_agent == config.sec_user_agent


@pytest.mark.parametrize("stage", sorted({"resolve", "materiality", "review", "publish"}))
def test_missing_required_stage_inputs_fail_closed_without_fabrication(
    tmp_path: Path, stage: str
):
    components, store = _components(tmp_path)
    adapter = build_production_stage_adapters(components)[stage]

    with pytest.raises(RequiredStageUnavailable, match="required production input"):
        adapter.run(
            _context(stage),
            StageExecutionControl(
                task_id=f"task-{stage}",
                publication_effect_key="b" * 64 if stage == "publish" else None,
            ),
        )

    assert store.list_agent_executions("workflow-production-test") == ()


def test_ingestion_adapter_uses_connector_and_evidence_store_before_checkpoint(
    tmp_path: Path,
):
    checkpoints = _CheckpointStore()
    source = DocumentInput(
        source_type="issuer_release",
        url="https://example.com/release",
        publisher="Example Issuer",
        published_at=NOW,
        retrieved_at=NOW,
        content="A primary-source release describes a material capacity expansion.",
    )
    components, store = _components(
        tmp_path,
        document_connectors=(_DocumentConnector(source),),
        checkpoints=checkpoints,
    )

    outcome = build_production_stage_adapters(components)["ingestion"].run(
        _context("ingestion"), StageExecutionControl(task_id="task-ingestion")
    )

    assert outcome.result_ref.startswith("ingestion-")
    assert outcome.result_hash and len(outcome.result_hash) == 64
    assert checkpoints.values["fixture_news"].cursor == "cursor-1"
    with store.connect() as connection:
        row = connection.execute(
            "SELECT publisher, extraction_status FROM source_documents"
        ).fetchone()
    assert tuple(row) == ("Example Issuer", "complete")


def test_ingestion_preview_hashes_current_sources_without_advancing_checkpoint(
    tmp_path: Path,
):
    checkpoints = _CheckpointStore()
    source = DocumentInput(
        source_type="issuer_release",
        url="https://example.com/preview-release",
        publisher="Example Issuer",
        published_at=NOW,
        retrieved_at=NOW,
        content="A current primary source enters the workflow identity.",
    )
    connector = _DocumentConnector(source)
    components, store = _components(
        tmp_path,
        document_connectors=(connector,),
        checkpoints=checkpoints,
    )
    adapter = build_production_stage_adapters(components)["ingestion"]

    hashes = adapter.preview_source_hashes(NOW)

    assert hashes == (
        hashlib.sha256(source.content.encode("utf-8")).hexdigest(),
    )
    assert checkpoints.values == {}
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM source_documents").fetchone() == (0,)

    adapter.run(
        _context("ingestion"), StageExecutionControl(task_id="task-ingestion")
    )

    assert checkpoints.values["fixture_news"].cursor == "cursor-1"
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM source_documents").fetchone() == (1,)


def test_ingestion_rejects_stale_upstream_without_advancing_checkpoint(tmp_path: Path):
    checkpoints = _CheckpointStore()
    source = DocumentInput(
        source_type="issuer_release",
        url="https://example.com/stale-release",
        publisher="Example Issuer",
        published_at=NOW - timedelta(hours=169),
        retrieved_at=NOW,
        content="An old release must not enter the current research packet.",
    )
    components, store = _components(
        tmp_path,
        document_connectors=(_DocumentConnector(source),),
        checkpoints=checkpoints,
    )

    with pytest.raises(RequiredStageUnavailable, match="freshness window"):
        build_production_stage_adapters(components)["ingestion"].run(
            _context("ingestion"), StageExecutionControl(task_id="task-ingestion")
        )

    assert checkpoints.values == {}
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM source_documents").fetchone()[0] == 0


def test_ingestion_connector_failure_becomes_safe_required_stage_failure(
    tmp_path: Path,
):
    checkpoints = _CheckpointStore()

    class FailingConnector:
        name = "failing_source"

        def fetch(self, _checkpoint):
            raise ConnectorError(
                self.name,
                retryable=True,
                diagnostic_code="upstream_unavailable",
            )

    components, _ = _components(
        tmp_path,
        document_connectors=(FailingConnector(),),
        checkpoints=checkpoints,
    )

    with pytest.raises(RequiredStageUnavailable, match="connector is unavailable"):
        build_production_stage_adapters(components)["ingestion"].preview_source_hashes(
            NOW
        )


class _EventInputs(ProductionInputs):
    def agent_task(self, stage, context, role):
        assert stage == "materiality"
        assert role is AgentRole.EVIDENCE_ANALYST
        return AgentTask[EvidenceAnalystInput](
            task_id=context.task_id,
            run_id=context.workflow_id,
            input=EvidenceAnalystInput(
                evidence_ids=("passage-event",),
                as_of=context.as_of,
                question="Did the primary filing contain a material change?",
            ),
        )

    def is_material_event(self, context, output):
        return bool(output.claims)


def test_materiality_adapter_runs_real_agent_router_and_budget_guard(tmp_path: Path):
    components, store = _components(tmp_path, inputs=_EventInputs())
    store.insert_documents_with_passages(
        (
            (
                __import__("news_bot.research.models", fromlist=["SourceDocument"]).SourceDocument(
                    document_id="document-event",
                    source_type="sec_filing",
                    canonical_url="https://example.com/filing",
                    publisher="SEC",
                    published_at=NOW,
                    retrieved_at=NOW,
                    content_hash=hashlib.sha256(b"filing").hexdigest(),
                    raw_content_path=None,
                    extraction_status="complete",
                ),
                (
                    __import__("news_bot.research.store", fromlist=["DocumentPassageRecord"]).DocumentPassageRecord(
                        passage_id="passage-event",
                        document_id="document-event",
                        ordinal=0,
                        text="Capacity expansion was filed.",
                        content_hash=hashlib.sha256(b"Capacity expansion was filed.").hexdigest(),
                        start_offset=0,
                        end_offset=len("Capacity expansion was filed."),
                    ),
                ),
            ),
        )
    )

    outcome = build_production_stage_adapters(components)["materiality"].run(
        _context("materiality"), StageExecutionControl(task_id="task-materiality")
    )

    assert outcome.material_event is True
    assert outcome.new_agent_runs == 1
    assert outcome.result_ref.startswith("agent_attempt_")
    audits = store.list_agent_executions("workflow-production-test")
    assert len(audits) == 1
    assert audits[0].role is AgentRole.EVIDENCE_ANALYST
    assert components.budget.month_total(2026, 8) == Decimal("0")


class _OfflineReviewProvider:
    name = "offline-production-review"
    model = "offline-test-model"

    def generate(self, request: ModelRequest) -> ModelResponse:
        assert request.role is AgentRole.SKEPTICAL_REVIEWER
        evidence_ids = tuple(
            json.loads(request.canonical_evidence)["task_input"]["evidence_ids"]
        )
        output = ReviewerOutput(
            schema_version="1",
            evidence_ids=evidence_ids,
            as_of=NOW,
            confidence=0.9,
            inference_mode=InferenceMode.LOCAL_ONLY,
            verdict=ReviewVerdict.PASS,
            issues=(),
        )
        return ModelResponse(
            data=output,
            raw_response_hash=hashlib.sha256(b"review-output").hexdigest(),
            input_tokens=10,
            output_tokens=10,
            reasoning_tokens=0,
            model=self.model,
            latency_ms=1,
            provider=self.name,
            inference_mode=InferenceMode.LOCAL_ONLY,
            run_id=request.run_id,
        )


class _ReviewInputs(ProductionInputs):
    def __init__(self, passage_id: str, document_id: str) -> None:
        self.passage_id = passage_id
        self.document_id = document_id

    def agent_task(self, stage, context, role):
        assert stage == "review"
        assert role is AgentRole.SKEPTICAL_REVIEWER
        return AgentTask[ReviewerInput](
            task_id=context.task_id,
            run_id=context.workflow_id,
            input=ReviewerInput(
                evidence_ids=(self.passage_id,),
                claim_ids=("claim-review",),
                target_claim_ids=("claim-review",),
                as_of=context.as_of,
            ),
        )

    def quality_input(self, context, review):
        assert review.verdict is ReviewVerdict.PASS
        reference = EvidenceReference(
            evidence_id=self.passage_id,
            canonical_source_id=self.document_id,
            source_family="sec",
            source_type="sec_filing",
            stored=True,
            primary=True,
            published_at=NOW,
            retrieved_at=NOW,
        )
        return QualityGateInput(
            as_of=context.as_of,
            publication_kind=PublicationKind.EVENT_REPORT,
            requested_rating=RecommendationRating.NO_RATING,
            portfolio_snapshot_at=NOW,
            price=None,
            filing=FilingAvailability(
                filing_due=True, available_as_of=True, filed_at=NOW
            ),
            security=SecurityEligibility(
                entity_id="entity-review",
                security_id="security-review",
                symbol="NVDA",
                asset_kind="equity",
                resolved=True,
                public=True,
                tradable=True,
            ),
            claims=(
                ClaimQualityInput(
                    claim_id="claim-review",
                    section="event",
                    material=True,
                    evidence=(reference,),
                ),
            ),
            reviewer_verdict=review.verdict,
            reviewer_approved_claim_ids=("claim-review",),
            editor_claim_ids=("claim-review",),
            inference_disclosure=InferenceDisclosure(
                mode=InferenceMode.LOCAL_ONLY,
                provider="offline-production-review",
                model="offline-test-model",
                confidence=Decimal("0.9"),
            ),
            event_source_type="sec_filing",
            event_published_at=NOW,
        )


def test_review_adapter_runs_real_skeptic_and_deterministic_quality_gate(tmp_path: Path):
    components, store = _components(
        tmp_path,
        provider=_OfflineReviewProvider(),
    )
    source = DocumentInput(
        source_type="sec_filing",
        url="https://example.com/review-filing",
        publisher="SEC",
        published_at=NOW,
        retrieved_at=NOW,
        content="A filed primary-source fact supports the reviewed claim.",
    )
    document = components.evidence.ingest(source)
    passage = document.passages[0]
    components = replace(
        components,
        inputs=_ReviewInputs(passage.passage_id, document.document_id),
    )
    store.insert_claim_with_lineage(
        EvidenceClaim(
            claim_id="claim-review",
            entity_id=None,
            kind=ClaimKind.FACT,
            text="The filing supports the reviewed claim.",
            as_of=NOW,
            confidence=Decimal("0.9"),
            status="active",
        ),
        passage_links=((passage.passage_id, "supports"),),
        supporting_claim_ids=(),
    )

    outcome = build_production_stage_adapters(components)["review"].run(
        _context("review"), StageExecutionControl(task_id="task-review")
    )

    assert outcome.reviewer_verdict is ReviewVerdict.PASS
    assert outcome.quality_gate.publication_verdict is PublicationVerdict.PARTIAL
    assert outcome.published_claim_ids == ("claim-review",)
