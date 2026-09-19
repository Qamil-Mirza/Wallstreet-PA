"""Deterministic, offline acceptance harness for the research pipeline.

The fixtures contain test-authored paraphrases of public facts.  This module
deliberately composes production domain models, SQLite repositories, budget
controls, report rendering, and the synthetic runtime instead of replacing
those boundaries with a second fake implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
from math import ceil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_UP, Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

from news_bot.research.agents.analysts import EvidenceAnalyst, FundamentalAnalyst
from news_bot.research.agents.contracts import (
    AgentTask,
    EditorInput,
    EvidenceAnalystInput,
    EvidenceAnalystOutput,
    FundamentalAnalystInput,
    FundamentalAnalystOutput,
    ResearchEditorOutput,
    ReviewerInput,
    ReviewerOutput,
    SecurityEligibility,
)
from news_bot.research.agents.editor import ResearchEditor
from news_bot.research.agents.reviewer import SkepticalReviewer

from news_bot.research.budget import (
    BudgetExceeded,
    BudgetLedger,
    SoftBudgetExceeded,
)
from news_bot.research.config import ResearchConfig
from news_bot.research.models import (
    AgentRole,
    ClaimKind,
    EvidenceClaim,
    InferenceMode,
    RecommendationRating,
    SourceDocument,
)
from news_bot.research.providers.base import ModelRequest, ModelResponse
from news_bot.research.providers.router import ProviderRouter
from news_bot.research.reports import (
    Citation,
    EventUpdate,
    PublicationPrivacyContext,
    ReportMetadata,
    ReportRenderer,
    ReportSection,
    RenderedReportArtifact,
)
from news_bot.research.reports.exhibits import (
    StoredExposureRow,
    build_exposure_exhibit,
)
from news_bot.research.runtime import RuntimeFactories, RuntimeWorkflowService
from news_bot.research.quality import CalculatedExhibit, CalculatedRow
from news_bot.research.store import (
    AgentRunAudit,
    DocumentPassageRecord,
    ResearchStore,
)


GOLDEN_ROOT = Path(__file__).parent / "golden"
SOURCE_PACKET = GOLDEN_ROOT / "source_packet"
_ADMINISTRATIVE_DISCLOSURE = (
    "This report is research, not an order: no margin, no forced liquidation, "
    "and human review is required for thin-liquidity ideas."
)


class ReplayDateError(ValueError):
    """The historical boundary cannot establish a source publication date."""


@dataclass(frozen=True)
class MaterialClaim:
    claim_id: str
    text: str
    kind: ClaimKind
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReplayResult:
    as_of: date
    documents: tuple[SourceDocument, ...]


@dataclass(frozen=True)
class GoldenRun:
    store: ResearchStore
    report: EventUpdate
    material_claims: tuple[MaterialClaim, ...]
    documents: tuple[SourceDocument, ...]
    rating: RecommendationRating
    paid_cost: Decimal
    fundamental: FundamentalAnalystOutput
    reviewer: ReviewerOutput
    editor: ResearchEditorOutput
    agent_audits: tuple[AgentRunAudit, ...]
    artifact: RenderedReportArtifact | None = None


@dataclass(frozen=True)
class GoldenEvaluation:
    scores: dict[str, Decimal]
    minimum_scores: dict[str, Decimal]
    failures: tuple[str, ...]

    @property
    def passes(self) -> bool:
        return not self.failures and all(
            self.scores[name] >= minimum
            for name, minimum in self.minimum_scores.items()
        )


@dataclass(frozen=True)
class CostSimulationResult:
    paid_cost: Decimal
    deferred_tasks: int
    completed_tasks: int


@dataclass(frozen=True)
class SyntheticEndToEndResult:
    html_path: Path
    pdf_path: Path
    html: str
    workflow_status: str
    omissions: tuple[str, ...]
    external_calls: tuple[str, ...]
    order_or_trade_requests: tuple[str, ...]
    email_attempts: int


class DeterministicGoldenProvider:
    """Offline structured provider that still exercises routing and accounting."""

    name = "deterministic-golden-provider"
    model = "deterministic-golden-v2"
    input_per_million = Decimal("0.20")
    output_per_million = Decimal("0.60")
    microdollar = Decimal("0.000001")

    def __init__(
        self,
        ledger: BudgetLedger,
        *,
        as_of: datetime,
    ) -> None:
        self.ledger = ledger
        self.as_of = as_of
        self.roles: list[AgentRole] = []

    def _payload(self, request: ModelRequest) -> dict[str, object]:
        evidence = json.loads(request.canonical_evidence)
        passage_text = " ".join(
            passage["text"] for passage in evidence["passages"]
        )
        required_facts = (
            "NVIDIA uses third parties to manufacture",
            "TSMC manufactures semiconductors for customers",
            "continued investment in advanced process",
        )
        if not all(fact in passage_text for fact in required_facts):
            raise AssertionError("agent request omitted canonical fixture evidence")
        task_input = evidence["task_input"]
        common: dict[str, object] = {
            "schema_version": "1",
            "evidence_ids": task_input["evidence_ids"],
            "as_of": task_input["as_of"],
            "confidence": 0.8,
            "inference_mode": "local_only",
        }
        if request.role is AgentRole.EVIDENCE_ANALYST:
            if not task_input["claim_ids"]:
                common["claims"] = [
                    {
                        "claim_id": "provider-placeholder-supply",
                        "text": (
                            "NVIDIA depends on third-party manufacturing and "
                            "packaging capacity."
                        ),
                        "kind": "fact",
                        "evidence_ids": ["passage-nvda-supply"],
                        "supporting_claim_ids": [],
                    },
                    {
                        "claim_id": "provider-placeholder-foundry",
                        "text": (
                            "TSMC is an upstream dedicated foundry for fabless "
                            "designers."
                        ),
                        "kind": "fact",
                        "evidence_ids": ["passage-tsmc-foundry"],
                        "supporting_claim_ids": [],
                    },
                ]
            else:
                facts_by_text = {
                    claim["text"]: claim["claim_id"]
                    for claim in evidence["claims"]
                }
                fact_ids = sorted(
                    (
                        facts_by_text[
                            "NVIDIA depends on third-party manufacturing and "
                            "packaging capacity."
                        ],
                        facts_by_text[
                            "TSMC is an upstream dedicated foundry for fabless "
                            "designers."
                        ],
                    )
                )
                common["claims"] = [
                    {
                        "claim_id": "provider-placeholder-chain",
                        "text": (
                            "Advanced packaging capacity can constrain NVIDIA "
                            "product availability."
                        ),
                        "kind": "inference",
                        "evidence_ids": [
                            "passage-nvda-supply",
                            "passage-tsmc-foundry",
                        ],
                        "supporting_claim_ids": fact_ids,
                    },
                    {
                        "claim_id": "provider-placeholder-counter",
                        "text": (
                            "Additional foundry and packaging capacity may ease "
                            "constraints."
                        ),
                        "kind": "inference",
                        "evidence_ids": ["passage-tsmc-capacity"],
                        "supporting_claim_ids": fact_ids,
                    },
                    {
                        "claim_id": "provider-placeholder-rating",
                        "text": (
                            "A low-liquidity account warrants HOLD, no margin, no "
                            "forced liquidation, and human review."
                        ),
                        "kind": "inference",
                        "evidence_ids": [
                            "passage-nvda-supply",
                            "passage-tsmc-foundry",
                            "passage-tsmc-capacity",
                        ],
                        "supporting_claim_ids": fact_ids,
                    },
                ]
        elif request.role is AgentRole.FUNDAMENTAL_ANALYST:
            security = task_input["security"]
            common.update(
                {
                    "security_id": security["security_id"],
                    "thesis": (
                        "Hold while upstream supply evidence lacks a current "
                        "valuation basis."
                    ),
                    "horizon_months": task_input["horizon_months"],
                    "valuation": {
                        "low": "0",
                        "high": "0",
                        "currency": "USD",
                        "as_of": task_input["as_of"],
                    },
                    "assumptions": ["No margin or forced sale is available."],
                    "catalysts": ["Verified expansion in packaging capacity."],
                    "counter_thesis": (
                        "Capacity expansion may remove the observed bottleneck."
                    ),
                    "risks": ["Valuation evidence is intentionally unavailable."],
                    "invalidation_conditions": [
                        "Fresh valuation evidence changes the risk-reward balance."
                    ],
                    "rating": "hold",
                    "eligible": True,
                }
            )
        elif request.role is AgentRole.SKEPTICAL_REVIEWER:
            common.update({"verdict": "pass", "issues": []})
        elif request.role is AgentRole.RESEARCH_EDITOR:
            approved = set(task_input["approved_claim_ids"])
            claims_by_text = {
                claim["text"]: claim["claim_id"]
                for claim in evidence["claims"]
                if claim["claim_id"] in approved
            }

            def selected(fragment: str) -> list[str]:
                matches = [
                    claim_id
                    for text, claim_id in claims_by_text.items()
                    if fragment in text
                ]
                if not matches:
                    raise AssertionError("editor fixture claim was not persisted")
                return matches

            common.update(
                {
                    "title": "Semiconductor value-chain research replay",
                    "sections": [
                        {
                            "heading": "Thesis and rating",
                            "approved_claim_ids": selected("low-liquidity"),
                        },
                        {
                            "heading": "Documented evidence",
                            "approved_claim_ids": [
                                *selected("third-party manufacturing"),
                                *selected("upstream dedicated foundry"),
                            ],
                        },
                        {
                            "heading": "Causal chain",
                            "approved_claim_ids": selected("can constrain"),
                        },
                        {
                            "heading": "Counter-thesis",
                            "approved_claim_ids": selected("may ease constraints"),
                        },
                    ],
                }
            )
        else:
            raise AssertionError(f"unexpected golden role: {request.role.value}")
        return common

    def generate(self, request: ModelRequest) -> ModelResponse:
        expected_schema = {
            AgentRole.EVIDENCE_ANALYST: EvidenceAnalystOutput,
            AgentRole.FUNDAMENTAL_ANALYST: FundamentalAnalystOutput,
            AgentRole.SKEPTICAL_REVIEWER: ReviewerOutput,
            AgentRole.RESEARCH_EDITOR: ResearchEditorOutput,
        }[request.role]
        if request.output_schema is not expected_schema:
            raise AssertionError("role was routed to the wrong output contract")
        payload = self._payload(request)
        output = request.output_schema.model_validate(payload)
        canonical = json.dumps(
            output.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        input_tokens = ceil(
            len(
                (
                    request.system_prompt
                    + request.provider_input
                    + request.canonical_schema
                ).encode("utf-8")
            )
            / 4
        )
        output_tokens = ceil(len(canonical.encode("utf-8")) / 4)
        reasoning_tokens = max(1, output_tokens // 10)

        def cost(input_count: int, output_count: int) -> Decimal:
            amount = (
                Decimal(input_count) * self.input_per_million
                + Decimal(output_count) * self.output_per_million
            ) / Decimal("1000000")
            return amount.quantize(self.microdollar, rounding=ROUND_UP)

        estimated_cost = cost(input_tokens, request.max_output_tokens)
        actual_cost = cost(input_tokens, output_tokens + reasoning_tokens)
        reservation = self.ledger.reserve(
            f"{request.run_id}:{request.role.value}",
            estimated_cost,
            now=self.as_of,
            role=request.role,
        )
        reconciled = self.ledger.reconcile(
            reservation.id,
            actual_cost,
            now=self.as_of,
        )
        self.roles.append(request.role)
        return ModelResponse(
            data=output,
            raw_response_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            model=self.model,
            latency_ms=1,
            provider=self.name,
            inference_mode=InferenceMode.LOCAL_ONLY,
            run_id=request.run_id,
            reservation_id=reconciled.id,
            reservation_state=reconciled.state,
            reserved_cost_usd=reconciled.amount,
        )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"golden fixture must be an object: {path.name}")
    return value


def load_expected_characteristics() -> dict[str, Any]:
    return _load_json(GOLDEN_ROOT / "expected_characteristics.json")


def _parse_publication_date(value: object, *, document_id: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ReplayDateError(
            f"document {document_id!r} has no trustworthy published_at"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ReplayDateError(
            f"document {document_id!r} has an invalid published_at"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReplayDateError(
            f"document {document_id!r} has a timezone-unknown published_at"
        )
    return parsed.astimezone(timezone.utc)


def _packet_records(
    packet_root: Path = SOURCE_PACKET,
) -> tuple[tuple[SourceDocument, tuple[DocumentPassageRecord, ...]], ...]:
    manifest = _load_json(packet_root / "manifest.json")
    if (
        manifest.get("created_for_test") is not True
        or manifest.get("redistribution_license") != "CC0-1.0"
        or not manifest.get("provenance")
    ):
        raise ValueError("golden packet must declare test provenance and license")

    records = []
    for filename in manifest.get("documents", ()):
        raw = _load_json(packet_root / str(filename))
        document_id = str(raw.get("document_id", ""))
        if (
            raw.get("created_for_test") is not True
            or raw.get("redistribution_license") != "CC0-1.0"
            or not raw.get("provenance")
        ):
            raise ValueError(f"fixture {filename!r} lacks redistribution metadata")
        published_at = _parse_publication_date(
            raw.get("published_at"), document_id=document_id
        )
        retrieved_at = _parse_publication_date(
            raw.get("retrieved_at"), document_id=document_id
        )
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        document = SourceDocument(
            document_id=document_id,
            source_type=str(raw["source_type"]),
            canonical_url=str(raw["canonical_url"]),
            publisher=str(raw["publisher"]),
            published_at=published_at,
            retrieved_at=retrieved_at,
            content_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            raw_content_path=str(packet_root / str(filename)),
            extraction_status="complete",
        )
        passages = []
        for ordinal, item in enumerate(raw["passages"]):
            text = str(item["text"])
            passages.append(
                DocumentPassageRecord(
                    passage_id=str(item["passage_id"]),
                    document_id=document_id,
                    ordinal=ordinal,
                    text=text,
                    content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    start_offset=0,
                    end_offset=len(text),
                    published_at=published_at,
                    retrieved_at=retrieved_at,
                )
            )
        records.append((document, tuple(passages)))
    return tuple(records)


def run_historical_replay(
    tmp_path: Path,
    *,
    as_of: date,
    packet_root: Path = SOURCE_PACKET,
) -> ReplayResult:
    """Persist fixtures, then select only documents known by the replay date."""
    if type(as_of) is not date:
        raise TypeError("as_of must be date")
    store = ResearchStore(tmp_path / "replay.db")
    store.migrate()
    store.insert_documents_with_passages(_packet_records(packet_root))
    cutoff = datetime.combine(as_of, datetime.max.time(), tzinfo=timezone.utc)
    documents = store.list_documents_as_of(cutoff)
    return ReplayResult(as_of=as_of, documents=documents)


def _research_config(data_dir: Path) -> ResearchConfig:
    return ResearchConfig(
        enabled=True,
        data_dir=data_dir,
        openai_api_key=None,
        inference_mode=InferenceMode.LOCAL_ONLY,
        ollama_base_url="http://localhost:11434",
        ollama_model="deterministic-golden-v2",
        budget_soft_usd=Decimal("4.00"),
        budget_hard_usd=Decimal("5.00"),
    )


def _persist_analyst_claims(
    store: ResearchStore,
    output: EvidenceAnalystOutput,
) -> tuple[MaterialClaim, ...]:
    material: list[MaterialClaim] = []
    for draft in output.claims:
        store.insert_claim_with_lineage(
            EvidenceClaim(
                claim_id=draft.claim_id,
                entity_id=None,
                kind=draft.kind,
                text=draft.text,
                as_of=output.as_of,
                confidence=Decimal(str(output.confidence)),
                status="active",
            ),
            passage_links=tuple(
                (evidence_id, "supports")
                for evidence_id in draft.evidence_ids
            ),
            supporting_claim_ids=draft.supporting_claim_ids,
        )
        material.append(
            MaterialClaim(
                claim_id=draft.claim_id,
                text=draft.text,
                kind=draft.kind,
                evidence_ids=draft.evidence_ids,
            )
        )
    return tuple(material)


def _run_specialists(
    store: ResearchStore,
    *,
    as_of: datetime,
    data_dir: Path,
) -> tuple[
    tuple[MaterialClaim, ...],
    FundamentalAnalystOutput,
    ReviewerOutput,
    ResearchEditorOutput,
    tuple[AgentRunAudit, ...],
    Decimal,
]:
    reservation_ids = iter(f"golden-reservation-{index}" for index in range(1, 9))
    ledger = BudgetLedger(
        store=store,
        soft_limit=Decimal("4.00"),
        hard_limit=Decimal("5.00"),
        id_factory=lambda: next(reservation_ids),
    )
    provider = DeterministicGoldenProvider(ledger, as_of=as_of)
    router = ProviderRouter(
        _research_config(data_dir),
        external=None,
        ollama=provider,
        clock=lambda: as_of,
        monotonic=lambda: 1.0,
    )
    evidence_ids = (
        "passage-nvda-supply",
        "passage-tsmc-capacity",
        "passage-tsmc-foundry",
    )
    workflow_id = "golden-specialist-workflow"

    fact_output = EvidenceAnalyst(
        router, store, clock=lambda: as_of
    ).run(
        AgentTask[EvidenceAnalystInput](
            task_id="01-evidence-facts",
            run_id=workflow_id,
            input=EvidenceAnalystInput(
                evidence_ids=evidence_ids,
                as_of=as_of,
                question=(
                    "How do upstream manufacturing constraints affect the "
                    "holding and its conservative research posture?"
                ),
            ),
        )
    )
    fact_claims = _persist_analyst_claims(store, fact_output)
    fact_claim_ids = tuple(claim.claim_id for claim in fact_claims)
    inference_output = EvidenceAnalyst(
        router, store, clock=lambda: as_of
    ).run(
        AgentTask[EvidenceAnalystInput](
            task_id="02-evidence-inferences",
            run_id=workflow_id,
            input=EvidenceAnalystInput(
                evidence_ids=evidence_ids,
                claim_ids=fact_claim_ids,
                as_of=as_of,
                question=(
                    "What causal, counter-thesis, and portfolio-posture "
                    "inferences follow from the documented facts?"
                ),
            ),
        )
    )
    inference_claims = _persist_analyst_claims(store, inference_output)
    material_claims = (*fact_claims, *inference_claims)
    claim_ids = tuple(claim.claim_id for claim in material_claims)

    fundamental = FundamentalAnalyst(
        router, store, clock=lambda: as_of
    ).run(
        AgentTask[FundamentalAnalystInput](
            task_id="03-fundamental-analyst",
            run_id=workflow_id,
            input=FundamentalAnalystInput(
                evidence_ids=evidence_ids,
                claim_ids=claim_ids,
                as_of=as_of,
                security=SecurityEligibility(
                    entity_id="entity-nvidia",
                    security_id="security-nvda",
                    symbol="NVDA",
                    asset_kind="equity",
                    resolved=True,
                    public=True,
                    tradable=True,
                ),
                horizon_months=12,
            ),
        )
    )
    reviewer = SkepticalReviewer(router, store, clock=lambda: as_of).run(
        AgentTask[ReviewerInput](
            task_id="04-skeptical-reviewer",
            run_id=workflow_id,
            input=ReviewerInput(
                evidence_ids=evidence_ids,
                claim_ids=claim_ids,
                target_claim_ids=claim_ids,
                as_of=as_of,
            ),
        )
    )
    editor = ResearchEditor(router, store, clock=lambda: as_of).run(
        AgentTask[EditorInput](
            task_id="05-research-editor",
            run_id=workflow_id,
            input=EditorInput(
                evidence_ids=evidence_ids,
                claim_ids=claim_ids,
                approved_claim_ids=claim_ids,
                as_of=as_of,
            ),
        )
    )
    selected_ids = {
        claim_id
        for section in editor.sections
        for claim_id in section.approved_claim_ids
    }
    selected_claims = tuple(
        claim for claim in material_claims if claim.claim_id in selected_ids
    )
    return (
        selected_claims,
        fundamental,
        reviewer,
        editor,
        store.list_agent_executions(workflow_id),
        ledger.month_total(as_of.year, as_of.month),
    )


def _golden_report(
    store: ResearchStore,
    documents: tuple[SourceDocument, ...],
    editor: ResearchEditorOutput,
    agent_audits: tuple[AgentRunAudit, ...],
    as_of: datetime,
) -> EventUpdate:
    provenance = {(audit.provider, audit.model) for audit in agent_audits}
    if (
        not provenance
        or len(provenance) != 1
        or any(provider is None or model is None for provider, model in provenance)
    ):
        raise AssertionError("golden agent audits must share one provider and model")
    provider, model = provenance.pop()
    by_id = {document.document_id: document for document in documents}
    evidence_to_document = {
        "passage-nvda-supply": by_id["doc-nvda-2025-10k"],
        "passage-tsmc-foundry": by_id["doc-tsmc-2024-annual"],
        "passage-tsmc-capacity": by_id["doc-tsmc-2024-annual"],
    }
    citations = tuple(
        Citation(
            evidence_id=evidence_id,
            source=document.publisher,
            url=document.canonical_url,
            source_date=document.published_at.date(),
            data_date=document.published_at.date(),
            content_hash=document.content_hash,
        )
        for evidence_id, document in evidence_to_document.items()
    )

    editor_sections = {section.heading: section for section in editor.sections}

    def section(heading: str) -> ReportSection:
        selected = editor_sections[heading]
        claims = []
        evidence_ids: set[str] = set()
        for claim_id in selected.approved_claim_ids:
            claim = store.get_claim(claim_id)
            lineage = store.list_claim_lineage(claim_id)
            if claim is None or not lineage:
                raise AssertionError("editor selected a claim without stored lineage")
            claims.append(claim.text)
            evidence_ids.update(item.passage.passage_id for item in lineage)
        return ReportSection(
            title=heading,
            body=" ".join(claims),
            evidence_ids=tuple(sorted(evidence_ids)),
        )

    thesis = section("Thesis and rating")
    documented = section("Documented evidence")
    causal = section("Causal chain")
    counter = section("Counter-thesis")
    exposure_row = StoredExposureRow(
        row_id="NVDA",
        symbol="NVDA",
        label="Synthetic NVIDIA exposure",
        weight=Decimal("1.0"),
        currency="USD",
        source_note="Synthetic acceptance portfolio; cited supply evidence",
        source_date=as_of.date(),
    )
    calculated_row = CalculatedRow(
        row_id=exposure_row.row_id,
        values={"weight": exposure_row.weight},
    )
    exhibit = build_exposure_exhibit(
        (exposure_row,),
        calculation=CalculatedExhibit(
            exhibit_id="exposure",
            normalized_rows=(calculated_row,),
            rendered_rows=(calculated_row,),
        ),
        evidence_ids=("passage-nvda-supply",),
    )
    return EventUpdate(
        metadata=ReportMetadata(
            report_id="golden-semiconductor-value-chain",
            report_type="event_update",
            title="Semiconductor value-chain research replay",
            as_of=as_of,
            inference_mode=InferenceMode.LOCAL_ONLY,
            provider=provider,
            model=model,
            freshness=(
                "Only sources published on or before the replay date are eligible."
            ),
            citations=citations,
            methodology=(
                "Facts are fixture-backed. Inference is explicitly labeled and traced "
                "to cited passages; no future documents are available to analysis."
            ),
            omissions=("Live prices and valuation inputs are intentionally omitted.",),
            disclosure=_ADMINISTRATIVE_DISCLOSURE,
        ),
        thesis=thesis,
        event_decomposition=(documented,),
        causal_decomposition=(causal,),
        read_through=(causal,),
        exhibits=(exhibit,),
        thesis_changes=(thesis,),
        unchanged_assumptions=(counter,),
        questions=(counter,),
        signposts=(documented,),
    )


def assert_report_claim_binding(
    report: EventUpdate,
    store: ResearchStore,
    editor: ResearchEditorOutput,
) -> None:
    """Prove every material section is exact prose from editor-approved claims."""
    if report.metadata.disclosure != _ADMINISTRATIVE_DISCLOSURE:
        raise AssertionError("report disclosure must remain administrative only")
    editor_sections = {section.heading: section for section in editor.sections}
    if len(editor_sections) != len(editor.sections):
        raise AssertionError("editor section headings must be unique")
    report_sections = (
        report.thesis,
        *report.event_decomposition,
        *report.causal_decomposition,
        *report.read_through,
        *report.thesis_changes,
        *report.unchanged_assumptions,
        *report.questions,
        *report.signposts,
    )
    observed_headings: set[str] = set()
    for section in report_sections:
        selected = editor_sections.get(section.title)
        if selected is None:
            raise AssertionError("report section is not backed by approved claims")
        claims = tuple(
            store.get_claim(claim_id)
            for claim_id in selected.approved_claim_ids
        )
        if any(claim is None for claim in claims):
            raise AssertionError("report section references an unknown approved claim")
        expected_body = " ".join(claim.text for claim in claims if claim is not None)
        expected_evidence = {
            lineage.passage.passage_id
            for claim_id in selected.approved_claim_ids
            for lineage in store.list_claim_lineage(claim_id)
        }
        if (
            section.body != expected_body
            or set(section.evidence_ids) != expected_evidence
        ):
            raise AssertionError("report section differs from its approved claims")
        observed_headings.add(section.title)
    if observed_headings != set(editor_sections):
        raise AssertionError("report omits editor-approved claims")


def build_golden_run(tmp_path: Path, *, render: bool = False) -> GoldenRun:
    expected = load_expected_characteristics()
    replay = run_historical_replay(
        tmp_path, as_of=date.fromisoformat(expected["as_of"])
    )
    store = ResearchStore(tmp_path / "replay.db")
    as_of = datetime.combine(replay.as_of, datetime.max.time(), tzinfo=timezone.utc)
    (
        material_claims,
        fundamental,
        reviewer,
        editor,
        agent_audits,
        paid_cost,
    ) = _run_specialists(store, as_of=as_of, data_dir=tmp_path)
    report = _golden_report(
        store,
        replay.documents,
        editor,
        agent_audits,
        as_of,
    )
    assert_report_claim_binding(report, store, editor)
    artifact = None
    if render:
        homebrew_lib = Path("/opt/homebrew/lib")
        if homebrew_lib.is_dir():
            os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", str(homebrew_lib))
        artifact = ReportRenderer(
            tmp_path / "golden-reports",
            privacy_context=PublicationPrivacyContext(
                mode="no_sensitive_data",
                sensitive_literals=(),
                account_identifiers=(),
                portfolio_values=(),
            ),
        ).render_event_update(report)
        if artifact.pdf_path is None:
            raise AssertionError(
                f"native PDF rendering failed: {artifact.pdf_error}"
            )
    return GoldenRun(
        store=store,
        report=report,
        material_claims=material_claims,
        documents=replay.documents,
        rating=fundamental.rating,
        paid_cost=paid_cost,
        fundamental=fundamental,
        reviewer=reviewer,
        editor=editor,
        agent_audits=agent_audits,
        artifact=artifact,
    )


def run_golden_evaluation(tmp_path: Path) -> GoldenEvaluation:
    expected = load_expected_characteristics()
    run = build_golden_run(tmp_path)
    report_text = " ".join(
        str(value)
        for value in run.report.model_dump(mode="json").values()
    ).casefold()
    actual_claim_ids = {claim.claim_id for claim in run.material_claims}
    expected_claim_ids = set(expected["expected_claim_ids"])
    actual_claim_texts = {claim.text for claim in run.material_claims}
    expected_claim_texts = set(expected["expected_claim_texts"])
    actual_facts = {
        claim.text for claim in run.material_claims if claim.kind is ClaimKind.FACT
    }
    expected_facts = set(expected["expected_facts"])
    cited_ids = {item.evidence_id for item in run.report.metadata.citations}
    expected_evidence_ids = set(expected["expected_evidence_ids"])

    def fraction(items: list[str], predicate: Callable[[str], bool]) -> Decimal:
        matched = sum(1 for item in items if predicate(item))
        return Decimal(matched) / Decimal(len(items)) if items else Decimal("1")

    scores = {
        "factual_accuracy": (
            Decimal("1")
            if (
                actual_claim_ids == expected_claim_ids
                and actual_claim_texts == expected_claim_texts
                and actual_facts == expected_facts
            )
            else Decimal("0")
        ),
        "citation_coverage": (
            Decimal("1")
            if all(
                claim.evidence_ids
                and set(claim.evidence_ids) <= cited_ids
                and run.store.list_claim_lineage(claim.claim_id)
                for claim in run.material_claims
            )
            and cited_ids == expected_evidence_ids
            else Decimal("0")
        ),
        "causal_links": fraction(
            expected["required_causal_links"],
            lambda term: term.casefold() in report_text,
        ),
        "counterarguments": fraction(
            expected["required_counterarguments"],
            lambda term: term.casefold() in report_text,
        ),
        "rating_consistency": (
            Decimal("1")
            if (
                run.rating.value == expected["expected_rating"]
                and all(
                    term.casefold() in report_text
                    for term in expected["required_posture_terms"]
                )
            )
            else Decimal("0")
        ),
        "inference_disclosure": (
            Decimal("1")
            if (
                "inference is explicitly labeled" in report_text
                and any(
                    claim.kind is ClaimKind.INFERENCE
                    for claim in run.material_claims
                )
            )
            else Decimal("0")
        ),
        "cost": (
            Decimal("1")
            if run.paid_cost <= Decimal(expected["maximum_paid_cost_usd"])
            else Decimal("0")
        ),
    }
    minimum_scores = {
        name: Decimal(value) for name, value in expected["minimum_scores"].items()
    }
    failures = tuple(
        name for name, minimum in minimum_scores.items() if scores[name] < minimum
    )
    return GoldenEvaluation(
        scores=scores,
        minimum_scores=minimum_scores,
        failures=failures,
    )


def run_cost_simulation(tmp_path: Path, *, days: int) -> CostSimulationResult:
    store = ResearchStore(tmp_path / "cost.db")
    store.migrate()
    ids = iter(f"reservation-{index:02d}" for index in range(1, days + 1))
    ledger = BudgetLedger(
        store=store,
        soft_limit=Decimal("4.00"),
        hard_limit=Decimal("5.00"),
        id_factory=lambda: next(ids),
    )
    completed = 0
    deferred = 0
    for day in range(days):
        now = datetime(2026, 8, 1, 12, tzinfo=timezone.utc) + timedelta(days=day)
        try:
            reservation = ledger.reserve(
                f"daily-evidence-{day + 1}",
                Decimal("0.40"),
                now=now,
                role=AgentRole.EVIDENCE_ANALYST,
            )
            ledger.reconcile(reservation.id, Decimal("0.40"), now=now)
            completed += 1
        except (SoftBudgetExceeded, BudgetExceeded):
            deferred += 1
    return CostSimulationResult(
        paid_cost=ledger.month_total(2026, 8),
        deferred_tasks=deferred,
        completed_tasks=completed,
    )


def run_synthetic_end_to_end(tmp_path: Path) -> SyntheticEndToEndResult:
    """Exercise the production synthetic workflow while trapping external effects."""
    calls: list[str] = []
    order_or_trade_requests: list[str] = []
    email_attempts = 0
    config = ResearchConfig(
        enabled=True,
        data_dir=tmp_path,
        openai_api_key=None,
        inference_mode=InferenceMode.LOCAL_ONLY,
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1:8b",
        budget_soft_usd=Decimal("4.00"),
        budget_hard_usd=Decimal("5.00"),
    )
    store = ResearchStore(config.database_path)
    store.migrate()

    def forbidden_flex(_config: object) -> object:
        calls.append("ibkr-flex")
        raise AssertionError("synthetic dry run attempted IBKR Flex")

    def forbidden_http(*args: object, **kwargs: object) -> object:
        url = str(kwargs.get("url", args[1] if len(args) > 1 else "unknown"))
        calls.append(f"http:{url}")
        if "order" in url.casefold() or "trade" in url.casefold():
            order_or_trade_requests.append(url)
        raise AssertionError("synthetic dry run attempted HTTP")

    def forbidden_email(*_args: object, **_kwargs: object) -> object:
        nonlocal email_attempts
        email_attempts += 1
        calls.append("email")
        raise AssertionError("synthetic dry run attempted email")

    service = RuntimeWorkflowService(
        config,
        store,
        RuntimeFactories(
            flex_client_factory=forbidden_flex,
            owner_id_factory=lambda: "golden-e2e-owner",
        ),
    )
    homebrew_lib = Path("/opt/homebrew/lib")
    if homebrew_lib.is_dir():
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", str(homebrew_lib))
    with (
        patch("requests.sessions.Session.request", forbidden_http),
        patch("news_bot.email_client.send_email", forbidden_email),
        patch("smtplib.SMTP", forbidden_email),
    ):
        result = service.run_daily(
            as_of=datetime(2026, 8, 24, tzinfo=timezone.utc),
            dry_run=True,
            synthetic_portfolio=True,
        )

    html_paths = tuple(config.report_dir.glob("event_update-*.html"))
    pdf_paths = tuple(config.report_dir.glob("event_update-*.pdf"))
    if len(html_paths) != 1 or len(pdf_paths) != 1:
        raise AssertionError("synthetic runtime did not produce one HTML and one PDF")
    return SyntheticEndToEndResult(
        html_path=html_paths[0],
        pdf_path=pdf_paths[0],
        html=html_paths[0].read_text(encoding="utf-8"),
        workflow_status=result.status.value,
        omissions=result.omissions,
        external_calls=tuple(calls),
        order_or_trade_requests=tuple(order_or_trade_requests),
        email_attempts=email_attempts,
    )
