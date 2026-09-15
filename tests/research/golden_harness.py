"""Deterministic, offline acceptance harness for the research pipeline.

The fixtures contain test-authored paraphrases of public facts.  This module
deliberately composes production domain models, SQLite repositories, budget
controls, report rendering, and the synthetic runtime instead of replacing
those boundaries with a second fake implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

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
from news_bot.research.reports import (
    Citation,
    EventUpdate,
    ReportMetadata,
    ReportRenderer,
    ReportSection,
)
from news_bot.research.runtime import RuntimeFactories, RuntimeWorkflowService
from news_bot.research.store import DocumentPassageRecord, ResearchStore


GOLDEN_ROOT = Path(__file__).parent / "golden"
SOURCE_PACKET = GOLDEN_ROOT / "source_packet"


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
    with store.connect() as connection:
        rows = connection.execute(
            "SELECT document_id, source_type, canonical_url, publisher, "
            "published_at, retrieved_at, content_hash, raw_content_path, "
            "extraction_status FROM source_documents "
            "WHERE published_at <= ? ORDER BY published_at, document_id",
            (cutoff.isoformat(timespec="microseconds"),),
        ).fetchall()
    documents = tuple(
        SourceDocument(
            document_id=row[0],
            source_type=row[1],
            canonical_url=row[2],
            publisher=row[3],
            published_at=_parse_publication_date(row[4], document_id=row[0]),
            retrieved_at=_parse_publication_date(row[5], document_id=row[0]),
            content_hash=row[6],
            raw_content_path=row[7],
            extraction_status=row[8],
        )
        for row in rows
    )
    return ReplayResult(as_of=as_of, documents=documents)


def _seed_material_claims(
    store: ResearchStore, *, as_of: datetime
) -> tuple[MaterialClaim, ...]:
    definitions = (
        (
            "claim-nvda-supply",
            ClaimKind.FACT,
            "NVIDIA depends on third-party manufacturing and packaging capacity.",
            ("passage-nvda-supply",),
            (),
        ),
        (
            "claim-tsmc-foundry",
            ClaimKind.FACT,
            "TSMC is an upstream dedicated foundry for fabless designers.",
            ("passage-tsmc-foundry",),
            (),
        ),
        (
            "claim-capacity-bottleneck",
            ClaimKind.INFERENCE,
            "Advanced packaging capacity can constrain NVIDIA product availability.",
            ("passage-nvda-supply", "passage-tsmc-foundry"),
            ("claim-nvda-supply", "claim-tsmc-foundry"),
        ),
        (
            "claim-capacity-counterargument",
            ClaimKind.INFERENCE,
            "Additional foundry and packaging capacity may ease constraints.",
            ("passage-tsmc-capacity",),
            ("claim-capacity-bottleneck",),
        ),
        (
            "claim-conservative-rating",
            ClaimKind.INFERENCE,
            (
                "A low-liquidity account warrants HOLD, no margin, no forced "
                "liquidation, and human review."
            ),
            (
                "passage-nvda-supply",
                "passage-tsmc-foundry",
                "passage-tsmc-capacity",
            ),
            ("claim-capacity-bottleneck", "claim-capacity-counterargument"),
        ),
    )
    material = []
    for claim_id, kind, text, passage_ids, supporting_ids in definitions:
        claim = EvidenceClaim(
            claim_id=claim_id,
            entity_id=None,
            kind=kind,
            text=text,
            as_of=as_of,
            confidence=Decimal("0.90") if kind is ClaimKind.FACT else Decimal("0.70"),
            status="active",
        )
        store.insert_claim_with_lineage(
            claim,
            passage_links=tuple((item, "supports") for item in passage_ids),
            supporting_claim_ids=supporting_ids,
        )
        material.append(
            MaterialClaim(
                claim_id=claim_id,
                text=text,
                kind=kind,
                evidence_ids=passage_ids,
            )
        )
    return tuple(material)


def _golden_report(
    documents: tuple[SourceDocument, ...], as_of: datetime
) -> EventUpdate:
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

    def section(title: str, body: str, *evidence_ids: str) -> ReportSection:
        return ReportSection(title=title, body=body, evidence_ids=evidence_ids)

    all_evidence = tuple(evidence_to_document)
    return EventUpdate(
        metadata=ReportMetadata(
            report_id="golden-semiconductor-value-chain",
            report_type="event_update",
            title="Semiconductor value-chain research replay",
            as_of=as_of,
            inference_mode=InferenceMode.LOCAL_ONLY,
            provider="offline",
            model="deterministic-golden-v1",
            freshness=(
                "Only sources published on or before the replay date are eligible."
            ),
            citations=citations,
            methodology=(
                "Facts are fixture-backed. Inference is explicitly labeled and traced "
                "to cited passages; no future documents are available to analysis."
            ),
            omissions=("Live prices and valuation inputs are intentionally omitted.",),
            disclosure=(
                "HOLD is research for a low-liquidity account, not an order: no "
                "margin, no forced liquidation, and thin-liquidity ideas require "
                "human review."
            ),
        ),
        thesis=section(
            "Thesis",
            (
                "HOLD: supply-chain evidence is relevant, but it does not establish "
                "valuation upside."
            ),
            *all_evidence,
        ),
        event_decomposition=(
            section(
                "Documented structure",
                (
                    "NVIDIA depends on third-party manufacturing; TSMC sits "
                    "upstream as a foundry."
                ),
                "passage-nvda-supply",
                "passage-tsmc-foundry",
            ),
        ),
        causal_decomposition=(
            section(
                "Causal chain",
                (
                    "Third-party manufacturing and advanced packaging capacity "
                    "can limit product availability."
                ),
                "passage-nvda-supply",
                "passage-tsmc-foundry",
            ),
        ),
        read_through=(
            section(
                "One layer deeper",
                (
                    "Foundry and packaging capacity are upstream variables to "
                    "monitor for a fabless chip holding."
                ),
                "passage-tsmc-foundry",
                "passage-tsmc-capacity",
            ),
        ),
        thesis_changes=(
            section(
                "Rating discipline",
                (
                    "The rating remains HOLD until valuation and demand evidence "
                    "justify a change."
                ),
                *all_evidence,
            ),
        ),
        unchanged_assumptions=(
            section(
                "Counter-thesis",
                "Additional foundry and packaging capacity may ease constraints.",
                "passage-tsmc-capacity",
            ),
        ),
        questions=(
            section(
                "Open question",
                "Will capacity additions arrive before demand or product mix changes?",
                "passage-tsmc-capacity",
            ),
        ),
        signposts=(
            section(
                "Signposts",
                (
                    "Track foundry investment, packaging availability, and issuer "
                    "dependency disclosures."
                ),
                *all_evidence,
            ),
        ),
    )


def build_golden_run(tmp_path: Path) -> GoldenRun:
    expected = load_expected_characteristics()
    replay = run_historical_replay(
        tmp_path, as_of=date.fromisoformat(expected["as_of"])
    )
    store = ResearchStore(tmp_path / "replay.db")
    as_of = datetime.combine(replay.as_of, datetime.min.time(), tzinfo=timezone.utc)
    material_claims = _seed_material_claims(store, as_of=as_of)
    return GoldenRun(
        store=store,
        report=_golden_report(replay.documents, as_of),
        material_claims=material_claims,
        documents=replay.documents,
        rating=RecommendationRating.HOLD,
        paid_cost=Decimal("0.00"),
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

    def deterministic_pdf(destination: Path, _html: str) -> None:
        destination.write_bytes(
            b"%PDF-1.4\n% deterministic offline acceptance artifact\n%%EOF\n"
        )

    service = RuntimeWorkflowService(
        config,
        store,
        RuntimeFactories(
            flex_client_factory=forbidden_flex,
            owner_id_factory=lambda: "golden-e2e-owner",
        ),
    )
    with (
        patch("requests.sessions.Session.request", forbidden_http),
        patch("news_bot.email_client.send_email", forbidden_email),
        patch("smtplib.SMTP", forbidden_email),
        patch.object(
            ReportRenderer, "_atomic_pdf", staticmethod(deterministic_pdf)
        ),
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
