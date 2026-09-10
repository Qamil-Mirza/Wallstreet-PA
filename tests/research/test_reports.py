"""Institutional research report models, exhibits, and rendering."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from jinja2 import StrictUndefined
from pydantic import ValidationError

from news_bot.research.models import InferenceMode
from news_bot.research.reports.exhibits import (
    ExposureRow,
    ScenarioRow,
    ValuationRow,
    build_exposure_exhibit,
    build_scenario_matrix,
    build_valuation_exhibit,
)
from news_bot.research.reports.models import (
    Citation,
    EmergingCompanyMonitor,
    EventUpdate,
    IndustryLandscape,
    PortfolioBrief,
    ReportMetadata,
    ReportSection,
)
from news_bot.research.reports.renderer import ReportRenderer


NOW = datetime(2026, 9, 9, 12, tzinfo=timezone.utc)


def metadata(report_type: str) -> ReportMetadata:
    return ReportMetadata(
        report_id=f"RPT-{report_type}-001",
        report_type=report_type,
        title=f"Test {report_type}",
        as_of=NOW,
        inference_mode=InferenceMode.LOCAL_ONLY,
        provider="ollama",
        model="qwen3",
        freshness="Current through market close",
        citations=(
            Citation(
                evidence_id="E-001",
                source="Issuer filing",
                url="https://example.com/filing",
                source_date=date(2026, 9, 8),
                data_date=date(2026, 6, 30),
                content_hash="a" * 64,
            ),
        ),
        methodology="Evidence-weighted fundamental analysis.",
        omissions=("No management interview.",),
        disclosure="Research only; no order or transaction instruction.",
    )


def section(title: str, body: str = "Evidence E-001 supports this view.") -> ReportSection:
    return ReportSection(title=title, body=body, evidence_ids=("E-001",))


def event_report(*, title: str | None = None) -> EventUpdate:
    meta = metadata("event_update")
    if title is not None:
        meta = meta.model_copy(update={"title": title})
    return EventUpdate(
        metadata=meta,
        thesis=section("Thesis"),
        event_decomposition=(section("Event"),),
        causal_decomposition=(section("Causal chain"),),
        read_through=(section("Portfolio read-through"),),
        exhibits=(),
        thesis_changes=(section("Changed"),),
        unchanged_assumptions=(section("Unchanged"),),
        questions=(section("Questions"),),
        signposts=(section("Signposts"),),
    )


def portfolio_report() -> PortfolioBrief:
    return PortfolioBrief(
        metadata=metadata("portfolio_brief"),
        exposure_summary=(section("Rounded exposure summary"),),
        relevant_news=(section("Relevant news"),),
        value_chain_developments=(section("Value-chain developments"),),
        research_views=(section("Research views"),),
        change_history=(section("Change history"),),
        concentration_and_correlation=(section("Concentration and correlation"),),
    )


def industry_report() -> IndustryLandscape:
    return IndustryLandscape(
        metadata=metadata("industry_landscape"),
        value_chain=(section("Value chain"),),
        profit_pools=(section("Profit pools"),),
        bottlenecks=(section("Bottlenecks"),),
        emerging_technology_and_companies=(section("Emerging technology and companies"),),
        long_term_scenarios=(section("Base, upside, downside — 5–10 years"),),
        signposts=(section("Signposts"),),
        invalidation=(section("Invalidation"),),
        public_beneficiaries=(section("Public beneficiaries"),),
        threats=(section("Threats"),),
        portfolio_relevance=(section("Portfolio relevance"),),
    )


def emerging_report() -> EmergingCompanyMonitor:
    return EmergingCompanyMonitor(
        metadata=metadata("emerging_monitor"),
        company_and_technology_map=(section("Company and technology map"),),
        adoption_signals=(section("Evidence-backed adoption signals"),),
        confidence_and_limits=(section("Confidence and limits"),),
        public_market_translation=(section("Public-market translation"),),
        private_company_rating=None,
    )


def test_report_models_are_frozen_strict_and_forbid_sensitive_fields() -> None:
    report = event_report()
    with pytest.raises(ValidationError):
        report.metadata.title = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ReportMetadata(**{**metadata("event_update").model_dump(), "account_id": "secret"})
    with pytest.raises(ValidationError):
        ReportMetadata(**{**metadata("event_update").model_dump(), "as_of": NOW.isoformat()})
    with pytest.raises(ValidationError):
        Citation(**{**metadata("event_update").citations[0].model_dump(), "content_hash": "bad"})
    with pytest.raises(ValidationError):
        Citation(**{**metadata("event_update").citations[0].model_dump(), "url": "file:///etc/passwd"})


def test_report_type_must_match_specific_model() -> None:
    with pytest.raises(ValidationError):
        EventUpdate(
            **event_report().model_dump(exclude={"metadata"}),
            metadata=metadata("portfolio_brief"),
        )


def test_emerging_monitor_prohibits_private_company_rating() -> None:
    values = emerging_report().model_dump()
    values["private_company_rating"] = "buy"
    with pytest.raises(ValidationError):
        EmergingCompanyMonitor.model_validate(values)


def test_exposure_builder_formats_weight_and_orders_deterministically() -> None:
    rows = (
        ExposureRow(
            symbol="NVDA", label="NVIDIA", weight=Decimal("0.25"), currency="USD",
            source_note="Broker statement", source_date=date(2026, 9, 8),
        ),
        ExposureRow(
            symbol="CASH", label="Cash", weight=Decimal("0.75"), currency="USD",
            source_note="Broker statement", source_date=date(2026, 9, 8),
        ),
    )
    exhibit = build_exposure_exhibit(rows, declared_total=Decimal("1.0"))
    assert [row.symbol for row in exhibit.rows] == ["CASH", "NVDA"]
    assert exhibit.rows[1].display_weight == "25.0%"
    assert exhibit.source_notes == ("Broker statement (2026-09-08)",)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, 0.25])
def test_exhibit_rows_reject_non_decimal_numeric_values(bad: object) -> None:
    with pytest.raises(ValidationError):
        ExposureRow(
            symbol="NVDA", label="NVIDIA", weight=bad, currency="USD",  # type: ignore[arg-type]
            source_note="Statement", source_date=date(2026, 9, 8),
        )


def test_exposure_builder_reconciles_with_exact_tolerance() -> None:
    rows = (
        ExposureRow(symbol="A", label="A", weight=Decimal("0.4"), currency="USD", source_note="S", source_date=date(2026, 9, 8)),
        ExposureRow(symbol="B", label="B", weight=Decimal("0.5"), currency="USD", source_note="S", source_date=date(2026, 9, 8)),
    )
    with pytest.raises(ValueError, match="reconcile"):
        build_exposure_exhibit(rows, declared_total=Decimal("1"), tolerance=Decimal("0.000001"))
    with pytest.raises(ValueError, match="currency"):
        build_exposure_exhibit(
            rows + (ExposureRow(symbol="C", label="C", weight=Decimal("0.1"), currency="EUR", source_note="S", source_date=date(2026, 9, 8)),),
            declared_total=Decimal("1"),
        )


def test_valuation_and_scenario_builders_validate_units_and_values() -> None:
    valuation = build_valuation_exhibit((ValuationRow(
        label="EV / sales", low=Decimal("5"), base=Decimal("7"), high=Decimal("9"),
        currency="USD", unit="multiple", source_note="Model; Evidence E-001", source_date=date(2026, 9, 8),
    ),))
    assert valuation.rows[0].display_range == "5–9 multiple"
    matrix = build_scenario_matrix((ScenarioRow(
        scenario="Base", probability=Decimal("0.6"), value=Decimal("100"),
        currency="USD", unit="per_share", source_note="Model; Evidence E-001", source_date=date(2026, 9, 8),
    ),), declared_probability=Decimal("0.6"))
    assert matrix.rows[0].display_probability == "60.0%"
    with pytest.raises(ValidationError):
        ValuationRow(label="Bad", low=Decimal("9"), base=Decimal("7"), high=Decimal("5"), currency="USD", unit="multiple", source_note="S", source_date=date(2026, 9, 8))


@pytest.mark.parametrize(
    ("method", "report"),
    [
        ("render_event_update", event_report()),
        ("render_portfolio_brief", portfolio_report()),
        ("render_industry_landscape", industry_report()),
        ("render_emerging_monitor", emerging_report()),
    ],
)
def test_all_templates_include_audit_metadata_and_disclosure(
    tmp_path: Path, method: str, report: object
) -> None:
    artifact = getattr(ReportRenderer(tmp_path), method)(report)
    assert "Evidence E-001" in artifact.html
    assert "Inference mode: local_only" in artifact.html
    assert "Research only; no order or transaction instruction." in artifact.html
    assert "No management interview." in artifact.html
    assert "2026-09-08" in artifact.html
    assert artifact.pdf_path is not None
    assert artifact.pdf_path.read_bytes().startswith(b"%PDF")


def test_renderer_autoescapes_all_display_values(tmp_path: Path) -> None:
    artifact = ReportRenderer(tmp_path).render_event_update(
        event_report(title='<script>alert("x")</script>')
    )
    assert "<script>" not in artifact.html
    assert "&lt;script&gt;" in artifact.html


def test_renderer_uses_strict_undefined(tmp_path: Path) -> None:
    renderer = ReportRenderer(tmp_path)
    assert renderer.environment.undefined is StrictUndefined
    with pytest.raises(Exception):
        renderer.environment.from_string("{{ missing }}").render()


def test_renderer_contains_paths_and_sanitizes_filenames(tmp_path: Path) -> None:
    values = metadata("event_update").model_dump()
    values["report_id"] = "../escape"
    with pytest.raises(ValidationError):
        ReportMetadata.model_validate(values)
    artifact = ReportRenderer(tmp_path).render_event_update(event_report())
    assert artifact.html_path.parent == tmp_path.resolve()
    assert ".." not in artifact.html_path.name


def test_pdf_failure_retains_html_without_partial_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_pdf(*args: object, **kwargs: object) -> None:
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr("news_bot.research.reports.renderer.HTML.write_pdf", fail_pdf)
    artifact = ReportRenderer(tmp_path).render_event_update(event_report())
    assert artifact.html_path.exists()
    assert artifact.pdf_path is None
    assert "renderer unavailable" in (artifact.pdf_error or "")
    assert list(tmp_path.glob("*.pdf")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_remote_resource_fetcher_blocks_network(tmp_path: Path) -> None:
    renderer = ReportRenderer(tmp_path)
    with pytest.raises(ValueError, match="remote"):
        renderer.url_fetcher("https://tracker.example/pixel.png")
