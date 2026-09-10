"""Institutional research report models, exhibits, and rendering."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import builtins
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest
from jinja2 import StrictUndefined
from pydantic import ValidationError

from news_bot.research.models import InferenceMode, RecommendationRating
from news_bot.research.quality import CalculatedExhibit, CalculatedRow
from news_bot.research.reports import models as report_models
from news_bot.research.reports.exhibits import (
    ExposureExhibit,
    ExposureRow,
    ScenarioRow,
    ValuationRow,
    build_exposure_exhibit,
    build_scenario_matrix,
    build_valuation_exhibit,
)
from news_bot.research.reports.models import (
    Citation,
    ConcentrationCorrelation,
    EmergingCompanyMonitor,
    EventUpdate,
    IndustryLandscape,
    PortfolioBrief,
    PortfolioNewsItem,
    ReportMetadata,
    ReportSection,
    ResearchView,
    ResearchViewChange,
    RoundedExposure,
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
        exposure_summary=(
            RoundedExposure(
                symbol="NVDA",
                label="NVIDIA",
                weight_band="20%+",
                rounded_weight_percent=Decimal("25"),
                evidence_ids=("E-001",),
            ),
        ),
        relevant_news=(
            PortfolioNewsItem(
                headline="New platform",
                thesis_impact="Supports demand durability",
                evidence_ids=("E-001",),
            ),
        ),
        value_chain_developments=(section("Value-chain developments"),),
        research_views=(
            ResearchView(
                symbol="NVDA",
                thesis="Demand remains durable",
                rating=RecommendationRating.HOLD,
                evidence_ids=("E-001",),
            ),
        ),
        change_history=(
            ResearchViewChange(
                symbol="NVDA",
                previous_rating=RecommendationRating.BUY,
                new_rating=RecommendationRating.HOLD,
                rationale="Valuation now balanced",
                changed_on=date(2026, 9, 9),
                evidence_ids=("E-001",),
            ),
        ),
        concentration_and_correlation=ConcentrationCorrelation(
            summary="High semiconductor concentration",
            risk_level="high",
            evidence_ids=("E-001",),
        ),
    )


def industry_report() -> IndustryLandscape:
    return IndustryLandscape(
        metadata=metadata("industry_landscape"),
        industry_definition=section("Industry definition"),
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
    authority = calculation(
        "exposure",
        tuple(
            CalculatedRow(row_id=row.symbol, values={"weight": row.weight})
            for row in rows
        ),
    )
    exhibit = build_exposure_exhibit(
        rows, calculation=authority, evidence_ids=("E-001",)
    )
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


def test_exposure_builder_requires_fixed_total_and_one_currency() -> None:
    rows = (
        ExposureRow(symbol="A", label="A", weight=Decimal("0.4"), currency="USD", source_note="S", source_date=date(2026, 9, 8)),
        ExposureRow(symbol="B", label="B", weight=Decimal("0.5"), currency="USD", source_note="S", source_date=date(2026, 9, 8)),
    )
    authority = calculation(
        "exposure",
        tuple(
            CalculatedRow(row_id=row.symbol, values={"weight": row.weight})
            for row in rows
        ),
    )
    with pytest.raises(ValueError, match="reconcile"):
        build_exposure_exhibit(
            rows, calculation=authority, evidence_ids=("E-001",)
        )
    with pytest.raises(ValueError, match="currency"):
        mixed_rows = rows + (
            ExposureRow(symbol="C", label="C", weight=Decimal("0.1"), currency="EUR", source_note="S", source_date=date(2026, 9, 8)),
        )
        build_exposure_exhibit(mixed_rows, calculation=calculation(
            "exposure",
            tuple(CalculatedRow(row_id=row.symbol, values={"weight": row.weight}) for row in mixed_rows),
        ), evidence_ids=("E-001",))


def test_valuation_and_scenario_builders_validate_units_and_values() -> None:
    valuation_row = ValuationRow(
        label="EV / sales", low=Decimal("5"), base=Decimal("7"), high=Decimal("9"),
        currency="USD", unit="multiple", source_note="Model; Evidence E-001", source_date=date(2026, 9, 8),
    )
    valuation = build_valuation_exhibit(
        (valuation_row,),
        calculation=calculation("valuation", (
            CalculatedRow(row_id="EV-sales", values={"low": Decimal("5"), "base": Decimal("7"), "high": Decimal("9")}),
        )),
        evidence_ids=("E-001",),
    )
    assert valuation.rows[0].display_range == "5–9 multiple"
    scenario_row = ScenarioRow(
        scenario="Base", probability=Decimal("0.6"), value=Decimal("100"),
        currency="USD", unit="per_share", source_note="Model; Evidence E-001", source_date=date(2026, 9, 8),
    )
    downside_row = ScenarioRow(
        scenario="Downside", probability=Decimal("0.4"), value=Decimal("70"),
        currency="USD", unit="per_share", source_note="Model; Evidence E-001", source_date=date(2026, 9, 8),
    )
    scenario_rows = (scenario_row, downside_row)
    matrix = build_scenario_matrix(
        scenario_rows,
        calculation=calculation("scenario", tuple(
            CalculatedRow(row_id=row.scenario, values={"probability": row.probability, "value": row.value})
            for row in scenario_rows
        )),
        evidence_ids=("E-001",),
    )
    assert next(row for row in matrix.rows if row.scenario == "Base").display_probability == "60.0%"
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

    monkeypatch.setattr(ReportRenderer, "_atomic_pdf", fail_pdf)
    artifact = ReportRenderer(tmp_path).render_event_update(event_report())
    assert artifact.html_path.exists()
    assert artifact.pdf_path is None
    assert "renderer unavailable" in (artifact.pdf_error or "")
    assert list(tmp_path.glob("*.pdf")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_pdf_failure_removes_stale_deterministic_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = ReportRenderer(tmp_path)
    report = event_report()
    stale_pdf = tmp_path / (
        renderer._filename(
            report.metadata.report_type,
            report.metadata.report_id,
            report.metadata.as_of,
        )
        + ".pdf"
    )
    stale_pdf.write_bytes(b"%PDF-stale")

    def fail_pdf(*args: object, **kwargs: object) -> None:
        raise RuntimeError("failed")

    monkeypatch.setattr(ReportRenderer, "_atomic_pdf", fail_pdf)
    artifact = renderer.render_event_update(report)

    assert artifact.html_path.exists()
    assert artifact.pdf_path is None
    assert not stale_pdf.exists()


def test_remote_resource_fetcher_blocks_network(tmp_path: Path) -> None:
    renderer = ReportRenderer(tmp_path)
    with pytest.raises(ValueError, match="remote"):
        renderer.url_fetcher("https://tracker.example/pixel.png")


@pytest.mark.parametrize(
    "private_text",
    [
        "api_key=sk-live-secret",
        "Authorization: Bearer bearer-secret",
        "Raw brokerage account U1234567",
        "account_id=12345678",
        "Account number is 12345678",
        "NAV: USD 1,234.56",
        "cash balance = $1200.25",
        "position value: USD 9876.54",
    ],
)
def test_all_display_text_rejects_private_material(private_text: str) -> None:
    with pytest.raises(ValidationError, match="private|sensitive"):
        ReportSection(title="Analysis", body=private_text, evidence_ids=("E-001",))
    citation_values = metadata("event_update").citations[0].model_dump()
    citation_values["source"] = private_text
    with pytest.raises(ValidationError, match="private|sensitive"):
        Citation.model_validate(citation_values)


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.com/source",
        "https://example.com/source?api_key=secret",
        "https://example.com/source?access_token=secret",
        "https://example.com/source?account_id=12345678",
        "https://example.com/source?next=api_key%3Dsecret",
        "https://example.com/source?next=NAV%3DUSD%201234.56",
        "https://example.com/source#token=secret",
        "https://example.com/source#access_token%3Dsecret",
    ],
)
def test_citation_url_rejects_embedded_credentials(url: str) -> None:
    values = metadata("event_update").citations[0].model_dump()
    values["url"] = url
    with pytest.raises(ValidationError, match="credential|sensitive"):
        Citation.model_validate(values)


def test_renderer_revalidates_copied_models_at_privacy_boundary(tmp_path: Path) -> None:
    unsafe = event_report().model_copy(
        update={"thesis": section("Thesis").model_copy(update={"body": "token=secret"})}
    )
    with pytest.raises(ValidationError, match="private|sensitive"):
        ReportRenderer(tmp_path).render_event_update(unsafe)


def test_metadata_accepts_actual_research_config_model_name() -> None:
    values = metadata("event_update").model_dump()
    values["model"] = "library/llama3.1:8b"
    assert ReportMetadata.model_validate(values).model == "library/llama3.1:8b"


@pytest.mark.parametrize(
    "model_name", ("provider//model", "bad model", "<model>", "token=secret")
)
def test_metadata_rejects_unsafe_model_names(model_name: str) -> None:
    values = metadata("event_update").model_dump()
    values["model"] = model_name
    with pytest.raises(ValidationError):
        ReportMetadata.model_validate(values)


def test_display_privacy_filter_allows_legitimate_valuation_language() -> None:
    section_value = ReportSection(
        title="Valuation",
        body="NAV multiple: 5x; valuation range is USD 70–100 per share.",
        evidence_ids=("E-001",),
    )
    assert "5x" in section_value.body


def test_material_sections_require_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        ReportSection(title="Uncited thesis", body="Material factual claim")


def test_all_referenced_evidence_must_be_declared_in_metadata() -> None:
    values = event_report().model_dump()
    values["thesis"]["evidence_ids"] = ("E-999",)
    with pytest.raises(ValidationError, match="absent from citations"):
        EventUpdate.model_validate(values)


def test_industry_definition_is_required() -> None:
    values = industry_report().model_dump()
    values.pop("industry_definition")
    with pytest.raises(ValidationError, match="industry_definition"):
        IndustryLandscape.model_validate(values)


def test_industry_definition_renders_before_value_chain(tmp_path: Path) -> None:
    html = ReportRenderer(tmp_path).render_industry_landscape(industry_report()).html
    assert html.index("Industry definition") < html.index("Value chain")


def calculation(
    exhibit_id: str, rows: tuple[CalculatedRow, ...], rendered: tuple[CalculatedRow, ...] | None = None
) -> CalculatedExhibit:
    return CalculatedExhibit(
        exhibit_id=exhibit_id,
        normalized_rows=rows,
        rendered_rows=rows if rendered is None else rendered,
    )


def test_exposure_builder_requires_reconciled_authoritative_rows() -> None:
    rows = (
        ExposureRow(symbol="NVDA", label="NVIDIA", weight=Decimal("0.25"), currency="USD", source_note="Statement", source_date=date(2026, 9, 8)),
        ExposureRow(symbol="OTHER", label="Other", weight=Decimal("0.75"), currency="USD", source_note="Statement", source_date=date(2026, 9, 8)),
    )
    authority_rows = tuple(
        CalculatedRow(row_id=row.symbol, values={"weight": row.weight}) for row in rows
    )
    exhibit = build_exposure_exhibit(
        rows,
        calculation=calculation("exposure", authority_rows),
        evidence_ids=("E-001",),
    )
    assert exhibit.verified_reconciliation is True
    assert exhibit.rows[1].display_weight == "75.0%"

    wrong = calculation(
        "exposure",
        authority_rows,
        rendered=(
            CalculatedRow(row_id="NVDA", values={"weight": Decimal("0.26")}),
            authority_rows[1],
        ),
    )
    with pytest.raises((ValueError, ValidationError), match="reconcile"):
        build_exposure_exhibit(rows, calculation=wrong, evidence_ids=("E-001",))


def test_exhibit_rejects_stored_vs_rendered_row_mismatch_and_direct_bypass() -> None:
    row = ExposureRow(symbol="NVDA", label="NVIDIA", weight=Decimal("1"), currency="USD", source_note="Statement", source_date=date(2026, 9, 8))
    mismatched = calculation(
        "exposure",
        (CalculatedRow(row_id="NVDA", values={"weight": Decimal("0.9")}),),
    )
    with pytest.raises(ValidationError, match="authoritative|reconcile"):
        ExposureExhibit(
            title="Exposure", source_notes=("Statement (2026-09-08)",),
            evidence_ids=("E-001",), rows=(row,), total_weight=Decimal("1"),
            calculation=mismatched, verified_reconciliation=True,
        )


def test_event_renderer_preserves_typed_exhibit_rows_and_citation(tmp_path: Path) -> None:
    row = ExposureRow(
        symbol="NVDA", label="NVIDIA", weight=Decimal("1"), currency="USD",
        source_note="Statement", source_date=date(2026, 9, 8),
    )
    authority = calculation("exposure", (
        CalculatedRow(row_id="NVDA", values={"weight": Decimal("1")}),
    ))
    exhibit = build_exposure_exhibit(
        (row,), calculation=authority, evidence_ids=("E-001",)
    )
    report = event_report().model_copy(update={"exhibits": (exhibit,)})

    artifact = ReportRenderer(tmp_path).render_event_update(report)

    assert "100.0%" in artifact.html
    assert "Statement (2026-09-08)" in artifact.html
    assert "Exhibit evidence: E-001" in artifact.html


def test_renderer_revalidates_copied_exhibit_at_authority_boundary(tmp_path: Path) -> None:
    row = ExposureRow(
        symbol="NVDA", label="NVIDIA", weight=Decimal("1"), currency="USD",
        source_note="Statement", source_date=date(2026, 9, 8),
    )
    authority = calculation("exposure", (
        CalculatedRow(row_id="NVDA", values={"weight": Decimal("1")}),
    ))
    exhibit = build_exposure_exhibit(
        (row,), calculation=authority, evidence_ids=("E-001",)
    ).model_copy(update={"total_weight": Decimal("0.9")})
    report = event_report().model_copy(update={"exhibits": (exhibit,)})

    with pytest.raises(ValidationError, match="reconcile"):
        ReportRenderer(tmp_path).render_event_update(report)


def test_valuation_and_scenario_builders_reconcile_authoritative_values() -> None:
    valuation_row = ValuationRow(
        label="EV sales", low=Decimal("5"), base=Decimal("7"), high=Decimal("9"),
        currency="USD", unit="multiple", source_note="Model", source_date=date(2026, 9, 8),
    )
    valuation_authority = calculation("valuation", (
        CalculatedRow(row_id="EV-sales", values={"low": Decimal("5"), "base": Decimal("7"), "high": Decimal("9")}),
    ))
    assert build_valuation_exhibit(
        (valuation_row,), calculation=valuation_authority, evidence_ids=("E-001",)
    ).verified_reconciliation

    scenario_rows = (
        ScenarioRow(scenario="Base", probability=Decimal("0.6"), value=Decimal("100"), currency="USD", unit="per_share", source_note="Model", source_date=date(2026, 9, 8)),
        ScenarioRow(scenario="Downside", probability=Decimal("0.4"), value=Decimal("70"), currency="USD", unit="per_share", source_note="Model", source_date=date(2026, 9, 8)),
    )
    scenario_authority = calculation("scenario", tuple(
        CalculatedRow(row_id=row.scenario, values={"probability": row.probability, "value": row.value}) for row in scenario_rows
    ))
    assert build_scenario_matrix(
        scenario_rows, calculation=scenario_authority, evidence_ids=("E-001",)
    ).verified_reconciliation


def test_uncited_exhibit_is_rejected() -> None:
    row = ExposureRow(symbol="NVDA", label="NVIDIA", weight=Decimal("1"), currency="USD", source_note="Statement", source_date=date(2026, 9, 8))
    authority = calculation("exposure", (
        CalculatedRow(row_id="NVDA", values={"weight": Decimal("1")}),
    ))
    with pytest.raises((TypeError, ValueError, ValidationError), match="evidence"):
        build_exposure_exhibit((row,), calculation=authority, evidence_ids=())


def test_typed_portfolio_brief_exposes_only_rounded_research_views() -> None:
    RoundedExposure = getattr(report_models, "RoundedExposure")
    PortfolioNewsItem = getattr(report_models, "PortfolioNewsItem")
    ResearchView = getattr(report_models, "ResearchView")
    ResearchViewChange = getattr(report_models, "ResearchViewChange")
    ConcentrationCorrelation = getattr(report_models, "ConcentrationCorrelation")
    from news_bot.research.models import RecommendationRating

    report = PortfolioBrief(
        metadata=metadata("portfolio_brief"),
        exposure_summary=(RoundedExposure(symbol="NVDA", label="NVIDIA", weight_band="20%+", rounded_weight_percent=Decimal("25"), evidence_ids=("E-001",)),),
        relevant_news=(PortfolioNewsItem(headline="New platform", thesis_impact="Supports demand durability", evidence_ids=("E-001",)),),
        value_chain_developments=(section("Value chain"),),
        research_views=(ResearchView(symbol="NVDA", thesis="Demand remains durable", rating=RecommendationRating.HOLD, evidence_ids=("E-001",)),),
        change_history=(ResearchViewChange(symbol="NVDA", previous_rating=RecommendationRating.BUY, new_rating=RecommendationRating.HOLD, rationale="Valuation now balanced", changed_on=date(2026, 9, 9), evidence_ids=("E-001",)),),
        concentration_and_correlation=ConcentrationCorrelation(summary="High semiconductor concentration", risk_level="high", evidence_ids=("E-001",)),
    )
    html = ReportRenderer(Path(os.environ.get("TMPDIR", "/tmp")) / "typed-report-test").render_portfolio_brief(report).html
    assert "25%" in html
    assert "Supports demand durability" in html
    assert "hold" in html

    with pytest.raises(ValidationError):
        RoundedExposure(symbol="NVDA", label="NVIDIA", weight_band="20%+", rounded_weight_percent=Decimal("25.4"), evidence_ids=("E-001",), market_value=Decimal("1000"))


def test_reports_import_without_pdf_backend(tmp_path: Path) -> None:
    blocker = tmp_path / "weasyprint.py"
    blocker.write_text("raise ImportError('backend unavailable')\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(Path.cwd())))
    result = subprocess.run(
        [sys.executable, "-c", "import news_bot.research.reports; print('models available')"],
        text=True, capture_output=True, env=env, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "models available" in result.stdout


def test_pdf_backend_import_failure_retains_html(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def unavailable(name: str, *args: object, **kwargs: object) -> object:
        if name == "weasyprint":
            raise ImportError("backend unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    artifact = ReportRenderer(tmp_path).render_event_update(event_report())
    assert artifact.html_path.exists()
    assert artifact.pdf_path is None
    assert "backend unavailable" in (artifact.pdf_error or "")
    assert not tuple(tmp_path.glob("*.pdf"))


def test_wheel_configuration_includes_every_report_template() -> None:
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["setuptools"]["package-data"]["news_bot.research.reports"] == ["templates/*.html"]
    assert {path.name for path in Path("news_bot/research/reports/templates").glob("*.html")} == {
        "base.html", "event_update.html", "portfolio_brief.html",
        "industry_landscape.html", "emerging_monitor.html",
    }
