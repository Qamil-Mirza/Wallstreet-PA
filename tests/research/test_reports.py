"""Institutional research report models, exhibits, and rendering."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import builtins
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
from urllib.parse import quote

import pytest
from jinja2 import StrictUndefined
from pydantic import ValidationError

from news_bot.research.models import InferenceMode, RecommendationRating
from news_bot.research.quality import CalculatedExhibit, CalculatedRow
from news_bot.research.reports import exhibits as exhibit_models
from news_bot.research.reports import models as report_models
from news_bot.research.reports.exhibits import (
    ExposureExhibit,
    ExposureRow,
    ScenarioRow,
    StoredExposureRow,
    StoredScenarioRow,
    StoredValuationRow,
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
PRIVACY_AMOUNT_SIGNS = ("+", "-", "−", "＋", "－", "﹢", "﹣")


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


def publication_privacy_context(**updates: object) -> object:
    context_type = getattr(report_models, "PublicationPrivacyContext")
    values: dict[str, object] = {
        "mode": "no_sensitive_data",
        "sensitive_literals": (),
        "account_identifiers": (),
        "portfolio_values": (),
    }
    values.update(updates)
    if "mode" not in updates and any(
        values[field]
        for field in (
            "sensitive_literals",
            "account_identifiers",
            "portfolio_values",
        )
    ):
        values["mode"] = "enforced"
    return context_type(**values)


def renderer(output_dir: Path) -> ReportRenderer:
    return ReportRenderer(
        output_dir, privacy_context=publication_privacy_context()
    )


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
        StoredExposureRow(
            row_id="NVDA",
            symbol="NVDA", label="NVIDIA", weight=Decimal("0.25"), currency="USD",
            source_note="Broker statement", source_date=date(2026, 9, 8),
        ),
        StoredExposureRow(
            row_id="CASH",
            symbol="CASH", label="Cash", weight=Decimal("0.75"), currency="USD",
            source_note="Broker statement", source_date=date(2026, 9, 8),
        ),
    )
    authority = calculation(
        "exposure",
        tuple(
            CalculatedRow(row_id=row.row_id, values={"weight": row.weight})
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
            row_id="NVDA",
            symbol="NVDA", label="NVIDIA", weight=bad, currency="USD",  # type: ignore[arg-type]
            source_note="Statement", source_date=date(2026, 9, 8),
        )


def test_exposure_builder_requires_fixed_total_and_one_currency() -> None:
    rows = (
        StoredExposureRow(row_id="A", symbol="A", label="A", weight=Decimal("0.4"), currency="USD", source_note="S", source_date=date(2026, 9, 8)),
        StoredExposureRow(row_id="B", symbol="B", label="B", weight=Decimal("0.5"), currency="USD", source_note="S", source_date=date(2026, 9, 8)),
    )
    authority = calculation(
        "exposure",
        tuple(
            CalculatedRow(row_id=row.row_id, values={"weight": row.weight})
            for row in rows
        ),
    )
    with pytest.raises(ValueError, match="reconcile"):
        build_exposure_exhibit(
            rows, calculation=authority, evidence_ids=("E-001",)
        )
    with pytest.raises(ValueError, match="currency"):
        mixed_rows = rows + (
            StoredExposureRow(row_id="C", symbol="C", label="C", weight=Decimal("0.1"), currency="EUR", source_note="S", source_date=date(2026, 9, 8)),
        )
        build_exposure_exhibit(mixed_rows, calculation=calculation(
            "exposure",
            tuple(CalculatedRow(row_id=row.row_id, values={"weight": row.weight}) for row in mixed_rows),
        ), evidence_ids=("E-001",))


def test_valuation_and_scenario_builders_validate_units_and_values() -> None:
    valuation_row = StoredValuationRow(
        row_id="EV-sales",
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
    scenario_row = StoredScenarioRow(
        row_id="Base",
        scenario="Base", probability=Decimal("0.6"), value=Decimal("100"),
        currency="USD", unit="per_share", source_note="Model; Evidence E-001", source_date=date(2026, 9, 8),
    )
    downside_row = StoredScenarioRow(
        row_id="Downside",
        scenario="Downside", probability=Decimal("0.4"), value=Decimal("70"),
        currency="USD", unit="per_share", source_note="Model; Evidence E-001", source_date=date(2026, 9, 8),
    )
    scenario_rows = (scenario_row, downside_row)
    matrix = build_scenario_matrix(
        scenario_rows,
        calculation=calculation("scenario", tuple(
            CalculatedRow(row_id=row.row_id, values={"probability": row.probability, "value": row.value})
            for row in scenario_rows
        )),
        evidence_ids=("E-001",),
    )
    assert next(row for row in matrix.rows if row.scenario == "Base").display_probability == "60.0%"
    with pytest.raises(ValidationError):
        ValuationRow(row_id="Bad", label="Bad", low=Decimal("9"), base=Decimal("7"), high=Decimal("5"), currency="USD", unit="multiple", source_note="S", source_date=date(2026, 9, 8))


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
    artifact = getattr(renderer(tmp_path), method)(report)
    assert "Evidence E-001" in artifact.html
    assert "Inference mode: local_only" in artifact.html
    assert "Research only; no order or transaction instruction." in artifact.html
    assert "No management interview." in artifact.html
    assert "2026-09-08" in artifact.html
    assert artifact.pdf_path is not None
    assert artifact.pdf_path.read_bytes().startswith(b"%PDF")


def test_renderer_autoescapes_all_display_values(tmp_path: Path) -> None:
    artifact = renderer(tmp_path).render_event_update(
        event_report(title='<script>alert("x")</script>')
    )
    assert "<script>" not in artifact.html
    assert "&lt;script&gt;" in artifact.html


def test_renderer_uses_strict_undefined(tmp_path: Path) -> None:
    report_renderer = renderer(tmp_path)
    assert report_renderer.environment.undefined is StrictUndefined
    with pytest.raises(Exception):
        report_renderer.environment.from_string("{{ missing }}").render()


def test_renderer_contains_paths_and_sanitizes_filenames(tmp_path: Path) -> None:
    values = metadata("event_update").model_dump()
    values["report_id"] = "../escape"
    with pytest.raises(ValidationError):
        ReportMetadata.model_validate(values)
    artifact = renderer(tmp_path).render_event_update(event_report())
    assert artifact.html_path.parent == tmp_path.resolve()
    assert ".." not in artifact.html_path.name


def test_pdf_failure_retains_html_without_partial_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_pdf(*args: object, **kwargs: object) -> None:
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(ReportRenderer, "_atomic_pdf", fail_pdf)
    artifact = renderer(tmp_path).render_event_update(event_report())
    assert artifact.html_path.exists()
    assert artifact.pdf_path is None
    assert artifact.pdf_error == "pdf_render_failed"
    assert list(tmp_path.glob("*.pdf")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_pdf_failure_removes_stale_deterministic_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_renderer = renderer(tmp_path)
    report = event_report()
    stale_pdf = tmp_path / (
        report_renderer._filename(
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
    artifact = report_renderer.render_event_update(report)

    assert artifact.html_path.exists()
    assert artifact.pdf_path is None
    assert not stale_pdf.exists()


def test_remote_resource_fetcher_blocks_network(tmp_path: Path) -> None:
    report_renderer = renderer(tmp_path)
    with pytest.raises(ValueError, match="remote"):
        report_renderer.url_fetcher("https://tracker.example/pixel.png")


@pytest.mark.parametrize(
    "private_text",
    [
        "api_key=sk-live-secret",
        "Authorization: Bearer bearer-secret",
        "Raw brokerage account U1234567",
        "account_id=12345678",
        "Account number is 12345678",
        "The account identifier is 12345678",
        "Account ID: 12 345 678",
        "NAV: USD 1,234.56",
        "NAV totals USD 1,234.56",
        "NAV: USD .50",
        "cash balance = $1200.25",
        "position value: USD 9876.54",
        "An arbitrary portfolio amount is USD .50",
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
        "https://example.com/source?X-Amz-Signature=secret",
        "https://example.com/source?X-Goog-Signature=secret",
        "https://example.com/source?sig=secret",
        "https://example.com/source?se=2026-09-09&sp=r&sv=1",
        "https://example.com/source?credential=secret",
        "https://example.com/source?%58-Amz-Signature=secret",
        "https://example.com/source?benign=tracking",
        "https://example.com/source#section-1",
        "https://example.com/source?",
        "https://example.com/source#",
    ],
)
def test_citation_url_rejects_embedded_credentials(url: str) -> None:
    values = metadata("event_update").citations[0].model_dump()
    values["url"] = url
    with pytest.raises(ValidationError, match="canonical|credential|sensitive"):
        Citation.model_validate(values)


def test_renderer_revalidates_copied_models_at_privacy_boundary(tmp_path: Path) -> None:
    unsafe = event_report().model_copy(
        update={"thesis": section("Thesis").model_copy(update={"body": "token=secret"})}
    )
    with pytest.raises(ValidationError, match="private|sensitive"):
        renderer(tmp_path).render_event_update(unsafe)


def test_renderer_requires_explicit_publication_privacy_context(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="privacy_context"):
        ReportRenderer(tmp_path)


def test_publication_privacy_context_is_strict_and_frozen() -> None:
    context_type = getattr(report_models, "PublicationPrivacyContext")
    context = publication_privacy_context(sensitive_literals=("opaque-secret",))
    with pytest.raises(ValidationError):
        context_type(
            mode="enforced",
            sensitive_literals=["opaque-secret"],
            account_identifiers=(),
            portfolio_values=(),
        )
    with pytest.raises(ValidationError):
        context.sensitive_literals = ()


def test_publication_privacy_context_requires_affirmative_consistent_mode() -> None:
    context_type = getattr(report_models, "PublicationPrivacyContext")
    no_sensitive = context_type(
        mode="no_sensitive_data",
        sensitive_literals=(),
        account_identifiers=(),
        portfolio_values=(),
    )
    enforced = context_type(
        mode="enforced",
        sensitive_literals=("private-marker",),
        account_identifiers=(),
        portfolio_values=(),
    )
    assert no_sensitive.mode == "no_sensitive_data"
    assert enforced.mode == "enforced"
    with pytest.raises(ValidationError, match="enforced|sensitive"):
        context_type(
            mode="enforced",
            sensitive_literals=(),
            account_identifiers=(),
            portfolio_values=(),
        )
    with pytest.raises(ValidationError, match="no_sensitive_data|empty"):
        context_type(
            mode="no_sensitive_data",
            sensitive_literals=("private-marker",),
            account_identifiers=(),
            portfolio_values=(),
        )


@pytest.mark.parametrize(
    "private_body",
    (
        "Operational note blue heron remains pending.",
        "Reference code 12-345 678 is internal.",
        "An arbitrary metric is USD .50.",
        "An arbitrary metric is (USD 1,234.56).",
    ),
)
def test_runtime_privacy_context_blocks_registered_values_before_html_write(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    private_body: str,
) -> None:
    context = publication_privacy_context(
        sensitive_literals=("blue heron",),
        account_identifiers=("12345678",),
        portfolio_values=(Decimal(".50"), Decimal("1234.56")),
    )
    unsafe = event_report().model_copy(
        update={
            "thesis": section("Thesis").model_copy(update={"body": private_body})
        }
    )
    publication_error = getattr(report_models, "ReportPublicationError")

    with pytest.raises(publication_error) as exc_info:
        ReportRenderer(tmp_path, privacy_context=context).render_event_update(unsafe)

    assert str(exc_info.value) == "report publication blocked by privacy policy"
    assert "blue heron" not in caplog.text
    assert "12345678" not in caplog.text
    assert not tuple(tmp_path.glob("*.html"))


def test_runtime_privacy_context_allows_unrelated_valuation_prose(
    tmp_path: Path,
) -> None:
    context = publication_privacy_context(
        sensitive_literals=("unrelated-secret",),
        account_identifiers=("87654321",),
        portfolio_values=(Decimal("999.99"),),
    )
    safe = event_report().model_copy(
        update={
            "thesis": section(
                "Thesis",
                "NAV was -20%; NAV multiple was 5x; valuation range is USD 70–100.",
            )
        }
    )

    artifact = ReportRenderer(
        tmp_path, privacy_context=context
    ).render_event_update(safe)

    assert "NAV was -20%" in artifact.html


def test_runtime_privacy_context_scans_nested_metadata_and_percent_encoding(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    context = publication_privacy_context(sensitive_literals=("blue heron",))
    unsafe_metadata = event_report().metadata.model_copy(
        update={"title": "Review of blue%20heron operations"}
    )
    unsafe = event_report().model_copy(update={"metadata": unsafe_metadata})
    publication_error = getattr(report_models, "ReportPublicationError")

    with pytest.raises(publication_error) as exc_info:
        ReportRenderer(tmp_path, privacy_context=context).render_event_update(unsafe)

    assert str(exc_info.value) == "report publication blocked by privacy policy"
    assert "blue heron" not in caplog.text
    assert not tuple(tmp_path.glob("*.html"))


def test_runtime_privacy_context_scans_exhibit_strings_before_write(
    tmp_path: Path,
) -> None:
    stored = StoredExposureRow(
        row_id="NVDA",
        symbol="NVDA",
        label="NVIDIA",
        weight=Decimal("1"),
        currency="USD",
        source_note="blue heron statement",
        source_date=date(2026, 9, 8),
    )
    authority = calculation(
        "exposure",
        (CalculatedRow(row_id="NVDA", values={"weight": Decimal("1")}),),
    )
    exhibit = build_exposure_exhibit(
        (stored,), calculation=authority, evidence_ids=("E-001",)
    )
    unsafe = event_report().model_copy(update={"exhibits": (exhibit,)})
    publication_error = getattr(report_models, "ReportPublicationError")

    with pytest.raises(publication_error, match="publication blocked"):
        ReportRenderer(
            tmp_path,
            privacy_context=publication_privacy_context(
                sensitive_literals=("blue heron",)
            ),
        ).render_event_update(unsafe)

    assert not tuple(tmp_path.glob("*.html"))


def test_runtime_privacy_context_fails_closed_on_exact_typed_numeric_collision(
    tmp_path: Path,
) -> None:
    publication_error = getattr(report_models, "ReportPublicationError")

    with pytest.raises(publication_error, match="publication blocked"):
        ReportRenderer(
            tmp_path,
            privacy_context=publication_privacy_context(
                portfolio_values=(Decimal("25"),)
            ),
        ).render_portfolio_brief(portfolio_report())

    assert not tuple(tmp_path.glob("*.html"))


@pytest.mark.parametrize(
    "formatted_value",
    (
        "1 234.56",
        "1\u00a0234.56",
        "1\u202f234.56",
        "1\u2009234.56",
        "1,234.56",
        "+1 234.56",
        "−1\u202f234.56",
        "(1 234.56)",
        "USD 1 234.56",
        "1.234,56",
    ),
)
def test_runtime_privacy_context_blocks_grouped_registered_decimal_forms(
    tmp_path: Path,
    formatted_value: str,
) -> None:
    unsafe = event_report().model_copy(
        update={
            "thesis": section("Thesis").model_copy(
                update={"body": f"Portfolio worth {formatted_value}."}
            )
        }
    )
    publication_error = getattr(report_models, "ReportPublicationError")

    with pytest.raises(publication_error) as exc_info:
        ReportRenderer(
            tmp_path,
            privacy_context=publication_privacy_context(
                portfolio_values=(Decimal("1234.56"),)
            ),
        ).render_event_update(unsafe)

    assert str(exc_info.value) == "report publication blocked by privacy policy"
    assert not tuple(tmp_path.glob("*.html"))


@pytest.mark.parametrize(
    "body",
    (
        "DCF fair value is USD 100 per share.",
        "Price target is $100.",
        "Revenue reached USD 5 billion.",
    ),
)
def test_display_text_allows_public_valuation_and_revenue_prose(body: str) -> None:
    value = ReportSection(title="Valuation", body=body, evidence_ids=("E-001",))
    assert value.body == body


@pytest.mark.parametrize(
    "mode",
    ("no_sensitive_data", "enforced"),
)
def test_renderer_allows_noncolliding_public_money_prose(
    tmp_path: Path,
    mode: str,
) -> None:
    context = (
        publication_privacy_context()
        if mode == "no_sensitive_data"
        else publication_privacy_context(portfolio_values=(Decimal("999.99"),))
    )
    safe = event_report().model_copy(
        update={
            "thesis": section("Thesis").model_copy(
                update={"body": "DCF fair value is USD 100 per share."}
            )
        }
    )

    artifact = ReportRenderer(
        tmp_path, privacy_context=context
    ).render_event_update(safe)

    assert "DCF fair value is USD 100 per share." in artifact.html


def test_renderer_blocks_known_value_even_in_valuation_prose(tmp_path: Path) -> None:
    unsafe = event_report().model_copy(
        update={
            "thesis": section("Thesis").model_copy(
                update={"body": "DCF fair value is USD 100 per share."}
            )
        }
    )
    publication_error = getattr(report_models, "ReportPublicationError")

    with pytest.raises(publication_error, match="publication blocked"):
        ReportRenderer(
            tmp_path,
            privacy_context=publication_privacy_context(
                portfolio_values=(Decimal("100"),)
            ),
        ).render_event_update(unsafe)

    assert not tuple(tmp_path.glob("*.html"))


def test_privacy_scanner_has_bounded_linear_adjacent_checks() -> None:
    source = inspect.getsource(report_models._publication_number_values)
    assert "normalized[: match.start()]" not in source
    assert "normalized[match.end() :]" not in source


def test_privacy_number_parser_never_splits_malformed_grouping() -> None:
    parser = report_models._publication_number_values
    assert parser("Malformed 12 34.56 and 1,23.45 values.") == ()


def test_publication_limits_reject_oversize_field_without_echoing_input() -> None:
    max_chars = getattr(report_models, "MAX_PUBLICATION_STRING_CHARS")
    marker = "PRIVATE-OVERSIZE-MARKER"
    with pytest.raises(ValidationError) as exc_info:
        ReportSection(
            title="Analysis",
            body="x" * (max_chars + 1) + marker,
            evidence_ids=("E-001",),
        )
    assert marker not in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "limit_name", "prefix"),
    (
        ("sensitive_literals", "MAX_PRIVACY_LITERAL_CHARS", "x"),
        ("account_identifiers", "MAX_PRIVACY_ACCOUNT_CHARS", "A1"),
    ),
)
def test_publication_limits_reject_oversize_context_without_echoing_input(
    field: str,
    limit_name: str,
    prefix: str,
) -> None:
    max_chars = getattr(report_models, limit_name)
    marker = "PRIVATE-CONTEXT-MARKER"
    context_type = getattr(report_models, "PublicationPrivacyContext")
    values: dict[str, object] = {
        "mode": "enforced",
        "sensitive_literals": (),
        "account_identifiers": (),
        "portfolio_values": (),
    }
    values[field] = (prefix + "x" * max_chars + marker,)
    with pytest.raises(ValidationError) as exc_info:
        context_type(**values)
    assert marker not in str(exc_info.value)


@pytest.mark.parametrize(
    "field",
    ("sensitive_literals", "account_identifiers", "portfolio_values"),
)
def test_publication_limits_reject_excess_context_items(field: str) -> None:
    limit = getattr(report_models, "MAX_PRIVACY_CONTEXT_ITEMS")
    values: dict[str, object] = {
        "mode": "enforced",
        "sensitive_literals": (),
        "account_identifiers": (),
        "portfolio_values": (),
    }
    if field == "sensitive_literals":
        values[field] = tuple(f"secret-{index}" for index in range(limit + 1))
    elif field == "account_identifiers":
        values[field] = tuple(f"A{index:05d}" for index in range(limit + 1))
    else:
        values[field] = tuple(Decimal(index) for index in range(limit + 1))
    context_type = getattr(report_models, "PublicationPrivacyContext")
    with pytest.raises(ValidationError, match="length|items|tuple"):
        context_type(**values)


def test_publication_limits_reject_aggregate_report_text_before_write(
    tmp_path: Path,
) -> None:
    per_string = getattr(report_models, "MAX_PUBLICATION_STRING_CHARS")
    total_text = getattr(report_models, "MAX_PUBLICATION_TOTAL_TEXT_CHARS")
    body = "x" * (per_string - 100)
    section_count = (total_text // len(body)) + 1
    oversized_sections = tuple(
        ReportSection(title=f"Section {index}", body=body, evidence_ids=("E-001",))
        for index in range(section_count)
    )
    unsafe = event_report().model_copy(
        update={"event_decomposition": oversized_sections}
    )
    publication_error = getattr(report_models, "ReportPublicationError")

    with pytest.raises(publication_error) as exc_info:
        renderer(tmp_path).render_event_update(unsafe)

    assert str(exc_info.value) == "report publication blocked by privacy policy"
    assert not tuple(tmp_path.glob("*.html"))


def test_publication_limits_reject_excess_report_nodes_before_write(
    tmp_path: Path,
) -> None:
    node_limit = getattr(report_models, "MAX_PUBLICATION_NODES")
    repeated = section("Repeated")
    unsafe = event_report().model_copy(
        update={"event_decomposition": (repeated,) * (node_limit + 1)}
    )
    publication_error = getattr(report_models, "ReportPublicationError")

    with pytest.raises(publication_error, match="publication blocked"):
        renderer(tmp_path).render_event_update(unsafe)

    assert not tuple(tmp_path.glob("*.html"))


def test_large_allowed_numeric_report_scans_within_generous_bound() -> None:
    max_chars = getattr(report_models, "MAX_PUBLICATION_STRING_CHARS")
    body = ("Metric 12,345.67; " * 10_000)[: max_chars - 1]
    report = event_report().model_copy(
        update={
            "thesis": section("Thesis").model_copy(update={"body": body})
        }
    )
    context = publication_privacy_context(
        portfolio_values=(Decimal("999999999.99"),)
    )

    started = time.monotonic()
    context.assert_safe(report)

    assert time.monotonic() - started < 2.0


@pytest.mark.parametrize("sign", PRIVACY_AMOUNT_SIGNS)
def test_renderer_rejects_copied_private_compatibility_sign_amounts(
    tmp_path: Path, sign: str
) -> None:
    private_body = f"NAV was USD {sign}1,234.56"
    unsafe = event_report().model_copy(
        update={
            "thesis": section("Thesis").model_copy(update={"body": private_body})
        }
    )

    with pytest.raises(ValidationError, match="private|sensitive"):
        renderer(tmp_path).render_event_update(unsafe)


def test_display_privacy_rejects_invisible_keyword_split() -> None:
    with pytest.raises(ValidationError, match="format|invisible|private|sensitive"):
        ReportSection(
            title="Analysis",
            body="N\u200bAV was USD 1,234.56",
            evidence_ids=("E-001",),
        )


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


@pytest.mark.parametrize("sign", (*PRIVACY_AMOUNT_SIGNS, "±"))
def test_display_privacy_filter_allows_legitimate_valuation_language(
    sign: str,
) -> None:
    body = (
        f"NAV was {sign}20%; NAV multiple was {sign}5x; "
        "valuation range is USD 70–100 per share."
    )
    section_value = ReportSection(
        title="Valuation",
        body=body,
        evidence_ids=("E-001",),
    )
    assert "5x" in section_value.body
    assert section_value.body == body


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
    html = renderer(tmp_path).render_industry_landscape(industry_report()).html
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
        StoredExposureRow(row_id="NVDA", symbol="NVDA", label="NVIDIA", weight=Decimal("0.25"), currency="USD", source_note="Statement", source_date=date(2026, 9, 8)),
        StoredExposureRow(row_id="OTHER", symbol="OTHER", label="Other", weight=Decimal("0.75"), currency="USD", source_note="Statement", source_date=date(2026, 9, 8)),
    )
    authority_rows = tuple(
        CalculatedRow(row_id=row.row_id, values={"weight": row.weight}) for row in rows
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
    stored = StoredExposureRow(row_id="NVDA", symbol="NVDA", label="NVIDIA", weight=Decimal("1"), currency="USD", source_note="Statement", source_date=date(2026, 9, 8))
    row = ExposureRow.model_validate(stored.model_dump())
    mismatched = calculation(
        "exposure",
        (CalculatedRow(row_id="NVDA", values={"weight": Decimal("0.9")}),),
    )
    with pytest.raises(ValidationError, match="authoritative|reconcile"):
        ExposureExhibit(
            title="Exposure", source_notes=("Statement (2026-09-08)",),
            evidence_ids=("E-001",), authority_rows=(stored,), rows=(row,), total_weight=Decimal("1"),
            calculation=mismatched, verified_reconciliation=True,
        )


def test_event_renderer_preserves_typed_exhibit_rows_and_citation(tmp_path: Path) -> None:
    row = StoredExposureRow(
        row_id="NVDA",
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

    artifact = renderer(tmp_path).render_event_update(report)

    assert "100.0%" in artifact.html
    assert "Statement (2026-09-08)" in artifact.html
    assert "Exhibit evidence: E-001" in artifact.html


def test_renderer_revalidates_copied_exhibit_at_authority_boundary(tmp_path: Path) -> None:
    row = StoredExposureRow(
        row_id="NVDA",
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
        renderer(tmp_path).render_event_update(report)


def test_valuation_and_scenario_builders_reconcile_authoritative_values() -> None:
    valuation_row = StoredValuationRow(
        row_id="EV-sales",
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
        StoredScenarioRow(row_id="Base", scenario="Base", probability=Decimal("0.6"), value=Decimal("100"), currency="USD", unit="per_share", source_note="Model", source_date=date(2026, 9, 8)),
        StoredScenarioRow(row_id="Downside", scenario="Downside", probability=Decimal("0.4"), value=Decimal("70"), currency="USD", unit="per_share", source_note="Model", source_date=date(2026, 9, 8)),
    )
    scenario_authority = calculation("scenario", tuple(
        CalculatedRow(row_id=row.row_id, values={"probability": row.probability, "value": row.value}) for row in scenario_rows
    ))
    assert build_scenario_matrix(
        scenario_rows, calculation=scenario_authority, evidence_ids=("E-001",)
    ).verified_reconciliation


def test_uncited_exhibit_is_rejected() -> None:
    row = StoredExposureRow(row_id="NVDA", symbol="NVDA", label="NVIDIA", weight=Decimal("1"), currency="USD", source_note="Statement", source_date=date(2026, 9, 8))
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
    html = renderer(
        Path(os.environ.get("TMPDIR", "/tmp")) / "typed-report-test"
    ).render_portfolio_brief(report).html
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
    artifact = renderer(tmp_path).render_event_update(event_report())
    assert artifact.html_path.exists()
    assert artifact.pdf_path is None
    assert artifact.pdf_error == "pdf_backend_unavailable"
    assert not tuple(tmp_path.glob("*.pdf"))


def test_wheel_configuration_includes_every_report_template() -> None:
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["setuptools"]["package-data"]["news_bot.research.reports"] == ["templates/*.html"]
    assert {path.name for path in Path("news_bot/research/reports/templates").glob("*.html")} == {
        "base.html", "event_update.html", "portfolio_brief.html",
        "industry_landscape.html", "emerging_monitor.html",
    }


@pytest.mark.parametrize(
    "private_text",
    (
        "U%31%32%33%34%35%36%37",
        "api%5Fkey%3Dsk-live-secret",
        "Brokerage account 12345678",
        "NAV was USD 1,234.56",
        "NAV: USD -1,234.56",
        "NAV was -USD 1,234.56",
        "NAV was USD −1,234.56",
        "cash balance was $1200.25",
        "cash balance was +EUR 1,200.25",
        "position value = (USD 9,876.54)",
    ),
)
def test_display_privacy_rejects_encoded_and_accounting_variants(
    private_text: str,
) -> None:
    with pytest.raises(ValidationError, match="private|sensitive"):
        ReportSection(title="Analysis", body=private_text, evidence_ids=("E-001",))


@pytest.mark.parametrize("sign", PRIVACY_AMOUNT_SIGNS)
def test_display_privacy_rejects_all_compatibility_sign_amounts(sign: str) -> None:
    with pytest.raises(ValidationError, match="private|sensitive"):
        ReportSection(
            title="Analysis",
            body=f"NAV was USD {sign}1,234.56",
            evidence_ids=("E-001",),
        )


@pytest.mark.parametrize("sign", ("＋", "－", "﹢", "﹣"))
def test_display_privacy_rejects_encoded_compatibility_sign_amounts(
    sign: str,
) -> None:
    encoded = quote(f"NAV was USD {sign}1,234.56", safe="")
    with pytest.raises(ValidationError, match="private|sensitive"):
        ReportSection(title="Analysis", body=encoded, evidence_ids=("E-001",))


def test_display_privacy_decoding_is_bounded_and_fail_closed() -> None:
    deeply_encoded = "api_key=sk-live-secret"
    for _ in range(20):
        deeply_encoded = quote(deeply_encoded, safe="")
    with pytest.raises(ValidationError, match="encoded|private|sensitive"):
        ReportSection(
            title="Analysis", body=deeply_encoded, evidence_ids=("E-001",)
        )

    benign = ReportSection(
        title="Analysis",
        body="Revenue rose 20%; malformed %ZZ stays literal.",
        evidence_ids=("E-001",),
    )
    assert "%ZZ" in benign.body


@pytest.mark.parametrize("sign", PRIVACY_AMOUNT_SIGNS)
def test_rounded_exposure_identity_rejects_bare_money(sign: str) -> None:
    with pytest.raises(ValidationError, match="money|private|sensitive"):
        RoundedExposure(
            symbol="NVDA",
            label=f"NVIDIA — USD {sign}1,234.56",
            weight_band="20%+",
            rounded_weight_percent=Decimal("25"),
            evidence_ids=("E-001",),
        )


def stored_exposure(**updates: object) -> object:
    StoredExposureRow = getattr(exhibit_models, "StoredExposureRow")
    values = {
        "row_id": "position-nvda",
        "symbol": "NVDA",
        "label": "NVIDIA",
        "weight": Decimal("1"),
        "currency": "USD",
        "source_note": "Broker statement",
        "source_date": date(2026, 9, 8),
    }
    values.update(updates)
    return StoredExposureRow(**values)


def stored_valuation(**updates: object) -> object:
    StoredValuationRow = getattr(exhibit_models, "StoredValuationRow")
    values = {
        "row_id": "nvda-ev-sales",
        "label": "NVDA EV / sales",
        "low": Decimal("5"),
        "base": Decimal("7"),
        "high": Decimal("9"),
        "currency": "USD",
        "unit": "multiple",
        "source_note": "Analyst model",
        "source_date": date(2026, 9, 8),
    }
    values.update(updates)
    return StoredValuationRow(**values)


def test_exhibit_builder_derives_display_and_provenance_from_stored_rows() -> None:
    authority = stored_exposure()
    numeric = calculation(
        "exposure",
        (CalculatedRow(row_id="position-nvda", values={"weight": Decimal("1")}),),
    )
    exhibit = build_exposure_exhibit(
        (authority,), calculation=numeric, evidence_ids=("E-001",)
    )

    assert exhibit.rows[0].row_id == "position-nvda"
    assert exhibit.rows[0].label == "NVIDIA"
    assert exhibit.rows[0].currency == "USD"
    assert exhibit.source_notes == ("Broker statement (2026-09-08)",)
    assert exhibit.authority_rows == (authority,)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("row_id", "wrong-id"),
        ("label", "Wrong label"),
        ("currency", "EUR"),
        ("source_note", "Wrong source"),
        ("source_date", date(2026, 9, 7)),
    ),
)
def test_exposure_exhibit_rejects_rendered_provenance_mismatch(
    field: str, bad_value: object
) -> None:
    authority = stored_exposure()
    numeric = calculation(
        "exposure",
        (CalculatedRow(row_id="position-nvda", values={"weight": Decimal("1")}),),
    )
    valid = build_exposure_exhibit(
        (authority,), calculation=numeric, evidence_ids=("E-001",)
    )
    values = valid.model_dump()
    values["rows"][0][field] = bad_value

    with pytest.raises(ValidationError, match="authoritative|stored|provenance"):
        ExposureExhibit.model_validate(values)


def test_valuation_exhibit_rejects_rendered_unit_mismatch() -> None:
    authority = stored_valuation()
    numeric = calculation(
        "valuation",
        (
            CalculatedRow(
                row_id="nvda-ev-sales",
                values={
                    "low": Decimal("5"),
                    "base": Decimal("7"),
                    "high": Decimal("9"),
                },
            ),
        ),
    )
    valid = build_valuation_exhibit(
        (authority,), calculation=numeric, evidence_ids=("E-001",)
    )
    values = valid.model_dump()
    values["rows"][0]["unit"] = "currency"

    with pytest.raises(ValidationError, match="authoritative|stored|provenance"):
        exhibit_models.ValuationExhibit.model_validate(values)


def test_exhibit_builder_rejects_numeric_authority_row_id_mismatch() -> None:
    authority = stored_exposure()
    wrong_numeric = calculation(
        "exposure",
        (CalculatedRow(row_id="wrong-id", values={"weight": Decimal("1")}),),
    )
    with pytest.raises((ValueError, ValidationError), match="authoritative|stored"):
        build_exposure_exhibit(
            (authority,), calculation=wrong_numeric, evidence_ids=("E-001",)
        )


def test_rating_history_allows_unchanged_current_view() -> None:
    unchanged = ResearchViewChange(
        symbol="NVDA",
        previous_rating=RecommendationRating.HOLD,
        new_rating=RecommendationRating.HOLD,
        rationale="Thesis and valuation remain balanced",
        changed_on=date(2026, 9, 9),
        evidence_ids=("E-001",),
    )
    values = portfolio_report().model_dump()
    values["change_history"] = (unchanged.model_dump(),)
    assert PortfolioBrief.model_validate(values).change_history == (unchanged,)


def test_portfolio_history_maps_exactly_once_to_each_current_view() -> None:
    values = portfolio_report().model_dump()
    nvda_view = ResearchView(
        symbol="NVDA",
        thesis="Demand remains durable",
        rating=RecommendationRating.HOLD,
        evidence_ids=("E-001",),
    ).model_dump()
    msft_view = ResearchView(
        symbol="MSFT",
        thesis="Cloud demand remains durable",
        rating=RecommendationRating.BUY,
        evidence_ids=("E-001",),
    ).model_dump()
    nvda_history = {
        "symbol": "nvda",
        "previous_rating": RecommendationRating.BUY,
        "new_rating": RecommendationRating.HOLD,
        "rationale": "Valuation now balanced",
        "changed_on": date(2026, 9, 9),
        "evidence_ids": ("E-001",),
    }
    msft_history = {
        "symbol": "MSFT",
        "previous_rating": RecommendationRating.BUY,
        "new_rating": RecommendationRating.BUY,
        "rationale": "No change to the current view",
        "changed_on": date(2026, 9, 9),
        "evidence_ids": ("E-001",),
    }
    values["research_views"] = (nvda_view, msft_view)
    values["change_history"] = (nvda_history, msft_history)
    assert len(PortfolioBrief.model_validate(values).change_history) == 2

    extra = {
        **msft_history,
        "symbol": "AAPL",
        "rationale": "No current view exists",
    }
    mismatch = {**nvda_history, "new_rating": RecommendationRating.BUY}
    for bad_history in (
        (nvda_history,),
        (nvda_history, msft_history, nvda_history),
        (nvda_history, msft_history, extra),
        (mismatch, msft_history),
    ):
        with pytest.raises(ValidationError, match="history|rating|symbol"):
            PortfolioBrief.model_validate({**values, "change_history": bad_history})

    with pytest.raises(ValidationError, match="current|view|symbol"):
        PortfolioBrief.model_validate(
            {**values, "research_views": (nvda_view, nvda_view)}
        )


def test_pdf_stale_cleanup_failure_never_escapes_or_surfaces_stale_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    report_renderer = renderer(tmp_path)
    report = event_report()
    stale_pdf = tmp_path / (
        report_renderer._filename(
            report.metadata.report_type,
            report.metadata.report_id,
            report.metadata.as_of,
        )
        + ".pdf"
    )
    stale_pdf.write_bytes(b"%PDF-stale")
    real_unlink = Path.unlink

    def deny_stale_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == stale_pdf:
            raise PermissionError("cleanup denied")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_stale_unlink)
    publication_error = getattr(report_models, "ReportPublicationError")
    with pytest.raises(publication_error) as exc_info:
        report_renderer.render_event_update(report)

    assert str(exc_info.value) == "report output conflict"
    assert tuple(tmp_path.glob("*.html"))
    assert "cleanup denied" not in caplog.text
    assert "cleanup denied" not in str(exc_info.value)
    assert str(stale_pdf) not in caplog.text
    assert stale_pdf.read_bytes() == b"%PDF-stale"
    assert not tuple(tmp_path.glob("*.tmp"))
