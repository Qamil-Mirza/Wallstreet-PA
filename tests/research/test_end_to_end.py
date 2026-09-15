from decimal import Decimal

from tests.research.golden_harness import (
    run_cost_simulation,
    run_synthetic_end_to_end,
)


def test_synthetic_portfolio_dry_run_is_offline_and_outputs_html_and_pdf(tmp_path):
    result = run_synthetic_end_to_end(tmp_path)
    assert result.external_calls == ()
    assert result.order_or_trade_requests == ()
    assert result.email_attempts == 0
    assert result.html_path.is_file()
    assert result.pdf_path.is_file()
    assert result.pdf_path.read_bytes().startswith(b"%PDF-")
    assert "Institutional research" in result.html
    assert "Validation thesis" in result.html
    assert "Deterministic offline fixture" in result.html
    assert "All content is synthetic" in result.html
    assert result.workflow_status == "partial"
    assert "publication_dry_run" in result.omissions
    assert "DU123456" not in result.html
    assert "NAV: USD" not in result.html


def test_monthly_cost_simulation_stops_at_five_dollars(tmp_path):
    result = run_cost_simulation(tmp_path, days=31)
    assert result.paid_cost <= Decimal("5.00")
    assert result.paid_cost == Decimal("4.00")
    assert result.completed_tasks == 10
    assert result.deferred_tasks > 0
    assert result.completed_tasks + result.deferred_tasks == 31
