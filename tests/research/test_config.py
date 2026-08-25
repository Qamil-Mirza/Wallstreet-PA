"""Tests for research configuration and foundational domain types."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from news_bot.research.config import ResearchConfig, ResearchConfigError
from news_bot.research.models import (
    AgentRunResult,
    AgentRole,
    ClaimKind,
    EvidenceClaim,
    InferenceMode,
    ModelUsage,
    PortfolioSnapshot,
    Position,
    RecommendationRating,
    ReviewVerdict,
    SourceDocument,
)


@pytest.fixture(autouse=True)
def clear_research_environment(monkeypatch):
    """Keep research tests independent of workstation configuration and secrets."""
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_API_KEY_FILE",
        "RESEARCH_ENABLED",
        "RESEARCH_DATA_DIR",
        "OLLAMA_BASE_URL",
        "OLLAMA_RESEARCH_MODEL",
        "MODEL_BUDGET_SOFT_USD",
        "MODEL_BUDGET_HARD_USD",
    ):
        monkeypatch.delenv(name, raising=False)


def test_missing_openai_key_selects_local_only(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(tmp_path))
    config = ResearchConfig.from_env()
    assert config.inference_mode is InferenceMode.LOCAL_ONLY
    assert config.ollama_base_url == "http://localhost:11434"


def test_secret_file_wins_over_environment(monkeypatch, tmp_path):
    secret = tmp_path / "openai_key"
    secret.write_text("file-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(secret))
    assert ResearchConfig.from_env().openai_api_key == "file-key"


def test_portfolio_snapshot_is_timezone_aware():
    snapshot = PortfolioSnapshot(
        snapshot_id="snapshot-1",
        as_of=datetime(2026, 8, 24, tzinfo=timezone.utc),
        base_currency="USD",
        nav=Decimal("4500.00"),
        cash=Decimal("500.00"),
        is_stale=False,
    )
    assert snapshot.as_of.tzinfo is not None


def test_portfolio_snapshot_rejects_naive_datetime():
    with pytest.raises(
        ValueError, match="PortfolioSnapshot.as_of must be timezone-aware"
    ):
        PortfolioSnapshot(
            snapshot_id="snapshot-1",
            as_of=datetime(2026, 8, 24),
            base_currency="USD",
            nav=Decimal("4500.00"),
            cash=Decimal("500.00"),
            is_stale=False,
        )


def test_domain_enums_have_approved_values():
    assert {mode.value for mode in InferenceMode} == {"external", "local_only"}
    assert {role.value for role in AgentRole} == {
        "research_director",
        "portfolio_mapper",
        "event_scout",
        "emerging_company_scout",
        "evidence_analyst",
        "industry_strategist",
        "fundamental_analyst",
        "skeptical_reviewer",
        "research_editor",
    }
    assert {kind.value for kind in ClaimKind} == {
        "fact",
        "guidance",
        "estimate",
        "inference",
    }
    assert {rating.value for rating in RecommendationRating} == {
        "buy",
        "hold",
        "sell_reduce",
        "no_rating",
    }
    assert {verdict.value for verdict in ReviewVerdict} == {"pass", "revise", "block"}


def test_portfolio_snapshot_is_frozen():
    snapshot = PortfolioSnapshot(
        snapshot_id="snapshot-1",
        as_of=datetime(2026, 8, 24, tzinfo=timezone.utc),
        base_currency="USD",
        nav=Decimal("4500.00"),
        cash=Decimal("500.00"),
        is_stale=False,
    )

    with pytest.raises(FrozenInstanceError):
        snapshot.cash = Decimal("0")  # type: ignore[misc]


def test_money_and_quantity_fields_use_decimal_values():
    position = Position(
        position_id="position-1",
        snapshot_id="snapshot-1",
        symbol="NVDA",
        quantity=Decimal("2.5"),
        market_value=Decimal("450.00"),
        currency="USD",
        cost_basis=Decimal("300.00"),
    )

    assert isinstance(position.quantity, Decimal)
    assert isinstance(position.market_value, Decimal)
    assert isinstance(position.cost_basis, Decimal)


def test_money_and_quantity_fields_reject_non_decimal_values():
    with pytest.raises(TypeError, match="PortfolioSnapshot.nav must be Decimal"):
        PortfolioSnapshot(
            snapshot_id="snapshot-1",
            as_of=datetime(2026, 8, 24, tzinfo=timezone.utc),
            base_currency="USD",
            nav="4500.00",  # type: ignore[arg-type]
            cash=Decimal("500.00"),
            is_stale=False,
        )

    with pytest.raises(TypeError, match="Position.quantity must be Decimal"):
        Position(
            position_id="position-1",
            snapshot_id="snapshot-1",
            symbol="NVDA",
            quantity=2.5,  # type: ignore[arg-type]
            market_value=Decimal("450.00"),
            currency="USD",
        )


def test_timestamp_bearing_domain_records_reject_naive_datetimes():
    aware = datetime(2026, 8, 24, tzinfo=timezone.utc)
    naive = datetime(2026, 8, 24)

    with pytest.raises(ValueError, match="SourceDocument.published_at"):
        SourceDocument(
            document_id="document-1",
            source_type="sec_filing",
            canonical_url="https://www.sec.gov/example",
            publisher="SEC",
            published_at=naive,
            retrieved_at=aware,
            content_hash="abc123",
            raw_content_path=None,
            extraction_status="complete",
        )

    with pytest.raises(ValueError, match="EvidenceClaim.as_of"):
        EvidenceClaim(
            claim_id="claim-1",
            entity_id=None,
            kind=ClaimKind.FACT,
            text="Revenue grew 20%.",
            as_of=naive,
            confidence=Decimal("0.90"),
            status="active",
        )

    with pytest.raises(ValueError, match="ModelUsage.recorded_at"):
        ModelUsage(
            usage_id="usage-1",
            run_id="run-1",
            provider="openai",
            model="example-model",
            input_tokens=100,
            output_tokens=50,
            cost_usd=Decimal("0.02"),
            recorded_at=naive,
        )

    with pytest.raises(ValueError, match="AgentRunResult.completed_at"):
        AgentRunResult(
            run_id="run-1",
            role=AgentRole.EVIDENCE_ANALYST,
            started_at=aware,
            completed_at=naive,
            status="complete",
            output="{}",
        )


def test_direct_secret_value_is_stripped(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "  environment-key\n")

    config = ResearchConfig.from_env()

    assert config.openai_api_key == "environment-key"
    assert config.inference_mode is InferenceMode.EXTERNAL


def test_unreadable_secret_file_has_clear_error(monkeypatch, tmp_path):
    missing = tmp_path / "missing-key"
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(missing))

    with pytest.raises(ResearchConfigError, match="OPENAI_API_KEY.*secret file"):
        ResearchConfig.from_env()


@pytest.mark.parametrize("value", ["maybe", "2", ""])
def test_invalid_research_enabled_has_clear_error(monkeypatch, value):
    monkeypatch.setenv("RESEARCH_ENABLED", value)

    with pytest.raises(ResearchConfigError, match="RESEARCH_ENABLED.*boolean"):
        ResearchConfig.from_env()


@pytest.mark.parametrize("name", ["MODEL_BUDGET_SOFT_USD", "MODEL_BUDGET_HARD_USD"])
def test_invalid_budget_decimal_has_clear_error(monkeypatch, name):
    monkeypatch.setenv(name, "not-a-decimal")

    with pytest.raises(ResearchConfigError, match=f"{name}.*decimal"):
        ResearchConfig.from_env()


def test_soft_budget_must_be_less_than_hard_budget(monkeypatch):
    monkeypatch.setenv("MODEL_BUDGET_SOFT_USD", "5.00")
    monkeypatch.setenv("MODEL_BUDGET_HARD_USD", "5.00")

    with pytest.raises(
        ResearchConfigError,
        match="MODEL_BUDGET_SOFT_USD must be less than MODEL_BUDGET_HARD_USD",
    ):
        ResearchConfig.from_env()


@pytest.mark.parametrize(
    ("soft", "hard", "message"),
    [
        ("-2.00", "-1.00", "must be non-negative"),
        ("4.00", "5.01", "must not exceed 5.00"),
    ],
)
def test_budget_limits_preserve_monthly_safety_ceiling(
    monkeypatch, soft, hard, message
):
    monkeypatch.setenv("MODEL_BUDGET_SOFT_USD", soft)
    monkeypatch.setenv("MODEL_BUDGET_HARD_USD", hard)

    with pytest.raises(ResearchConfigError, match=message):
        ResearchConfig.from_env()


def test_research_paths_are_derived_without_creating_them(monkeypatch, tmp_path):
    data_dir = tmp_path / "research"
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(data_dir))

    config = ResearchConfig.from_env()

    assert config.database_path == data_dir / "research.db"
    assert config.cache_dir == data_dir / "cache"
    assert config.report_dir == data_dir / "reports"
    assert config.backup_dir == data_dir / "backups"
    assert not data_dir.exists()
