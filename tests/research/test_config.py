"""Tests for research configuration and foundational domain types."""

from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from news_bot.research.config import CronSchedule, ResearchConfig, ResearchConfigError
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
        "OLLAMA_HEALTH_TIMEOUT_SECONDS",
        "IBKR_FLEX_BASE_URL",
        "IBKR_FLEX_POLL_TIMEOUT_SECONDS",
        "PORTFOLIO_MAX_STALENESS_HOURS",
        "SOURCE_MAX_STALENESS_HOURS",
        "SEC_USER_AGENT",
        "MODEL_PRICE_EFFECTIVE_UNTIL",
        "MODEL_INPUT_PRICE_PER_MILLION_USD",
        "MODEL_OUTPUT_PRICE_PER_MILLION_USD",
        "MODEL_BUDGET_SOFT_USD",
        "MODEL_BUDGET_HARD_USD",
        "RESEARCH_DAILY_SCHEDULE",
        "RESEARCH_WEEKLY_SCHEDULE",
        "RESEARCH_MONTHLY_SCHEDULE",
        "RESEARCH_MONTHLY_INDUSTRY",
        "IBKR_FLEX_TOKEN",
        "IBKR_FLEX_TOKEN_FILE",
        "IBKR_FLEX_QUERY_ID",
        "IBKR_FLEX_QUERY_ID_FILE",
        "IBKR_FLEX_ACCOUNT_SALT",
        "IBKR_FLEX_ACCOUNT_SALT_FILE",
        *(f"MODEL_ROUTE_{role.value.upper()}" for role in AgentRole),
    ):
        monkeypatch.delenv(name, raising=False)


def valid_research_config() -> ResearchConfig:
    """Build a valid config directly so construction invariants are testable."""
    return ResearchConfig(
        enabled=False,
        data_dir=Path("research_data"),
        openai_api_key=None,
        inference_mode=InferenceMode.LOCAL_ONLY,
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1:8b",
        budget_soft_usd=Decimal("4.00"),
        budget_hard_usd=Decimal("5.00"),
    )


def test_missing_openai_key_selects_local_only(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(tmp_path))
    config = ResearchConfig.from_env()
    assert config.inference_mode is InferenceMode.LOCAL_ONLY
    assert config.ollama_base_url == "http://localhost:11434"


def test_advertised_runtime_policy_is_typed_from_environment(monkeypatch):
    monkeypatch.setenv(
        "IBKR_FLEX_BASE_URL",
        "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService",
    )
    monkeypatch.setenv("IBKR_FLEX_POLL_TIMEOUT_SECONDS", "75")
    monkeypatch.setenv("PORTFOLIO_MAX_STALENESS_HOURS", "36")
    monkeypatch.setenv("SOURCE_MAX_STALENESS_HOURS", "96")
    monkeypatch.setenv("OLLAMA_HEALTH_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("SEC_USER_AGENT", "Research Bot research@example.com")
    monkeypatch.setenv("MODEL_PRICE_EFFECTIVE_UNTIL", "2027-01-31")
    monkeypatch.setenv("MODEL_INPUT_PRICE_PER_MILLION_USD", "12.50")
    monkeypatch.setenv("MODEL_OUTPUT_PRICE_PER_MILLION_USD", "50.00")
    monkeypatch.setenv("MODEL_ROUTE_EVENT_SCOUT", "gpt-route-scout")

    config = ResearchConfig.from_env()

    assert config.ibkr_flex_base_url.endswith("/FlexWebService")
    assert config.ibkr_flex_poll_timeout_seconds == 75.0
    assert config.portfolio_max_staleness_hours == 36.0
    assert config.source_max_staleness_hours == 96.0
    assert config.ollama_health_timeout_seconds == 45.0
    assert config.sec_user_agent == "Research Bot research@example.com"
    assert config.model_price_effective_until == date(2027, 1, 31)
    assert config.model_input_price_per_million_usd == Decimal("12.50")
    assert config.model_output_price_per_million_usd == Decimal("50.00")
    assert config.model_routes[AgentRole.EVENT_SCOUT] == "gpt-route-scout"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("IBKR_FLEX_POLL_TIMEOUT_SECONDS", "zero"),
        ("PORTFOLIO_MAX_STALENESS_HOURS", "0"),
        ("SOURCE_MAX_STALENESS_HOURS", "NaN"),
        ("OLLAMA_HEALTH_TIMEOUT_SECONDS", "-1"),
        ("MODEL_PRICE_EFFECTIVE_UNTIL", "not-a-date"),
        ("MODEL_INPUT_PRICE_PER_MILLION_USD", "-1"),
        ("MODEL_INPUT_PRICE_PER_MILLION_USD", "0"),
        ("MODEL_OUTPUT_PRICE_PER_MILLION_USD", "NaN"),
        ("MODEL_OUTPUT_PRICE_PER_MILLION_USD", "0"),
    ],
)
def test_advertised_runtime_policy_rejects_invalid_environment(
    monkeypatch, name, value
):
    monkeypatch.setenv(name, value)

    with pytest.raises(ResearchConfigError, match=name):
        ResearchConfig.from_env()


def test_flex_base_url_with_malformed_port_is_a_configuration_error(monkeypatch):
    monkeypatch.setenv(
        "IBKR_FLEX_BASE_URL",
        "https://ndcdyn.interactivebrokers.com:notaport/AccountManagement/FlexWebService",
    )

    with pytest.raises(ResearchConfigError, match="ibkr_flex_base_url"):
        ResearchConfig.from_env()


def test_scheduler_cron_settings_are_typed_from_environment(monkeypatch):
    monkeypatch.setenv("RESEARCH_DAILY_SCHEDULE", "5 6 * * *")
    monkeypatch.setenv("RESEARCH_WEEKLY_SCHEDULE", "10 7 * * tue")
    monkeypatch.setenv("RESEARCH_MONTHLY_SCHEDULE", "15 8 2 * *")

    config = ResearchConfig.from_env()

    assert config.daily_schedule == CronSchedule.from_crontab("5 6 * * *")
    assert config.weekly_schedule == CronSchedule.from_crontab("10 7 * * tue")
    assert config.monthly_schedule == CronSchedule.from_crontab("15 8 2 * *")


def test_invalid_scheduler_cron_is_rejected(monkeypatch):
    monkeypatch.setenv("RESEARCH_DAILY_SCHEDULE", "not a cron")

    with pytest.raises(ResearchConfigError, match="RESEARCH_DAILY_SCHEDULE"):
        ResearchConfig.from_env()


def test_out_of_range_scheduler_cron_is_rejected(monkeypatch):
    monkeypatch.setenv("RESEARCH_DAILY_SCHEDULE", "0 99 * * *")

    with pytest.raises(ResearchConfigError, match="RESEARCH_DAILY_SCHEDULE"):
        ResearchConfig.from_env()


@pytest.mark.parametrize(
    "schedule",
    [
        lambda: CronSchedule("bad", "7", "*", "*", "mon"),
        lambda: CronSchedule.from_crontab("0 7 * * fri-mon"),
        lambda: CronSchedule.from_crontab("*/99 7 * * mon"),
        lambda: CronSchedule.from_crontab("0 7 * * 1-5"),
    ],
)
def test_cron_schedule_rejects_direct_bypass_and_ambiguous_ranges(schedule) -> None:
    with pytest.raises(ResearchConfigError):
        schedule()


def test_default_weekday_crons_use_apscheduler_names() -> None:
    config = ResearchConfig.from_env()

    assert config.daily_schedule.day_of_week == "mon-fri"
    assert config.weekly_schedule.day_of_week == "mon"


def test_default_weekday_crons_have_deterministic_next_fire_times() -> None:
    cron_module = pytest.importorskip("apscheduler.triggers.cron")
    trigger_type = cron_module.CronTrigger
    config = ResearchConfig.from_env()
    friday_after_daily = datetime(2026, 8, 21, 8, tzinfo=timezone.utc)

    daily = trigger_type(
        **config.daily_schedule.as_kwargs(), timezone=timezone.utc
    )
    weekly = trigger_type(
        **config.weekly_schedule.as_kwargs(), timezone=timezone.utc
    )

    assert daily.get_next_fire_time(None, friday_after_daily) == datetime(
        2026, 8, 24, 7, tzinfo=timezone.utc
    )
    assert weekly.get_next_fire_time(None, friday_after_daily) == datetime(
        2026, 8, 24, 8, tzinfo=timezone.utc
    )


@pytest.mark.parametrize(
    "value",
    (
        "*/60 7 * * mon",
        "0 7 */31 * mon",
        "0 7 * */12 mon",
        "0 7 * * */7",
        "1-2/2 7 * * mon",
        "0 7 * jan-feb/2 mon",
    ),
)
def test_cron_steps_cannot_exceed_apscheduler_available_span(value: str) -> None:
    with pytest.raises(ResearchConfigError):
        CronSchedule.from_crontab(value)


@pytest.mark.parametrize(
    "value",
    (
        "0 7 * jan-mar/2 mon",
        "0 7 * * mon-fri/2",
    ),
)
def test_named_cron_ranges_reject_ambiguous_steps(value: str) -> None:
    with pytest.raises(ResearchConfigError):
        CronSchedule.from_crontab(value)


def test_cron_steps_accept_apscheduler_span_boundaries() -> None:
    schedule = CronSchedule.from_crontab("*/59 */23 */30 */11 */6")

    assert schedule.as_kwargs() == {
        "minute": "*/59",
        "hour": "*/23",
        "day": "*/30",
        "month": "*/11",
        "day_of_week": "*/6",
    }
    assert CronSchedule.from_crontab("1-3/2 7 * * mon").minute == "1-3/2"


def test_real_cron_trigger_matches_validated_step_spans() -> None:
    cron_module = pytest.importorskip("apscheduler.triggers.cron")
    schedule = CronSchedule.from_crontab("*/59 */23 */30 */11 */6")

    cron_module.CronTrigger(**schedule.as_kwargs(), timezone=timezone.utc)
    cron_module.CronTrigger(
        **CronSchedule.from_crontab("1-3/2 7 * * mon").as_kwargs(),
        timezone=timezone.utc,
    )


def test_real_cron_trigger_preserves_step_free_named_ranges() -> None:
    cron_module = pytest.importorskip("apscheduler.triggers.cron")
    schedule = CronSchedule.from_crontab("0 7 * jan-mar mon-fri")

    trigger = cron_module.CronTrigger(
        **schedule.as_kwargs(), timezone=timezone.utc
    )
    fields = {field.name: str(field) for field in trigger.fields}

    assert fields["month"] == "jan-mar"
    assert fields["day_of_week"] == "mon-fri"


def test_short_flex_account_salt_is_rejected() -> None:
    with pytest.raises(ResearchConfigError, match="at least 16 UTF-8 bytes"):
        replace(
            valid_research_config(),
            ibkr_flex_token="token",
            ibkr_flex_query_id="query",
            ibkr_flex_account_salt="too-short",
        )


def test_monthly_industry_is_typed_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_MONTHLY_INDUSTRY", "robotic-actuators")
    assert ResearchConfig.from_env().monthly_industry == "robotic-actuators"


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


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize(
    ("record", "field_name"),
    [
        (
            PortfolioSnapshot(
                snapshot_id="snapshot-1",
                as_of=datetime(2026, 8, 24, tzinfo=timezone.utc),
                base_currency="USD",
                nav=Decimal("4500.00"),
                cash=Decimal("500.00"),
                is_stale=False,
            ),
            "nav",
        ),
        (
            PortfolioSnapshot(
                snapshot_id="snapshot-1",
                as_of=datetime(2026, 8, 24, tzinfo=timezone.utc),
                base_currency="USD",
                nav=Decimal("4500.00"),
                cash=Decimal("500.00"),
                is_stale=False,
            ),
            "cash",
        ),
        (
            Position(
                position_id="position-1",
                snapshot_id="snapshot-1",
                symbol="NVDA",
                quantity=Decimal("2.5"),
                market_value=Decimal("450.00"),
                currency="USD",
                cost_basis=Decimal("300.00"),
            ),
            "quantity",
        ),
        (
            Position(
                position_id="position-1",
                snapshot_id="snapshot-1",
                symbol="NVDA",
                quantity=Decimal("2.5"),
                market_value=Decimal("450.00"),
                currency="USD",
                cost_basis=Decimal("300.00"),
            ),
            "market_value",
        ),
        (
            Position(
                position_id="position-1",
                snapshot_id="snapshot-1",
                symbol="NVDA",
                quantity=Decimal("2.5"),
                market_value=Decimal("450.00"),
                currency="USD",
                cost_basis=Decimal("300.00"),
            ),
            "cost_basis",
        ),
        (
            EvidenceClaim(
                claim_id="claim-1",
                entity_id=None,
                kind=ClaimKind.FACT,
                text="Revenue grew 20%.",
                as_of=datetime(2026, 8, 24, tzinfo=timezone.utc),
                confidence=Decimal("0.90"),
                status="active",
            ),
            "confidence",
        ),
        (
            ModelUsage(
                usage_id="usage-1",
                run_id="run-1",
                provider="openai",
                model="example-model",
                input_tokens=100,
                output_tokens=50,
                cost_usd=Decimal("0.02"),
                recorded_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
            ),
            "cost_usd",
        ),
    ],
)
def test_domain_decimal_fields_reject_non_finite_values(record, field_name, value):
    with pytest.raises(
        ValueError,
        match=rf"{type(record).__name__}.{field_name} must be finite$",
    ):
        replace(record, **{field_name: Decimal(value)})


@pytest.mark.parametrize("confidence", ["-0.01", "1.01"])
def test_evidence_claim_confidence_must_be_between_zero_and_one(confidence):
    with pytest.raises(
        ValueError, match="EvidenceClaim.confidence must be between 0 and 1"
    ):
        EvidenceClaim(
            claim_id="claim-1",
            entity_id=None,
            kind=ClaimKind.FACT,
            text="Revenue grew 20%.",
            as_of=datetime(2026, 8, 24, tzinfo=timezone.utc),
            confidence=Decimal(confidence),
            status="active",
        )


def test_model_usage_cost_must_be_non_negative():
    with pytest.raises(ValueError, match="ModelUsage.cost_usd must be non-negative"):
        ModelUsage(
            usage_id="usage-1",
            run_id="run-1",
            provider="openai",
            model="example-model",
            input_tokens=100,
            output_tokens=50,
            cost_usd=Decimal("-0.01"),
            recorded_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize("field_name", ["input_tokens", "output_tokens"])
def test_model_usage_token_counts_must_be_non_negative(field_name):
    usage = ModelUsage(
        usage_id="usage-1",
        run_id="run-1",
        provider="openai",
        model="example-model",
        input_tokens=100,
        output_tokens=50,
        cost_usd=Decimal("0.02"),
        recorded_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
    )

    with pytest.raises(
        ValueError, match=rf"ModelUsage.{field_name} must be non-negative$"
    ):
        replace(usage, **{field_name: -1})


def test_direct_secret_value_is_stripped(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "  environment-key\n")

    config = ResearchConfig.from_env()

    assert config.openai_api_key == "environment-key"
    assert config.inference_mode is InferenceMode.EXTERNAL


def test_research_config_repr_does_not_expose_openai_key(monkeypatch):
    secret = "sk-exact-secret-that-must-not-leak"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    config = ResearchConfig.from_env()

    assert secret not in repr(config)


@pytest.mark.parametrize("field_name", ["budget_soft_usd", "budget_hard_usd"])
def test_direct_config_rejects_non_decimal_budgets(field_name):
    with pytest.raises(
        ResearchConfigError, match=rf"{field_name} must be Decimal$"
    ):
        replace(valid_research_config(), **{field_name: "4.00"})


@pytest.mark.parametrize("field_name", ["budget_soft_usd", "budget_hard_usd"])
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_direct_config_rejects_non_finite_budgets(field_name, value):
    with pytest.raises(
        ResearchConfigError, match=rf"{field_name} must be finite$"
    ):
        replace(valid_research_config(), **{field_name: Decimal(value)})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"budget_soft_usd": Decimal("-0.01")},
            "Model budget limits must be non-negative",
        ),
        (
            {
                "budget_soft_usd": Decimal("5.00"),
                "budget_hard_usd": Decimal("5.00"),
            },
            "MODEL_BUDGET_SOFT_USD must be less than MODEL_BUDGET_HARD_USD",
        ),
        (
            {"budget_hard_usd": Decimal("5.01")},
            "MODEL_BUDGET_HARD_USD must not exceed 5.00",
        ),
    ],
)
def test_direct_config_enforces_budget_safety_bounds(changes, message):
    with pytest.raises(ResearchConfigError, match=message):
        replace(valid_research_config(), **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"inference_mode": InferenceMode.EXTERNAL},
            "inference_mode must be local_only when OPENAI_API_KEY is absent",
        ),
        (
            {"openai_api_key": "configured-key"},
            "inference_mode must be external when OPENAI_API_KEY is configured",
        ),
        (
            {"openai_api_key": " "},
            "openai_api_key must be non-empty when provided",
        ),
        (
            {"inference_mode": "local_only"},
            "inference_mode must be InferenceMode",
        ),
    ],
)
def test_direct_config_enforces_inference_routing(changes, message):
    with pytest.raises(ResearchConfigError, match=message):
        replace(valid_research_config(), **changes)


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


@pytest.mark.parametrize(
    ("env_name", "field_name", "default"),
    [
        ("RESEARCH_DATA_DIR", "data_dir", Path("research_data")),
        ("OLLAMA_BASE_URL", "ollama_base_url", "http://localhost:11434"),
        ("OLLAMA_RESEARCH_MODEL", "ollama_model", "llama3.1:8b"),
    ],
)
def test_blank_text_environment_settings_use_defaults(
    monkeypatch, env_name, field_name, default
):
    monkeypatch.setenv(env_name, " \t ")

    config = ResearchConfig.from_env()

    assert getattr(config, field_name) == default


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"data_dir": Path(" ")}, "data_dir must be a non-empty path"),
        ({"ollama_base_url": " \t"}, "ollama_base_url must be non-empty"),
        ({"ollama_model": " \t"}, "ollama_model must be non-empty"),
    ],
)
def test_direct_config_rejects_blank_required_settings(changes, message):
    with pytest.raises(ResearchConfigError, match=message):
        replace(valid_research_config(), **changes)
