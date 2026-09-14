"""Tests for safe research workflow CLI and scheduling boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import replace
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from news_bot.research.cli import (
    CliServices,
    ExitCode,
    _validate_credentials,
    default_services,
    main,
)
from news_bot.research.config import CronSchedule, ResearchConfig, ResearchConfigError
from news_bot.research.ibkr_flex import (
    FlexTransportError,
    PortfolioFreshness,
    PortfolioSyncResult,
)
from news_bot.research.models import (
    InferenceMode,
    PortfolioSnapshot,
    Position,
    RecommendationRating,
    ReviewVerdict,
)
from news_bot.research.orchestrator import (
    ResearchOrchestrator,
    StageOutcome,
    WorkflowBusy,
    WorkflowKind,
    WorkflowRunResult,
    WorkflowRunState,
)
from news_bot.research.quality import PublicationVerdict, QualityGateResult
from news_bot.research.reports import ReportRenderer
from news_bot.research.runtime import RuntimeFactories, RuntimeStageRunner
from news_bot.research.scheduler import (
    RunAlreadyActive,
    RunLease,
    RunLeaseLost,
    SchedulerClockError,
    build_scheduler,
)
from news_bot.research.store import ResearchStore


def FakeResult(
    *,
    status: str = "completed",
    report_ids: tuple[str, ...] = ("report-private-123",),
    omissions: tuple[str, ...] = (),
) -> WorkflowRunResult:
    valid_statuses = {state.value: state for state in WorkflowRunState}
    result = WorkflowRunResult(
        workflow_id="wf-cli-result",
        workflow_kind=WorkflowKind.DAILY,
        status=valid_statuses.get(status, WorkflowRunState.COMPLETED),
        completed_stages=(),
        report_ids=report_ids,
        new_agent_runs=0,
        pending_tasks=(),
        omissions=omissions,
        dry_run=False,
    )
    if status not in valid_statuses:
        return result.model_copy(update={"status": status})
    return result


class RecordingWorkflows:
    def __init__(self, result: object | None = None) -> None:
        self.result = result or FakeResult()
        self.calls: list[tuple[str, dict[str, object]]] = []

    def _record(self, command: str, **kwargs: object) -> object:
        self.calls.append((command, kwargs))
        return self.result

    def run_daily(self, **kwargs: object) -> object:
        return self._record("daily", **kwargs)

    def run_weekly(self, **kwargs: object) -> object:
        return self._record("weekly", **kwargs)

    def run_monthly(self, **kwargs: object) -> object:
        return self._record("monthly", **kwargs)

    def run_backfill(self, **kwargs: object) -> object:
        return self._record("backfill", **kwargs)

    def regenerate(self, **kwargs: object) -> object:
        return self._record("regenerate", **kwargs)


def research_config(tmp_path: Path) -> ResearchConfig:
    return ResearchConfig(
        enabled=True,
        data_dir=tmp_path,
        openai_api_key=None,
        inference_mode=InferenceMode.LOCAL_ONLY,
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1:8b",
        budget_soft_usd=Decimal("4.00"),
        budget_hard_usd=Decimal("5.00"),
        daily_schedule=CronSchedule.from_crontab("5 6 * * *"),
        weekly_schedule=CronSchedule.from_crontab("10 7 * * tue"),
        monthly_schedule=CronSchedule.from_crontab("15 8 2 * *"),
        monthly_industry="robotic-actuators",
    )


def fake_services(
    tmp_path: Path,
    *,
    workflows: RecordingWorkflows | None = None,
    validator=None,
) -> tuple[CliServices, RecordingWorkflows, dict[str, object]]:
    config = research_config(tmp_path)
    workflow_service = workflows or RecordingWorkflows()
    events: dict[str, object] = {
        "config_loads": 0,
        "store_opens": 0,
        "workflow_builds": 0,
        "validations": [],
    }

    def load_config() -> ResearchConfig:
        events["config_loads"] = int(events["config_loads"]) + 1
        return config

    def open_store(loaded: ResearchConfig) -> ResearchStore:
        assert loaded is config
        events["store_opens"] = int(events["store_opens"]) + 1
        store = ResearchStore(loaded.database_path)
        store.migrate()
        return store

    def build_workflows(
        loaded: ResearchConfig, store: ResearchStore
    ) -> RecordingWorkflows:
        assert loaded is config
        assert store.database_path == config.database_path
        events["workflow_builds"] = int(events["workflow_builds"]) + 1
        return workflow_service

    def validate(
        command: str, loaded: ResearchConfig, requires_portfolio: bool
    ) -> None:
        cast_validations = events["validations"]
        assert isinstance(cast_validations, list)
        cast_validations.append((command, requires_portfolio))
        if validator is not None:
            validator(command, loaded, requires_portfolio)

    return (
        CliServices(
            load_config=load_config,
            open_store=open_store,
            build_workflows=build_workflows,
            validate_credentials=validate,
        ),
        workflow_service,
        events,
    )


@pytest.mark.parametrize(
    ("command", "workflow"),
    [
        ("daily", "daily"),
        ("weekly", "weekly"),
        ("monthly", "monthly"),
        ("backfill", "backfill"),
        ("dry-run", "daily"),
        ("regenerate", "regenerate"),
    ],
)
def test_all_commands_dispatch_exactly_one_workflow(
    tmp_path: Path, capsys, command: str, workflow: str
) -> None:
    services, recorder, events = fake_services(tmp_path)

    argv = [command, "--as-of", "2026-08-24"]
    if command == "monthly":
        argv.extend(("--industry", "robotic-actuators"))
    if command == "regenerate":
        argv.extend(("--run-id", "wf_existing"))
    exit_code = main(argv, services=services)

    assert exit_code == ExitCode.OK
    assert [name for name, _ in recorder.calls] == [workflow]
    assert events == {
        "config_loads": 1,
        "store_opens": 1,
        "workflow_builds": 1,
        "validations": [(command, command in {"daily", "weekly", "dry-run"})],
    }
    assert recorder.calls[0][1]["as_of"] == datetime(
        2026, 8, 24, tzinfo=timezone.utc
    )
    output = capsys.readouterr().out
    assert f"command={command}" in output
    assert "status=completed" in output


def test_dry_run_marks_workflow_and_never_calls_external_sender(tmp_path: Path) -> None:
    services, recorder, _ = fake_services(tmp_path)

    assert main(["dry-run", "--as-of", "2026-08-24"], services=services) == 0

    assert recorder.calls == [
        (
            "daily",
            {
                "as_of": datetime(2026, 8, 24, tzinfo=timezone.utc),
                "dry_run": True,
                "synthetic_portfolio": False,
            },
        )
    ]


def test_synthetic_dry_run_skips_real_portfolio_and_marks_dispatch(
    tmp_path: Path,
) -> None:
    services, recorder, events = fake_services(tmp_path)

    code = main(
        [
            "dry-run",
            "--as-of",
            "2026-08-24",
            "--synthetic-portfolio",
        ],
        services=services,
    )

    assert code == 0
    assert events["validations"] == [("dry-run", False)]
    assert recorder.calls[0][1]["synthetic_portfolio"] is True


def test_backfill_is_bounded_and_evidence_only_by_default(tmp_path: Path) -> None:
    services, recorder, _ = fake_services(tmp_path)

    code = main(
        ["backfill", "--as-of", "2026-08-24", "--max-documents", "25"],
        services=services,
    )

    assert code == 0
    assert recorder.calls[0][1] == {
        "as_of": datetime(2026, 8, 24, tzinfo=timezone.utc),
        "max_documents": 25,
        "authorize_analysis": False,
    }


def test_backfill_analysis_requires_explicit_authorization(tmp_path: Path) -> None:
    services, recorder, _ = fake_services(tmp_path)

    code = main(
        [
            "backfill",
            "--as-of",
            "2026-08-24",
            "--max-documents",
            "25",
            "--authorize-analysis",
        ],
        services=services,
    )

    assert code == 0
    assert recorder.calls[0][1]["authorize_analysis"] is True


@pytest.mark.parametrize(
    "bad_date", ["2026-8-24", "20260824", "24-08-2026", "2026-02-30"]
)
def test_invalid_as_of_is_a_stable_input_error(
    tmp_path: Path, capsys, bad_date: str
) -> None:
    services, recorder, events = fake_services(tmp_path)

    code = main(["daily", "--as-of", bad_date], services=services)

    assert code == ExitCode.INVALID_INPUT
    assert not recorder.calls
    assert events["config_loads"] == 0
    assert capsys.readouterr().err == "research error: invalid input\n"


def test_help_returns_instead_of_raising_system_exit(capsys) -> None:
    assert main(["--help"]) == ExitCode.OK
    assert "portfolio-research" in capsys.readouterr().out


def test_invalid_credentials_do_not_invoke_workflow_or_leak_details(
    tmp_path: Path, capsys
) -> None:
    def reject(
        _command: str, _config: ResearchConfig, _requires_portfolio: bool
    ) -> None:
        raise ResearchConfigError("token=secret-value account=DU123456")

    services, recorder, _ = fake_services(tmp_path, validator=reject)

    code = main(["daily", "--as-of", "2026-08-24"], services=services)

    assert code == ExitCode.INVALID_CONFIG
    assert not recorder.calls
    captured = capsys.readouterr()
    assert captured.err == "research error: invalid configuration\n"
    assert "secret-value" not in captured.out + captured.err
    assert "DU123456" not in captured.out + captured.err


def test_operational_failure_does_not_leak_exception_internals(
    tmp_path: Path, capsys
) -> None:
    class FailingWorkflows(RecordingWorkflows):
        def run_daily(self, **kwargs: object) -> object:
            raise RuntimeError("NAV=9999 account DU123456 token secret-value")

    services, recorder, _ = fake_services(tmp_path, workflows=FailingWorkflows())

    code = main(["daily", "--as-of", "2026-08-24"], services=services)

    assert code == ExitCode.OPERATIONAL_FAILURE
    assert not recorder.calls
    captured = capsys.readouterr()
    assert captured.err == "research error: operational failure\n"
    assert "9999" not in captured.out + captured.err
    assert "DU123456" not in captured.out + captured.err
    assert "secret-value" not in captured.out + captured.err


def test_inner_workflow_busy_uses_busy_exit_code(tmp_path: Path, capsys) -> None:
    class BusyWorkflows(RecordingWorkflows):
        def run_daily(self, **kwargs: object) -> object:
            raise WorkflowBusy("private owner and workflow identifiers")

    services, _, _ = fake_services(tmp_path, workflows=BusyWorkflows())

    code = main(["daily", "--as-of", "2026-08-24"], services=services)

    assert code == ExitCode.BUSY
    assert capsys.readouterr().err == "research error: run already active\n"


def test_summary_prints_counts_not_private_identifiers_or_values(
    tmp_path: Path, capsys
) -> None:
    result = FakeResult(
        status="partial",
        report_ids=("report-private-123", "account-DU123456-NAV-9999"),
        omissions=("missing_claim_lineage",),
    )
    services, _, _ = fake_services(
        tmp_path, workflows=RecordingWorkflows(result)
    )

    assert main(["weekly", "--as-of", "2026-08-24"], services=services) == 0

    output = capsys.readouterr().out
    assert "reports=2" in output
    assert "omissions=1" in output
    assert "report-private" not in output
    assert "DU123456" not in output
    assert "9999" not in output


def test_blocked_required_output_has_explicit_exit_code(tmp_path: Path) -> None:
    services, _, _ = fake_services(
        tmp_path, workflows=RecordingWorkflows(FakeResult(status="blocked"))
    )
    assert (
        main(
            [
                "monthly",
                "--as-of",
                "2026-08-24",
                "--industry",
                "robotic-actuators",
            ],
            services=services,
        )
        == ExitCode.BLOCKED
    )


@pytest.mark.parametrize(
    ("status", "omissions", "expected"),
    [
        ("completed", (), ExitCode.OK),
        ("partial", ("missing_claim_lineage",), ExitCode.OK),
        ("partial", (), ExitCode.OPERATIONAL_FAILURE),
        ("running", (), ExitCode.OPERATIONAL_FAILURE),
        ("deferred", (), ExitCode.OPERATIONAL_FAILURE),
        ("failed", (), ExitCode.OPERATIONAL_FAILURE),
        ("unknown", (), ExitCode.OPERATIONAL_FAILURE),
    ],
)
def test_exit_status_matrix_is_fail_closed(
    tmp_path: Path,
    status: str,
    omissions: tuple[str, ...],
    expected: ExitCode,
) -> None:
    services, _, _ = fake_services(
        tmp_path,
        workflows=RecordingWorkflows(
            FakeResult(status=status, omissions=omissions)
        ),
    )

    assert main(["weekly", "--as-of", "2026-08-24"], services=services) == expected


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(status="partial", report_ids=(), omissions=[None]),
        WorkflowRunResult(
            workflow_id="wf-malformed-partial",
            workflow_kind=WorkflowKind.DAILY,
            status=WorkflowRunState.PARTIAL,
            completed_stages=(),
            report_ids=(),
            new_agent_runs=0,
            pending_tasks=(),
            omissions=("dry_run",),
            dry_run=True,
        ).model_copy(update={"omissions": [None]}),
        FakeResult(status="partial", omissions=()),
        FakeResult(status="completed", omissions=("dry_run",)),
    ],
)
def test_cli_rejects_malformed_partial_before_printing_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    result: object,
) -> None:
    services, _, _ = fake_services(
        tmp_path, workflows=RecordingWorkflows(result)
    )

    assert main(
        ["dry-run", "--as-of", "2026-08-24"], services=services
    ) == ExitCode.OPERATIONAL_FAILURE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "research error: operational failure\n"


def test_unrelated_commands_do_not_read_unused_secret_files(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("IBKR_FLEX_TOKEN_FILE", str(tmp_path / "missing-flex"))

    monthly = main(
        [
            "monthly",
            "--as-of",
            "2026-08-24",
            "--industry",
            "robotic-actuators",
        ]
    )
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(tmp_path / "missing-model"))
    backfill = main(
        ["backfill", "--as-of", "2026-08-24", "--max-documents", "5"]
    )

    assert monthly == ExitCode.BLOCKED
    assert backfill == ExitCode.BLOCKED


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            [
                "monthly",
                "--as-of",
                "2026-08-24",
                "--industry",
                "robotic-actuators",
            ],
            ExitCode.BLOCKED,
        ),
        (
            ["backfill", "--as-of", "2026-08-24", "--max-documents", "5"],
            ExitCode.BLOCKED,
        ),
        (
            [
                "backfill",
                "--as-of",
                "2026-08-24",
                "--max-documents",
                "5",
                "--authorize-analysis",
            ],
            ExitCode.BLOCKED,
        ),
        (
            [
                "regenerate",
                "--as-of",
                "2026-08-24",
                "--report-id",
                "missing-report",
            ],
            ExitCode.OPERATIONAL_FAILURE,
        ),
        (
            [
                "dry-run",
                "--as-of",
                "2026-08-24",
                "--synthetic-portfolio",
            ],
            ExitCode.OK,
        ),
    ],
)
def test_commands_ignore_unused_partial_flex_and_unreadable_model_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected: ExitCode,
) -> None:
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("IBKR_FLEX_TOKEN", "unused-partial-token")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(tmp_path / "missing-model"))

    assert main(argv) == expected


def test_required_command_still_rejects_unreadable_flex_secret(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("IBKR_FLEX_TOKEN_FILE", str(tmp_path / "missing-flex"))

    assert main(["daily", "--as-of", "2026-08-24"]) == ExitCode.INVALID_CONFIG


def test_monthly_requires_strict_industry_input(tmp_path: Path, capsys) -> None:
    services, recorder, events = fake_services(tmp_path)

    assert (
        main(["monthly", "--as-of", "2026-08-24"], services=services)
        == ExitCode.INVALID_INPUT
    )
    assert not recorder.calls
    assert events["config_loads"] == 0
    assert capsys.readouterr().err == "research error: invalid input\n"


def test_flex_credentials_are_only_required_for_real_portfolio_workflows(
    tmp_path: Path,
) -> None:
    config = research_config(tmp_path)

    for command in ("monthly", "backfill", "regenerate"):
        _validate_credentials(command, config, False)
    for command in ("daily", "weekly", "dry-run"):
        with pytest.raises(ResearchConfigError, match="IBKR Flex"):
            _validate_credentials(command, config, True)
    _validate_credentials("dry-run", config, False)


def portfolio_sync_result(
    *,
    as_of: datetime = datetime(2026, 8, 24, tzinfo=timezone.utc),
    stale: bool = False,
    snapshot_stale: bool | None = False,
    positions: tuple[Position, ...] = (),
    generated_at: datetime | None = None,
    evaluated_at: datetime | None = None,
) -> PortfolioSyncResult:
    evaluated_at = evaluated_at or (
        as_of + (timedelta(hours=25) if stale else timedelta())
    )
    generated_at = generated_at or as_of
    snapshot = PortfolioSnapshot(
        snapshot_id="snapshot-runtime-test",
        as_of=as_of,
        base_currency="USD",
        nav=Decimal("100.00"),
        cash=Decimal("100.00"),
        is_stale=snapshot_stale,
    )
    return PortfolioSyncResult(
        snapshot=snapshot,
        positions=positions,
        account_ref="acct_0123456789abcdef01234567",
        generated_at=generated_at,
        freshness=PortfolioFreshness(
            evaluated_at=evaluated_at,
            as_of=as_of,
            max_hours=24,
            age=evaluated_at - as_of,
            is_stale=stale,
        ),
    )


class PersistingFlexClient:
    def __init__(
        self,
        result: object | None = None,
        *,
        persist: bool = True,
        persisted_result: PortfolioSyncResult | None = None,
    ) -> None:
        self.calls = 0
        self.result = result or portfolio_sync_result()
        self.persist = persist
        self.persisted_result = persisted_result

    def sync(self, store: ResearchStore) -> object:
        self.calls += 1
        if self.persist and isinstance(self.result, PortfolioSyncResult):
            persisted = self.persisted_result or self.result
            store.insert_portfolio_snapshot(
                persisted.snapshot,
                persisted.positions,
                persisted.account_ref,
            )
        return self.result

    def close(self) -> None:
        return None


def test_default_composition_runs_real_orchestrator_and_persists_progress(
    tmp_path: Path, capsys
) -> None:
    flex = PersistingFlexClient()
    config = replace(
        research_config(tmp_path),
        ibkr_flex_token="read-only-token",
        ibkr_flex_query_id="query-id",
        ibkr_flex_account_salt="0123456789abcdef",
    )
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=lambda flex_config: flex,
            owner_id_factory=lambda: "runtime-test-owner",
        ),
    )

    code = main(["daily", "--as-of", "2026-08-24"], services=services)

    assert code == ExitCode.BLOCKED
    assert flex.calls == 1
    with ResearchStore(config.database_path).connect() as connection:
        run = connection.execute(
            "SELECT workflow_kind, state FROM workflow_runs"
        ).fetchall()
        tasks = connection.execute(
            "SELECT stage, state FROM workflow_tasks ORDER BY ordinal"
        ).fetchall()
    assert run == [("daily", "blocked")]
    assert tasks[:2] == [("portfolio", "completed"), ("ingestion", "failed")]
    assert "status=blocked" in capsys.readouterr().out


def _passing_quality_gate() -> QualityGateResult:
    return QualityGateResult(
        reason_codes=(),
        allowed_sections=("synthetic-analysis",),
        allowed_claim_ids=("synthetic-claim",),
        allow_event_report=True,
        allow_sizing=False,
        publication_verdict=PublicationVerdict.FINAL,
        review_verdict=ReviewVerdict.PASS,
        effective_rating=RecommendationRating.HOLD,
        inference_mode=InferenceMode.LOCAL_ONLY,
        inference_provider="ollama",
        inference_model="offline-fixture",
    )


def test_configured_stage_adapters_complete_daily_workflow(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []

    class Adapter:
        def run(self, context, control) -> StageOutcome:
            control.checkpoint()
            calls.append((context.stage, control.task_id))
            if context.stage == "materiality":
                return StageOutcome(
                    result_ref="materiality-configured",
                    material_event=True,
                )
            if context.stage == "event_update":
                return StageOutcome(
                    result_ref="report-configured",
                    report_id="report-configured",
                )
            if context.stage == "review":
                return StageOutcome(
                    result_ref="review-configured",
                    reviewer_verdict=ReviewVerdict.PASS,
                    quality_gate=_passing_quality_gate(),
                    published_claim_ids=("synthetic-claim",),
                )
            if context.stage == "publish":
                return StageOutcome(
                    result_ref="report-configured",
                    report_id="report-configured",
                    published_claim_ids=("synthetic-claim",),
                    publication_receipt_hash="f" * 64,
                )
            return StageOutcome(result_ref=f"{context.stage}-configured")

    flex = PersistingFlexClient()
    config = replace(
        research_config(tmp_path),
        ibkr_flex_token="read-only-token",
        ibkr_flex_query_id="query-id",
        ibkr_flex_account_salt="0123456789abcdef",
    )
    configured = {
        stage: Adapter()
        for stage in (
            "ingestion",
            "resolve",
            "materiality",
            "event_analysis",
            "event_update",
            "review",
            "publish",
        )
    }
    factories = RuntimeFactories(
        flex_client_factory=lambda _config: flex,
        owner_id_factory=lambda: "configured-runtime-owner",
        stage_adapters=configured,
    )
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=factories,
    )

    code = main(["daily", "--as-of", "2026-08-24"], services=services)

    assert code == ExitCode.OK
    assert [stage for stage, _ in calls] == list(configured)
    assert len({task_id for _, task_id in calls}) == len(calls)
    with ResearchStore(config.database_path).connect() as connection:
        assert connection.execute(
            "SELECT state FROM workflow_runs"
        ).fetchone() == ("completed",)
        assert connection.execute(
            "SELECT COUNT(*) FROM workflow_tasks WHERE state != 'completed'"
        ).fetchone() == (0,)
    with pytest.raises(TypeError):
        factories.stage_adapters["ingestion"] = Adapter()  # type: ignore[index]


def test_default_composition_persists_fail_closed_source_outcome(
    tmp_path: Path,
) -> None:
    class UnavailableFlexClient:
        def sync(self, _store: ResearchStore) -> object:
            raise FlexTransportError("private upstream failure")

        def close(self) -> None:
            return None

    config = replace(
        research_config(tmp_path),
        ibkr_flex_token="read-only-token",
        ibkr_flex_query_id="query-id",
        ibkr_flex_account_salt="0123456789abcdef",
    )
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=lambda _config: UnavailableFlexClient(),
            owner_id_factory=lambda: "unavailable-flex-owner",
        ),
    )

    code = main(["daily", "--as-of", "2026-08-24"], services=services)

    assert code == ExitCode.BLOCKED
    store = ResearchStore(config.database_path)
    with store.connect() as connection:
        workflow_id = connection.execute(
            "SELECT workflow_id FROM workflow_runs"
        ).fetchone()[0]
        assert connection.execute(
            "SELECT state FROM workflow_runs"
        ).fetchone() == ("blocked",)
        assert connection.execute(
            "SELECT stage, state, defer_reason FROM workflow_tasks "
            "ORDER BY ordinal LIMIT 1"
        ).fetchone() == (
            "portfolio",
            "failed",
            "required_stage_unavailable",
        )
    typed_task = ResearchOrchestrator(
        store,
        SimpleNamespace(run=lambda *_args: None),
        owner_id="reason-code-reader",
    ).list_tasks(workflow_id)[0]
    assert typed_task.defer_reason == "required_stage_unavailable"


@pytest.mark.parametrize(
    "result,persist",
    [
        (SimpleNamespace(snapshot=portfolio_sync_result().snapshot), True),
        (portfolio_sync_result(stale=True), True),
        (
            portfolio_sync_result(
                as_of=datetime(2026, 8, 23, tzinfo=timezone.utc)
            ),
            True,
        ),
        (
            portfolio_sync_result(
                as_of=datetime(2026, 8, 25, tzinfo=timezone.utc)
            ),
            True,
        ),
        (portfolio_sync_result(), False),
    ],
)
def test_portfolio_stage_rejects_untrusted_or_inconsistent_sync_results(
    tmp_path: Path,
    result: object,
    persist: bool,
) -> None:
    config = replace(
        research_config(tmp_path),
        ibkr_flex_token="read-only-token",
        ibkr_flex_query_id="query-id",
        ibkr_flex_account_salt="0123456789abcdef",
    )
    flex = PersistingFlexClient(result, persist=persist)
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=lambda _config: flex,
            owner_id_factory=lambda: "invalid-flex-owner",
        ),
    )

    code = main(["daily", "--as-of", "2026-08-24"], services=services)

    assert code == ExitCode.BLOCKED
    with ResearchStore(config.database_path).connect() as connection:
        assert connection.execute(
            "SELECT stage, state FROM workflow_tasks ORDER BY ordinal LIMIT 1"
        ).fetchone() == ("portfolio", "failed")


def test_portfolio_stage_accepts_real_flex_nullable_snapshot_freshness(
    tmp_path: Path,
) -> None:
    config = replace(
        research_config(tmp_path),
        ibkr_flex_token="read-only-token",
        ibkr_flex_query_id="query-id",
        ibkr_flex_account_salt="0123456789abcdef",
    )
    flex = PersistingFlexClient(
        portfolio_sync_result(
            snapshot_stale=None,
            generated_at=datetime(2026, 8, 24, 12, tzinfo=timezone.utc),
            evaluated_at=datetime(2026, 8, 24, 13, tzinfo=timezone.utc),
        )
    )
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=lambda _config: flex,
            owner_id_factory=lambda: "nullable-freshness-owner",
        ),
    )

    assert main(
        ["daily", "--as-of", "2026-08-24"], services=services
    ) == ExitCode.BLOCKED
    with ResearchStore(config.database_path).connect() as connection:
        assert connection.execute(
            "SELECT stage, state FROM workflow_tasks ORDER BY ordinal LIMIT 1"
        ).fetchone() == ("portfolio", "completed")


def test_portfolio_stage_compares_snapshot_date_in_utc(tmp_path: Path) -> None:
    local_timestamp = datetime(
        2026,
        8,
        24,
        23,
        30,
        tzinfo=timezone(timedelta(hours=-7)),
    )
    config = replace(
        research_config(tmp_path),
        ibkr_flex_token="read-only-token",
        ibkr_flex_query_id="query-id",
        ibkr_flex_account_salt="0123456789abcdef",
    )
    flex = PersistingFlexClient(portfolio_sync_result(as_of=local_timestamp))
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=lambda _config: flex,
            owner_id_factory=lambda: "utc-date-owner",
        ),
    )

    assert main(
        ["daily", "--as-of", "2026-08-24"], services=services
    ) == ExitCode.BLOCKED
    with ResearchStore(config.database_path).connect() as connection:
        assert connection.execute(
            "SELECT stage, state FROM workflow_tasks ORDER BY ordinal LIMIT 1"
        ).fetchone() == ("portfolio", "failed")


def test_portfolio_stage_rejects_mismatched_persisted_position_content(
    tmp_path: Path,
) -> None:
    snapshot_id = "snapshot-runtime-test"

    def position(quantity: str) -> Position:
        return Position(
            position_id="position-runtime-test",
            snapshot_id=snapshot_id,
            symbol="TEST",
            quantity=Decimal(quantity),
            market_value=Decimal("25.00"),
            currency="USD",
            security_id="security-runtime-test",
            asset_class="STK",
        )

    returned = portfolio_sync_result(positions=(position("1"),))
    persisted = portfolio_sync_result(positions=(position("2"),))
    flex = PersistingFlexClient(returned, persisted_result=persisted)
    config = replace(
        research_config(tmp_path),
        ibkr_flex_token="read-only-token",
        ibkr_flex_query_id="query-id",
        ibkr_flex_account_salt="0123456789abcdef",
    )
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=lambda _config: flex,
            owner_id_factory=lambda: "position-mismatch-owner",
        ),
    )

    assert main(
        ["daily", "--as-of", "2026-08-24"], services=services
    ) == ExitCode.BLOCKED
    with ResearchStore(config.database_path).connect() as connection:
        assert connection.execute(
            "SELECT stage, state FROM workflow_tasks ORDER BY ordinal LIMIT 1"
        ).fetchone() == ("portfolio", "failed")


def test_portfolio_stage_rejects_mismatched_persisted_security_identity(
    tmp_path: Path,
) -> None:
    snapshot_id = "snapshot-runtime-test"

    def position(conid: str) -> Position:
        return Position(
            position_id="position-runtime-test",
            snapshot_id=snapshot_id,
            symbol="TEST",
            quantity=Decimal("1"),
            market_value=Decimal("25.00"),
            currency="USD",
            security_id="security-runtime-test",
            asset_class="STK",
            conid=conid,
            security_id_type="CONID",
        )

    returned = portfolio_sync_result(positions=(position("4815747"),))
    persisted = portfolio_sync_result(positions=(position("DIFFERENT-CONID"),))
    flex = PersistingFlexClient(returned, persisted_result=persisted)
    config = replace(
        research_config(tmp_path),
        ibkr_flex_token="read-only-token",
        ibkr_flex_query_id="query-id",
        ibkr_flex_account_salt="0123456789abcdef",
    )
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=lambda _config: flex,
            owner_id_factory=lambda: "security-mismatch-owner",
        ),
    )

    assert main(
        ["daily", "--as-of", "2026-08-24"], services=services
    ) == ExitCode.BLOCKED
    with ResearchStore(config.database_path).connect() as connection:
        assert connection.execute(
            "SELECT stage, state FROM workflow_tasks ORDER BY ordinal LIMIT 1"
        ).fetchone() == ("portfolio", "failed")


def test_default_monthly_composition_never_requires_or_calls_flex(
    tmp_path: Path,
) -> None:
    config = research_config(tmp_path)

    def forbidden_flex(_config):
        raise AssertionError("monthly must not initialize Flex")

    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=forbidden_flex,
            owner_id_factory=lambda: "monthly-runtime-owner",
        ),
    )

    code = main(
        [
            "monthly",
            "--as-of",
            "2026-08-24",
            "--industry",
            "robotic-actuators",
        ],
        services=services,
    )

    assert code == ExitCode.BLOCKED
    with ResearchStore(config.database_path).connect() as connection:
        assert connection.execute(
            "SELECT workflow_kind, industry_key, state FROM workflow_runs"
        ).fetchone() == ("monthly", "robotic-actuators", "blocked")


def test_default_synthetic_dry_run_completes_offline_with_safe_report(
    tmp_path: Path,
) -> None:
    config = research_config(tmp_path)

    def forbidden_flex(_config):
        raise AssertionError("synthetic dry-run must not initialize Flex")

    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=forbidden_flex,
            owner_id_factory=lambda: "synthetic-runtime-owner",
        ),
    )

    code = main(
        [
            "dry-run",
            "--as-of",
            "2026-08-24",
            "--synthetic-portfolio",
        ],
        services=services,
    )

    assert code == ExitCode.OK
    with ResearchStore(config.database_path).connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM portfolio_snapshots"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT dry_run, state FROM workflow_runs"
        ).fetchone() == (1, "partial")
        report = connection.execute(
            "SELECT report_id, body, status, metadata_json FROM reports"
        ).fetchone()
    assert report is not None
    report_id, body, status, metadata_json = report
    metadata = json.loads(metadata_json)
    assert status == "reviewed"
    assert json.loads(body)["metadata"]["report_id"] == report_id
    assert metadata["schema"] == "trusted_report_payload/v1"
    assert metadata["payload_sha256"] == hashlib.sha256(
        body.encode("utf-8")
    ).hexdigest()
    assert metadata["pdf_status"] in {
        "rendered",
        "pdf_backend_unavailable",
        "pdf_render_failed",
    }
    artifacts = tuple(config.report_dir.glob("event_update-*.html"))
    assert len(artifacts) == 1
    html = artifacts[0].read_text(encoding="utf-8")
    assert "Institutional research" in html
    assert "<script" not in html.casefold()
    assert "DU123456" not in html
    assert "NAV: USD" not in html


def test_synthetic_cli_suppresses_pdf_backend_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    config = research_config(tmp_path)

    def noisy_pdf(
        _renderer: ReportRenderer, destination: Path, _html: str
    ) -> None:
        os.write(2, b"private PDF diagnostic NAV=9999 DU123456\n")
        destination.write_bytes(b"%PDF-test")

    monkeypatch.setattr(ReportRenderer, "_atomic_pdf", noisy_pdf)
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=lambda _config: pytest.fail("Flex was called"),
            owner_id_factory=lambda: "quiet-pdf-owner",
        ),
    )

    assert main(
        ["dry-run", "--as-of", "2026-08-24", "--synthetic-portfolio"],
        services=services,
    ) == ExitCode.OK
    captured = capfd.readouterr()
    assert captured.out == (
        "research command=dry-run status=partial reports=1 omissions=1\n"
    )
    assert captured.err == ""


def test_regenerate_replays_existing_run_without_workflow_or_source_calls(
    tmp_path: Path,
) -> None:
    config = research_config(tmp_path)
    flex_calls: list[str] = []
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=lambda _config: flex_calls.append("flex"),
            owner_id_factory=lambda: "regenerate-runtime-owner",
        ),
    )
    assert main(
        [
            "monthly",
            "--as-of",
            "2026-08-24",
            "--industry",
            "robotic-actuators",
        ],
        services=services,
    ) == ExitCode.BLOCKED
    with ResearchStore(config.database_path).connect() as connection:
        workflow_id = connection.execute(
            "SELECT workflow_id FROM workflow_runs"
        ).fetchone()[0]
        before = connection.execute(
            "SELECT COUNT(*) FROM workflow_runs"
        ).fetchone()

    code = main(
        [
            "regenerate",
            "--as-of",
            "2026-08-24",
            "--run-id",
            workflow_id,
        ],
        services=services,
    )

    assert code == ExitCode.BLOCKED
    assert flex_calls == []
    with ResearchStore(config.database_path).connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM workflow_runs"
        ).fetchone() == before


def _seed_trusted_synthetic_report(
    config: ResearchConfig,
) -> tuple[str, str, str, dict[str, object]]:
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=lambda _config: pytest.fail("Flex was called"),
            owner_id_factory=lambda: "regenerate-seed-owner",
        ),
    )
    assert main(
        [
            "dry-run",
            "--as-of",
            "2026-08-24",
            "--synthetic-portfolio",
        ],
        services=services,
    ) == ExitCode.OK
    with ResearchStore(config.database_path).connect() as connection:
        workflow_id = connection.execute(
            "SELECT workflow_id FROM workflow_runs"
        ).fetchone()[0]
        report_id, body, metadata_json = connection.execute(
            "SELECT report_id, body, metadata_json FROM reports"
        ).fetchone()
    return workflow_id, report_id, body, json.loads(metadata_json)


def _clear_report_artifacts(config: ResearchConfig) -> None:
    for path in config.report_dir.iterdir():
        if path.is_file():
            path.unlink()


def test_regenerate_rebuilds_typed_attested_report_through_renderer(
    tmp_path: Path,
) -> None:
    config = research_config(tmp_path)
    _, report_id, body, _ = _seed_trusted_synthetic_report(config)
    _clear_report_artifacts(config)
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=lambda _config: pytest.fail("Flex was called"),
            owner_id_factory=lambda: "regenerate-artifact-owner",
        ),
    )

    code = main(
        [
            "regenerate",
            "--as-of",
            "2026-08-24",
            "--report-id",
            report_id,
        ],
        services=services,
    )

    assert code == ExitCode.OK
    artifact_sets = tuple(config.report_dir.glob("regenerated-*"))
    assert len(artifact_sets) == 1
    artifacts = tuple(artifact_sets[0].iterdir())
    html = next(path for path in artifacts if path.suffix == ".html")
    assert html.read_text(encoding="utf-8") != body
    assert "Institutional research" in html.read_text(encoding="utf-8")
    assert any(path.suffix == ".pdf" for path in artifacts) or any(
        path.name == "pdf-status.txt" for path in artifacts
    )


def test_regenerate_rejects_incomplete_existing_artifact_set(
    tmp_path: Path,
) -> None:
    config = research_config(tmp_path)
    _, report_id, _, _ = _seed_trusted_synthetic_report(config)
    _clear_report_artifacts(config)
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=lambda _config: pytest.fail("Flex was called"),
            owner_id_factory=lambda: "regenerate-completeness-owner",
        ),
    )
    argv = [
        "regenerate",
        "--as-of",
        "2026-08-24",
        "--report-id",
        report_id,
    ]
    assert main(argv, services=services) == ExitCode.OK
    artifact_set = next(config.report_dir.glob("regenerated-*"))
    secondary = next(path for path in artifact_set.iterdir() if path.suffix != ".html")
    secondary.unlink()

    assert main(argv, services=services) == ExitCode.OPERATIONAL_FAILURE


def test_regenerate_rejects_corrupt_existing_pdf_artifact(
    tmp_path: Path,
) -> None:
    config = research_config(tmp_path)
    _, report_id, _, _ = _seed_trusted_synthetic_report(config)
    _clear_report_artifacts(config)
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=lambda _config: pytest.fail("Flex was called"),
            owner_id_factory=lambda: "regenerate-integrity-owner",
        ),
    )
    argv = [
        "regenerate",
        "--as-of",
        "2026-08-24",
        "--report-id",
        report_id,
    ]
    assert main(argv, services=services) == ExitCode.OK
    artifact_set = next(config.report_dir.glob("regenerated-*"))
    pdf = next(iter(artifact_set.glob("*.pdf")), None)
    if pdf is None:
        pytest.skip("PDF backend is unavailable")
    assert main(argv, services=services) == ExitCode.OK
    pdf.write_bytes(b"%PDF-corrupt")

    assert main(argv, services=services) == ExitCode.OPERATIONAL_FAILURE


@pytest.mark.parametrize(
    "mutation",
    [
        "malformed_json",
        "missing_attestation",
        "wrong_attestation",
        "forged_provenance",
        "active_script",
        "private_portfolio_values",
        "wrong_as_of",
        "unreviewed",
    ],
)
def test_regenerate_rejects_untrusted_or_unsafe_stored_report_without_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    config = research_config(tmp_path)
    if mutation == "wrong_as_of":
        original_report = RuntimeStageRunner._synthetic_event_report

        def mismatched_report(self, context):
            report = original_report(self, context)
            metadata = report.metadata.model_copy(
                update={"as_of": datetime(2026, 8, 25, tzinfo=timezone.utc)}
            )
            return report.model_copy(update={"metadata": metadata})

        monkeypatch.setattr(
            RuntimeStageRunner, "_synthetic_event_report", mismatched_report
        )
    workflow_id, report_id, body, metadata = _seed_trusted_synthetic_report(config)
    store = ResearchStore(config.database_path)
    if mutation == "malformed_json":
        body = "{not-json"
    elif mutation == "missing_attestation":
        metadata = {}
    elif mutation == "wrong_attestation":
        metadata["payload_sha256"] = "0" * 64
    elif mutation == "forged_provenance":
        metadata["workflow_id"] = "wf-forged"
    elif mutation == "active_script":
        payload = json.loads(body)
        payload["metadata"]["title"] = "<script>alert('unsafe')</script>"
        body = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        metadata["payload_sha256"] = hashlib.sha256(
            body.encode("utf-8")
        ).hexdigest()
    elif mutation == "private_portfolio_values":
        portfolio_ref = str(metadata["portfolio_snapshot_ref"])
        store.insert_portfolio_snapshot(
            PortfolioSnapshot(
                snapshot_id=portfolio_ref,
                as_of=datetime(2026, 8, 24, tzinfo=timezone.utc),
                base_currency="USD",
                nav=Decimal("4321.99"),
                cash=Decimal("111.11"),
                is_stale=False,
            ),
            (),
            "acct_0123456789abcdef01234567",
        )
        payload = json.loads(body)
        payload["thesis"]["body"] = (
            "Observed figure 4321.99 and account acct_0123456789abcdef01234567."
        )
        body = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        metadata["payload_sha256"] = hashlib.sha256(
            body.encode("utf-8")
        ).hexdigest()
    status = "draft" if mutation == "unreviewed" else "reviewed"
    with store.transaction() as connection:
        connection.execute(
            "UPDATE reports SET body = ?, metadata_json = ?, status = ? "
            "WHERE report_id = ?",
            (
                body,
                json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                status,
                report_id,
            ),
        )
    _clear_report_artifacts(config)
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=lambda _config: pytest.fail("Flex was called"),
            owner_id_factory=lambda: f"regenerate-reject-{mutation}",
        ),
    )

    code = main(
        [
            "regenerate",
            "--as-of",
            "2026-08-24",
            "--report-id",
            report_id,
        ],
        services=services,
    )

    assert code == ExitCode.OPERATIONAL_FAILURE
    assert tuple(config.report_dir.iterdir()) == ()


def test_second_holder_cannot_acquire_active_lease_and_release_allows_next(
    migrated_store: ResearchStore,
) -> None:
    first = RunLease.acquire(migrated_store, "research-run", ttl_seconds=30)

    with pytest.raises(RunAlreadyActive):
        RunLease.acquire(migrated_store, "research-run", ttl_seconds=30)

    assert first.release() is True
    replacement = RunLease.acquire(
        migrated_store, "research-run", ttl_seconds=30
    )
    assert replacement.release() is True


def test_run_lease_default_is_short_and_ttl_is_bounded(
    migrated_store: ResearchStore,
) -> None:
    lease = RunLease.acquire(migrated_store, "short-default")
    try:
        assert lease._ttl_seconds <= 30
    finally:
        lease.release()

    with pytest.raises(ValueError, match="at most"):
        RunLease.acquire(migrated_store, "too-long", ttl_seconds=31)


def test_cli_requires_confident_lease_cleanup_before_success_summary(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services, _, _ = fake_services(tmp_path)

    class UncertainLease:
        def assert_current(self) -> None:
            return None

        def release(self) -> bool:
            return False

    monkeypatch.setattr(
        RunLease,
        "acquire",
        classmethod(lambda cls, *args, **kwargs: UncertainLease()),
    )

    code = main(["weekly", "--as-of", "2026-08-24"], services=services)

    output = capsys.readouterr()
    assert code == ExitCode.OPERATIONAL_FAILURE
    assert "research command=" not in output.out
    assert output.err.strip() == "research error: operational failure"


def test_default_runtime_receives_global_lease_loss_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flex_calls: list[str] = []
    registered_events: list[object] = []

    class LostLease:
        def register_cancellation_event(self, event) -> None:
            registered_events.append(event)
            event.set()

        def unregister_cancellation_event(self, event) -> None:
            return None

        def assert_current(self) -> None:
            raise RunLeaseLost("lost")

        def release(self) -> bool:
            return False

    monkeypatch.setattr(
        RunLease,
        "acquire",
        classmethod(lambda cls, *args, **kwargs: LostLease()),
    )
    config = replace(
        research_config(tmp_path),
        ibkr_flex_token="read-only-token",
        ibkr_flex_query_id="query-id",
        ibkr_flex_account_salt="0123456789abcdef",
    )
    services = default_services(
        config_loader=lambda: config,
        runtime_factories=RuntimeFactories(
            flex_client_factory=lambda _config: flex_calls.append("flex"),
            owner_id_factory=lambda: "global-lease-loss-owner",
        ),
    )

    assert main(
        ["daily", "--as-of", "2026-08-24"], services=services
    ) == ExitCode.OPERATIONAL_FAILURE
    assert registered_events
    assert flex_calls == []


def test_cli_redacts_release_database_errors_and_returns_nonzero(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services, _, _ = fake_services(tmp_path)

    class BrokenReleaseLease:
        def assert_current(self) -> None:
            return None

        def release(self) -> bool:
            raise sqlite3.DatabaseError(
                "NAV=9999 account=DU123456 token=private"
            )

    monkeypatch.setattr(
        RunLease,
        "acquire",
        classmethod(lambda cls, *args, **kwargs: BrokenReleaseLease()),
    )

    code = main(["weekly", "--as-of", "2026-08-24"], services=services)

    output = capsys.readouterr()
    assert code == ExitCode.OPERATIONAL_FAILURE
    assert "research command=" not in output.out
    assert output.err.strip() == "research error: operational failure"
    assert "9999" not in output.err
    assert "DU123456" not in output.err
    assert "private" not in output.err


def test_run_lease_release_never_joins_heartbeat_without_a_bound(
    migrated_store: ResearchStore,
) -> None:
    lease = RunLease.acquire(
        migrated_store, "bounded-heartbeat", ttl_seconds=0.09
    )
    lease._stop.set()
    assert lease._thread is not None
    lease._thread.join(timeout=1)
    joins: list[float | None] = []

    class StuckThread:
        def join(self, timeout: float | None = None) -> None:
            joins.append(timeout)

        def is_alive(self) -> bool:
            return True

    lease._thread = StuckThread()  # type: ignore[assignment]

    assert lease.release() is False
    assert len(joins) == 1
    assert joins[0] is not None
    assert joins[0] <= 1


def test_expired_stale_holder_cannot_release_replacement(tmp_path: Path) -> None:
    now = [datetime(2026, 8, 24, tzinfo=timezone.utc)]
    store = ResearchStore(tmp_path / "lease.db", clock=lambda: now[0])
    store.migrate()
    stale = RunLease.acquire(store, "research-run", ttl_seconds=10)
    now[0] += timedelta(seconds=10)
    current = RunLease.acquire(store, "research-run", ttl_seconds=10)

    with pytest.raises(RunLeaseLost):
        stale.assert_current()
    stale.release()
    with pytest.raises(RunAlreadyActive):
        RunLease.acquire(store, "research-run", ttl_seconds=10)

    current.release()


def test_run_lease_renews_beyond_original_ttl(migrated_store: ResearchStore) -> None:
    lease = RunLease.acquire(
        migrated_store, "research-heartbeat", ttl_seconds=0.09
    )
    try:
        time.sleep(0.24)
        lease.assert_current()
        with pytest.raises(RunAlreadyActive):
            RunLease.acquire(
                migrated_store, "research-heartbeat", ttl_seconds=0.09
            )
    finally:
        lease.release()


class RecordingScheduler:
    def __init__(self) -> None:
        self.jobs: list[tuple[object, dict[str, object]]] = []

    def add_job(self, function, **kwargs: object) -> None:
        self.jobs.append((function, kwargs))


def test_scheduler_registration_value_error_is_stable(tmp_path: Path) -> None:
    class RejectingScheduler:
        def add_job(self, _function, **_kwargs: object) -> None:
            raise ValueError("private scheduler diagnostic")

    with pytest.raises(RuntimeError) as exc_info:
        build_scheduler(
            research_config(tmp_path),
            scheduler_factory=RejectingScheduler,
        )

    assert str(exc_info.value) == "scheduler registration failed"
    assert exc_info.value.__cause__ is None


def test_scheduler_registers_typed_cadences_and_nonoverlap_options(
    tmp_path: Path,
) -> None:
    config = research_config(tmp_path)
    services, recorder, events = fake_services(tmp_path)
    scheduler = RecordingScheduler()

    built = build_scheduler(
        config,
        services=services,
        scheduler_factory=lambda: scheduler,
        clock=lambda: datetime(2026, 8, 24, tzinfo=timezone.utc),
    )

    assert built is scheduler
    assert events["config_loads"] == 0
    assert events["store_opens"] == 0
    assert [job[1]["id"] for job in scheduler.jobs] == [
        "research-daily",
        "research-weekly",
        "research-monthly",
    ]
    for (_, options), cadence in zip(
        scheduler.jobs,
        (config.daily_schedule, config.weekly_schedule, config.monthly_schedule),
        strict=True,
    ):
        assert options["trigger"] == "cron"
        for name, value in cadence.as_kwargs().items():
            assert options[name] == value
        assert options["max_instances"] == 1
        assert options["coalesce"] is True
        assert options["replace_existing"] is True

    function, options = scheduler.jobs[0]
    function(*options["args"])
    assert [name for name, _ in recorder.calls] == ["daily"]
    monthly_function, monthly_options = scheduler.jobs[2]
    monthly_function(*monthly_options["args"])
    assert recorder.calls[-1][1]["industry_key"] == "robotic-actuators"


def test_scheduler_derives_as_of_from_aware_utc_clock(tmp_path: Path) -> None:
    config = research_config(tmp_path)
    services, recorder, _ = fake_services(tmp_path)
    scheduler = RecordingScheduler()
    build_scheduler(
        config,
        services=services,
        scheduler_factory=lambda: scheduler,
        clock=lambda: datetime(2026, 8, 25, 0, 5, tzinfo=timezone.utc),
    )

    function, options = scheduler.jobs[0]
    function(*options["args"])

    assert recorder.calls[0][1]["as_of"] == datetime(
        2026, 8, 25, tzinfo=timezone.utc
    )


def test_scheduler_rejects_naive_clock(tmp_path: Path) -> None:
    config = research_config(tmp_path)
    services, _, _ = fake_services(tmp_path)
    scheduler = RecordingScheduler()
    build_scheduler(
        config,
        services=services,
        scheduler_factory=lambda: scheduler,
        clock=lambda: datetime(2026, 8, 24),
    )

    function, options = scheduler.jobs[0]
    with pytest.raises(SchedulerClockError):
        function(*options["args"])


def test_legacy_main_selects_pipeline_only_when_research_enabled(monkeypatch) -> None:
    import news_bot.main as newsletter_main

    calls: list[str] = []
    disabled = SimpleNamespace(enabled=False)
    assert newsletter_main.main(
        research_config_loader=lambda: disabled,
        research_entrypoint=lambda argv: calls.append("research") or 0,
        legacy_runner=lambda: calls.append("legacy"),
    ) == 0
    assert calls == ["legacy"]

    calls.clear()
    enabled = SimpleNamespace(enabled=True)
    assert newsletter_main.main(
        research_config_loader=lambda: enabled,
        research_entrypoint=lambda argv: calls.append(" ".join(argv)) or 0,
        legacy_runner=lambda: calls.append("legacy"),
        today=lambda: datetime(2026, 8, 24, tzinfo=timezone.utc).date(),
    ) == 0
    assert calls == ["daily --as-of 2026-08-24"]


def test_research_main_default_date_uses_aware_utc_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import news_bot.main as newsletter_main

    class FixedDatetime:
        @classmethod
        def now(cls, tz):
            assert tz is timezone.utc
            return datetime(2026, 8, 25, 0, 5, tzinfo=timezone.utc)

    monkeypatch.setattr(newsletter_main, "datetime", FixedDatetime)
    calls: list[str] = []

    code = newsletter_main.main(
        research_config_loader=lambda: SimpleNamespace(enabled=True),
        research_entrypoint=lambda argv: calls.append(" ".join(argv)) or 0,
        legacy_runner=lambda: calls.append("legacy"),
    )

    assert code == 0
    assert calls == ["daily --as-of 2026-08-25"]


def test_disabled_legacy_path_does_not_load_unrelated_research_secrets(
    monkeypatch,
) -> None:
    import news_bot.main as newsletter_main

    monkeypatch.setenv("RESEARCH_ENABLED", "false")
    monkeypatch.setenv(
        "IBKR_FLEX_TOKEN_FILE", "/definitely/missing/private/secret"
    )
    calls: list[str] = []

    code = newsletter_main.main(
        research_entrypoint=lambda argv: calls.append("research") or 0,
        legacy_runner=lambda: calls.append("legacy"),
    )

    assert code == 0
    assert calls == ["legacy"]


def test_research_main_does_not_log_entrypoint_exception_internals(
    caplog,
) -> None:
    import news_bot.main as newsletter_main

    def fail(_argv: Sequence[str]) -> int:
        raise RuntimeError("NAV=9999 account=DU123456 token=secret")

    code = newsletter_main.main(
        research_config_loader=lambda: SimpleNamespace(enabled=True),
        research_entrypoint=fail,
        legacy_runner=lambda: None,
    )

    assert code == 1
    assert "9999" not in caplog.text
    assert "DU123456" not in caplog.text
    assert "secret" not in caplog.text


def test_run_script_uses_project_venv_interpreter_directly() -> None:
    script = Path("scripts/run.sh").read_text(encoding="utf-8")
    assert '"$PROJECT_DIR/venv/bin/python" -m news_bot.main' in script
    assert "source venv/bin/activate" not in script
    assert "\npython -m news_bot.main" not in script
