"""Tests for safe research workflow CLI and scheduling boundaries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from news_bot.research.cli import CliServices, ExitCode, main
from news_bot.research.config import CronSchedule, ResearchConfig, ResearchConfigError
from news_bot.research.models import InferenceMode
from news_bot.research.orchestrator import WorkflowBusy
from news_bot.research.scheduler import RunAlreadyActive, RunLease, build_scheduler
from news_bot.research.store import ResearchStore


@dataclass
class FakeResult:
    status: str = "completed"
    report_ids: tuple[str, ...] = ("report-private-123",)
    omissions: tuple[str, ...] = ()


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

    def validate(command: str, loaded: ResearchConfig) -> None:
        cast_validations = events["validations"]
        assert isinstance(cast_validations, list)
        cast_validations.append(command)
        if validator is not None:
            validator(command, loaded)

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

    exit_code = main(
        [command, "--as-of", "2026-08-24"], services=services
    )

    assert exit_code == ExitCode.OK
    assert [name for name, _ in recorder.calls] == [workflow]
    assert events == {
        "config_loads": 1,
        "store_opens": 1,
        "workflow_builds": 1,
        "validations": [command],
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
            },
        )
    ]


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
    def reject(_command: str, _config: ResearchConfig) -> None:
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
        report_ids=("report-private-123", "account-DU123456-NAV-9999"),
        omissions=("portfolio_value_9999",),
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
        main(["monthly", "--as-of", "2026-08-24"], services=services)
        == ExitCode.BLOCKED
    )


def test_second_holder_cannot_acquire_active_lease_and_release_allows_next(
    migrated_store: ResearchStore,
) -> None:
    first = RunLease.acquire(migrated_store, "research-run", ttl_seconds=3600)

    with pytest.raises(RunAlreadyActive):
        RunLease.acquire(migrated_store, "research-run", ttl_seconds=3600)

    first.release()
    replacement = RunLease.acquire(
        migrated_store, "research-run", ttl_seconds=3600
    )
    replacement.release()


def test_expired_stale_holder_cannot_release_replacement(tmp_path: Path) -> None:
    now = [datetime(2026, 8, 24, tzinfo=timezone.utc)]
    store = ResearchStore(tmp_path / "lease.db", clock=lambda: now[0])
    store.migrate()
    stale = RunLease.acquire(store, "research-run", ttl_seconds=10)
    now[0] += timedelta(seconds=10)
    current = RunLease.acquire(store, "research-run", ttl_seconds=10)

    stale.release()
    with pytest.raises(RunAlreadyActive):
        RunLease.acquire(store, "research-run", ttl_seconds=10)

    current.release()


class RecordingScheduler:
    def __init__(self) -> None:
        self.jobs: list[tuple[object, dict[str, object]]] = []

    def add_job(self, function, **kwargs: object) -> None:
        self.jobs.append((function, kwargs))


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
        today=lambda: datetime(2026, 8, 24, tzinfo=timezone.utc).date(),
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
