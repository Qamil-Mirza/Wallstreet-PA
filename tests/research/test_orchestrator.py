"""Durability and safety tests for dependency-aware research workflows."""

from __future__ import annotations

import json
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Lock

import pytest
from pydantic import ValidationError

from news_bot.research.models import AgentRole, ReviewVerdict
from news_bot.research.orchestrator import (
    DurableTaskView,
    RecommendationTrigger,
    ResearchOrchestrator,
    StageContext,
    StageOutcome,
    WorkflowKind,
    WorkflowRunResult,
    WorkflowRunState,
    WorkflowTaskState,
    WorkflowBusy,
    WorkflowConflict,
)
from news_bot.research.quality import (
    GateReasonCode,
    PublicationVerdict,
    QualityGateResult,
)
from news_bot.research.models import InferenceMode, RecommendationRating
from news_bot.research.store import ResearchStore


def utc(day: int = 24, hour: int = 12) -> datetime:
    return datetime(2026, 8, day, hour, tzinfo=timezone.utc)


class RecordingRunner:
    """Network-free typed stage service used by orchestration tests."""

    def __init__(
        self,
        outcomes: dict[str, StageOutcome | list[StageOutcome | Exception]] | None = None,
    ) -> None:
        self.outcomes = outcomes or {}
        self.calls: list[StageContext] = []
        self._lock = Lock()

    def run(self, context: StageContext) -> StageOutcome:
        with self._lock:
            self.calls.append(context)
            configured = self.outcomes.get(context.stage)
            if isinstance(configured, list):
                value = configured.pop(0)
            elif configured is not None:
                value = configured
            elif context.stage == "publish":
                value = StageOutcome(
                    result_ref=f"report-{context.workflow_id[-12:]}",
                    report_id=f"report-{context.workflow_id[-12:]}",
                )
            else:
                value = StageOutcome(result_ref=f"result-{context.task_id[-12:]}")
        if isinstance(value, BaseException):
            raise value
        return value


@pytest.fixture
def store(tmp_path: Path) -> ResearchStore:
    result = ResearchStore(tmp_path / "research.db")
    result.migrate()
    return result


@pytest.fixture
def runner() -> RecordingRunner:
    return RecordingRunner()


@pytest.fixture
def orchestrator(store: ResearchStore, runner: RecordingRunner) -> ResearchOrchestrator:
    return ResearchOrchestrator(store, runner, owner_id="scheduler-a")


def test_contracts_are_immutable_and_json_roundtrippable() -> None:
    task = DurableTaskView(
        task_id="task-1", idempotency_key="a" * 64, stage="portfolio",
        dependency_ids=(), state=WorkflowTaskState.PENDING, attempt_count=0,
        max_attempts=2,
    )
    result = WorkflowRunResult(
        workflow_id="workflow-1", workflow_kind=WorkflowKind.DAILY,
        status=WorkflowRunState.RUNNING, completed_stages=(), report_ids=(),
        new_agent_runs=0, pending_tasks=(task,), omissions=(), dry_run=False,
    )

    assert WorkflowRunResult.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValidationError):
        result.status = WorkflowRunState.COMPLETED  # type: ignore[misc]
    with pytest.raises(ValidationError):
        WorkflowRunResult.model_validate(
            {**result.model_dump(), "workflow_id": "account-U123", "nav": "4999"}
        )


def test_daily_run_does_not_recompute_all_ratings(
    orchestrator: ResearchOrchestrator, runner: RecordingRunner
) -> None:
    result = orchestrator.run_daily(as_of=utc(), source_hashes=("a" * 64,))

    assert result.completed_stages == (
        "portfolio", "ingestion", "materiality", "event_update",
    )
    assert "full_recommendation_refresh" not in result.completed_stages
    assert "full_recommendation_refresh" not in [call.stage for call in runner.calls]


def test_same_source_and_period_is_idempotent(
    orchestrator: ResearchOrchestrator, runner: RecordingRunner
) -> None:
    first = orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    calls = len(runner.calls)
    second = orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))

    assert second.workflow_id == first.workflow_id
    assert second.report_ids == first.report_ids
    assert second.new_agent_runs == 0
    assert len(runner.calls) == calls


def test_different_source_hash_creates_a_distinct_run(
    orchestrator: ResearchOrchestrator,
) -> None:
    first = orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    second = orchestrator.run_weekly(as_of=utc(), source_hashes=("b" * 64,))

    assert second.workflow_id != first.workflow_id


def test_weekly_revise_returns_task_to_fundamental_analyst(store: ResearchStore) -> None:
    runner = RecordingRunner({"review": StageOutcome(
        reviewer_verdict=ReviewVerdict.REVISE,
        originating_role=AgentRole.FUNDAMENTAL_ANALYST,
        defer_reason="strengthen valuation support",
    )})
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")

    result = orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))

    assert result.status is WorkflowRunState.BLOCKED
    returned = result.pending_tasks[0]
    assert returned.assigned_role is AgentRole.FUNDAMENTAL_ANALYST
    assert returned.originating_role is AgentRole.FUNDAMENTAL_ANALYST
    assert returned.defer_reason == "strengthen valuation support"
    assert returned.dependency_ids
    assert "publish" not in [call.stage for call in runner.calls]


def test_workflow_dags_persist_explicit_dependencies(
    orchestrator: ResearchOrchestrator,
) -> None:
    result = orchestrator.run_monthly(
        as_of=utc(), source_hashes=("a" * 64,), industry_key="robotic-actuators"
    )
    tasks = {task.stage: task for task in orchestrator.list_tasks(result.workflow_id)}

    assert set(tasks) == {
        "public_signals", "emerging_map", "industry_refresh", "review", "publish",
    }
    assert tasks["emerging_map"].dependency_ids == (tasks["public_signals"].task_id,)
    assert tasks["industry_refresh"].dependency_ids == (tasks["emerging_map"].task_id,)
    assert tasks["review"].dependency_ids == (tasks["industry_refresh"].task_id,)
    assert tasks["publish"].dependency_ids == (tasks["review"].task_id,)


def test_monthly_refreshes_exactly_one_explicit_industry(store: ResearchStore) -> None:
    runner = RecordingRunner()
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")
    orchestrator.run_monthly(
        as_of=utc(), source_hashes=("a" * 64,), industry_key="robotic-actuators"
    )

    refreshes = [call for call in runner.calls if call.stage == "industry_refresh"]
    assert len(refreshes) == 1
    assert refreshes[0].industry_key == "robotic-actuators"


def test_backfill_is_bounded_and_evidence_only_by_default(
    orchestrator: ResearchOrchestrator, runner: RecordingRunner
) -> None:
    result = orchestrator.run_backfill(
        as_of=utc(), source_hashes=("a" * 64, "b" * 64), max_documents=2
    )

    assert result.completed_stages == ("historical_documents", "evidence")
    assert [call.stage for call in runner.calls] == ["historical_documents", "evidence"]


@pytest.mark.parametrize("maximum", [0, -1, True, 1001])
def test_backfill_rejects_unsafe_bounds(
    orchestrator: ResearchOrchestrator, maximum: int
) -> None:
    with pytest.raises((TypeError, ValueError), match="max_documents"):
        orchestrator.run_backfill(
            as_of=utc(), source_hashes=("a" * 64,), max_documents=maximum
        )


def test_backfill_analysis_requires_explicit_authorization(store: ResearchStore) -> None:
    runner = RecordingRunner()
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")
    orchestrator.run_backfill(
        as_of=utc(), source_hashes=("a" * 64,), max_documents=1,
        authorize_analysis=True,
    )

    assert [call.stage for call in runner.calls] == [
        "historical_documents", "evidence", "analysis", "review", "publish"
    ]


def test_all_datetimes_must_be_strict_utc(orchestrator: ResearchOrchestrator) -> None:
    with pytest.raises(ValueError, match="UTC"):
        orchestrator.run_daily(
            as_of=datetime(2026, 8, 24, 12), source_hashes=("a" * 64,)
        )
    with pytest.raises(ValueError, match="UTC"):
        orchestrator.run_daily(
            as_of=datetime(2026, 8, 24, 12, tzinfo=timezone(timedelta(hours=1))),
            source_hashes=("a" * 64,),
        )


def test_migration_008_has_foreign_keys_and_supporting_indexes(store: ResearchStore) -> None:
    with store.connect() as connection:
        version = connection.execute(
            "SELECT name FROM schema_migrations WHERE version = 8"
        ).fetchone()
        fk_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        dependency_fks = connection.execute(
            "PRAGMA foreign_key_list(workflow_task_dependencies)"
        ).fetchall()
        task_indexes = connection.execute("PRAGMA index_list(workflow_tasks)").fetchall()

    assert version == ("008_durable_workflow_orchestration.sql",)
    assert fk_errors == []
    assert len({row[0] for row in dependency_fks}) == 3
    assert any(row[1] == "idx_workflow_tasks_runnable" for row in task_indexes)


def test_daily_material_event_runs_analysis_review_and_publish(store: ResearchStore) -> None:
    runner = RecordingRunner({"materiality": StageOutcome(material_event=True)})
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")

    result = orchestrator.run_daily(as_of=utc(), source_hashes=("a" * 64,))

    assert [call.stage for call in runner.calls] == [
        "portfolio", "ingestion", "resolve", "materiality", "event_analysis",
        "event_update", "review", "publish",
    ]
    assert result.status is WorkflowRunState.COMPLETED
    assert len(result.report_ids) == 1


def test_transient_stage_failure_retries_with_bounded_attempts(store: ResearchStore) -> None:
    runner = RecordingRunner({
        "changed_theses": [RuntimeError("private provider detail"), StageOutcome(result_ref="thesis-1")]
    })
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")

    result = orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    task = next(item for item in orchestrator.list_tasks(result.workflow_id)
                if item.stage == "changed_theses")

    assert result.status is WorkflowRunState.COMPLETED
    assert task.attempt_count == 2
    assert "private provider detail" not in json.dumps(result.model_dump(mode="json"))


def test_exhausted_stage_failure_is_safe_and_does_not_publish(store: ResearchStore) -> None:
    runner = RecordingRunner({
        "selected_recommendations": [RuntimeError("secret one"), RuntimeError("secret two")]
    })
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")

    result = orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))

    assert result.status is WorkflowRunState.FAILED
    assert "publish" not in [call.stage for call in runner.calls]
    assert "secret" not in result.model_dump_json()


def test_budget_defer_stops_monthly_frontier_work(store: ResearchStore) -> None:
    runner = RecordingRunner({
        "emerging_map": StageOutcome(defer_reason="monthly_budget_deferred")
    })
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")

    result = orchestrator.run_monthly(
        as_of=utc(), source_hashes=("a" * 64,), industry_key="robotic-actuators"
    )

    assert result.status is WorkflowRunState.DEFERRED
    assert [call.stage for call in runner.calls] == ["public_signals", "emerging_map"]
    assert {task.stage for task in result.pending_tasks} == {
        "industry_refresh", "review", "publish"
    }


def valid_quality(*, partial: bool = False) -> QualityGateResult:
    return QualityGateResult(
        reason_codes=(
            (GateReasonCode.MISSING_CLAIM_LINEAGE,) if partial else ()
        ),
        allowed_sections=("portfolio",), allowed_claim_ids=("claim-1",),
        allow_event_report=True, allow_sizing=False,
        publication_verdict=(
            PublicationVerdict.PARTIAL if partial else PublicationVerdict.FINAL
        ),
        review_verdict=ReviewVerdict.PASS,
        effective_rating=(
            RecommendationRating.NO_RATING if partial else RecommendationRating.HOLD
        ),
        portfolio_age_hours=None, inference_mode=InferenceMode.EXTERNAL,
        inference_provider="openai", inference_model="gpt-test",
    )


def draft_quality() -> QualityGateResult:
    return QualityGateResult(
        reason_codes=(GateReasonCode.REVIEWER_NOT_PASSED,),
        allowed_sections=(), allowed_claim_ids=(), allow_event_report=False,
        allow_sizing=False, publication_verdict=PublicationVerdict.DRAFT,
        review_verdict=ReviewVerdict.BLOCK,
        effective_rating=RecommendationRating.NO_RATING,
        portfolio_age_hours=None, inference_mode=None,
        inference_provider=None, inference_model=None,
    )


@pytest.mark.parametrize(
    "outcome",
    [
        StageOutcome(quality_gate=draft_quality()),
        StageOutcome(
            quality_gate=valid_quality(), published_claim_ids=("claim-outside",)
        ),
    ],
)
def test_quality_boundary_blocks_draft_and_unapproved_claims(
    store: ResearchStore, outcome: StageOutcome
) -> None:
    runner = RecordingRunner({"review": outcome})
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")

    result = orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))

    assert result.status is WorkflowRunState.BLOCKED
    assert "publish" not in [call.stage for call in runner.calls]


def test_partial_quality_publishes_only_allowed_claims(store: ResearchStore) -> None:
    runner = RecordingRunner({
        "review": StageOutcome(
            quality_gate=valid_quality(partial=True),
            published_claim_ids=("claim-1",), omissions=("missing_claim_lineage",),
        )
    })
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")

    result = orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))

    assert result.status is WorkflowRunState.PARTIAL
    assert result.omissions == ("missing_claim_lineage",)
    assert [call.stage for call in runner.calls][-1] == "publish"
    assert runner.calls[-1].allowed_claim_ids == ("claim-1",)


def test_named_lease_blocks_second_owner_and_reclaims_at_exact_expiry(tmp_path: Path) -> None:
    now = [utc()]
    store = ResearchStore(tmp_path / "research.db", clock=lambda: now[0])
    store.migrate()
    first = ResearchOrchestrator(store, RecordingRunner(), owner_id="owner-a")
    second = ResearchOrchestrator(store, RecordingRunner(), owner_id="owner-b")
    lease = first.acquire_lease("workflow:test")

    with pytest.raises(WorkflowBusy):
        second.acquire_lease("workflow:test")
    now[0] = lease.expires_at
    replacement = second.acquire_lease("workflow:test")
    assert replacement.lease_token != lease.lease_token
    with pytest.raises(WorkflowConflict, match="stale"):
        first.renew_lease(lease)
    with pytest.raises(WorkflowConflict, match="stale"):
        first.release_lease(lease)
    second.release_lease(replacement)


def test_expired_task_claim_is_resumed_without_repeating_completed_effects(
    tmp_path: Path,
) -> None:
    now = [utc()]
    store = ResearchStore(tmp_path / "research.db", clock=lambda: now[0])
    store.migrate()
    runner = RecordingRunner({"ingestion": [KeyboardInterrupt(), StageOutcome(result_ref="ingest-1")]})
    first = ResearchOrchestrator(store, runner, owner_id="owner-a")
    with pytest.raises(KeyboardInterrupt):
        first.run_daily(as_of=utc(), source_hashes=("a" * 64,))
    assert [call.stage for call in runner.calls] == ["portfolio", "ingestion"]

    now[0] += timedelta(minutes=15)
    second = ResearchOrchestrator(store, runner, owner_id="owner-b")
    result = second.run_daily(as_of=utc(), source_hashes=("a" * 64,))

    assert result.status is WorkflowRunState.COMPLETED
    assert [call.stage for call in runner.calls].count("portfolio") == 1
    assert [call.stage for call in runner.calls].count("ingestion") == 2


def test_same_idempotency_key_concurrently_runs_each_effect_once(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    store.migrate()
    entered = Event()
    release = Event()

    class BlockingRunner(RecordingRunner):
        def run(self, context: StageContext) -> StageOutcome:
            if context.stage == "portfolio":
                entered.set()
                assert release.wait(timeout=5)
            return super().run(context)

    runner = BlockingRunner()
    first = ResearchOrchestrator(store, runner, owner_id="owner-a")
    second = ResearchOrchestrator(store, runner, owner_id="owner-b")
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(
            first.run_weekly, as_of=utc(), source_hashes=("a" * 64,)
        )
        assert entered.wait(timeout=5)
        with pytest.raises(WorkflowBusy):
            second.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
        release.set()
        result = future.result(timeout=5)

    assert result.status is WorkflowRunState.COMPLETED
    assert [call.stage for call in runner.calls].count("portfolio") == 1
    assert [call.stage for call in runner.calls].count("publish") == 1


def test_task_results_store_identifiers_and_hashes_not_raw_output(
    orchestrator: ResearchOrchestrator,
) -> None:
    result = orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    with orchestrator.store.connect() as connection:
        rows = connection.execute(
            "SELECT result_ref, result_hash, outcome_json FROM workflow_tasks "
            "WHERE workflow_id = ?", (result.workflow_id,),
        ).fetchall()

    assert all(row[1] is not None and len(row[1]) == 64 for row in rows)
    assert all("account" not in (row[2] or "").casefold() for row in rows)
    assert all("nav" not in (row[2] or "").casefold() for row in rows)


def test_weekly_selection_exposes_only_explicit_safe_triggers(store: ResearchStore) -> None:
    runner = RecordingRunner()
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")
    orchestrator.run_weekly(
        as_of=utc(), source_hashes=("a" * 64,),
        recommendation_triggers=(
            RecommendationTrigger.VALUATION, RecommendationTrigger.MATERIAL,
        ),
    )

    selected = next(call for call in runner.calls if call.stage == "selected_recommendations")
    assert selected.recommendation_triggers == (
        RecommendationTrigger.MATERIAL, RecommendationTrigger.VALUATION,
    )
    with pytest.raises(ValueError, match="recommendation trigger"):
        orchestrator.run_weekly(
            as_of=utc(25), source_hashes=("b" * 64,), recommendation_triggers=(),
        )


def test_reviewer_cannot_redirect_revision_away_from_originating_role(
    store: ResearchStore,
) -> None:
    runner = RecordingRunner({"review": StageOutcome(
        reviewer_verdict=ReviewVerdict.BLOCK,
        originating_role=AgentRole.RESEARCH_EDITOR,
        defer_reason="rebuild evidence bridge",
    )})
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")

    result = orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))

    assert result.pending_tasks[0].assigned_role is AgentRole.FUNDAMENTAL_ANALYST


def test_populated_v7_database_upgrades_without_data_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "legacy.db"
    migration_root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    legacy_files = []
    for name in (
        "001_initial.sql", "002_nullable_portfolio_freshness.sql",
        "003_claim_dependencies.sql", "004_entity_resolution.sql",
        "005_agent_execution_audit.sql", "006_agent_replay_lease.sql",
        "007_workflow_task_instances.sql",
    ):
        target = tmp_path / name
        shutil.copyfile(migration_root / name, target)
        legacy_files.append(target)
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: tuple(legacy_files))
    legacy.migrate()
    with legacy.transaction() as connection:
        connection.execute(
            "INSERT INTO research_tasks (task_id, task_kind, scope_json, state, "
            "priority, created_at) VALUES ('legacy-v7-task', 'evidence', '{}', "
            "'pending', 0, '2026-08-24T00:00:00.000000Z')"
        )

    upgraded = ResearchStore(database)
    upgraded.migrate()
    with upgraded.connect() as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        legacy_row = connection.execute(
            "SELECT task_kind, state FROM research_tasks WHERE task_id = 'legacy-v7-task'"
        ).fetchone()
        fk_errors = connection.execute("PRAGMA foreign_key_check").fetchall()

    assert versions == [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)]
    assert legacy_row == ("evidence", "pending")
    assert fk_errors == []


def test_migration_008_failure_rolls_back_all_new_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "rollback.db"
    migration_root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    legacy_files = tuple(
        migration_root / name for name in (
            "001_initial.sql", "002_nullable_portfolio_freshness.sql",
            "003_claim_dependencies.sql", "004_entity_resolution.sql",
            "005_agent_execution_audit.sql", "006_agent_replay_lease.sql",
            "007_workflow_task_instances.sql",
        )
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: legacy_files)
    legacy.migrate()
    broken = tmp_path / "008_broken.sql"
    broken.write_text(
        "CREATE TABLE workflow_runs (workflow_id TEXT PRIMARY KEY);\n"
        "INSERT INTO missing_table VALUES (1);\n", encoding="utf-8",
    )
    attempted = ResearchStore(database)
    monkeypatch.setattr(
        attempted, "_migration_files", lambda: (*legacy_files, broken)
    )

    with pytest.raises(sqlite3.OperationalError):
        attempted.migrate()
    assert "workflow_runs" not in attempted.table_names()
    with attempted.connect() as connection:
        assert connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall() == [(1,), (2,), (3,), (4,), (5,), (6,), (7,)]
