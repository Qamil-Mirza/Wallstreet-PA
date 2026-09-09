"""Durability and safety tests for dependency-aware research workflows."""

from __future__ import annotations

import hashlib
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
                    publication_receipt_hash="e" * 64,
                )
            elif context.stage == "review":
                value = StageOutcome(
                    result_ref=f"result-{context.task_id[-12:]}",
                    quality_gate=valid_quality(),
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


def test_same_utc_date_with_different_times_reuses_workflow_identity(
    store: ResearchStore,
) -> None:
    runner = RecordingRunner()
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")

    first = orchestrator.run_weekly(
        as_of=utc(hour=12), portfolio_date=utc(hour=10),
        source_hashes=("a" * 64,)
    )
    second = orchestrator.run_weekly(
        as_of=utc(hour=13), portfolio_date=utc(hour=11),
        source_hashes=("a" * 64,)
    )
    next_date = orchestrator.run_weekly(
        as_of=utc(day=25, hour=12), portfolio_date=utc(day=25, hour=10),
        source_hashes=("a" * 64,)
    )

    assert second.workflow_id == first.workflow_id
    assert second.report_ids == first.report_ids
    assert second.new_agent_runs == 0
    assert next_date.workflow_id != first.workflow_id
    assert [call.stage for call in runner.calls].count("publish") == 2


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
    assert returned.defer_reason == "review_revise"
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


def test_review_without_explicit_quality_authorization_never_calls_publisher(
    store: ResearchStore,
) -> None:
    runner = RecordingRunner({
        "review": StageOutcome(result_ref="review-without-gate"),
    })

    result = ResearchOrchestrator(
        store, runner, owner_id="scheduler-a"
    ).run_weekly(as_of=utc(), source_hashes=("a" * 64,))

    assert result.status is WorkflowRunState.BLOCKED
    assert "publish" not in [call.stage for call in runner.calls]
    publish_task = next(
        task for task in ResearchOrchestrator(
            store, runner, owner_id="reader"
        ).list_tasks(result.workflow_id)
        if task.stage == "publish"
    )
    assert publish_task.state is not WorkflowTaskState.COMPLETED


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


def test_publisher_persists_claims_allowed_by_completed_review(
    store: ResearchStore,
) -> None:
    runner = RecordingRunner({
        "review": StageOutcome(quality_gate=valid_quality()),
        "publish": StageOutcome(
            result_ref="publisher-temporary-ref",
            result_hash="a" * 64,
            report_id="report-safe",
            material_event=True,
            reviewer_verdict=ReviewVerdict.BLOCK,
            originating_role=AgentRole.RESEARCH_DIRECTOR,
            defer_reason="dry_run",
            quality_gate=draft_quality(),
            published_claim_ids=("claim-1",),
            publication_receipt_hash="e" * 64,
        ),
    })

    result = ResearchOrchestrator(
        store, runner, owner_id="scheduler-a"
    ).run_weekly(as_of=utc(), source_hashes=("a" * 64,))

    assert result.status is WorkflowRunState.COMPLETED
    assert result.report_ids == ("report-safe",)
    with store.connect() as connection:
        task_json, effect_json = connection.execute(
            "SELECT task.outcome_json, effect.outcome_json FROM workflow_tasks task "
            "JOIN publication_effects effect ON effect.workflow_id = task.workflow_id "
            "AND effect.task_id = task.task_id WHERE task.workflow_id = ? "
            "AND task.stage = 'publish'",
            (result.workflow_id,),
        ).fetchone()
    assert task_json == effect_json
    stored = StageOutcome.model_validate_json(task_json)
    assert stored.result_ref == stored.report_id == "report-safe"
    assert stored.published_claim_ids == ("claim-1",)
    assert stored.result_hash is None
    assert stored.material_event is None
    assert stored.reviewer_verdict is None
    assert stored.originating_role is None
    assert stored.defer_reason is None
    assert stored.quality_gate is None


def test_crash_after_publication_confirmation_recovers_full_safe_outcome_once(
    store: ResearchStore,
) -> None:
    runner = RecordingRunner({
        "review": StageOutcome(quality_gate=valid_quality()),
        "publish": StageOutcome(
            result_ref="report-safe",
            report_id="report-safe",
            new_agent_runs=2,
            omissions=("missing_claim_lineage",),
            published_claim_ids=("claim-1",),
            publication_receipt_hash="e" * 64,
        ),
    })
    first = ResearchOrchestrator(store, runner, owner_id="scheduler-a")
    original_finalize = first.finalize_task

    def crash_after_confirmation(claim, **kwargs):
        outcome = kwargs.get("outcome")
        if outcome is not None and outcome.report_id is not None:
            raise KeyboardInterrupt
        return original_finalize(claim, **kwargs)

    first.finalize_task = crash_after_confirmation  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        first.run_weekly(as_of=utc(), source_hashes=("a" * 64,))

    recovered = ResearchOrchestrator(
        store, runner, owner_id="scheduler-b"
    ).run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    replay = ResearchOrchestrator(
        store, runner, owner_id="scheduler-c"
    ).run_weekly(as_of=utc(), source_hashes=("a" * 64,))

    assert recovered.status is WorkflowRunState.PARTIAL
    assert recovered.report_ids == ("report-safe",)
    assert recovered.new_agent_runs == 2
    assert recovered.omissions == ("missing_claim_lineage",)
    assert replay.new_agent_runs == 0
    assert [call.stage for call in runner.calls].count("publish") == 1
    with store.connect() as connection:
        task_json, effect_json, stored_runs = connection.execute(
            "SELECT task.outcome_json, effect.outcome_json, run.new_agent_runs "
            "FROM workflow_tasks task JOIN publication_effects effect "
            "ON effect.workflow_id = task.workflow_id AND effect.task_id = task.task_id "
            "JOIN workflow_runs run ON run.workflow_id = task.workflow_id "
            "WHERE task.workflow_id = ? AND task.stage = 'publish'",
            (recovered.workflow_id,),
        ).fetchone()
    assert task_json == effect_json
    assert stored_runs == 2
    assert StageOutcome.model_validate_json(task_json).published_claim_ids == ("claim-1",)


def test_ordinary_failure_after_confirmation_recovers_without_republication(
    store: ResearchStore,
) -> None:
    runner = RecordingRunner({
        "review": StageOutcome(quality_gate=valid_quality()),
        "publish": StageOutcome(
            result_ref="report-safe",
            report_id="report-safe",
            new_agent_runs=2,
            omissions=("missing_claim_lineage",),
            published_claim_ids=("claim-1",),
            publication_receipt_hash="e" * 64,
        ),
    })
    first = ResearchOrchestrator(store, runner, owner_id="scheduler-a")
    original_finalize = first.finalize_task
    failed_once = False

    def fail_after_confirmation(claim, **kwargs):
        nonlocal failed_once
        outcome = kwargs.get("outcome")
        if not failed_once and outcome is not None and outcome.report_id is not None:
            failed_once = True
            raise RuntimeError("transient task completion failure")
        return original_finalize(claim, **kwargs)

    first.finalize_task = fail_after_confirmation  # type: ignore[method-assign]
    interrupted = first.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    recovered = ResearchOrchestrator(
        store, runner, owner_id="scheduler-b"
    ).run_weekly(as_of=utc(hour=13), source_hashes=("a" * 64,))

    assert interrupted.status is WorkflowRunState.DEFERRED
    assert recovered.status is WorkflowRunState.PARTIAL
    assert recovered.report_ids == ("report-safe",)
    assert recovered.new_agent_runs == 2
    assert recovered.omissions == ("missing_claim_lineage",)
    assert [call.stage for call in runner.calls].count("publish") == 1


@pytest.mark.parametrize(
    "published_claim_ids",
    (("claim-outside",), ("claim private",)),
)
def test_publisher_claims_are_rejected_before_effect_confirmation(
    store: ResearchStore, published_claim_ids: tuple[str, ...]
) -> None:
    class UntrustedPublisher(RecordingRunner):
        def run(self, context: StageContext):
            if context.stage == "publish":
                return {
                    "result_ref": "report-safe",
                    "report_id": "report-safe",
                    "published_claim_ids": published_claim_ids,
                    "publication_receipt_hash": "e" * 64,
                }
            return super().run(context)

    runner = UntrustedPublisher({
        "review": StageOutcome(quality_gate=valid_quality()),
    })
    result = ResearchOrchestrator(
        store, runner, owner_id="scheduler-a"
    ).run_weekly(as_of=utc(), source_hashes=("a" * 64,))

    assert result.status is WorkflowRunState.DEFERRED
    with store.connect() as connection:
        assert connection.execute(
            "SELECT state, outcome_json, result_hash FROM publication_effects "
            "WHERE workflow_id = ?", (result.workflow_id,),
        ).fetchone() == ("outcome_unknown", None, None)


def test_direct_effect_confirmation_cannot_bypass_review_claim_allowlist(
    store: ResearchStore,
) -> None:
    runner = RecordingRunner({
        "review": StageOutcome(quality_gate=valid_quality()),
        "publish": [KeyboardInterrupt()],
    })
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")
    with pytest.raises(KeyboardInterrupt):
        orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    publish_context = next(call for call in runner.calls if call.stage == "publish")
    outcome = StageOutcome(
        result_ref="report-safe",
        report_id="report-safe",
        published_claim_ids=("claim-outside",),
        publication_receipt_hash="e" * 64,
    )
    outcome_json = json.dumps(
        outcome.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    with store.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "UPDATE publication_effects SET state = 'confirmed', "
            "claim_token = NULL, claim_expires_at = NULL, report_id = ?, "
            "receipt_hash = ?, outcome_json = ?, result_hash = ?, updated_at = ? "
            "WHERE effect_key = ?",
            (
                "report-safe",
                "e" * 64,
                outcome_json,
                hashlib.sha256(outcome_json.encode()).hexdigest(),
                "2026-08-24T12:01:00.000000Z",
                publish_context.publication_effect_key,
            ),
        )


def test_direct_insert_cannot_create_terminal_publication_effect(
    store: ResearchStore,
) -> None:
    runner = RecordingRunner({
        "review": StageOutcome(quality_gate=valid_quality()),
    })
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")

    def stop_before_effect(*args, **kwargs):
        raise KeyboardInterrupt

    orchestrator._prepare_publication_task = stop_before_effect  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    with store.connect() as connection:
        workflow_id = connection.execute(
            "SELECT workflow_id FROM workflow_runs"
        ).fetchone()[0]
    publish_task = next(
        task for task in orchestrator.list_tasks(workflow_id)
        if task.stage == "publish"
    )
    effect_key = orchestrator._publication_key(workflow_id, publish_task.task_id)
    outcome = StageOutcome(
        result_ref="report-safe",
        report_id="report-safe",
        published_claim_ids=("claim-outside",),
        publication_receipt_hash="e" * 64,
    )
    outcome_json = json.dumps(
        outcome.model_dump(mode="json"), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False,
    )

    with store.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO publication_effects (publication_effect_id, effect_key, "
            "workflow_id, task_id, state, report_id, receipt_hash, outcome_json, "
            "result_hash, created_at, updated_at) VALUES (?, ?, ?, ?, 'confirmed', "
            "?, ?, ?, ?, ?, ?)",
            (
                f"pub_{effect_key[:40]}", effect_key, workflow_id,
                publish_task.task_id, "report-safe", "e" * 64, outcome_json,
                hashlib.sha256(outcome_json.encode()).hexdigest(),
                "2026-08-24T12:00:00.000000Z",
                "2026-08-24T12:00:00.000000Z",
            ),
        )


def test_direct_claimed_effect_insert_requires_stable_identity_and_task_lease(
    store: ResearchStore,
) -> None:
    runner = RecordingRunner({
        "review": StageOutcome(quality_gate=valid_quality()),
    })
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")

    def stop_before_effect(*args, **kwargs):
        raise KeyboardInterrupt

    orchestrator._prepare_publication_task = stop_before_effect  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    with store.connect() as connection:
        workflow_id = connection.execute(
            "SELECT workflow_id FROM workflow_runs"
        ).fetchone()[0]
    publish_task = next(
        task for task in orchestrator.list_tasks(workflow_id)
        if task.stage == "publish"
    )
    claim = orchestrator.claim_task(workflow_id, publish_task.task_id)
    effect_key = orchestrator._publication_key(workflow_id, publish_task.task_id)
    expiry = claim.lease_expires_at.isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")

    with store.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO publication_effects (publication_effect_id, effect_key, "
            "workflow_id, task_id, state, claim_token, claim_expires_at, "
            "created_at, updated_at) VALUES ('pub_wrong_identity', ?, ?, ?, "
            "'claimed', ?, ?, ?, ?)",
            (
                effect_key, workflow_id, publish_task.task_id, claim.lease_token,
                expiry, "2026-08-24T12:00:00.000000Z",
                "2026-08-24T12:00:00.000000Z",
            ),
        )


@pytest.mark.parametrize(
    "link_forged_review",
    (
        pytest.param(False, id="unlinked"),
        pytest.param(True, id="extra-nonadjacent-dependency"),
    ),
)
def test_duplicate_review_cannot_authorize_publication_effect(
    store: ResearchStore, link_forged_review: bool,
) -> None:
    runner = RecordingRunner({
        "review": StageOutcome(quality_gate=valid_quality()),
        "publish": [KeyboardInterrupt()],
    })
    orchestrator = ResearchOrchestrator(store, runner, owner_id="scheduler-a")
    with pytest.raises(KeyboardInterrupt):
        orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    publish_context = next(call for call in runner.calls if call.stage == "publish")
    forged_gate = valid_quality().model_copy(
        update={"allowed_claim_ids": ("claim-outside",)}
    )
    forged_review = StageOutcome(
        result_ref="forged-review", quality_gate=forged_gate
    )
    forged_review_json = json.dumps(
        forged_review.model_dump(mode="json"), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False,
    )
    outcome = StageOutcome(
        result_ref="report-safe",
        report_id="report-safe",
        published_claim_ids=("claim-outside",),
        publication_receipt_hash="e" * 64,
    )
    outcome_json = json.dumps(
        outcome.model_dump(mode="json"), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False,
    )
    now_text = "2026-08-24T12:01:00.000000Z"

    with store.connect() as connection:
        publish_ordinal = connection.execute(
            "SELECT ordinal FROM workflow_tasks WHERE task_id = ?",
            (publish_context.task_id,),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO workflow_tasks (task_id, workflow_id, idempotency_key, "
            "stage, ordinal, state, max_attempts, result_ref, result_hash, "
            "outcome_json, created_at, completed_at) VALUES (?, ?, ?, 'review', "
            "?, 'completed', 2, ?, ?, ?, ?, ?)",
            (
                "wft_forged_review", publish_context.workflow_id, "f" * 64,
                publish_ordinal + 1, "forged-review",
                hashlib.sha256(forged_review_json.encode()).hexdigest(),
                forged_review_json, now_text, now_text,
            ),
        )
        if link_forged_review:
            connection.execute(
                "INSERT INTO workflow_task_dependencies "
                "(workflow_id, task_id, dependency_task_id) VALUES (?, ?, ?)",
                (
                    publish_context.workflow_id,
                    publish_context.task_id,
                    "wft_forged_review",
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE publication_effects SET state = 'confirmed', "
                "claim_token = NULL, claim_expires_at = NULL, report_id = ?, "
                "receipt_hash = ?, outcome_json = ?, result_hash = ?, "
                "updated_at = ? WHERE effect_key = ?",
                (
                    "report-safe", "e" * 64, outcome_json,
                    hashlib.sha256(outcome_json.encode()).hexdigest(), now_text,
                    publish_context.publication_effect_key,
                ),
            )


@pytest.mark.parametrize(
    ("now_offset", "accepted"),
    (
        pytest.param(timedelta(microseconds=-1), True, id="one-microsecond-before"),
        pytest.param(timedelta(0), False, id="exact-expiry"),
    ),
)
def test_claimed_effect_insert_requires_active_task_lease(
    tmp_path: Path, now_offset: timedelta, accepted: bool,
) -> None:
    now = [utc()]
    store = ResearchStore(tmp_path / "research.db", clock=lambda: now[0])
    store.migrate()
    orchestrator = ResearchOrchestrator(
        store,
        RecordingRunner({"review": StageOutcome(quality_gate=valid_quality())}),
        owner_id="scheduler-a",
    )

    def stop_before_effect(*args, **kwargs):
        raise KeyboardInterrupt

    orchestrator._prepare_publication_task = stop_before_effect  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    with store.connect() as connection:
        workflow_id = connection.execute(
            "SELECT workflow_id FROM workflow_runs"
        ).fetchone()[0]
    publish_task = next(
        task for task in orchestrator.list_tasks(workflow_id)
        if task.stage == "publish"
    )
    claim = orchestrator.claim_task(workflow_id, publish_task.task_id)
    effect_key = orchestrator._publication_key(workflow_id, publish_task.task_id)
    expiry = claim.lease_expires_at.isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")
    now[0] = claim.lease_expires_at + now_offset
    created_at = now[0].isoformat(timespec="microseconds").replace("+00:00", "Z")
    values = (
        f"pub_{effect_key[:40]}", effect_key, workflow_id, publish_task.task_id,
        claim.lease_token, expiry, created_at, created_at,
    )
    statement = (
        "INSERT INTO publication_effects (publication_effect_id, effect_key, "
        "workflow_id, task_id, state, claim_token, claim_expires_at, created_at, "
        "updated_at) VALUES (?, ?, ?, ?, 'claimed', ?, ?, ?, ?)"
    )

    with store.connect() as connection:
        if accepted:
            connection.execute(statement, values)
            assert connection.execute(
                "SELECT state FROM publication_effects WHERE effect_key = ?",
                (effect_key,),
            ).fetchone() == ("claimed",)
        else:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(statement, values)


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

    assert versions == [
        (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,),
        (11,), (12,), (13,), (14,), (15,),
    ]
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


def test_publish_expiry_records_unknown_outcome_and_never_repeats_effect(
    tmp_path: Path,
) -> None:
    now = [utc()]
    store = ResearchStore(tmp_path / "research.db", clock=lambda: now[0])
    store.migrate()

    class ExpiringPublisher(RecordingRunner):
        def run(self, context: StageContext) -> StageOutcome:
            outcome = super().run(context)
            if context.stage == "publish":
                assert context.publication_effect_key
                now[0] += timedelta(minutes=15)
            return outcome

    runner = ExpiringPublisher()
    first = ResearchOrchestrator(store, runner, owner_id="owner-a")
    result = first.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    second = ResearchOrchestrator(store, runner, owner_id="owner-b")
    replay = second.run_weekly(as_of=utc(), source_hashes=("a" * 64,))

    assert result.status is WorkflowRunState.DEFERRED
    assert replay.status is WorkflowRunState.DEFERRED
    assert [call.stage for call in runner.calls].count("publish") == 1
    assert "publication_outcome_unknown" in {
        task.defer_reason for task in second.list_tasks(result.workflow_id)
    }


def test_material_event_crash_resume_hydrates_safe_completed_outcomes(
    tmp_path: Path,
) -> None:
    now = [utc()]
    store = ResearchStore(tmp_path / "research.db", clock=lambda: now[0])
    store.migrate()
    runner = RecordingRunner({
        "materiality": StageOutcome(
            material_event=True, new_agent_runs=2,
            omissions=("pre_crash_omission",), result_ref="material-1",
        ),
        "event_analysis": [KeyboardInterrupt(), StageOutcome(
            result_ref="analysis-1", new_agent_runs=3,
        )],
    })
    first = ResearchOrchestrator(store, runner, owner_id="owner-a")
    with pytest.raises(KeyboardInterrupt):
        first.run_daily(as_of=utc(), source_hashes=("a" * 64,))

    now[0] += timedelta(minutes=15)
    resumed = ResearchOrchestrator(store, runner, owner_id="owner-b")
    result = resumed.run_daily(as_of=utc(), source_hashes=("a" * 64,))

    stages = [call.stage for call in runner.calls]
    assert stages.count("portfolio") == 1
    assert stages.count("materiality") == 1
    assert stages.count("event_analysis") == 2
    assert "event_update" in stages and "publish" in stages
    assert result.new_agent_runs == 5
    assert result.status is WorkflowRunState.PARTIAL
    assert result.omissions == ("omission_sanitized",)
    assert len(result.report_ids) == 1


def test_daily_event_update_persistently_depends_on_event_analysis(
    store: ResearchStore,
) -> None:
    runner = RecordingRunner({"materiality": StageOutcome(material_event=True)})
    orchestrator = ResearchOrchestrator(store, runner, owner_id="owner-a")
    result = orchestrator.run_daily(as_of=utc(), source_hashes=("a" * 64,))
    tasks = {task.stage: task for task in orchestrator.list_tasks(result.workflow_id)}

    assert tasks["event_analysis"].task_id in tasks["event_update"].dependency_ids


def test_monthly_revision_returns_to_industry_strategist(store: ResearchStore) -> None:
    runner = RecordingRunner({"review": StageOutcome(
        reviewer_verdict=ReviewVerdict.REVISE,
        defer_reason="refresh industry assumptions",
    )})
    orchestrator = ResearchOrchestrator(store, runner, owner_id="owner-a")

    result = orchestrator.run_monthly(
        as_of=utc(), source_hashes=("a" * 64,), industry_key="robotic-actuators"
    )

    assert result.pending_tasks[0].assigned_role is AgentRole.INDUSTRY_STRATEGIST
    assert result.pending_tasks[0].originating_role is AgentRole.INDUSTRY_STRATEGIST
    assert result.pending_tasks[0].defer_reason == "review_revise"


def test_untrusted_reason_text_is_never_persisted_or_returned(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    store.migrate()
    private = "account U1234567 NAV 4999 api key sk-private"
    runner = RecordingRunner({"review": StageOutcome(
        reviewer_verdict=ReviewVerdict.BLOCK,
        defer_reason=private,
        omissions=(private,),
    )})
    orchestrator = ResearchOrchestrator(store, runner, owner_id="owner-a")

    result = orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    with store.connect() as connection:
        stored = " ".join(
            str(value) for row in connection.execute(
                "SELECT defer_reason, outcome_json FROM workflow_tasks"
            ).fetchall() for value in row if value is not None
        )

    assert private not in stored
    assert private not in result.model_dump_json()
    assert result.pending_tasks[0].defer_reason == "review_block"
    assert result.omissions == ("omission_sanitized",)


def test_unknown_publication_requires_explicit_idempotent_reconciliation(
    tmp_path: Path,
) -> None:
    now = [utc()]
    store = ResearchStore(tmp_path / "research.db", clock=lambda: now[0])
    store.migrate()

    class UnknownPublisher(RecordingRunner):
        def run(self, context: StageContext) -> StageOutcome:
            outcome = super().run(context)
            if context.stage == "publish":
                now[0] += timedelta(minutes=15)
            return outcome

    runner = UnknownPublisher()
    orchestrator = ResearchOrchestrator(store, runner, owner_id="owner-a")
    deferred = orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    effect_key = next(
        call.publication_effect_key for call in runner.calls if call.stage == "publish"
    )
    assert effect_key is not None

    receipt = orchestrator.reconcile_publication(
        effect_key, report_id="report-reconciled", receipt_hash="f" * 64
    )
    assert orchestrator.reconcile_publication(
        effect_key, report_id="report-reconciled", receipt_hash="f" * 64
    ) == receipt
    with pytest.raises(WorkflowConflict, match="conflicts"):
        orchestrator.reconcile_publication(
            effect_key, report_id="report-conflict", receipt_hash="a" * 64
        )
    resumed = ResearchOrchestrator(store, runner, owner_id="owner-b").run_weekly(
        as_of=utc(), source_hashes=("a" * 64,)
    )

    assert deferred.status is WorkflowRunState.DEFERRED
    assert resumed.report_ids == ("report-reconciled",)
    assert [call.stage for call in runner.calls].count("publish") == 1


def test_v9_triggers_reject_terminal_and_identity_rewrites(
    store: ResearchStore,
) -> None:
    orchestrator = ResearchOrchestrator(store, RecordingRunner(), owner_id="owner-a")
    result = orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    tasks = orchestrator.list_tasks(result.workflow_id)
    with store.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE workflow_runs SET state = 'running' WHERE workflow_id = ?",
                (result.workflow_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE workflow_tasks SET assigned_role = 'research_editor' "
                "WHERE task_id = ?", (tasks[0].task_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE workflow_tasks SET result_hash = ? WHERE task_id = ?",
                ("a" * 64, tasks[0].task_id),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE publication_effects SET report_id = 'changed' "
                "WHERE workflow_id = ?", (result.workflow_id,),
            )


def test_v9_publication_schema_has_foreign_keys_and_indexes(store: ResearchStore) -> None:
    with store.connect() as connection:
        version = connection.execute(
            "SELECT name FROM schema_migrations WHERE version = 9"
        ).fetchone()
        foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(publication_effects)"
        ).fetchall()
        indexes = connection.execute(
            "PRAGMA index_list(publication_effects)"
        ).fetchall()
        errors = connection.execute("PRAGMA foreign_key_check").fetchall()

    assert version == ("009_workflow_recovery_publication.sql",)
    assert len({row[0] for row in foreign_keys}) == 2
    assert any(row[1] == "idx_publication_effects_workflow_state" for row in indexes)
    assert errors == []


def test_daily_revision_returns_to_event_scout(store: ResearchStore) -> None:
    runner = RecordingRunner({
        "materiality": StageOutcome(material_event=True),
        "review": StageOutcome(reviewer_verdict=ReviewVerdict.REVISE),
    })
    result = ResearchOrchestrator(store, runner, owner_id="owner-a").run_daily(
        as_of=utc(), source_hashes=("a" * 64,)
    )

    assert result.pending_tasks[0].assigned_role is AgentRole.EVENT_SCOUT


def test_populated_v8_workflow_upgrades_to_v9_without_data_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v8.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(root / name for name in (
        "001_initial.sql", "002_nullable_portfolio_freshness.sql",
        "003_claim_dependencies.sql", "004_entity_resolution.sql",
        "005_agent_execution_audit.sql", "006_agent_replay_lease.sql",
        "007_workflow_task_instances.sql", "008_durable_workflow_orchestration.sql",
    ))
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    with legacy.transaction() as connection:
        connection.execute(
            "INSERT INTO workflow_runs (workflow_id, idempotency_key, workflow_kind, "
            "period_key, as_of, source_hashes_json, definition_hash, state, dry_run, "
            "authorize_analysis, completed_stages_json, report_ids_json, omissions_json, "
            "created_at, updated_at, completed_at) VALUES ('wf_legacy', ?, 'daily', "
            "'2026-08-24', '2026-08-24T00:00:00.000000Z', '[]', ?, 'completed', "
            "0, 0, '[]', '[]', '[]', ?, ?, ?)",
            ("a" * 64, "b" * 64, *("2026-08-24T00:00:00.000000Z",) * 3),
        )
    upgraded = ResearchStore(database)
    upgraded.migrate()
    with upgraded.connect() as connection:
        row = connection.execute(
            "SELECT workflow_kind, state FROM workflow_runs WHERE workflow_id = 'wf_legacy'"
        ).fetchone()
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()

    assert row == ("daily", "completed")
    assert versions[-1] == (15,)


def test_migration_009_failure_rolls_back_publication_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v9-rollback.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(root / name for name in (
        "001_initial.sql", "002_nullable_portfolio_freshness.sql",
        "003_claim_dependencies.sql", "004_entity_resolution.sql",
        "005_agent_execution_audit.sql", "006_agent_replay_lease.sql",
        "007_workflow_task_instances.sql", "008_durable_workflow_orchestration.sql",
    ))
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    broken = tmp_path / "009_broken.sql"
    broken.write_text(
        "CREATE TABLE publication_effects (id TEXT);\n"
        "INSERT INTO missing_table VALUES (1);\n", encoding="utf-8",
    )
    attempted = ResearchStore(database)
    monkeypatch.setattr(attempted, "_migration_files", lambda: (*files, broken))

    with pytest.raises(sqlite3.OperationalError):
        attempted.migrate()
    assert "publication_effects" not in attempted.table_names()
    with attempted.connect() as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (8,)


def test_failed_task_reason_is_immutable_until_repository_retry(
    store: ResearchStore,
) -> None:
    runner = RecordingRunner({
        "changed_theses": [RuntimeError("private"), RuntimeError("private")]
    })
    orchestrator = ResearchOrchestrator(store, runner, owner_id="owner-a")
    result = orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    failed = next(task for task in orchestrator.list_tasks(result.workflow_id)
                  if task.state is WorkflowTaskState.FAILED)

    with store.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "UPDATE workflow_tasks SET defer_reason = 'changed' WHERE task_id = ?",
            (failed.task_id,),
        )


def test_expired_publish_intent_reconcile_bypasses_attempt_limit_without_recall(
    tmp_path: Path,
) -> None:
    now = [utc()]
    store = ResearchStore(tmp_path / "research.db", clock=lambda: now[0])
    store.migrate()

    class CrashingPublisher(RecordingRunner):
        def run(self, context: StageContext) -> StageOutcome:
            try:
                return super().run(context)
            except KeyboardInterrupt:
                if context.stage == "publish":
                    now[0] += timedelta(minutes=15)
                raise

    runner = CrashingPublisher({"publish": [KeyboardInterrupt()]})
    first = ResearchOrchestrator(store, runner, owner_id="owner-a")
    with pytest.raises(KeyboardInterrupt):
        first.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    effect_key = next(
        call.publication_effect_key for call in runner.calls if call.stage == "publish"
    )
    assert effect_key is not None

    deferred = ResearchOrchestrator(store, runner, owner_id="owner-b").run_weekly(
        as_of=utc(), source_hashes=("a" * 64,)
    )
    publish_task = next(
        task for task in first.list_tasks(deferred.workflow_id)
        if task.stage == "publish"
    )
    assert deferred.status is WorkflowRunState.DEFERRED
    assert publish_task.attempt_count == 1
    assert [call.stage for call in runner.calls].count("publish") == 1

    first.reconcile_publication(
        effect_key, report_id="report-reconciled", receipt_hash="f" * 64
    )
    completed = ResearchOrchestrator(store, runner, owner_id="owner-c").run_weekly(
        as_of=utc(), source_hashes=("a" * 64,)
    )

    assert completed.status is WorkflowRunState.COMPLETED
    assert completed.report_ids == ("report-reconciled",)
    assert [call.stage for call in runner.calls].count("publish") == 1
    assert next(
        task for task in first.list_tasks(completed.workflow_id)
        if task.stage == "publish"
    ).attempt_count == 1


def test_populated_v9_unknown_receipt_state_upgrades_to_v10(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v9.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    names = tuple(f"{number:03d}_" for number in range(1, 10))
    files = tuple(
        path for prefix in names for path in root.iterdir()
        if path.name.startswith(prefix)
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    timestamp = "2026-08-24T00:00:00.000000Z"
    with legacy.transaction() as connection:
        connection.execute(
            "INSERT INTO workflow_runs (workflow_id, idempotency_key, workflow_kind, "
            "period_key, as_of, source_hashes_json, definition_hash, state, dry_run, "
            "authorize_analysis, created_at, updated_at, completed_at) VALUES "
            "('wf_v9', ?, 'weekly', '2026-W35', ?, '[]', ?, 'deferred', 0, 0, ?, ?, ?)",
            ("a" * 64, timestamp, "b" * 64, timestamp, timestamp, timestamp),
        )
        connection.execute(
            "INSERT INTO workflow_tasks (task_id, workflow_id, idempotency_key, stage, "
            "ordinal, state, attempt_count, max_attempts, assigned_role, "
            "originating_role, defer_reason, created_at, completed_at) VALUES "
            "('wft_v9', 'wf_v9', ?, 'publish', 0, 'deferred', 1, 2, "
            "'research_editor', 'research_editor', 'publication_outcome_unknown', ?, ?)",
            ("c" * 64, timestamp, timestamp),
        )
        connection.execute(
            "INSERT INTO publication_effects (publication_effect_id, effect_key, "
            "workflow_id, task_id, state, created_at, updated_at) VALUES "
            "('pub_v9', ?, 'wf_v9', 'wft_v9', 'outcome_unknown', ?, ?)",
            ("d" * 64, timestamp, timestamp),
        )

    upgraded = ResearchStore(database)
    upgraded.migrate()
    with upgraded.connect() as connection:
        assert connection.execute(
            "SELECT state, attempt_count FROM workflow_tasks WHERE task_id = 'wft_v9'"
        ).fetchone() == ("deferred", 1)
        assert connection.execute(
            "SELECT state FROM publication_effects WHERE publication_effect_id = 'pub_v9'"
        ).fetchone() == ("outcome_unknown",)
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (15,)


def test_migration_010_failure_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v10-rollback.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 10) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    broken = tmp_path / "010_broken.sql"
    broken.write_text(
        "DROP TRIGGER workflow_tasks_transition_guard;\n"
        "INSERT INTO missing_table VALUES (1);\n", encoding="utf-8",
    )
    attempted = ResearchStore(database)
    monkeypatch.setattr(attempted, "_migration_files", lambda: (*files, broken))

    with pytest.raises(sqlite3.OperationalError):
        attempted.migrate()
    with attempted.connect() as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (9,)
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'workflow_tasks_transition_guard'"
        ).fetchone() == ("workflow_tasks_transition_guard",)


def test_publish_lock_conflict_does_not_consume_external_attempt(
    store: ResearchStore,
) -> None:
    runner = RecordingRunner()
    orchestrator = ResearchOrchestrator(store, runner, owner_id="owner-a")
    original = orchestrator.acquire_lease
    held = None

    def conflict_on_publish(name: str, **kwargs):
        nonlocal held
        if name.startswith("publish:"):
            blocker = ResearchOrchestrator(store, runner, owner_id="owner-b")
            held = original(name, **kwargs)
            return blocker.acquire_lease(name, **kwargs)
        return original(name, **kwargs)

    orchestrator.acquire_lease = conflict_on_publish  # type: ignore[method-assign]
    with pytest.raises(WorkflowBusy):
        orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    publish = next(call for call in runner.calls if call.stage == "review")
    workflow_id = publish.workflow_id
    publish_task = next(
        task for task in orchestrator.list_tasks(workflow_id) if task.stage == "publish"
    )
    with store.connect() as connection:
        effects = connection.execute("SELECT COUNT(*) FROM publication_effects").fetchone()

    assert publish_task.attempt_count == 0
    assert effects == (0,)
    assert held is not None
    ResearchOrchestrator(store, runner, owner_id="owner-a").release_lease(held)


def test_direct_sql_cannot_complete_confirmed_publish_with_forged_receipt(
    store: ResearchStore,
) -> None:
    runner = RecordingRunner()
    orchestrator = ResearchOrchestrator(store, runner, owner_id="owner-a")
    original_finalize = orchestrator.finalize_task

    def crash_after_confirmation(claim, **kwargs):
        outcome = kwargs.get("outcome")
        if outcome is not None and outcome.report_id is not None:
            raise KeyboardInterrupt
        return original_finalize(claim, **kwargs)

    orchestrator.finalize_task = crash_after_confirmation  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    publish_context = next(call for call in runner.calls if call.stage == "publish")
    report_id = f"report-{publish_context.workflow_id[-12:]}"
    forged = StageOutcome(
        result_ref=report_id, report_id=report_id,
        publication_receipt_hash="f" * 64,
    )
    forged_json = json.dumps(
        forged.model_dump(mode="json"), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False,
    )

    with store.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "UPDATE workflow_tasks SET state = 'completed', result_ref = ?, "
            "result_hash = ?, outcome_json = ?, lease_token = NULL, "
            "lease_expires_at = NULL, completed_at = ? WHERE task_id = ?",
            (report_id, hashlib.sha256(forged_json.encode()).hexdigest(), forged_json,
             "2026-08-24T12:01:00.000000Z", publish_context.task_id),
        )

    valid_receipt = StageOutcome(
        result_ref=report_id, report_id=report_id,
        publication_receipt_hash="e" * 64,
    )
    valid_json = json.dumps(
        valid_receipt.model_dump(mode="json"), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False,
    )
    with store.connect() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "UPDATE workflow_tasks SET state = 'completed', result_ref = ?, "
            "result_hash = ?, outcome_json = ?, lease_token = NULL, "
            "lease_expires_at = NULL, completed_at = ? WHERE task_id = ?",
            (report_id, "a" * 64, valid_json,
             "2026-08-24T12:01:00.000000Z", publish_context.task_id),
        )


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        pytest.param("set", ("result_ref", "different-result"), id="result-ref"),
        pytest.param("set", ("raw_secret", "sk-private"), id="private-key"),
        pytest.param("drop", "result_hash", id="missing-key"),
        pytest.param("set", ("result_hash", "a" * 64), id="inner-result-hash"),
        pytest.param("set", ("new_agent_runs", False), id="bool-agent-count"),
        pytest.param("set", ("material_event", False), id="material-event"),
        pytest.param("set", ("omissions", {}), id="wrong-omissions-type"),
        pytest.param("set", ("reviewer_verdict", "pass"), id="reviewer-verdict"),
        pytest.param("set", ("originating_role", "director"), id="origin-role"),
        pytest.param("set", ("defer_reason", "dry_run"), id="defer-reason"),
        pytest.param("set", ("quality_gate", {}), id="quality-gate"),
        pytest.param(
            "set", ("published_claim_ids", ["claim-safe"]), id="published-claims"
        ),
        pytest.param("pretty", None, id="noncanonical-bytes"),
    ),
)
def test_direct_sql_rejects_noncanonical_publication_outcome(
    store: ResearchStore, mutation: str, value: object
) -> None:
    runner = RecordingRunner()
    orchestrator = ResearchOrchestrator(store, runner, owner_id="owner-a")
    original_finalize = orchestrator.finalize_task

    def crash_after_confirmation(claim, **kwargs):
        outcome = kwargs.get("outcome")
        if outcome is not None and outcome.report_id is not None:
            raise KeyboardInterrupt
        return original_finalize(claim, **kwargs)

    orchestrator.finalize_task = crash_after_confirmation  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        orchestrator.run_weekly(as_of=utc(), source_hashes=("a" * 64,))
    publish_context = next(call for call in runner.calls if call.stage == "publish")
    report_id = f"report-{publish_context.workflow_id[-12:]}"
    outcome = StageOutcome(
        result_ref=report_id,
        report_id=report_id,
        publication_receipt_hash="e" * 64,
    ).model_dump(mode="json")
    if mutation == "set":
        field, replacement = value
        outcome[field] = replacement
    elif mutation == "drop":
        outcome.pop(value)
    outcome_json = (
        json.dumps(outcome, ensure_ascii=False)
        if mutation == "pretty"
        else json.dumps(
            outcome, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    )

    with store.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError) as error:
            connection.execute(
                "UPDATE workflow_tasks SET state = 'completed', result_ref = ?, "
                "result_hash = ?, outcome_json = ?, lease_token = NULL, "
                "lease_expires_at = NULL, completed_at = ? WHERE task_id = ?",
                (
                    report_id,
                    hashlib.sha256(outcome_json.encode()).hexdigest(),
                    outcome_json,
                    "2026-08-24T12:01:00.000000Z",
                    publish_context.task_id,
                ),
            )
        retained = connection.execute(
            "SELECT state, outcome_json FROM workflow_tasks WHERE task_id = ?",
            (publish_context.task_id,),
        ).fetchone()

    assert "sk-private" not in str(error.value)
    assert retained == ("running", None)


def _seed_legacy_completed_publication(
    store: ResearchStore,
) -> tuple[str, str]:
    workflow_id = "wf_legacy_publication"
    task_id = "wft_legacy_publication"
    report_id = "report-legacy"
    receipt_hash = "e" * 64
    timestamp = "2026-08-24T12:00:00.000000Z"
    outcome = StageOutcome(
        result_ref=report_id,
        report_id=report_id,
        publication_receipt_hash=receipt_hash,
    )
    outcome_json = json.dumps(
        outcome.model_dump(mode="json"), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False,
    )
    result_hash = hashlib.sha256(outcome_json.encode()).hexdigest()
    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO workflow_runs (workflow_id, idempotency_key, workflow_kind, "
            "period_key, as_of, source_hashes_json, definition_hash, state, dry_run, "
            "authorize_analysis, created_at, updated_at, completed_at) VALUES "
            "(?, ?, 'weekly', '2026-W35', ?, '[]', ?, 'completed', 0, 0, ?, ?, ?)",
            (workflow_id, "a" * 64, timestamp, "b" * 64,
             timestamp, timestamp, timestamp),
        )
        connection.execute(
            "INSERT INTO workflow_tasks (task_id, workflow_id, idempotency_key, "
            "stage, ordinal, state, attempt_count, max_attempts, result_ref, "
            "result_hash, outcome_json, created_at, started_at, completed_at) "
            "VALUES (?, ?, ?, 'publish', 0, 'completed', 1, 2, ?, ?, ?, ?, ?, ?)",
            (task_id, workflow_id, "c" * 64, report_id, result_hash, outcome_json,
             timestamp, timestamp, timestamp),
        )
        connection.execute(
            "INSERT INTO publication_effects (publication_effect_id, effect_key, "
            "workflow_id, task_id, state, report_id, receipt_hash, created_at, "
            "updated_at) VALUES ('pub_legacy_publication', ?, ?, ?, 'confirmed', "
            "?, ?, ?, ?)",
            ("d" * 64, workflow_id, task_id, report_id, receipt_hash,
             timestamp, timestamp),
        )
    return workflow_id, report_id


def test_populated_v10_publication_receipt_upgrades_without_data_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v10.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 11) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    workflow_id, report_id = _seed_legacy_completed_publication(legacy)

    upgraded = ResearchStore(database)
    upgraded.migrate()
    with upgraded.connect() as connection:
        assert connection.execute(
            "SELECT state, report_id, receipt_hash FROM publication_effects "
            "WHERE workflow_id = ?", (workflow_id,),
        ).fetchone() == ("confirmed", report_id, "e" * 64)
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (15,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_migration_011_rolls_back_trigger_replacement_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v11-rollback.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 11) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    broken = tmp_path / "011_broken.sql"
    broken.write_text(
        (root / "011_bind_publication_receipts.sql").read_text(encoding="utf-8")
        + "\nINSERT INTO missing_table VALUES (1);\n",
        encoding="utf-8",
    )
    attempted = ResearchStore(database)
    monkeypatch.setattr(attempted, "_migration_files", lambda: (*files, broken))

    with pytest.raises(sqlite3.OperationalError):
        attempted.migrate()
    with attempted.connect() as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (10,)
        assert connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'workflow_tasks_transition_guard'"
        ).fetchone()[0].find("sha256_hex") == -1


def test_migration_011_fails_closed_without_sha256_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v11-no-hash.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 11) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    attempted = ResearchStore(database)
    monkeypatch.setattr(
        attempted, "_register_sql_functions",
        lambda connection: None,
    )

    with pytest.raises(sqlite3.OperationalError, match="sha256_hex"):
        attempted.migrate()
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (10,)


def test_populated_v11_publication_outcome_upgrades_without_data_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v11.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 12) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    workflow_id, report_id = _seed_legacy_completed_publication(legacy)

    upgraded = ResearchStore(database)
    upgraded.migrate()
    with upgraded.connect() as connection:
        assert connection.execute(
            "SELECT state, report_id, receipt_hash FROM publication_effects "
            "WHERE workflow_id = ?", (workflow_id,),
        ).fetchone() == ("confirmed", report_id, "e" * 64)
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (15,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name IN "
            "('workflow_tasks_transition_guard', "
            "'workflow_tasks_terminal_output_guard', "
            "'workflow_tasks_terminal_reason_immutable') ORDER BY name"
        ).fetchall()
    assert len(trigger_sql) == 3
    assert all(
        "is_canonical_publication_effect_outcome" in row[0]
        for row in trigger_sql
    )


def test_migration_012_rolls_back_trigger_replacement_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v12-rollback.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 12) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    broken = tmp_path / "012_broken.sql"
    broken.write_text(
        (root / "012_canonical_publication_outcomes.sql").read_text(encoding="utf-8")
        + "\nINSERT INTO missing_table VALUES (1);\n",
        encoding="utf-8",
    )
    attempted = ResearchStore(database)
    monkeypatch.setattr(attempted, "_migration_files", lambda: (*files, broken))

    with pytest.raises(sqlite3.OperationalError):
        attempted.migrate()
    with attempted.connect() as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (11,)
        assert "is_canonical_publication_outcome" not in connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'workflow_tasks_transition_guard'"
        ).fetchone()[0]


def test_migration_012_fails_closed_without_canonical_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v12-no-validator.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 12) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    attempted = ResearchStore(database)

    def register_hash_only(connection):
        connection.create_function(
            "sha256_hex", 1,
            lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
            deterministic=True,
        )

    monkeypatch.setattr(attempted, "_register_sql_functions", register_hash_only)
    with pytest.raises(sqlite3.OperationalError, match="canonical_publication"):
        attempted.migrate()
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (11,)


def test_populated_v12_effects_backfill_canonical_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v12.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 13) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    timestamp = "2026-08-24T12:00:00.000000Z"
    with legacy.transaction() as connection:
        for ordinal, state in enumerate(("confirmed", "reconciled")):
            workflow_id = f"wf_legacy_{state}"
            task_id = f"wft_legacy_{state}"
            report_id = f"report-{state}"
            receipt_hash = ("e" if state == "confirmed" else "f") * 64
            outcome = StageOutcome(
                result_ref=report_id,
                report_id=report_id,
                publication_receipt_hash=receipt_hash,
            )
            outcome_json = json.dumps(
                outcome.model_dump(mode="json"), sort_keys=True,
                separators=(",", ":"), ensure_ascii=False,
            )
            result_hash = hashlib.sha256(outcome_json.encode()).hexdigest()
            connection.execute(
                "INSERT INTO workflow_runs (workflow_id, idempotency_key, "
                "workflow_kind, period_key, as_of, source_hashes_json, "
                "definition_hash, state, dry_run, authorize_analysis, created_at, "
                "updated_at, completed_at) VALUES (?, ?, 'weekly', '2026-W35', ?, "
                "'[]', ?, 'completed', 0, 0, ?, ?, ?)",
                (workflow_id, str(ordinal) * 64, timestamp, "a" * 64,
                 timestamp, timestamp, timestamp),
            )
            connection.execute(
                "INSERT INTO workflow_tasks (task_id, workflow_id, idempotency_key, "
                "stage, ordinal, state, attempt_count, max_attempts, result_ref, "
                "result_hash, outcome_json, created_at, started_at, completed_at) "
                "VALUES (?, ?, ?, 'publish', 0, 'completed', 1, 2, ?, ?, ?, ?, ?, ?)",
                (task_id, workflow_id, str(ordinal + 2) * 64, report_id,
                 result_hash, outcome_json, timestamp, timestamp, timestamp),
            )
            connection.execute(
                "INSERT INTO publication_effects (publication_effect_id, effect_key, "
                "workflow_id, task_id, state, report_id, receipt_hash, created_at, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"pub_legacy_{state}", str(ordinal + 4) * 64, workflow_id,
                 task_id, state, report_id, receipt_hash, timestamp, timestamp),
            )

    upgraded = ResearchStore(database)
    upgraded.migrate()
    with upgraded.connect() as connection:
        rows = connection.execute(
            "SELECT effect.state, effect.outcome_json, effect.result_hash, "
            "task.outcome_json, task.result_hash FROM publication_effects effect "
            "JOIN workflow_tasks task ON task.workflow_id = effect.workflow_id "
            "AND task.task_id = effect.task_id WHERE effect.workflow_id IN (?, ?) "
            "ORDER BY effect.state",
            ("wf_legacy_confirmed", "wf_legacy_reconciled"),
        ).fetchall()
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (15,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert [row[0] for row in rows] == ["confirmed", "reconciled"]
    assert all(row[1] == row[3] and row[2] == row[4] for row in rows)
    assert all(hashlib.sha256(row[1].encode()).hexdigest() == row[2] for row in rows)


def test_migration_013_rolls_back_effect_columns_and_triggers_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v13-rollback.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 13) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    broken = tmp_path / "013_broken.sql"
    broken.write_text(
        (root / "013_publication_effect_outcomes.sql").read_text(encoding="utf-8")
        + "\nINSERT INTO missing_table VALUES (1);\n",
        encoding="utf-8",
    )
    attempted = ResearchStore(database)
    monkeypatch.setattr(attempted, "_migration_files", lambda: (*files, broken))

    with pytest.raises(sqlite3.OperationalError):
        attempted.migrate()
    with attempted.connect() as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (12,)
        columns = {
            row[1] for row in connection.execute(
                "PRAGMA table_info(publication_effects)"
            ).fetchall()
        }
        assert "is_canonical_publication_effect_outcome" not in connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'workflow_tasks_transition_guard'"
        ).fetchone()[0]
    assert {"outcome_json", "result_hash"}.isdisjoint(columns)


def test_migration_013_fails_closed_without_effect_outcome_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v13-no-validator.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 13) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    attempted = ResearchStore(database)

    def register_hash_only(connection):
        connection.create_function(
            "sha256_hex", 1,
            lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
            deterministic=True,
        )

    monkeypatch.setattr(attempted, "_register_sql_functions", register_hash_only)
    with pytest.raises(sqlite3.OperationalError, match="publication_effect_outcome"):
        attempted.migrate()
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (12,)


def test_populated_v13_upgrades_effect_insert_guards_without_data_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v13.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 14) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    result = ResearchOrchestrator(
        legacy, RecordingRunner(), owner_id="owner-a"
    ).run_weekly(as_of=utc(), source_hashes=("a" * 64,))

    upgraded = ResearchStore(database)
    upgraded.migrate()
    with upgraded.connect() as connection:
        assert connection.execute(
            "SELECT state, report_id, outcome_json, result_hash "
            "FROM publication_effects WHERE workflow_id = ?",
            (result.workflow_id,),
        ).fetchone()[0:2] == ("confirmed", result.report_ids[0])
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (15,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        insert_guard = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'publication_effects_insert_guard'"
        ).fetchone()[0]
        transition_guard = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'publication_effects_transition_guard'"
        ).fetchone()[0]
    assert "publication_effect_key" in insert_guard
    assert "quality_gate.review_verdict" in transition_guard


def test_migration_014_rolls_back_insert_and_transition_guards_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v14-rollback.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 14) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    with legacy.connect() as connection:
        previous_transition_guard = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'publication_effects_transition_guard'"
        ).fetchone()[0]
    broken = tmp_path / "014_broken.sql"
    broken.write_text(
        (root / "014_publication_effect_insert_guards.sql").read_text(
            encoding="utf-8"
        ) + "\nINSERT INTO missing_table VALUES (1);\n",
        encoding="utf-8",
    )
    attempted = ResearchStore(database)
    monkeypatch.setattr(attempted, "_migration_files", lambda: (*files, broken))

    with pytest.raises(sqlite3.OperationalError):
        attempted.migrate()
    with attempted.connect() as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (13,)
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'publication_effects_insert_guard'"
        ).fetchone() is None
        assert connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'publication_effects_transition_guard'"
        ).fetchone()[0] == previous_transition_guard


def test_migration_014_fails_closed_without_effect_key_scalar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v14-no-key.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 14) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    attempted = ResearchStore(database)
    register_all = attempted._register_sql_functions

    def register_without_effect_key(connection):
        register_all(connection)
        connection.create_function("publication_effect_key", 2, None)

    monkeypatch.setattr(
        attempted, "_register_sql_functions", register_without_effect_key
    )
    with pytest.raises(sqlite3.OperationalError):
        attempted.migrate()
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (13,)


def test_populated_v14_upgrades_bound_publication_authority_without_data_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v14.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 15) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    result = ResearchOrchestrator(
        legacy, RecordingRunner(), owner_id="owner-a"
    ).run_weekly(as_of=utc(), source_hashes=("a" * 64,))

    upgraded = ResearchStore(database)
    upgraded.migrate()
    with upgraded.connect() as connection:
        effect = connection.execute(
            "SELECT state, report_id, outcome_json, result_hash "
            "FROM publication_effects WHERE workflow_id = ?",
            (result.workflow_id,),
        ).fetchone()
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (15,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        insert_guard = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'publication_effects_insert_guard'"
        ).fetchone()[0]
        transition_guard = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'publication_effects_transition_guard'"
        ).fetchone()[0]
    assert effect[0:2] == ("confirmed", result.report_ids[0])
    assert effect[2] is not None and effect[3] is not None
    assert "research_utc_now" in insert_guard
    assert "workflow_task_dependencies" in transition_guard
    assert "review.ordinal + 1 = publish.ordinal" in transition_guard


def test_migration_015_rolls_back_all_guard_replacements_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v15-rollback.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 15) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    guard_names = (
        "publication_effects_insert_guard",
        "publication_effects_transition_guard",
        "workflow_tasks_transition_guard",
        "workflow_tasks_terminal_output_guard",
        "workflow_tasks_terminal_reason_immutable",
    )
    placeholders = ",".join("?" for _ in guard_names)
    with legacy.connect() as connection:
        previous_guards = dict(connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' "
            f"AND name IN ({placeholders})",
            guard_names,
        ).fetchall())
    broken = tmp_path / "015_broken.sql"
    broken.write_text(
        (root / "015_bind_publication_authority.sql").read_text(
            encoding="utf-8"
        ) + "\nINSERT INTO missing_table VALUES (1);\n",
        encoding="utf-8",
    )
    attempted = ResearchStore(database)
    monkeypatch.setattr(attempted, "_migration_files", lambda: (*files, broken))

    with pytest.raises(sqlite3.OperationalError):
        attempted.migrate()
    with attempted.connect() as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (14,)
        restored_guards = dict(connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' "
            f"AND name IN ({placeholders})",
            guard_names,
        ).fetchall())
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'research_utc_now_function_probe'"
        ).fetchone() is None
    assert restored_guards == previous_guards


def test_migration_015_fails_closed_without_authoritative_clock_scalar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "v15-no-clock.db"
    root = Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    files = tuple(
        path for number in range(1, 15) for path in root.iterdir()
        if path.name.startswith(f"{number:03d}_")
    )
    legacy = ResearchStore(database)
    monkeypatch.setattr(legacy, "_migration_files", lambda: files)
    legacy.migrate()
    attempted = ResearchStore(database)
    register_all = attempted._register_sql_functions

    def register_without_clock(connection):
        register_all(connection)
        connection.create_function("research_utc_now", 0, None)

    monkeypatch.setattr(
        attempted, "_register_sql_functions", register_without_clock
    )
    with pytest.raises(sqlite3.OperationalError):
        attempted.migrate()
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT MAX(version) FROM schema_migrations"
        ).fetchone() == (14,)
        assert connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'publication_effects_insert_guard'"
        ).fetchone()[0].find("research_utc_now") == -1
