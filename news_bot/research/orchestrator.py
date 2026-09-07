"""Dependency-aware, durable orchestration for qualitative research workflows.

The orchestrator deliberately knows nothing about HTTP clients or model providers.
It persists an explicit DAG, claims each effect before calling an injected stage
runner, and stores only identifiers, hashes, and release-control metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from .models import AgentRole, ReviewVerdict
from .quality import GateReasonCode, PublicationVerdict, QualityGateResult
from .store import ResearchStore, _parse_utc, _utc_text


_HASH = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")
_DEFAULT_LEASE = timedelta(minutes=15)
_MAX_BACKFILL_DOCUMENTS = 1000
_INTERNAL_REASON_CODES = {
    "budget_deferred",
    "deferred_by_stage",
    "dry_run",
    "no_material_event",
    "omission_sanitized",
    "publication_dry_run",
    "publication_outcome_unknown",
    "publication_quality_blocked",
    "review_block",
    "review_not_passed",
    "review_requires_revision",
    "review_revise",
    "stage_execution_failed",
    "task_lease_expired",
}
_ALLOWED_REASON_CODES = _INTERNAL_REASON_CODES | {
    reason.value for reason in GateReasonCode
}
_REASON_ALIASES = {"monthly_budget_deferred": "budget_deferred"}


class WorkflowError(RuntimeError):
    """Base class for safe orchestration failures."""


class WorkflowBusy(WorkflowError):
    """Raised when another owner has an unexpired workflow or publish lease."""


class WorkflowConflict(WorkflowError):
    """Raised when durable workflow state fails a compare-and-swap boundary."""


class WorkflowKind(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    BACKFILL = "backfill"


class WorkflowRunState(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"
    DEFERRED = "deferred"


class WorkflowTaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEFERRED = "deferred"


class RecommendationTrigger(str, Enum):
    """Allowed reasons for the bounded weekly recommendation selection."""

    SCHEDULED = "scheduled"
    MATERIAL = "material"
    VALUATION = "valuation"


class PublicationEffectState(str, Enum):
    CLAIMED = "claimed"
    CONFIRMED = "confirmed"
    OUTCOME_UNKNOWN = "outcome_unknown"
    RECONCILED = "reconciled"


def _strict_utc(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise ValueError("datetime must be UTC")
    return value.astimezone(timezone.utc)


def _identifier(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("identifier is invalid")
    normalized = unicodedata.normalize("NFC", value)
    if _SAFE_ID.fullmatch(normalized) is None:
        raise ValueError("identifier is invalid")
    return normalized


def _hash(value: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError("hash must be lowercase SHA-256")
    return value


def _optional_identifier(value: str | None) -> str | None:
    return None if value is None else _identifier(value)


def _safe_reason(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("defer reason is invalid")
    normalized = unicodedata.normalize("NFC", value).strip().casefold()
    normalized = _REASON_ALIASES.get(normalized, normalized)
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", normalized) is None:
        return "deferred_by_stage"
    return normalized if normalized in _ALLOWED_REASON_CODES else "deferred_by_stage"


def _omission_codes(values: Sequence[str]) -> tuple[str, ...]:
    sanitized = []
    for value in values:
        reason = _safe_reason(value)
        sanitized.append(
            reason if reason != "deferred_by_stage" else "omission_sanitized"
        )
    return tuple(sorted(set(sanitized)))


def _canonical_ids(values: Sequence[str], *, hashes: bool = False) -> tuple[str, ...]:
    validator = _hash if hashes else _identifier
    normalized = tuple(validator(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("identifiers must be unique")
    return tuple(sorted(normalized))


def _ordered_ids(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(_identifier(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("identifiers must be unique")
    return normalized


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


class FrozenWorkflowContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, revalidate_instances="always", validate_default=True
    )


class DurableTaskView(FrozenWorkflowContract):
    """Safe public view of one durable task and its dependency lineage."""

    task_id: str
    idempotency_key: str
    stage: str
    dependency_ids: tuple[str, ...]
    state: WorkflowTaskState
    attempt_count: StrictInt = Field(ge=0)
    max_attempts: StrictInt = Field(gt=0)
    assigned_role: AgentRole | None = None
    originating_role: AgentRole | None = None
    defer_reason: str | None = None
    result_ref: str | None = None
    result_hash: str | None = None

    _task_id = field_validator("task_id")(_identifier)
    _key = field_validator("idempotency_key")(_hash)
    _stage = field_validator("stage")(_identifier)
    _dependencies = field_validator("dependency_ids")(_canonical_ids)
    _reason = field_validator("defer_reason")(_safe_reason)
    _result_ref = field_validator("result_ref")(_optional_identifier)

    @field_validator("result_hash")
    @classmethod
    def _result_digest(cls, value: str | None) -> str | None:
        return None if value is None else _hash(value)

    @model_validator(mode="after")
    def _coherent_attempts(self) -> "DurableTaskView":
        if self.attempt_count > self.max_attempts:
            raise ValueError("attempt count exceeds maximum")
        return self


class WorkflowRunResult(FrozenWorkflowContract):
    """Replayable summary of one workflow without portfolio-private values."""

    workflow_id: str
    workflow_kind: WorkflowKind
    status: WorkflowRunState
    completed_stages: tuple[str, ...]
    report_ids: tuple[str, ...]
    new_agent_runs: StrictInt = Field(ge=0)
    pending_tasks: tuple[DurableTaskView, ...]
    omissions: tuple[str, ...]
    dry_run: StrictBool

    _workflow_id = field_validator("workflow_id")(_identifier)
    _stages = field_validator("completed_stages")(_ordered_ids)
    _reports = field_validator("report_ids")(_canonical_ids)
    _omissions = field_validator("omissions")(_omission_codes)


class StageContext(FrozenWorkflowContract):
    """Identifier-only context passed to a stage service outside transactions."""

    workflow_id: str
    workflow_kind: WorkflowKind
    task_id: str
    stage: str
    as_of: datetime
    period_key: str
    source_hashes: tuple[str, ...]
    dependency_result_refs: tuple[str, ...] = ()
    allowed_claim_ids: tuple[str, ...] = ()
    industry_key: str | None = None
    max_documents: StrictInt | None = Field(default=None, gt=0)
    authorize_analysis: StrictBool = False
    dry_run: StrictBool = False
    recommendation_triggers: tuple[RecommendationTrigger, ...] = ()
    publication_effect_key: str | None = None
    task_lease_token: str

    _workflow_id = field_validator("workflow_id")(_identifier)
    _task_id = field_validator("task_id")(_identifier)
    _stage = field_validator("stage")(_identifier)
    _period = field_validator("period_key")(_identifier)
    _sources = field_validator("source_hashes")(
        lambda values: _canonical_ids(values, hashes=True)
    )
    _dependency_refs = field_validator("dependency_result_refs")(_canonical_ids)
    _allowed_claims = field_validator("allowed_claim_ids")(_canonical_ids)
    _industry = field_validator("industry_key")(_optional_identifier)
    _as_of = field_validator("as_of")(_strict_utc)
    _task_token = field_validator("task_lease_token")(_hash)

    @field_validator("publication_effect_key")
    @classmethod
    def _effect_key(cls, value: str | None) -> str | None:
        return None if value is None else _hash(value)

    @field_validator("recommendation_triggers")
    @classmethod
    def _triggers(
        cls, values: tuple[RecommendationTrigger, ...]
    ) -> tuple[RecommendationTrigger, ...]:
        if len(values) != len(set(values)):
            raise ValueError("recommendation triggers must be unique")
        return tuple(sorted(values, key=lambda value: value.value))

    @model_validator(mode="after")
    def _publication_identity(self) -> "StageContext":
        if (self.stage == "publish") != (self.publication_effect_key is not None):
            raise ValueError("publish stage requires its durable effect key")
        return self


class StageOutcome(FrozenWorkflowContract):
    """Small safe stage result; narrative output remains in its owning service."""

    result_ref: str | None = None
    result_hash: str | None = None
    report_id: str | None = None
    new_agent_runs: StrictInt = Field(default=0, ge=0)
    material_event: StrictBool | None = None
    omissions: tuple[str, ...] = ()
    reviewer_verdict: ReviewVerdict | None = None
    originating_role: AgentRole | None = None
    defer_reason: str | None = None
    quality_gate: QualityGateResult | None = None
    published_claim_ids: tuple[str, ...] = ()
    publication_receipt_hash: str | None = None

    _result_ref = field_validator("result_ref")(_optional_identifier)
    _report_id = field_validator("report_id")(_optional_identifier)
    _omissions = field_validator("omissions")(_omission_codes)
    _claims = field_validator("published_claim_ids")(_canonical_ids)
    _reason = field_validator("defer_reason")(_safe_reason)

    @field_validator("result_hash")
    @classmethod
    def _result_digest(cls, value: str | None) -> str | None:
        return None if value is None else _hash(value)

    @field_validator("publication_receipt_hash")
    @classmethod
    def _receipt_digest(cls, value: str | None) -> str | None:
        return None if value is None else _hash(value)


class WorkflowLease(FrozenWorkflowContract):
    lease_name: str
    workflow_id: str | None = None
    owner_id: str
    lease_token: str
    expires_at: datetime

    _name = field_validator("lease_name")(_identifier)
    _workflow = field_validator("workflow_id")(_optional_identifier)
    _owner = field_validator("owner_id")(_identifier)
    _token = field_validator("lease_token")(_hash)
    _expiry = field_validator("expires_at")(_strict_utc)


class TaskClaim(FrozenWorkflowContract):
    task_id: str
    lease_token: str
    attempt_count: StrictInt = Field(gt=0)
    lease_expires_at: datetime

    _task = field_validator("task_id")(_identifier)
    _token = field_validator("lease_token")(_hash)
    _expiry = field_validator("lease_expires_at")(_strict_utc)


class PublicationEffectView(FrozenWorkflowContract):
    effect_key: str
    workflow_id: str
    task_id: str
    state: PublicationEffectState
    report_id: str | None = None
    receipt_hash: str | None = None

    _key = field_validator("effect_key")(_hash)
    _workflow = field_validator("workflow_id")(_identifier)
    _task = field_validator("task_id")(_identifier)
    _report = field_validator("report_id")(_optional_identifier)

    @field_validator("receipt_hash")
    @classmethod
    def _receipt(cls, value: str | None) -> str | None:
        return None if value is None else _hash(value)

    @model_validator(mode="after")
    def _receipt_state(self) -> "PublicationEffectView":
        known = self.state in {
            PublicationEffectState.CONFIRMED, PublicationEffectState.RECONCILED
        }
        if known != (self.report_id is not None and self.receipt_hash is not None):
            raise ValueError("publication receipt state is inconsistent")
        return self


class StageRunner(Protocol):
    def run(self, context: StageContext) -> StageOutcome:
        """Perform one claimed stage and return identifier-only metadata."""


class _TaskDefinition(FrozenWorkflowContract):
    stage: str
    dependencies: tuple[str, ...] = ()
    assigned_role: AgentRole | None = None
    originating_role: AgentRole | None = None
    max_attempts: StrictInt = Field(default=2, gt=0)
    optional: StrictBool = False

    _stage = field_validator("stage")(_identifier)
    _dependencies = field_validator("dependencies")(_canonical_ids)


_ROLES: Mapping[str, AgentRole] = {
    "portfolio": AgentRole.PORTFOLIO_MAPPER,
    "ingestion": AgentRole.EVENT_SCOUT,
    "resolve": AgentRole.EVIDENCE_ANALYST,
    "materiality": AgentRole.EVIDENCE_ANALYST,
    "event_analysis": AgentRole.FUNDAMENTAL_ANALYST,
    "event_update": AgentRole.EVENT_SCOUT,
    "changed_theses": AgentRole.RESEARCH_DIRECTOR,
    "selected_recommendations": AgentRole.FUNDAMENTAL_ANALYST,
    "public_signals": AgentRole.EMERGING_COMPANY_SCOUT,
    "emerging_map": AgentRole.INDUSTRY_STRATEGIST,
    "industry_refresh": AgentRole.INDUSTRY_STRATEGIST,
    "historical_documents": AgentRole.EVIDENCE_ANALYST,
    "evidence": AgentRole.EVIDENCE_ANALYST,
    "analysis": AgentRole.FUNDAMENTAL_ANALYST,
    "review": AgentRole.SKEPTICAL_REVIEWER,
    "publish": AgentRole.RESEARCH_EDITOR,
}


def _linear(
    *stages: str,
    review_origin: AgentRole = AgentRole.FUNDAMENTAL_ANALYST,
) -> tuple[_TaskDefinition, ...]:
    definitions: list[_TaskDefinition] = []
    previous: str | None = None
    for stage in stages:
        definitions.append(
            _TaskDefinition(
                stage=stage,
                dependencies=() if previous is None else (previous,),
                assigned_role=_ROLES[stage],
                originating_role=(
                    review_origin if stage == "review" else _ROLES[stage]
                ),
            )
        )
        previous = stage
    return tuple(definitions)


class ResearchOrchestrator:
    """Coordinate explicit persisted DAGs through exclusive renewable leases."""

    def __init__(
        self,
        store: ResearchStore,
        runner: StageRunner,
        *,
        owner_id: str,
        lease_duration: timedelta = _DEFAULT_LEASE,
    ) -> None:
        if not isinstance(store, ResearchStore):
            raise TypeError("store must be ResearchStore")
        if not hasattr(runner, "run"):
            raise TypeError("runner must implement run")
        self.store = store
        self.runner = runner
        self.owner_id = _identifier(owner_id)
        if not isinstance(lease_duration, timedelta) or lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        self.lease_duration = lease_duration

    def _now(self) -> datetime:
        return _strict_utc(self.store.clock())

    @staticmethod
    def _period(kind: WorkflowKind, as_of: datetime) -> str:
        if kind is WorkflowKind.DAILY or kind is WorkflowKind.BACKFILL:
            return as_of.date().isoformat()
        if kind is WorkflowKind.WEEKLY:
            year, week, _ = as_of.isocalendar()
            return f"{year}-W{week:02d}"
        return as_of.strftime("%Y-%m")

    @staticmethod
    def _definitions(
        kind: WorkflowKind, *, authorize_analysis: bool
    ) -> tuple[_TaskDefinition, ...]:
        if kind is WorkflowKind.DAILY:
            return (
                _TaskDefinition(stage="portfolio", assigned_role=_ROLES["portfolio"]),
                _TaskDefinition(stage="ingestion", dependencies=("portfolio",), assigned_role=_ROLES["ingestion"]),
                _TaskDefinition(stage="resolve", dependencies=("ingestion",), assigned_role=_ROLES["resolve"]),
                _TaskDefinition(stage="materiality", dependencies=("resolve",), assigned_role=_ROLES["materiality"]),
                _TaskDefinition(stage="event_analysis", dependencies=("materiality",), assigned_role=_ROLES["event_analysis"], optional=True),
                _TaskDefinition(stage="event_update", dependencies=("materiality", "event_analysis"), assigned_role=_ROLES["event_update"]),
                _TaskDefinition(stage="review", dependencies=("event_update",), assigned_role=_ROLES["review"], originating_role=AgentRole.EVENT_SCOUT, optional=True),
                _TaskDefinition(stage="publish", dependencies=("review",), assigned_role=_ROLES["publish"], optional=True),
            )
        if kind is WorkflowKind.WEEKLY:
            return _linear("portfolio", "changed_theses", "selected_recommendations", "review", "publish")
        if kind is WorkflowKind.MONTHLY:
            return _linear(
                "public_signals", "emerging_map", "industry_refresh", "review", "publish",
                review_origin=AgentRole.INDUSTRY_STRATEGIST,
            )
        stages = ["historical_documents", "evidence"]
        if authorize_analysis:
            stages.extend(("analysis", "review", "publish"))
        return _linear(*stages)

    def acquire_lease(
        self,
        lease_name: str,
        *,
        workflow_id: str | None = None,
        duration: timedelta | None = None,
    ) -> WorkflowLease:
        """Atomically acquire or reclaim a named lease at exact UTC expiry."""
        name = _identifier(lease_name)
        workflow = _optional_identifier(workflow_id)
        ttl = self.lease_duration if duration is None else duration
        if not isinstance(ttl, timedelta) or ttl <= timedelta(0):
            raise ValueError("lease duration must be positive")
        now = self._now()
        expires = now + ttl
        token = secrets.token_hex(32)
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT owner_id, lease_token, expires_at FROM workflow_leases "
                "WHERE lease_name = ?", (name,),
            ).fetchone()
            if row is not None and _parse_utc(row[2]) > now:
                raise WorkflowBusy("workflow lease is held by another owner")
            if row is None:
                connection.execute(
                    "INSERT INTO workflow_leases (lease_name, workflow_id, owner_id, "
                    "lease_token, expires_at, acquired_at, renewed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (name, workflow, self.owner_id, token, _utc_text(expires), _utc_text(now), _utc_text(now)),
                )
            else:
                connection.execute(
                    "UPDATE workflow_leases SET workflow_id = ?, owner_id = ?, "
                    "lease_token = ?, expires_at = ?, acquired_at = ?, renewed_at = ? "
                    "WHERE lease_name = ? AND expires_at <= ?",
                    (workflow, self.owner_id, token, _utc_text(expires), _utc_text(now), _utc_text(now), name, _utc_text(now)),
                )
        return WorkflowLease(
            lease_name=name, workflow_id=workflow, owner_id=self.owner_id,
            lease_token=token, expires_at=expires,
        )

    def renew_lease(
        self, lease: WorkflowLease, *, duration: timedelta | None = None
    ) -> WorkflowLease:
        lease = WorkflowLease.model_validate(lease)
        ttl = self.lease_duration if duration is None else duration
        if not isinstance(ttl, timedelta) or ttl <= timedelta(0):
            raise ValueError("lease duration must be positive")
        now = self._now()
        expires = now + ttl
        with self.store.transaction() as connection:
            cursor = connection.execute(
                "UPDATE workflow_leases SET expires_at = ?, renewed_at = ? "
                "WHERE lease_name = ? AND owner_id = ? AND lease_token = ? "
                "AND expires_at > ?",
                (_utc_text(expires), _utc_text(now), lease.lease_name, self.owner_id, lease.lease_token, _utc_text(now)),
            )
            if cursor.rowcount != 1:
                raise WorkflowConflict("workflow lease is stale")
        return lease.model_copy(update={"expires_at": expires})

    def release_lease(self, lease: WorkflowLease) -> None:
        lease = WorkflowLease.model_validate(lease)
        with self.store.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM workflow_leases WHERE lease_name = ? AND owner_id = ? "
                "AND lease_token = ? AND expires_at > ?",
                (lease.lease_name, self.owner_id, lease.lease_token, _utc_text(self._now())),
            )
            if cursor.rowcount != 1:
                raise WorkflowConflict("workflow lease is stale")

    def _ensure_run(
        self,
        *,
        kind: WorkflowKind,
        as_of: datetime,
        portfolio_date: datetime | None,
        source_hashes: tuple[str, ...],
        period_key: str,
        dry_run: bool,
        authorize_analysis: bool,
        industry_key: str | None,
        max_documents: int | None,
        recommendation_triggers: tuple[RecommendationTrigger, ...],
    ) -> tuple[str, str, bool]:
        definitions = self._definitions(kind, authorize_analysis=authorize_analysis)
        definition_payload = [item.model_dump(mode="json") for item in definitions]
        definition_hash = _digest(definition_payload)
        identity = {
            "kind": kind.value,
            "period": period_key,
            "as_of": _utc_text(as_of),
            "portfolio_date": None if portfolio_date is None else _utc_text(portfolio_date),
            "source_hashes": source_hashes,
            "definition_hash": definition_hash,
            "dry_run": dry_run,
            "authorize_analysis": authorize_analysis,
            "industry_key": industry_key,
            "max_documents": max_documents,
            "recommendation_triggers": tuple(
                trigger.value for trigger in recommendation_triggers
            ),
        }
        key = _digest(identity)
        workflow_id = f"wf_{key[:40]}"
        now = self._now()
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT workflow_id FROM workflow_runs WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if row is not None:
                return row[0], key, False
            connection.execute(
                "INSERT INTO workflow_runs (workflow_id, idempotency_key, workflow_kind, "
                "period_key, as_of, portfolio_date, source_hashes_json, definition_hash, "
                "state, dry_run, authorize_analysis, industry_key, max_documents, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?)",
                (
                    workflow_id, key, kind.value, period_key, _utc_text(as_of),
                    None if portfolio_date is None else _utc_text(portfolio_date),
                    _json(source_hashes), definition_hash, int(dry_run),
                    int(authorize_analysis), industry_key, max_documents,
                    _utc_text(now), _utc_text(now),
                ),
            )
            task_ids: dict[str, str] = {}
            for ordinal, definition in enumerate(definitions):
                task_key = _digest({
                    "workflow": key, "ordinal": ordinal,
                    "definition": definition.model_dump(mode="json"),
                })
                task_id = f"wft_{task_key[:40]}"
                task_ids[definition.stage] = task_id
                connection.execute(
                    "INSERT INTO workflow_tasks (task_id, workflow_id, idempotency_key, "
                    "stage, ordinal, state, max_attempts, assigned_role, originating_role, "
                    "created_at) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)",
                    (
                        task_id, workflow_id, task_key, definition.stage, ordinal,
                        definition.max_attempts,
                        None if definition.assigned_role is None else definition.assigned_role.value,
                        None if definition.originating_role is None else definition.originating_role.value,
                        _utc_text(now),
                    ),
                )
            for definition in definitions:
                for dependency in definition.dependencies:
                    connection.execute(
                        "INSERT INTO workflow_task_dependencies "
                        "(workflow_id, task_id, dependency_task_id) VALUES (?, ?, ?)",
                        (workflow_id, task_ids[definition.stage], task_ids[dependency]),
                    )
        return workflow_id, key, True

    def claim_task(self, workflow_id: str, task_id: str) -> TaskClaim:
        """Claim one dependency-ready task, reclaiming expired work safely."""
        workflow_id = _identifier(workflow_id)
        task_id = _identifier(task_id)
        now = self._now()
        expires = now + self.lease_duration
        token = secrets.token_hex(32)
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT state, attempt_count, max_attempts, lease_expires_at "
                "FROM workflow_tasks WHERE workflow_id = ? AND task_id = ?",
                (workflow_id, task_id),
            ).fetchone()
            if row is None:
                raise WorkflowConflict("workflow task does not exist")
            state, attempts, maximum, lease_expiry = row
            if state == "running" and lease_expiry is not None and _parse_utc(lease_expiry) > now:
                raise WorkflowBusy("workflow task is already claimed")
            if state == "running":
                connection.execute(
                    "UPDATE workflow_tasks SET state = 'failed', "
                    "defer_reason = 'task_lease_expired', lease_token = NULL, "
                    "lease_expires_at = NULL, completed_at = ? "
                    "WHERE workflow_id = ? AND task_id = ? AND state = 'running' "
                    "AND lease_expires_at <= ?",
                    (_utc_text(now), workflow_id, task_id, _utc_text(now)),
                )
                state = "failed"
            if state in {"completed", "deferred"}:
                raise WorkflowConflict("workflow task is terminal")
            if attempts >= maximum:
                raise WorkflowConflict("workflow task attempts are exhausted")
            if state == "failed":
                connection.execute(
                    "UPDATE workflow_tasks SET state = 'pending', defer_reason = NULL, "
                    "completed_at = NULL WHERE workflow_id = ? AND task_id = ? "
                    "AND state = 'failed'",
                    (workflow_id, task_id),
                )
                state = "pending"
            blocked = connection.execute(
                "SELECT 1 FROM workflow_task_dependencies AS dependency "
                "JOIN workflow_tasks AS required ON required.task_id = dependency.dependency_task_id "
                "WHERE dependency.workflow_id = ? AND dependency.task_id = ? "
                "AND required.state <> 'completed' LIMIT 1",
                (workflow_id, task_id),
            ).fetchone()
            if blocked is not None:
                raise WorkflowConflict("workflow task dependencies are incomplete")
            cursor = connection.execute(
                "UPDATE workflow_tasks SET state = 'running', attempt_count = attempt_count + 1, "
                "lease_token = ?, lease_expires_at = ?, started_at = COALESCE(started_at, ?) "
                "WHERE workflow_id = ? AND task_id = ? AND attempt_count = ?",
                (token, _utc_text(expires), _utc_text(now), workflow_id, task_id, attempts),
            )
            if cursor.rowcount != 1:
                raise WorkflowBusy("workflow task claim conflicted")
        return TaskClaim(
            task_id=task_id, lease_token=token,
            attempt_count=attempts + 1, lease_expires_at=expires,
        )

    def finalize_task(
        self,
        claim: TaskClaim,
        *,
        outcome: StageOutcome | None = None,
        failure_code: str | None = None,
        deferred: bool = False,
    ) -> None:
        """Finalize a claimed task with an expiry-aware compare-and-swap."""
        claim = TaskClaim.model_validate(claim)
        valid_success = outcome is not None and failure_code is None
        valid_failure = outcome is None and failure_code is not None and not deferred
        if not (valid_success or valid_failure):
            raise ValueError("task finalization must select exactly one outcome")
        now = self._now()
        state = "completed" if outcome is not None and not deferred else (
            "deferred" if deferred else "failed"
        )
        safe_failure = None if failure_code is None else _identifier(failure_code)
        result_ref = None if outcome is None else (outcome.result_ref or outcome.report_id)
        result_hash = None if outcome is None else (outcome.result_hash or _digest(outcome.model_dump(mode="json")))
        outcome_json = None if outcome is None else _json(outcome.model_dump(mode="json"))
        defer_reason = None if outcome is None else outcome.defer_reason
        with self.store.transaction() as connection:
            if outcome is None:
                cursor = connection.execute(
                    "UPDATE workflow_tasks SET state = 'failed', defer_reason = ?, "
                    "lease_token = NULL, lease_expires_at = NULL, completed_at = ? "
                    "WHERE task_id = ? AND state = 'running' AND lease_token = ? "
                    "AND lease_expires_at > ?",
                    (safe_failure, _utc_text(now), claim.task_id,
                     claim.lease_token, _utc_text(now)),
                )
            else:
                cursor = connection.execute(
                    "UPDATE workflow_tasks SET state = ?, defer_reason = ?, result_ref = ?, "
                    "result_hash = ?, outcome_json = ?, lease_token = NULL, "
                    "lease_expires_at = NULL, completed_at = ? "
                    "WHERE task_id = ? AND state = 'running' AND lease_token = ? "
                    "AND lease_expires_at > ?",
                    (
                        state, defer_reason, result_ref, result_hash,
                        outcome_json, _utc_text(now), claim.task_id,
                        claim.lease_token, _utc_text(now),
                    ),
                )
            if cursor.rowcount != 1:
                raise WorkflowConflict("workflow task lease is stale")

    def _dependency_refs(self, workflow_id: str, task_id: str) -> tuple[str, ...]:
        with self.store.connect() as connection:
            rows = connection.execute(
                "SELECT required.result_ref FROM workflow_task_dependencies AS dependency "
                "JOIN workflow_tasks AS required ON required.task_id = dependency.dependency_task_id "
                "WHERE dependency.workflow_id = ? AND dependency.task_id = ? "
                "AND required.result_ref IS NOT NULL ORDER BY required.ordinal",
                (workflow_id, task_id),
            ).fetchall()
        return tuple(row[0] for row in rows)

    def _allowed_claims_for_publish(self, workflow_id: str) -> tuple[str, ...]:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT outcome_json FROM workflow_tasks WHERE workflow_id = ? "
                "AND stage = 'review' AND state = 'completed'",
                (workflow_id,),
            ).fetchone()
        if row is None or row[0] is None:
            return ()
        outcome = StageOutcome.model_validate_json(row[0])
        return () if outcome.quality_gate is None else outcome.quality_gate.allowed_claim_ids

    @staticmethod
    def _publication_key(workflow_id: str, task_id: str) -> str:
        return _digest({"workflow_id": workflow_id, "task_id": task_id, "effect": "publish"})

    def _publication_effect(self, effect_key: str) -> PublicationEffectView | None:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT effect_key, workflow_id, task_id, state, report_id, "
                "receipt_hash FROM publication_effects WHERE effect_key = ?",
                (_hash(effect_key),),
            ).fetchone()
        return None if row is None else PublicationEffectView(
            effect_key=row[0], workflow_id=row[1], task_id=row[2],
            state=PublicationEffectState(row[3]), report_id=row[4], receipt_hash=row[5],
        )

    @staticmethod
    def _complete_publication_task(
        connection: sqlite3.Connection,
        *,
        workflow_id: str,
        task_id: str,
        report_id: str,
        receipt_hash: str,
        completed_at: datetime,
    ) -> None:
        outcome = StageOutcome(
            result_ref=report_id,
            report_id=report_id,
            publication_receipt_hash=receipt_hash,
        )
        outcome_json = _json(outcome.model_dump(mode="json"))
        connection.execute(
            "UPDATE workflow_tasks SET state = 'completed', defer_reason = NULL, "
            "result_ref = ?, result_hash = ?, outcome_json = ?, lease_token = NULL, "
            "lease_expires_at = NULL, completed_at = ? WHERE workflow_id = ? "
            "AND task_id = ? AND stage = 'publish' AND state <> 'completed'",
            (report_id, _digest(outcome.model_dump(mode="json")), outcome_json,
             _utc_text(completed_at), workflow_id, task_id),
        )

    def _prepare_publication_task(
        self, workflow_id: str, task_id: str
    ) -> tuple[TaskClaim | None, PublicationEffectView]:
        """Inspect receipt state before atomically consuming an external attempt."""
        effect_key = self._publication_key(workflow_id, task_id)
        now = self._now()
        expires = now + self.lease_duration
        token = secrets.token_hex(32)
        with self.store.transaction() as connection:
            task = connection.execute(
                "SELECT state, attempt_count, max_attempts, lease_expires_at "
                "FROM workflow_tasks WHERE workflow_id = ? AND task_id = ? "
                "AND stage = 'publish'", (workflow_id, task_id),
            ).fetchone()
            if task is None:
                raise WorkflowConflict("publish task does not exist")
            state, attempts, maximum, task_expiry = task
            effect = connection.execute(
                "SELECT state, claim_token, claim_expires_at, report_id, receipt_hash "
                "FROM publication_effects WHERE effect_key = ?", (effect_key,),
            ).fetchone()
            if effect is not None:
                effect_state, _, effect_expiry, report_id, receipt_hash = effect
                if effect_state == "claimed":
                    if _parse_utc(effect_expiry) > now:
                        raise WorkflowBusy("publication effect is already claimed")
                    connection.execute(
                        "UPDATE publication_effects SET state = 'outcome_unknown', "
                        "claim_token = NULL, claim_expires_at = NULL, updated_at = ? "
                        "WHERE effect_key = ? AND state = 'claimed' "
                        "AND claim_expires_at <= ?",
                        (_utc_text(now), effect_key, _utc_text(now)),
                    )
                    effect_state = "outcome_unknown"
                if effect_state == "outcome_unknown":
                    if state == "running" and (
                        task_expiry is None or _parse_utc(task_expiry) > now
                    ):
                        raise WorkflowBusy("publication task is already claimed")
                    connection.execute(
                        "UPDATE workflow_tasks SET state = 'deferred', "
                        "defer_reason = 'publication_outcome_unknown', "
                        "lease_token = NULL, lease_expires_at = NULL, completed_at = ? "
                        "WHERE workflow_id = ? AND task_id = ? "
                        "AND state IN ('pending', 'running', 'failed')",
                        (_utc_text(now), workflow_id, task_id),
                    )
                    return None, PublicationEffectView(
                        effect_key=effect_key, workflow_id=workflow_id,
                        task_id=task_id,
                        state=PublicationEffectState.OUTCOME_UNKNOWN,
                    )
                if effect_state in {"confirmed", "reconciled"}:
                    self._complete_publication_task(
                        connection, workflow_id=workflow_id, task_id=task_id,
                        report_id=report_id, receipt_hash=receipt_hash,
                        completed_at=now,
                    )
                    return None, PublicationEffectView(
                        effect_key=effect_key, workflow_id=workflow_id,
                        task_id=task_id, state=PublicationEffectState(effect_state),
                        report_id=report_id, receipt_hash=receipt_hash,
                    )
            if state == "completed":
                raise WorkflowConflict("completed publish task lacks a receipt")
            if state == "running":
                if task_expiry is not None and _parse_utc(task_expiry) > now:
                    raise WorkflowBusy("publication task is already claimed")
                connection.execute(
                    "UPDATE workflow_tasks SET state = 'failed', "
                    "defer_reason = 'task_lease_expired', lease_token = NULL, "
                    "lease_expires_at = NULL, completed_at = ? WHERE workflow_id = ? "
                    "AND task_id = ? AND state = 'running' AND lease_expires_at <= ?",
                    (_utc_text(now), workflow_id, task_id, _utc_text(now)),
                )
                state = "failed"
            if state == "deferred":
                raise WorkflowConflict("publication task is deferred")
            if attempts >= maximum:
                raise WorkflowConflict("publication task attempts are exhausted")
            if state == "failed":
                connection.execute(
                    "UPDATE workflow_tasks SET state = 'pending', defer_reason = NULL, "
                    "completed_at = NULL WHERE workflow_id = ? AND task_id = ? "
                    "AND state = 'failed'", (workflow_id, task_id),
                )
            blocked = connection.execute(
                "SELECT 1 FROM workflow_task_dependencies AS dependency "
                "JOIN workflow_tasks AS required "
                "ON required.task_id = dependency.dependency_task_id "
                "WHERE dependency.workflow_id = ? AND dependency.task_id = ? "
                "AND required.state <> 'completed' LIMIT 1",
                (workflow_id, task_id),
            ).fetchone()
            if blocked is not None:
                raise WorkflowConflict("publication dependencies are incomplete")
            connection.execute(
                "UPDATE workflow_tasks SET state = 'running', "
                "attempt_count = attempt_count + 1, lease_token = ?, "
                "lease_expires_at = ?, started_at = COALESCE(started_at, ?) "
                "WHERE workflow_id = ? AND task_id = ? AND state = 'pending' "
                "AND attempt_count = ?",
                (token, _utc_text(expires), _utc_text(now), workflow_id,
                 task_id, attempts),
            )
            connection.execute(
                "INSERT INTO publication_effects (publication_effect_id, effect_key, "
                "workflow_id, task_id, state, claim_token, claim_expires_at, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, 'claimed', ?, ?, ?, ?)",
                (f"pub_{effect_key[:40]}", effect_key, workflow_id, task_id,
                 token, _utc_text(expires), _utc_text(now), _utc_text(now)),
            )
        claim = TaskClaim(
            task_id=task_id, lease_token=token, attempt_count=attempts + 1,
            lease_expires_at=expires,
        )
        return claim, PublicationEffectView(
            effect_key=effect_key, workflow_id=workflow_id, task_id=task_id,
            state=PublicationEffectState.CLAIMED,
        )

    def _confirm_publication_effect(
        self, effect: PublicationEffectView, claim: TaskClaim, outcome: StageOutcome
    ) -> PublicationEffectView:
        if outcome.report_id is None or outcome.publication_receipt_hash is None:
            raise WorkflowConflict("publication receipt is missing")
        now = self._now()
        with self.store.transaction() as connection:
            cursor = connection.execute(
                "UPDATE publication_effects SET state = 'confirmed', claim_token = NULL, "
                "claim_expires_at = NULL, report_id = ?, receipt_hash = ?, updated_at = ? "
                "WHERE effect_key = ? AND state = 'claimed' AND claim_token = ? "
                "AND claim_expires_at > ? AND EXISTS (SELECT 1 FROM workflow_tasks "
                "WHERE task_id = ? AND state = 'running' AND lease_token = ? "
                "AND lease_expires_at > ?)",
                (outcome.report_id, outcome.publication_receipt_hash, _utc_text(now),
                 effect.effect_key, claim.lease_token, _utc_text(now), claim.task_id,
                 claim.lease_token, _utc_text(now)),
            )
            if cursor.rowcount != 1:
                raise WorkflowConflict("publication receipt lease is stale")
        return PublicationEffectView(
            effect_key=effect.effect_key, workflow_id=effect.workflow_id,
            task_id=effect.task_id, state=PublicationEffectState.CONFIRMED,
            report_id=outcome.report_id,
            receipt_hash=outcome.publication_receipt_hash,
        )

    def _mark_publication_unknown(
        self, effect: PublicationEffectView, claim: TaskClaim
    ) -> None:
        now = self._now()
        with self.store.transaction() as connection:
            connection.execute(
                "UPDATE publication_effects SET state = 'outcome_unknown', "
                "claim_token = NULL, claim_expires_at = NULL, updated_at = ? "
                "WHERE effect_key = ? AND state = 'claimed' AND claim_token = ?",
                (_utc_text(now), effect.effect_key, claim.lease_token),
            )
            connection.execute(
                "UPDATE workflow_tasks SET state = 'deferred', "
                "defer_reason = 'publication_outcome_unknown', lease_token = NULL, "
                "lease_expires_at = NULL, completed_at = ? WHERE task_id = ? "
                "AND state = 'running' AND lease_token = ?",
                (_utc_text(now), claim.task_id, claim.lease_token),
            )

    def reconcile_publication(
        self, effect_key: str, *, report_id: str, receipt_hash: str
    ) -> PublicationEffectView:
        """Record an externally verified receipt and make deferred work resumable."""
        key = _hash(effect_key)
        report = _identifier(report_id)
        receipt = _hash(receipt_hash)
        now = self._now()
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT workflow_id, task_id, state, report_id, receipt_hash "
                "FROM publication_effects WHERE effect_key = ?", (key,),
            ).fetchone()
            if row is None:
                raise WorkflowConflict("publication effect does not exist")
            workflow_id, task_id, state, stored_report, stored_receipt = row
            if state in {"confirmed", "reconciled"}:
                if (stored_report, stored_receipt) != (report, receipt):
                    raise WorkflowConflict("publication receipt conflicts")
            elif state != "outcome_unknown":
                raise WorkflowConflict("publication effect is not reconcilable")
            else:
                connection.execute(
                    "UPDATE publication_effects SET state = 'reconciled', report_id = ?, "
                    "receipt_hash = ?, updated_at = ? WHERE effect_key = ? "
                    "AND state = 'outcome_unknown'",
                    (report, receipt, _utc_text(now), key),
                )
                self._complete_publication_task(
                    connection, workflow_id=workflow_id, task_id=task_id,
                    report_id=report, receipt_hash=receipt, completed_at=now,
                )
                connection.execute(
                    "UPDATE workflow_runs SET state = 'running', completed_at = NULL, "
                    "updated_at = ? WHERE workflow_id = ? AND state = 'deferred'",
                    (_utc_text(now), workflow_id),
                )
        return PublicationEffectView(
            effect_key=key, workflow_id=workflow_id, task_id=task_id,
            state=(PublicationEffectState.RECONCILED
                   if state == "outcome_unknown" else PublicationEffectState(state)),
            report_id=report, receipt_hash=receipt,
        )

    def _task_rows(self, workflow_id: str) -> list[tuple[object, ...]]:
        with self.store.connect() as connection:
            return connection.execute(
                "SELECT task_id, idempotency_key, stage, state, attempt_count, "
                "max_attempts, assigned_role, originating_role, defer_reason, "
                "result_ref, result_hash, outcome_json, ordinal FROM workflow_tasks "
                "WHERE workflow_id = ? ORDER BY ordinal", (workflow_id,),
            ).fetchall()

    def list_tasks(self, workflow_id: str) -> tuple[DurableTaskView, ...]:
        workflow_id = _identifier(workflow_id)
        rows = self._task_rows(workflow_id)
        with self.store.connect() as connection:
            dependencies = {
                row[0]: tuple(item[0] for item in connection.execute(
                    "SELECT dependency_task_id FROM workflow_task_dependencies "
                    "WHERE workflow_id = ? AND task_id = ? ORDER BY dependency_task_id",
                    (workflow_id, row[0]),
                ).fetchall()) for row in rows
            }
        return tuple(
            DurableTaskView(
                task_id=row[0], idempotency_key=row[1], stage=row[2],
                dependency_ids=dependencies[row[0]], state=WorkflowTaskState(row[3]),
                attempt_count=row[4], max_attempts=row[5],
                assigned_role=None if row[6] is None else AgentRole(row[6]),
                originating_role=None if row[7] is None else AgentRole(row[7]),
                defer_reason=row[8], result_ref=row[9], result_hash=row[10],
            ) for row in rows
        )

    def _stored_result(self, workflow_id: str, *, replay: bool) -> WorkflowRunResult:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT workflow_kind, state, completed_stages_json, report_ids_json, "
                "new_agent_runs, omissions_json, dry_run FROM workflow_runs "
                "WHERE workflow_id = ?", (workflow_id,),
            ).fetchone()
        if row is None:
            raise WorkflowConflict("workflow does not exist")
        tasks = self.list_tasks(workflow_id)
        pending = tuple(task for task in tasks if task.state is WorkflowTaskState.PENDING)
        return WorkflowRunResult(
            workflow_id=workflow_id, workflow_kind=WorkflowKind(row[0]),
            status=WorkflowRunState(row[1]), completed_stages=tuple(json.loads(row[2])),
            report_ids=tuple(json.loads(row[3])),
            new_agent_runs=0 if replay else row[4], pending_tasks=pending,
            omissions=tuple(json.loads(row[5])), dry_run=bool(row[6]),
        )

    def _defer_optional(self, workflow_id: str, stages: Sequence[str], reason: str) -> None:
        now = self._now()
        with self.store.transaction() as connection:
            connection.execute(
                f"UPDATE workflow_tasks SET state = 'deferred', defer_reason = ?, "
                f"lease_token = NULL, lease_expires_at = NULL, completed_at = ? "
                f"WHERE workflow_id = ? AND stage IN "
                f"({','.join('?' for _ in stages)}) AND state = 'running' "
                f"AND lease_expires_at <= ?",
                (reason, _utc_text(now), workflow_id, *stages, _utc_text(now)),
            )
            connection.execute(
                f"UPDATE workflow_tasks SET state = 'deferred', defer_reason = ?, "
                f"completed_at = ? WHERE workflow_id = ? AND stage IN "
                f"({','.join('?' for _ in stages)}) AND state = 'pending'",
                (reason, _utc_text(now), workflow_id, *stages),
            )

    def _return_for_revision(
        self, workflow_id: str, review_task_id: str, outcome: StageOutcome
    ) -> None:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT originating_role FROM workflow_tasks WHERE task_id = ? "
                "AND workflow_id = ?", (review_task_id, workflow_id),
            ).fetchone()
        role = (
            AgentRole.FUNDAMENTAL_ANALYST
            if row is None or row[0] is None else AgentRole(row[0])
        )
        reason = (
            "review_block"
            if outcome.reviewer_verdict is ReviewVerdict.BLOCK
            else "review_revise"
            if outcome.reviewer_verdict is ReviewVerdict.REVISE
            else "review_requires_revision"
        )
        key = _digest({"workflow": workflow_id, "review": review_task_id, "role": role.value})
        task_id = f"wft_{key[:40]}"
        now = self._now()
        with self.store.transaction() as connection:
            ordinal = connection.execute(
                "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM workflow_tasks WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()[0]
            connection.execute(
                "UPDATE workflow_tasks SET state = 'deferred', defer_reason = ?, completed_at = ? "
                "WHERE workflow_id = ? AND stage = 'publish' AND state = 'pending'",
                ("review_not_passed", _utc_text(now), workflow_id),
            )
            connection.execute(
                "INSERT OR IGNORE INTO workflow_tasks (task_id, workflow_id, idempotency_key, "
                "stage, ordinal, state, max_attempts, assigned_role, originating_role, "
                "defer_reason, created_at) VALUES (?, ?, ?, 'revision', ?, 'pending', 2, ?, ?, ?, ?)",
                (task_id, workflow_id, key, ordinal, role.value, role.value, reason, _utc_text(now)),
            )
            connection.execute(
                "INSERT OR IGNORE INTO workflow_task_dependencies "
                "(workflow_id, task_id, dependency_task_id) VALUES (?, ?, ?)",
                (workflow_id, task_id, review_task_id),
            )

    @staticmethod
    def _publication_allowed(outcome: StageOutcome) -> bool:
        gate = outcome.quality_gate
        if outcome.reviewer_verdict in {ReviewVerdict.REVISE, ReviewVerdict.BLOCK}:
            return False
        if gate is None:
            return True
        if gate.publication_verdict is PublicationVerdict.DRAFT:
            return False
        if gate.review_verdict is not ReviewVerdict.PASS:
            return False
        return set(outcome.published_claim_ids) <= set(gate.allowed_claim_ids)

    def _execute(
        self,
        workflow_id: str,
        *,
        workflow_lease: WorkflowLease,
        kind: WorkflowKind,
        as_of: datetime,
        period_key: str,
        source_hashes: tuple[str, ...],
        industry_key: str | None,
        max_documents: int | None,
        authorize_analysis: bool,
        dry_run: bool,
        recommendation_triggers: tuple[RecommendationTrigger, ...],
    ) -> None:
        material_event = False
        accumulated_runs = 0
        reports: set[str] = set()
        omissions: set[str] = set()
        terminal = WorkflowRunState.COMPLETED
        partial_release = False
        for stored_row in self._task_rows(workflow_id):
            if stored_row[3] != "completed" or stored_row[11] is None:
                continue
            stored_outcome = StageOutcome.model_validate_json(stored_row[11])
            accumulated_runs += stored_outcome.new_agent_runs
            omissions.update(stored_outcome.omissions)
            if stored_outcome.report_id is not None:
                reports.add(stored_outcome.report_id)
            partial_release = partial_release or bool(stored_outcome.omissions) or (
                stored_outcome.quality_gate is not None
                and stored_outcome.quality_gate.publication_verdict
                is PublicationVerdict.PARTIAL
            )
            if stored_row[2] == "materiality":
                material_event = bool(stored_outcome.material_event)
        while True:
            workflow_lease = self.renew_lease(workflow_lease)
            rows = self._task_rows(workflow_id)
            next_row = next((row for row in rows if row[3] in {"pending", "failed", "running"}), None)
            if next_row is None:
                break
            task_id, _, stage, state, attempts, maximum, *_ = next_row
            if state == "failed" and attempts >= maximum:
                terminal = WorkflowRunState.FAILED
                break
            if kind is WorkflowKind.DAILY and stage == "event_analysis" and not material_event:
                self._defer_optional(
                    workflow_id,
                    ("event_analysis", "event_update", "review", "publish"),
                    "no_material_event",
                )
                continue
            if stage == "publish" and dry_run:
                self._defer_optional(workflow_id, ("publish",), "dry_run")
                omissions.add("publication_dry_run")
                terminal = WorkflowRunState.PARTIAL
                continue
            publication_effect: PublicationEffectView | None = None
            publish_lease: WorkflowLease | None = None
            if stage == "publish":
                publish_lease = self.acquire_lease(
                    f"publish:{workflow_id}", workflow_id=workflow_id
                )
            try:
                if stage == "publish":
                    claim, publication_effect = self._prepare_publication_task(
                        workflow_id, task_id
                    )
                else:
                    claim = self.claim_task(workflow_id, task_id)
            except WorkflowBusy:
                if publish_lease is not None:
                    self.release_lease(publish_lease)
                raise
            except WorkflowConflict:
                if publish_lease is not None:
                    self.release_lease(publish_lease)
                terminal = WorkflowRunState.FAILED
                break
            if publication_effect is not None:
                if publication_effect.state is PublicationEffectState.OUTCOME_UNKNOWN:
                    if publish_lease is not None:
                        self.release_lease(publish_lease)
                    terminal = WorkflowRunState.DEFERRED
                    break
                if publication_effect.state in {
                    PublicationEffectState.CONFIRMED,
                    PublicationEffectState.RECONCILED,
                }:
                    reports.add(publication_effect.report_id)
                    if publish_lease is not None:
                        self.release_lease(publish_lease)
                    continue
            if claim is None:  # pragma: no cover - guarded by effect states above
                raise WorkflowConflict("publication preflight did not claim work")
            context = StageContext(
                workflow_id=workflow_id, workflow_kind=kind, task_id=task_id,
                stage=stage, as_of=as_of, period_key=period_key,
                source_hashes=source_hashes,
                dependency_result_refs=self._dependency_refs(workflow_id, task_id),
                allowed_claim_ids=(
                    self._allowed_claims_for_publish(workflow_id)
                    if stage == "publish" else ()
                ),
                industry_key=industry_key, max_documents=max_documents,
                authorize_analysis=authorize_analysis, dry_run=dry_run,
                recommendation_triggers=recommendation_triggers,
                publication_effect_key=(
                    None if publication_effect is None
                    else publication_effect.effect_key
                ),
                task_lease_token=claim.lease_token,
            )
            try:
                if publication_effect is not None and publication_effect.state in {
                    PublicationEffectState.CONFIRMED,
                    PublicationEffectState.RECONCILED,
                }:
                    outcome = StageOutcome(
                        result_ref=publication_effect.report_id,
                        report_id=publication_effect.report_id,
                        publication_receipt_hash=publication_effect.receipt_hash,
                    )
                else:
                    outcome = StageOutcome.model_validate(self.runner.run(context))
                if stage == "publish" and (
                    not self._publication_allowed(outcome)
                    or not set(outcome.published_claim_ids)
                    <= set(context.allowed_claim_ids)
                ):
                    if publication_effect is not None:
                        self._mark_publication_unknown(publication_effect, claim)
                        terminal = WorkflowRunState.DEFERRED
                    else:  # pragma: no cover - publish always has an intent
                        self.finalize_task(
                            claim, failure_code="publication_quality_blocked"
                        )
                        terminal = WorkflowRunState.BLOCKED
                    break
                if (
                    publication_effect is not None
                    and publication_effect.state is PublicationEffectState.CLAIMED
                ):
                    try:
                        self._confirm_publication_effect(
                            publication_effect, claim, outcome
                        )
                    except WorkflowConflict:
                        self._mark_publication_unknown(publication_effect, claim)
                        terminal = WorkflowRunState.DEFERRED
                        break
                if stage == "review" and not self._publication_allowed(outcome):
                    self.finalize_task(claim, outcome=outcome)
                    self._return_for_revision(workflow_id, task_id, outcome)
                    omissions.update(outcome.omissions)
                    terminal = WorkflowRunState.BLOCKED
                    break
                if outcome.defer_reason is not None and stage != "review":
                    self.finalize_task(claim, outcome=outcome, deferred=True)
                    omissions.update(outcome.omissions)
                    terminal = WorkflowRunState.DEFERRED
                    break
                self.finalize_task(claim, outcome=outcome)
            except Exception:
                if publication_effect is not None:
                    self._mark_publication_unknown(publication_effect, claim)
                    terminal = WorkflowRunState.DEFERRED
                    break
                try:
                    self.finalize_task(claim, failure_code="stage_execution_failed")
                except WorkflowConflict:
                    pass
                if claim.attempt_count >= maximum:
                    terminal = WorkflowRunState.FAILED
                    break
                continue
            finally:
                if publish_lease is not None:
                    try:
                        self.release_lease(publish_lease)
                    except WorkflowConflict:
                        pass
            accumulated_runs += outcome.new_agent_runs
            omissions.update(outcome.omissions)
            partial_release = partial_release or bool(outcome.omissions) or (
                outcome.quality_gate is not None
                and outcome.quality_gate.publication_verdict
                is PublicationVerdict.PARTIAL
            )
            if outcome.report_id is not None:
                reports.add(outcome.report_id)
            if stage == "materiality":
                material_event = bool(outcome.material_event)

        tasks = self.list_tasks(workflow_id)
        completed = [task.stage for task in tasks if task.state is WorkflowTaskState.COMPLETED]
        if kind is WorkflowKind.DAILY and not material_event:
            completed = [
                stage for stage in ("portfolio", "ingestion", "materiality")
                if stage in completed
            ]
            completed.append("event_update")
        pending = any(task.state is WorkflowTaskState.PENDING for task in tasks)
        if terminal is WorkflowRunState.COMPLETED and pending:
            terminal = WorkflowRunState.PARTIAL
        if terminal is WorkflowRunState.COMPLETED and partial_release:
            terminal = WorkflowRunState.PARTIAL
        now = self._now()
        with self.store.transaction() as connection:
            connection.execute(
                "UPDATE workflow_runs SET state = ?, completed_stages_json = ?, "
                "report_ids_json = ?, omissions_json = ?, new_agent_runs = ?, "
                "updated_at = ?, completed_at = ? WHERE workflow_id = ? AND state = 'running'",
                (
                    terminal.value, _json(completed), _json(sorted(reports)),
                    _json(sorted(omissions)), accumulated_runs, _utc_text(now),
                    _utc_text(now), workflow_id,
                ),
            )

    def _run(
        self,
        kind: WorkflowKind,
        *,
        as_of: datetime,
        source_hashes: Sequence[str],
        period_key: str | None = None,
        portfolio_date: datetime | None = None,
        dry_run: bool = False,
        authorize_analysis: bool = False,
        industry_key: str | None = None,
        max_documents: int | None = None,
        recommendation_triggers: Sequence[RecommendationTrigger] = (),
    ) -> WorkflowRunResult:
        as_of = _strict_utc(as_of)
        if not isinstance(dry_run, bool):
            raise TypeError("dry_run must be bool")
        if portfolio_date is not None:
            portfolio_date = _strict_utc(portfolio_date)
        sources = _canonical_ids(tuple(source_hashes), hashes=True)
        period = _identifier(period_key or self._period(kind, as_of))
        industry = _optional_identifier(industry_key)
        triggers = tuple(RecommendationTrigger(trigger) for trigger in recommendation_triggers)
        if len(triggers) != len(set(triggers)):
            raise ValueError("recommendation triggers must be unique")
        triggers = tuple(sorted(triggers, key=lambda trigger: trigger.value))
        workflow_id, key, created = self._ensure_run(
            kind=kind, as_of=as_of, portfolio_date=portfolio_date,
            source_hashes=sources, period_key=period, dry_run=dry_run,
            authorize_analysis=authorize_analysis, industry_key=industry,
            max_documents=max_documents, recommendation_triggers=triggers,
        )
        stored = self._stored_result(workflow_id, replay=not created)
        if stored.status is not WorkflowRunState.RUNNING:
            return stored
        lease = self.acquire_lease(f"workflow:{key}", workflow_id=workflow_id)
        executed = False
        try:
            current = self._stored_result(workflow_id, replay=not created)
            if current.status is WorkflowRunState.RUNNING:
                self._execute(
                    workflow_id, workflow_lease=lease, kind=kind, as_of=as_of,
                    period_key=period,
                    source_hashes=sources, industry_key=industry,
                    max_documents=max_documents, authorize_analysis=authorize_analysis,
                    dry_run=dry_run, recommendation_triggers=triggers,
                )
                executed = True
            return self._stored_result(
                workflow_id, replay=not (created or executed)
            )
        finally:
            try:
                self.release_lease(lease)
            except WorkflowConflict:
                pass

    def run_daily(self, *, as_of: datetime, source_hashes: Sequence[str] = (),
                  period_key: str | None = None, portfolio_date: datetime | None = None,
                  dry_run: bool = False) -> WorkflowRunResult:
        return self._run(
            WorkflowKind.DAILY, as_of=as_of, source_hashes=source_hashes,
            period_key=period_key, portfolio_date=portfolio_date, dry_run=dry_run,
        )

    def run_weekly(self, *, as_of: datetime, source_hashes: Sequence[str] = (),
                   period_key: str | None = None, portfolio_date: datetime | None = None,
                   dry_run: bool = False,
                   recommendation_triggers: Sequence[RecommendationTrigger] = (
                       RecommendationTrigger.SCHEDULED,
                   )) -> WorkflowRunResult:
        if not recommendation_triggers:
            raise ValueError("weekly workflow requires a recommendation trigger")
        return self._run(
            WorkflowKind.WEEKLY, as_of=as_of, source_hashes=source_hashes,
            period_key=period_key, portfolio_date=portfolio_date, dry_run=dry_run,
            recommendation_triggers=recommendation_triggers,
        )

    def run_monthly(self, *, as_of: datetime, industry_key: str,
                    source_hashes: Sequence[str] = (), period_key: str | None = None,
                    dry_run: bool = False) -> WorkflowRunResult:
        return self._run(
            WorkflowKind.MONTHLY, as_of=as_of, source_hashes=source_hashes,
            period_key=period_key, industry_key=industry_key, dry_run=dry_run,
        )

    def run_backfill(self, *, as_of: datetime, max_documents: int,
                     source_hashes: Sequence[str] = (), period_key: str | None = None,
                     authorize_analysis: bool = False,
                     dry_run: bool = False) -> WorkflowRunResult:
        if isinstance(max_documents, bool) or not isinstance(max_documents, int):
            raise TypeError("max_documents must be an integer")
        if not 1 <= max_documents <= _MAX_BACKFILL_DOCUMENTS:
            raise ValueError("max_documents must be between 1 and 1000")
        if not isinstance(authorize_analysis, bool):
            raise TypeError("authorize_analysis must be bool")
        return self._run(
            WorkflowKind.BACKFILL, as_of=as_of, source_hashes=source_hashes,
            period_key=period_key, max_documents=max_documents,
            authorize_analysis=authorize_analysis, dry_run=dry_run,
        )


__all__ = [
    "DurableTaskView", "PublicationEffectState", "PublicationEffectView",
    "RecommendationTrigger", "ResearchOrchestrator", "StageContext", "StageOutcome",
    "StageRunner", "TaskClaim", "WorkflowBusy", "WorkflowConflict", "WorkflowKind",
    "WorkflowLease", "WorkflowRunResult", "WorkflowRunState", "WorkflowTaskState",
]
