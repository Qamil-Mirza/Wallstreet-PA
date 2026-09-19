"""Durable SQLite persistence for research evidence and revision history."""

import hashlib
import json
import re
import secrets
import sqlite3
from importlib import resources
from importlib.resources.abc import Traversable
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path

from .models import (
    AgentRole,
    ClaimKind,
    EvidenceClaim,
    InferenceMode,
    PortfolioSnapshot,
    Position,
    SourceDocument,
)


_MIGRATION_NAME = re.compile(r"^(?P<version>[0-9]+)_.+\.sql$")
_AGENT_EXECUTION_LEASE = timedelta(minutes=15)
_WORKFLOW_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PUBLICATION_OMISSION_CODES = frozenset({
    "approved_claim_unknown",
    "budget_deferred",
    "calendar_coverage_unknown",
    "deferred_by_stage",
    "dry_run",
    "editor_claim_unknown",
    "editor_selection_missing",
    "editor_unapproved_claim",
    "event_freshness_unknown",
    "event_outside_materiality_window",
    "exhibit_mismatch",
    "filing_status_unknown",
    "inference_disclosure_missing",
    "insufficient_corroboration",
    "local_confidence_weak",
    "missing_claim_lineage",
    "no_material_event",
    "omission_sanitized",
    "portfolio_missing",
    "portfolio_stale",
    "price_currency_mismatch",
    "price_missing",
    "price_not_latest_session",
    "price_not_trading_session",
    "price_session_unknown",
    "price_stale",
    "publication_dry_run",
    "publication_outcome_unknown",
    "publication_quality_blocked",
    "recommendation_evidence_missing",
    "required_filing_unavailable",
    "research_claims_missing",
    "review_block",
    "review_not_passed",
    "review_requires_revision",
    "review_revise",
    "reviewer_approval_missing",
    "reviewer_not_passed",
    "search_snippet_inadmissible",
    "security_ineligible",
    "stage_execution_failed",
    "task_lease_expired",
    "unresolved_contradiction",
})


def _system_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_hex(value: str) -> str:
    """Hash exact SQLite TEXT for receipt/result integrity triggers."""
    if not isinstance(value, str):
        raise ValueError("sha256_hex requires text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _publication_effect_key(
    workflow_id: object, task_id: object
) -> str | None:
    """Return the canonical publish-effect key for two safe identifiers."""
    if (
        not isinstance(workflow_id, str)
        or not isinstance(task_id, str)
        or _WORKFLOW_IDENTIFIER.fullmatch(workflow_id) is None
        or _WORKFLOW_IDENTIFIER.fullmatch(task_id) is None
    ):
        return None
    payload = {
        "workflow_id": workflow_id,
        "task_id": task_id,
        "effect": "publish",
    }
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _is_canonical_publication_outcome(
    outcome_json: object,
    result_ref: object,
    effect_report_id: object,
    effect_receipt_hash: object,
) -> int:
    """Validate the exact safe StageOutcome envelope without importing it."""
    try:
        if not all(
            isinstance(value, str)
            for value in (
                outcome_json, result_ref, effect_report_id, effect_receipt_hash
            )
        ):
            return 0
        if (
            result_ref != effect_report_id
            or _WORKFLOW_IDENTIFIER.fullmatch(result_ref) is None
            or _SHA256.fullmatch(effect_receipt_hash) is None
        ):
            return 0
        parsed = json.loads(outcome_json)
        if not isinstance(parsed, dict):
            return 0
        expected = {
            "result_ref": result_ref,
            "result_hash": None,
            "report_id": effect_report_id,
            "new_agent_runs": 0,
            "material_event": None,
            "omissions": [],
            "reviewer_verdict": None,
            "originating_role": None,
            "defer_reason": None,
            "quality_gate": None,
            "published_claim_ids": [],
            "publication_receipt_hash": effect_receipt_hash,
        }
        canonical = json.dumps(
            expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return int(outcome_json == canonical)
    except (RecursionError, TypeError, ValueError):
        return 0


def _is_canonical_publication_effect_outcome(
    outcome_json: object,
    result_ref: object,
    effect_report_id: object,
    effect_receipt_hash: object,
) -> int:
    """Validate an exact safe publish envelope with bounded audit metadata."""
    try:
        if not all(
            isinstance(value, str)
            for value in (
                outcome_json, result_ref, effect_report_id, effect_receipt_hash
            )
        ):
            return 0
        if (
            result_ref != effect_report_id
            or _WORKFLOW_IDENTIFIER.fullmatch(result_ref) is None
            or _SHA256.fullmatch(effect_receipt_hash) is None
        ):
            return 0
        parsed = json.loads(outcome_json)
        if not isinstance(parsed, dict):
            return 0
        new_agent_runs = parsed.get("new_agent_runs")
        omissions = parsed.get("omissions")
        published_claim_ids = parsed.get("published_claim_ids")
        if type(new_agent_runs) is not int or new_agent_runs < 0:
            return 0
        if (
            not isinstance(omissions, list)
            or any(
                not isinstance(code, str)
                or code not in _PUBLICATION_OMISSION_CODES
                for code in omissions
            )
            or omissions != sorted(set(omissions))
        ):
            return 0
        if (
            not isinstance(published_claim_ids, list)
            or any(
                not isinstance(claim_id, str)
                or _WORKFLOW_IDENTIFIER.fullmatch(claim_id) is None
                for claim_id in published_claim_ids
            )
            or published_claim_ids != sorted(set(published_claim_ids))
        ):
            return 0
        expected = {
            "result_ref": result_ref,
            "result_hash": None,
            "report_id": effect_report_id,
            "new_agent_runs": new_agent_runs,
            "material_event": None,
            "omissions": omissions,
            "reviewer_verdict": None,
            "originating_role": None,
            "defer_reason": None,
            "quality_gate": None,
            "published_claim_ids": published_claim_ids,
            "publication_receipt_hash": effect_receipt_hash,
        }
        canonical = json.dumps(
            expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return int(outcome_json == canonical)
    except (RecursionError, TypeError, ValueError):
        return 0


class _ExactDecimalSum:
    """SQLite aggregate that preserves arbitrary Decimal text precision."""

    def __init__(self) -> None:
        self.values: list[Decimal] = []

    def step(self, value: object) -> None:
        if value is None:
            return
        if not isinstance(value, str):
            raise ValueError("decimal aggregate requires text")
        amount = Decimal(value)
        if not amount.is_finite() or amount < 0:
            raise ValueError(
                "decimal aggregate requires a finite non-negative value"
            )
        self.values.append(amount)

    def finalize(self) -> str | None:
        if not self.values:
            return None
        nonzero = [value for value in self.values if value]
        if not nonzero:
            return "0"
        highest_place = max(value.adjusted() for value in nonzero)
        lowest_place = min(value.as_tuple().exponent for value in nonzero)
        with localcontext() as context:
            context.prec = max(
                28,
                highest_place - lowest_place + len(self.values).bit_length() + 2,
            )
            total = sum(self.values, Decimal("0"))
        return _decimal_text(total)


class ResearchStoreError(RuntimeError):
    """Raised when the research store cannot safely initialize."""


class AgentExecutionConflict(ResearchStoreError):
    """Raised when a logical execution is already claimed or task state conflicts."""


def _has_sql_content(script: str) -> bool:
    """Return whether text contains anything other than whitespace/comments."""
    index = 0
    while index < len(script):
        if script[index].isspace():
            index += 1
            continue
        if script.startswith("--", index):
            newline = script.find("\n", index + 2)
            index = len(script) if newline < 0 else newline + 1
            continue
        if script.startswith("/*", index):
            closing = script.find("*/", index + 2)
            index = len(script) if closing < 0 else closing + 2
            continue
        return True
    return False


def _utc_text(value: datetime, field_name: str = "store datetime") -> str:
    """Serialize an aware datetime in the store's canonical UTC format."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_utc(value: str) -> datetime:
    """Parse canonical UTC text back to an aware datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _decimal_text(value: Decimal) -> str:
    """Serialize a finite Decimal without exponent notation."""
    if not isinstance(value, Decimal):
        raise TypeError("store decimal values must be Decimal")
    if not value.is_finite():
        raise ValueError("store decimal values must be finite")
    return format(value, "f")


def _validate_lease_token(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value or "") is None:
        raise ValueError("agent execution lease token is invalid")
    return value


def _canonical_json(value: Mapping[str, object]) -> str:
    """Serialize optional structured metadata deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ThesisRevision:
    """One immutable revision in an industry's thesis history."""

    thesis_id: int
    industry_key: str
    version: int
    thesis_text: str
    created_at: datetime
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class DocumentPassageRecord:
    """One immutable, source-relative passage persisted for citation."""

    passage_id: str
    document_id: str
    ordinal: int
    text: str
    content_hash: str
    start_offset: int
    end_offset: int
    published_at: datetime | None = None
    retrieved_at: datetime | None = None


@dataclass(frozen=True)
class ClaimLineageRecord:
    """One exact passage linked to a claim with an epistemic stance."""

    claim_id: str
    passage: DocumentPassageRecord
    stance: str


@dataclass(frozen=True)
class AgentEvidencePacket:
    """Deterministically ordered, citation-ready evidence for one agent call."""

    passages: tuple[DocumentPassageRecord, ...]
    claims: tuple[EvidenceClaim, ...]


@dataclass(frozen=True)
class AgentExecutionClaim:
    """Exclusive lease or validated terminal replay for one logical execution."""

    attempt_id: str
    lease_token: str | None
    replay_output_json: str | None = None
    replay_output_hash: str | None = None
    replay_evidence_hash: str | None = None


@dataclass(frozen=True)
class AgentRunAudit:
    """Redacted metadata retained for one completed bounded agent run."""

    run_id: str
    task_id: str
    role: AgentRole
    started_at: datetime
    completed_at: datetime | None
    provider: str | None
    model: str | None
    inference_mode: InferenceMode | None
    prompt_hash: str
    evidence_hash: str | None
    output_hash: str | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    fallback_reason: str | None = None
    attempt_id: str | None = None
    status: str = "succeeded"
    safe_failure_code: str | None = None
    usage_known: bool = True
    provider_attempt_count: int = 0
    reservation_state: str | None = None
    reserved_cost_usd: Decimal | None = None


@dataclass(frozen=True)
class ProviderAttemptAudit:
    """One redacted physical provider call retained under an agent execution."""

    status: str
    provider: str
    model: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    inference_mode: InferenceMode
    recorded_at: datetime
    fallback_reason: str | None = None
    response_hash: str | None = None
    failure_code: str | None = None
    usage_known: bool = True
    reservation_id: str | None = None
    reservation_state: str | None = None
    reserved_cost_usd: Decimal | None = None


@dataclass(frozen=True)
class ProviderAttemptRecord:
    """One immutable physical provider attempt with deterministic ordinal."""

    provider_attempt_id: str
    attempt_id: str
    ordinal: int
    status: str
    provider: str
    model: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    usage_known: bool
    inference_mode: InferenceMode
    recorded_at: datetime
    fallback_reason: str | None = None
    response_hash: str | None = None
    failure_code: str | None = None
    reservation_id: str | None = None
    reservation_state: str | None = None
    reserved_cost_usd: Decimal | None = None


class ResearchStore:
    """Connection-per-operation SQLite research store."""

    def __init__(
        self,
        database_path: Path,
        *,
        clock: Callable[[], datetime] = _system_utc_now,
    ) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path must be Path")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.database_path = database_path
        self.clock = clock

    def connect(self) -> sqlite3.Connection:
        """Open one configured SQLite connection owned by the caller."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, isolation_level=None)
        try:
            self._register_sql_functions(connection)
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA busy_timeout = 5000")
        except BaseException:
            connection.close()
            raise
        return connection

    def _research_utc_now(self) -> str:
        """Return the authoritative store clock as strict canonical UTC text."""
        value = self.clock()
        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() != timedelta(0)
        ):
            raise ValueError("research store clock must return a UTC datetime")
        return _utc_text(value, "research store clock")

    def _register_sql_functions(self, connection: sqlite3.Connection) -> None:
        connection.create_aggregate("decimal_sum_exact", 1, _ExactDecimalSum)
        connection.create_function("sha256_hex", 1, _sha256_hex, deterministic=True)
        connection.create_function(
            "publication_effect_key", 2, _publication_effect_key, deterministic=True
        )
        connection.create_function(
            "research_utc_now", 0, self._research_utc_now, deterministic=False
        )
        connection.create_function(
            "is_canonical_publication_outcome",
            4,
            _is_canonical_publication_outcome,
            deterministic=True,
        )
        connection.create_function(
            "is_canonical_publication_effect_outcome",
            4,
            _is_canonical_publication_effect_outcome,
            deterministic=True,
        )

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Run a bounded immediate transaction and always close its connection."""
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _migration_files(self) -> tuple[Traversable, ...]:
        migration_dir = resources.files("news_bot.research").joinpath("migrations")
        if not migration_dir.is_dir():
            return ()
        return tuple(
            child
            for child in migration_dir.iterdir()
            if child.is_file() and child.name.endswith(".sql")
        )

    @staticmethod
    def _migration_version(path: Traversable) -> int:
        match = _MIGRATION_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Invalid migration filename: {path.name}")
        return int(match.group("version"))

    @staticmethod
    def _execute_script_atomically(
        connection: sqlite3.Connection, script: str
    ) -> None:
        """Execute a script without sqlite3.executescript's implicit COMMIT."""
        statement_parts: list[str] = []
        for character in script:
            statement_parts.append(character)
            statement = "".join(statement_parts)
            if sqlite3.complete_statement(statement):
                if _has_sql_content(statement):
                    connection.execute(statement)
                statement_parts.clear()
        if _has_sql_content("".join(statement_parts)):
            raise sqlite3.OperationalError("incomplete SQL migration statement")

    def migrate(self) -> None:
        """Apply each packaged numbered migration exactly once, in order."""
        migration_files = self._migration_files()
        if not migration_files:
            raise ResearchStoreError("No research store migrations found")
        migrations = [
            (self._migration_version(path), path) for path in migration_files
        ]
        versions = [version for version, _ in migrations]
        if len(versions) != len(set(versions)):
            raise ValueError("Migration versions must be unique")
        migrations.sort(key=lambda item: item[0])

        with self.transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, "
                "name TEXT NOT NULL UNIQUE, "
                "applied_at TEXT NOT NULL)"
            )

        for version, path in migrations:
            with self.transaction() as connection:
                applied = connection.execute(
                    "SELECT name FROM schema_migrations WHERE version = ?",
                    (version,),
                ).fetchone()
                if applied is not None:
                    if applied[0] != path.name:
                        raise sqlite3.IntegrityError(
                            f"Migration version {version} is already {applied[0]}"
                        )
                    continue
                self._execute_script_atomically(
                    connection, path.read_text(encoding="utf-8")
                )
                connection.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) "
                    "VALUES (?, ?, ?)",
                    (version, path.name, _utc_text(datetime.now(timezone.utc))),
                )

    def table_names(self) -> set[str]:
        """Return user-defined table names for focused schema inspection."""
        connection = self.connect()
        try:
            rows = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            return {row[0] for row in rows}
        finally:
            connection.close()

    def upsert_resolved_entity(
        self, resolved: object, aliases: Sequence[object], provenance: object
    ) -> None:
        """Atomically persist an idempotent canonical entity resolution."""
        from .entities import Alias, ResolutionProvenance, ResolvedEntity

        if not isinstance(resolved, ResolvedEntity):
            raise TypeError("resolved must be ResolvedEntity")
        alias_records = tuple(aliases)
        if any(not isinstance(alias, Alias) for alias in alias_records):
            raise TypeError("aliases must contain Alias records")
        if not isinstance(provenance, ResolutionProvenance):
            raise TypeError("provenance must be ResolutionProvenance")
        if any(alias.entity_id != resolved.entity_id for alias in alias_records):
            raise sqlite3.IntegrityError("alias entity_id does not match entity")
        if provenance.entity_id != resolved.entity_id:
            raise sqlite3.IntegrityError(
                "resolution provenance entity_id does not match entity"
            )
        with self.transaction() as connection:
            entity_values = (
                resolved.entity_id,
                resolved.canonical_name,
                "company",
            )
            existing_entity = connection.execute(
                "SELECT entity_id, canonical_name, entity_type FROM entities "
                "WHERE entity_id = ?",
                (resolved.entity_id,),
            ).fetchone()
            created_at = _utc_text(datetime.now(timezone.utc))
            if existing_entity is None:
                connection.execute(
                    "INSERT INTO entities (entity_id, canonical_name, entity_type, "
                    "created_at) VALUES (?, ?, ?, ?)",
                    (*entity_values, created_at),
                )
            elif tuple(existing_entity) != entity_values:
                raise sqlite3.IntegrityError(
                    "conflicting canonical entity already exists"
                )
            for alias in alias_records:
                values = (
                    alias.alias_id,
                    alias.entity_id,
                    alias.value,
                    alias.alias_type,
                    alias.market,
                )
                existing_alias = connection.execute(
                    "SELECT alias_id, entity_id, value, alias_type, market "
                    "FROM entity_aliases WHERE alias_id = ?",
                    (alias.alias_id,),
                ).fetchone()
                if existing_alias is None:
                    connection.execute(
                        "INSERT INTO entity_aliases (alias_id, entity_id, value, "
                        "alias_type, market, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (*values, created_at),
                    )
                elif tuple(existing_alias) != values:
                    raise sqlite3.IntegrityError("conflicting entity alias exists")
            provenance_values = (
                provenance.provenance_id,
                provenance.entity_id,
                provenance.method,
                provenance.source,
                provenance.matched_identifier,
            )
            existing_provenance = connection.execute(
                "SELECT provenance_id, entity_id, method, source, "
                "matched_identifier FROM entity_resolution_provenance "
                "WHERE provenance_id = ?",
                (provenance.provenance_id,),
            ).fetchone()
            if existing_provenance is None:
                connection.execute(
                    "INSERT INTO entity_resolution_provenance (provenance_id, "
                    "entity_id, method, source, matched_identifier, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (*provenance_values, created_at),
                )
            elif tuple(existing_provenance) != provenance_values:
                raise sqlite3.IntegrityError(
                    "conflicting entity resolution provenance exists"
                )

    def list_entity_aliases(self, entity_id: str) -> tuple[object, ...]:
        """Return immutable aliases in deterministic order."""
        from .entities import Alias

        connection = self.connect()
        try:
            rows = connection.execute(
                "SELECT alias_id, entity_id, value, alias_type, market "
                "FROM entity_aliases WHERE entity_id = ? ORDER BY alias_id",
                (entity_id,),
            ).fetchall()
        finally:
            connection.close()
        return tuple(Alias(*row) for row in rows)

    def list_resolution_provenance(self, entity_id: str) -> tuple[object, ...]:
        """Return resolution provenance in deterministic order."""
        from .entities import ResolutionProvenance

        connection = self.connect()
        try:
            rows = connection.execute(
                "SELECT provenance_id, entity_id, method, source, "
                "matched_identifier FROM entity_resolution_provenance "
                "WHERE entity_id = ? ORDER BY provenance_id",
                (entity_id,),
            ).fetchall()
        finally:
            connection.close()
        return tuple(ResolutionProvenance(*row) for row in rows)

    def upsert_unresolved_research_task(self, task: object) -> None:
        """Persist a deduplicated entity-resolution task without guessing."""
        from .entities import UnresolvedResearchTask

        if not isinstance(task, UnresolvedResearchTask):
            raise TypeError("task must be UnresolvedResearchTask")
        values = (
            task.task_id,
            "entity_resolution",
            _canonical_json(
                {
                    "candidate_entity_ids": list(task.candidate_entity_ids),
                    "normalized_query": task.normalized_query,
                    "reason": task.reason,
                }
            ),
            "pending",
            0,
        )
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT task_id, task_kind, scope_json, state, priority "
                "FROM research_tasks WHERE task_id = ?",
                (task.task_id,),
            ).fetchone()
            if existing is not None:
                if tuple(existing) == values:
                    return
                raise sqlite3.IntegrityError(
                    "conflicting unresolved research task exists"
                )
            connection.execute(
                "INSERT INTO research_tasks (task_id, task_kind, scope_json, "
                "state, priority, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (*values, _utc_text(datetime.now(timezone.utc))),
            )

    @staticmethod
    def _references_exist(
        connection: sqlite3.Connection,
        passage_ids: tuple[str, ...],
        claim_ids: tuple[str, ...],
    ) -> bool:
        if not passage_ids and not claim_ids:
            return False
        if passage_ids:
            placeholders = ",".join("?" for _ in passage_ids)
            passage_count = connection.execute(
                f"SELECT COUNT(*) FROM document_passages "
                f"WHERE passage_id IN ({placeholders})",
                passage_ids,
            ).fetchone()[0]
            if passage_count != len(passage_ids):
                return False
        if claim_ids:
            placeholders = ",".join("?" for _ in claim_ids)
            claim_count = connection.execute(
                f"SELECT COUNT(*) FROM claims WHERE claim_id IN ({placeholders}) "
                "AND lineage_sealed = 1",
                claim_ids,
            ).fetchone()[0]
            if claim_count != len(claim_ids):
                return False
        return True

    def evidence_references_exist(
        self, passage_ids: Sequence[str], claim_ids: Sequence[str]
    ) -> bool:
        """Verify every relationship lineage reference without accepting a subset."""
        passages = tuple(passage_ids)
        claims = tuple(claim_ids)
        if len(passages) != len(set(passages)) or len(claims) != len(set(claims)):
            return False
        connection = self.connect()
        try:
            return self._references_exist(connection, passages, claims)
        finally:
            connection.close()

    def insert_relationship(self, relationship: object) -> None:
        """Atomically persist an idempotent relationship and exact lineage."""
        from .entities import Relationship

        if not isinstance(relationship, Relationship):
            raise TypeError("relationship must be Relationship")
        values = (
            relationship.relationship_id,
            relationship.source_entity_id,
            relationship.target_entity_id,
            relationship.kind,
            _utc_text(relationship.as_of, "Relationship.as_of"),
            _decimal_text(relationship.confidence),
            relationship.stance,
            relationship.provenance,
            relationship.evidence_ids[0] if relationship.evidence_ids else None,
            (
                relationship.supporting_claim_ids[0]
                if relationship.supporting_claim_ids
                else None
            ),
        )
        with self.transaction() as connection:
            if not self._references_exist(
                connection,
                relationship.evidence_ids,
                relationship.supporting_claim_ids,
            ):
                raise sqlite3.IntegrityError(
                    "relationship evidence references do not exist"
                )
            existing = connection.execute(
                "SELECT relationship_id, source_entity_id, target_entity_id, kind, "
                "as_of, confidence, stance, provenance, primary_passage_id, "
                "primary_claim_id FROM relationships "
                "WHERE relationship_id = ?",
                (relationship.relationship_id,),
            ).fetchone()
            if existing is not None:
                stored_passages = tuple(
                    row[0]
                    for row in connection.execute(
                        "SELECT passage_id FROM relationship_evidence "
                        "WHERE relationship_id = ? ORDER BY passage_id",
                        (relationship.relationship_id,),
                    ).fetchall()
                )
                stored_claims = tuple(
                    row[0]
                    for row in connection.execute(
                        "SELECT claim_id FROM relationship_claim_evidence "
                        "WHERE relationship_id = ? ORDER BY claim_id",
                        (relationship.relationship_id,),
                    ).fetchall()
                )
                if (
                    tuple(existing) == values
                    and stored_passages == relationship.evidence_ids
                    and stored_claims == relationship.supporting_claim_ids
                ):
                    return
                raise sqlite3.IntegrityError(
                    "conflicting relationship assertion already exists"
                )
            created_at = _utc_text(datetime.now(timezone.utc))
            connection.execute(
                "INSERT INTO relationships (relationship_id, source_entity_id, "
                "target_entity_id, kind, as_of, confidence, stance, provenance, "
                "primary_passage_id, primary_claim_id, lineage_sealed, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                (*values, created_at),
            )
            connection.executemany(
                "INSERT INTO relationship_evidence (relationship_id, passage_id) "
                "VALUES (?, ?)",
                (
                    (relationship.relationship_id, passage_id)
                    for passage_id in relationship.evidence_ids
                ),
            )
            connection.executemany(
                "INSERT INTO relationship_claim_evidence (relationship_id, claim_id) "
                "VALUES (?, ?)",
                (
                    (relationship.relationship_id, claim_id)
                    for claim_id in relationship.supporting_claim_ids
                ),
            )
            connection.execute(
                "UPDATE relationships SET lineage_sealed = 1 "
                "WHERE relationship_id = ?",
                (relationship.relationship_id,),
            )

    def list_relationships(self, source_entity_id: str) -> tuple[object, ...]:
        """Return directed relationship assertions without collapsing contradictions."""
        from .entities import Relationship

        connection = self.connect()
        try:
            rows = connection.execute(
                "SELECT relationship_id, source_entity_id, target_entity_id, kind, "
                "as_of, confidence, stance, provenance FROM relationships "
                "WHERE source_entity_id = ? AND lineage_sealed = 1 "
                "ORDER BY stance, relationship_id",
                (source_entity_id,),
            ).fetchall()
            records = []
            for row in rows:
                evidence_ids = tuple(
                    item[0]
                    for item in connection.execute(
                        "SELECT passage_id FROM relationship_evidence "
                        "WHERE relationship_id = ? ORDER BY passage_id",
                        (row[0],),
                    ).fetchall()
                )
                claim_ids = tuple(
                    item[0]
                    for item in connection.execute(
                        "SELECT claim_id FROM relationship_claim_evidence "
                        "WHERE relationship_id = ? ORDER BY claim_id",
                        (row[0],),
                    ).fetchall()
                )
                records.append(
                    Relationship(
                        relationship_id=row[0],
                        source_entity_id=row[1],
                        target_entity_id=row[2],
                        kind=row[3],
                        as_of=_parse_utc(row[4]),
                        confidence=Decimal(row[5]),
                        stance=row[6],
                        evidence_ids=evidence_ids,
                        supporting_claim_ids=claim_ids,
                        provenance=row[7],
                    )
                )
        finally:
            connection.close()
        return tuple(records)

    def insert_portfolio_snapshot(
        self,
        snapshot: PortfolioSnapshot,
        positions: Sequence[Position],
        account_ref: str,
    ) -> None:
        """Atomically insert an idempotent snapshot and all of its positions."""
        position_records = tuple(positions)
        if not isinstance(account_ref, str) or re.fullmatch(
            r"acct_[0-9a-f]{24}", account_ref
        ) is None:
            raise ValueError("account_ref must be a hashed local reference")
        if any(position.snapshot_id != snapshot.snapshot_id for position in position_records):
            raise sqlite3.IntegrityError("position snapshot_id does not match snapshot")
        if any(position.security_id is None for position in position_records):
            raise sqlite3.IntegrityError("position security_id is required")
        position_ids = [position.position_id for position in position_records]
        if len(position_ids) != len(set(position_ids)):
            raise sqlite3.IntegrityError("duplicate position IDs are not allowed")

        snapshot_values = (
            snapshot.snapshot_id,
            _utc_text(snapshot.as_of, "PortfolioSnapshot.as_of"),
            snapshot.base_currency,
            _decimal_text(snapshot.nav),
            _decimal_text(snapshot.cash),
            int(snapshot.is_stale) if snapshot.is_stale is not None else None,
            account_ref,
        )
        security_records = tuple(
            sorted(
                (
                    position.security_id,
                    position.symbol,
                    position.asset_class or "UNKNOWN",
                    None,
                    position.currency,
                    _canonical_json(
                        {
                            key: value
                            for key, value in (
                                ("conid", position.conid),
                                ("isin", position.isin),
                                ("cusip", position.cusip),
                                ("figi", position.figi),
                                (
                                    "external_security_id",
                                    position.external_security_id,
                                ),
                                ("security_id_type", position.security_id_type),
                            )
                            if value is not None
                        }
                    ),
                )
                for position in position_records
            )
        )
        expected_positions = tuple(
            sorted(
                (
                    position.position_id,
                    position.snapshot_id,
                    position.symbol,
                    _decimal_text(position.quantity),
                    _decimal_text(position.market_value),
                    position.currency,
                    (
                        _decimal_text(position.cost_basis)
                        if position.cost_basis is not None
                        else None
                    ),
                    position.security_id,
                )
                for position in position_records
            )
        )

        with self.transaction() as connection:
            created_at = _utc_text(datetime.now(timezone.utc))
            for security in security_records:
                existing_security = connection.execute(
                    "SELECT security_id, symbol, security_type, exchange, currency, "
                    "identifiers_json FROM securities WHERE security_id = ?",
                    (security[0],),
                ).fetchone()
                if existing_security is None:
                    connection.execute(
                        "INSERT INTO securities (security_id, entity_id, symbol, "
                        "security_type, exchange, currency, identifiers_json, "
                        "metadata_json, created_at) VALUES (?, NULL, ?, ?, ?, ?, ?, "
                        "NULL, ?)",
                        (*security, created_at),
                    )
                elif tuple(existing_security) != security:
                    raise sqlite3.IntegrityError(
                        "conflicting security identity already exists"
                    )

            existing = connection.execute(
                "SELECT snapshot_id, as_of, base_currency, nav, cash, is_stale, "
                "account_ref FROM portfolio_snapshots WHERE snapshot_id = ?",
                (snapshot.snapshot_id,),
            ).fetchone()
            if existing is not None:
                stored_positions = tuple(
                    connection.execute(
                        "SELECT position_id, snapshot_id, symbol, quantity, "
                        "market_value, currency, cost_basis, security_id FROM positions "
                        "WHERE snapshot_id = ? ORDER BY position_id",
                        (snapshot.snapshot_id,),
                    ).fetchall()
                )
                if tuple(existing) == snapshot_values and stored_positions == expected_positions:
                    return
                raise sqlite3.IntegrityError("conflicting snapshot already exists")

            connection.execute(
                "INSERT INTO portfolio_snapshots ("
                "snapshot_id, as_of, base_currency, nav, cash, is_stale, "
                "account_ref, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (*snapshot_values, created_at),
            )
            connection.executemany(
                "INSERT INTO positions (position_id, snapshot_id, symbol, quantity, "
                "market_value, currency, cost_basis, security_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                expected_positions,
            )

    def insert_source_document(self, document: SourceDocument) -> None:
        """Insert source metadata needed to seed evidence integration tests."""
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO source_documents ("
                "document_id, source_type, canonical_url, publisher, published_at, "
                "retrieved_at, content_hash, raw_content_path, extraction_status, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    document.document_id,
                    document.source_type,
                    document.canonical_url,
                    document.publisher,
                    _utc_text(document.published_at),
                    _utc_text(document.retrieved_at),
                    document.content_hash,
                    document.raw_content_path,
                    document.extraction_status,
                    _utc_text(datetime.now(timezone.utc)),
                ),
            )

    def insert_document_with_passages(
        self,
        document: SourceDocument,
        passages: Sequence[DocumentPassageRecord],
    ) -> None:
        """Atomically persist an idempotent document and all exact passages."""
        self.insert_documents_with_passages(((document, passages),))

    def insert_documents_with_passages(
        self,
        records: Sequence[tuple[SourceDocument, Sequence[DocumentPassageRecord]]],
    ) -> tuple[tuple[SourceDocument, tuple[DocumentPassageRecord, ...]], ...]:
        """Persist a complete evidence batch in one SQLite transaction.

        Content-addressed cache files may already exist when this transaction
        rolls back. They are immutable and reusable by a later ingestion.
        """
        prepared = []
        for document, passages in records:
            passage_records = tuple(passages)
            if any(item.document_id != document.document_id for item in passage_records):
                raise sqlite3.IntegrityError(
                    "passage document_id does not match document"
                )
            if tuple(item.ordinal for item in passage_records) != tuple(
                range(len(passage_records))
            ):
                raise sqlite3.IntegrityError("passage ordinals must be contiguous")
            document_values = (
                document.document_id,
                document.source_type,
                document.canonical_url,
                document.publisher,
                _utc_text(document.published_at, "SourceDocument.published_at"),
                _utc_text(document.retrieved_at, "SourceDocument.retrieved_at"),
                document.content_hash,
                document.raw_content_path,
                document.extraction_status,
            )
            passage_values = tuple(
                (
                    passage.passage_id,
                    passage.document_id,
                    passage.ordinal,
                    passage.text,
                    passage.content_hash,
                    _canonical_json(
                        {
                            "end_offset": passage.end_offset,
                            "start_offset": passage.start_offset,
                        }
                    ),
                )
                for passage in passage_records
            )
            prepared.append(
                (document, passage_records, document_values, passage_values)
            )

        stored_records = []
        with self.transaction() as connection:
            for document, passage_records, document_values, passage_values in prepared:
                existing = connection.execute(
                    "SELECT document_id, source_type, canonical_url, publisher, "
                    "published_at, retrieved_at, content_hash, raw_content_path, "
                    "extraction_status FROM source_documents WHERE content_hash = ?",
                    (document.content_hash,),
                ).fetchone()
                if existing is not None:
                    stored_rows = tuple(
                        connection.execute(
                            "SELECT passage_id, document_id, ordinal, text, content_hash, "
                            "locator_json FROM document_passages WHERE document_id = ? "
                            "ORDER BY ordinal",
                            (existing[0],),
                        ).fetchall()
                    )
                    if existing[0] != document.document_id or stored_rows != passage_values:
                        raise sqlite3.IntegrityError(
                            "conflicting document content already exists"
                        )
                    stored_document = SourceDocument(
                        document_id=existing[0],
                        source_type=existing[1],
                        canonical_url=existing[2],
                        publisher=existing[3],
                        published_at=_parse_utc(existing[4]),
                        retrieved_at=_parse_utc(existing[5]),
                        content_hash=existing[6],
                        raw_content_path=existing[7],
                        extraction_status=existing[8],
                    )
                    stored_passages = tuple(
                        DocumentPassageRecord(
                            passage_id=row[0],
                            document_id=row[1],
                            ordinal=row[2],
                            text=row[3],
                            content_hash=row[4],
                            start_offset=json.loads(row[5])["start_offset"],
                            end_offset=json.loads(row[5])["end_offset"],
                        )
                        for row in stored_rows
                    )
                    stored_records.append((stored_document, stored_passages))
                    continue

                created_at = _utc_text(datetime.now(timezone.utc))
                connection.execute(
                    "INSERT INTO source_documents ("
                    "document_id, source_type, canonical_url, publisher, published_at, "
                    "retrieved_at, content_hash, raw_content_path, extraction_status, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (*document_values, created_at),
                )
                connection.executemany(
                    "INSERT INTO document_passages ("
                    "passage_id, document_id, ordinal, text, content_hash, locator_json, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ((*values, created_at) for values in passage_values),
                )
                stored_records.append((document, passage_records))
        return tuple(stored_records)

    def get_document_by_content_hash(
        self, content_hash: str
    ) -> SourceDocument | None:
        """Load the canonical document associated with a content hash."""
        connection = self.connect()
        try:
            row = connection.execute(
                "SELECT document_id, source_type, canonical_url, publisher, "
                "published_at, retrieved_at, content_hash, raw_content_path, "
                "extraction_status FROM source_documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return SourceDocument(
            document_id=row[0],
            source_type=row[1],
            canonical_url=row[2],
            publisher=row[3],
            published_at=_parse_utc(row[4]),
            retrieved_at=_parse_utc(row[5]),
            content_hash=row[6],
            raw_content_path=row[7],
            extraction_status=row[8],
        )

    def list_documents_as_of(
        self, as_of: datetime
    ) -> tuple[SourceDocument, ...]:
        """Return only documents published and retrieved by an exact UTC cutoff.

        Historical replay treats malformed or timezone-unknown persisted dates as
        an integrity failure.  It never silently drops an unclassifiable source.
        """
        if (
            not isinstance(as_of, datetime)
            or as_of.tzinfo is None
            or as_of.utcoffset() != timezone.utc.utcoffset(as_of)
        ):
            raise ValueError("document as_of must be an aware UTC datetime")
        cutoff = as_of.astimezone(timezone.utc)
        cutoff_text = _utc_text(cutoff, "document as_of")
        connection = self.connect()
        try:
            persisted_timestamps = connection.execute(
                "SELECT published_at, retrieved_at FROM source_documents"
            ).fetchall()
            for published_text, retrieved_text in persisted_timestamps:
                try:
                    published_at = datetime.fromisoformat(
                        published_text.replace("Z", "+00:00")
                    )
                    retrieved_at = datetime.fromisoformat(
                        retrieved_text.replace("Z", "+00:00")
                    )
                    canonical = (
                        _utc_text(published_at),
                        _utc_text(retrieved_at),
                    )
                except (AttributeError, TypeError, ValueError):
                    raise ValueError(
                        "stored document publication date is invalid or unknown; "
                        "timestamps must use canonical UTC"
                    ) from None
                if canonical != (published_text, retrieved_text):
                    raise ValueError(
                        "stored document publication date is invalid or unknown; "
                        "timestamps must use canonical UTC"
                    )
            rows = connection.execute(
                "SELECT document_id, source_type, canonical_url, publisher, "
                "published_at, retrieved_at, content_hash, raw_content_path, "
                "extraction_status FROM source_documents "
                "WHERE published_at <= ? AND retrieved_at <= ? "
                "ORDER BY published_at, document_id",
                (cutoff_text, cutoff_text),
            ).fetchall()
        finally:
            connection.close()

        documents: list[SourceDocument] = []
        for row in rows:
            try:
                published_at = _parse_utc(row[4])
                retrieved_at = _parse_utc(row[5])
            except (AttributeError, TypeError, ValueError):
                raise ValueError(
                    "stored document publication date is invalid or unknown"
                ) from None
            if published_at > cutoff or retrieved_at > cutoff:
                raise ValueError("document replay query exceeded its UTC cutoff")
            documents.append(
                SourceDocument(
                    document_id=row[0],
                    source_type=row[1],
                    canonical_url=row[2],
                    publisher=row[3],
                    published_at=published_at,
                    retrieved_at=retrieved_at,
                    content_hash=row[6],
                    raw_content_path=row[7],
                    extraction_status=row[8],
                )
            )
        return tuple(documents)

    def list_document_passages(
        self, document_id: str
    ) -> tuple[DocumentPassageRecord, ...]:
        """Load a document's passages in deterministic source order."""
        connection = self.connect()
        try:
            rows = connection.execute(
                "SELECT passage_id, document_id, ordinal, text, content_hash, "
                "locator_json FROM document_passages WHERE document_id = ? "
                "ORDER BY ordinal",
                (document_id,),
            ).fetchall()
        finally:
            connection.close()
        records: list[DocumentPassageRecord] = []
        for row in rows:
            locator = json.loads(row[5])
            records.append(
                DocumentPassageRecord(
                    passage_id=row[0],
                    document_id=row[1],
                    ordinal=row[2],
                    text=row[3],
                    content_hash=row[4],
                    start_offset=locator["start_offset"],
                    end_offset=locator["end_offset"],
                )
            )
        return tuple(records)

    def insert_document_passage(
        self,
        *,
        passage_id: str,
        document_id: str,
        ordinal: int,
        text: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Insert one document passage for real evidence-link tests."""
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO document_passages ("
                "passage_id, document_id, ordinal, text, content_hash, "
                "metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    passage_id,
                    document_id,
                    ordinal,
                    text,
                    content_hash,
                    _canonical_json(metadata) if metadata is not None else None,
                    _utc_text(datetime.now(timezone.utc)),
                ),
            )

    def insert_claim(
        self,
        claim: EvidenceClaim,
        evidence_ids: Sequence[str],
        stance: str = "supports",
    ) -> None:
        """Atomically insert a typed claim and its required passage evidence."""
        passage_ids = tuple(evidence_ids)
        self.insert_claim_with_lineage(
            claim,
            passage_links=tuple((passage_id, stance) for passage_id in passage_ids),
            supporting_claim_ids=(),
        )

    def insert_claim_with_lineage(
        self,
        claim: EvidenceClaim,
        *,
        passage_links: Sequence[tuple[str, str]],
        supporting_claim_ids: Sequence[str],
    ) -> None:
        """Atomically insert a claim and all passage/claim dependencies."""
        links = tuple(passage_links)
        dependency_ids = tuple(supporting_claim_ids)
        supporting_passages = tuple(
            passage_id for passage_id, stance in links if stance == "supports"
        )
        if claim.kind is ClaimKind.INFERENCE:
            if not supporting_passages and not dependency_ids:
                raise sqlite3.IntegrityError(
                    "inference claims require a supporting claim or passage"
                )
        elif not supporting_passages:
            raise sqlite3.IntegrityError(
                "fact, guidance, and estimate claims require passage evidence"
            )
        if claim.claim_id in dependency_ids:
            raise sqlite3.IntegrityError("claims cannot depend on themselves")
        if len(links) != len({passage_id for passage_id, _ in links}):
            raise sqlite3.IntegrityError("duplicate passage links are not allowed")
        if len(dependency_ids) != len(set(dependency_ids)):
            raise sqlite3.IntegrityError("duplicate claim dependencies are not allowed")
        confidence = _decimal_text(claim.confidence)
        if not Decimal("0") <= claim.confidence <= Decimal("1"):
            raise ValueError("claim confidence must be between 0 and 1")

        with self.transaction() as connection:
            primary_passage_id = (
                supporting_passages[0] if supporting_passages else None
            )
            primary_supporting_claim_id = (
                dependency_ids[0] if dependency_ids else None
            )
            connection.execute(
                "INSERT INTO claims ("
                "claim_id, entity_id, kind, text, as_of, confidence, status, "
                "primary_passage_id, primary_supporting_claim_id, lineage_sealed, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    claim.claim_id,
                    claim.entity_id,
                    claim.kind.value,
                    claim.text,
                    _utc_text(claim.as_of, "EvidenceClaim.as_of"),
                    confidence,
                    claim.status,
                    primary_passage_id,
                    primary_supporting_claim_id,
                    0,
                    _utc_text(datetime.now(timezone.utc)),
                ),
            )
            connection.executemany(
                "INSERT INTO claim_evidence (claim_id, passage_id, stance) "
                "VALUES (?, ?, ?)",
                (
                    (claim.claim_id, passage_id, link_stance)
                    for passage_id, link_stance in links
                ),
            )
            connection.executemany(
                "INSERT INTO claim_dependencies (claim_id, supporting_claim_id) "
                "VALUES (?, ?)",
                (
                    (claim.claim_id, supporting_claim_id)
                    for supporting_claim_id in dependency_ids
                ),
            )
            connection.execute(
                "UPDATE claims SET lineage_sealed = 1 WHERE claim_id = ?",
                (claim.claim_id,),
            )

    def get_claim(self, claim_id: str) -> EvidenceClaim | None:
        """Load one typed claim, preserving exact Decimal representation."""
        connection = self.connect()
        try:
            row = connection.execute(
                "SELECT claim_id, entity_id, kind, text, as_of, confidence, status "
                "FROM claims WHERE claim_id = ? AND lineage_sealed = 1",
                (claim_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return EvidenceClaim(
            claim_id=row[0],
            entity_id=row[1],
            kind=ClaimKind(row[2]),
            text=row[3],
            as_of=_parse_utc(row[4]),
            confidence=Decimal(row[5]),
            status=row[6],
        )

    def list_claim_lineage(self, claim_id: str) -> tuple[ClaimLineageRecord, ...]:
        """Return exact passage lineage for one claim without rewriting evidence."""
        connection = self.connect()
        try:
            rows = connection.execute(
                "SELECT ce.claim_id, ce.stance, p.passage_id, p.document_id, "
                "p.ordinal, p.text, p.content_hash, p.locator_json "
                "FROM claim_evidence AS ce "
                "JOIN document_passages AS p ON p.passage_id = ce.passage_id "
                "JOIN claims AS c ON c.claim_id = ce.claim_id "
                "WHERE ce.claim_id = ? AND c.lineage_sealed = 1 "
                "ORDER BY p.document_id, p.ordinal, ce.stance",
                (claim_id,),
            ).fetchall()
        finally:
            connection.close()
        lineage: list[ClaimLineageRecord] = []
        for row in rows:
            locator = json.loads(row[7]) if row[7] is not None else {}
            lineage.append(
                ClaimLineageRecord(
                    claim_id=row[0],
                    stance=row[1],
                    passage=DocumentPassageRecord(
                        passage_id=row[2],
                        document_id=row[3],
                        ordinal=row[4],
                        text=row[5],
                        content_hash=row[6],
                        start_offset=locator.get("start_offset", 0),
                        end_offset=locator.get("end_offset", len(row[5])),
                    ),
                )
            )
        return tuple(lineage)

    def list_supporting_claim_ids(self, claim_id: str) -> tuple[str, ...]:
        """List immutable claim-to-claim dependencies deterministically."""
        connection = self.connect()
        try:
            rows = connection.execute(
                "SELECT dependency.supporting_claim_id FROM claim_dependencies "
                "AS dependency JOIN claims AS c ON c.claim_id = dependency.claim_id "
                "JOIN claims AS supporting_claim ON supporting_claim.claim_id = "
                "dependency.supporting_claim_id "
                "WHERE dependency.claim_id = ? AND c.lineage_sealed = 1 "
                "AND supporting_claim.lineage_sealed = 1 "
                "ORDER BY dependency.supporting_claim_id",
                (claim_id,),
            ).fetchall()
        finally:
            connection.close()
        return tuple(row[0] for row in rows)

    def update_claim_status(self, claim_id: str, status: str) -> None:
        """Apply an intentional status transition without rewriting claim content."""
        allowed_statuses = {"active", "contradicted", "superseded"}
        if status not in allowed_statuses:
            raise ValueError(
                "claim status must be active, contradicted, or superseded"
            )
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE claims SET status = ? "
                "WHERE claim_id = ? AND lineage_sealed = 1",
                (status, claim_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown claim_id: {claim_id}")

    def load_agent_evidence(
        self,
        passage_ids: Sequence[str],
        claim_ids: Sequence[str],
        *,
        as_of: datetime,
    ) -> AgentEvidencePacket | None:
        """Load a complete safe evidence packet, or ``None`` for any bad reference."""
        cutoff = _parse_utc(_utc_text(as_of, "agent evidence as_of"))
        requested_passages = tuple(sorted(set(passage_ids)))
        requested_claims = tuple(sorted(set(claim_ids)))
        if (
            not requested_passages
            or len(requested_passages) != len(tuple(passage_ids))
            or len(requested_claims) != len(tuple(claim_ids))
        ):
            return None
        connection = self.connect()
        try:
            connection.execute("BEGIN")
            placeholders = ",".join("?" for _ in requested_passages)
            passage_rows = connection.execute(
                "SELECT p.passage_id, p.document_id, p.ordinal, p.text, "
                "p.content_hash, p.locator_json, d.published_at, d.retrieved_at, "
                "d.extraction_status "
                "FROM document_passages AS p JOIN source_documents AS d "
                "ON d.document_id = p.document_id "
                f"WHERE p.passage_id IN ({placeholders}) ORDER BY p.passage_id",
                requested_passages,
            ).fetchall()
            if len(passage_rows) != len(requested_passages) or any(
                row[8] not in {"complete", "extracted", "normalized", "success"}
                for row in passage_rows
            ):
                return None
            try:
                if any(
                    _parse_utc(row[6]) > cutoff or _parse_utc(row[7]) > cutoff
                    for row in passage_rows
                ):
                    return None
            except (TypeError, ValueError):
                return None
            claim_rows = ()
            if requested_claims:
                placeholders = ",".join("?" for _ in requested_claims)
                closure_rows = connection.execute(
                    "WITH RECURSIVE claim_closure(claim_id) AS ("
                    "SELECT claim_id FROM claims "
                    f"WHERE claim_id IN ({placeholders}) "
                    "UNION "
                    "SELECT dependency.supporting_claim_id "
                    "FROM claim_dependencies AS dependency "
                    "JOIN claim_closure AS parent "
                    "ON parent.claim_id = dependency.claim_id "
                    "LIMIT 257"
                    ") SELECT claim_id FROM claim_closure ORDER BY claim_id",
                    requested_claims,
                ).fetchall()
                closure_ids = tuple(row[0] for row in closure_rows)
                if (
                    len(closure_ids) > 256
                    or not set(requested_claims) <= set(closure_ids)
                ):
                    return None
                placeholders = ",".join("?" for _ in closure_ids)
                claim_rows = connection.execute(
                    "SELECT claim_id, entity_id, kind, text, as_of, confidence, status "
                    "FROM claims "
                    f"WHERE claim_id IN ({placeholders}) AND lineage_sealed = 1 "
                    "AND status = 'active' ORDER BY claim_id",
                    closure_ids,
                ).fetchall()
                if len(claim_rows) != len(closure_ids):
                    return None
                try:
                    if any(_parse_utc(row[4]) > cutoff for row in claim_rows):
                        return None
                except (TypeError, ValueError):
                    return None
                lineage_rows = connection.execute(
                    "SELECT DISTINCT passage_id FROM claim_evidence "
                    f"WHERE claim_id IN ({placeholders}) ORDER BY passage_id",
                    closure_ids,
                ).fetchall()
                required_passages = {row[0] for row in lineage_rows}
                if not required_passages <= set(requested_passages):
                    return None
            connection.commit()
        finally:
            if connection.in_transaction:
                connection.rollback()
            connection.close()
        passages = []
        for row in passage_rows:
            locator = json.loads(row[5]) if row[5] is not None else {}
            passages.append(DocumentPassageRecord(
                passage_id=row[0], document_id=row[1], ordinal=row[2], text=row[3],
                content_hash=row[4], start_offset=locator.get("start_offset", 0),
                end_offset=locator.get("end_offset", len(row[3])),
                published_at=_parse_utc(row[6]), retrieved_at=_parse_utc(row[7]),
            ))
        claims = tuple(EvidenceClaim(
            claim_id=row[0], entity_id=row[1], kind=ClaimKind(row[2]), text=row[3],
            as_of=_parse_utc(row[4]), confidence=Decimal(row[5]), status=row[6],
        ) for row in claim_rows)
        return AgentEvidencePacket(tuple(passages), claims)

    def claim_agent_execution(
        self,
        *,
        workflow_run_id: str,
        task_id: str,
        role: AgentRole,
        schema_version: str,
        input_hash: str,
        prompt_hash: str,
        evidence_hash: str | None,
        started_at: datetime,
    ) -> AgentExecutionClaim:
        """Atomically acquire a live lease or return an exact terminal replay."""
        for name, value in (
            ("workflow_run_id", workflow_run_id),
            ("task_id", task_id),
            ("schema_version", schema_version),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 256
                or any(character.isspace() for character in value)
            ):
                raise ValueError(f"{name} is invalid")
        if not isinstance(role, AgentRole):
            raise TypeError("role must be AgentRole")
        for digest in (input_hash, prompt_hash):
            if re.fullmatch(r"[0-9a-f]{64}", digest or "") is None:
                raise ValueError("agent execution hashes must be SHA-256 digests")
        if evidence_hash is not None and re.fullmatch(
            r"[0-9a-f]{64}", evidence_hash
        ) is None:
            raise ValueError("agent execution evidence hash must be SHA-256")
        started_text = _utc_text(started_at, "agent execution started_at")
        lease_token = secrets.token_hex(32)
        identity = _canonical_json({
            "role": role.value,
            "schema_version": schema_version,
            "task_id": task_id,
            "workflow_run_id": workflow_run_id,
        })
        attempt_id = "agent_attempt_" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()
        task_instance_id = "task_instance_" + attempt_id.removeprefix(
            "agent_attempt_"
        )
        task_scope = _canonical_json({
            "contract_version": schema_version,
            "input_hash": input_hash,
            "role": role.value,
        })
        with self.transaction() as connection:
            lease_now_text = _utc_text(self.clock(), "research store clock")
            lease_expires_text = _utc_text(
                _parse_utc(lease_now_text) + _AGENT_EXECUTION_LEASE,
                "agent execution lease expiry",
            )
            existing_attempt = connection.execute(
                "SELECT state, input_hash, prompt_hash, evidence_hash, "
                "schema_version, output_json, output_hash, lease_token, "
                "lease_expires_at, task_instance_id "
                "FROM agent_executions WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            existing_instance = connection.execute(
                "SELECT workflow_run_id, task_id, role, schema_version, "
                "input_hash, prompt_hash, state, owns_research_task_state "
                "FROM workflow_task_instances WHERE task_instance_id = ?",
                (task_instance_id,),
            ).fetchone()
            existing_task = connection.execute(
                "SELECT task_kind, scope_json, state FROM research_tasks "
                "WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if existing_attempt is not None:
                if (
                    existing_attempt[1] != input_hash
                    or existing_attempt[2] != prompt_hash
                    or (
                        evidence_hash is not None
                        and existing_attempt[3] != evidence_hash
                    )
                    or existing_attempt[4] != schema_version
                    or existing_attempt[9] != task_instance_id
                    or existing_instance is None
                    or existing_instance[:6] != (
                        workflow_run_id, task_id, role.value, schema_version,
                        input_hash, prompt_hash,
                    )
                    or existing_task is None
                    or existing_task[0] != role.value
                    or existing_task[1] != task_scope
                ):
                    raise AgentExecutionConflict(
                        "agent execution identity or task scope conflicts"
                    )
                if existing_attempt[0] == "succeeded":
                    evidence_digest, output_json, output_hash = (
                        existing_attempt[3], existing_attempt[5],
                        existing_attempt[6],
                    )
                    if (
                        existing_instance[6] != "completed"
                        or re.fullmatch(
                            r"[0-9a-f]{64}", evidence_digest or ""
                        ) is None
                        or not isinstance(output_json, str)
                        or re.fullmatch(r"[0-9a-f]{64}", output_hash or "") is None
                        or hashlib.sha256(output_json.encode("utf-8")).hexdigest()
                        != output_hash
                    ):
                        raise AgentExecutionConflict(
                            "terminal agent replay is unavailable"
                        )
                    return AgentExecutionClaim(
                        attempt_id=attempt_id, lease_token=None,
                        replay_output_json=output_json,
                        replay_output_hash=output_hash,
                        replay_evidence_hash=evidence_digest,
                    )
                if (
                    existing_attempt[0] != "running"
                    or existing_instance[6] != "running"
                ):
                    raise AgentExecutionConflict(
                        "agent execution is already terminal"
                    )
                lease_expiry = existing_attempt[8]
                if (
                    isinstance(lease_expiry, str)
                    and _parse_utc(lease_expiry) > _parse_utc(lease_now_text)
                ):
                    raise AgentExecutionConflict(
                        "agent execution is already claimed by a live lease"
                    )
                cursor = connection.execute(
                    "UPDATE agent_executions SET lease_token = ?, "
                    "lease_expires_at = ? WHERE attempt_id = ? "
                    "AND state = 'running' AND (lease_expires_at IS NULL "
                    "OR lease_expires_at <= ?) AND lease_token IS ? "
                    "AND lease_expires_at IS ?",
                    (
                        lease_token, lease_expires_text, attempt_id,
                        lease_now_text, existing_attempt[7], existing_attempt[8],
                    ),
                )
                if cursor.rowcount != 1:
                    raise AgentExecutionConflict(
                        "agent execution lease could not be reclaimed"
                    )
                return AgentExecutionClaim(attempt_id, lease_token)
            if existing_task is None:
                connection.execute(
                    "INSERT INTO research_tasks (task_id, task_kind, scope_json, "
                    "state, priority, created_at, started_at) "
                    "VALUES (?, ?, ?, 'running', 0, ?, ?)",
                    (task_id, role.value, task_scope, started_text, started_text),
                )
                owns_research_task_state = 1
            else:
                if (
                    existing_task[0] != role.value
                    or existing_task[1] != task_scope
                ):
                    raise AgentExecutionConflict(
                        "research task role or scope conflicts"
                    )
                owner_exists = connection.execute(
                    "SELECT 1 FROM workflow_task_instances "
                    "WHERE task_id = ? AND owns_research_task_state = 1",
                    (task_id,),
                ).fetchone()
                owns_research_task_state = int(
                    owner_exists is None
                    and existing_task[2] in {"pending", "running"}
                )
                if owns_research_task_state and existing_task[2] == "pending":
                    cursor = connection.execute(
                        "UPDATE research_tasks SET state = 'running', "
                        "started_at = ? WHERE task_id = ? AND state = 'pending'",
                        (started_text, task_id),
                    )
                    if cursor.rowcount != 1:
                        raise AgentExecutionConflict(
                            "research task could not be claimed"
                        )
            connection.execute(
                "INSERT INTO workflow_task_instances (task_instance_id, "
                "workflow_run_id, task_id, role, schema_version, input_hash, "
                "prompt_hash, state, started_at, owns_research_task_state) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)",
                (
                    task_instance_id, workflow_run_id, task_id, role.value,
                    schema_version, input_hash, prompt_hash, started_text,
                    owns_research_task_state,
                ),
            )
            connection.execute(
                "INSERT INTO agent_executions (attempt_id, workflow_run_id, "
                "task_id, task_instance_id, role, schema_version, input_hash, "
                "prompt_hash, evidence_hash, state, started_at, lease_token, "
                "lease_expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)",
                (
                    attempt_id, workflow_run_id, task_id, task_instance_id,
                    role.value, schema_version, input_hash, prompt_hash,
                    evidence_hash, started_text, lease_token, lease_expires_text,
                ),
            )
        return AgentExecutionClaim(attempt_id, lease_token)

    def set_agent_execution_evidence_hash(
        self, attempt_id: str, evidence_hash: str, *, lease_token: str
    ) -> None:
        """Attach the canonical evidence hash once before the provider call."""
        _validate_lease_token(lease_token)
        if re.fullmatch(r"[0-9a-f]{64}", evidence_hash or "") is None:
            raise ValueError("agent execution evidence hash must be SHA-256")
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT state, evidence_hash, lease_token FROM agent_executions "
                "WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if existing == ("running", evidence_hash, lease_token):
                return
            cursor = connection.execute(
                "UPDATE agent_executions SET evidence_hash = ? "
                "WHERE attempt_id = ? AND state = 'running' "
                "AND evidence_hash IS NULL AND lease_token = ?",
                (evidence_hash, attempt_id, lease_token),
            )
            if cursor.rowcount != 1:
                raise AgentExecutionConflict(
                    "agent execution evidence hash or lease conflicts"
                )

    def record_provider_attempt(
        self,
        attempt_id: str,
        attempt: ProviderAttemptAudit,
        *,
        lease_token: str,
    ) -> None:
        """Append one immutable provider call while its execution is running."""
        _validate_lease_token(lease_token)
        if not isinstance(attempt, ProviderAttemptAudit):
            raise TypeError("attempt must be ProviderAttemptAudit")
        if attempt.status not in {"succeeded", "failed"}:
            raise ValueError("provider attempt status is invalid")
        for name, value in (("provider", attempt.provider), ("model", attempt.model)):
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 256
                or any(ord(character) < 32 or ord(character) == 127 for character in value)
            ):
                raise ValueError(f"provider attempt {name} is invalid")
        if not isinstance(attempt.inference_mode, InferenceMode):
            raise TypeError("provider attempt inference_mode must be InferenceMode")
        if type(attempt.usage_known) is not bool:
            raise TypeError("provider attempt usage_known must be bool")
        for name, value in (("latency_ms", attempt.latency_ms),):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        usage = (
            attempt.input_tokens, attempt.output_tokens, attempt.reasoning_tokens
        )
        if attempt.usage_known:
            if any(type(value) is not int or value < 0 for value in usage):
                raise ValueError("known provider usage must be non-negative integers")
        elif any(value is not None for value in usage):
            raise ValueError("unknown provider usage must use null token counts")
        if attempt.status == "failed" and re.fullmatch(
            r"[a-z0-9_.-]+", attempt.failure_code or ""
        ) is None:
            raise ValueError("failed provider attempts require a safe failure code")
        if attempt.status == "succeeded" and attempt.failure_code is not None:
            raise ValueError("successful provider attempts cannot have a failure code")
        if attempt.response_hash is not None and re.fullmatch(
            r"[0-9a-f]{64}", attempt.response_hash
        ) is None:
            raise ValueError("provider response hash must be SHA-256")
        if attempt.fallback_reason is not None and re.fullmatch(
            r"[a-z0-9_.-]+", attempt.fallback_reason
        ) is None:
            raise ValueError("provider fallback reason must be a safe code")
        reservation = (
            attempt.reservation_id,
            attempt.reservation_state,
            attempt.reserved_cost_usd,
        )
        if any(value is not None for value in reservation):
            if any(value is None for value in reservation):
                raise ValueError("provider reservation metadata must be complete")
            if re.fullmatch(r"[A-Za-z0-9_.:-]+", attempt.reservation_id or "") is None:
                raise ValueError("provider reservation id is invalid")
            if attempt.reservation_state not in {
                "reserved", "usage_unknown", "reconciled", "released",
                "consumed", "expired",
            }:
                raise ValueError("provider reservation state is invalid")
            if (
                not isinstance(attempt.reserved_cost_usd, Decimal)
                or not attempt.reserved_cost_usd.is_finite()
                or attempt.reserved_cost_usd < 0
            ):
                raise ValueError("provider reserved cost is invalid")
        recorded_text = _utc_text(attempt.recorded_at, "provider attempt recorded_at")
        with self.transaction() as connection:
            now_text = _utc_text(self.clock(), "research store clock")
            execution = connection.execute(
                "SELECT state, lease_token FROM agent_executions WHERE "
                "attempt_id = ? AND state = 'running' AND lease_token = ? "
                "AND lease_expires_at > ?",
                (attempt_id, lease_token, now_text),
            ).fetchone()
            if execution != ("running", lease_token):
                raise AgentExecutionConflict(
                    "agent execution lease is not current"
                )
            ordinal = connection.execute(
                "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM provider_attempts "
                "WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()[0]
            provider_attempt_id = "provider_attempt_" + hashlib.sha256(
                f"{attempt_id}:{ordinal}".encode("ascii")
            ).hexdigest()
            connection.execute(
                "INSERT INTO provider_attempts (provider_attempt_id, attempt_id, "
                "ordinal, status, provider, model, latency_ms, input_tokens, "
                "output_tokens, reasoning_tokens, usage_known, inference_mode, "
                "fallback_reason, response_hash, failure_code, reservation_id, "
                "reservation_state, reserved_cost_usd, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    provider_attempt_id, attempt_id, ordinal, attempt.status,
                    attempt.provider, attempt.model, attempt.latency_ms,
                    attempt.input_tokens, attempt.output_tokens,
                    attempt.reasoning_tokens, int(attempt.usage_known),
                    attempt.inference_mode.value, attempt.fallback_reason,
                    attempt.response_hash, attempt.failure_code,
                    attempt.reservation_id, attempt.reservation_state,
                    (
                        None if attempt.reserved_cost_usd is None
                        else _decimal_text(attempt.reserved_cost_usd)
                    ),
                    recorded_text,
                ),
            )
            aggregate = self._provider_attempt_aggregate(connection, attempt_id)
            cursor = connection.execute(
                "UPDATE agent_executions SET provider_attempt_count = ?, "
                "usage_known = ?, input_tokens = ?, output_tokens = ?, "
                "reasoning_tokens = ?, reservation_state = ?, "
                "reserved_cost_usd = ?, provider = ?, model = ?, "
                "inference_mode = ?, fallback_reason = ? "
                "WHERE attempt_id = ? AND state = 'running' "
                "AND lease_token = ? AND lease_expires_at > ?",
                (*aggregate, attempt_id, lease_token, now_text),
            )
            if cursor.rowcount != 1:
                raise AgentExecutionConflict(
                    "agent execution lease could not aggregate provider usage"
                )

    @staticmethod
    def _provider_attempt_aggregate(
        connection: sqlite3.Connection, attempt_id: str
    ) -> tuple[
        int, int, int | None, int | None, int | None, str | None,
        str | None, str | None, str | None, str | None, str | None,
    ]:
        rows = connection.execute(
            "SELECT usage_known, input_tokens, output_tokens, reasoning_tokens, "
            "reservation_state, reserved_cost_usd, provider, model, "
            "inference_mode, fallback_reason FROM provider_attempts "
            "WHERE attempt_id = ? ORDER BY ordinal",
            (attempt_id,),
        ).fetchall()
        if not rows:
            return (0, 1, 0, 0, 0, None, None, None, None, None, None)
        usage_known = int(all(row[0] == 1 for row in rows))
        token_sums: tuple[int | None, int | None, int | None]
        if usage_known:
            token_sums = tuple(
                sum(row[index] for row in rows) for index in (1, 2, 3)
            )
        else:
            token_sums = (None, None, None)
        reservation_rows = [row for row in rows if row[4] is not None]
        reservation_state = (
            "usage_unknown"
            if any(row[4] == "usage_unknown" for row in reservation_rows)
            else (reservation_rows[-1][4] if reservation_rows else None)
        )
        reserved_cost = (
            sum(
                (Decimal(row[5]) for row in reservation_rows), Decimal("0")
            )
            if reservation_rows
            else None
        )
        last = rows[-1]
        return (
            len(rows), usage_known, *token_sums, reservation_state,
            None if reserved_cost is None else _decimal_text(reserved_cost),
            last[6], last[7], last[8], last[9],
        )

    def provider_attempt_count(self, attempt_id: str) -> int:
        """Return the number of immutable physical calls under an execution."""
        connection = self.connect()
        try:
            return connection.execute(
                "SELECT COUNT(*) FROM provider_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()[0]
        finally:
            connection.close()

    def finalize_agent_execution(
        self,
        attempt_id: str,
        *,
        lease_token: str,
        succeeded: bool,
        completed_at: datetime,
        output_hash: str | None = None,
        output_json: str | None = None,
        failure_code: str | None = None,
    ) -> None:
        """Seal a running execution once with aggregate provider telemetry."""
        _validate_lease_token(lease_token)
        completed_text = _utc_text(completed_at, "agent execution completed_at")
        if succeeded:
            try:
                decoded_output = json.loads(output_json or "")
                if not isinstance(decoded_output, dict):
                    raise ValueError
                canonical_output = json.dumps(
                    decoded_output, ensure_ascii=False, allow_nan=False,
                    sort_keys=True, separators=(",", ":"),
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                raise ValueError(
                    "successful execution requires canonical typed output JSON"
                ) from None
            expected_hash = hashlib.sha256(
                canonical_output.encode("utf-8")
            ).hexdigest()
            if output_hash != expected_hash:
                raise ValueError("successful execution output hash conflicts")
            if failure_code is not None:
                raise ValueError("successful execution cannot have failure code")
        elif (
            output_hash is not None
            or output_json is not None
            or not isinstance(failure_code, str)
            or re.fullmatch(r"[a-z0-9_.-]+", failure_code) is None
        ):
            raise ValueError("failed execution requires a safe failure code")
        with self.transaction() as connection:
            now_text = _utc_text(self.clock(), "research store clock")
            execution = connection.execute(
                "SELECT task_id, task_instance_id, state, lease_token "
                "FROM agent_executions WHERE attempt_id = ? "
                "AND state = 'running' AND lease_token = ? "
                "AND lease_expires_at > ?",
                (attempt_id, lease_token, now_text),
            ).fetchone()
            if (
                execution is None
                or execution[2] != "running"
                or execution[3] != lease_token
            ):
                raise AgentExecutionConflict(
                    "agent execution lease is not current"
                )
            aggregate = self._provider_attempt_aggregate(connection, attempt_id)
            (
                attempt_count, usage_known, input_tokens, output_tokens,
                reasoning_tokens, reservation_state, reserved_cost_usd,
                provider, model, inference_mode, fallback_reason,
            ) = aggregate
            state = "succeeded" if succeeded else "failed"
            cursor = connection.execute(
                "UPDATE agent_executions SET state = ?, completed_at = ?, "
                "safe_failure_code = ?, output_hash = ?, output_json = ?, "
                "provider = ?, model = ?, "
                "inference_mode = ?, fallback_reason = ?, "
                "provider_attempt_count = ?, input_tokens = ?, output_tokens = ?, "
                "reasoning_tokens = ?, usage_known = ?, reservation_state = ?, "
                "reserved_cost_usd = ? "
                "WHERE attempt_id = ? AND state = 'running' AND lease_token = ? "
                "AND lease_expires_at > ?",
                (
                    state, completed_text, failure_code, output_hash,
                    canonical_output if succeeded else None, provider, model,
                    inference_mode, fallback_reason, attempt_count, input_tokens,
                    output_tokens, reasoning_tokens, usage_known,
                    reservation_state, reserved_cost_usd, attempt_id, lease_token,
                    now_text,
                ),
            )
            if cursor.rowcount != 1:
                raise AgentExecutionConflict(
                    "agent execution lease could not terminalize"
                )
            task_state = "completed" if succeeded else "failed"
            instance = connection.execute(
                "SELECT state, owns_research_task_state "
                "FROM workflow_task_instances WHERE task_instance_id = ?",
                (execution[1],),
            ).fetchone()
            if instance is None or instance[0] != "running":
                raise AgentExecutionConflict(
                    "workflow task instance state conflicts"
                )
            cursor = connection.execute(
                "UPDATE workflow_task_instances SET state = ?, "
                "completed_at = ?, safe_failure_code = ? "
                "WHERE task_instance_id = ? AND state = 'running'",
                (task_state, completed_text, failure_code, execution[1]),
            )
            if cursor.rowcount != 1:
                raise AgentExecutionConflict(
                    "workflow task instance terminal state conflicts"
                )
            if instance[1]:
                cursor = connection.execute(
                    "UPDATE research_tasks SET state = ?, completed_at = ?, "
                    "error_text = ? WHERE task_id = ? AND state = 'running'",
                    (task_state, completed_text, failure_code, execution[0]),
                )
                if cursor.rowcount != 1:
                    raise AgentExecutionConflict(
                        "research task terminal state conflicts"
                    )

    def record_agent_run_audit(self, audit: AgentRunAudit) -> None:
        """Atomically retain hashes and usage, never prompt/evidence/output bodies."""
        if not isinstance(audit, AgentRunAudit):
            raise TypeError("audit must be AgentRunAudit")
        digests = (audit.prompt_hash, audit.evidence_hash, audit.output_hash)
        if any(re.fullmatch(r"[0-9a-f]{64}", item) is None for item in digests):
            raise ValueError("agent audit hashes must be SHA-256 digests")
        for value in (audit.input_tokens, audit.output_tokens, audit.reasoning_tokens):
            if type(value) is not int or value < 0:
                raise ValueError("agent usage must be non-negative integers")
        metadata = {
            "evidence_hash": audit.evidence_hash,
            "fallback_reason": audit.fallback_reason,
            "inference_mode": audit.inference_mode.value,
            "input_tokens": audit.input_tokens,
            "output_hash": audit.output_hash,
            "output_tokens": audit.output_tokens,
            "prompt_hash": audit.prompt_hash,
            "reasoning_tokens": audit.reasoning_tokens,
            "schema_version": "1",
        }
        task_scope = _canonical_json({"contract_version": "1"})
        run_values = (
            audit.run_id, audit.task_id, audit.role.value, "completed",
            _utc_text(audit.started_at), _utc_text(audit.completed_at),
            audit.provider, audit.model, _canonical_json(metadata),
        )
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO research_tasks (task_id, task_kind, scope_json, "
                "state, priority, created_at, started_at, completed_at) "
                "VALUES (?, ?, ?, 'completed', 0, ?, ?, ?)",
                (audit.task_id, audit.role.value, task_scope,
                 _utc_text(audit.started_at), _utc_text(audit.started_at),
                 _utc_text(audit.completed_at)),
            )
            existing = connection.execute(
                "SELECT run_id, task_id, role, status, started_at, completed_at, "
                "provider, model, metadata_json FROM agent_runs WHERE run_id = ?",
                (audit.run_id,),
            ).fetchone()
            if existing is not None:
                if tuple(existing) == run_values:
                    return
                raise sqlite3.IntegrityError("conflicting agent run already exists")
            connection.execute(
                "INSERT INTO agent_runs (run_id, task_id, role, status, started_at, "
                "completed_at, output_text, error_text, provider, model, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)",
                run_values,
            )

    @staticmethod
    def _agent_execution_audit(row: sqlite3.Row | tuple) -> AgentRunAudit:
        return AgentRunAudit(
            run_id=row[1], task_id=row[2], role=AgentRole(row[3]),
            started_at=_parse_utc(row[5]),
            completed_at=None if row[6] is None else _parse_utc(row[6]),
            provider=row[7], model=row[8],
            inference_mode=(
                None if row[9] is None else InferenceMode(row[9])
            ),
            prompt_hash=row[10], evidence_hash=row[11], output_hash=row[12],
            input_tokens=row[13], output_tokens=row[14],
            reasoning_tokens=row[15], fallback_reason=row[16],
            attempt_id=row[0], status=row[4], safe_failure_code=row[17],
            usage_known=bool(row[18]),
            provider_attempt_count=row[19], reservation_state=row[20],
            reserved_cost_usd=(
                None if row[21] is None else Decimal(row[21])
            ),
        )

    @staticmethod
    def _agent_execution_select() -> str:
        return (
            "SELECT attempt_id, workflow_run_id, task_id, role, state, started_at, "
            "completed_at, provider, model, inference_mode, prompt_hash, "
            "evidence_hash, output_hash, input_tokens, output_tokens, "
            "reasoning_tokens, fallback_reason, safe_failure_code, usage_known, "
            "provider_attempt_count, reservation_state, reserved_cost_usd "
            "FROM agent_executions "
        )

    def get_agent_execution(self, attempt_id: str) -> AgentRunAudit | None:
        """Return one redacted execution audit by its unambiguous attempt id."""
        connection = self.connect()
        try:
            row = connection.execute(
                self._agent_execution_select() + "WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        finally:
            connection.close()
        return None if row is None else self._agent_execution_audit(row)

    def list_agent_executions(
        self, workflow_run_id: str
    ) -> tuple[AgentRunAudit, ...]:
        """List redacted workflow executions in stable task/role/id order."""
        connection = self.connect()
        try:
            rows = connection.execute(
                self._agent_execution_select()
                + "WHERE workflow_run_id = ? "
                "ORDER BY task_id, role, attempt_id",
                (workflow_run_id,),
            ).fetchall()
        finally:
            connection.close()
        return tuple(self._agent_execution_audit(row) for row in rows)

    def list_provider_attempts(
        self, attempt_id: str
    ) -> tuple[ProviderAttemptRecord, ...]:
        """List immutable physical calls in their exact recorded order."""
        connection = self.connect()
        try:
            rows = connection.execute(
                "SELECT provider_attempt_id, attempt_id, ordinal, status, "
                "provider, model, latency_ms, input_tokens, output_tokens, "
                "reasoning_tokens, usage_known, inference_mode, recorded_at, "
                "fallback_reason, response_hash, failure_code, reservation_id, "
                "reservation_state, reserved_cost_usd FROM provider_attempts "
                "WHERE attempt_id = ? ORDER BY ordinal, provider_attempt_id",
                (attempt_id,),
            ).fetchall()
        finally:
            connection.close()
        return tuple(ProviderAttemptRecord(
            provider_attempt_id=row[0], attempt_id=row[1], ordinal=row[2],
            status=row[3], provider=row[4], model=row[5], latency_ms=row[6],
            input_tokens=row[7], output_tokens=row[8], reasoning_tokens=row[9],
            usage_known=bool(row[10]), inference_mode=InferenceMode(row[11]),
            recorded_at=_parse_utc(row[12]), fallback_reason=row[13],
            response_hash=row[14], failure_code=row[15], reservation_id=row[16],
            reservation_state=row[17],
            reserved_cost_usd=(
                None if row[18] is None else Decimal(row[18])
            ),
        ) for row in rows)

    def get_agent_run_audit(self, run_id: str) -> AgentRunAudit | None:
        """Return a legacy audit lookup, rejecting ambiguous workflow ids."""
        connection = self.connect()
        try:
            execution = connection.execute(
                self._agent_execution_select()
                + "WHERE attempt_id = ? AND state IN ('succeeded', 'failed')",
                (run_id,),
            ).fetchone()
            executions = connection.execute(
                self._agent_execution_select()
                + "WHERE workflow_run_id = ? "
                "ORDER BY task_id, role, attempt_id",
                (run_id,),
            ).fetchall()
            if execution is not None and (
                len(executions) != 0
                and not (
                    len(executions) == 1
                    and executions[0][0] == execution[0]
                )
            ):
                raise AgentExecutionConflict(
                    "agent audit lookup is ambiguous across attempt and workflow "
                    "namespaces; use explicit getters"
                )
            if execution is None:
                if len(executions) > 1:
                    raise AgentExecutionConflict(
                        "workflow audit lookup is ambiguous; use explicit list APIs"
                    )
                execution = (
                    executions[0]
                    if executions and executions[0][4] in {"succeeded", "failed"}
                    else None
                )
            if execution is not None:
                return self._agent_execution_audit(execution)
            row = connection.execute(
                "SELECT run_id, task_id, role, started_at, completed_at, provider, "
                "model, metadata_json FROM agent_runs WHERE run_id = ? "
                "AND status = 'completed' AND output_text IS NULL AND error_text IS NULL",
                (run_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        metadata = json.loads(row[7])
        return AgentRunAudit(
            run_id=row[0], task_id=row[1], role=AgentRole(row[2]),
            started_at=_parse_utc(row[3]), completed_at=_parse_utc(row[4]),
            provider=row[5], model=row[6],
            inference_mode=InferenceMode(metadata["inference_mode"]),
            prompt_hash=metadata["prompt_hash"], evidence_hash=metadata["evidence_hash"],
            output_hash=metadata["output_hash"], input_tokens=metadata["input_tokens"],
            output_tokens=metadata["output_tokens"],
            reasoning_tokens=metadata["reasoning_tokens"],
            fallback_reason=metadata["fallback_reason"],
            usage_known=True,
        )

    def append_thesis_revision(
        self,
        industry_key: str,
        thesis_text: str,
        evidence_ids: Sequence[str],
    ) -> ThesisRevision:
        """Append the next thesis version under a write-locking transaction."""
        passage_ids = tuple(evidence_ids)
        created_at = datetime.now(timezone.utc)
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 "
                "FROM industry_theses WHERE industry_key = ?",
                (industry_key,),
            ).fetchone()
            version = row[0]
            cursor = connection.execute(
                "INSERT INTO industry_theses ("
                "industry_key, version, thesis_text, created_at"
                ") VALUES (?, ?, ?, ?)",
                (industry_key, version, thesis_text, _utc_text(created_at)),
            )
            thesis_id = cursor.lastrowid
            connection.executemany(
                "INSERT INTO thesis_evidence (thesis_id, passage_id, ordinal) "
                "VALUES (?, ?, ?)",
                (
                    (thesis_id, passage_id, ordinal)
                    for ordinal, passage_id in enumerate(passage_ids)
                ),
            )
        return ThesisRevision(
            thesis_id=thesis_id,
            industry_key=industry_key,
            version=version,
            thesis_text=thesis_text,
            created_at=created_at,
            evidence_ids=passage_ids,
        )

    def list_thesis_revisions(self, industry_key: str) -> list[ThesisRevision]:
        """List immutable thesis revisions in ascending version order."""
        connection = self.connect()
        try:
            rows = connection.execute(
                "SELECT thesis_id, industry_key, version, thesis_text, created_at "
                "FROM industry_theses WHERE industry_key = ? ORDER BY version",
                (industry_key,),
            ).fetchall()
            revisions: list[ThesisRevision] = []
            for row in rows:
                evidence = connection.execute(
                    "SELECT passage_id FROM thesis_evidence "
                    "WHERE thesis_id = ? ORDER BY ordinal",
                    (row[0],),
                ).fetchall()
                revisions.append(
                    ThesisRevision(
                        thesis_id=row[0],
                        industry_key=row[1],
                        version=row[2],
                        thesis_text=row[3],
                        created_at=_parse_utc(row[4]),
                        evidence_ids=tuple(item[0] for item in evidence),
                    )
                )
            return revisions
        finally:
            connection.close()
