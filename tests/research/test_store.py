"""Integration tests for the durable SQLite research store."""

import hashlib
import json
import sqlite3
import shutil
import subprocess
import sys
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from sqlite3 import IntegrityError

import pytest

from news_bot.research.ibkr_flex import parse_statement
from news_bot.research.models import AgentRole, ClaimKind, InferenceMode, SourceDocument
from news_bot.research.store import (
    AgentExecutionConflict,
    ProviderAttemptAudit,
    ResearchStore,
)

from .conftest import fixture, make_claim, make_migrated_store, utc


REQUIRED_TABLES = {
    "schema_migrations",
    "portfolio_snapshots",
    "positions",
    "portfolio_snapshot_sources",
    "entities",
    "securities",
    "relationships",
    "entity_aliases",
    "entity_resolution_provenance",
    "relationship_evidence",
    "relationship_claim_evidence",
    "source_documents",
    "document_passages",
    "claims",
    "claim_evidence",
    "claim_dependencies",
    "industry_theses",
    "scenarios",
    "recommendations",
    "research_tasks",
    "agent_runs",
    "agent_executions",
    "workflow_task_instances",
    "provider_attempts",
    "reports",
    "model_usage",
    "budget_reservations",
}


def make_document(
    document_id: str = "document-1", content_hash: str = "sha256:document-1"
) -> SourceDocument:
    return SourceDocument(
        document_id=document_id,
        source_type="filing",
        canonical_url=f"https://example.test/{document_id}",
        publisher="Example Publisher",
        published_at=utc(2026, 8, 23),
        retrieved_at=utc(2026, 8, 24),
        content_hash=content_hash,
        raw_content_path=None,
        extraction_status="complete",
    )


def test_portfolio_snapshot_fallback_is_bound_to_one_flex_source(tmp_path):
    store = make_migrated_store(tmp_path)
    result = parse_statement(
        fixture("ibkr_statement.xml"),
        account_salt="local-test-salt-strong",
    )
    store.insert_portfolio_snapshot(
        result.snapshot, result.positions, result.account_ref
    )
    source_a = "a" * 64
    source_b = "b" * 64

    store.bind_portfolio_snapshot_source(result.snapshot.snapshot_id, source_a)

    assert store.latest_portfolio_snapshot_for_source(
        source_a, utc(2026, 8, 25)
    ) == result.snapshot.snapshot_id
    assert store.latest_portfolio_snapshot_for_source(
        source_b, utc(2026, 8, 25)
    ) is None
    with pytest.raises(sqlite3.IntegrityError, match="source binding"):
        store.bind_portfolio_snapshot_source(result.snapshot.snapshot_id, source_b)


def seed_passage(store: ResearchStore, passage_id: str = "passage-1") -> None:
    store.insert_source_document(make_document())
    store.insert_document_passage(
        passage_id=passage_id,
        document_id="document-1",
        ordinal=0,
        text="Reported revenue increased by ten percent.",
    )


def seed_claim_with_evidence(store: ResearchStore) -> None:
    seed_passage(store)
    store.insert_claim(make_claim("claim-1"), evidence_ids=["passage-1"])


def claim_execution(store, *, now=utc(2026, 8, 24), task_id="task-1"):
    return store.claim_agent_execution(
        workflow_run_id="workflow-1", task_id=task_id,
        role=AgentRole.EVENT_SCOUT, schema_version="1",
        input_hash="a" * 64, prompt_hash="b" * 64,
        evidence_hash=None, started_at=now,
    )


def test_store_migrates_empty_database(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    store.migrate()
    names = store.table_names()
    assert REQUIRED_TABLES <= names


def test_publication_outcome_scalar_accepts_only_exact_canonical_envelope(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    report_id = "report-safe"
    receipt_hash = "e" * 64
    expected = {
        "result_ref": report_id,
        "result_hash": None,
        "report_id": report_id,
        "new_agent_runs": 0,
        "material_event": None,
        "omissions": [],
        "reviewer_verdict": None,
        "originating_role": None,
        "defer_reason": None,
        "quality_gate": None,
        "published_claim_ids": [],
        "publication_receipt_hash": receipt_hash,
    }
    canonical = json.dumps(
        expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    invalid = (
        "{",
        "null",
        "[]",
        canonical.replace('"new_agent_runs":0', '"new_agent_runs":"0"'),
        canonical.replace('"report_id":"report-safe"', '"report_id":7'),
        canonical.replace('"publication_receipt_hash":"' + receipt_hash + '"',
                          '"publication_receipt_hash":null'),
        canonical[:-1] + ',"raw_secret":"sk-private"}',
        json.dumps(expected, ensure_ascii=False),
    )

    with store.connect() as connection:
        accepted = connection.execute(
            "SELECT is_canonical_publication_outcome(?, ?, ?, ?)",
            (canonical, report_id, report_id, receipt_hash),
        ).fetchone()
        rejected = [
            connection.execute(
                "SELECT is_canonical_publication_outcome(?, ?, ?, ?)",
                (candidate, report_id, report_id, receipt_hash),
            ).fetchone()[0]
            for candidate in invalid
        ]

    assert accepted == (1,)
    assert rejected == [0] * len(invalid)


def test_publication_effect_scalar_accepts_canonical_safe_audit_fields(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    report_id = "report-safe"
    receipt_hash = "e" * 64
    outcome = {
        "result_ref": report_id,
        "result_hash": None,
        "report_id": report_id,
        "new_agent_runs": 2,
        "material_event": None,
        "omissions": ["missing_claim_lineage"],
        "reviewer_verdict": None,
        "originating_role": None,
        "defer_reason": None,
        "quality_gate": None,
        "published_claim_ids": ["claim-1"],
        "publication_receipt_hash": receipt_hash,
    }
    canonical = json.dumps(
        outcome, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    unsafe_omission = canonical.replace(
        "missing_claim_lineage", "sk_private"
    )
    unsorted_claims = canonical.replace(
        '["claim-1"]', '["claim-2","claim-1"]'
    )

    with store.connect() as connection:
        assert connection.execute(
            "SELECT is_canonical_publication_effect_outcome(?, ?, ?, ?)",
            (canonical, report_id, report_id, receipt_hash),
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT is_canonical_publication_effect_outcome(?, ?, ?, ?)",
            (unsafe_omission, report_id, report_id, receipt_hash),
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT is_canonical_publication_effect_outcome(?, ?, ?, ?)",
            (unsorted_claims, report_id, report_id, receipt_hash),
        ).fetchone() == (0,)


def test_publication_effect_key_scalar_matches_canonical_orchestrator_identity(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    workflow_id = "wf_safe"
    task_id = "wft_safe"
    expected = hashlib.sha256(json.dumps(
        {"workflow_id": workflow_id, "task_id": task_id, "effect": "publish"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()

    with store.connect() as connection:
        assert connection.execute(
            "SELECT publication_effect_key(?, ?)", (workflow_id, task_id)
        ).fetchone() == (expected,)
        assert connection.execute(
            "SELECT publication_effect_key(?, ?)", ("unsafe id", task_id)
        ).fetchone() == (None,)


def test_research_utc_now_scalar_uses_store_clock(tmp_path):
    now = datetime(2026, 8, 24, 12, 34, 56, 789, tzinfo=timezone.utc)
    store = ResearchStore(tmp_path / "research.db", clock=lambda: now)

    with store.connect() as connection:
        assert connection.execute(
            "SELECT research_utc_now()"
        ).fetchone() == ("2026-08-24T12:34:56.000789Z",)


@pytest.mark.parametrize(
    "invalid_now",
    (datetime(2026, 8, 24, 12), "2026-08-24T12:00:00Z"),
)
def test_research_utc_now_scalar_rejects_invalid_clock_values(
    tmp_path, invalid_now,
):
    store = ResearchStore(tmp_path / "research.db", clock=lambda: invalid_now)

    with store.connect() as connection, pytest.raises(sqlite3.OperationalError):
        connection.execute("SELECT research_utc_now()").fetchone()


def test_relationship_cannot_be_sealed_without_matching_lineage(migrated_store):
    created_at = "2026-08-24T00:00:00.000000Z"
    with migrated_store.transaction() as connection:
        connection.executemany(
            "INSERT INTO entities (entity_id, canonical_name, entity_type, created_at) "
            "VALUES (?, ?, 'company', ?)",
            (
                ("entity_" + "a" * 64, "Company A", created_at),
                ("entity_" + "b" * 64, "Company B", created_at),
            ),
        )
        connection.execute(
            "INSERT INTO relationships (relationship_id, source_entity_id, "
            "target_entity_id, kind, as_of, confidence, stance, provenance, "
            "lineage_sealed, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
            (
                "relationship_" + "c" * 64,
                "entity_" + "a" * 64,
                "entity_" + "b" * 64,
                "supplier",
                created_at,
                "0.8",
                "supports",
                "test",
                created_at,
            ),
        )

        with pytest.raises(sqlite3.IntegrityError, match="lineage"):
            connection.execute(
                "UPDATE relationships SET lineage_sealed = 1 "
                "WHERE relationship_id = ?",
                ("relationship_" + "c" * 64,),
            )


def test_migration_is_idempotent_and_recorded_once(tmp_path):
    store = ResearchStore(tmp_path / "research.db")

    store.migrate()
    store.migrate()

    with store.connect() as connection:
        rows = connection.execute(
            "SELECT version, name, applied_at FROM schema_migrations"
        ).fetchall()
    assert len(rows) == 18
    assert rows[0][0:2] == (1, "001_initial.sql")
    assert rows[1][0:2] == (2, "002_nullable_portfolio_freshness.sql")
    assert rows[2][0:2] == (3, "003_claim_dependencies.sql")
    assert rows[3][0:2] == (4, "004_entity_resolution.sql")
    assert rows[4][0:2] == (5, "005_agent_execution_audit.sql")
    assert rows[5][0:2] == (6, "006_agent_replay_lease.sql")
    assert rows[6][0:2] == (7, "007_workflow_task_instances.sql")
    assert rows[7][0:2] == (8, "008_durable_workflow_orchestration.sql")
    assert rows[8][0:2] == (9, "009_workflow_recovery_publication.sql")
    assert rows[9][0:2] == (10, "010_publication_receipt_recovery.sql")
    assert rows[10][0:2] == (11, "011_bind_publication_receipts.sql")
    assert rows[11][0:2] == (12, "012_canonical_publication_outcomes.sql")
    assert rows[12][0:2] == (13, "013_publication_effect_outcomes.sql")
    assert rows[13][0:2] == (14, "014_publication_effect_insert_guards.sql")
    assert rows[14][0:2] == (15, "015_bind_publication_authority.sql")
    assert rows[15][0:2] == (16, "016_unique_workflow_task_stages.sql")
    assert rows[16][0:2] == (
        17, "017_require_active_publication_confirmation.sql"
    )
    assert rows[17][0:2] == (18, "018_bind_portfolio_snapshot_sources.sql")
    assert all(row[2].endswith("Z") for row in rows)


def test_existing_v1_database_upgrades_nullable_freshness_without_data_loss(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "research.db"
    v1_migration = tmp_path / "001_initial.sql"
    packaged_v1 = (
        Path(__file__).resolve().parents[2]
        / "news_bot/research/migrations/001_initial.sql"
    )
    shutil.copyfile(packaged_v1, v1_migration)
    v1_store = ResearchStore(database_path)
    monkeypatch.setattr(v1_store, "_migration_files", lambda: (v1_migration,))
    v1_store.migrate()

    snapshot_row = (
        "snapshot_existing_v1",
        "2026-08-23T00:00:00.000000Z",
        "USD",
        "4725.50",
        "525.50",
        0,
        "acct_111111111111111111111111",
        '{"source":"v1"}',
        "2026-08-24T12:00:00.000000Z",
    )
    security_row = (
        "security_existing_v1",
        None,
        "NVDA",
        "STK",
        "NASDAQ",
        "USD",
        '{"conid":"4815747","isin":"US67066G1040"}',
        '{"source":"v1"}',
        "2026-08-24T12:00:00.000000Z",
    )
    position_row = (
        "position_existing_v1",
        "snapshot_existing_v1",
        "NVDA",
        "10",
        "1200.00",
        "USD",
        "900.00",
        "security_existing_v1",
        '{"source":"v1"}',
    )
    with v1_store.transaction() as connection:
        connection.execute(
            "INSERT INTO portfolio_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            snapshot_row,
        )
        connection.execute(
            "INSERT INTO securities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            security_row,
        )
        connection.execute(
            "INSERT INTO positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            position_row,
        )

    upgraded = ResearchStore(database_path)
    upgraded.migrate()
    upgraded.migrate()

    current = parse_statement(
        fixture("ibkr_statement.xml"),
        account_salt="local-test-salt-strong",
    )
    upgraded.insert_portfolio_snapshot(
        current.snapshot, current.positions, current.account_ref
    )

    with upgraded.connect() as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        stored_snapshot = connection.execute(
            "SELECT * FROM portfolio_snapshots WHERE snapshot_id = ?",
            (snapshot_row[0],),
        ).fetchone()
        stored_security = connection.execute(
            "SELECT * FROM securities WHERE security_id = ?", (security_row[0],)
        ).fetchone()
        stored_position = connection.execute(
            "SELECT * FROM positions WHERE position_id = ?", (position_row[0],)
        ).fetchone()
        nullable = {
            row[1]: row[3]
            for row in connection.execute("PRAGMA table_info(portfolio_snapshots)")
        }
        current_staleness = connection.execute(
            "SELECT is_stale FROM portfolio_snapshots WHERE snapshot_id = ?",
            (current.snapshot.snapshot_id,),
        ).fetchone()
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        position_indexes = connection.execute(
            "SELECT name, tbl_name FROM sqlite_master "
            "WHERE type = 'index' AND name LIKE 'idx_positions_%' ORDER BY name"
        ).fetchall()

    assert versions == [
        (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,),
        (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,),
    ]
    assert stored_snapshot == snapshot_row
    assert stored_security == security_row
    assert stored_position == position_row
    assert nullable["is_stale"] == 0
    assert current_staleness == (None,)
    assert foreign_key_errors == []
    assert position_indexes == [
        ("idx_positions_security_id", "positions"),
        ("idx_positions_snapshot_id", "positions"),
        ("idx_positions_symbol", "positions"),
    ]


@pytest.mark.parametrize("stance", ["supports", "contradicts"])
def test_existing_v2_claims_upgrade_with_lineage_anchor_without_data_loss(
    tmp_path, monkeypatch, stance
):
    database_path = tmp_path / "research.db"
    migration_root = (
        Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    )
    v1 = tmp_path / "001_initial.sql"
    v2 = tmp_path / "002_nullable_portfolio_freshness.sql"
    shutil.copyfile(migration_root / v1.name, v1)
    shutil.copyfile(migration_root / v2.name, v2)
    old_store = ResearchStore(database_path)
    monkeypatch.setattr(old_store, "_migration_files", lambda: (v1, v2))
    old_store.migrate()
    seed_passage(old_store)
    claim = make_claim("claim-v2")
    with old_store.transaction() as connection:
        connection.execute(
            "INSERT INTO claims ("
            "claim_id, entity_id, kind, text, as_of, confidence, status, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                claim.claim_id,
                claim.entity_id,
                claim.kind.value,
                claim.text,
                "2026-08-24T00:00:00.000000Z",
                "0.90",
                claim.status,
                "2026-08-24T00:00:00.000000Z",
            ),
        )
        connection.execute(
            "INSERT INTO claim_evidence (claim_id, passage_id, stance) "
            "VALUES (?, ?, ?)",
            (claim.claim_id, "passage-1", stance),
        )

    upgraded = ResearchStore(database_path)
    upgraded.migrate()

    with upgraded.connect() as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        primary_anchor = connection.execute(
            "SELECT primary_passage_id, primary_supporting_claim_id, lineage_sealed "
            "FROM claims WHERE claim_id = ?",
            (claim.claim_id,),
        ).fetchone()
        retained_evidence = connection.execute(
            "SELECT passage_id, stance FROM claim_evidence WHERE claim_id = ?",
            (claim.claim_id,),
        ).fetchall()
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
    assert versions == [
        (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,),
        (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,),
    ]
    assert retained_evidence == [("passage-1", stance)]
    if stance == "supports":
        assert upgraded.get_claim(claim.claim_id) == claim
        lineage = upgraded.list_claim_lineage(claim.claim_id)[0]
        assert lineage.passage.passage_id == "passage-1"
        assert lineage.stance == stance
        assert primary_anchor == ("passage-1", None, 1)
    else:
        assert upgraded.get_claim(claim.claim_id) is None
        assert upgraded.list_claim_lineage(claim.claim_id) == ()
        assert primary_anchor == ("passage-1", None, 0)
    assert foreign_key_errors == []


def test_populated_v3_relationship_upgrade_preserves_and_quarantines_legacy_data(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "research.db"
    migration_root = (
        Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    )
    legacy_migrations = []
    for migration_name in (
        "001_initial.sql",
        "002_nullable_portfolio_freshness.sql",
        "003_claim_dependencies.sql",
    ):
        source = migration_root / migration_name
        copied = tmp_path / source.name
        shutil.copyfile(source, copied)
        legacy_migrations.append(copied)
    old_store = ResearchStore(database_path)
    monkeypatch.setattr(
        old_store, "_migration_files", lambda: tuple(legacy_migrations)
    )
    old_store.migrate()
    seed_claim_with_evidence(old_store)

    created_at = "2026-08-24T00:00:00.000000Z"
    source_entity_id = "entity_" + "a" * 64
    target_entity_id = "entity_" + "b" * 64
    relationship_id = "relationship_" + "c" * 64
    legacy_metadata = '{"evidence_id":"passage-1","status":"active"}'
    with old_store.transaction() as connection:
        connection.executemany(
            "INSERT INTO entities (entity_id, canonical_name, entity_type, created_at) "
            "VALUES (?, ?, 'company', ?)",
            (
                (source_entity_id, "Company A", created_at),
                (target_entity_id, "Company B", created_at),
            ),
        )
        connection.execute(
            "INSERT INTO relationships (relationship_id, source_entity_id, "
            "target_entity_id, kind, as_of, confidence, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                relationship_id,
                source_entity_id,
                target_entity_id,
                "supplier",
                created_at,
                "0.8",
                legacy_metadata,
                created_at,
            ),
        )

    upgraded = ResearchStore(database_path)
    upgraded.migrate()

    with upgraded.connect() as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        relationship = connection.execute(
            "SELECT source_entity_id, target_entity_id, kind, as_of, confidence, "
            "stance, provenance, primary_passage_id, primary_claim_id, "
            "lineage_sealed, metadata_json FROM relationships "
            "WHERE relationship_id = ?",
            (relationship_id,),
        ).fetchone()
        claim_state = connection.execute(
            "SELECT status, lineage_sealed FROM claims WHERE claim_id = 'claim-1'"
        ).fetchone()
        claim_evidence = connection.execute(
            "SELECT passage_id, stance FROM claim_evidence "
            "WHERE claim_id = 'claim-1'"
        ).fetchall()
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()

    assert versions == [
        (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,),
        (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,),
    ]
    assert relationship == (
        source_entity_id,
        target_entity_id,
        "supplier",
        created_at,
        "0.8",
        "supports",
        "legacy",
        None,
        None,
        0,
        legacy_metadata,
    )
    assert upgraded.list_relationships(source_entity_id) == ()
    assert claim_state == ("active", 1)
    assert claim_evidence == [("passage-1", "supports")]
    assert {
        "idx_relationships_source_entity_id",
        "idx_relationships_target_entity_id",
        "idx_relationships_kind_as_of",
    } <= indexes
    assert foreign_key_errors == []

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with upgraded.transaction() as connection:
            connection.execute(
                "UPDATE relationships SET confidence = '0.9' "
                "WHERE relationship_id = ?",
                (relationship_id,),
            )
    with pytest.raises(sqlite3.IntegrityError, match="lineage"):
        with upgraded.transaction() as connection:
            connection.execute(
                "UPDATE relationships SET lineage_sealed = 1 "
                "WHERE relationship_id = ?",
                (relationship_id,),
            )


def test_populated_v4_database_upgrades_agent_audit_without_data_loss(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "research.db"
    migration_root = (
        Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    )
    legacy_migrations = []
    for migration_name in (
        "001_initial.sql",
        "002_nullable_portfolio_freshness.sql",
        "003_claim_dependencies.sql",
        "004_entity_resolution.sql",
    ):
        copied = tmp_path / migration_name
        shutil.copyfile(migration_root / migration_name, copied)
        legacy_migrations.append(copied)
    old_store = ResearchStore(database_path)
    monkeypatch.setattr(
        old_store, "_migration_files", lambda: tuple(legacy_migrations)
    )
    old_store.migrate()
    created_at = "2026-08-24T00:00:00.000000Z"
    with old_store.transaction() as connection:
        connection.execute(
            "INSERT INTO research_tasks (task_id, task_kind, scope_json, state, "
            "priority, created_at, started_at, completed_at) "
            "VALUES ('legacy-task', 'event_scout', '{}', 'completed', 0, ?, ?, ?)",
            (created_at, created_at, created_at),
        )
        connection.execute(
            "INSERT INTO agent_runs (run_id, task_id, role, status, started_at, "
            "completed_at, provider, model, metadata_json) "
            "VALUES ('legacy-run', 'legacy-task', 'event_scout', 'completed', "
            "?, ?, 'ollama', 'legacy', '{}')",
            (created_at, created_at),
        )

    upgraded = ResearchStore(database_path)
    upgraded.migrate()

    with upgraded.connect() as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        legacy = connection.execute(
            "SELECT run_id, task_id, status FROM agent_runs"
        ).fetchall()
        foreign_key_errors = connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
    assert versions == [
        (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,),
        (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,),
    ]
    assert legacy == [("legacy-run", "legacy-task", "completed")]
    assert {"agent_executions", "provider_attempts"} <= upgraded.table_names()
    assert foreign_key_errors == []


def test_populated_v5_database_upgrades_replay_lease_without_data_loss(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "research.db"
    migration_root = (
        Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    )
    legacy_migrations = []
    for migration_name in (
        "001_initial.sql",
        "002_nullable_portfolio_freshness.sql",
        "003_claim_dependencies.sql",
        "004_entity_resolution.sql",
        "005_agent_execution_audit.sql",
    ):
        copied = tmp_path / migration_name
        shutil.copyfile(migration_root / migration_name, copied)
        legacy_migrations.append(copied)
    old_store = ResearchStore(database_path)
    monkeypatch.setattr(
        old_store, "_migration_files", lambda: tuple(legacy_migrations)
    )
    old_store.migrate()
    timestamp = "2026-08-24T00:00:00.000000Z"
    attempt_id = "agent_attempt_" + "a" * 64
    with old_store.transaction() as connection:
        connection.execute(
            "INSERT INTO research_tasks (task_id, task_kind, scope_json, state, "
            "priority, created_at, started_at) VALUES ("
            "'v5-task', 'event_scout', '{}', 'running', 0, ?, ?)",
            (timestamp, timestamp),
        )
        connection.execute(
            "INSERT INTO agent_executions (attempt_id, workflow_run_id, task_id, "
            "role, schema_version, input_hash, prompt_hash, evidence_hash, state, "
            "started_at) VALUES (?, 'v5-workflow', 'v5-task', 'event_scout', "
            "'1', ?, ?, ?, 'running', ?)",
            (attempt_id, "b" * 64, "c" * 64, "d" * 64, timestamp),
        )
        connection.execute(
            "INSERT INTO provider_attempts (provider_attempt_id, attempt_id, "
            "ordinal, status, provider, model, latency_ms, input_tokens, "
            "output_tokens, reasoning_tokens, inference_mode, recorded_at) "
            "VALUES (?, ?, 1, 'succeeded', 'ollama', 'legacy', 4, 3, 2, 1, "
            "'local_only', ?)",
            ("provider_attempt_" + "f" * 64, attempt_id, timestamp),
        )
        connection.execute(
            "UPDATE agent_executions SET state = 'succeeded', completed_at = ?, "
            "output_hash = ?, provider = 'ollama', model = 'legacy', "
            "inference_mode = 'local_only', provider_attempt_count = 1, "
            "input_tokens = 3, output_tokens = 2, reasoning_tokens = 1 "
            "WHERE attempt_id = ?",
            (timestamp, "e" * 64, attempt_id),
        )
        connection.execute(
            "UPDATE research_tasks SET state = 'completed', completed_at = ? "
            "WHERE task_id = 'v5-task'",
            (timestamp,),
        )

    upgraded = ResearchStore(database_path)
    upgraded.migrate()

    with upgraded.connect() as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        execution = connection.execute(
            "SELECT state, output_hash, output_json, lease_token, "
            "lease_expires_at FROM agent_executions WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        provider_attempt = connection.execute(
            "SELECT provider, model, input_tokens, output_tokens, "
            "reasoning_tokens, usage_known FROM provider_attempts "
            "WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        foreign_key_errors = connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
    assert versions == [
        (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,),
        (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,),
    ]
    assert execution == ("succeeded", "e" * 64, None, None, None)
    assert provider_attempt == ("ollama", "legacy", 3, 2, 1, 1)
    assert foreign_key_errors == []


def test_populated_v6_database_upgrades_workflow_instances_without_data_loss(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "research.db"
    migration_root = (
        Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    )
    legacy_migrations = []
    for migration_name in (
        "001_initial.sql",
        "002_nullable_portfolio_freshness.sql",
        "003_claim_dependencies.sql",
        "004_entity_resolution.sql",
        "005_agent_execution_audit.sql",
        "006_agent_replay_lease.sql",
    ):
        copied = tmp_path / migration_name
        shutil.copyfile(migration_root / migration_name, copied)
        legacy_migrations.append(copied)
    old_store = ResearchStore(database_path)
    monkeypatch.setattr(
        old_store, "_migration_files", lambda: tuple(legacy_migrations)
    )
    old_store.migrate()

    first_attempt = "agent_attempt_" + "a" * 64
    second_attempt = "agent_attempt_" + "b" * 64
    started_at = "2026-08-24T00:00:00.000000Z"
    later_at = "2026-08-24T00:01:00.000000Z"
    completed_at = "2026-08-24T00:02:00.000000Z"
    output_json = '{"schema_version":"1","summary":"approved"}'
    with old_store.transaction() as connection:
        connection.execute(
            "INSERT INTO research_tasks (task_id, task_kind, scope_json, state, "
            "priority, created_at, started_at) VALUES ("
            "'v6-task', 'event_scout', '{}', 'running', 0, ?, ?)",
            (started_at, started_at),
        )
        for attempt_id, workflow_run_id, attempt_started_at in (
            (first_attempt, "v6-workflow-one", started_at),
            (second_attempt, "v6-workflow-two", later_at),
        ):
            connection.execute(
                "INSERT INTO agent_executions (attempt_id, workflow_run_id, "
                "task_id, role, schema_version, input_hash, prompt_hash, state, "
                "started_at, lease_token, lease_expires_at) VALUES (?, ?, "
                "'v6-task', 'event_scout', '1', ?, ?, 'running', ?, ?, ?)",
                (
                    attempt_id,
                    workflow_run_id,
                    "c" * 64,
                    "d" * 64,
                    attempt_started_at,
                    "lease_" + attempt_id,
                    "2026-08-24T01:00:00.000000Z",
                ),
            )
        connection.execute(
            "INSERT INTO provider_attempts (provider_attempt_id, attempt_id, "
            "ordinal, status, provider, model, latency_ms, input_tokens, "
            "output_tokens, reasoning_tokens, usage_known, inference_mode, "
            "failure_code, reservation_id, reservation_state, reserved_cost_usd, "
            "recorded_at) VALUES (?, ?, 1, 'failed', 'openai', 'gpt-test', 7, "
            "NULL, NULL, NULL, 0, 'external', 'provider_usage_unavailable', "
            "'reservation-v6-1', 'usage_unknown', '0.123456789123456789', ?)",
            ("provider_attempt_" + "e" * 64, first_attempt, completed_at),
        )
        connection.execute(
            "INSERT INTO provider_attempts (provider_attempt_id, attempt_id, "
            "ordinal, status, provider, model, latency_ms, input_tokens, "
            "output_tokens, reasoning_tokens, usage_known, inference_mode, "
            "reservation_id, reservation_state, reserved_cost_usd, recorded_at) "
            "VALUES (?, ?, 2, 'succeeded', 'openai', 'gpt-test', 5, 3, 2, 1, "
            "1, 'external', 'reservation-v6-2', 'reconciled', "
            "'0.000000000000000001', ?)",
            ("provider_attempt_" + "f" * 64, first_attempt, completed_at),
        )
        connection.execute(
            "UPDATE agent_executions SET evidence_hash = ? WHERE attempt_id = ?",
            ("e" * 64, first_attempt),
        )
        connection.execute(
            "UPDATE agent_executions SET state = 'succeeded', completed_at = ?, "
            "output_hash = ?, output_json = ?, provider = 'openai', "
            "model = 'gpt-test', inference_mode = 'external', "
            "provider_attempt_count = 2, input_tokens = NULL, "
            "output_tokens = NULL, reasoning_tokens = NULL, usage_known = 0 "
            "WHERE attempt_id = ?",
            (completed_at, "f" * 64, output_json, first_attempt),
        )
        connection.execute(
            "UPDATE agent_executions SET state = 'failed', completed_at = ?, "
            "safe_failure_code = 'agent_execution' WHERE attempt_id = ?",
            (completed_at, second_attempt),
        )
        connection.execute(
            "UPDATE research_tasks SET state = 'completed', completed_at = ? "
            "WHERE task_id = 'v6-task'",
            (completed_at,),
        )

    upgraded = ResearchStore(database_path)
    upgraded.migrate()

    with upgraded.connect() as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        instances = connection.execute(
            "SELECT workflow_run_id, state, owns_research_task_state "
            "FROM workflow_task_instances WHERE task_id = 'v6-task' "
            "ORDER BY started_at, task_instance_id"
        ).fetchall()
        executions = connection.execute(
            "SELECT attempt_id, task_instance_id, state, output_json, usage_known, "
            "input_tokens, reservation_state, reserved_cost_usd "
            "FROM agent_executions WHERE task_id = 'v6-task' "
            "ORDER BY started_at, attempt_id"
        ).fetchall()
        provider_attempts = connection.execute(
            "SELECT usage_known, input_tokens, reservation_id, reservation_state, "
            "reserved_cost_usd FROM provider_attempts WHERE attempt_id = ? "
            "ORDER BY ordinal",
            (first_attempt,),
        ).fetchall()
        foreign_key_errors = connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

    assert versions == [
        (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,),
        (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,),
    ]
    assert instances == [
        ("v6-workflow-one", "completed", 1),
        ("v6-workflow-two", "failed", 0),
    ]
    assert executions == [
        (
            first_attempt,
            "task_instance_" + "a" * 64,
            "succeeded",
            output_json,
            0,
            None,
            "usage_unknown",
            "0.123456789123456790",
        ),
        (
            second_attempt,
            "task_instance_" + "b" * 64,
            "failed",
            None,
            1,
            0,
            None,
            None,
        ),
    ]
    assert provider_attempts == [
        (
            0, None, "reservation-v6-1", "usage_unknown",
            "0.123456789123456789",
        ),
        (
            1, 3, "reservation-v6-2", "reconciled",
            "0.000000000000000001",
        ),
    ]
    assert foreign_key_errors == []


def test_migration_007_rolls_back_without_exact_decimal_aggregate(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "research.db"
    migration_root = (
        Path(__file__).resolve().parents[2] / "news_bot/research/migrations"
    )
    legacy_migrations = []
    for migration_name in (
        "001_initial.sql",
        "002_nullable_portfolio_freshness.sql",
        "003_claim_dependencies.sql",
        "004_entity_resolution.sql",
        "005_agent_execution_audit.sql",
        "006_agent_replay_lease.sql",
    ):
        copied = tmp_path / migration_name
        shutil.copyfile(migration_root / migration_name, copied)
        legacy_migrations.append(copied)
    old_store = ResearchStore(database_path)
    monkeypatch.setattr(
        old_store, "_migration_files", lambda: tuple(legacy_migrations)
    )
    old_store.migrate()

    upgraded = ResearchStore(database_path)
    monkeypatch.setattr(
        upgraded, "_register_sql_functions", lambda connection: None,
        raising=False,
    )
    with pytest.raises(sqlite3.OperationalError, match="decimal_sum_exact"):
        upgraded.migrate()

    with upgraded.connect() as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert versions == [(1,), (2,), (3,), (4,), (5,), (6,)]
    assert "workflow_task_instances" not in upgraded.table_names()


def test_agent_execution_terminal_rows_and_provider_attempts_are_immutable(tmp_path):
    store = make_migrated_store(tmp_path)
    timestamp = "2026-08-24T00:00:00.000000Z"
    claim = claim_execution(store)
    attempt_id = claim.attempt_id
    store.set_agent_execution_evidence_hash(
        attempt_id, "d" * 64, lease_token=claim.lease_token
    )
    store.record_provider_attempt(
        attempt_id,
        ProviderAttemptAudit(
            status="succeeded", provider="ollama", model="local", latency_ms=1,
            input_tokens=1, output_tokens=1, reasoning_tokens=0,
            inference_mode=InferenceMode.LOCAL_ONLY,
            recorded_at=utc(2026, 8, 24),
        ),
        lease_token=claim.lease_token,
    )
    output_json = '{"approved":true}'
    store.finalize_agent_execution(
        attempt_id, lease_token=claim.lease_token, succeeded=True,
        completed_at=utc(2026, 8, 24), output_json=output_json,
        output_hash=hashlib.sha256(output_json.encode()).hexdigest(),
    )

    with pytest.raises(sqlite3.IntegrityError, match="terminal"):
        with store.transaction() as connection:
            connection.execute(
                "UPDATE agent_executions SET safe_failure_code = 'changed' "
                "WHERE attempt_id = ?",
                (attempt_id,),
            )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with store.transaction() as connection:
            connection.execute(
                "UPDATE agent_executions SET lease_token = ? WHERE attempt_id = ?",
                ("f" * 64, attempt_id),
            )
    with pytest.raises(AgentExecutionConflict, match="identity"):
        store.claim_agent_execution(
            workflow_run_id="workflow-1", task_id="task-1",
            role=AgentRole.EVENT_SCOUT, schema_version="1",
            input_hash="a" * 64, prompt_hash="b" * 64,
            evidence_hash="f" * 64,
            started_at=utc(2026, 8, 24) + timedelta(minutes=1),
        )
    with pytest.raises(sqlite3.IntegrityError, match="running"):
        with store.transaction() as connection:
            connection.execute(
                "INSERT INTO provider_attempts (provider_attempt_id, attempt_id, "
                "ordinal, status, provider, model, latency_ms, input_tokens, "
                "output_tokens, reasoning_tokens, usage_known, inference_mode, "
                "recorded_at) "
                "VALUES ('provider_attempt_terminal', ?, 2, 'failed', 'ollama', "
                "'local', 0, 0, 0, 0, 1, 'local_only', ?)",
                (attempt_id, timestamp),
            )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with store.transaction() as connection:
            connection.execute(
                "UPDATE provider_attempts SET latency_ms = 2 "
                "WHERE attempt_id = ?",
                (attempt_id,),
            )
    with pytest.raises(sqlite3.IntegrityError):
        with store.transaction() as connection:
            connection.execute(
                "INSERT INTO provider_attempts (provider_attempt_id, attempt_id, "
                "ordinal, status, provider, model, latency_ms, input_tokens, "
                "output_tokens, reasoning_tokens, usage_known, inference_mode, "
                "recorded_at) "
                "VALUES ('provider_attempt_missing', 'missing', 1, 'failed', "
                "'ollama', 'local', 0, 0, 0, 0, 1, 'local_only', ?)",
                (timestamp,),
            )


def test_stale_agent_execution_lease_is_reclaimed_by_one_new_owner(tmp_path):
    authoritative_now = [utc(2026, 8, 24)]
    store = ResearchStore(
        tmp_path / "research.db", clock=lambda: authoritative_now[0]
    )
    store.migrate()
    first = claim_execution(store, now=authoritative_now[0])

    authoritative_now[0] += timedelta(minutes=14)
    with pytest.raises(AgentExecutionConflict, match="already"):
        claim_execution(store, now=utc(2026, 8, 24) + timedelta(hours=2))

    authoritative_now[0] += timedelta(minutes=1)
    recovered = claim_execution(
        store, now=utc(2026, 8, 24)
    )
    assert recovered.attempt_id == first.attempt_id
    assert recovered.lease_token != first.lease_token
    assert recovered.replay_output_json is None

    attempt = ProviderAttemptAudit(
        status="failed", provider="openai", model="gpt-test", latency_ms=1,
        input_tokens=None, output_tokens=None, reasoning_tokens=None,
        inference_mode=InferenceMode.EXTERNAL,
        recorded_at=utc(2026, 8, 24) + timedelta(minutes=16),
        failure_code="provider_usage_unavailable", usage_known=False,
        reservation_id="reservation-1", reservation_state="usage_unknown",
        reserved_cost_usd=Decimal("0.01"),
    )
    with pytest.raises(AgentExecutionConflict, match="lease"):
        store.record_provider_attempt(
            first.attempt_id, attempt, lease_token=first.lease_token
        )
    with pytest.raises(ValueError, match="lease token"):
        store.record_provider_attempt(
            recovered.attempt_id, attempt, lease_token=None
        )
    store.record_provider_attempt(
        recovered.attempt_id, attempt, lease_token=recovered.lease_token
    )
    running = store.get_agent_execution(recovered.attempt_id)
    assert running is not None and running.status == "running"
    assert running.usage_known is False
    assert running.provider_attempt_count == 1
    assert (
        running.input_tokens, running.output_tokens, running.reasoning_tokens
    ) == (None, None, None)
    assert running.reservation_state == "usage_unknown"
    assert running.reserved_cost_usd == Decimal("0.01")
    with pytest.raises(AgentExecutionConflict, match="lease"):
        store.finalize_agent_execution(
            first.attempt_id, lease_token=first.lease_token, succeeded=False,
            completed_at=utc(2026, 8, 24) + timedelta(minutes=17),
            failure_code="agent_execution",
        )
    store.finalize_agent_execution(
        recovered.attempt_id, lease_token=recovered.lease_token,
        succeeded=False,
        completed_at=utc(2026, 8, 24) + timedelta(minutes=17),
        failure_code="agent_execution",
    )
    audit = store.get_agent_execution(recovered.attempt_id)
    assert audit is not None and audit.usage_known is False
    assert audit.provider_attempt_count == 1
    assert (
        audit.input_tokens, audit.output_tokens, audit.reasoning_tokens
    ) == (None, None, None)
    assert audit.reservation_state == "usage_unknown"
    assert audit.reserved_cost_usd == Decimal("0.01")


def test_future_caller_timestamp_cannot_steal_live_agent_lease(tmp_path):
    authoritative_now = utc(2026, 8, 24)
    store = ResearchStore(
        tmp_path / "research.db", clock=lambda: authoritative_now
    )
    store.migrate()
    first = claim_execution(store, now=authoritative_now)

    with pytest.raises(AgentExecutionConflict, match="live lease"):
        claim_execution(
            store, now=authoritative_now + timedelta(hours=1)
        )

    with store.connect() as connection:
        lease = connection.execute(
            "SELECT lease_token, lease_expires_at FROM agent_executions "
            "WHERE attempt_id = ?",
            (first.attempt_id,),
        ).fetchone()
    assert lease == (
        first.lease_token,
        "2026-08-24T00:15:00.000000Z",
    )


@pytest.mark.parametrize("late_by", (timedelta(), timedelta(hours=1)))
def test_provider_attempt_rejects_current_token_at_or_after_lease_expiry(
    tmp_path, late_by
):
    now = [utc(2026, 8, 24)]
    store = ResearchStore(tmp_path / "research.db", clock=lambda: now[0])
    store.migrate()
    claim = claim_execution(store, now=now[0])
    now[0] += timedelta(minutes=15) + late_by
    attempt = ProviderAttemptAudit(
        status="failed", provider="openai", model="gpt-test", latency_ms=1,
        input_tokens=0, output_tokens=0, reasoning_tokens=0,
        inference_mode=InferenceMode.EXTERNAL, recorded_at=now[0],
        failure_code="provider_unavailable",
    )

    with pytest.raises(AgentExecutionConflict, match="lease"):
        store.record_provider_attempt(
            claim.attempt_id, attempt, lease_token=claim.lease_token
        )

    assert store.list_provider_attempts(claim.attempt_id) == ()


@pytest.mark.parametrize("late_by", (timedelta(), timedelta(hours=1)))
def test_finalize_rejects_current_token_at_or_after_lease_expiry(
    tmp_path, late_by
):
    now = [utc(2026, 8, 24)]
    store = ResearchStore(tmp_path / "research.db", clock=lambda: now[0])
    store.migrate()
    claim = claim_execution(store, now=now[0])
    now[0] += timedelta(minutes=15) + late_by

    with pytest.raises(AgentExecutionConflict, match="lease"):
        store.finalize_agent_execution(
            claim.attempt_id, lease_token=claim.lease_token, succeeded=False,
            completed_at=now[0], failure_code="agent_execution",
        )

    execution = store.get_agent_execution(claim.attempt_id)
    assert execution is not None and execution.status == "running"


def test_agent_terminalization_rolls_back_if_task_state_diverged(tmp_path):
    store = make_migrated_store(tmp_path)
    claim = claim_execution(store)
    with store.transaction() as connection:
        connection.execute(
            "UPDATE research_tasks SET state = 'failed' WHERE task_id = 'task-1'"
        )

    with pytest.raises(AgentExecutionConflict, match="task"):
        store.finalize_agent_execution(
            claim.attempt_id, lease_token=claim.lease_token, succeeded=False,
            completed_at=utc(2026, 8, 24) + timedelta(minutes=1),
            failure_code="agent_execution",
        )

    with store.connect() as connection:
        state = connection.execute(
            "SELECT state FROM agent_executions WHERE attempt_id = ?",
            (claim.attempt_id,),
        ).fetchone()
    assert state == ("running",)


def test_legacy_agent_audit_lookup_rejects_attempt_workflow_namespace_collision(
    tmp_path,
):
    store = make_migrated_store(tmp_path)
    first = claim_execution(store)
    store.finalize_agent_execution(
        first.attempt_id, lease_token=first.lease_token, succeeded=False,
        completed_at=utc(2026, 8, 24), failure_code="agent_execution",
    )
    second = store.claim_agent_execution(
        workflow_run_id=first.attempt_id, task_id="task-2",
        role=AgentRole.EVENT_SCOUT, schema_version="1",
        input_hash="a" * 64, prompt_hash="b" * 64,
        evidence_hash=None, started_at=utc(2026, 8, 24),
    )
    store.finalize_agent_execution(
        second.attempt_id, lease_token=second.lease_token, succeeded=False,
        completed_at=utc(2026, 8, 24), failure_code="agent_execution",
    )

    assert store.get_agent_execution(first.attempt_id) is not None
    assert len(store.list_agent_executions(first.attempt_id)) == 1
    with pytest.raises(AgentExecutionConflict, match="ambiguous"):
        store.get_agent_run_audit(first.attempt_id)


def test_store_fails_closed_when_no_migrations_are_available(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path / "research.db")
    monkeypatch.setattr(store, "_migration_files", lambda: ())

    with pytest.raises(RuntimeError, match="No research store migrations found"):
        store.migrate()

    assert not store.database_path.exists()


def test_built_wheel_contains_and_applies_all_migrations(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    source_copy = tmp_path / "source"
    wheel_dir = tmp_path / "wheelhouse"
    shutil.copytree(
        project_root,
        source_copy,
        ignore=shutil.ignore_patterns(
            ".git", ".worktrees", "venv", "__pycache__", ".pytest_cache", "*.pyc"
        ),
    )
    wheel_dir.mkdir()

    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_dir),
            str(source_copy),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    migration_names = {
        "news_bot/research/migrations/001_initial.sql",
        "news_bot/research/migrations/002_nullable_portfolio_freshness.sql",
        "news_bot/research/migrations/003_claim_dependencies.sql",
        "news_bot/research/migrations/004_entity_resolution.sql",
        "news_bot/research/migrations/005_agent_execution_audit.sql",
        "news_bot/research/migrations/006_agent_replay_lease.sql",
        "news_bot/research/migrations/007_workflow_task_instances.sql",
        "news_bot/research/migrations/008_durable_workflow_orchestration.sql",
        "news_bot/research/migrations/009_workflow_recovery_publication.sql",
        "news_bot/research/migrations/010_publication_receipt_recovery.sql",
        "news_bot/research/migrations/011_bind_publication_receipts.sql",
        "news_bot/research/migrations/012_canonical_publication_outcomes.sql",
        "news_bot/research/migrations/013_publication_effect_outcomes.sql",
        "news_bot/research/migrations/014_publication_effect_insert_guards.sql",
        "news_bot/research/migrations/015_bind_publication_authority.sql",
        "news_bot/research/migrations/016_unique_workflow_task_stages.sql",
        "news_bot/research/migrations/017_require_active_publication_confirmation.sql",
        "news_bot/research/migrations/018_bind_portfolio_snapshot_sources.sql",
    }
    prompt_names = {
        "news_bot/research/prompts/director.md",
        "news_bot/research/prompts/emerging_scout.md",
        "news_bot/research/prompts/event_scout.md",
        "news_bot/research/prompts/evidence_analyst.md",
        "news_bot/research/prompts/fundamental_analyst.md",
        "news_bot/research/prompts/industry_strategist.md",
        "news_bot/research/prompts/research_editor.md",
        "news_bot/research/prompts/skeptical_reviewer.md",
    }
    with zipfile.ZipFile(wheel) as archive:
        assert migration_names <= set(archive.namelist())
        assert prompt_names <= set(archive.namelist())

    resource_probe = (
        "import sqlite3, sys, tempfile; from importlib import resources; "
        "from pathlib import Path; "
        f"sys.path.insert(0, {str(wheel)!r}); "
        "migration_dir = resources.files('news_bot.research').joinpath('migrations'); "
        "assert {p.name for p in migration_dir.iterdir()} >= "
        "{'001_initial.sql', '002_nullable_portfolio_freshness.sql', "
        "'003_claim_dependencies.sql', '004_entity_resolution.sql', "
        "'005_agent_execution_audit.sql', '006_agent_replay_lease.sql', "
        "'007_workflow_task_instances.sql', "
        "'008_durable_workflow_orchestration.sql', "
        "'009_workflow_recovery_publication.sql', "
        "'010_publication_receipt_recovery.sql', "
        "'011_bind_publication_receipts.sql', "
        "'012_canonical_publication_outcomes.sql', "
        "'013_publication_effect_outcomes.sql', "
        "'014_publication_effect_insert_guards.sql', "
        "'015_bind_publication_authority.sql', "
        "'016_unique_workflow_task_stages.sql', "
        "'017_require_active_publication_confirmation.sql', "
        "'018_bind_portfolio_snapshot_sources.sql'}; "
        "prompt_dir = resources.files('news_bot.research').joinpath('prompts'); "
        "assert {p.name for p in prompt_dir.iterdir()} >= "
        "{'director.md', 'emerging_scout.md', 'event_scout.md', "
        "'evidence_analyst.md', 'fundamental_analyst.md', "
        "'industry_strategist.md', 'research_editor.md', "
        "'skeptical_reviewer.md'}; "
        "from news_bot.research.store import ResearchStore; "
        "db = Path(tempfile.mkdtemp()) / 'research.db'; "
        "store = ResearchStore(db); store.migrate(); "
        "connection = sqlite3.connect(db); "
        "assert connection.execute('SELECT version FROM schema_migrations "
        "ORDER BY version').fetchall() == "
        "[(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), "
        "(11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,)]; "
        "connection.close()"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", resource_probe],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_failed_multi_statement_migration_is_atomic(tmp_path, monkeypatch):
    store = make_migrated_store(tmp_path)
    bad_migration = tmp_path / "019_broken.sql"
    bad_migration.write_text(
        "CREATE TABLE should_roll_back (id INTEGER);\n"
        "INSERT INTO missing_table (id) VALUES (1);\n",
        encoding="utf-8",
    )
    original_files = store._migration_files()
    monkeypatch.setattr(
        store, "_migration_files", lambda: (*original_files, bad_migration)
    )

    with pytest.raises(sqlite3.OperationalError):
        store.migrate()

    assert "should_roll_back" not in store.table_names()
    with store.connect() as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert versions == [
        (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,),
        (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,),
    ]


def test_migrations_are_sorted_by_numeric_version(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path / "research.db")
    migration_two = tmp_path / "2_create.sql"
    migration_ten = tmp_path / "10_insert.sql"
    migration_two.write_text(
        "CREATE TABLE ordered_migrations (value TEXT);", encoding="utf-8"
    )
    migration_ten.write_text(
        "INSERT INTO ordered_migrations (value) VALUES ('ten');", encoding="utf-8"
    )
    monkeypatch.setattr(
        store, "_migration_files", lambda: (migration_ten, migration_two)
    )

    store.migrate()

    with store.connect() as connection:
        values = connection.execute(
            "SELECT value FROM ordered_migrations"
        ).fetchall()
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert values == [("ten",)]
    assert versions == [(2,), (10,)]


def test_duplicate_numeric_migration_versions_are_rejected(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path / "research.db")
    first = tmp_path / "2_first.sql"
    duplicate = tmp_path / "02_duplicate.sql"
    first.write_text("SELECT 1;", encoding="utf-8")
    duplicate.write_text("SELECT 2;", encoding="utf-8")
    monkeypatch.setattr(store, "_migration_files", lambda: (first, duplicate))

    with pytest.raises(ValueError, match="Migration versions must be unique"):
        store.migrate()


@pytest.mark.parametrize("filename", ["migration_2.sql", "2.sql", "2_bad.txt"])
def test_migration_filenames_require_strict_numeric_prefix(
    tmp_path, monkeypatch, filename
):
    store = ResearchStore(tmp_path / "research.db")
    invalid = tmp_path / filename
    invalid.write_text("SELECT 1;", encoding="utf-8")
    monkeypatch.setattr(store, "_migration_files", lambda: (invalid,))

    with pytest.raises(ValueError, match=rf"Invalid migration filename: {filename}"):
        store.migrate()


def test_migration_parser_splits_multiple_statements_on_one_line(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    with store.transaction() as connection:
        store._execute_script_atomically(
            connection,
            "CREATE TABLE first_table (id INTEGER); "
            "CREATE TABLE second_table (id INTEGER);",
        )

    assert {"first_table", "second_table"} <= store.table_names()


def test_migration_parser_supports_multiline_trigger_bodies(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    script = """
    CREATE TABLE source_rows (id INTEGER PRIMARY KEY, value TEXT);
    CREATE TABLE audit_rows (source_id INTEGER, old_value TEXT, new_value TEXT);
    CREATE TRIGGER audit_source_update
    AFTER UPDATE ON source_rows
    BEGIN
        INSERT INTO audit_rows (source_id, old_value, new_value)
        VALUES (OLD.id, OLD.value, NEW.value);
        UPDATE audit_rows SET new_value = upper(new_value) WHERE source_id = NEW.id;
    END;
    INSERT INTO source_rows (id, value) VALUES (1, 'before');
    UPDATE source_rows SET value = 'after' WHERE id = 1;
    """

    with store.transaction() as connection:
        store._execute_script_atomically(connection, script)

    with store.connect() as connection:
        audit = connection.execute(
            "SELECT source_id, old_value, new_value FROM audit_rows"
        ).fetchall()
    assert audit == [(1, "before", "AFTER")]


@pytest.mark.parametrize(
    "script",
    [
        "",
        "   \n\t",
        "-- comment only, with a semicolon;\n",
        "/* block comment only; */",
    ],
)
def test_migration_parser_accepts_blank_or_comment_only_scripts(tmp_path, script):
    store = ResearchStore(tmp_path / "research.db")

    with store.transaction() as connection:
        store._execute_script_atomically(connection, script)


def test_migration_parser_accepts_trailing_comments(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    with store.transaction() as connection:
        store._execute_script_atomically(
            connection,
            "CREATE TABLE before_comment (id INTEGER);\n"
            "-- migration explanation without a final semicolon\n",
        )

    assert "before_comment" in store.table_names()


def test_one_line_migration_failure_rolls_back_all_statements(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    with pytest.raises(sqlite3.Error):
        with store.transaction() as connection:
            store._execute_script_atomically(
                connection,
                "CREATE TABLE rolled_back (id INTEGER); "
                "INSERT INTO missing_table (id) VALUES (1);",
            )

    assert "rolled_back" not in store.table_names()


def test_connections_enable_required_sqlite_pragmas(tmp_path):
    store = ResearchStore(tmp_path / "research.db")

    with store.connect() as connection:
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]

    assert foreign_keys == 1
    assert journal_mode == "wal"
    assert busy_timeout == 5000


def test_transaction_rolls_back_on_exception(tmp_path):
    store = make_migrated_store(tmp_path)

    with pytest.raises(RuntimeError, match="abort"):
        with store.transaction() as connection:
            connection.execute(
                "INSERT INTO entities "
                "(entity_id, canonical_name, entity_type, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("entity-1", "Example", "company", "2026-08-24T00:00:00.000000Z"),
            )
            raise RuntimeError("abort")

    with store.connect() as connection:
        count = connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    assert count == 0


def test_foreign_key_rejection_is_enabled(tmp_path):
    store = make_migrated_store(tmp_path)

    with pytest.raises(IntegrityError):
        with store.transaction() as connection:
            connection.execute(
                "INSERT INTO positions "
                "(position_id, snapshot_id, symbol, quantity, market_value, currency) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("position-1", "missing", "EXM", "1", "10.00", "USD"),
            )


def test_schema_uses_text_for_decimal_values_and_required_indexes(tmp_path):
    store = make_migrated_store(tmp_path)

    with store.connect() as connection:
        decimal_columns = {
            (table, row[1]): row[2]
            for table in ("portfolio_snapshots", "positions", "claims", "model_usage")
            for row in connection.execute(f"PRAGMA table_info({table})")
            if row[1]
            in {"nav", "cash", "quantity", "market_value", "cost_basis", "confidence", "cost_usd"}
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }

    assert set(decimal_columns.values()) == {"TEXT"}
    assert {
        "idx_source_documents_canonical_url",
        "idx_entities_canonical_name",
        "idx_securities_symbol",
        "idx_claims_status",
        "idx_research_tasks_state",
    } <= indexes


def test_budget_reservation_supports_fresh_owner_and_reconciliation_states(tmp_path):
    store = make_migrated_store(tmp_path)
    created_at = "2026-08-24T00:00:00.000000Z"

    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO budget_reservations ("
            "reservation_id, owner_key, amount_usd, state, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (
                "reservation-1",
                "fresh-run-1",
                "1.2500",
                "reserved",
                created_at,
                created_at,
            ),
        )
        connection.execute(
            "UPDATE budget_reservations SET state = ?, updated_at = ? "
            "WHERE reservation_id = ?",
            ("usage_unknown", "2026-08-24T01:00:00.000000Z", "reservation-1"),
        )
        connection.execute(
            "UPDATE budget_reservations SET state = ?, updated_at = ? "
            "WHERE reservation_id = ?",
            ("reconciled", "2026-08-24T02:00:00.000000Z", "reservation-1"),
        )

    with store.connect() as connection:
        stored = connection.execute(
            "SELECT owner_key, task_id, run_id, amount_usd, state "
            "FROM budget_reservations WHERE reservation_id = ?",
            ("reservation-1",),
        ).fetchone()
    assert stored == ("fresh-run-1", None, None, "1.2500", "reconciled")


def test_schema_indexes_child_foreign_keys_and_research_workflows(tmp_path):
    store = make_migrated_store(tmp_path)
    with store.connect() as connection:
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }

    assert {
        "idx_positions_snapshot_id",
        "idx_positions_security_id",
        "idx_securities_entity_id",
        "idx_relationships_source_entity_id",
        "idx_relationships_target_entity_id",
        "idx_document_passages_document_id",
        "idx_claims_entity_as_of",
        "idx_claim_evidence_passage_id",
        "idx_thesis_evidence_passage_id",
        "idx_recommendations_entity_id",
        "idx_recommendations_security_id",
        "idx_research_tasks_parent_task_id",
        "idx_agent_runs_task_id",
        "idx_reports_created_by_run_id",
        "idx_model_usage_run_id",
        "idx_model_usage_recorded_at",
        "idx_budget_reservations_owner_key",
        "idx_budget_reservations_task_id",
        "idx_budget_reservations_run_id",
        "idx_budget_reservations_state_created_at",
    } <= indexes


def test_claim_requires_evidence(tmp_path):
    store = make_migrated_store(tmp_path)
    with pytest.raises(IntegrityError):
        store.insert_claim(make_claim("claim-1"), evidence_ids=[])
    assert store.get_claim("claim-1") is None


@pytest.mark.parametrize("kind", ["fact", "guidance", "estimate", "inference"])
def test_database_rejects_claim_without_required_lineage_anchor(tmp_path, kind):
    store = make_migrated_store(tmp_path)

    with pytest.raises(IntegrityError):
        with store.transaction() as connection:
            connection.execute(
                "INSERT INTO claims ("
                "claim_id, entity_id, kind, text, as_of, confidence, status, created_at"
                ") VALUES (?, NULL, ?, ?, ?, ?, ?, ?)",
                (
                    f"claim-{kind}",
                    kind,
                    "Uncited material claim.",
                    "2026-08-24T00:00:00.000000Z",
                    "0.5",
                    "active",
                    "2026-08-24T00:00:00.000000Z",
                ),
            )


def test_direct_claim_rows_must_begin_unsealed(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_passage(store)
    values = (
        "claim-direct",
        "fact",
        "Direct claim.",
        "2026-08-24T00:00:00.000000Z",
        "0.5",
        "active",
        "passage-1",
        "2026-08-24T00:00:00.000000Z",
    )

    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO claims (claim_id, kind, text, as_of, confidence, status, "
            "primary_passage_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )

    with store.connect() as connection:
        sealed = connection.execute(
            "SELECT lineage_sealed FROM claims WHERE claim_id = ?",
            ("claim-direct",),
        ).fetchone()
    assert sealed == (0,)
    assert store.get_claim("claim-direct") is None
    assert store.list_claim_lineage("claim-direct") == ()


def test_direct_claim_insert_cannot_start_sealed(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_passage(store)

    with pytest.raises(IntegrityError, match="begin unsealed"):
        with store.transaction() as connection:
            connection.execute(
                "INSERT INTO claims (claim_id, kind, text, as_of, confidence, status, "
                "primary_passage_id, lineage_sealed, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "claim-direct",
                    "fact",
                    "Direct claim.",
                    "2026-08-24T00:00:00.000000Z",
                    "0.5",
                    "active",
                    "passage-1",
                    1,
                    "2026-08-24T00:00:00.000000Z",
                ),
            )


def test_claim_seal_requires_matching_supporting_passage_link(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_passage(store)
    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO claims (claim_id, kind, text, as_of, confidence, status, "
            "primary_passage_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "claim-direct",
                "fact",
                "Direct claim.",
                "2026-08-24T00:00:00.000000Z",
                "0.5",
                "active",
                "passage-1",
                "2026-08-24T00:00:00.000000Z",
            ),
        )
        connection.execute(
            "INSERT INTO claim_evidence (claim_id, passage_id, stance) "
            "VALUES (?, ?, ?)",
            ("claim-direct", "passage-1", "contradicts"),
        )

    with pytest.raises(IntegrityError, match="matching supporting lineage"):
        with store.transaction() as connection:
            connection.execute(
                "UPDATE claims SET lineage_sealed = 1 WHERE claim_id = ?",
                ("claim-direct",),
            )

    assert store.get_claim("claim-direct") is None
    assert store.list_claim_lineage("claim-direct") == ()


def test_inference_seal_requires_matching_claim_dependency(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_claim_with_evidence(store)
    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO claims (claim_id, kind, text, as_of, confidence, status, "
            "primary_supporting_claim_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "inference-direct",
                "inference",
                "Capacity may tighten.",
                "2026-08-24T00:00:00.000000Z",
                "0.5",
                "active",
                "claim-1",
                "2026-08-24T00:00:00.000000Z",
            ),
        )

    with pytest.raises(IntegrityError, match="matching supporting lineage"):
        with store.transaction() as connection:
            connection.execute(
                "UPDATE claims SET lineage_sealed = 1 WHERE claim_id = ?",
                ("inference-direct",),
            )

    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO claim_dependencies (claim_id, supporting_claim_id) "
            "VALUES (?, ?)",
            ("inference-direct", "claim-1"),
        )
        connection.execute(
            "UPDATE claims SET lineage_sealed = 1 WHERE claim_id = ?",
            ("inference-direct",),
        )

    assert store.get_claim("inference-direct") is not None
    assert store.list_supporting_claim_ids("inference-direct") == ("claim-1",)


def test_inference_cannot_seal_against_quarantined_supporting_claim(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_passage(store)
    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO claims (claim_id, kind, text, as_of, confidence, status, "
            "primary_passage_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "claim-quarantined",
                "fact",
                "Quarantined claim.",
                "2026-08-24T00:00:00.000000Z",
                "0.5",
                "active",
                "passage-1",
                "2026-08-24T00:00:00.000000Z",
            ),
        )
        connection.execute(
            "INSERT INTO claim_evidence (claim_id, passage_id, stance) "
            "VALUES (?, ?, ?)",
            ("claim-quarantined", "passage-1", "contradicts"),
        )
        connection.execute(
            "INSERT INTO claims (claim_id, kind, text, as_of, confidence, status, "
            "primary_supporting_claim_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "inference-direct",
                "inference",
                "Capacity may tighten.",
                "2026-08-24T00:00:00.000000Z",
                "0.5",
                "active",
                "claim-quarantined",
                "2026-08-24T00:00:00.000000Z",
            ),
        )
    with pytest.raises(IntegrityError, match="supporting claim must be sealed"):
        with store.transaction() as connection:
            connection.execute(
                "INSERT INTO claim_dependencies (claim_id, supporting_claim_id) "
                "VALUES (?, ?)",
                ("inference-direct", "claim-quarantined"),
            )

    assert store.get_claim("inference-direct") is None


def test_public_inference_with_mixed_dependency_validity_is_atomic(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_claim_with_evidence(store)
    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO claims (claim_id, kind, text, as_of, confidence, status, "
            "primary_passage_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "claim-quarantined",
                "fact",
                "Quarantined claim.",
                "2026-08-24T00:00:00.000000Z",
                "0.5",
                "active",
                "passage-1",
                "2026-08-24T00:00:00.000000Z",
            ),
        )
        connection.execute(
            "INSERT INTO claim_evidence (claim_id, passage_id, stance) "
            "VALUES (?, ?, ?)",
            ("claim-quarantined", "passage-1", "contradicts"),
        )
    inference = replace(
        make_claim("inference-mixed"),
        kind=ClaimKind.INFERENCE,
        text="Capacity may tighten.",
    )

    with pytest.raises(IntegrityError, match="supporting claim must be sealed"):
        store.insert_claim_with_lineage(
            inference,
            passage_links=(),
            supporting_claim_ids=("claim-1", "claim-quarantined"),
        )

    assert store.get_claim("inference-mixed") is None
    with store.connect() as connection:
        raw_claim = connection.execute(
            "SELECT claim_id FROM claims WHERE claim_id = ?", ("inference-mixed",)
        ).fetchone()
        raw_links = connection.execute(
            "SELECT supporting_claim_id FROM claim_dependencies WHERE claim_id = ?",
            ("inference-mixed",),
        ).fetchall()
    assert raw_claim is None
    assert raw_links == []


def test_direct_dependency_to_quarantined_target_is_rejected(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_claim_with_evidence(store)
    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO claims (claim_id, kind, text, as_of, confidence, status, "
            "primary_passage_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "claim-quarantined",
                "fact",
                "Quarantined claim.",
                "2026-08-24T00:00:00.000000Z",
                "0.5",
                "active",
                "passage-1",
                "2026-08-24T00:00:00.000000Z",
            ),
        )
        connection.execute(
            "INSERT INTO claims (claim_id, kind, text, as_of, confidence, status, "
            "primary_supporting_claim_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "inference-direct",
                "inference",
                "Capacity may tighten.",
                "2026-08-24T00:00:00.000000Z",
                "0.5",
                "active",
                "claim-1",
                "2026-08-24T00:00:00.000000Z",
            ),
        )

    with pytest.raises(IntegrityError, match="supporting claim must be sealed"):
        with store.transaction() as connection:
            connection.execute(
                "INSERT INTO claim_dependencies (claim_id, supporting_claim_id) "
                "VALUES (?, ?)",
                ("inference-direct", "claim-quarantined"),
            )


def test_supporting_claim_reads_filter_quarantined_targets_defensively(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_claim_with_evidence(store)
    inference = replace(
        make_claim("inference-sealed"),
        kind=ClaimKind.INFERENCE,
        text="Capacity may tighten.",
    )
    store.insert_claim_with_lineage(
        inference,
        passage_links=(),
        supporting_claim_ids=("claim-1",),
    )
    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO claims (claim_id, kind, text, as_of, confidence, status, "
            "primary_passage_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "claim-quarantined",
                "fact",
                "Quarantined claim.",
                "2026-08-24T00:00:00.000000Z",
                "0.5",
                "active",
                "passage-1",
                "2026-08-24T00:00:00.000000Z",
            ),
        )
        connection.execute("DROP TRIGGER claim_dependencies_no_insert_after_seal")
        connection.execute(
            "DROP TRIGGER IF EXISTS claim_dependencies_require_sealed_target"
        )
        connection.execute(
            "INSERT INTO claim_dependencies (claim_id, supporting_claim_id) "
            "VALUES (?, ?)",
            ("inference-sealed", "claim-quarantined"),
        )

    assert store.list_supporting_claim_ids("inference-sealed") == ("claim-1",)


def test_inference_accepts_multiple_sealed_dependencies(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_claim_with_evidence(store)
    store.insert_source_document(
        make_document("document-2", content_hash="sha256:document-2")
    )
    store.insert_document_passage(
        passage_id="passage-2",
        document_id="document-2",
        ordinal=0,
        text="A second source passage.",
    )
    store.insert_claim(make_claim("claim-2"), evidence_ids=("passage-2",))
    inference = replace(
        make_claim("inference-multiple"),
        kind=ClaimKind.INFERENCE,
        text="Capacity may tighten.",
    )

    store.insert_claim_with_lineage(
        inference,
        passage_links=(),
        supporting_claim_ids=("claim-2", "claim-1"),
    )

    assert store.get_claim("inference-multiple") == inference
    assert store.list_supporting_claim_ids("inference-multiple") == (
        "claim-1",
        "claim-2",
    )


def test_claim_missing_passage_is_atomic(tmp_path):
    store = make_migrated_store(tmp_path)

    with pytest.raises(IntegrityError):
        store.insert_claim(make_claim("claim-1"), evidence_ids=["missing"])

    assert store.get_claim("claim-1") is None


def test_inference_missing_claim_dependency_is_atomic(tmp_path):
    store = make_migrated_store(tmp_path)
    inference = replace(
        make_claim("inference-1"),
        kind=ClaimKind.INFERENCE,
        text="Capacity may tighten.",
    )

    with pytest.raises(IntegrityError):
        store.insert_claim_with_lineage(
            inference,
            passage_links=(),
            supporting_claim_ids=("missing",),
        )

    assert store.get_claim("inference-1") is None


def test_self_referential_claim_dependency_is_atomic(tmp_path):
    store = make_migrated_store(tmp_path)
    inference = replace(
        make_claim("inference-1"),
        kind=ClaimKind.INFERENCE,
        text="Capacity may tighten.",
    )

    with pytest.raises(IntegrityError, match="themselves"):
        store.insert_claim_with_lineage(
            inference,
            passage_links=(),
            supporting_claim_ids=("inference-1",),
        )

    assert store.get_claim("inference-1") is None


def test_claim_dependencies_are_indexed_restrictive_and_immutable(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_claim_with_evidence(store)
    inference = replace(
        make_claim("inference-1"),
        kind=ClaimKind.INFERENCE,
        text="Capacity may tighten.",
    )
    store.insert_claim_with_lineage(
        inference,
        passage_links=(),
        supporting_claim_ids=("claim-1",),
    )

    assert store.list_supporting_claim_ids("inference-1") == ("claim-1",)
    with store.connect() as connection:
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
    assert "idx_claim_dependencies_supporting_claim_id" in indexes

    with pytest.raises(IntegrityError, match="claim dependency links are immutable"):
        with store.transaction() as connection:
            connection.execute(
                "UPDATE claim_dependencies SET supporting_claim_id = ? "
                "WHERE claim_id = ?",
                ("inference-1", "inference-1"),
            )
    with pytest.raises(IntegrityError, match="claim dependency links are immutable"):
        with store.transaction() as connection:
            connection.execute(
                "DELETE FROM claim_dependencies WHERE claim_id = ?", ("inference-1",)
            )


def test_claim_lineage_rejects_links_appended_after_atomic_creation(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_claim_with_evidence(store)
    store.insert_source_document(
        make_document("document-2", content_hash="sha256:document-2")
    )
    store.insert_document_passage(
        passage_id="passage-2",
        document_id="document-2",
        ordinal=0,
        text="A second source passage.",
    )
    store.insert_claim(make_claim("claim-2"), evidence_ids=("passage-2",))

    with pytest.raises(IntegrityError, match="claim evidence links are immutable"):
        with store.transaction() as connection:
            connection.execute(
                "INSERT INTO claim_evidence (claim_id, passage_id, stance) "
                "VALUES (?, ?, ?)",
                ("claim-1", "passage-2", "contradicts"),
            )

    with pytest.raises(IntegrityError, match="claim dependency links are immutable"):
        with store.transaction() as connection:
            connection.execute(
                "INSERT INTO claim_dependencies (claim_id, supporting_claim_id) "
                "VALUES (?, ?)",
                ("claim-1", "claim-2"),
            )

    with pytest.raises(IntegrityError, match="claim lineage seal is immutable"):
        with store.transaction() as connection:
            connection.execute(
                "UPDATE claims SET lineage_sealed = 0 WHERE claim_id = ?",
                ("claim-1",),
            )


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        (
            "UPDATE source_documents SET publisher = 'changed' "
            "WHERE document_id = 'document-1'",
            "source documents are immutable",
        ),
        (
            "DELETE FROM source_documents WHERE document_id = 'document-1'",
            "source documents are immutable",
        ),
        (
            "UPDATE document_passages SET text = 'changed' "
            "WHERE passage_id = 'passage-1'",
            "document passages are immutable",
        ),
        (
            "DELETE FROM document_passages WHERE passage_id = 'passage-1'",
            "document passages are immutable",
        ),
    ],
)
def test_source_evidence_is_immutable_at_database_boundary(
    tmp_path, statement, message
):
    store = make_migrated_store(tmp_path)
    seed_passage(store)

    with pytest.raises(IntegrityError, match=message):
        with store.transaction() as connection:
            connection.execute(statement)


def test_claim_invalid_stance_is_atomic(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_passage(store)

    with pytest.raises(IntegrityError):
        store.insert_claim(
            make_claim("claim-1"), evidence_ids=["passage-1"], stance="unclear"
        )

    assert store.get_claim("claim-1") is None


def test_claim_round_trips_canonical_decimal_and_normalized_utc(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_passage(store)
    local_time = utc(2026, 8, 24).astimezone(
        timezone(timedelta(hours=5, minutes=30))
    )
    claim = replace(
        make_claim("claim-1"),
        as_of=local_time,
        confidence=Decimal("0.9000"),
    )

    store.insert_claim(claim, evidence_ids=["passage-1"])

    loaded = store.get_claim("claim-1")
    assert loaded == replace(claim, as_of=utc(2026, 8, 24))
    with store.connect() as connection:
        stored = connection.execute(
            "SELECT confidence, as_of FROM claims WHERE claim_id = ?",
            ("claim-1",),
        ).fetchone()
    assert stored == ("0.9000", "2026-08-24T00:00:00.000000Z")


def test_claim_repository_rejects_naive_datetime_atomically(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_passage(store)
    claim = make_claim("claim-1")
    object.__setattr__(claim, "as_of", datetime(2026, 8, 24))

    with pytest.raises(ValueError, match="EvidenceClaim.as_of"):
        store.insert_claim(claim, evidence_ids=["passage-1"])

    assert store.get_claim("claim-1") is None


@pytest.mark.parametrize(
    ("statement", "parameters", "message"),
    [
        (
            "UPDATE claims SET text = ? WHERE claim_id = ?",
            ("rewritten", "claim-1"),
            "claim immutable fields cannot be updated",
        ),
        (
            "DELETE FROM claims WHERE claim_id = ?",
            ("claim-1",),
            "claims cannot be deleted",
        ),
        (
            "UPDATE claim_evidence SET stance = ? WHERE claim_id = ?",
            ("contradicts", "claim-1"),
            "claim evidence links are immutable",
        ),
        (
            "DELETE FROM claim_evidence WHERE claim_id = ?",
            ("claim-1",),
            "claim evidence links are immutable",
        ),
    ],
)
def test_claim_history_rejects_direct_mutation(
    tmp_path, statement, parameters, message
):
    store = make_migrated_store(tmp_path)
    seed_claim_with_evidence(store)

    with pytest.raises(IntegrityError, match=message):
        with store.transaction() as connection:
            connection.execute(statement, parameters)

    assert store.get_claim("claim-1") == make_claim("claim-1")


def test_claim_status_transition_uses_focused_store_method(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_claim_with_evidence(store)

    store.update_claim_status("claim-1", "contradicted")

    assert store.get_claim("claim-1") == replace(
        make_claim("claim-1"), status="contradicted"
    )


def test_source_content_hash_is_unique(tmp_path):
    store = make_migrated_store(tmp_path)
    store.insert_source_document(make_document())

    with pytest.raises(IntegrityError):
        store.insert_source_document(
            make_document("document-2", content_hash="sha256:document-1")
        )


def test_thesis_revisions_are_append_only(tmp_path):
    store = make_migrated_store(tmp_path)
    first = store.append_thesis_revision("semiconductors", "base thesis", [])
    second = store.append_thesis_revision("semiconductors", "revised thesis", [])
    assert second.version == first.version + 1
    assert store.list_thesis_revisions("semiconductors") == [first, second]


def test_thesis_revision_preserves_quotes_and_evidence(tmp_path):
    store = make_migrated_store(tmp_path)
    seed_passage(store)

    revision = store.append_thesis_revision(
        "semiconductor's tools",
        "Demand isn't uniformly strong.",
        ["passage-1"],
    )

    assert revision.industry_key == "semiconductor's tools"
    assert revision.thesis_text == "Demand isn't uniformly strong."
    assert revision.evidence_ids == ("passage-1",)
    assert store.list_thesis_revisions("semiconductor's tools") == [revision]


def test_thesis_missing_passage_does_not_create_revision(tmp_path):
    store = make_migrated_store(tmp_path)

    with pytest.raises(IntegrityError):
        store.append_thesis_revision("semiconductors", "base thesis", ["missing"])

    assert store.list_thesis_revisions("semiconductors") == []


@pytest.mark.parametrize(
    ("statement", "parameters", "message"),
    [
        (
            "UPDATE industry_theses SET thesis_text = ? WHERE thesis_id = ?",
            ("rewritten", 1),
            "industry thesis revisions are immutable",
        ),
        (
            "DELETE FROM industry_theses WHERE thesis_id = ?",
            (1,),
            "industry thesis revisions cannot be deleted",
        ),
        (
            "UPDATE thesis_evidence SET ordinal = ? WHERE thesis_id = ?",
            (2, 1),
            "thesis evidence links are immutable",
        ),
        (
            "DELETE FROM thesis_evidence WHERE thesis_id = ?",
            (1,),
            "thesis evidence links are immutable",
        ),
    ],
)
def test_thesis_history_rejects_direct_mutation(
    tmp_path, statement, parameters, message
):
    store = make_migrated_store(tmp_path)
    seed_passage(store)
    revision = store.append_thesis_revision(
        "semiconductors", "base thesis", ["passage-1"]
    )

    with pytest.raises(IntegrityError, match=message):
        with store.transaction() as connection:
            connection.execute(statement, parameters)

    assert store.list_thesis_revisions("semiconductors") == [revision]
