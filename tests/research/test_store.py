"""Integration tests for the durable SQLite research store."""

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
from news_bot.research.models import ClaimKind, SourceDocument
from news_bot.research.store import ResearchStore

from .conftest import fixture, make_claim, make_migrated_store, utc


REQUIRED_TABLES = {
    "schema_migrations",
    "portfolio_snapshots",
    "positions",
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


def test_store_migrates_empty_database(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    store.migrate()
    names = store.table_names()
    assert REQUIRED_TABLES <= names


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
    assert len(rows) == 4
    assert rows[0][0:2] == (1, "001_initial.sql")
    assert rows[1][0:2] == (2, "002_nullable_portfolio_freshness.sql")
    assert rows[2][0:2] == (3, "003_claim_dependencies.sql")
    assert rows[3][0:2] == (4, "004_entity_resolution.sql")
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

    assert versions == [(1,), (2,), (3,), (4,)]
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
    assert versions == [(1,), (2,), (3,), (4,)]
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

    assert versions == [(1,), (2,), (3,), (4,)]
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
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
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
    }
    with zipfile.ZipFile(wheel) as archive:
        assert migration_names <= set(archive.namelist())

    resource_probe = (
        "import sqlite3, sys, tempfile; from importlib import resources; "
        "from pathlib import Path; "
        f"sys.path.insert(0, {str(wheel)!r}); "
        "migration_dir = resources.files('news_bot.research').joinpath('migrations'); "
        "assert {p.name for p in migration_dir.iterdir()} >= "
        "{'001_initial.sql', '002_nullable_portfolio_freshness.sql', "
        "'003_claim_dependencies.sql', '004_entity_resolution.sql'}; "
        "from news_bot.research.store import ResearchStore; "
        "db = Path(tempfile.mkdtemp()) / 'research.db'; "
        "store = ResearchStore(db); store.migrate(); "
        "connection = sqlite3.connect(db); "
        "assert connection.execute('SELECT version FROM schema_migrations "
        "ORDER BY version').fetchall() == [(1,), (2,), (3,), (4,)]; connection.close()"
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
    bad_migration = tmp_path / "005_broken.sql"
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
    assert versions == [(1,), (2,), (3,), (4,)]


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
