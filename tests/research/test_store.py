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

from news_bot.research.models import SourceDocument
from news_bot.research.store import ResearchStore

from .conftest import make_claim, make_migrated_store, utc


REQUIRED_TABLES = {
    "schema_migrations",
    "portfolio_snapshots",
    "positions",
    "entities",
    "securities",
    "relationships",
    "source_documents",
    "document_passages",
    "claims",
    "claim_evidence",
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


def test_migration_is_idempotent_and_recorded_once(tmp_path):
    store = ResearchStore(tmp_path / "research.db")

    store.migrate()
    store.migrate()

    with store.connect() as connection:
        rows = connection.execute(
            "SELECT version, name, applied_at FROM schema_migrations"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0:2] == (1, "001_initial.sql")
    assert rows[0][2].endswith("Z")


def test_store_fails_closed_when_no_migrations_are_available(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path / "research.db")
    monkeypatch.setattr(store, "_migration_files", lambda: ())

    with pytest.raises(RuntimeError, match="No research store migrations found"):
        store.migrate()

    assert not store.database_path.exists()


def test_built_wheel_contains_discoverable_initial_migration(tmp_path):
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
    migration_name = "news_bot/research/migrations/001_initial.sql"
    with zipfile.ZipFile(wheel) as archive:
        assert migration_name in archive.namelist()

    resource_probe = (
        "import sys; from importlib import resources; "
        f"sys.path.insert(0, {str(wheel)!r}); "
        "migration = resources.files('news_bot.research').joinpath("
        "'migrations', '001_initial.sql'); "
        "assert 'CREATE TABLE portfolio_snapshots' in migration.read_text('utf-8')"
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
    bad_migration = tmp_path / "002_broken.sql"
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
    assert versions == [(1,)]


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


def test_claim_missing_passage_is_atomic(tmp_path):
    store = make_migrated_store(tmp_path)

    with pytest.raises(IntegrityError):
        store.insert_claim(make_claim("claim-1"), evidence_ids=["missing"])

    assert store.get_claim("claim-1") is None


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
