"""Integration tests for the durable SQLite research store."""

import sqlite3
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
