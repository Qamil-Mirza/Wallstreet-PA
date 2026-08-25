"""Durable SQLite persistence for research evidence and revision history."""

import hashlib
import json
import re
import sqlite3
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from .models import ClaimKind, EvidenceClaim, SourceDocument


_MIGRATION_NAME = re.compile(r"^(?P<version>[0-9]+)_.+\.sql$")


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


class ResearchStore:
    """Connection-per-operation SQLite research store."""

    def __init__(self, database_path: Path) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path must be Path")
        self.database_path = database_path

    def connect(self) -> sqlite3.Connection:
        """Open one configured SQLite connection owned by the caller."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA busy_timeout = 5000")
        except BaseException:
            connection.close()
            raise
        return connection

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

    def _migration_files(self) -> tuple[Path, ...]:
        migration_dir = Path(__file__).with_name("migrations")
        return tuple(sorted(migration_dir.glob("[0-9]*_*.sql")))

    @staticmethod
    def _migration_version(path: Path) -> int:
        match = _MIGRATION_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Invalid migration filename: {path.name}")
        return int(match.group("version"))

    @staticmethod
    def _execute_script_atomically(
        connection: sqlite3.Connection, script: str
    ) -> None:
        """Execute a script without sqlite3.executescript's implicit COMMIT."""
        statement_lines: list[str] = []
        for line in script.splitlines(keepends=True):
            statement_lines.append(line)
            statement = "".join(statement_lines)
            if sqlite3.complete_statement(statement):
                if statement.strip():
                    connection.execute(statement)
                statement_lines.clear()
        if "".join(statement_lines).strip():
            raise sqlite3.OperationalError("incomplete SQL migration statement")

    def migrate(self) -> None:
        """Apply each packaged numbered migration exactly once, in order."""
        migration_files = self._migration_files()
        versions = [self._migration_version(path) for path in migration_files]
        if len(versions) != len(set(versions)):
            raise ValueError("Migration versions must be unique")

        with self.transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, "
                "name TEXT NOT NULL UNIQUE, "
                "applied_at TEXT NOT NULL)"
            )

        for path, version in zip(migration_files, versions, strict=True):
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
        if not passage_ids:
            raise sqlite3.IntegrityError("claims require at least one evidence passage")
        confidence = _decimal_text(claim.confidence)
        if not Decimal("0") <= claim.confidence <= Decimal("1"):
            raise ValueError("claim confidence must be between 0 and 1")

        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO claims ("
                "claim_id, entity_id, kind, text, as_of, confidence, status, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    claim.claim_id,
                    claim.entity_id,
                    claim.kind.value,
                    claim.text,
                    _utc_text(claim.as_of, "EvidenceClaim.as_of"),
                    confidence,
                    claim.status,
                    _utc_text(datetime.now(timezone.utc)),
                ),
            )
            connection.executemany(
                "INSERT INTO claim_evidence (claim_id, passage_id, stance) "
                "VALUES (?, ?, ?)",
                ((claim.claim_id, passage_id, stance) for passage_id in passage_ids),
            )

    def get_claim(self, claim_id: str) -> EvidenceClaim | None:
        """Load one typed claim, preserving exact Decimal representation."""
        connection = self.connect()
        try:
            row = connection.execute(
                "SELECT claim_id, entity_id, kind, text, as_of, confidence, status "
                "FROM claims WHERE claim_id = ?",
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
