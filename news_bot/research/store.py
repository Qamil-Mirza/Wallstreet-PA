"""Durable SQLite persistence for research evidence and revision history."""

import hashlib
import json
import re
import sqlite3
from importlib import resources
from importlib.resources.abc import Traversable
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from .models import (
    ClaimKind,
    EvidenceClaim,
    PortfolioSnapshot,
    Position,
    SourceDocument,
)


_MIGRATION_NAME = re.compile(r"^(?P<version>[0-9]+)_.+\.sql$")


class ResearchStoreError(RuntimeError):
    """Raised when the research store cannot safely initialize."""


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


@dataclass(frozen=True)
class ClaimLineageRecord:
    """One exact passage linked to a claim with an epistemic stance."""

    claim_id: str
    passage: DocumentPassageRecord
    stance: str


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
        passage_records = tuple(passages)
        if any(item.document_id != document.document_id for item in passage_records):
            raise sqlite3.IntegrityError("passage document_id does not match document")
        expected_ordinals = tuple(range(len(passage_records)))
        if tuple(item.ordinal for item in passage_records) != expected_ordinals:
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

        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT document_id, source_type, canonical_url, publisher, "
                "published_at, retrieved_at, content_hash, raw_content_path, "
                "extraction_status FROM source_documents WHERE content_hash = ?",
                (document.content_hash,),
            ).fetchone()
            if existing is not None:
                stored_passages = tuple(
                    connection.execute(
                        "SELECT passage_id, document_id, ordinal, text, content_hash, "
                        "locator_json FROM document_passages WHERE document_id = ? "
                        "ORDER BY ordinal",
                        (existing[0],),
                    ).fetchall()
                )
                if (
                    existing[0] == document.document_id
                    and stored_passages == passage_values
                ):
                    return
                raise sqlite3.IntegrityError(
                    "conflicting document content already exists"
                )

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
                "primary_passage_id, primary_supporting_claim_id, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    def list_claim_lineage(self, claim_id: str) -> tuple[ClaimLineageRecord, ...]:
        """Return exact passage lineage for one claim without rewriting evidence."""
        connection = self.connect()
        try:
            rows = connection.execute(
                "SELECT ce.claim_id, ce.stance, p.passage_id, p.document_id, "
                "p.ordinal, p.text, p.content_hash, p.locator_json "
                "FROM claim_evidence AS ce "
                "JOIN document_passages AS p ON p.passage_id = ce.passage_id "
                "WHERE ce.claim_id = ? ORDER BY p.document_id, p.ordinal, ce.stance",
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
                "SELECT supporting_claim_id FROM claim_dependencies "
                "WHERE claim_id = ? ORDER BY supporting_claim_id",
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
                "UPDATE claims SET status = ? WHERE claim_id = ?",
                (status, claim_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown claim_id: {claim_id}")

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
