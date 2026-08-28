"""Canonical, content-addressed evidence ingestion and claim traceability."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import sqlite3
import stat
import tempfile
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

from .models import ClaimKind, EvidenceClaim, SourceDocument
from .store import DocumentPassageRecord, ResearchStore


_TRACKING_PARAMETERS = {"dclid", "fbclid", "gclid", "mc_cid", "mc_eid"}
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_MALFORMED_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


class EvidenceError(RuntimeError):
    """Base error for safe evidence-ingestion failures."""


class EvidenceValidationError(EvidenceError, ValueError):
    """Raised when document metadata or content is unsafe."""


class EvidenceCacheError(EvidenceError):
    """Raised when immutable cached content cannot be trusted."""


class EvidencePolicyError(EvidenceError, ValueError):
    """Raised when claim lineage violates the evidence policy."""


class EvidencePersistenceError(EvidenceError):
    """Raised when canonical evidence cannot be persisted consistently."""


def _valid_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceValidationError(f"{field_name} must be nonblank text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise EvidenceValidationError(f"{field_name} must be valid UTF-8") from None


def _aware(value: datetime, field_name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise EvidenceValidationError(f"{field_name} must be timezone-aware")


def _valid_lineage_identifier(value: str, field_name: str) -> None:
    _valid_text(value, field_name)
    if (
        len(value) > 256
        or any(character.isspace() for character in value)
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise EvidenceValidationError("lineage identifier is invalid")


@dataclass(frozen=True)
class DocumentInput:
    """Validated source material accepted at the ingestion trust boundary."""

    source_type: str
    url: str
    publisher: str
    published_at: datetime
    retrieved_at: datetime
    content: str | bytes

    def __post_init__(self) -> None:
        _valid_text(self.source_type, "source_type")
        _valid_text(self.url, "url")
        _valid_text(self.publisher, "publisher")
        _aware(self.published_at, "published_at")
        _aware(self.retrieved_at, "retrieved_at")
        if not isinstance(self.content, (str, bytes)):
            raise EvidenceValidationError("content must be UTF-8 text or bytes")
        canonicalize_url(self.url)
        canonicalize_content(self.content)


@dataclass(frozen=True)
class EvidencePassage:
    """A stable, exact span in a canonical source document."""

    passage_id: str
    document_id: str
    ordinal: int
    text: str
    content_hash: str
    start_offset: int
    end_offset: int

    def __post_init__(self) -> None:
        _valid_lineage_identifier(self.passage_id, "passage_id")
        _valid_lineage_identifier(self.document_id, "document_id")
        _valid_text(self.text, "passage text")
        if not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise EvidenceValidationError("passage ordinal must be non-negative")
        if re.fullmatch(r"[0-9a-f]{64}", self.content_hash) is None:
            raise EvidenceValidationError("passage content_hash must be SHA-256")
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.content_hash:
            raise EvidenceValidationError("passage content_hash does not match text")
        if (
            not isinstance(self.start_offset, int)
            or not isinstance(self.end_offset, int)
            or self.start_offset < 0
            or self.end_offset - self.start_offset != len(self.text)
        ):
            raise EvidenceValidationError("passage offsets must exactly span text")


@dataclass(frozen=True)
class IngestedDocument:
    """Canonical document identity plus its citation-ready passages."""

    document_id: str
    canonical_url: str
    content_hash: str
    raw_content_path: Path
    passages: tuple[EvidencePassage, ...]

    def __post_init__(self) -> None:
        _valid_lineage_identifier(self.document_id, "document_id")
        if canonicalize_url(self.canonical_url) != self.canonical_url:
            raise EvidenceValidationError("canonical_url must already be canonical")
        if re.fullmatch(r"[0-9a-f]{64}", self.content_hash) is None:
            raise EvidenceValidationError("content_hash must be SHA-256")
        if not isinstance(self.raw_content_path, Path):
            raise EvidenceValidationError("raw_content_path must be Path")
        if not isinstance(self.passages, tuple) or any(
            item.document_id != self.document_id for item in self.passages
        ):
            raise EvidenceValidationError("passages must belong to the document")


@dataclass(frozen=True)
class ClaimLineage:
    """One immutable passage relationship supporting or contradicting a claim."""

    claim_id: str
    passage_id: str
    document_id: str
    text: str
    start_offset: int
    end_offset: int
    stance: str

    def __post_init__(self) -> None:
        for field_name in ("claim_id", "passage_id", "document_id"):
            _valid_lineage_identifier(getattr(self, field_name), field_name)
        _valid_text(self.text, "text")
        if self.stance not in {"supports", "contradicts"}:
            raise EvidenceValidationError("lineage stance is invalid")
        if (
            not isinstance(self.start_offset, int)
            or not isinstance(self.end_offset, int)
            or self.start_offset < 0
            or self.end_offset - self.start_offset != len(self.text)
        ):
            raise EvidenceValidationError("lineage offsets must exactly span text")


def _contains_control(value: str) -> bool:
    return any(unicodedata.category(character) == "Cc" for character in value)


def _contains_content_control(value: str) -> bool:
    return any(
        character != "\n" and unicodedata.category(character) == "Cc"
        for character in value
    )


def canonicalize_url(value: str) -> str:
    """Return a stable HTTP(S) URL without credentials or tracking metadata."""
    try:
        if (
            not isinstance(value, str)
            or any(character.isspace() for character in value)
            or _contains_control(value)
            or _MALFORMED_PERCENT_ESCAPE.search(value)
            or _contains_control(unquote(value, errors="strict"))
        ):
            raise ValueError
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError
        raw_hostname = parsed.hostname.rstrip(".")
        if not raw_hostname:
            raise ValueError
        try:
            address = ipaddress.ip_address(raw_hostname)
        except ValueError:
            if ":" in raw_hostname:
                raise ValueError
            hostname = (
                raw_hostname.encode("idna").decode("ascii").lower().rstrip(".")
            )
            labels = hostname.split(".")
            if (
                not hostname
                or len(hostname) > 253
                or any(_HOST_LABEL.fullmatch(label) is None for label in labels)
                or (re.fullmatch(r"[0-9.]+", hostname) and len(labels) > 1)
            ):
                raise ValueError
        else:
            hostname = address.compressed.lower()
        port = parsed.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError
        if ":" in hostname:
            hostname = f"[{hostname}]"
        default_port = (parsed.scheme.lower() == "http" and port == 80) or (
            parsed.scheme.lower() == "https" and port == 443
        )
        netloc = hostname if port is None or default_port else f"{hostname}:{port}"
        query_items = []
        for key, item in parse_qsl(parsed.query, keep_blank_values=True):
            if _contains_control(key) or _contains_control(item):
                raise ValueError
            lowered = key.lower()
            if lowered.startswith("utm_") or lowered in _TRACKING_PARAMETERS:
                continue
            query_items.append((key, item))
        query = urlencode(sorted(query_items), doseq=True)
        return urlunsplit(
            (parsed.scheme.lower(), netloc, parsed.path or "/", query, "")
        )
    except (UnicodeError, ValueError):
        raise EvidenceValidationError("document URL is invalid") from None


def canonicalize_content(value: str | bytes) -> str:
    """Decode and normalize source text while preserving paragraph boundaries."""
    try:
        text = (
            value.decode("utf-8", errors="strict")
            if isinstance(value, bytes)
            else value
        )
        text.encode("utf-8", errors="strict")
    except (AttributeError, UnicodeError):
        raise EvidenceValidationError("document content must be valid UTF-8") from None
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "\n".join(
        _HORIZONTAL_WHITESPACE.sub(" ", line).strip()
        for line in text.split("\n")
    )
    text = _EXCESS_BLANK_LINES.sub("\n\n", text).strip("\n")
    if not text or _contains_content_control(text):
        raise EvidenceValidationError("document content is invalid")
    return text


def _split_passages(
    text: str, document_id: str, max_chars: int = 1200
) -> tuple[EvidencePassage, ...]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text):
        start, end = match.span()
        while end - start > max_chars:
            boundary = text.rfind(" ", start, start + max_chars + 1)
            if boundary <= start:
                boundary = start + max_chars
            spans.append((start, boundary))
            start = boundary
            while start < end and text[start] == " ":
                start += 1
        if start < end:
            spans.append((start, end))
    passages: list[EvidencePassage] = []
    for ordinal, (start, end) in enumerate(spans):
        passage_text = text[start:end]
        digest = hashlib.sha256(passage_text.encode("utf-8")).hexdigest()
        identity = hashlib.sha256(
            f"{document_id}:{start}:{end}:{digest}".encode("ascii")
        ).hexdigest()
        passages.append(
            EvidencePassage(
                passage_id=f"passage_{identity}",
                document_id=document_id,
                ordinal=ordinal,
                text=passage_text,
                content_hash=digest,
                start_offset=start,
                end_offset=end,
            )
        )
    return tuple(passages)


class EvidenceIngestor:
    """Persist canonical source material and expose exact claim lineage."""

    def __init__(self, store: ResearchStore, cache_dir: Path | None = None) -> None:
        if not isinstance(store, ResearchStore):
            raise TypeError("store must be ResearchStore")
        if cache_dir is not None and not isinstance(cache_dir, Path):
            raise TypeError("cache_dir must be Path")
        self.store = store
        self.cache_dir = cache_dir or store.database_path.parent / "source_cache"

    def _cache(self, content_hash: str, content: bytes) -> Path:
        try:
            return self._cache_unchecked(content_hash, content)
        except EvidenceCacheError:
            raise
        except OSError:
            raise EvidenceCacheError("source content could not be cached") from None

    def _cache_unchecked(self, content_hash: str, content: bytes) -> Path:
        """Perform cache I/O; the public helper redacts all OS failures."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / content_hash
        if target.exists() or target.is_symlink():
            self._fsync_cache_directory()
            return self._verify_cached(target, content)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".evidence-", dir=self.cache_dir
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                self._fsync_cache_directory()
                return self._verify_cached(target, content)
            try:
                self._fsync_cache_directory()
            except OSError:
                target.unlink(missing_ok=True)
                raise
            return self._verify_cached(target, content)
        finally:
            temporary.unlink(missing_ok=True)

    def _fsync_cache_directory(self) -> None:
        """Durably publish cache directory entries before evidence persistence."""
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(self.cache_dir, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _verify_cached(target: Path, expected: bytes) -> Path:
        try:
            metadata = target.lstat()
            if not stat.S_ISREG(metadata.st_mode) or target.read_bytes() != expected:
                raise EvidenceCacheError(
                    "cached source content failed integrity validation"
                )
        except EvidenceCacheError:
            raise
        except OSError:
            raise EvidenceCacheError(
                "cached source content failed integrity validation"
            ) from None
        return target

    def ingest(self, source: DocumentInput) -> IngestedDocument:
        """Canonicalize and atomically persist one idempotent source document."""
        return self.ingest_batch((source,))[0]

    def ingest_batch(
        self, sources: Sequence[DocumentInput]
    ) -> tuple[IngestedDocument, ...]:
        """Cache then atomically persist an ordered source-document batch.

        Cache publication intentionally precedes the database transaction.
        Content-addressed cache artifacts are immutable and reusable if the
        database transaction subsequently rolls back.
        """
        source_records = tuple(sources)
        if any(not isinstance(source, DocumentInput) for source in source_records):
            raise TypeError("sources must contain only DocumentInput records")
        prepared = []
        cached_paths: dict[str, Path] = {}
        for source in source_records:
            canonical_url = canonicalize_url(source.url)
            text = canonicalize_content(source.content)
            encoded = text.encode("utf-8")
            digest = hashlib.sha256(encoded).hexdigest()
            document_id = f"document_{digest}"
            cached_path = self._cache(digest, encoded)
            cached_paths[digest] = cached_path
            passages = _split_passages(text, document_id)
            document = SourceDocument(
                document_id=document_id,
                source_type=source.source_type,
                canonical_url=canonical_url,
                publisher=source.publisher,
                published_at=source.published_at,
                retrieved_at=source.retrieved_at,
                content_hash=digest,
                raw_content_path=str(cached_path),
                extraction_status="complete",
            )
            prepared.append(
                (
                    document,
                    tuple(
                        DocumentPassageRecord(
                            passage_id=item.passage_id,
                            document_id=item.document_id,
                            ordinal=item.ordinal,
                            text=item.text,
                            content_hash=item.content_hash,
                            start_offset=item.start_offset,
                            end_offset=item.end_offset,
                        )
                        for item in passages
                    ),
                )
            )
        try:
            stored_records = self.store.insert_documents_with_passages(prepared)
        except sqlite3.Error:
            raise EvidencePersistenceError(
                "canonical evidence could not be persisted"
            ) from None
        return tuple(
            IngestedDocument(
                document_id=document.document_id,
                canonical_url=document.canonical_url,
                content_hash=document.content_hash,
                raw_content_path=Path(
                    document.raw_content_path or cached_paths[document.content_hash]
                ),
                passages=tuple(
                    EvidencePassage(
                        passage_id=item.passage_id,
                        document_id=item.document_id,
                        ordinal=item.ordinal,
                        text=item.text,
                        content_hash=item.content_hash,
                        start_offset=item.start_offset,
                        end_offset=item.end_offset,
                    )
                    for item in passages
                ),
            )
            for document, passages in stored_records
        )

    def add_claim(
        self,
        text: str,
        kind: ClaimKind,
        passage_ids: Sequence[str],
        *,
        supporting_claim_ids: Sequence[str] = (),
        contradicting_passage_ids: Sequence[str] = (),
        entity_id: str | None = None,
        as_of: datetime | None = None,
        confidence: Decimal = Decimal("0.5"),
    ) -> EvidenceClaim:
        """Create one typed claim backed by exact stored passages."""
        _valid_text(text, "claim text")
        if not isinstance(kind, ClaimKind):
            raise EvidencePolicyError("claim kind is invalid")
        ids = tuple(passage_ids)
        dependencies = tuple(supporting_claim_ids)
        contradictions = tuple(contradicting_passage_ids)
        for identifier in ids:
            _valid_lineage_identifier(identifier, "passage_id")
        for identifier in dependencies:
            _valid_lineage_identifier(identifier, "supporting_claim_id")
        for identifier in contradictions:
            _valid_lineage_identifier(identifier, "contradicting_passage_id")
        if kind is ClaimKind.INFERENCE:
            if not ids and not dependencies:
                raise EvidencePolicyError(
                    "inference requires a supporting claim or passage"
                )
        elif not ids:
            raise EvidencePolicyError(
                "fact, guidance, and estimate claims require passage evidence"
            )
        claim_id = self._claim_id(kind, text, ids, dependencies, contradictions)
        if claim_id in dependencies:
            raise EvidencePolicyError("a claim cannot depend on itself")
        claim = EvidenceClaim(
            claim_id=claim_id,
            entity_id=entity_id,
            kind=kind,
            text=text,
            as_of=as_of or datetime.now(timezone.utc),
            confidence=confidence,
            status="active",
        )
        try:
            self.store.insert_claim_with_lineage(
                claim,
                passage_links=(
                    *((passage_id, "supports") for passage_id in ids),
                    *((passage_id, "contradicts") for passage_id in contradictions),
                ),
                supporting_claim_ids=dependencies,
            )
        except sqlite3.IntegrityError:
            raise EvidencePolicyError("claim lineage references are invalid") from None
        return claim

    @staticmethod
    def _claim_id(
        kind: ClaimKind,
        text: str,
        passage_ids: Sequence[str],
        supporting_claim_ids: Sequence[str],
        contradicting_passage_ids: Sequence[str],
    ) -> str:
        identity_payload = json.dumps(
            {
                "contradicting_passage_ids": sorted(contradicting_passage_ids),
                "kind": kind.value,
                "passage_ids": sorted(passage_ids),
                "supporting_claim_ids": sorted(supporting_claim_ids),
                "text": text,
                "version": 1,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        identity = hashlib.sha256(identity_payload).hexdigest()
        return f"claim_{identity}"

    def lineage(self, claim_id: str) -> tuple[ClaimLineage, ...]:
        """Resolve a claim to the exact canonical passages used to create it."""
        _valid_lineage_identifier(claim_id, "claim_id")
        return tuple(
            ClaimLineage(
                claim_id=item.claim_id,
                passage_id=item.passage.passage_id,
                document_id=item.passage.document_id,
                text=item.passage.text,
                start_offset=item.passage.start_offset,
                end_offset=item.passage.end_offset,
                stance=item.stance,
            )
            for item in self.store.list_claim_lineage(claim_id)
        )

    def supporting_claim_ids(self, claim_id: str) -> tuple[str, ...]:
        """Return the immutable claims used to support an inference."""
        _valid_lineage_identifier(claim_id, "claim_id")
        return self.store.list_supporting_claim_ids(claim_id)
