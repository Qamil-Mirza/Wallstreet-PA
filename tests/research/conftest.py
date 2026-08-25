"""Shared helpers for deterministic research tests."""

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from news_bot.research.models import ClaimKind, EvidenceClaim
from news_bot.research.store import ResearchStore


def utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    """Build a timezone-aware UTC datetime."""
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def fixture(name: str) -> str:
    """Read a UTF-8 research fixture by name."""
    return (Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8")


def fixture_json(name: str) -> dict[str, object]:
    """Read and parse a JSON research fixture."""
    return json.loads(fixture(name))


def make_migrated_store(tmp_path: Path) -> ResearchStore:
    """Create a migrated store backed by a deterministic temporary path."""
    store = ResearchStore(tmp_path / "research.db")
    store.migrate()
    return store


@pytest.fixture
def migrated_store(tmp_path: Path) -> ResearchStore:
    """Provide a fresh migrated SQLite research store."""
    return make_migrated_store(tmp_path)


def make_claim(claim_id: str) -> EvidenceClaim:
    """Build a fixed evidence claim for persistence tests."""
    return EvidenceClaim(
        claim_id=claim_id,
        entity_id=None,
        kind=ClaimKind.FACT,
        text="Revenue increased year over year.",
        as_of=utc(2026, 8, 24),
        confidence=Decimal("0.90"),
        status="active",
    )
