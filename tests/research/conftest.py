"""Shared helpers for deterministic research tests."""

import json
from datetime import datetime, timezone
from pathlib import Path


def utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    """Build a timezone-aware UTC datetime."""
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def fixture(name: str) -> str:
    """Read a UTF-8 research fixture by name."""
    return (Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8")


def fixture_json(name: str) -> dict[str, object]:
    """Read and parse a JSON research fixture."""
    return json.loads(fixture(name))
