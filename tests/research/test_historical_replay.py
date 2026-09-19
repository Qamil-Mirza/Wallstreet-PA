import json
from datetime import date, datetime, timedelta, timezone

import pytest

from news_bot.research.store import ResearchStore

from tests.research.golden_harness import (
    ReplayDateError,
    SOURCE_PACKET,
    run_historical_replay,
)


def test_replay_cannot_use_future_document(tmp_path):
    result = run_historical_replay(tmp_path, as_of=date(2025, 10, 1))
    assert all(
        document.published_at.date() <= result.as_of
        for document in result.documents
    )
    assert {document.document_id for document in result.documents} == {
        "doc-nvda-2025-10k",
        "doc-tsmc-2024-annual",
    }


def test_replay_fails_closed_when_source_date_is_unknown(tmp_path):
    packet = tmp_path / "packet"
    packet.mkdir()
    manifest = json.loads(
        (SOURCE_PACKET / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["documents"] = ["unknown.json"]
    (packet / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    unknown = json.loads(
        (SOURCE_PACKET / "nvidia_2025_10k.json").read_text(encoding="utf-8")
    )
    unknown["document_id"] = "doc-unknown-date"
    unknown["published_at"] = None
    (packet / "unknown.json").write_text(json.dumps(unknown), encoding="utf-8")

    with pytest.raises(ReplayDateError, match="no trustworthy published_at"):
        run_historical_replay(
            tmp_path / "run",
            as_of=date(2025, 10, 1),
            packet_root=packet,
        )


def test_production_replay_query_includes_exact_boundary_and_known_retrieval(
    tmp_path,
):
    run_historical_replay(tmp_path, as_of=date(2025, 10, 1))
    store = ResearchStore(tmp_path / "replay.db")
    boundary = store.list_documents_as_of(
        datetime(2025, 10, 1, 12, tzinfo=timezone.utc)
    )
    assert {document.document_id for document in boundary} == {
        "doc-nvda-2025-10k",
        "doc-tsmc-2024-annual",
    }


def test_production_replay_query_fails_closed_on_corrupt_stored_date(tmp_path):
    store = ResearchStore(tmp_path / "replay.db")
    store.migrate()
    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO source_documents (document_id, source_type, "
            "canonical_url, publisher, published_at, retrieved_at, "
            "content_hash, raw_content_path, extraction_status, created_at) "
            "VALUES ('doc-corrupt', 'filing', 'https://example.com/corrupt', "
            "'Fixture', 'unknown', '2025-01-01T00:00:00.000000Z', ?, NULL, "
            "'complete', '2025-01-01T00:00:00.000000Z')",
            ("f" * 64,),
        )

    with pytest.raises(ValueError, match="publication date"):
        store.list_documents_as_of(datetime(2025, 10, 1, tzinfo=timezone.utc))


@pytest.mark.parametrize(
    "published_at,retrieved_at",
    (
        ("2025-01-01 00:00:00", "2025-01-01T00:00:00.000000Z"),
        ("2025-01-01T00:00:00.000000Z", "2025-01-01T01:00:00+01:00"),
    ),
)
def test_production_replay_query_rejects_noncanonical_persisted_utc_text(
    tmp_path,
    published_at,
    retrieved_at,
):
    store = ResearchStore(tmp_path / "replay.db")
    store.migrate()
    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO source_documents (document_id, source_type, "
            "canonical_url, publisher, published_at, retrieved_at, "
            "content_hash, raw_content_path, extraction_status, created_at) "
            "VALUES ('doc-noncanonical', 'filing', "
            "'https://example.com/noncanonical', 'Fixture', ?, ?, ?, NULL, "
            "'complete', '2025-01-01T00:00:00.000000Z')",
            (published_at, retrieved_at, "e" * 64),
        )

    with pytest.raises(ValueError, match="canonical UTC"):
        store.list_documents_as_of(datetime(2025, 10, 1, tzinfo=timezone.utc))


def test_production_replay_query_excludes_document_retrieved_after_cutoff(tmp_path):
    store = ResearchStore(tmp_path / "replay.db")
    store.migrate()
    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO source_documents (document_id, source_type, "
            "canonical_url, publisher, published_at, retrieved_at, "
            "content_hash, raw_content_path, extraction_status, created_at) "
            "VALUES ('doc-late-retrieval', 'filing', "
            "'https://example.com/late', 'Fixture', "
            "'2025-01-01T00:00:00.000000Z', "
            "'2025-10-01T12:00:00.000001Z', ?, NULL, 'complete', "
            "'2025-01-01T00:00:00.000000Z')",
            ("d" * 64,),
        )

    documents = store.list_documents_as_of(
        datetime(2025, 10, 1, 12, tzinfo=timezone.utc)
    )
    assert documents == ()


@pytest.mark.parametrize(
    "cutoff",
    (
        datetime(2025, 10, 1),
        datetime(2025, 10, 1, tzinfo=timezone(timedelta(hours=1))),
    ),
)
def test_production_replay_query_requires_aware_utc_cutoff(tmp_path, cutoff):
    store = ResearchStore(tmp_path / "replay.db")
    store.migrate()
    with pytest.raises(ValueError, match="aware UTC"):
        store.list_documents_as_of(cutoff)
