import json
from datetime import date

import pytest

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
