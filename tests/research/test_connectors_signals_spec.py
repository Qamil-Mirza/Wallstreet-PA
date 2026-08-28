"""Spec regressions for official signal formats and real evidence lineage."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from news_bot.research.connectors.awards import (
    SBIRConfig,
    SBIRConnector,
    USASpendingConfig,
    USASpendingConnector,
)
from news_bot.research.connectors.base import ConnectorCheckpoint, ConnectorError
from news_bot.research.connectors.clinical_trials import (
    ClinicalTrialsConfig,
    ClinicalTrialsConnector,
)
from news_bot.research.connectors.form_d import FormDConfig, FormDConnector
from news_bot.research.connectors.manual_import import ManualImportConnector
from news_bot.research.connectors.uspto import USPTOConfig, USPTOConnector
from news_bot.research.evidence import EvidenceIngestor, EvidencePersistenceError
from news_bot.research.store import ResearchStore


FIXTURES = Path(__file__).parent / "fixtures"


def fixture_json(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def ingestor(tmp_path: Path) -> EvidenceIngestor:
    store = ResearchStore(tmp_path / "research.db")
    store.migrate()
    return EvidenceIngestor(store, tmp_path / "cache")


def official_form_d_zip(*, extra_column: bool = True) -> bytes:
    suffix = "\tDOCUMENTED_EXTRA" if extra_column else ""
    value_suffix = "\textra" if extra_column else ""
    tables = {
        "FORMDSUBMISSION.txt": (
            f"ACCESSIONNUMBER\tFILING_DATE{suffix}\n"
            f"0001234567-26-000001\t20-AUG-26{value_suffix}\n"
        ),
        "ISSUERS.txt": (
            f"ACCESSIONNUMBER\tISSUER_SEQ_KEY\tCIK\tENTITYNAME\tSTATEORCOUNTRY{suffix}\n"
            f"0001234567-26-000001\t1\t0001234567\tAtlas Robotics, Inc.\tCA{value_suffix}\n"
        ),
        "OFFERING.txt": (
            f"ACCESSIONNUMBER\tTOTALOFFERINGAMOUNT\tINDUSTRYGROUPTYPE{suffix}\n"
            f"0001234567-26-000001\t1250000.50\tTechnology{value_suffix}\n"
        ),
        "RECIPIENTS.txt": "ACCESSIONNUMBER\tRECIPIENT_SEQ_KEY\tRECIPIENTNAME\n0001234567-26-000001\t1\tBroker LLC\n",
        "RELATEDPERSONS.txt": "ACCESSIONNUMBER\tRELATEDPERSON_SEQ_KEY\tFIRSTNAME\n0001234567-26-000001\t1\tAvery\n",
        "SIGNATURES.txt": "ACCESSIONNUMBER\tSIGNATURE_SEQ_KEY\tSIGNATURENAME\n0001234567-26-000001\t1\tMorgan Lee\n",
    }
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in tables.items():
            archive.writestr(f"2026q3/{name.swapcase()}", content.encode("utf-8"))
    return result.getvalue()


def _without_synthetic_evidence(name: str):
    payload = fixture_json(name)
    if name == "uspto_sample.json":
        payload["patentFileWrapperDataBag"][0].pop("evidencePassageId", None)
    elif name == "sbir_sample.json":
        payload[0].pop("evidence_passage_id", None)
    elif name == "usaspending_sample.json":
        payload["results"][0].pop("evidence_passage_id", None)
    elif name == "clinical_trials_sample.json":
        payload["studies"][0].pop("evidencePassageId", None)
    return payload


@pytest.mark.parametrize(
    ("factory", "fixture_name", "source_field"),
    [
        (lambda sink: USPTOConnector(ingestor=sink), "uspto_sample.json", "applicationNumberText"),
        (lambda sink: SBIRConnector(ingestor=sink), "sbir_sample.json", "contract"),
        (
            lambda sink: USASpendingConnector(ingestor=sink),
            "usaspending_sample.json",
            "Award ID",
        ),
        (
            lambda sink: ClinicalTrialsConnector(ingestor=sink),
            "clinical_trials_sample.json",
            "protocolSection",
        ),
    ],
)
def test_documented_payload_creates_real_exact_evidence(
    tmp_path, factory, fixture_name, source_field
):
    sink = ingestor(tmp_path)
    payload = _without_synthetic_evidence(fixture_name)
    signal = factory(sink).parse(payload)[0]

    passages = sink.store.list_document_passages(signal.source_document_id)
    assert signal.source_document_id.startswith("document_")
    assert signal.evidence_passage_id == passages[0].passage_id
    assert source_field in passages[0].text
    assert "evidencePassageId" not in passages[0].text
    assert "evidence_passage_id" not in passages[0].text


def test_uspto_passage_is_exact_canonical_raw_record(tmp_path):
    sink = ingestor(tmp_path)
    payload = _without_synthetic_evidence("uspto_sample.json")
    row = payload["patentFileWrapperDataBag"][0]

    signal = USPTOConnector(ingestor=sink).parse(payload)[0]

    passage = sink.store.list_document_passages(signal.source_document_id)[0]
    assert passage.text == json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def test_official_form_d_six_table_archive_joins_and_persists(tmp_path):
    sink = ingestor(tmp_path)

    signal = FormDConnector(ingestor=sink).parse_zip(official_form_d_zip())[0]

    assert signal.company == "Atlas Robotics, Inc."
    assert str(signal.amount) == "1250000.50"
    assert signal.effective_date.isoformat() == "2026-08-20"
    assert signal.geography == "ca"
    assert signal.technology_terms == ("technology",)
    passages = sink.store.list_document_passages(signal.source_document_id)
    assert signal.evidence_passage_id == passages[0].passage_id
    assert '"FORMDSUBMISSION"' in passages[0].text
    assert '"ISSUERS"' in passages[0].text
    assert '"OFFERING"' in passages[0].text


def test_form_d_rejects_missing_duplicate_and_aggregate_bomb_tables(tmp_path):
    sink = ingestor(tmp_path)
    missing = io.BytesIO()
    with zipfile.ZipFile(missing, "w") as archive:
        archive.writestr("FORMDSUBMISSION.txt", "ACCESSIONNUMBER\tFILED\n")
    with pytest.raises(ConnectorError) as missing_error:
        FormDConnector(ingestor=sink).parse_zip(missing.getvalue())

    duplicate = io.BytesIO(official_form_d_zip())
    with zipfile.ZipFile(duplicate, "a") as archive:
        archive.writestr("duplicate/ISSUERS.TXT", "ACCESSIONNUMBER\tENTITYNAME\tSTATEORCOUNTRY\n")
    with pytest.raises(ConnectorError) as duplicate_error:
        FormDConnector(ingestor=sink).parse_zip(duplicate.getvalue())

    with pytest.raises(ConnectorError) as bomb_error:
        FormDConnector(
            FormDConfig(max_total_uncompressed_bytes=64), ingestor=sink
        ).parse_zip(official_form_d_zip())

    assert missing_error.value.diagnostic_code == "invalid_archive"
    assert duplicate_error.value.diagnostic_code == "invalid_archive"
    assert bomb_error.value.diagnostic_code == "archive_total_too_large"


def test_form_d_supports_multiple_official_issuer_rows(tmp_path):
    source = zipfile.ZipFile(io.BytesIO(official_form_d_zip()))
    rebuilt = io.BytesIO()
    with source, zipfile.ZipFile(rebuilt, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            content = source.read(item)
            if Path(item.filename).stem.upper() == "ISSUERS":
                content += (
                    b"0001234567-26-000001\t2\t0007654321\tAtlas Controls LLC"
                    b"\tNY\textra\n"
                )
            target.writestr(item.filename, content)

    signals = FormDConnector(ingestor=ingestor(tmp_path)).parse_zip(
        rebuilt.getvalue()
    )

    assert [signal.company for signal in signals] == [
        "Atlas Robotics, Inc.",
        "Atlas Controls LLC",
    ]
    assert len({signal.source_locator for signal in signals}) == 2


def test_manual_import_persists_row_instead_of_accepting_lineage_id(tmp_path):
    sink = ingestor(tmp_path)
    content = (
        "company,profile_date,geography,technology_terms,source_locator,stage,amount\n"
        "Atlas Robotics Inc.,2026-08-15,California,Robotic Actuators|Harmonic Drives,"
        "licensed-export:atlas-2026-08-15,Series A,5000000\n"
    )

    signal = ManualImportConnector(ingestor=sink).parse(content)[0]

    passages = sink.store.list_document_passages(signal.source_document_id)
    assert signal.evidence_passage_id == passages[0].passage_id
    assert "Atlas Robotics Inc." in passages[0].text


def test_evidence_failure_is_redacted_and_prevents_form_d_batch(tmp_path, monkeypatch):
    sink = ingestor(tmp_path)

    def fail(_source):
        raise EvidencePersistenceError("private-record-value")

    monkeypatch.setattr(sink, "ingest", fail)
    checkpoint = ConnectorCheckpoint("form_d", cursor="2026Q2")
    connector = FormDConnector(
        ingestor=sink,
        archive_loader=lambda incoming: (official_form_d_zip(), "2026Q3"),
    )

    with pytest.raises(ConnectorError) as error:
        connector.fetch(checkpoint)

    assert error.value.diagnostic_code == "evidence_persistence_failed"
    assert error.value.retryable is False
    assert "private-record-value" not in str(error.value)
    with sink.store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM source_documents").fetchone()[0] == 0


def test_sbir_bulk_fallback_is_size_bounded_before_json_parse(tmp_path):
    connector = SBIRConnector(
        SBIRConfig(max_response_bytes=32),
        ingestor=ingestor(tmp_path),
        bulk_loader=lambda: b"private-bulk-value" * 100,
    )

    with pytest.raises(ConnectorError) as error:
        connector._bulk_batch(ConnectorCheckpoint("sbir"))

    assert error.value.diagnostic_code == "bulk_too_large"
    assert "private-bulk-value" not in str(error.value)


@pytest.mark.parametrize(
    "build",
    [
        lambda: USPTOConfig(min_interval_seconds=0),
        lambda: SBIRConfig(min_interval_seconds=0),
        lambda: USASpendingConfig(min_interval_seconds=0),
        lambda: ClinicalTrialsConfig(min_interval_seconds=0),
    ],
)
def test_api_pacing_must_be_strictly_positive(build):
    with pytest.raises(ValueError, match="positive"):
        build()
