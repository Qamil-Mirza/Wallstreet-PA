"""Contracts for emerging-company public-signal connectors."""

from __future__ import annotations

import dataclasses
import io
import json
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from news_bot.research.connectors.awards import (
    SBIRConfig,
    SBIRConnector,
    USASpendingConfig,
    USASpendingConnector,
)
from news_bot.research.connectors.base import (
    ConnectorCheckpoint,
    ConnectorError,
    ConnectorStatus,
    EmergingSignal,
    SignalResearchConnector,
    SignalConnectorBatch,
)
from news_bot.research.connectors.clinical_trials import (
    ClinicalTrialsConfig,
    ClinicalTrialsConnector,
)
from news_bot.research.connectors.form_d import FormDConfig, FormDConnector
from news_bot.research.connectors.manual_import import ManualImportConnector
from news_bot.research.connectors.uspto import USPTOConfig, USPTOConnector


FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("connector", "fixture_name", "expected_signal"),
    [
        (FormDConnector(), "form_d_sample.tsv", "funding"),
        (USPTOConnector(), "uspto_sample.json", "patent"),
        (SBIRConnector(), "sbir_sample.json", "grant"),
        (USASpendingConnector(), "usaspending_sample.json", "contract"),
        (ClinicalTrialsConnector(), "clinical_trials_sample.json", "clinical_trial"),
        (
            ManualImportConnector(),
            "manual_company_import.csv",
            "private_market_profile",
        ),
    ],
)
def test_public_signal_normalization(connector, fixture_name, expected_signal):
    records = connector.parse(fixture(fixture_name))

    assert records[0].signal_type == expected_signal
    assert records[0].source_document_id
    assert records[0].source_locator
    assert records[0].evidence_passage_id


def test_emerging_signal_is_frozen_validated_and_normalized():
    signal = EmergingSignal(
        source_document_id="doc-1",
        source_locator="source:item-1",
        company="  Atlas\u00a0Robotics  ",
        signal_type="patent",
        amount=Decimal("125.50"),
        stage="  Phase I  ",
        effective_date=date(2026, 8, 19),
        geography="  CALIFORNIA ",
        technology_terms=(" Robotic  Actuators ", "HARMONIC Drives"),
        evidence_passage_id="passage-1",
    )

    assert signal.company == "Atlas Robotics"
    assert signal.geography == "california"
    assert signal.technology_terms == ("robotic actuators", "harmonic drives")
    with pytest.raises(dataclasses.FrozenInstanceError):
        signal.company = "Changed"
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(signal, amount=Decimal("NaN"))
    with pytest.raises(ValueError, match="evidence_passage_id"):
        dataclasses.replace(signal, evidence_passage_id=" ")


def test_source_ids_are_deterministic_and_source_identity_changes_them():
    connector = FormDConnector()
    first = connector.parse(fixture("form_d_sample.tsv"))[0]
    second = connector.parse(fixture("form_d_sample.tsv"))[0]
    changed = fixture("form_d_sample.tsv").replace(
        "0001234567-26-000001", "0001234567-26-000002"
    )

    assert first.source_document_id == second.source_document_id
    assert first.source_document_id != connector.parse(changed)[0].source_document_id


def _zip_bytes(name: str, content: bytes, *, declared_size: int | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo(name)
        if declared_size is not None:
            info.file_size = declared_size
        archive.writestr(info, content)
    return buffer.getvalue()


def test_form_d_accepts_one_bounded_tsv_zip_member():
    connector = FormDConnector(FormDConfig(max_archive_bytes=20_000, max_member_bytes=5_000))

    records = connector.parse_zip(
        _zip_bytes("2026Q3/FORM_D.tsv", fixture("form_d_sample.tsv").encode())
    )

    assert records[0].amount == Decimal("1250000.50")


@pytest.mark.parametrize("member", ["../escape.tsv", "/absolute.tsv", "safe.csv"])
def test_form_d_rejects_unsafe_or_wrong_archive_member(member):
    with pytest.raises(ConnectorError) as error:
        FormDConnector().parse_zip(
            _zip_bytes(member, fixture("form_d_sample.tsv").encode())
        )

    assert error.value.retryable is False
    assert error.value.diagnostic_code == "invalid_archive"


def test_form_d_rejects_archive_bombs_and_invalid_utf8_without_echoing_data():
    connector = FormDConnector(FormDConfig(max_archive_bytes=20_000, max_member_bytes=32))
    secret = b"private-token-123\xff"

    with pytest.raises(ConnectorError) as too_large:
        connector.parse_zip(_zip_bytes("FORM_D.tsv", b"a" * 1_000))
    with pytest.raises(ConnectorError) as invalid_encoding:
        FormDConnector().parse_zip(_zip_bytes("FORM_D.tsv", secret))

    assert too_large.value.diagnostic_code == "archive_member_too_large"
    assert invalid_encoding.value.diagnostic_code == "invalid_encoding"
    assert "private-token" not in str(invalid_encoding.value)


def test_form_d_and_manual_import_require_exact_schema():
    with pytest.raises(ConnectorError) as form_error:
        FormDConnector().parse("ACCESSIONNUMBER\tENTITYNAME\n1\tAtlas\n")
    with pytest.raises(ConnectorError) as manual_error:
        ManualImportConnector().parse(
            fixture("manual_company_import.csv").replace(",amount\n", ",amount,secret\n")
        )

    assert form_error.value.diagnostic_code == "invalid_schema"
    assert manual_error.value.diagnostic_code == "invalid_schema"


@pytest.mark.parametrize("payload", ["=2+2", "+cmd", "-10", "@SUM(A1)"])
def test_manual_import_rejects_spreadsheet_formulas(payload):
    content = fixture("manual_company_import.csv").replace(
        "Atlas Robotics Inc.", payload
    )

    with pytest.raises(ConnectorError) as error:
        ManualImportConnector().parse(content)

    assert error.value.diagnostic_code == "unsafe_cell"
    assert payload not in str(error.value)


def test_manual_import_is_size_and_utf8_bounded():
    connector = ManualImportConnector(max_bytes=32)

    with pytest.raises(ConnectorError) as too_large:
        connector.parse(fixture("manual_company_import.csv"))
    with pytest.raises(ConnectorError) as encoding:
        ManualImportConnector().parse(b"company\xff")

    assert too_large.value.diagnostic_code == "input_too_large"
    assert encoding.value.diagnostic_code == "invalid_encoding"


class Response:
    def __init__(self, payload, *, url, status=200, headers=None, history=()):
        self.payload = payload
        self.url = url
        self.status_code = status
        self.headers = headers or {"Content-Type": "application/json"}
        self.history = history
        self.closed = False

    def iter_content(self, chunk_size=65_536):
        raw = json.dumps(self.payload).encode("utf-8")
        yield from (
            raw[offset : offset + chunk_size]
            for offset in range(0, len(raw), chunk_size)
        )

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False
        self.trust_env = True

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.responses.pop(0)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def test_uspto_uses_authenticated_safe_transport_and_closes_response():
    response = Response(
        json.loads(fixture("uspto_sample.json")),
        url="https://api.uspto.gov/api/v1/patent/applications/search",
    )
    session = Session([response])
    connector = USPTOConnector(
        USPTOConfig(api_key="private-uspto-key"), session=session, query="robotics"
    )

    batch = connector.fetch(ConnectorCheckpoint("uspto", cursor="0"))

    method, url, request = session.calls[0]
    assert method == "GET"
    assert url == "https://api.uspto.gov/api/v1/patent/applications/search"
    assert request["headers"]["X-API-KEY"] == "private-uspto-key"
    assert request["allow_redirects"] is False
    assert request["stream"] is True
    assert batch.signals[0].signal_type == "patent"
    assert request["params"]["start"] == 0
    assert request["params"]["rows"] == 100
    assert batch.next_checkpoint.cursor == "1"
    assert response.closed is True
    assert "private-uspto-key" not in repr(connector.config)


@pytest.mark.parametrize("status", [401, 403])
def test_uspto_auth_failure_is_nonretryable_and_attempted_once(status):
    session = Session(
        [Response({}, url="https://api.uspto.gov/api/v1/patent/applications/search", status=status)]
    )

    with pytest.raises(ConnectorError) as error:
        USPTOConnector(
            USPTOConfig(api_key="private-uspto-key"), session=session, query="robotics"
        ).fetch(ConnectorCheckpoint("uspto"))

    assert error.value.retryable is False
    assert len(session.calls) == 1
    assert "private-uspto-key" not in str(error.value)


def test_maintenance_marks_only_connector_unavailable_and_preserves_cursor():
    checkpoint = ConnectorCheckpoint("uspto", cursor="1")
    session = Session(
        [
            Response(
                {"message": "maintenance"},
                url="https://api.uspto.gov/api/v1/patent/applications/search",
                status=503,
            )
        ]
    )
    batch = USPTOConnector(
        USPTOConfig(api_key="private-uspto-key"), session=session, query="robotics"
    ).fetch(checkpoint)

    assert batch == SignalConnectorBatch(
        connector="uspto",
        signals=(),
        next_checkpoint=checkpoint,
        status=ConnectorStatus(
            available=False, retryable=True, diagnostic_code="maintenance"
        ),
    )
    independent = ClinicalTrialsConnector().parse(fixture("clinical_trials_sample.json"))
    assert independent[0].signal_type == "clinical_trial"


def test_empty_api_result_preserves_checkpoint():
    checkpoint = ConnectorCheckpoint("clinical_trials", cursor="trial-page-1")
    response = Response(
        {"studies": []},
        url="https://clinicaltrials.gov/api/v2/studies",
    )
    batch = ClinicalTrialsConnector(
        ClinicalTrialsConfig(), session=Session([response]), query="robotics"
    ).fetch(checkpoint)

    assert batch.signals == ()
    assert batch.next_checkpoint == checkpoint


def test_usaspending_uses_documented_post_schema_and_exact_host():
    response = Response(
        json.loads(fixture("usaspending_sample.json")),
        url="https://api.usaspending.gov/api/v2/search/spending_by_award/",
    )
    session = Session([response])
    connector = USASpendingConnector(
        USASpendingConfig(), session=session, query="robotic actuators"
    )

    batch = connector.fetch(ConnectorCheckpoint("usaspending"))

    method, _, request = session.calls[0]
    assert method == "POST"
    assert request["json"]["filters"]["keywords"] == ["robotic actuators"]
    assert request["json"]["fields"] == [
        "Award ID",
        "Recipient Name",
        "Award Amount",
        "Start Date",
        "Recipient Location",
        "Description",
    ]
    assert request["allow_redirects"] is False
    assert batch.signals[0].amount == Decimal("880000")


def test_api_connectors_reject_disallowed_endpoint_before_request():
    session = Session([])

    with pytest.raises(ValueError, match="allowlisted"):
        USPTOConfig(api_key="private-uspto-key", base_url="https://evil.example/api")
    with pytest.raises(ValueError, match="allowlisted"):
        USASpendingConfig(base_url="http://api.usaspending.gov/api/v2")
    with pytest.raises(ValueError, match="allowlisted"):
        ClinicalTrialsConfig(base_url="https://clinicaltrials.gov.evil.test/api/v2")
    assert session.calls == []


def test_sbir_health_outage_uses_bulk_fallback_but_auth_failure_does_not():
    maintenance_session = Session(
        [
            Response(
                {},
                url="https://api.www.sbir.gov/public/api/awards",
                status=503,
            )
        ]
    )
    fallback_calls = []
    connector = SBIRConnector(
        SBIRConfig(),
        session=maintenance_session,
        bulk_loader=lambda: fallback_calls.append(True) or fixture("sbir_sample.json"),
    )

    maintenance_batch = connector.fetch(ConnectorCheckpoint("sbir"))

    assert fallback_calls == [True]
    assert maintenance_batch.signals[0].signal_type == "grant"

    auth_session = Session(
        [
            Response(
                {},
                url="https://api.www.sbir.gov/public/api/awards",
                status=401,
            )
        ]
    )
    forbidden_fallback = []
    with pytest.raises(ConnectorError) as error:
        SBIRConnector(
            SBIRConfig(api_key="private-sbir-key"),
            session=auth_session,
            bulk_loader=lambda: forbidden_fallback.append(True) or "{}",
        ).fetch(ConnectorCheckpoint("sbir"))
    assert error.value.retryable is False
    assert forbidden_fallback == []
    assert len(auth_session.calls) == 1


def test_source_specific_pacing_is_thread_safe():
    times = iter([0.0, 0.0, 0.0, 0.0])
    sleeps = []
    session = Session(
        [
            Response(json.loads(fixture("uspto_sample.json")), url="https://api.uspto.gov/api/v1/patent/applications/search"),
            Response(json.loads(fixture("uspto_sample.json")), url="https://api.uspto.gov/api/v1/patent/applications/search"),
        ]
    )
    connector = USPTOConnector(
        USPTOConfig(api_key="private-uspto-key", min_interval_seconds=0.25),
        session=session,
        query="robotics",
        clock=lambda: next(times),
        sleeper=sleeps.append,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        tuple(pool.map(lambda _: connector.fetch(ConnectorCheckpoint("uspto")), range(2)))

    assert sleeps == [pytest.approx(0.25)]


def test_connector_owns_only_its_internal_session(monkeypatch):
    from news_bot.research.connectors import uspto as uspto_module

    internal = Session([])
    monkeypatch.setattr(uspto_module.requests, "Session", lambda: internal)
    owned = USPTOConnector(USPTOConfig(api_key="private-uspto-key"), query="robotics")
    owned.close()
    assert internal.trust_env is False
    assert internal.closed is True

    external = Session([])
    injected = USPTOConnector(
        USPTOConfig(api_key="private-uspto-key"), session=external, query="robotics"
    )
    injected.close()
    assert external.closed is False


def test_download_and_manual_sources_implement_typed_checkpoint_fetch():
    archive = _zip_bytes("FORM_D.tsv", fixture("form_d_sample.tsv").encode())
    form_checkpoint = ConnectorCheckpoint("form_d", cursor="2026Q2")
    form_d = FormDConnector(
        archive_loader=lambda checkpoint: (archive, "2026Q3")
    )
    manual_checkpoint = ConnectorCheckpoint("manual_import", cursor="import-1")
    manual = ManualImportConnector(content=fixture("manual_company_import.csv"))

    form_batch = form_d.fetch(form_checkpoint)
    manual_batch = manual.fetch(manual_checkpoint)

    assert isinstance(form_d, SignalResearchConnector)
    assert isinstance(manual, SignalResearchConnector)
    assert form_batch.next_checkpoint.cursor == "2026Q3"
    assert form_batch.signals[0].signal_type == "funding"
    assert manual_batch.next_checkpoint == manual_checkpoint
    assert manual_batch.signals[0].signal_type == "private_market_profile"


def test_empty_download_batch_never_advances_checkpoint():
    header_only = fixture("form_d_sample.tsv").splitlines()[0] + "\n"
    archive = _zip_bytes("FORM_D.tsv", header_only.encode())
    checkpoint = ConnectorCheckpoint("form_d", cursor="2026Q2")
    connector = FormDConnector(
        archive_loader=lambda incoming: (archive, "2026Q3")
    )

    batch = connector.fetch(checkpoint)

    assert batch.signals == ()
    assert batch.next_checkpoint == checkpoint
