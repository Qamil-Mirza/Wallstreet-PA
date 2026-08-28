"""Regression contracts for connector safety and orchestration boundaries."""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal

import pytest
import requests

from news_bot.news_client import ArticleMeta
from news_bot.research.connectors import fmp as fmp_module
from news_bot.research.connectors.base import (
    ConnectorBatch,
    ConnectorCheckpoint,
    ConnectorError,
    ResearchConnector,
)
from news_bot.research.connectors.fmp import FMPConfig, FMPConnector
from news_bot.research.connectors.investor_relations import (
    InvestorRelationsConfig,
    InvestorRelationsConnector,
)
from news_bot.research.connectors.news import MarketAuxResearchConnector
from news_bot.research.connectors.sec import SECConfig, SECConnector
from news_bot.research.evidence import EvidenceIngestor

from .conftest import fixture_json


NOW = datetime(2026, 8, 24, 12, tzinfo=timezone.utc)


class ClosingResponse:
    def __init__(
        self,
        payload,
        *,
        url: str,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        history: tuple[object, ...] = (),
        fail_iteration: bool = False,
    ) -> None:
        self.payload = payload
        self.url = url
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "application/json"}
        self.history = history
        self.fail_iteration = fail_iteration
        self.closed = False

    def iter_content(self, chunk_size: int = 65_536):
        if self.fail_iteration:
            raise OSError("https://private.test/?apikey=private")
        raw = json.dumps(self.payload).encode("utf-8")
        yield from (
            raw[index : index + chunk_size]
            for index in range(0, len(raw), chunk_size)
        )

    def close(self) -> None:
        self.closed = True


class RecordingSession:
    def __init__(self, responses: list[ClosingResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.trust_env = True
        self.closed = False

    def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def sec_config(**overrides) -> SECConfig:
    values = {
        "user_agent": "newsletter-research research@example.com",
        "min_interval_seconds": 0.11,
        "timeout_seconds": 5.0,
        "max_response_bytes": 100_000,
    }
    values.update(overrides)
    return SECConfig(**values)


def test_fmp_log_redaction_chains_once_and_preserves_unrelated_records(caplog):
    original_factory = logging.getLogRecordFactory()
    factory_calls = []

    def prior_factory(*args, **kwargs):
        record = original_factory(*args, **kwargs)
        record.prior_factory_marker = True
        factory_calls.append(record.name)
        return record

    logging.setLogRecordFactory(prior_factory)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            tuple(pool.map(lambda _: fmp_module.install_fmp_log_redaction(), range(32)))
        installed_factory = logging.getLogRecordFactory()
        fmp_module.install_fmp_log_redaction()
        assert logging.getLogRecordFactory() is installed_factory

        logger = logging.getLogger("urllib3.connectionpool")
        with caplog.at_level(logging.DEBUG, logger=logger.name):
            logger.debug(
                "GET /stable/quote?symbol=NVDA&apikey=private-query headers=%r",
                {"apikey": "private-header", "Accept": "application/json"},
            )
            logger.debug("headers={'apikey': 'private-rendered'}")
            logger.debug("unrelated %s", "value")
            try:
                raise requests.RequestException(
                    "failed https://financialmodelingprep.com/stable/quote?apikey=private-exception"
                )
            except requests.RequestException:
                logger.exception("request failed")

        assert "private-query" not in caplog.text
        assert "private-header" not in caplog.text
        assert "private-exception" not in caplog.text
        assert "private-rendered" not in caplog.text
        assert "apikey=[REDACTED]" in caplog.text
        unrelated = next(record for record in caplog.records if record.msg == "unrelated %s")
        assert unrelated.args == ("value",)
        assert unrelated.prior_factory_marker is True
        assert factory_calls
    finally:
        logging.setLogRecordFactory(original_factory)


def test_fmp_secret_registration_and_log_redaction_are_thread_safe(monkeypatch, caplog):
    iteration_started = threading.Event()
    registration_finished = threading.Event()
    original_factory = logging.getLogRecordFactory()

    class CoordinatedSecrets(set):
        def __iter__(self):
            iterator = super().__iter__()
            yield next(iterator)
            iteration_started.set()
            assert registration_finished.wait(timeout=2)
            yield from iterator

    first_secret = "first-concurrent-secret"
    second_secret = "second-concurrent-secret"
    monkeypatch.setattr(
        fmp_module, "_CONFIGURED_SECRETS", CoordinatedSecrets({first_secret})
    )
    logging.setLogRecordFactory(logging.LogRecord)
    logger = logging.getLogger("fmp.concurrent.redaction")

    def register_secret() -> None:
        assert iteration_started.wait(timeout=2)
        try:
            FMPConnector(
                FMPConfig(api_key=second_secret), session=RecordingSession([])
            )
        finally:
            registration_finished.set()

    try:
        fmp_module.install_fmp_log_redaction()
        with caplog.at_level(logging.ERROR, logger=logger.name):
            with ThreadPoolExecutor(max_workers=2) as pool:
                registration = pool.submit(register_secret)
                logged = pool.submit(logger.error, "request failed: %s", first_secret)
                registration.result(timeout=5)
                logged.result(timeout=5)
            logger.error("request failed: %s", second_secret)

        assert first_secret not in caplog.text
        assert second_secret not in caplog.text
        assert caplog.text.count("[REDACTED]") == 2
    finally:
        logging.setLogRecordFactory(original_factory)


def test_fmp_disabled_configuration_returns_explicit_sec_fallback():
    fallback_calls = []

    def sec_facts(symbol: str):
        fallback_calls.append(symbol)
        return {"revenue": Decimal("30000000000")}

    connector = FMPConnector(
        FMPConfig(api_key=None),
        symbol="NVDA",
        session=RecordingSession([]),
        sec_facts_fallback=sec_facts,
    )

    packet = connector.fetch_fundamentals()

    assert fallback_calls == ["NVDA"]
    assert packet.price is None
    assert packet.statements == {"revenue": Decimal("30000000000")}
    assert packet.metadata["fundamental_source"] == "sec_companyfacts"
    assert packet.unavailable["fmp"] == "configuration_missing"
    assert packet.use_sec_fallback is True


def test_fmp_empty_sec_fallback_is_explicitly_unavailable():
    connector = FMPConnector(
        FMPConfig(api_key=None),
        symbol="NVDA",
        session=RecordingSession([]),
        sec_facts_fallback=lambda symbol: {},
    )

    packet = connector.fetch_fundamentals()

    assert packet.statements == {}
    assert packet.unavailable["sec_fallback"] == "fallback_unavailable"
    assert packet.use_sec_fallback is True


def test_fmp_disallowed_endpoint_never_attempts_transport_or_leaks_config():
    session = RecordingSession([])
    config = FMPConfig(
        api_key="private-key",
        base_url="http://evil.example/stable?apikey=private-config",
    )
    connector = FMPConnector(config, symbol="NVDA", session=session)

    packet = connector.fetch_fundamentals()

    assert session.calls == []
    assert packet.unavailable["fmp"] == "endpoint_disallowed"
    assert packet.use_sec_fallback is True
    assert "private-key" not in repr(config)
    assert "private-config" not in repr(config)


def test_news_document_input_is_canonical_before_ingestion():
    connector = MarketAuxResearchConnector(now=lambda: NOW)
    source = ArticleMeta(
        id="a-1",
        title="Chip demand",
        url="HTTPS://News.Example:443/story?utm_source=feed#fragment",
        summary=None,
        content="  Revenue\tgrew.\r\n\r\n Margin   expanded. ",
        published_at=datetime(2026, 8, 23, 8),
        source="Reuters",
    )

    document = connector.from_article(source, section="US Tech")

    assert document.evidence.url == "https://news.example/story"
    assert document.evidence.content == "Revenue grew.\n\nMargin expanded."


def test_news_pipeline_ingests_before_advancing_checkpoint(migrated_store, tmp_path):
    source = ArticleMeta(
        id="a-1",
        title="Chip demand",
        url="https://news.example/story?utm_source=feed",
        summary=None,
        content="Revenue grew.",
        published_at=NOW,
        source="Reuters",
    )
    fetch_calls = []
    ingestor = EvidenceIngestor(migrated_store, tmp_path / "cache")
    connector = MarketAuxResearchConnector(
        news_config=object(),
        section_fetcher=lambda config, per_section_limit: (
            fetch_calls.append((config, per_section_limit)) or {"US Tech": [source]}
        ),
        ingestor=ingestor,
        now=lambda: NOW,
    )
    checkpoints = []

    def advance(checkpoint):
        assert migrated_store.get_document_by_content_hash(
            "1ccfbd0e54d401dbddd1c26566778fb94b44e2a404960b8c3a7320984362150d"
        ) is not None
        checkpoints.append(checkpoint)

    ingested = connector.fetch_and_persist(
        ConnectorCheckpoint("marketaux"), persist_checkpoint=advance
    )

    assert len(ingested) == 1
    assert len(fetch_calls) == 1
    assert checkpoints == [ConnectorCheckpoint("marketaux", cursor=NOW.isoformat())]


def test_connector_error_is_immutable_exception():
    error = ConnectorError(
        "sec", status=429, retryable=True, diagnostic_code="http_429"
    )

    assert isinstance(error, RuntimeError)
    assert error.args == (
        "sec connector failed (http_429, status=429, retryable=true)",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        error.status = 500
    with pytest.raises(dataclasses.FrozenInstanceError):
        error.diagnostic_code = "changed"


def test_sec_ir_and_fmp_structurally_implement_research_connector():
    connectors = (
        SECConnector(sec_config(), session=RecordingSession([])),
        InvestorRelationsConnector(
            InvestorRelationsConfig(
                allowed_prefixes={"NVDA": ("https://investor.nvidia.com/news/",)}
            )
        ),
        FMPConnector(FMPConfig(api_key=None), session=RecordingSession([])),
    )

    assert all(isinstance(connector, ResearchConnector) for connector in connectors)


def test_sec_fetch_uses_validated_constructor_context():
    response = ClosingResponse(
        fixture_json("sec_submissions.json"),
        url="https://data.sec.gov/submissions/CIK0001045810.json",
    )
    connector = SECConnector(
        sec_config(), cik="1045810", session=RecordingSession([response])
    )

    batch = connector.fetch(ConnectorCheckpoint("sec"))

    assert isinstance(batch, ConnectorBatch)
    assert batch.documents[0].tags == ("SEC", "10-Q")
    assert batch.next_checkpoint.cursor == "0001045810"


def test_ir_fetch_uses_validated_release_context():
    response = ClosingResponse(
        {},
        url="https://investor.nvidia.com/news/release-1",
        headers={"Content-Type": "text/html"},
    )
    connector = InvestorRelationsConnector(
        InvestorRelationsConfig(
            allowed_prefixes={"NVDA": ("https://investor.nvidia.com/news/",)}
        ),
        issuer="NVDA",
        release_url="https://investor.nvidia.com/news/release-1",
        published_at=NOW,
        session=RecordingSession([response]),
        extractor=lambda html, url: "Revenue grew.",
        clock=lambda: NOW,
    )

    batch = connector.fetch(ConnectorCheckpoint("investor_relations"))

    assert isinstance(batch, ConnectorBatch)
    assert batch.documents[0].tags == ("investor-relations", "NVDA")
    assert batch.next_checkpoint.connector == "investor_relations"


def test_fmp_fetch_uses_validated_symbol_context():
    responses = [
        ClosingResponse(
            fixture_json("fmp_quote.json"),
            url="https://financialmodelingprep.com/stable/quote",
        ),
        ClosingResponse(
            fixture_json("fmp_income_statement.json"),
            url="https://financialmodelingprep.com/stable/income-statement",
        ),
    ]
    connector = FMPConnector(
        FMPConfig(api_key="private-key"),
        symbol="NVDA",
        session=RecordingSession(responses),
        clock=lambda: NOW,
    )

    batch = connector.fetch(ConnectorCheckpoint("fmp"))

    assert isinstance(batch, ConnectorBatch)
    assert batch.documents[0].evidence.source_type == "fmp_fundamentals"
    assert batch.next_checkpoint.cursor == "NVDA"


@pytest.mark.parametrize(
    "mode", ["success", "not_modified", "redirect", "status", "overflow", "iteration"]
)
def test_sec_response_is_closed_for_every_outcome(mode):
    response = ClosingResponse(
        fixture_json("sec_submissions.json"),
        url="https://data.sec.gov/submissions/CIK0001045810.json",
        status_code=304 if mode == "not_modified" else (429 if mode == "status" else 200),
        history=(object(),) if mode == "redirect" else (),
        fail_iteration=mode == "iteration",
    )
    config = sec_config(max_response_bytes=8 if mode == "overflow" else 100_000)
    connector = SECConnector(config, session=RecordingSession([response]))

    if mode in {"redirect", "status", "overflow", "iteration"}:
        with pytest.raises(ConnectorError):
            connector.fetch_submissions("1045810")
    else:
        connector.fetch_submissions("1045810")

    assert response.closed is True


@pytest.mark.parametrize("mode", ["success", "redirect", "status", "overflow", "iteration"])
def test_fmp_response_is_closed_for_every_outcome(mode):
    response = ClosingResponse(
        fixture_json("fmp_quote.json"),
        url="https://financialmodelingprep.com/stable/quote",
        status_code=429 if mode == "status" else 200,
        history=(object(),) if mode == "redirect" else (),
        fail_iteration=mode == "iteration",
    )
    config = FMPConfig(
        api_key="private-key", max_response_bytes=8 if mode == "overflow" else 100_000
    )
    connector = FMPConnector(config, session=RecordingSession([response]))

    if mode == "success":
        connector.fetch_quote("NVDA")
    else:
        with pytest.raises(ConnectorError):
            connector.fetch_quote("NVDA")

    assert response.closed is True


@pytest.mark.parametrize("connector_kind", ["sec", "fmp"])
def test_connectors_do_not_close_or_mutate_injected_sessions(connector_kind):
    session = RecordingSession([])
    connector = (
        SECConnector(sec_config(), session=session)
        if connector_kind == "sec"
        else FMPConnector(FMPConfig(api_key=None), session=session)
    )

    connector.close()

    assert session.closed is False
    assert session.trust_env is True
