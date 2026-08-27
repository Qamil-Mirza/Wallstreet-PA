"""Quality regressions for connector trust and identity boundaries."""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from news_bot.news_client import ArticleMeta
from news_bot.research.connectors.base import ConnectorCheckpoint, ConnectorError
from news_bot.research.connectors.fmp import FMPConfig, FMPConnector
from news_bot.research.connectors.investor_relations import (
    InvestorRelationsConfig,
    InvestorRelationsConnector,
)
from news_bot.research.connectors.news import (
    MarketAuxResearchConnector,
    RSSResearchConnector,
)
from news_bot.research.connectors.sec import SECConfig, SECConnector

from .conftest import fixture_json


NOW = datetime(2026, 8, 24, 12, tzinfo=timezone.utc)


class Response:
    def __init__(
        self,
        payload=None,
        *,
        raw: bytes | None = None,
        url: str,
        status: int = 200,
        headers: dict[str, str] | None = None,
        history=(),
    ) -> None:
        self.payload = payload
        self.raw = raw
        self.url = url
        self.status_code = status
        self.headers = headers or {"Content-Type": "application/json"}
        self.history = history
        self.closed = False

    def iter_content(self, chunk_size=65_536):
        content = self.raw if self.raw is not None else json.dumps(self.payload).encode()
        yield from (
            content[index : index + chunk_size]
            for index in range(0, len(content), chunk_size)
        )

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.trust_env = True
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def sec_config():
    return SECConfig(
        user_agent="newsletter-research research@example.com",
        min_interval_seconds=0.11,
    )


def ir_config(**overrides):
    values = {
        "allowed_prefixes": {
            "NVDA": ("https://investor.nvidia.com/news/",)
        },
        "timeout_seconds": 4.0,
        "max_response_bytes": 100_000,
    }
    values.update(overrides)
    return InvestorRelationsConfig(**values)


def test_ir_default_transport_rejects_redirect_and_closes_response():
    response = Response(
        raw=b"",
        url="https://investor.nvidia.com/news/release-1",
        status=302,
        headers={"Location": "http://127.0.0.1/private"},
    )
    session = Session([response])
    connector = InvestorRelationsConnector(ir_config(), session=session)

    with pytest.raises(ConnectorError) as error:
        connector.fetch_release(
            "NVDA", "https://investor.nvidia.com/news/release-1", published_at=NOW
        )

    assert error.value.diagnostic_code == "redirect_rejected"
    assert "127.0.0.1" not in str(error.value)
    assert response.closed is True
    assert session.calls[0][1]["allow_redirects"] is False
    assert session.calls[0][1]["stream"] is True
    assert session.calls[0][1]["timeout"] == 4.0
    assert session.trust_env is True


def test_ir_transport_revalidates_final_url_and_reuses_html_extractor():
    html = (
        b"<html><article><p>Quarterly revenue increased substantially across the "
        b"company's data-center product lines.</p><p>Management maintained its "
        b"published outlook for the next quarter.</p></article></html>"
    )
    response = Response(
        raw=html,
        url="https://investor.nvidia.com/news/release-1",
        headers={"Content-Type": "text/html; charset=utf-8"},
    )
    captured = []
    connector = InvestorRelationsConnector(
        ir_config(),
        session=Session([response]),
        extractor=lambda body, url: captured.append((body, url)) or "Revenue increased.",
    )

    document = connector.fetch_release(
        "NVDA", "https://investor.nvidia.com/news/release-1", published_at=NOW
    )

    assert captured == [(html.decode(), "https://investor.nvidia.com/news/release-1")]
    assert document.evidence.content == "Revenue increased."
    assert response.closed is True


def test_ir_transport_rejects_unapproved_final_url():
    response = Response(
        raw=b"<html></html>",
        url="https://internal.example/private",
        headers={"Content-Type": "text/html"},
    )
    connector = InvestorRelationsConnector(ir_config(), session=Session([response]))

    with pytest.raises(ConnectorError) as error:
        connector.fetch_release(
            "NVDA", "https://investor.nvidia.com/news/release-1", published_at=NOW
        )

    assert error.value.diagnostic_code == "response_origin_rejected"
    assert response.closed is True


@pytest.mark.parametrize(
    "quote,statements",
    [
        ({"Error Message": "private provider body"}, []),
        ([["not-a-mapping"]], []),
        ([], {"error": "private provider body"}),
    ],
)
def test_fmp_parse_rejects_malformed_top_level_schema(quote, statements):
    connector = FMPConnector(FMPConfig(api_key="private"), session=Session([]))

    with pytest.raises(ConnectorError) as error:
        connector.parse(quote=quote, statements=statements)

    assert error.value.diagnostic_code in {"provider_error", "invalid_payload"}
    assert "private provider body" not in str(error.value)


def test_fmp_http_200_provider_error_invokes_explicit_sec_fallback():
    response = Response(
        {"Error Message": "private provider failure"},
        url="https://financialmodelingprep.com/stable/quote",
    )
    fallback_calls = []
    connector = FMPConnector(
        FMPConfig(api_key="private"),
        symbol="NVDA",
        session=Session([response]),
        sec_facts_fallback=lambda symbol: (
            fallback_calls.append(symbol) or {"revenue": Decimal("30")}
        ),
    )

    packet = connector.fetch_fundamentals()

    assert fallback_calls == ["NVDA"]
    assert packet.statements["revenue"] == Decimal("30")
    assert packet.unavailable["fmp"] == "provider_unavailable"
    assert response.closed is True


@pytest.mark.parametrize("mismatch_at", ["quote", "statement"])
def test_fmp_rejects_provider_symbol_mismatch_and_uses_fallback(mismatch_at):
    quote = copy.deepcopy(fixture_json("fmp_quote.json"))
    statement = copy.deepcopy(fixture_json("fmp_income_statement.json"))
    if mismatch_at == "quote":
        quote[0]["symbol"] = "AMD"
    else:
        statement[0]["symbol"] = "AMD"
    responses = [
        Response(quote, url="https://financialmodelingprep.com/stable/quote"),
        Response(
            statement,
            url="https://financialmodelingprep.com/stable/income-statement",
        ),
    ]
    fallback_calls = []
    connector = FMPConnector(
        FMPConfig(api_key="private"),
        symbol="NVDA",
        session=Session(responses),
        sec_facts_fallback=lambda symbol: fallback_calls.append(symbol) or {},
    )

    packet = connector.fetch_fundamentals()

    assert fallback_calls == ["NVDA"]
    assert packet.unavailable["fmp"] == "provider_unavailable"
    assert packet.price is None


def test_sec_submissions_rejects_response_cik_mismatch():
    payload = copy.deepcopy(fixture_json("sec_submissions.json"))
    payload["cik"] = "9999999"
    response = Response(
        payload, url="https://data.sec.gov/submissions/CIK0001045810.json"
    )

    with pytest.raises(ConnectorError) as error:
        SECConnector(sec_config(), session=Session([response])).fetch_submissions(
            "1045810"
        )

    assert error.value.diagnostic_code == "identity_mismatch"
    assert response.closed is True


def test_sec_companyfacts_rejects_response_cik_mismatch():
    payload = copy.deepcopy(fixture_json("sec_companyfacts.json"))
    payload["cik"] = 9999999
    response = Response(
        payload,
        url="https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json",
    )

    with pytest.raises(ConnectorError) as error:
        SECConnector(sec_config(), session=Session([response])).fetch_companyfacts(
            "1045810"
        )

    assert error.value.diagnostic_code == "identity_mismatch"
    assert response.closed is True


@pytest.mark.parametrize("connector_name", ["marketaux", "rss"])
def test_empty_news_batch_preserves_incoming_cursor(connector_name):
    checkpoint = ConnectorCheckpoint(connector_name, cursor="2026-08-24T12:00:00+00:00")
    connector = (
        MarketAuxResearchConnector(
            news_config=object(), section_fetcher=lambda config, per_section_limit: {}
        )
        if connector_name == "marketaux"
        else RSSResearchConnector(article_provider=lambda: {})
    )

    batch = connector.fetch(checkpoint)

    assert batch.documents == ()
    assert batch.next_checkpoint.cursor == checkpoint.cursor


def test_news_batch_never_rewinds_cursor_for_older_document():
    checkpoint = ConnectorCheckpoint(
        "marketaux", cursor="2026-08-24T12:00:00+00:00"
    )
    stale = ArticleMeta(
        id="old",
        title="Older article",
        url="https://news.example/old",
        summary="Older evidence.",
        content="Older evidence.",
        published_at=datetime(2026, 8, 23, tzinfo=timezone.utc),
        source="Reuters",
    )
    connector = MarketAuxResearchConnector(
        news_config=object(),
        section_fetcher=lambda config, per_section_limit: {"World": [stale]},
    )

    batch = connector.fetch(checkpoint)

    assert batch.next_checkpoint.cursor == checkpoint.cursor


def test_fmp_redaction_keeps_ordinary_authorization_but_masks_configured_secret(caplog):
    connector = FMPConnector(
        FMPConfig(api_key="configured-private-key"), session=Session([])
    )
    assert connector.config is not None
    logger = logging.getLogger("urllib3.connectionpool")

    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("policy=%r", {"authorization": "approved by legal"})
        logger.info("diagnostic=%s", "configured-private-key")
        logger.info("sector=%s", "private markets")

    assert "approved by legal" in caplog.text
    assert "configured-private-key" not in caplog.text
    assert "private markets" in caplog.text
