"""Contract tests for public research-source connectors."""

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from news_bot.news_client import ArticleMeta
from news_bot.research.connectors.base import (
    ConnectorBatch,
    ConnectorCheckpoint,
    ConnectorError,
    NormalizedResearchDocument,
    commit_connector_batch,
)
from news_bot.research.connectors.fmp import DatedPrice, FMPConfig, FMPConnector
from news_bot.research.connectors.investor_relations import (
    InvestorRelationsConfig,
    InvestorRelationsConnector,
)
from news_bot.research.connectors.news import (
    MarketAuxResearchConnector,
    RSSResearchConnector,
)
from news_bot.research.connectors.sec import SECConfig, SECConnector


NOW = datetime(2026, 8, 24, 12, tzinfo=timezone.utc)
FIXTURES = Path(__file__).parent / "fixtures"


def fixture_json(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(
        self,
        payload,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        url: str = "https://data.sec.gov/submissions/CIK0001045810.json",
        history: tuple[object, ...] = (),
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "application/json"}
        self.url = url
        self.history = history

    def iter_content(self, chunk_size: int = 65536):
        raw = json.dumps(self.payload).encode("utf-8")
        yield from (raw[index : index + chunk_size] for index in range(0, len(raw), chunk_size))


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.trust_env = True

    def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class BrokenBodyResponse(FakeResponse):
    def iter_content(self, chunk_size: int = 65536):
        raise OSError("https://private.test/?token=private")
        yield b""  # pragma: no cover - preserves generator semantics


def article(*, source: str = "Reuters", published_at: datetime | None = None):
    return ArticleMeta(
        id="a-1",
        title="Chip demand expands",
        url="https://News.Example/story?utm_source=feed",
        summary="Demand expanded across data-center customers.",
        content="Demand expanded across data-center customers.",
        published_at=published_at or datetime(2026, 8, 23, 8),
        source=source,
    )


@pytest.mark.parametrize(
    ("connector", "expected_type"),
    [
        (MarketAuxResearchConnector(), "marketaux_news"),
        (RSSResearchConnector(), "rss_news"),
    ],
)
def test_news_connectors_emit_canonical_tagged_documents(connector, expected_type):
    document = connector.from_article(article(), section="US Tech")

    assert isinstance(document, NormalizedResearchDocument)
    assert document.publisher == "Reuters"
    assert document.published_at.tzinfo is not None
    assert document.canonical_url == "https://news.example/story"
    assert document.tags == ("US Tech",)
    assert document.evidence.source_type == expected_type


def test_marketaux_connector_reuses_injected_section_fetcher():
    calls = []

    def fetcher(config, per_section_limit):
        calls.append((config, per_section_limit))
        return {"US Tech": [article()]}

    connector = MarketAuxResearchConnector(
        news_config=object(), section_fetcher=fetcher, per_section_limit=3
    )
    batch = connector.fetch(ConnectorCheckpoint("marketaux"))

    assert calls == [(connector.news_config, 3)]
    assert batch.documents[0].tags == ("US Tech",)
    assert batch.next_checkpoint.connector == "marketaux"


def test_connector_contract_is_frozen_and_checkpoint_commits_last():
    checkpoint = ConnectorCheckpoint("sec", cursor="cursor-1")
    batch = ConnectorBatch("sec", (), checkpoint)
    events = []

    commit_connector_batch(
        batch,
        persist_documents=lambda documents: events.append(("documents", documents)),
        persist_checkpoint=lambda value: events.append(("checkpoint", value)),
    )

    assert events == [("documents", ()), ("checkpoint", checkpoint)]
    with pytest.raises(dataclasses.FrozenInstanceError):
        checkpoint.cursor = "mutated"


def test_failed_batch_persistence_does_not_advance_checkpoint():
    batch = ConnectorBatch("sec", (), ConnectorCheckpoint("sec", cursor="next"))
    advanced = []

    with pytest.raises(RuntimeError, match="storage failed"):
        commit_connector_batch(
            batch,
            persist_documents=lambda documents: (_ for _ in ()).throw(RuntimeError("storage failed")),
            persist_checkpoint=advanced.append,
        )

    assert advanced == []


def test_connector_errors_are_structured_and_secret_safe():
    error = ConnectorError("sec", status=429, retryable=True, diagnostic_code="http_429")

    assert error.status == 429
    assert error.retryable is True
    assert str(error) == "sec connector failed (http_429, status=429, retryable=true)"
    with pytest.raises(ValueError):
        ConnectorError("sec", diagnostic_code="https://secret.test/?token=private")
    with pytest.raises(ValueError):
        ConnectorError(
            "https://secret.test/?token=private", diagnostic_code="transport_error"
        )


def sec_config(**overrides):
    values = {
        "user_agent": "newsletter-research research@example.com",
        "min_interval_seconds": 0.11,
        "timeout_seconds": 5.0,
        "max_response_bytes": 100_000,
    }
    values.update(overrides)
    return SECConfig(**values)


def test_sec_connector_sends_identity_conditionals_and_safe_transport():
    session = FakeSession(
        [FakeResponse(fixture_json("sec_submissions.json"), headers={"ETag": '"v2"'})]
    )
    connector = SECConnector(sec_config(), session=session, clock=lambda: 100.0)
    checkpoint = ConnectorCheckpoint(
        "sec", etag='"v1"', last_modified="Sat, 22 Aug 2026 00:00:00 GMT"
    )

    result = connector.fetch_submissions("1045810", checkpoint=checkpoint)

    url, request = session.calls[0]
    assert url == "https://data.sec.gov/submissions/CIK0001045810.json"
    assert request["headers"] == {
        "User-Agent": "newsletter-research research@example.com",
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
        "If-None-Match": '"v1"',
        "If-Modified-Since": "Sat, 22 Aug 2026 00:00:00 GMT",
    }
    assert request["allow_redirects"] is False
    assert request["stream"] is True
    assert request["timeout"] == 5.0
    assert session.trust_env is True
    assert result.checkpoint.etag == '"v2"'


@pytest.mark.parametrize("cik", ["", "abc", "12345678901", "-1"])
def test_sec_connector_rejects_invalid_cik_without_request(cik):
    session = FakeSession([])

    with pytest.raises(ValueError, match="CIK"):
        SECConnector(sec_config(), session=session).fetch_submissions(cik)

    assert session.calls == []


def test_sec_config_rejects_non_identifying_or_unsafe_settings():
    with pytest.raises(ValueError, match="identifying"):
        sec_config(user_agent="newsletter")
    with pytest.raises(ValueError, match="below 10 requests"):
        sec_config(min_interval_seconds=0.05)


def test_sec_connector_parses_only_approved_forms_as_evidence_documents():
    connector = SECConnector(sec_config(), session=FakeSession([]))
    documents = connector.parse_submissions(
        fixture_json("sec_submissions.json"), retrieved_at=NOW
    )

    assert len(documents) == 1
    assert documents[0].evidence.source_type == "sec_filing_metadata"
    assert documents[0].canonical_url.endswith("/nvda-20260726.htm")
    assert documents[0].tags == ("SEC", "10-Q")
    assert documents[0].published_at == datetime(2026, 8, 20, tzinfo=timezone.utc)


def test_sec_companyfacts_emits_canonical_document_and_rejects_redirects():
    session = FakeSession(
        [
            FakeResponse(
                fixture_json("sec_companyfacts.json"),
                url="https://evil.example/companyfacts.json",
                history=(object(),),
            )
        ]
    )

    with pytest.raises(ConnectorError) as error:
        SECConnector(sec_config(), session=session).fetch_companyfacts("1045810")

    assert error.value.diagnostic_code == "redirect_rejected"
    assert "evil.example" not in str(error.value)


@pytest.mark.parametrize(
    ("status", "retryable"), [(429, True), (503, True), (404, False), (401, False)]
)
def test_sec_http_failures_are_classified_without_response_body(status, retryable):
    session = FakeSession([FakeResponse({"token": "private"}, status_code=status)])

    with pytest.raises(ConnectorError) as error:
        SECConnector(sec_config(), session=session).fetch_submissions("1045810")

    assert error.value.retryable is retryable
    assert "private" not in str(error.value)


def test_sec_stream_failures_are_typed_and_redacted():
    session = FakeSession(
        [BrokenBodyResponse({}, url="https://data.sec.gov/submissions/CIK0001045810.json")]
    )

    with pytest.raises(ConnectorError) as error:
        SECConnector(sec_config(), session=session).fetch_submissions("1045810")

    assert error.value.diagnostic_code == "response_read_failed"
    assert "private.test" not in str(error.value)


def test_sec_rate_limiter_paces_concurrent_request_boundary():
    moments = iter([1.0, 1.02, 1.12])
    sleeps = []
    session = FakeSession(
        [
            FakeResponse(fixture_json("sec_submissions.json")),
            FakeResponse(fixture_json("sec_companyfacts.json")),
        ]
    )
    connector = SECConnector(
        sec_config(), session=session, clock=lambda: next(moments), sleeper=sleeps.append
    )

    connector.fetch_submissions("1045810")
    connector.fetch_companyfacts("1045810")

    assert sleeps == [pytest.approx(0.09)]


def test_investor_relations_enforces_issuer_host_and_path_allowlist():
    config = InvestorRelationsConfig(
        allowed_prefixes={"NVDA": ("https://investor.nvidia.com/news/",)}
    )
    connector = InvestorRelationsConnector(
        config,
        extractor=lambda url: ("Quarterly revenue increased.", False),
        clock=lambda: NOW,
    )

    document = connector.fetch_release(
        "NVDA", "https://investor.nvidia.com/news/release-1", published_at=NOW
    )

    assert document.publisher == "NVDA Investor Relations"
    assert document.tags == ("investor-relations", "NVDA")
    with pytest.raises(ValueError, match="allowlisted"):
        connector.fetch_release(
            "NVDA", "https://investor.nvidia.com/financials/secret", published_at=NOW
        )
    with pytest.raises(ValueError, match="allowlisted"):
        connector.fetch_release(
            "NVDA", "https://investor.nvidia.com.evil.test/news/release", published_at=NOW
        )


def test_investor_relations_rejects_encoded_path_traversal():
    connector = InvestorRelationsConnector(
        InvestorRelationsConfig(
            allowed_prefixes={"NVDA": ("https://investor.nvidia.com/news/",)}
        ),
        extractor=lambda url: ("Sensitive financial content.", False),
        clock=lambda: NOW,
    )

    with pytest.raises(ValueError, match="allowlisted"):
        connector.fetch_release(
            "NVDA",
            "https://investor.nvidia.com/news/%2e%2e/financials/secret",
            published_at=NOW,
        )


def test_fmp_connector_emits_dated_decimal_price_and_fundamentals():
    connector = FMPConnector(FMPConfig(api_key="private-key"), session=FakeSession([]))
    packet = connector.parse(
        quote=fixture_json("fmp_quote.json"),
        statements=fixture_json("fmp_income_statement.json"),
    )

    assert packet.price.value == Decimal("171.44")
    assert packet.price.effective_date == date(2026, 8, 21)
    assert packet.financial_period == "2026-Q2"
    assert packet.financial_effective_date == date(2026, 7, 26)
    assert packet.statements["revenue"] == Decimal("30000000000")
    assert packet.ratios["gross_profit_ratio"] == Decimal("0.734")
    assert packet.metadata["reported_currency"] == "USD"
    assert packet.unavailable["net_income"] == "provider_value_missing"
    assert packet.use_sec_fallback is True


def test_fmp_absence_is_explicit_and_never_fabricated():
    packet = FMPConnector(
        FMPConfig(api_key=None), session=FakeSession([])
    ).parse(quote=[], statements=[])

    assert packet.price is None
    assert packet.statements == {}
    assert packet.unavailable == {
        "price": "provider_value_missing",
        "fundamentals": "provider_value_missing",
    }
    assert packet.use_sec_fallback is True


def test_dated_price_rejects_implicit_or_nonfinite_numbers():
    with pytest.raises(TypeError, match="Decimal"):
        DatedPrice("NVDA", 171.44, date(2026, 8, 21))
    with pytest.raises(ValueError, match="finite"):
        DatedPrice("NVDA", Decimal("NaN"), date(2026, 8, 21))


def test_fmp_request_uses_allowlisted_https_base_and_never_leaks_key():
    session = FakeSession([FakeResponse([], url="https://financialmodelingprep.com/stable/quote")])
    connector = FMPConnector(FMPConfig(api_key="private-key"), session=session)

    connector.fetch_quote("NVDA")

    url, request = session.calls[0]
    assert url == "https://financialmodelingprep.com/stable/quote"
    assert request["params"] == {"symbol": "NVDA", "apikey": "private-key"}
    assert "private-key" not in repr(connector.config)
    assert FMPConfig(
        api_key="secret", base_url="http://financialmodelingprep.com/stable"
    ).unavailable_reason == "endpoint_disallowed"
    assert FMPConfig(
        api_key="secret", base_url="https://evil.example/stable"
    ).unavailable_reason == "endpoint_disallowed"


def test_fmp_http_error_is_redacted_and_classified():
    session = FakeSession(
        [
            FakeResponse(
                {"apikey": "private-key"},
                status_code=429,
                url="https://financialmodelingprep.com/stable/quote",
            )
        ]
    )
    connector = FMPConnector(FMPConfig(api_key="private-key"), session=session)

    with pytest.raises(ConnectorError) as error:
        connector.fetch_quote("NVDA")

    assert error.value.retryable is True
    assert "private-key" not in str(error.value)


def test_fmp_stream_failures_are_typed_and_redacted():
    session = FakeSession(
        [
            BrokenBodyResponse(
                {}, url="https://financialmodelingprep.com/stable/quote"
            )
        ]
    )
    connector = FMPConnector(FMPConfig(api_key="private-key"), session=session)

    with pytest.raises(ConnectorError) as error:
        connector.fetch_quote("NVDA")

    assert error.value.diagnostic_code == "response_read_failed"
    assert "private.test" not in str(error.value)
