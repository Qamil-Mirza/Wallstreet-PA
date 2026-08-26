"""Read-only IBKR Flex transport, parsing, and persistence tests."""

import logging
import sqlite3
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
import requests

from news_bot.research.ibkr_flex import (
    FlexAuthenticationError,
    FlexClient,
    FlexConfig,
    FlexError,
    FlexIPRestrictionError,
    FlexInvalidQueryError,
    FlexMalformedStatementError,
    FlexPollingExhaustedError,
    FlexThrottledError,
    FlexTransportError,
    FlexUnsafeResponseURLError,
    parse_statement,
    portfolio_is_stale,
)

from .conftest import fixture, make_migrated_store, utc


TOKEN = "test-token-never-log"
QUERY_ID = "query-12345"
ACCOUNT_ID = "U1234567"


@dataclass(frozen=True)
class _Request:
    method: str
    url: str
    params: dict[str, object]
    headers: dict[str, str]
    timeout: float
    allow_redirects: bool


class _Response:
    def __init__(
        self, text: str, status_code: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        self.content = text.encode("utf-8")
        self.status_code = status_code
        self.headers = headers or {}


class _Session:
    def __init__(self, adapter: "_RequestsMock") -> None:
        self.adapter = adapter

    def get(self, url, *, params, headers, timeout, allow_redirects=True):
        return self.adapter.request(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )


class _RequestsMock:
    def __init__(self) -> None:
        self._registered: dict[
            str, list[tuple[str, int, dict[str, str]] | BaseException]
        ] = {}
        self.request_history: list[_Request] = []

    def get(self, url, *, text="", status_code=200, headers=None, exc=None):
        response = exc if exc is not None else (text, status_code, headers or {})
        self._registered.setdefault(url, []).append(response)

    def request(self, url, *, params, headers, timeout, allow_redirects):
        self.request_history.append(
            _Request(
                "GET", url, dict(params), dict(headers), timeout, allow_redirects
            )
        )
        registered = self._registered.get(url, [])
        if not registered:
            raise AssertionError(f"Unexpected GET {url}")
        response = registered.pop(0)
        if isinstance(response, BaseException):
            raise response
        text, status_code, response_headers = response
        return _Response(text, status_code, response_headers)


@pytest.fixture
def requests_mock(monkeypatch):
    adapter = _RequestsMock()
    monkeypatch.setattr(
        "news_bot.research.ibkr_flex.requests.Session", lambda: _Session(adapter)
    )
    return adapter


@pytest.fixture
def flex_config():
    return FlexConfig(token=TOKEN, query_id=QUERY_ID, account_salt="local-test-salt")


def _error_xml(code: int) -> str:
    return (
        "<FlexStatementResponse><Status>Fail</Status>"
        f"<ErrorCode>{code}</ErrorCode><ErrorMessage>remote error</ErrorMessage>"
        "</FlexStatementResponse>"
    )


def _send_xml(url: str) -> str:
    return (
        "<FlexStatementResponse><Status>Success</Status>"
        "<ReferenceCode>REF-1</ReferenceCode>"
        f"<Url>{url}</Url></FlexStatementResponse>"
    )


def _assert_exception_redacted(error, markers, caplog):
    rendered = str(error) + repr(error)
    rendered += "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    caplog.clear()
    logging.getLogger("tests.ibkr.redaction").error(
        "sanitized Flex failure",
        exc_info=(type(error), error, error.__traceback__),
    )
    rendered += caplog.text
    for marker in markers:
        assert marker not in rendered


def test_sync_uses_reference_code_and_never_logs_token(
    requests_mock, caplog, flex_config
):
    requests_mock.get(flex_config.send_url, text=fixture("ibkr_send_success.xml"))
    requests_mock.get(flex_config.statement_url, text=fixture("ibkr_statement.xml"))
    result = FlexClient(flex_config).sync()
    assert result.snapshot.base_currency == "USD"
    assert result.positions[0].symbol == "NVDA"
    assert flex_config.token not in caplog.text


def test_account_id_is_replaced_by_stable_local_hash(flex_config):
    result = parse_statement(
        fixture("ibkr_statement.xml"), account_salt="local-test-salt"
    )
    assert result.account_ref.startswith("acct_")
    assert "U1234567" not in result.account_ref


def test_stale_snapshot_suppresses_sizing():
    assert portfolio_is_stale(utc(2026, 8, 22), utc(2026, 8, 24), max_hours=36)


@pytest.mark.parametrize("field", ["token", "query_id", "account_salt"])
def test_flex_config_rejects_blank_sensitive_values(field):
    values = {"token": TOKEN, "query_id": QUERY_ID, "account_salt": "salt"}
    values[field] = "  "
    with pytest.raises(ValueError, match=field):
        FlexConfig(**values)


def test_flex_config_hides_sensitive_fields_and_validates_limits(flex_config):
    rendered = repr(flex_config)
    assert TOKEN not in rendered
    assert QUERY_ID not in rendered
    assert "local-test-salt" not in rendered
    with pytest.raises(ValueError, match="max_polls"):
        replace(flex_config, max_polls=0)
    with pytest.raises(ValueError, match="HTTPS"):
        replace(flex_config, send_url="http://ndcdyn.interactivebrokers.com/send")


def test_https_requirement_cannot_be_disabled(flex_config):
    with pytest.raises(ValueError, match="ssl_required.*True"):
        replace(flex_config, ssl_required=False)


@pytest.mark.parametrize("field", ["send_url", "statement_url"])
def test_configured_http_endpoints_are_rejected_without_transport(
    requests_mock, flex_config, field
):
    with pytest.raises(ValueError, match="HTTPS"):
        replace(
            flex_config,
            **{field: "http://ndcdyn.interactivebrokers.com/FlexWebService"},
        )
    assert requests_mock.request_history == []


def test_default_https_port_is_allowed_but_other_ports_are_rejected(flex_config):
    replace(
        flex_config,
        send_url="https://ndcdyn.interactivebrokers.com:443/SendRequest",
        statement_url="https://ndcdyn.interactivebrokers.com:443/GetStatement",
    )
    with pytest.raises(ValueError, match="allowed HTTPS URL"):
        replace(
            flex_config,
            statement_url="https://ndcdyn.interactivebrokers.com:8443/GetStatement",
        )


def test_send_and_poll_use_get_params_headers_and_timeout(requests_mock, flex_config):
    requests_mock.get(flex_config.send_url, text=fixture("ibkr_send_success.xml"))
    requests_mock.get(flex_config.statement_url, text=fixture("ibkr_statement.xml"))

    FlexClient(flex_config).sync()

    assert [request.method for request in requests_mock.request_history] == ["GET", "GET"]
    send, statement = requests_mock.request_history
    assert send.params == {"t": TOKEN, "q": QUERY_ID, "v": "3"}
    assert statement.params == {"t": TOKEN, "q": "REF-20260824-001", "v": "3"}
    assert send.headers["User-Agent"] == flex_config.user_agent
    assert statement.headers["User-Agent"] == flex_config.user_agent
    assert send.timeout == statement.timeout == flex_config.request_timeout_seconds
    assert send.allow_redirects is False
    assert statement.allow_redirects is False
    assert TOKEN not in send.url and TOKEN not in statement.url


@pytest.mark.parametrize(
    ("status", "target"),
    [
        (301, "https://evil.test/steal"),
        (302, "http://127.0.0.1/private"),
    ],
)
def test_send_redirect_is_not_followed_or_leaked(
    requests_mock, caplog, flex_config, status, target
):
    requests_mock.get(
        flex_config.send_url,
        status_code=status,
        headers={"Location": target},
    )

    with pytest.raises(FlexTransportError) as captured:
        FlexClient(flex_config).sync()

    assert len(requests_mock.request_history) == 1
    request = requests_mock.request_history[0]
    assert request.url == flex_config.send_url
    assert request.allow_redirects is False
    _assert_exception_redacted(
        captured.value, (TOKEN, QUERY_ID, ACCOUNT_ID, target), caplog
    )


@pytest.mark.parametrize(
    ("status", "target"),
    [
        (301, "https://evil.test/steal"),
        (302, "http://127.0.0.1/private"),
    ],
)
def test_statement_redirect_is_not_followed_or_leaked(
    requests_mock, caplog, flex_config, status, target
):
    requests_mock.get(flex_config.send_url, text=fixture("ibkr_send_success.xml"))
    requests_mock.get(
        flex_config.statement_url,
        status_code=status,
        headers={"Location": target},
    )

    with pytest.raises(FlexTransportError) as captured:
        FlexClient(flex_config).sync()

    assert len(requests_mock.request_history) == 2
    assert requests_mock.request_history[-1].url == flex_config.statement_url
    assert all(not request.allow_redirects for request in requests_mock.request_history)
    _assert_exception_redacted(
        captured.value, (TOKEN, QUERY_ID, ACCOUNT_ID, target), caplog
    )


def test_1019_retries_with_capped_exponential_delay(requests_mock, flex_config):
    config = replace(flex_config, max_polls=3)
    requests_mock.get(config.send_url, text=fixture("ibkr_send_success.xml"))
    requests_mock.get(config.statement_url, text=_error_xml(1019))
    requests_mock.get(config.statement_url, text=_error_xml(1019))
    requests_mock.get(config.statement_url, text=fixture("ibkr_statement.xml"))
    delays = []

    FlexClient(config, sleeper=delays.append).sync()

    assert delays == [1, 2]


def test_1019_retry_delay_caps_at_sixteen_seconds(requests_mock, flex_config):
    config = replace(flex_config, max_polls=7)
    requests_mock.get(config.send_url, text=fixture("ibkr_send_success.xml"))
    for _ in range(6):
        requests_mock.get(config.statement_url, text=_error_xml(1019))
    requests_mock.get(config.statement_url, text=fixture("ibkr_statement.xml"))
    delays = []

    FlexClient(config, sleeper=delays.append).sync()

    assert delays == [1, 2, 4, 8, 16, 16]


def test_1019_exhaustion_uses_exact_poll_limit(requests_mock, flex_config):
    config = replace(flex_config, max_polls=2)
    requests_mock.get(config.send_url, text=fixture("ibkr_send_success.xml"))
    requests_mock.get(config.statement_url, text=_error_xml(1019))
    requests_mock.get(config.statement_url, text=_error_xml(1019))

    with pytest.raises(FlexPollingExhaustedError, match="2 polls"):
        FlexClient(config, sleeper=lambda _: None).sync()

    assert len(requests_mock.request_history) == 3


@pytest.mark.parametrize(
    ("code", "error_type"),
    [
        (1001, FlexError),
        (1003, FlexError),
        (1004, FlexError),
        (1005, FlexError),
        (1006, FlexError),
        (1007, FlexError),
        (1008, FlexError),
        (1009, FlexThrottledError),
        (1010, FlexError),
        (1011, FlexAuthenticationError),
        (1012, FlexAuthenticationError),
        (1013, FlexIPRestrictionError),
        (1014, FlexInvalidQueryError),
        (1015, FlexAuthenticationError),
        (1016, FlexAuthenticationError),
        (1017, FlexInvalidQueryError),
        (1018, FlexThrottledError),
        (1020, FlexInvalidQueryError),
        (1021, FlexError),
    ],
)
def test_non_retryable_codes_map_and_stop_immediately(
    requests_mock, flex_config, code, error_type
):
    requests_mock.get(flex_config.send_url, text=fixture("ibkr_send_success.xml"))
    requests_mock.get(flex_config.statement_url, text=_error_xml(code))

    with pytest.raises(error_type) as captured:
        FlexClient(flex_config).sync()

    assert len(requests_mock.request_history) == 2
    assert TOKEN not in str(captured.value)
    assert ACCOUNT_ID not in str(captured.value)


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "http://ndcdyn.interactivebrokers.com/GetStatement",
        "https://evil.test/GetStatement",
        "https://user:pass@ndcdyn.interactivebrokers.com/GetStatement",
        "https://ndcdyn.interactivebrokers.com:8443/GetStatement",
        "https://ndcdyn.interactivebrokers.com/GetStatement#fragment",
    ],
)
def test_unsafe_statement_url_is_rejected_before_token_is_sent_there(
    requests_mock, flex_config, unsafe_url
):
    requests_mock.get(flex_config.send_url, text=_send_xml(unsafe_url))

    with pytest.raises(FlexUnsafeResponseURLError):
        FlexClient(flex_config).sync()

    assert [item.url for item in requests_mock.request_history] == [flex_config.send_url]


def test_explicit_default_port_in_returned_url_is_allowed(requests_mock, flex_config):
    statement_url = (
        "https://ndcdyn.interactivebrokers.com:443/AccountManagement/"
        "FlexWebService/GetStatement"
    )
    requests_mock.get(flex_config.send_url, text=_send_xml(statement_url))
    requests_mock.get(statement_url, text=fixture("ibkr_statement.xml"))

    result = FlexClient(flex_config).sync()

    assert result.snapshot.base_currency == "USD"


@pytest.mark.parametrize(
    ("response_text", "status", "exception", "error_type"),
    [
        ("server failure", 500, None, FlexTransportError),
        ("", 200, requests.Timeout(f"timeout {TOKEN} {ACCOUNT_ID}"), FlexTransportError),
        ("<not-closed>", 200, None, FlexMalformedStatementError),
        ("<!DOCTYPE x [<!ENTITY e SYSTEM 'file:///etc/passwd'>]><x>&e;</x>", 200, None, FlexMalformedStatementError),
    ],
)
def test_transport_and_xml_errors_are_typed_and_redacted(
    requests_mock, flex_config, response_text, status, exception, error_type
):
    requests_mock.get(
        flex_config.send_url,
        text=response_text,
        status_code=status,
        exc=exception,
    )

    with pytest.raises(error_type) as captured:
        FlexClient(flex_config).sync()

    rendered = str(captured.value)
    assert TOKEN not in rendered
    assert ACCOUNT_ID not in rendered
    if response_text:
        assert response_text not in rendered


def test_request_exception_chain_is_fully_redacted(requests_mock, caplog, flex_config):
    marker = "request-secret-marker"
    requests_mock.get(
        flex_config.send_url,
        exc=requests.Timeout(f"{marker} {TOKEN} {QUERY_ID} {ACCOUNT_ID}"),
    )

    with pytest.raises(FlexTransportError) as captured:
        FlexClient(flex_config).sync()

    _assert_exception_redacted(
        captured.value, (marker, TOKEN, QUERY_ID, ACCOUNT_ID), caplog
    )


def test_malicious_url_exception_chain_is_fully_redacted(
    requests_mock, caplog, flex_config
):
    marker = "url-secret-marker"
    bad_url = f"https://ndcdyn.interactivebrokers.com:{marker}/GetStatement"
    requests_mock.get(flex_config.send_url, text=_send_xml(bad_url))

    with pytest.raises(FlexUnsafeResponseURLError) as captured:
        FlexClient(flex_config).sync()

    _assert_exception_redacted(captured.value, (marker, TOKEN, QUERY_ID), caplog)


def test_malformed_xml_exception_chain_does_not_retain_body(
    requests_mock, caplog, flex_config
):
    marker = "xml-secret-marker"
    body = f"<FlexStatementResponse><{marker}></FlexStatementResponse>"
    requests_mock.get(flex_config.send_url, text=body)

    with pytest.raises(FlexMalformedStatementError) as captured:
        FlexClient(flex_config).sync()

    assert captured.value.__cause__ is None
    _assert_exception_redacted(captured.value, (marker, TOKEN, QUERY_ID), caplog)


def test_invalid_decimal_exception_chain_does_not_retain_value(caplog):
    marker = "decimal-secret-marker"
    xml = fixture("ibkr_statement.xml").replace(
        'total="4725.50"', f'total="{marker}"'
    )

    with pytest.raises(FlexMalformedStatementError) as captured:
        parse_statement(xml, account_salt="salt")

    assert captured.value.__cause__ is None
    _assert_exception_redacted(captured.value, (marker, ACCOUNT_ID), caplog)


def test_oversized_response_is_rejected_without_body_in_error(requests_mock, flex_config):
    config = replace(flex_config, max_response_bytes=32)
    body = "x" * 33
    requests_mock.get(config.send_url, text=body)
    with pytest.raises(FlexTransportError) as captured:
        FlexClient(config).sync()
    assert body not in str(captured.value)


def test_statement_parses_exact_decimals_dates_and_positions():
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt="salt")

    assert result.snapshot.as_of == datetime(2026, 8, 24, 12, tzinfo=timezone.utc)
    assert result.snapshot.nav == Decimal("4725.50")
    assert result.snapshot.cash == Decimal("525.50")
    assert result.positions[0].quantity == Decimal("10")
    assert result.positions[0].market_value == Decimal("1200.00")
    assert result.positions[0].cost_basis == Decimal("900.00")
    assert len(result.positions) == 2


def test_statement_accepts_documented_iso_timestamp_variant():
    xml = fixture("ibkr_statement.xml").replace(
        "20260824;120000", "2026-08-24T12:00:00Z"
    )
    result = parse_statement(xml, account_salt="salt")
    assert result.snapshot.as_of == datetime(2026, 8, 24, 12, tzinfo=timezone.utc)


def test_statement_accepts_documented_dashed_flex_timestamp_variant():
    xml = fixture("ibkr_statement.xml").replace(
        'whenGenerated="20260824;120000"',
        'whenGenerated="2026-08-24;12:00:00"',
    )
    result = parse_statement(xml, account_salt="salt")
    assert result.snapshot.as_of == datetime(2026, 8, 24, 12, tzinfo=timezone.utc)


def test_statement_allows_all_cash_and_negative_quantities():
    xml = fixture("ibkr_statement.xml")
    positions_start = xml.index("      <OpenPositions>")
    positions_end = xml.index("      </OpenPositions>") + len("      </OpenPositions>\n")
    all_cash = xml[:positions_start] + xml[positions_end:]
    assert parse_statement(all_cash, account_salt="salt").positions == ()

    short_xml = xml.replace('position="10"', 'position="-10"', 1)
    assert parse_statement(short_xml, account_salt="salt").positions[0].quantity == Decimal("-10")


@pytest.mark.parametrize("bad_total", ["NaN", "Infinity", "-Infinity"])
def test_statement_rejects_nonfinite_numbers(bad_total):
    xml = fixture("ibkr_statement.xml").replace('total="4725.50"', f'total="{bad_total}"')
    with pytest.raises(FlexMalformedStatementError, match="finite"):
        parse_statement(xml, account_salt="salt")


def test_statement_rejects_missing_or_ambiguous_account_data():
    missing_nav = fixture("ibkr_statement.xml").replace(' total="4725.50"', "")
    with pytest.raises(FlexMalformedStatementError, match="NAV"):
        parse_statement(missing_nav, account_salt="salt")

    duplicate = fixture("ibkr_statement.xml").replace(
        "    </FlexStatement>\n", "    </FlexStatement>\n    <FlexStatement accountId=\"OTHER\" whenGenerated=\"20260824;120000\" />\n"
    )
    with pytest.raises(FlexMalformedStatementError, match="exactly one"):
        parse_statement(duplicate, account_salt="salt")


def test_account_hash_and_snapshot_ids_are_stable_and_salted():
    xml = fixture("ibkr_statement.xml")
    first = parse_statement(xml, account_salt="salt-a")
    repeat = parse_statement(xml, account_salt="salt-a")
    other = parse_statement(xml, account_salt="salt-b")
    assert first.account_ref == repeat.account_ref
    assert first.snapshot.snapshot_id == repeat.snapshot.snapshot_id
    assert first.account_ref != other.account_ref
    assert first.snapshot.snapshot_id != other.snapshot.snapshot_id


def test_portfolio_staleness_boundaries_and_time_validation():
    now = utc(2026, 8, 24)
    assert not portfolio_is_stale(now - timedelta(hours=36), now, max_hours=36)
    assert portfolio_is_stale(
        now - timedelta(hours=36, seconds=1), now, max_hours=36
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        portfolio_is_stale(datetime(2026, 8, 23), now, max_hours=36)
    with pytest.raises(ValueError, match="future"):
        portfolio_is_stale(now + timedelta(minutes=10), now, max_hours=36)
    with pytest.raises(ValueError, match="positive"):
        portfolio_is_stale(now, now, max_hours=0)


def test_sync_persists_atomically_and_never_stores_raw_credentials(
    requests_mock, flex_config, tmp_path
):
    store = make_migrated_store(tmp_path)
    requests_mock.get(flex_config.send_url, text=fixture("ibkr_send_success.xml"))
    requests_mock.get(flex_config.statement_url, text=fixture("ibkr_statement.xml"))

    result = FlexClient(flex_config).sync(store=store)

    with store.connect() as connection:
        snapshot_count = connection.execute("SELECT count(*) FROM portfolio_snapshots").fetchone()[0]
        position_count = connection.execute("SELECT count(*) FROM positions").fetchone()[0]
        dump = "\n".join(connection.iterdump())
    assert (snapshot_count, position_count) == (1, 2)
    assert result.account_ref in dump
    assert ACCOUNT_ID not in dump
    assert TOKEN not in dump
    assert QUERY_ID not in dump


def test_portfolio_persistence_is_idempotent_and_conflicts_do_not_overwrite(tmp_path):
    store = make_migrated_store(tmp_path)
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt="salt")
    store.insert_portfolio_snapshot(result.snapshot, result.positions, result.account_ref)
    store.insert_portfolio_snapshot(result.snapshot, result.positions, result.account_ref)

    with store.connect() as connection:
        assert connection.execute("SELECT count(*) FROM portfolio_snapshots").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM positions").fetchone()[0] == 2

    conflict = replace(result.snapshot, nav=Decimal("1.00"))
    with pytest.raises(sqlite3.IntegrityError, match="conflicting snapshot"):
        store.insert_portfolio_snapshot(conflict, result.positions, result.account_ref)

    with store.connect() as connection:
        assert connection.execute("SELECT nav FROM portfolio_snapshots").fetchone()[0] == "4725.50"


def test_invalid_positions_roll_back_entire_snapshot(tmp_path):
    store = make_migrated_store(tmp_path)
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt="salt")

    with pytest.raises(sqlite3.IntegrityError, match="duplicate position"):
        store.insert_portfolio_snapshot(
            result.snapshot, (result.positions[0], result.positions[0]), result.account_ref
        )

    with store.connect() as connection:
        assert connection.execute("SELECT count(*) FROM portfolio_snapshots").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM positions").fetchone()[0] == 0


def test_store_rejects_raw_account_identifier(tmp_path):
    store = make_migrated_store(tmp_path)
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt="salt")

    with pytest.raises(ValueError, match="hashed local reference"):
        store.insert_portfolio_snapshot(result.snapshot, result.positions, ACCOUNT_ID)

    with store.connect() as connection:
        assert connection.execute("SELECT count(*) FROM portfolio_snapshots").fetchone()[0] == 0
