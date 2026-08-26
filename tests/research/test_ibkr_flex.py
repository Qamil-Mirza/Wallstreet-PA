"""Read-only IBKR Flex transport, parsing, and persistence tests."""

import logging
import json
import sqlite3
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from threading import Barrier
from urllib.parse import urlsplit

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
ACCOUNT_SALT = "local-test-salt-strong"


@dataclass(frozen=True)
class _Request:
    method: str
    url: str
    params: dict[str, object]
    headers: dict[str, str]
    timeout: float
    allow_redirects: bool
    stream: bool


class _Response:
    def __init__(
        self,
        text: str,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        iteration_error: BaseException | None = None,
    ) -> None:
        self._chunks = chunks if chunks is not None else [text.encode("utf-8")]
        self.iteration_error = iteration_error
        self.status_code = status_code
        self.headers = headers or {}
        self.iteration_count = 0
        self.yield_count = 0
        self.closed = False

    @property
    def content(self):
        raise AssertionError("Flex transport must not access response.content")

    def iter_content(self, chunk_size):
        self.iteration_count += 1
        for chunk in self._chunks:
            self.yield_count += 1
            yield chunk
        if self.iteration_error is not None:
            raise self.iteration_error

    def close(self):
        self.closed = True


class _Session:
    def __init__(self, adapter: "_RequestsMock") -> None:
        self.adapter = adapter
        self.trust_env = True
        self.closed = False

    def get(
        self, url, *, params, headers, timeout, allow_redirects=True, stream=False
    ):
        return self.adapter.request(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            allow_redirects=allow_redirects,
            stream=stream,
        )

    def close(self):
        self.closed = True


class _RequestsMock:
    def __init__(self) -> None:
        self._registered: dict[
            str,
            list[
                tuple[
                    str,
                    int,
                    dict[str, str],
                    list[bytes] | None,
                    BaseException | None,
                ]
                | BaseException
            ],
        ] = {}
        self.request_history: list[_Request] = []
        self.responses: list[_Response] = []

    def get(
        self,
        url,
        *,
        text="",
        status_code=200,
        headers=None,
        chunks=None,
        iteration_error=None,
        exc=None,
    ):
        response = (
            exc
            if exc is not None
            else (text, status_code, headers or {}, chunks, iteration_error)
        )
        self._registered.setdefault(url, []).append(response)

    def request(self, url, *, params, headers, timeout, allow_redirects, stream):
        self.request_history.append(
            _Request(
                "GET",
                url,
                dict(params),
                dict(headers),
                timeout,
                allow_redirects,
                stream,
            )
        )
        registered = self._registered.get(url, [])
        if not registered:
            raise AssertionError(f"Unexpected GET {url}")
        response = registered.pop(0)
        if isinstance(response, BaseException):
            raise response
        text, status_code, response_headers, chunks, iteration_error = response
        result = _Response(
            text,
            status_code,
            response_headers,
            chunks,
            iteration_error,
        )
        self.responses.append(result)
        return result


@pytest.fixture
def requests_mock(monkeypatch):
    adapter = _RequestsMock()
    monkeypatch.setattr(
        "news_bot.research.ibkr_flex.requests.Session", lambda: _Session(adapter)
    )
    return adapter


@pytest.fixture
def flex_config():
    return FlexConfig(token=TOKEN, query_id=QUERY_ID, account_salt=ACCOUNT_SALT)


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
    return rendered


def test_sync_uses_reference_code_and_never_logs_token(
    requests_mock, caplog, flex_config
):
    requests_mock.get(flex_config.send_url, text=fixture("ibkr_send_success.xml"))
    requests_mock.get(flex_config.statement_url, text=fixture("ibkr_statement.xml"))
    result = FlexClient(flex_config).sync()
    assert result.snapshot.base_currency == "USD"
    assert result.positions[0].symbol == "NVDA"
    assert flex_config.token not in caplog.text


def test_official_lowercase_url_and_java_user_agent_are_used(
    requests_mock, flex_config
):
    assert flex_config.user_agent == "Java"
    requests_mock.get(flex_config.send_url, text=fixture("ibkr_send_success.xml"))
    requests_mock.get(flex_config.statement_url, text=fixture("ibkr_statement.xml"))

    FlexClient(flex_config).sync()

    assert requests_mock.request_history[0].headers["User-Agent"] == "Java"


def test_conflicting_lowercase_and_legacy_statement_urls_are_rejected(
    requests_mock, flex_config
):
    send_xml = fixture("ibkr_send_success.xml").replace(
        "</FlexStatementResponse>",
        "<Url>https://gdcdyn.interactivebrokers.com/GetStatement</Url>"
        "</FlexStatementResponse>",
    )
    requests_mock.get(flex_config.send_url, text=send_xml)

    with pytest.raises(FlexMalformedStatementError, match="Ambiguous"):
        FlexClient(flex_config).sync()


def test_account_id_is_replaced_by_stable_local_hash(flex_config):
    result = parse_statement(
        fixture("ibkr_statement.xml"), account_salt=ACCOUNT_SALT
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


@pytest.mark.parametrize("field", ["token", "query_id", "account_salt"])
@pytest.mark.parametrize(
    ("surrogate", "escaped"),
    [("\ud800", "\\ud800"), ("\udfff", "\\udfff")],
)
def test_flex_config_rejects_surrogates_in_sensitive_fields(
    caplog, flex_config, field, surrogate, escaped
):
    marker = f"{field}-unicode-secret-marker{surrogate}"

    with pytest.raises(ValueError, match=f"{field} must contain valid UTF-8 text") as captured:
        replace(flex_config, **{field: marker})

    assert captured.value.__cause__ is None
    _assert_exception_redacted(
        captured.value,
        (marker, "unicode-secret-marker", surrogate, escaped, "UnicodeEncodeError"),
        caplog,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("user_agent", "safe-prefix-unicode-secret-marker\ud800"),
        (
            "send_url",
            "https://ndcdyn.interactivebrokers.com/unicode-secret-marker\udfff",
        ),
    ],
)
def test_flex_config_rejects_surrogates_at_request_string_boundaries(
    caplog, flex_config, field, value
):
    with pytest.raises(ValueError, match=f"{field} must contain valid UTF-8 text") as captured:
        replace(flex_config, **{field: value})

    assert captured.value.__cause__ is None
    _assert_exception_redacted(
        captured.value,
        ("unicode-secret-marker", "\\ud800", "\\udfff", "UnicodeEncodeError"),
        caplog,
    )


def test_flex_config_hides_sensitive_fields_and_validates_limits(flex_config):
    rendered = repr(flex_config)
    assert TOKEN not in rendered
    assert QUERY_ID not in rendered
    assert ACCOUNT_SALT not in rendered
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
    assert send.stream is True
    assert statement.stream is True
    assert all(response.closed for response in requests_mock.responses)
    allowed_hosts = set(flex_config.allowed_statement_hosts)
    assert all(
        urlsplit(request.url).hostname in allowed_hosts
        for request in requests_mock.request_history
        if request.params.get("t") == TOKEN
    )


@pytest.mark.parametrize("content_length", ["-1", "not-an-integer", "999"])
def test_content_length_is_validated_before_body_iteration(
    requests_mock, flex_config, content_length
):
    config = replace(flex_config, max_response_bytes=64)
    requests_mock.get(
        config.send_url,
        headers={"Content-Length": content_length},
        chunks=[b"body-secret-marker"],
    )

    with pytest.raises(FlexTransportError) as captured:
        FlexClient(config).sync()

    response = requests_mock.responses[0]
    assert response.iteration_count == 0
    assert response.closed
    assert "body-secret-marker" not in str(captured.value)


def test_chunked_overflow_stops_early_and_closes_response(requests_mock, flex_config):
    config = replace(flex_config, max_response_bytes=10)
    requests_mock.get(
        config.send_url,
        chunks=[b"123456", b"secret7", b"must-not-be-read"],
    )

    with pytest.raises(FlexTransportError) as captured:
        FlexClient(config).sync()

    response = requests_mock.responses[0]
    assert response.yield_count == 2
    assert response.closed
    assert "secret7" not in str(captured.value)


def test_stream_iteration_error_is_redacted_and_response_closed(
    requests_mock, caplog, flex_config
):
    marker = "stream-iteration-secret-marker"
    requests_mock.get(
        flex_config.send_url,
        chunks=[b"partial"],
        iteration_error=requests.ConnectionError(f"{marker} {TOKEN} {QUERY_ID}"),
    )

    with pytest.raises(FlexTransportError) as captured:
        FlexClient(flex_config).sync()

    assert requests_mock.responses[0].closed
    _assert_exception_redacted(captured.value, (marker, TOKEN, QUERY_ID), caplog)


def test_redirect_and_http_error_responses_are_always_closed(
    requests_mock, flex_config
):
    requests_mock.get(
        flex_config.send_url,
        status_code=302,
        headers={"Location": "https://evil.test/secret"},
    )
    with pytest.raises(FlexTransportError):
        FlexClient(flex_config).sync()
    assert requests_mock.responses[0].closed


def test_internal_session_disables_environment_proxies_and_is_owned(
    requests_mock, flex_config
):
    client = FlexClient(flex_config)
    assert client.session.trust_env is False
    client.close()
    assert client.session.closed


def test_injected_session_remains_caller_owned(flex_config):
    adapter = _RequestsMock()
    session = _Session(adapter)
    client = FlexClient(flex_config, session=session)
    client.close()
    assert session.trust_env is True
    assert not session.closed


def test_http_library_debug_logs_redact_flex_query_values(caplog, flex_config):
    FlexClient(flex_config)
    token_marker = "logging-secret%2Ftoken"
    query_marker = "logging-secret-query"
    prepared = requests.Request(
        "GET",
        flex_config.send_url,
        params={"t": token_marker, "q": query_marker, "v": "3"},
    ).prepare()
    path_url = prepared.path_url

    with caplog.at_level(logging.DEBUG, logger="urllib3.connectionpool"):
        logging.getLogger("urllib3.connectionpool").debug(
            '%s://%s:%s "%s %s HTTP/%s" %s %s',
            "https",
            "ndcdyn.interactivebrokers.com",
            443,
            "GET",
            path_url,
            "1.1",
            200,
            123,
        )

    assert token_marker not in caplog.text
    assert "logging-secret%252Ftoken" not in caplog.text
    assert query_marker not in caplog.text
    assert "t=%5BREDACTED%5D" in caplog.text or "t=[REDACTED]" in caplog.text


@pytest.mark.parametrize(
    "logger_name",
    [
        "urllib3.util.retry",
        "urllib3.poolmanager",
        "urllib3.future.transport.detail",
        "requests.packages.urllib3.connectionpool",
        "requests.sessions",
    ],
)
def test_all_http_library_descendant_logs_are_redacted(
    caplog, flex_config, logger_name
):
    FlexClient(flex_config)
    marker = "descendant-log-secret-marker"
    prepared = requests.Request(
        "GET",
        flex_config.send_url,
        params={"t": marker, "q": f"{marker}-query", "v": "3"},
    ).prepare()

    with caplog.at_level(logging.DEBUG, logger=logger_name):
        logging.getLogger(logger_name).debug(
            "request data %s",
            {"nested": [{"target": prepared.url}]},
        )

    records = [record for record in caplog.records if record.name == logger_name]
    rendered = caplog.text + "".join(record.getMessage() for record in records)
    assert marker not in rendered
    assert "[REDACTED]" in rendered


def test_http_log_redaction_install_is_idempotent(flex_config):
    FlexClient(flex_config)
    installed = logging.getLogRecordFactory()
    FlexClient(flex_config)
    assert logging.getLogRecordFactory() is installed


def test_http_log_redaction_preserves_custom_factory_and_unrelated_logs(
    caplog, flex_config, monkeypatch
):
    previous = logging.getLogRecordFactory()

    def custom_factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        record.custom_factory_attribute = "preserved"
        return record

    monkeypatch.setattr(logging, "_logRecordFactory", custom_factory)
    FlexClient(flex_config)
    marker = "unrelated-app-query?t=must-remain&q=must-remain"
    with caplog.at_level(logging.INFO):
        logging.getLogger("portfolio.application").info(marker)
        logging.getLogger("urllib3.any.child").info(
            "https://example.test/?t=secret&q=secret"
        )

    app_record = next(
        record for record in caplog.records if record.name == "portfolio.application"
    )
    http_record = next(
        record for record in caplog.records if record.name == "urllib3.any.child"
    )
    assert app_record.getMessage() == marker
    assert app_record.custom_factory_attribute == "preserved"
    assert http_record.custom_factory_attribute == "preserved"
    assert "secret" not in http_record.getMessage()


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
        parse_statement(xml, account_salt=ACCOUNT_SALT)

    assert captured.value.__cause__ is None
    _assert_exception_redacted(captured.value, (marker, ACCOUNT_ID), caplog)


@pytest.mark.parametrize(
    ("surrogate", "escaped"),
    [("\ud800", "\\ud800"), ("\udfff", "\\udfff")],
)
def test_statement_xml_surrogates_raise_redacted_typed_error(
    caplog, surrogate, escaped
):
    marker = f"xml-unicode-secret-marker{surrogate}"
    xml = f"<FlexQueryResponse>{marker}</FlexQueryResponse>"

    with pytest.raises(FlexMalformedStatementError) as captured:
        parse_statement(xml, account_salt=ACCOUNT_SALT)

    assert captured.value.__cause__ is None
    _assert_exception_redacted(
        captured.value,
        (marker, "unicode-secret-marker", surrogate, escaped, "UnicodeEncodeError"),
        caplog,
    )


def test_direct_parser_rejects_surrogate_account_salt_without_leak(caplog):
    marker = "salt-unicode-secret-marker\ud800"

    with pytest.raises(FlexMalformedStatementError, match="account_salt") as captured:
        parse_statement(fixture("ibkr_statement.xml"), account_salt=marker)

    assert captured.value.__cause__ is None
    _assert_exception_redacted(
        captured.value,
        (marker, "unicode-secret-marker", "\\ud800", "UnicodeEncodeError"),
        caplog,
    )


@pytest.mark.parametrize(
    ("encoding", "with_bom"),
    [
        ("utf-16-le", False),
        ("utf-16-be", False),
        ("utf-32-le", False),
        ("utf-32-be", False),
        ("utf-16", True),
        ("utf-32", True),
    ],
)
def test_non_utf8_xml_encodings_are_rejected_before_entity_processing(
    caplog, encoding, with_bom
):
    marker = "encoded-entity-secret-marker"
    text = (
        '<!DoCtYpE FlexQueryResponse [<!EnTiTy x "'
        f'{marker}">]><FlexQueryResponse>&x;</FlexQueryResponse>'
    )
    payload = text.encode(encoding)

    with pytest.raises(FlexMalformedStatementError, match="UTF-8") as captured:
        parse_statement(payload, account_salt=ACCOUNT_SALT)

    assert captured.value.__cause__ is None
    _assert_exception_redacted(captured.value, (marker,), caplog)


@pytest.mark.parametrize(
    "payload",
    [
        b'<?xml version="1.0" encoding="ISO-8859-1"?><FlexQueryResponse/>',
        b"<FlexQuery\x00Response/>",
    ],
)
def test_unsupported_declaration_and_nul_xml_are_rejected(payload):
    with pytest.raises(FlexMalformedStatementError, match="UTF-8"):
        parse_statement(payload, account_salt=ACCOUNT_SALT)


def test_mixed_case_doctype_and_entity_are_rejected_without_expansion():
    xml = (
        '<!DoCtYpE FlexQueryResponse [<!EnTiTy x "entity-secret-marker">]>'
        "<FlexQueryResponse>&x;</FlexQueryResponse>"
    )
    with pytest.raises(FlexMalformedStatementError, match="declarations") as captured:
        parse_statement(xml, account_salt=ACCOUNT_SALT)
    assert "entity-secret-marker" not in str(captured.value)


def test_utf8_bom_statement_is_supported():
    payload = b"\xef\xbb\xbf" + fixture("ibkr_statement.xml").encode("utf-8")
    result = parse_statement(payload, account_salt=ACCOUNT_SALT)
    assert result.snapshot.base_currency == "USD"


def test_oversized_response_is_rejected_without_body_in_error(requests_mock, flex_config):
    config = replace(flex_config, max_response_bytes=32)
    body = "x" * 33
    requests_mock.get(config.send_url, text=body)
    with pytest.raises(FlexTransportError) as captured:
        FlexClient(config).sync()
    assert body not in str(captured.value)


def test_statement_parses_exact_decimals_dates_and_positions():
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt=ACCOUNT_SALT)

    assert result.snapshot.as_of == datetime(2026, 8, 24, tzinfo=timezone.utc)
    assert result.generated_at == datetime(2026, 8, 24, 12, tzinfo=timezone.utc)
    assert result.snapshot.nav == Decimal("4725.50")
    assert result.snapshot.cash == Decimal("525.50")
    assert result.positions[0].quantity == Decimal("10")
    assert result.positions[0].market_value == Decimal("1200.00")
    assert result.positions[0].cost_basis == Decimal("900.00")
    assert len(result.positions) == 2


def test_report_date_drives_source_freshness_not_generation_time():
    xml = fixture("ibkr_statement.xml").replace(
        'whenGenerated="20260824;120000"',
        'whenGenerated="20260826;120000"',
    )

    result = parse_statement(xml, account_salt=ACCOUNT_SALT)

    assert result.snapshot.as_of == utc(2026, 8, 24)
    assert result.generated_at == datetime(2026, 8, 26, 12, tzinfo=timezone.utc)
    assert portfolio_is_stale(result.snapshot.as_of, utc(2026, 8, 26), max_hours=24)


def test_parse_without_evaluation_context_has_no_freshness_result():
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt=ACCOUNT_SALT)
    assert result.freshness is None
    assert result.snapshot.is_stale is None


def test_sync_uses_clock_to_evaluate_source_freshness(
    requests_mock, flex_config
):
    requests_mock.get(flex_config.send_url, text=fixture("ibkr_send_success.xml"))
    statement = fixture("ibkr_statement.xml").replace(
        'whenGenerated="20260824;120000"',
        'whenGenerated="20260826;120000"',
    )
    requests_mock.get(flex_config.statement_url, text=statement)
    evaluated_at = utc(2026, 8, 26)

    result = FlexClient(flex_config, clock=lambda: evaluated_at).sync()

    assert result.snapshot.is_stale is None
    assert result.freshness is not None
    assert result.freshness.evaluated_at == evaluated_at
    assert result.freshness.as_of == result.snapshot.as_of
    assert result.freshness.max_hours == flex_config.max_staleness_hours
    assert result.freshness.is_stale is True


def test_evaluation_clock_does_not_change_snapshot_or_persistence(tmp_path):
    xml = fixture("ibkr_statement.xml")
    before = parse_statement(
        xml,
        account_salt=ACCOUNT_SALT,
        now=utc(2026, 8, 24),
        max_staleness_hours=24,
    )
    after = parse_statement(
        xml,
        account_salt=ACCOUNT_SALT,
        now=utc(2026, 8, 30),
        max_staleness_hours=24,
    )
    assert before.snapshot == after.snapshot
    assert before.positions == after.positions
    assert before.generated_at == after.generated_at
    assert before.freshness is not None
    assert after.freshness is not None
    assert before.freshness.is_stale is False
    assert after.freshness.is_stale is True
    assert before.snapshot.is_stale is None

    store = make_migrated_store(tmp_path)
    store.insert_portfolio_snapshot(before.snapshot, before.positions, before.account_ref)
    store.insert_portfolio_snapshot(after.snapshot, after.positions, after.account_ref)
    with store.connect() as connection:
        assert connection.execute(
            "SELECT is_stale FROM portfolio_snapshots"
        ).fetchone() == (None,)


def test_freshness_context_validation_is_explicit():
    xml = fixture("ibkr_statement.xml")
    with pytest.raises(ValueError, match="provided together"):
        parse_statement(xml, account_salt=ACCOUNT_SALT, now=utc(2026, 8, 24))
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_statement(
            xml,
            account_salt=ACCOUNT_SALT,
            now=datetime(2026, 8, 24),
            max_staleness_hours=24,
        )
    with pytest.raises(ValueError, match="positive"):
        parse_statement(
            xml,
            account_salt=ACCOUNT_SALT,
            now=utc(2026, 8, 24),
            max_staleness_hours=0,
        )


def test_sync_result_and_freshness_domain_invariants():
    result = parse_statement(
        fixture("ibkr_statement.xml"),
        account_salt=ACCOUNT_SALT,
        now=utc(2026, 8, 24),
        max_staleness_hours=24,
    )
    assert result.freshness is not None
    with pytest.raises(ValueError, match="generated_at.*timezone-aware"):
        replace(result, generated_at=datetime(2026, 8, 24))
    with pytest.raises(ValueError, match="account_ref"):
        replace(result, account_ref=ACCOUNT_ID)
    with pytest.raises(ValueError, match="snapshot_id"):
        replace(
            result,
            positions=(replace(result.positions[0], snapshot_id="other"),),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(result.freshness, evaluated_at=datetime(2026, 8, 24))
    with pytest.raises(ValueError, match="as_of"):
        replace(result.freshness, as_of=utc(2026, 8, 23))


def test_portfolio_snapshot_staleness_allows_only_bool_or_none():
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt=ACCOUNT_SALT)
    assert result.snapshot.is_stale is None
    replace(result.snapshot, is_stale=True)
    replace(result.snapshot, is_stale=False)
    with pytest.raises(TypeError, match="is_stale"):
        replace(result.snapshot, is_stale=0)


def test_store_schema_and_flex_persistence_keep_unevaluated_staleness_null(
    tmp_path,
):
    store = make_migrated_store(tmp_path)
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt=ACCOUNT_SALT)
    store.insert_portfolio_snapshot(result.snapshot, result.positions, result.account_ref)

    with store.connect() as connection:
        columns = {
            row[1]: row for row in connection.execute("PRAGMA table_info(portfolio_snapshots)")
        }
        stored = connection.execute(
            "SELECT is_stale FROM portfolio_snapshots"
        ).fetchone()
    assert columns["is_stale"][3] == 0
    assert stored == (None,)


def test_multiple_report_dates_select_latest_summary_and_holdings():
    xml = fixture("ibkr_statement.xml")
    earlier_summary = (
        '<EquitySummaryByReportDateInBase reportDate="20260823" '
        'total="4700.00" cash="500.00" />'
    )
    xml = xml.replace(
        "      <EquitySummaryInBase>",
        "      <EquitySummaryInBase>" + earlier_summary,
    )
    earlier_position = (
        '<OpenPosition accountId="U1234567" reportDate="20260823" '
        'assetCategory="STK" currency="USD" symbol="OLD" conid="111" '
        'position="1" positionValue="1.00" />'
    )
    xml = xml.replace("      <OpenPositions>", "      <OpenPositions>" + earlier_position)

    result = parse_statement(xml, account_salt=ACCOUNT_SALT)

    assert result.snapshot.nav == Decimal("4725.50")
    assert {position.symbol for position in result.positions} == {"NVDA", "US-TNOTE"}


def test_position_preserves_stable_security_identity_and_identifiers():
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt=ACCOUNT_SALT)
    nvda = result.positions[0]
    assert nvda.security_id.startswith("security_")
    assert nvda.asset_class == "STK"
    assert nvda.conid == "4815747"
    assert nvda.isin == "US67066G1040"
    assert nvda.figi == "BBG000BBJQV0"
    assert nvda.external_security_id == "67066G104"
    assert nvda.security_id_type == "CUSIP"


def test_cross_snapshot_security_identity_and_database_round_trip(tmp_path):
    first = parse_statement(fixture("ibkr_statement.xml"), account_salt=ACCOUNT_SALT)
    second_xml = fixture("ibkr_statement.xml").replace("20260824", "20260825")
    second = parse_statement(second_xml, account_salt=ACCOUNT_SALT)
    first_nvda = first.positions[0]
    second_nvda = second.positions[0]
    assert first_nvda.security_id == second_nvda.security_id
    assert first_nvda.position_id != second_nvda.position_id

    store = make_migrated_store(tmp_path)
    store.insert_portfolio_snapshot(first.snapshot, first.positions, first.account_ref)
    store.insert_portfolio_snapshot(second.snapshot, second.positions, second.account_ref)
    with store.connect() as connection:
        position_security_ids = connection.execute(
            "SELECT security_id FROM positions WHERE symbol = 'NVDA' ORDER BY snapshot_id"
        ).fetchall()
        row = connection.execute(
            "SELECT identifiers_json FROM securities WHERE security_id = ?",
            (first_nvda.security_id,),
        ).fetchone()
    assert position_security_ids == [
        (first_nvda.security_id,),
        (first_nvda.security_id,),
    ]
    assert json.loads(row[0]) == {
        "conid": "4815747",
        "external_security_id": "67066G104",
        "figi": "BBG000BBJQV0",
        "isin": "US67066G1040",
        "security_id_type": "CUSIP",
    }


def test_identifier_conflict_rolls_back_snapshot_and_security_changes(tmp_path):
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt=ACCOUNT_SALT)
    store = make_migrated_store(tmp_path)
    store.insert_portfolio_snapshot(result.snapshot, result.positions, result.account_ref)
    conflicting_snapshot = replace(
        result.snapshot,
        snapshot_id="snapshot_000000000000000000000000",
        as_of=utc(2026, 8, 25),
    )
    conflicting_position = replace(
        result.positions[0],
        position_id="position_000000000000000000000000",
        snapshot_id=conflicting_snapshot.snapshot_id,
        symbol="CONFLICT",
    )

    with pytest.raises(sqlite3.IntegrityError, match="security"):
        store.insert_portfolio_snapshot(
            conflicting_snapshot, (conflicting_position,), result.account_ref
        )

    with store.connect() as connection:
        assert connection.execute("SELECT count(*) FROM portfolio_snapshots").fetchone()[0] == 1
        assert connection.execute("SELECT symbol FROM securities WHERE security_id = ?", (result.positions[0].security_id,)).fetchone()[0] == "NVDA"


def test_security_identity_falls_back_to_asset_symbol_and_currency():
    xml = fixture("ibkr_statement.xml")
    for attribute in (
        ' conid="4815747"',
        ' isin="US67066G1040"',
        ' figi="BBG000BBJQV0"',
        ' securityID="67066G104"',
        ' securityIDType="CUSIP"',
    ):
        xml = xml.replace(attribute, "", 1)
    first = parse_statement(xml, account_salt=ACCOUNT_SALT)
    second = parse_statement(xml.replace("20260824", "20260825"), account_salt=ACCOUNT_SALT)
    assert first.positions[0].security_id == second.positions[0].security_id
    assert first.positions[0].position_id != second.positions[0].position_id


def test_weak_account_salt_is_rejected_before_hashing(flex_config):
    with pytest.raises(ValueError, match="at least 16 UTF-8 bytes"):
        replace(flex_config, account_salt="too-short")
    with pytest.raises(FlexMalformedStatementError, match="at least 16 UTF-8 bytes"):
        parse_statement(fixture("ibkr_statement.xml"), account_salt="too-short")


def test_statement_accepts_documented_iso_timestamp_variant():
    xml = fixture("ibkr_statement.xml").replace(
        "20260824;120000", "2026-08-24T12:00:00Z"
    )
    result = parse_statement(xml, account_salt=ACCOUNT_SALT)
    assert result.snapshot.as_of == utc(2026, 8, 24)
    assert result.generated_at == datetime(2026, 8, 24, 12, tzinfo=timezone.utc)


def test_statement_accepts_documented_dashed_flex_timestamp_variant():
    xml = fixture("ibkr_statement.xml").replace(
        'whenGenerated="20260824;120000"',
        'whenGenerated="2026-08-24;12:00:00"',
    )
    result = parse_statement(xml, account_salt=ACCOUNT_SALT)
    assert result.snapshot.as_of == utc(2026, 8, 24)
    assert result.generated_at == datetime(2026, 8, 24, 12, tzinfo=timezone.utc)


def test_statement_allows_all_cash_and_negative_quantities():
    xml = fixture("ibkr_statement.xml")
    positions_start = xml.index("      <OpenPositions>")
    positions_end = xml.index("      </OpenPositions>") + len("      </OpenPositions>\n")
    all_cash = xml[:positions_start] + xml[positions_end:]
    assert parse_statement(all_cash, account_salt=ACCOUNT_SALT).positions == ()

    short_xml = xml.replace('position="10"', 'position="-10"', 1)
    assert parse_statement(short_xml, account_salt=ACCOUNT_SALT).positions[0].quantity == Decimal("-10")


@pytest.mark.parametrize("bad_total", ["NaN", "Infinity", "-Infinity"])
def test_statement_rejects_nonfinite_numbers(bad_total):
    xml = fixture("ibkr_statement.xml").replace('total="4725.50"', f'total="{bad_total}"')
    with pytest.raises(FlexMalformedStatementError, match="finite"):
        parse_statement(xml, account_salt=ACCOUNT_SALT)


def test_statement_rejects_missing_or_ambiguous_account_data():
    missing_nav = fixture("ibkr_statement.xml").replace(' total="4725.50"', "")
    with pytest.raises(FlexMalformedStatementError, match="NAV"):
        parse_statement(missing_nav, account_salt=ACCOUNT_SALT)

    duplicate = fixture("ibkr_statement.xml").replace(
        "    </FlexStatement>\n", "    </FlexStatement>\n    <FlexStatement accountId=\"OTHER\" whenGenerated=\"20260824;120000\" />\n"
    )
    with pytest.raises(FlexMalformedStatementError, match="exactly one"):
        parse_statement(duplicate, account_salt=ACCOUNT_SALT)


def test_account_hash_and_snapshot_ids_are_stable_and_salted():
    xml = fixture("ibkr_statement.xml")
    first = parse_statement(xml, account_salt="salt-a-0123456789")
    repeat = parse_statement(xml, account_salt="salt-a-0123456789")
    other = parse_statement(xml, account_salt="salt-b-0123456789")
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
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt=ACCOUNT_SALT)
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
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt=ACCOUNT_SALT)

    with pytest.raises(sqlite3.IntegrityError, match="duplicate position"):
        store.insert_portfolio_snapshot(
            result.snapshot, (result.positions[0], result.positions[0]), result.account_ref
        )

    with store.connect() as connection:
        assert connection.execute("SELECT count(*) FROM portfolio_snapshots").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM positions").fetchone()[0] == 0


def test_store_rejects_raw_account_identifier(tmp_path):
    store = make_migrated_store(tmp_path)
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt=ACCOUNT_SALT)

    with pytest.raises(ValueError, match="hashed local reference"):
        store.insert_portfolio_snapshot(result.snapshot, result.positions, ACCOUNT_ID)

    with store.connect() as connection:
        assert connection.execute("SELECT count(*) FROM portfolio_snapshots").fetchone()[0] == 0


def test_concurrent_identical_snapshot_writers_are_idempotent(tmp_path):
    database_path = tmp_path / "research.db"
    first_store = make_migrated_store(tmp_path)
    second_store = type(first_store)(database_path)
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt=ACCOUNT_SALT)
    barrier = Barrier(2)

    def write(store):
        barrier.wait()
        store.insert_portfolio_snapshot(result.snapshot, result.positions, result.account_ref)
        return "ok"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(write, (first_store, second_store)))

    assert outcomes == ["ok", "ok"]
    with first_store.connect() as connection:
        assert connection.execute("SELECT count(*) FROM portfolio_snapshots").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM positions").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM securities").fetchone()[0] == 2


def test_concurrent_conflicting_snapshot_writers_leave_consistent_winner(tmp_path):
    database_path = tmp_path / "research.db"
    first_store = make_migrated_store(tmp_path)
    second_store = type(first_store)(database_path)
    result = parse_statement(fixture("ibkr_statement.xml"), account_salt=ACCOUNT_SALT)
    conflict = replace(result.snapshot, nav=Decimal("4000.00"))
    barrier = Barrier(2)

    def write(store, snapshot):
        barrier.wait()
        try:
            store.insert_portfolio_snapshot(snapshot, result.positions, result.account_ref)
        except sqlite3.IntegrityError:
            return "conflict"
        return "ok"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(write, first_store, result.snapshot),
            executor.submit(write, second_store, conflict),
        )
        outcomes = [future.result() for future in futures]

    assert sorted(outcomes) == ["conflict", "ok"]
    with first_store.connect() as connection:
        nav = connection.execute("SELECT nav FROM portfolio_snapshots").fetchone()[0]
        position_count = connection.execute("SELECT count(*) FROM positions").fetchone()[0]
        security_count = connection.execute("SELECT count(*) FROM securities").fetchone()[0]
    assert nav in {"4725.50", "4000.00"}
    assert position_count == 2
    assert security_count == 2
