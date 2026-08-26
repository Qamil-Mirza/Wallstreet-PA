"""Read-only IBKR Flex Web Service transport and portfolio normalization."""

from __future__ import annotations

import hashlib
import hmac
import math
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import requests

from .models import PortfolioSnapshot, Position

if TYPE_CHECKING:
    from .store import ResearchStore


DEFAULT_SEND_URL = (
    "https://ndcdyn.interactivebrokers.com/AccountManagement/"
    "FlexWebService/SendRequest"
)
DEFAULT_STATEMENT_URL = (
    "https://ndcdyn.interactivebrokers.com/AccountManagement/"
    "FlexWebService/GetStatement"
)
DEFAULT_ALLOWED_HOSTS = (
    "ndcdyn.interactivebrokers.com",
    "gdcdyn.interactivebrokers.com",
)
_FUTURE_CLOCK_TOLERANCE = timedelta(minutes=5)


class FlexError(RuntimeError):
    """Base class for redacted Flex failures."""

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code

    @classmethod
    def from_code(cls, code: int) -> "FlexError":
        """Map a Flex error code without retaining the remote message/body."""
        error_type: type[FlexError]
        if code in {1011, 1012, 1015, 1016}:
            error_type = FlexAuthenticationError
            label = "authentication failed"
        elif code == 1013:
            error_type = FlexIPRestrictionError
            label = "IP restriction rejected the request"
        elif code in {1014, 1017, 1020}:
            error_type = FlexInvalidQueryError
            label = "query or reference code is invalid"
        elif code in {1009, 1018}:
            error_type = FlexThrottledError
            label = "request was throttled"
        elif code == 1019:
            error_type = FlexNotReadyError
            label = "statement is still generating"
        else:
            error_type = FlexError
            label = "service rejected the request"
        return error_type(f"IBKR Flex error {code}: {label}", code=code)


class FlexAuthenticationError(FlexError):
    """Flex credentials or account authorization were rejected."""


class FlexIPRestrictionError(FlexError):
    """The caller did not satisfy the token's IP restriction."""


class FlexInvalidQueryError(FlexError):
    """The query or statement reference was invalid."""


class FlexThrottledError(FlexError):
    """IBKR rejected the request because of service load or rate limits."""


class FlexNotReadyError(FlexError):
    """The requested statement has not finished generating."""


class FlexPollingExhaustedError(FlexNotReadyError):
    """The statement remained unavailable through every configured poll."""


class FlexMalformedStatementError(FlexError):
    """A response or statement could not be safely interpreted."""


class FlexUnsafeResponseURLError(FlexError):
    """IBKR returned a statement URL outside the configured trust boundary."""


class FlexTransportError(FlexError):
    """The HTTPS exchange failed before a valid Flex document was available."""


@dataclass(frozen=True)
class FlexConfig:
    """Validated, repr-safe configuration for read-only Flex retrieval."""

    token: str = field(repr=False)
    query_id: str = field(repr=False)
    account_salt: str = field(repr=False)
    send_url: str = DEFAULT_SEND_URL
    statement_url: str = DEFAULT_STATEMENT_URL
    max_polls: int = 5
    request_timeout_seconds: float = 30.0
    max_response_bytes: int = 5 * 1024 * 1024
    user_agent: str = "news-bot-research/1.0"
    allowed_statement_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS
    ssl_required: bool = True
    max_staleness_hours: float = 24.0

    def __post_init__(self) -> None:
        for name in ("token", "query_id", "account_salt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonblank")
            _utf8_bytes(value, name)
        if not isinstance(self.max_polls, int) or isinstance(self.max_polls, bool) or self.max_polls <= 0:
            raise ValueError("max_polls must be a positive integer")
        if not _positive_finite(self.request_timeout_seconds):
            raise ValueError("request_timeout_seconds must be positive and finite")
        if not isinstance(self.max_response_bytes, int) or isinstance(self.max_response_bytes, bool) or self.max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        if not isinstance(self.user_agent, str) or not self.user_agent.strip():
            raise ValueError("user_agent must be nonblank")
        _utf8_bytes(self.user_agent, "user_agent")
        if not self.allowed_statement_hosts or any(
            not isinstance(host, str) or not host.strip()
            for host in self.allowed_statement_hosts
        ):
            raise ValueError("allowed_statement_hosts must contain nonblank hosts")
        for host in self.allowed_statement_hosts:
            _utf8_bytes(host, "allowed_statement_hosts")
        if self.ssl_required is not True:
            raise ValueError("ssl_required must be True")
        if not _positive_finite(self.max_staleness_hours):
            raise ValueError("max_staleness_hours must be positive and finite")
        for name in ("send_url", "statement_url"):
            value = getattr(self, name)
            if isinstance(value, str):
                _utf8_bytes(value, name)
        _validate_url(self.send_url, self, configuration=True)
        _validate_url(self.statement_url, self, configuration=True)


@dataclass(frozen=True)
class PortfolioSyncResult:
    """Immutable normalized output from one Flex statement."""

    snapshot: PortfolioSnapshot
    positions: tuple[Position, ...]
    account_ref: str


def _utf8_bytes(
    value: str,
    field_name: str,
    error_type: type[Exception] = ValueError,
) -> bytes:
    """Encode trusted text or raise a fixed-message error without a raw cause."""
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        raise error_type(f"{field_name} must contain valid UTF-8 text") from None


def _positive_finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return False
    try:
        return math.isfinite(float(value)) and value > 0
    except (OverflowError, TypeError, ValueError):
        return False


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_text(root: ET.Element, name: str) -> str | None:
    values = [
        (child.text or "").strip()
        for child in root
        if _local_name(child.tag) == name
    ]
    if len(values) > 1:
        raise FlexMalformedStatementError(f"Ambiguous {name} in Flex response")
    return values[0] if values else None


def _safe_xml_root(xml: str | bytes, max_bytes: int) -> ET.Element:
    if isinstance(xml, str):
        raw = _utf8_bytes(xml, "Flex XML", FlexMalformedStatementError)
    elif isinstance(xml, bytes):
        raw = xml
    else:
        raise TypeError("Flex XML must be str or bytes")
    if len(raw) > max_bytes:
        raise FlexTransportError("IBKR Flex response exceeded the configured size limit")
    upper = raw.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise FlexMalformedStatementError("Unsafe XML declarations are not allowed")
    try:
        return ET.fromstring(raw)
    except (ET.ParseError, UnicodeError):
        raise FlexMalformedStatementError("Malformed IBKR Flex XML") from None


def _validate_url(url: str, config: FlexConfig, *, configuration: bool = False) -> str:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except (TypeError, ValueError):
        if configuration:
            raise ValueError("Flex endpoint must be a valid URL") from None
        raise FlexUnsafeResponseURLError("IBKR returned an unsafe statement URL") from None
    allowed_hosts = {host.strip().lower() for host in config.allowed_statement_hosts}
    invalid = (
        parsed.scheme.lower() != "https"
        or parsed.hostname is None
        or parsed.hostname.lower() not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or bool(parsed.fragment)
        or bool(parsed.query)
        or not parsed.path
    )
    if invalid:
        if configuration:
            raise ValueError("Flex endpoint must be an allowed HTTPS URL")
        raise FlexUnsafeResponseURLError("IBKR returned an unsafe statement URL")
    return url


def _parse_timestamp(value: str, field_name: str) -> datetime:
    stripped = value.strip()
    formats = (
        "%Y%m%d;%H%M%S",
        "%Y-%m-%d;%H:%M:%S",
        "%Y%m%d %H%M%S",
        "%Y%m%d",
        "%Y-%m-%d",
    )
    for date_format in formats:
        try:
            return datetime.strptime(stripped, date_format).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
    except ValueError:
        raise FlexMalformedStatementError(f"Invalid {field_name} timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FlexMalformedStatementError(f"{field_name} timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _required_decimal(element: ET.Element, names: tuple[str, ...], label: str) -> Decimal:
    raw = next((element.get(name) for name in names if element.get(name) is not None), None)
    if raw is None or not raw.strip():
        raise FlexMalformedStatementError(f"Missing required {label}")
    try:
        value = Decimal(raw.strip())
    except InvalidOperation:
        raise FlexMalformedStatementError(f"Invalid decimal for {label}") from None
    if not value.is_finite():
        raise FlexMalformedStatementError(f"{label} must be finite")
    return value


def _optional_decimal(element: ET.Element, names: tuple[str, ...], label: str) -> Decimal | None:
    if not any(element.get(name) is not None for name in names):
        return None
    return _required_decimal(element, names, label)


def portfolio_is_stale(
    snapshot_as_of: datetime,
    now: datetime,
    max_hours: int | float | Decimal,
) -> bool:
    """Return whether age exceeds the limit; tolerate at most five minutes of skew."""
    for name, value in (("snapshot_as_of", snapshot_as_of), ("now", now)):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} must be timezone-aware")
    if not _positive_finite(max_hours):
        raise ValueError("max_hours must be positive and finite")
    snapshot_utc = snapshot_as_of.astimezone(timezone.utc)
    now_utc = now.astimezone(timezone.utc)
    if snapshot_utc - now_utc > _FUTURE_CLOCK_TOLERANCE:
        raise ValueError("snapshot timestamp is materially in the future")
    if snapshot_utc > now_utc:
        return False
    return now_utc - snapshot_utc > timedelta(hours=float(max_hours))


def parse_statement(
    xml: str | bytes,
    *,
    account_salt: str,
    now: datetime | None = None,
    max_staleness_hours: int | float | Decimal | None = None,
    max_bytes: int = 5 * 1024 * 1024,
) -> PortfolioSyncResult:
    """Normalize one complete FlexQueryResponse without retaining account IDs."""
    if not isinstance(account_salt, str) or not account_salt.strip():
        raise ValueError("account_salt must be nonblank")
    account_salt_bytes = _utf8_bytes(
        account_salt, "account_salt", FlexMalformedStatementError
    )
    root = _safe_xml_root(xml, max_bytes)
    if _local_name(root.tag) != "FlexQueryResponse":
        raise FlexMalformedStatementError("Expected a FlexQueryResponse statement")
    statements = [node for node in root.iter() if _local_name(node.tag) == "FlexStatement"]
    if len(statements) != 1:
        raise FlexMalformedStatementError("Statement must contain exactly one account statement")
    statement = statements[0]
    account_information = [
        node for node in statement.iter() if _local_name(node.tag) == "AccountInformation"
    ]
    if len(account_information) != 1:
        raise FlexMalformedStatementError("Statement must contain exactly one AccountInformation record")
    account_info = account_information[0]
    open_positions = [node for node in statement.iter() if _local_name(node.tag) == "OpenPosition"]
    account_ids = {
        value.strip()
        for value in (
            statement.get("accountId"),
            account_info.get("accountId"),
            *(node.get("accountId") for node in open_positions),
        )
        if value is not None and value.strip()
    }
    if len(account_ids) != 1:
        raise FlexMalformedStatementError("Missing or ambiguous account identifier")
    account_id = next(iter(account_ids))
    base_currency = (account_info.get("currency") or "").strip().upper()
    if not base_currency:
        raise FlexMalformedStatementError("Missing required base currency")

    summaries = [
        node
        for node in statement.iter()
        if _local_name(node.tag) == "EquitySummaryByReportDateInBase"
    ]
    if not summaries:
        raise FlexMalformedStatementError("Missing required equity summary")
    dated_summaries: list[tuple[datetime, ET.Element]] = []
    for summary in summaries:
        report_date = summary.get("reportDate")
        if not report_date:
            raise FlexMalformedStatementError("Missing equity summary report date")
        dated_summaries.append((_parse_timestamp(report_date, "reportDate"), summary))
    latest_date = max(item[0] for item in dated_summaries)
    latest = [item[1] for item in dated_summaries if item[0] == latest_date]
    if len(latest) != 1:
        raise FlexMalformedStatementError("Ambiguous latest equity summary")
    summary = latest[0]
    nav = _required_decimal(summary, ("total", "netLiquidationValue"), "NAV")
    cash = _required_decimal(summary, ("cash", "cashBalance", "totalCashValue"), "cash")

    generated = statement.get("whenGenerated") or statement.get("dateTime")
    if not generated:
        raise FlexMalformedStatementError("Missing required statement as-of timestamp")
    as_of = _parse_timestamp(generated, "statement as-of")
    account_ref = "acct_" + hmac.new(
        account_salt_bytes, account_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()[:24]
    snapshot_digest = hashlib.sha256(
        f"{account_ref}|{as_of.isoformat()}".encode("utf-8")
    ).hexdigest()[:24]
    snapshot_id = f"snapshot_{snapshot_digest}"

    positions: list[Position] = []
    position_ids: set[str] = set()
    for node in open_positions:
        symbol = (node.get("symbol") or "").strip()
        currency = (node.get("currency") or base_currency).strip().upper()
        if not symbol or not currency:
            raise FlexMalformedStatementError("OpenPosition requires symbol and currency")
        quantity = _required_decimal(node, ("position", "quantity"), "position quantity")
        market_value = _required_decimal(
            node, ("positionValue", "marketValue"), "position market value"
        )
        cost_basis = _optional_decimal(
            node, ("costBasisMoney", "costBasis"), "position cost basis"
        )
        identifiers = tuple(
            f"{name}:{node.get(name).strip()}"
            for name in ("conid", "isin", "cusip")
            if node.get(name) is not None and node.get(name).strip()
        )
        identity = "|".join(identifiers) or "|".join(
            ((node.get("assetCategory") or "").strip(), symbol, currency)
        )
        position_id = "position_" + hashlib.sha256(
            f"{snapshot_id}|{identity}".encode("utf-8")
        ).hexdigest()[:24]
        if position_id in position_ids:
            raise FlexMalformedStatementError("Duplicate or ambiguous OpenPosition")
        position_ids.add(position_id)
        positions.append(
            Position(
                position_id=position_id,
                snapshot_id=snapshot_id,
                symbol=symbol,
                quantity=quantity,
                market_value=market_value,
                currency=currency,
                cost_basis=cost_basis,
            )
        )

    if (now is None) != (max_staleness_hours is None):
        raise ValueError("now and max_staleness_hours must be provided together")
    is_stale = (
        portfolio_is_stale(as_of, now, max_staleness_hours)
        if now is not None and max_staleness_hours is not None
        else False
    )
    snapshot = PortfolioSnapshot(
        snapshot_id=snapshot_id,
        as_of=as_of,
        base_currency=base_currency,
        nav=nav,
        cash=cash,
        is_stale=is_stale,
    )
    return PortfolioSyncResult(snapshot, tuple(positions), account_ref)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FlexClient:
    """GET-only client for retrieving and normalizing one configured Flex query."""

    def __init__(
        self,
        config: FlexConfig,
        *,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.config = config
        self.session = session if session is not None else requests.Session()
        self.sleeper = sleeper
        self.clock = clock

    def _get(self, url: str, params: dict[str, str]) -> bytes:
        try:
            response = self.session.get(
                url,
                params=params,
                headers={"User-Agent": self.config.user_agent},
                timeout=self.config.request_timeout_seconds,
                allow_redirects=False,
            )
        except requests.RequestException:
            raise FlexTransportError("IBKR Flex HTTPS request failed") from None
        if 300 <= response.status_code < 400:
            raise FlexTransportError("IBKR Flex redirects are not allowed")
        if not 200 <= response.status_code < 300:
            raise FlexTransportError(
                f"IBKR Flex HTTP response status {response.status_code}"
            )
        content = response.content
        if len(content) > self.config.max_response_bytes:
            raise FlexTransportError("IBKR Flex response exceeded the configured size limit")
        return content

    def sync(self, store: ResearchStore | None = None) -> PortfolioSyncResult:
        """Retrieve, parse, and optionally atomically persist one statement."""
        send_body = self._get(
            self.config.send_url,
            {"t": self.config.token, "q": self.config.query_id, "v": "3"},
        )
        send_root = _safe_xml_root(send_body, self.config.max_response_bytes)
        if _local_name(send_root.tag) != "FlexStatementResponse":
            raise FlexMalformedStatementError("Unexpected SendRequest response type")
        send_error = _direct_text(send_root, "ErrorCode")
        if send_error:
            try:
                raise FlexError.from_code(int(send_error))
            except ValueError:
                raise FlexMalformedStatementError("Invalid Flex error code") from None
        if (_direct_text(send_root, "Status") or "").lower() != "success":
            raise FlexMalformedStatementError("SendRequest did not return success")
        reference_code = _direct_text(send_root, "ReferenceCode")
        statement_url = _direct_text(send_root, "Url")
        if not reference_code or not statement_url:
            raise FlexMalformedStatementError("SendRequest response is incomplete")
        statement_url = _validate_url(statement_url, self.config)

        statement_body: bytes | None = None
        for attempt in range(self.config.max_polls):
            candidate = self._get(
                statement_url,
                {"t": self.config.token, "q": reference_code, "v": "3"},
            )
            root = _safe_xml_root(candidate, self.config.max_response_bytes)
            root_name = _local_name(root.tag)
            if root_name == "FlexQueryResponse":
                statement_body = candidate
                break
            if root_name != "FlexStatementResponse":
                raise FlexMalformedStatementError("Unexpected GetStatement response type")
            error_code = _direct_text(root, "ErrorCode")
            if not error_code:
                raise FlexMalformedStatementError("GetStatement error response is incomplete")
            try:
                error = FlexError.from_code(int(error_code))
            except ValueError:
                raise FlexMalformedStatementError("Invalid Flex error code") from None
            if not isinstance(error, FlexNotReadyError):
                raise error
            if attempt + 1 == self.config.max_polls:
                raise FlexPollingExhaustedError(
                    f"IBKR Flex statement was not ready after {self.config.max_polls} polls",
                    code=1019,
                )
            self.sleeper(min(2**attempt, 16))

        if statement_body is None:  # Defensive; the loop exits by return/error above.
            raise FlexPollingExhaustedError("IBKR Flex polling ended without a statement")
        result = parse_statement(
            statement_body,
            account_salt=self.config.account_salt,
            now=self.clock(),
            max_staleness_hours=self.config.max_staleness_hours,
            max_bytes=self.config.max_response_bytes,
        )
        if store is not None:
            store.insert_portfolio_snapshot(
                result.snapshot, result.positions, result.account_ref
            )
        return result
