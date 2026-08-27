"""Personal-use Financial Modeling Prep fundamentals adapter."""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

import requests

from ..evidence import DocumentInput
from .base import (
    ConnectorBatch,
    ConnectorCheckpoint,
    ConnectorError,
    NormalizedResearchDocument,
)


_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9.-]{0,15}")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|apikey|x-api-key)\b['\"]?\s*[:=]\s*['\"]?)([^'\"&,\s}\]]+)"
)
_BEARER = re.compile(
    r"(?i)(\bauthorization\b['\"]?\s*[:=]\s*['\"]?bearer\s+)([^'\",\s}\]]+)"
)
_REDACTED = "[REDACTED]"
_LOG_FACTORY_LOCK = threading.RLock()
_FACTORY_MARKER = "_newsletter_fmp_redaction_factory"
_CONFIGURED_SECRETS: set[str] = set()


def _redact_text(value: str) -> str:
    redacted = _SECRET_ASSIGNMENT.sub(rf"\1{_REDACTED}", value)
    redacted = _BEARER.sub(rf"\1{_REDACTED}", redacted)
    for secret in sorted(
        (item for item in _CONFIGURED_SECRETS if len(item) >= 8),
        key=len,
        reverse=True,
    ):
        redacted = redacted.replace(secret, _REDACTED)
    return redacted


def _redact_log_value(value):
    if isinstance(value, str):
        if value in _CONFIGURED_SECRETS:
            return _REDACTED
        return _redact_text(value)
    if isinstance(value, Mapping):
        changed = False
        redacted: dict[object, object] = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("_", "-")
            if normalized in {"apikey", "api-key", "x-api-key"}:
                replacement = _REDACTED
            elif normalized == "authorization" and isinstance(item, str) and re.match(
                r"(?i)^\s*(?:bearer|basic)\s+\S+", item
            ):
                replacement = _REDACTED
            else:
                replacement = _redact_log_value(item)
            changed = changed or replacement != item
            redacted[key] = replacement
        return redacted if changed else value
    if isinstance(value, tuple):
        redacted = tuple(_redact_log_value(item) for item in value)
        return redacted if redacted != value else value
    if isinstance(value, list):
        redacted = [_redact_log_value(item) for item in value]
        return redacted if redacted != value else value
    return value


def install_fmp_log_redaction(secret: str | None = None) -> None:
    """Chain one process-wide secret-safe LogRecord factory."""
    with _LOG_FACTORY_LOCK:
        if isinstance(secret, str) and secret:
            _CONFIGURED_SECRETS.add(secret)
        prior_factory = logging.getLogRecordFactory()
        if getattr(prior_factory, _FACTORY_MARKER, False):
            return

        def redacting_factory(*args, **kwargs):
            record = prior_factory(*args, **kwargs)
            original_message = record.msg
            original_args = record.args
            record.msg = _redact_log_value(record.msg)
            record.args = _redact_log_value(record.args)
            if record.exc_text:
                record.exc_text = _redact_text(record.exc_text)
            if record.exc_info and record.exc_info[1] is not None:
                exception = record.exc_info[1]
                unsafe = str(exception)
                safe = _redact_text(unsafe)
                if safe != unsafe:
                    safe_exception = RuntimeError(safe)
                    record.exc_info = (RuntimeError, safe_exception, None)
                    record.exc_text = None
            if record.msg == original_message and record.args == original_args:
                record.msg = original_message
                record.args = original_args
            return record

        setattr(redacting_factory, _FACTORY_MARKER, True)
        logging.setLogRecordFactory(redacting_factory)


def _decimal(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field_name} is unavailable")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{field_name} is invalid") from None
    if not parsed.is_finite():
        raise ValueError(f"{field_name} is invalid")
    return parsed


def _normalized_symbol(value: str | None) -> str:
    normalized = value.strip().upper() if isinstance(value, str) else ""
    if _SYMBOL.fullmatch(normalized) is None:
        raise ValueError("FMP symbol is invalid")
    return normalized


@dataclass(frozen=True)
class FMPConfig:
    """FMP settings that can safely represent a disabled provider."""

    api_key: str | None = field(repr=False)
    base_url: str = field(
        default="https://financialmodelingprep.com/stable", repr=False
    )
    timeout_seconds: float = 10.0
    max_response_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key.strip()
        ):
            raise ValueError("FMP api_key must be nonblank when configured")
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("FMP base_url must be nonblank text")
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ValueError("FMP timeout_seconds must be positive")
        if not isinstance(self.max_response_bytes, int) or self.max_response_bytes <= 0:
            raise ValueError("FMP max_response_bytes must be positive")
        if self.transport_allowed:
            object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @property
    def transport_allowed(self) -> bool:
        try:
            parsed = urlsplit(self.base_url)
            return (
                parsed.scheme == "https"
                and parsed.hostname == "financialmodelingprep.com"
                and parsed.port in (None, 443)
                and parsed.username is None
                and parsed.password is None
                and not parsed.query
                and not parsed.fragment
                and parsed.path.rstrip("/") == "/stable"
            )
        except ValueError:
            return False

    @property
    def unavailable_reason(self) -> str | None:
        if self.api_key is None:
            return "configuration_missing"
        if not self.transport_allowed:
            return "endpoint_disallowed"
        return None


@dataclass(frozen=True)
class DatedPrice:
    """A provider price that retains its effective market date."""

    symbol: str
    value: Decimal
    effective_date: date

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or _SYMBOL.fullmatch(self.symbol) is None:
            raise ValueError("DatedPrice.symbol is invalid")
        if not isinstance(self.value, Decimal):
            raise TypeError("DatedPrice.value must be Decimal")
        if not self.value.is_finite():
            raise ValueError("DatedPrice.value must be finite")
        if not isinstance(self.effective_date, date) or isinstance(
            self.effective_date, datetime
        ):
            raise TypeError("DatedPrice.effective_date must be date")


@dataclass(frozen=True)
class FundamentalPacket:
    """Normalized values with explicit provider and fallback availability."""

    price: DatedPrice | None
    financial_period: str | None
    financial_effective_date: date | None
    statements: Mapping[str, Decimal]
    ratios: Mapping[str, Decimal]
    metadata: Mapping[str, str | Decimal]
    unavailable: Mapping[str, str]
    use_sec_fallback: bool

    def __post_init__(self) -> None:
        for field_name in ("statements", "ratios", "metadata", "unavailable"):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{field_name} must be a mapping")
            object.__setattr__(self, field_name, MappingProxyType(dict(value)))


class FMPConnector:
    """Read-only adapter for dated quotes and income-statement fundamentals."""

    name = "fmp"

    def __init__(
        self,
        config: FMPConfig | None,
        *,
        symbol: str | None = None,
        session=None,
        sec_facts_fallback: Callable[[str], Mapping[str, Decimal]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if config is not None and not isinstance(config, FMPConfig):
            raise TypeError("config must be FMPConfig or None")
        if symbol is not None:
            _normalized_symbol(symbol)
        if sec_facts_fallback is not None and not callable(sec_facts_fallback):
            raise TypeError("sec_facts_fallback must be callable")
        install_fmp_log_redaction(config.api_key if config is not None else None)
        self.config = config
        self.symbol = symbol.strip().upper() if symbol is not None else None
        self._sec_facts_fallback = sec_facts_fallback
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._owns_session = session is None
        self.session = session or requests.Session()
        if self._owns_session:
            self.session.trust_env = False

    def close(self) -> None:
        """Close only a session created and owned by this connector."""
        if self._owns_session:
            self.session.close()

    def _availability_reason(self) -> str | None:
        if self.config is None:
            return "configuration_missing"
        return self.config.unavailable_reason

    def _request(self, endpoint: str, symbol: str):
        normalized = _normalized_symbol(symbol)
        reason = self._availability_reason()
        if reason is not None:
            raise ConnectorError(self.name, retryable=False, diagnostic_code=reason)
        assert self.config is not None
        url = f"{self.config.base_url}/{endpoint}"
        try:
            response = self.session.get(
                url,
                params={"symbol": normalized, "apikey": self.config.api_key},
                timeout=self.config.timeout_seconds,
                allow_redirects=False,
                stream=True,
            )
        except Exception:
            raise ConnectorError(
                self.name, retryable=True, diagnostic_code="transport_error"
            ) from None
        try:
            final = urlsplit(getattr(response, "url", ""))
            if (
                getattr(response, "history", ())
                or final.scheme != "https"
                or final.hostname != "financialmodelingprep.com"
                or final.port not in (None, 443)
            ):
                raise ConnectorError(
                    self.name, retryable=False, diagnostic_code="redirect_rejected"
                )
            status = int(response.status_code)
            if not 200 <= status < 300:
                raise ConnectorError(
                    self.name,
                    status=status,
                    retryable=status in {408, 425, 429} or 500 <= status <= 599,
                    diagnostic_code=f"http_{status}",
                )
            raw = bytearray()
            try:
                for chunk in response.iter_content(chunk_size=65_536):
                    raw.extend(chunk)
                    if len(raw) > self.config.max_response_bytes:
                        raise ConnectorError(
                            self.name,
                            retryable=False,
                            diagnostic_code="response_too_large",
                        )
                payload = json.loads(bytes(raw).decode("utf-8", errors="strict"))
            except ConnectorError:
                raise
            except (UnicodeError, json.JSONDecodeError):
                raise ConnectorError(
                    self.name, retryable=False, diagnostic_code="invalid_json"
                ) from None
            except Exception:
                raise ConnectorError(
                    self.name, retryable=True, diagnostic_code="response_read_failed"
                ) from None
            if not isinstance(payload, (list, dict)):
                raise ConnectorError(
                    self.name, retryable=False, diagnostic_code="invalid_payload"
                )
            return payload
        finally:
            try:
                response.close()
            except Exception:
                pass

    def _validated_rows(
        self,
        payload,
        *,
        expected_symbol: str | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        if isinstance(payload, Mapping):
            lowered = {str(key).strip().lower() for key in payload}
            diagnostic = (
                "provider_error"
                if lowered & {"error", "error message", "error_message"}
                else "invalid_payload"
            )
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code=diagnostic
            )
        if not isinstance(payload, (list, tuple)):
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="invalid_payload"
            )
        rows: list[Mapping[str, Any]] = []
        normalized_expected = (
            _normalized_symbol(expected_symbol) if expected_symbol is not None else None
        )
        for item in payload:
            if not isinstance(item, Mapping):
                raise ConnectorError(
                    self.name, retryable=False, diagnostic_code="invalid_payload"
                )
            lowered = {str(key).strip().lower() for key in item}
            if lowered & {"error", "error message", "error_message"}:
                raise ConnectorError(
                    self.name, retryable=False, diagnostic_code="provider_error"
                )
            if normalized_expected is not None:
                try:
                    received = _normalized_symbol(str(item["symbol"]))
                except (KeyError, TypeError, ValueError):
                    raise ConnectorError(
                        self.name, retryable=False, diagnostic_code="invalid_identity"
                    ) from None
                if received != normalized_expected:
                    raise ConnectorError(
                        self.name, retryable=False, diagnostic_code="identity_mismatch"
                    )
            rows.append(item)
        return tuple(rows)

    def fetch_quote(self, symbol: str):
        normalized = _normalized_symbol(symbol)
        return self._validated_rows(
            self._request("quote", normalized), expected_symbol=normalized
        )

    def fetch_income_statement(self, symbol: str):
        normalized = _normalized_symbol(symbol)
        return self._validated_rows(
            self._request("income-statement", normalized), expected_symbol=normalized
        )

    def _resolve_symbol(self, value: str | None = None) -> str:
        return _normalized_symbol(value or self.symbol)

    def _apply_sec_fallback(
        self,
        symbol: str,
        reason: str,
        packet: FundamentalPacket | None = None,
    ) -> FundamentalPacket:
        statements = dict(packet.statements) if packet is not None else {}
        ratios = dict(packet.ratios) if packet is not None else {}
        metadata = dict(packet.metadata) if packet is not None else {}
        unavailable = dict(packet.unavailable) if packet is not None else {
            "price": "provider_value_missing",
            "fundamentals": "provider_value_missing",
        }
        unavailable["fmp"] = reason
        if self._sec_facts_fallback is not None:
            try:
                facts = self._sec_facts_fallback(symbol)
                if not isinstance(facts, Mapping):
                    raise TypeError
                accepted = False
                for key, value in facts.items():
                    if (
                        isinstance(key, str)
                        and key.strip()
                        and isinstance(value, Decimal)
                        and value.is_finite()
                    ):
                        if key not in statements:
                            statements[key] = value
                            unavailable.pop(key, None)
                            accepted = True
                    else:
                        raise ValueError
                if accepted:
                    metadata["fundamental_source"] = "sec_companyfacts"
                else:
                    raise ValueError
            except Exception:
                unavailable["sec_fallback"] = "fallback_unavailable"
        return FundamentalPacket(
            price=packet.price if packet is not None else None,
            financial_period=packet.financial_period if packet is not None else None,
            financial_effective_date=(
                packet.financial_effective_date if packet is not None else None
            ),
            statements=statements,
            ratios=ratios,
            metadata=metadata,
            unavailable=unavailable,
            use_sec_fallback=True,
        )

    def fetch_fundamentals(self, symbol: str | None = None) -> FundamentalPacket:
        normalized = self._resolve_symbol(symbol)
        reason = self._availability_reason()
        if reason is not None:
            return self._apply_sec_fallback(normalized, reason)
        try:
            packet = self.parse(
                quote=self.fetch_quote(normalized),
                statements=self.fetch_income_statement(normalized),
            )
        except ConnectorError:
            return self._apply_sec_fallback(normalized, "provider_unavailable")
        if packet.use_sec_fallback:
            return self._apply_sec_fallback(
                normalized, "provider_value_unavailable", packet
            )
        return packet

    def fetch(self, checkpoint: ConnectorCheckpoint) -> ConnectorBatch:
        if checkpoint.connector != self.name:
            raise ValueError("checkpoint belongs to another connector")
        symbol = self._resolve_symbol(checkpoint.cursor)
        packet = self.fetch_fundamentals(symbol)
        retrieved_at = self._clock()
        effective_date = packet.financial_effective_date or (
            packet.price.effective_date if packet.price is not None else None
        )
        published_at = (
            datetime.combine(effective_date, time.min, tzinfo=timezone.utc)
            if effective_date is not None
            else retrieved_at
        )
        payload = {
            "financial_effective_date": (
                packet.financial_effective_date.isoformat()
                if packet.financial_effective_date
                else None
            ),
            "financial_period": packet.financial_period,
            "metadata": {key: str(value) for key, value in packet.metadata.items()},
            "price": (
                {
                    "effective_date": packet.price.effective_date.isoformat(),
                    "symbol": packet.price.symbol,
                    "value": str(packet.price.value),
                }
                if packet.price
                else None
            ),
            "ratios": {key: str(value) for key, value in packet.ratios.items()},
            "statements": {key: str(value) for key, value in packet.statements.items()},
            "unavailable": dict(packet.unavailable),
        }
        document = NormalizedResearchDocument(
            evidence=DocumentInput(
                source_type="fmp_fundamentals",
                url=(
                    "https://financialmodelingprep.com/stable/quote"
                    f"?symbol={symbol}"
                ),
                publisher="Financial Modeling Prep",
                published_at=published_at,
                retrieved_at=retrieved_at,
                content=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            ),
            tags=("FMP", "fundamentals"),
        )
        return ConnectorBatch(
            self.name,
            (document,),
            ConnectorCheckpoint(self.name, cursor=symbol),
        )

    def parse(
        self,
        *,
        quote,
        statements,
    ) -> FundamentalPacket:
        quote = self._validated_rows(quote)
        statements = self._validated_rows(statements)
        unavailable: dict[str, str] = {}
        price: DatedPrice | None = None
        statement_values: dict[str, Decimal] = {}
        ratios: dict[str, Decimal] = {}
        metadata: dict[str, str | Decimal] = {}
        financial_period: str | None = None
        financial_effective_date: date | None = None

        quote_row = quote[0] if quote else None
        if isinstance(quote_row, Mapping):
            try:
                symbol = _normalized_symbol(str(quote_row["symbol"]))
                raw_timestamp = quote_row["timestamp"]
                if isinstance(raw_timestamp, bool) or not isinstance(
                    raw_timestamp, (int, float)
                ) or not math.isfinite(raw_timestamp):
                    raise ValueError
                effective_date = datetime.fromtimestamp(
                    raw_timestamp, tz=timezone.utc
                ).date()
                price = DatedPrice(
                    symbol, _decimal(quote_row["price"], "price"), effective_date
                )
            except (KeyError, TypeError, ValueError, OSError, OverflowError):
                unavailable["price"] = "provider_value_invalid"
            name = quote_row.get("name")
            if isinstance(name, str) and name.strip():
                metadata["company_name"] = name.strip()
            market_cap = quote_row.get("marketCap")
            if market_cap is not None:
                try:
                    metadata["market_cap"] = _decimal(market_cap, "market_cap")
                except ValueError:
                    unavailable["market_cap"] = "provider_value_invalid"
        else:
            unavailable["price"] = "provider_value_missing"

        statement = statements[0] if statements else None
        if isinstance(statement, Mapping):
            try:
                financial_effective_date = date.fromisoformat(statement.get("date"))
            except (TypeError, ValueError):
                unavailable["financial_effective_date"] = "provider_value_invalid"
            calendar_year = statement.get("calendarYear")
            period = statement.get("period")
            if isinstance(calendar_year, (str, int)) and isinstance(period, str) and period:
                financial_period = f"{calendar_year}-{period.upper()}"
            else:
                unavailable["financial_period"] = "provider_value_missing"
            currency = statement.get("reportedCurrency")
            if isinstance(currency, str) and currency.strip():
                metadata["reported_currency"] = currency.strip().upper()
            for provider_key, output_key in (
                ("revenue", "revenue"),
                ("netIncome", "net_income"),
            ):
                value = statement.get(provider_key)
                if value is None:
                    unavailable[output_key] = "provider_value_missing"
                else:
                    try:
                        statement_values[output_key] = _decimal(value, output_key)
                    except ValueError:
                        unavailable[output_key] = "provider_value_invalid"
            ratio = statement.get("grossProfitRatio")
            if ratio is None:
                unavailable["gross_profit_ratio"] = "provider_value_missing"
            else:
                try:
                    ratios["gross_profit_ratio"] = _decimal(
                        ratio, "gross_profit_ratio"
                    )
                except ValueError:
                    unavailable["gross_profit_ratio"] = "provider_value_invalid"
        else:
            unavailable["fundamentals"] = "provider_value_missing"

        return FundamentalPacket(
            price=price,
            financial_period=financial_period,
            financial_effective_date=financial_effective_date,
            statements=statement_values,
            ratios=ratios,
            metadata=metadata,
            unavailable=unavailable,
            use_sec_fallback=bool(unavailable),
        )
