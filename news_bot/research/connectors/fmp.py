"""Personal-use Financial Modeling Prep fundamentals adapter."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import requests

from .base import ConnectorError


_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9.-]{0,15}")


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


@dataclass(frozen=True)
class FMPConfig:
    """Restricted personal-use FMP endpoint configuration."""

    api_key: str | None = field(repr=False)
    base_url: str = "https://financialmodelingprep.com/stable"
    timeout_seconds: float = 10.0
    max_response_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "financialmodelingprep.com"
            or parsed.port not in (None, 443)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/") != "/stable"
        ):
            raise ValueError("FMP base_url must be the allowlisted HTTPS stable API")
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key.strip()
        ):
            raise ValueError("FMP api_key must be nonblank when configured")
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ValueError("FMP timeout_seconds must be positive")
        if not isinstance(self.max_response_bytes, int) or self.max_response_bytes <= 0:
            raise ValueError("FMP max_response_bytes must be positive")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))


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
    """Normalized FMP values with explicit field-level availability."""

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

    def __init__(self, config: FMPConfig, *, session=None) -> None:
        if not isinstance(config, FMPConfig):
            raise TypeError("config must be FMPConfig")
        self.config = config
        self.session = session or requests.Session()
        self.session.trust_env = False

    def _request(self, endpoint: str, symbol: str):
        normalized = symbol.strip().upper() if isinstance(symbol, str) else ""
        if _SYMBOL.fullmatch(normalized) is None:
            raise ValueError("FMP symbol is invalid")
        if self.config.api_key is None:
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="configuration_missing"
            )
        url = f"{self.config.base_url}/{endpoint}"
        try:
            response = self.session.get(
                url,
                params={"symbol": normalized, "apikey": self.config.api_key},
                timeout=self.config.timeout_seconds,
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException:
            raise ConnectorError(
                self.name, retryable=True, diagnostic_code="transport_error"
            ) from None
        except Exception:
            raise ConnectorError(
                self.name, retryable=True, diagnostic_code="transport_error"
            ) from None
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

    def fetch_quote(self, symbol: str):
        return self._request("quote", symbol)

    def fetch_income_statement(self, symbol: str):
        return self._request("income-statement", symbol)

    def parse(
        self,
        *,
        quote: Sequence[Mapping[str, Any]],
        statements: Sequence[Mapping[str, Any]],
    ) -> FundamentalPacket:
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
                symbol = str(quote_row["symbol"]).strip().upper()
                if _SYMBOL.fullmatch(symbol) is None:
                    raise ValueError
                raw_timestamp = quote_row["timestamp"]
                if isinstance(raw_timestamp, bool) or not isinstance(
                    raw_timestamp, (int, float)
                ) or not math.isfinite(raw_timestamp):
                    raise ValueError
                effective_date = datetime.fromtimestamp(
                    raw_timestamp, tz=timezone.utc
                ).date()
                price = DatedPrice(symbol, _decimal(quote_row["price"], "price"), effective_date)
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
            raw_statement_date = statement.get("date")
            try:
                financial_effective_date = date.fromisoformat(raw_statement_date)
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
