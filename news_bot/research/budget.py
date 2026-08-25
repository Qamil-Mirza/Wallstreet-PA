"""Transactional monthly spending controls for paid model calls."""

import sqlite3
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import ROUND_UP, Decimal
from types import MappingProxyType

from .models import AgentRole
from .store import ResearchStore


_MICRODOLLAR = Decimal("0.000001")
_ONE_MILLION = Decimal("1000000")
_COUNTED_STATES = ("reserved", "usage_unknown", "reconciled")
_CRITICAL_ROLES = {
    AgentRole.SKEPTICAL_REVIEWER,
    AgentRole.RESEARCH_EDITOR,
}


class BudgetError(Exception):
    """Base class for budget and pricing policy failures."""


class BudgetExceeded(BudgetError):
    """Raised when counted monthly spend would exceed the hard ceiling."""


class SoftBudgetExceeded(BudgetError):
    """Raised when a noncritical call would cross the soft ceiling."""


class ExpiredPricing(BudgetError):
    """Raised when estimates would rely on an expired price table."""


class UnknownModelPrice(BudgetError):
    """Raised when no approved price exists for a model."""


class InvalidBudgetValue(BudgetError, ValueError):
    """Raised when a public budget value violates its typed boundary."""


class InvalidReservationState(BudgetError):
    """Raised when a reservation cannot make the requested transition."""


class ReservationNotFound(BudgetError):
    """Raised when a reservation identifier is unknown."""


@dataclass(frozen=True)
class ModelPrice:
    """Exact per-million-token rates for one model."""

    input_per_million: Decimal
    output_per_million: Decimal

    def __post_init__(self) -> None:
        _require_amount("input_per_million", self.input_per_million)
        _require_amount("output_per_million", self.output_per_million)


@dataclass(frozen=True)
class Reservation:
    """Immutable view of one persisted budget reservation."""

    id: str
    owner_key: str
    amount: Decimal
    state: str
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise InvalidBudgetValue("reservation id must be a non-empty string")
        if not isinstance(self.owner_key, str) or not self.owner_key.strip():
            raise InvalidBudgetValue("owner_key must be a non-empty string")
        _require_amount("amount", self.amount)
        _require_utc("created_at", self.created_at)
        _require_utc("updated_at", self.updated_at)


class PriceTable:
    """Dated model prices used for pessimistic call estimates."""

    def __init__(
        self, effective_until: date, prices: Mapping[str, ModelPrice]
    ) -> None:
        if type(effective_until) is not date:
            raise InvalidBudgetValue("effective_until must be date")
        if not isinstance(prices, Mapping):
            raise InvalidBudgetValue("prices must be a mapping")
        for model, price in prices.items():
            if not isinstance(model, str) or not model.strip():
                raise InvalidBudgetValue("price model names must be non-empty strings")
            if not isinstance(price, ModelPrice):
                raise InvalidBudgetValue("price entries must be ModelPrice")
        self.effective_until = effective_until
        self.prices = MappingProxyType(dict(prices))

    def estimate(
        self,
        model: str,
        input_tokens: int,
        max_output_tokens: int,
        *,
        as_of: date | None = None,
    ) -> Decimal:
        """Return a pessimistic token-cost estimate rounded up to a microdollar."""
        estimate_date = date.today() if as_of is None else as_of
        if type(estimate_date) is not date:
            raise InvalidBudgetValue("as_of must be date")
        if estimate_date > self.effective_until:
            raise ExpiredPricing(
                f"price table expired on {self.effective_until.isoformat()}"
            )
        _require_tokens("input_tokens", input_tokens)
        _require_tokens("max_output_tokens", max_output_tokens)
        try:
            price = self.prices[model]
        except (KeyError, TypeError) as exc:
            raise UnknownModelPrice(f"no approved price for model {model!r}") from exc
        raw = (
            Decimal(input_tokens) * price.input_per_million
            + Decimal(max_output_tokens) * price.output_per_million
        ) / _ONE_MILLION
        return raw.quantize(_MICRODOLLAR, rounding=ROUND_UP)


class BudgetLedger:
    """SQLite-backed strict UTC calendar-month spending ledger.

    Counted spend belongs to the UTC month in which its reservation was created,
    even when reconciliation occurs later. Released reservations count as zero.
    """

    def __init__(
        self,
        store: ResearchStore,
        soft_limit: Decimal,
        hard_limit: Decimal,
    ) -> None:
        _require_amount("soft_limit", soft_limit)
        _require_amount("hard_limit", hard_limit)
        if not soft_limit < hard_limit <= Decimal("5.00"):
            raise InvalidBudgetValue(
                "limits must satisfy 0 <= soft_limit < hard_limit <= 5.00"
            )
        self.store = store
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit

    def reserve(
        self,
        owner_key: str,
        estimated_cost: Decimal,
        *,
        now: datetime | None = None,
        role: AgentRole | None = None,
    ) -> Reservation:
        """Atomically reserve pessimistic cost without crossing policy ceilings."""
        if not isinstance(owner_key, str) or not owner_key.strip():
            raise InvalidBudgetValue("owner_key must be a non-empty string")
        _require_amount("estimated_cost", estimated_cost)
        instant = _require_utc("now", now)
        if role is not None and not isinstance(role, AgentRole):
            raise InvalidBudgetValue("role must be AgentRole or None")
        month_start, month_end = _month_bounds(instant.year, instant.month)
        reservation_id = str(uuid.uuid4())

        with self.store.transaction() as connection:
            current = self._counted_total(connection, month_start, month_end)
            projected = current + estimated_cost
            if projected > self.hard_limit:
                raise BudgetExceeded(
                    f"monthly hard limit {self.hard_limit} would be exceeded"
                )
            if (
                role not in _CRITICAL_ROLES
                and (current >= self.soft_limit or projected > self.soft_limit)
            ):
                raise SoftBudgetExceeded(
                    f"monthly soft limit {self.soft_limit} requires a critical role"
                )
            timestamp = _utc_text(instant)
            connection.execute(
                "INSERT INTO budget_reservations ("
                "reservation_id, owner_key, amount_usd, state, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    reservation_id,
                    owner_key,
                    _decimal_text(estimated_cost),
                    "reserved",
                    timestamp,
                    timestamp,
                ),
            )
        return Reservation(
            id=reservation_id,
            owner_key=owner_key,
            amount=estimated_cost,
            state="reserved",
            created_at=instant,
            updated_at=instant,
        )

    def reconcile(
        self,
        reservation_id: str,
        actual_cost: Decimal,
        *,
        now: datetime | None = None,
    ) -> Reservation:
        """Replace a reserved estimate with provider-reported actual truth."""
        _require_amount("actual_cost", actual_cost)
        instant = _require_utc("now", now)
        overage = False
        with self.store.transaction() as connection:
            current = self._reservation_in(connection, reservation_id)
            if current.state == "reconciled":
                if current.amount == actual_cost:
                    return current
                raise InvalidReservationState(
                    "reservation is already reconciled with a different cost"
                )
            if current.state != "reserved":
                raise InvalidReservationState(
                    f"cannot reconcile reservation in state {current.state!r}"
                )
            timestamp = _utc_text(instant)
            connection.execute(
                "UPDATE budget_reservations "
                "SET amount_usd = ?, state = ?, updated_at = ? "
                "WHERE reservation_id = ?",
                (
                    _decimal_text(actual_cost),
                    "reconciled",
                    timestamp,
                    reservation_id,
                ),
            )
            reconciled = Reservation(
                id=current.id,
                owner_key=current.owner_key,
                amount=actual_cost,
                state="reconciled",
                created_at=current.created_at,
                updated_at=instant,
            )
            month_start, month_end = _month_bounds(
                current.created_at.year, current.created_at.month
            )
            overage = (
                self._counted_total(connection, month_start, month_end)
                > self.hard_limit
            )
        if overage:
            raise BudgetExceeded(
                "recorded provider overage above the monthly hard limit"
            )
        return reconciled

    def mark_usage_unknown(
        self, reservation_id: str, *, now: datetime | None = None
    ) -> Reservation:
        """Keep an uncertain provider outcome fully counted."""
        return self._transition_reserved(
            reservation_id, "usage_unknown", now=now
        )

    def release(
        self, reservation_id: str, *, now: datetime | None = None
    ) -> Reservation:
        """Release only a definitively unspent reserved amount."""
        return self._transition_reserved(reservation_id, "released", now=now)

    def get_reservation(self, reservation_id: str) -> Reservation:
        """Load one immutable reservation view and close the read connection."""
        connection = self.store.connect()
        try:
            return self._reservation_in(connection, reservation_id)
        finally:
            connection.close()

    def month_total(self, year: int, month: int) -> Decimal:
        """Return conservative counted spend for one UTC reservation month."""
        month_start, month_end = _month_bounds(year, month)
        connection = self.store.connect()
        try:
            return self._counted_total(connection, month_start, month_end)
        finally:
            connection.close()

    def _transition_reserved(
        self,
        reservation_id: str,
        target_state: str,
        *,
        now: datetime | None,
    ) -> Reservation:
        instant = _require_utc("now", now)
        with self.store.transaction() as connection:
            current = self._reservation_in(connection, reservation_id)
            if current.state == target_state:
                return current
            if current.state != "reserved":
                raise InvalidReservationState(
                    f"cannot transition {current.state!r} to {target_state!r}"
                )
            connection.execute(
                "UPDATE budget_reservations SET state = ?, updated_at = ? "
                "WHERE reservation_id = ?",
                (target_state, _utc_text(instant), reservation_id),
            )
            return Reservation(
                id=current.id,
                owner_key=current.owner_key,
                amount=current.amount,
                state=target_state,
                created_at=current.created_at,
                updated_at=instant,
            )

    @staticmethod
    def _reservation_in(
        connection: sqlite3.Connection, reservation_id: str
    ) -> Reservation:
        row = connection.execute(
            "SELECT reservation_id, owner_key, amount_usd, state, "
            "created_at, updated_at FROM budget_reservations "
            "WHERE reservation_id = ?",
            (reservation_id,),
        ).fetchone()
        if row is None:
            raise ReservationNotFound(f"unknown reservation: {reservation_id}")
        return Reservation(
            id=row[0],
            owner_key=row[1],
            amount=Decimal(row[2]),
            state=row[3],
            created_at=_parse_utc(row[4]),
            updated_at=_parse_utc(row[5]),
        )

    @staticmethod
    def _counted_total(
        connection: sqlite3.Connection,
        month_start: datetime,
        month_end: datetime,
    ) -> Decimal:
        placeholders = ", ".join("?" for _ in _COUNTED_STATES)
        rows = connection.execute(
            "SELECT amount_usd FROM budget_reservations "
            f"WHERE state IN ({placeholders}) "
            "AND created_at >= ? AND created_at < ?",
            (*_COUNTED_STATES, _utc_text(month_start), _utc_text(month_end)),
        ).fetchall()
        return sum((Decimal(row[0]) for row in rows), Decimal("0"))


def _require_amount(name: str, value: Decimal) -> None:
    if not isinstance(value, Decimal):
        raise InvalidBudgetValue(f"{name} must be Decimal")
    if not value.is_finite() or value < 0:
        raise InvalidBudgetValue(f"{name} must be finite and non-negative")


def _require_tokens(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise InvalidBudgetValue(f"{name} must be a non-negative integer")


def _require_utc(name: str, value: datetime | None) -> datetime:
    instant = datetime.now(timezone.utc) if value is None else value
    if not isinstance(instant, datetime):
        raise InvalidBudgetValue(f"{name} must be datetime")
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise InvalidBudgetValue(f"{name} must be timezone-aware")
    return instant.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    try:
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    except (TypeError, ValueError) as exc:
        raise InvalidBudgetValue("year and month must identify a calendar month") from exc
    return start, end
