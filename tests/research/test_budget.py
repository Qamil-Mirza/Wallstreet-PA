"""Tests for the transactional paid-model budget ledger."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from threading import Barrier

import pytest

from news_bot.research.budget import (
    BudgetError,
    BudgetExceeded,
    BudgetIntegrityError,
    BudgetLedger,
    ExpiredPricing,
    InvalidBudgetValue,
    InvalidReservationState,
    ModelPrice,
    PriceTable,
    Reservation,
    ReservationIdExhausted,
    ReservationNotFound,
    SoftBudgetExceeded,
    UnknownModelPrice,
)
from news_bot.research.models import AgentRole

from .conftest import utc


def test_reservation_cannot_cross_hard_limit(migrated_store):
    ledger = BudgetLedger(
        migrated_store, soft_limit=Decimal("4"), hard_limit=Decimal("5")
    )
    ledger.reserve(
        "run-1",
        Decimal("4.75"),
        now=utc(2026, 8, 1),
        role=AgentRole.SKEPTICAL_REVIEWER,
    )
    with pytest.raises(BudgetExceeded):
        ledger.reserve(
            "run-2",
            Decimal("0.26"),
            now=utc(2026, 8, 2),
            role=AgentRole.SKEPTICAL_REVIEWER,
        )


def test_reconcile_uses_provider_reported_cost(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = ledger.reserve(
        "run-1", Decimal("1.00"), now=utc(2026, 8, 1)
    )
    ledger.reconcile(reservation.id, actual_cost=Decimal("0.42"))
    assert ledger.month_total(2026, 8) == Decimal("0.42")


def test_expired_price_table_blocks_paid_call(migrated_store):
    table = PriceTable(effective_until=date(2026, 8, 23), prices={})
    with pytest.raises(ExpiredPricing):
        table.estimate(
            "gpt-5.6-sol", 1000, 100, as_of=date(2026, 8, 24)
        )


def test_price_table_estimates_pessimistically_with_exact_rounding():
    table = PriceTable(
        effective_until=date(2026, 8, 24),
        prices={
            "gpt-5.6-sol": ModelPrice(
                input_per_million=Decimal("1.25"),
                output_per_million=Decimal("10"),
            )
        },
    )

    estimate = table.estimate(
        "gpt-5.6-sol", 1000, 100, as_of=date(2026, 8, 24)
    )

    assert estimate == Decimal("0.002250")


def test_price_table_rounds_any_fractional_microdollar_up():
    table = PriceTable(
        effective_until=date(2026, 8, 24),
        prices={
            "small": ModelPrice(
                input_per_million=Decimal("0.1"),
                output_per_million=Decimal("0"),
            )
        },
    )

    assert table.estimate("small", 1, 0, as_of=date(2026, 8, 24)) == Decimal(
        "0.000001"
    )
    assert table.estimate("small", 0, 0, as_of=date(2026, 8, 24)) == Decimal(
        "0.000000"
    )


def test_price_table_expiry_boundary_is_inclusive():
    table = PriceTable(
        effective_until=date(2026, 8, 24),
        prices={
            "known": ModelPrice(Decimal("1"), Decimal("1")),
        },
    )

    assert table.estimate("known", 1, 1, as_of=date(2026, 8, 24)) == Decimal(
        "0.000002"
    )


def test_price_table_rejects_unknown_model():
    table = PriceTable(effective_until=date(2026, 8, 24), prices={})

    with pytest.raises(UnknownModelPrice):
        table.estimate("unknown", 0, 0, as_of=date(2026, 8, 24))


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [(-1, 0), (0, -1), (True, 0), (0, False), (Decimal("1"), 0)],
)
def test_price_table_rejects_invalid_token_counts(input_tokens, output_tokens):
    table = PriceTable(
        effective_until=date(2026, 8, 24),
        prices={"known": ModelPrice(Decimal("1"), Decimal("1"))},
    )

    with pytest.raises(InvalidBudgetValue):
        table.estimate(
            "known", input_tokens, output_tokens, as_of=date(2026, 8, 24)
        )


@pytest.mark.parametrize(
    ("input_rate", "output_rate"),
    [
        (Decimal("-0.01"), Decimal("1")),
        (Decimal("1"), Decimal("-0.01")),
        (Decimal("NaN"), Decimal("1")),
        (Decimal("1"), Decimal("Infinity")),
        ("1", Decimal("1")),
    ],
)
def test_model_price_rejects_invalid_rates(input_rate, output_rate):
    with pytest.raises(InvalidBudgetValue):
        ModelPrice(input_rate, output_rate)


@pytest.mark.parametrize(
    ("amount", "created_at", "updated_at"),
    [
        ("1", utc(2026, 8, 1), utc(2026, 8, 1)),
        (Decimal("NaN"), utc(2026, 8, 1), utc(2026, 8, 1)),
        (Decimal("1"), datetime(2026, 8, 1), utc(2026, 8, 1)),
        (Decimal("1"), utc(2026, 8, 1), datetime(2026, 8, 1)),
    ],
)
def test_reservation_record_rejects_invalid_exact_values_and_times(
    amount, created_at, updated_at
):
    with pytest.raises(InvalidBudgetValue):
        Reservation(
            id="reservation-1",
            owner_key="run-1",
            amount=amount,
            state="reserved",
            created_at=created_at,
            updated_at=updated_at,
        )


def test_reservation_record_normalizes_aware_timestamps_to_utc():
    plus_five_thirty = timezone(timedelta(hours=5, minutes=30))

    reservation = Reservation(
        id="reservation-1",
        owner_key="run-1",
        amount=Decimal("1"),
        state="reserved",
        created_at=datetime(2026, 8, 1, 5, 30, tzinfo=plus_five_thirty),
        updated_at=datetime(2026, 8, 1, 6, 30, tzinfo=plus_five_thirty),
    )

    assert reservation.created_at == utc(2026, 8, 1)
    assert reservation.updated_at == utc(2026, 8, 1, 1)
    assert reservation.created_at.tzinfo is timezone.utc
    assert reservation.updated_at.tzinfo is timezone.utc


def test_reservation_record_rejects_updated_at_before_created_at():
    with pytest.raises(InvalidBudgetValue, match="updated_at"):
        Reservation(
            id="reservation-1",
            owner_key="run-1",
            amount=Decimal("1"),
            state="reserved",
            created_at=utc(2026, 8, 1, 2),
            updated_at=utc(2026, 8, 1, 1),
        )


def test_reservation_record_rejects_unrecognized_state():
    with pytest.raises(InvalidBudgetValue, match="state"):
        Reservation(
            id="reservation-1",
            owner_key="run-1",
            amount=Decimal("1"),
            state="unexpected",
            created_at=utc(2026, 8, 1),
            updated_at=utc(2026, 8, 1),
        )


def test_price_table_rejects_non_price_entries():
    with pytest.raises(InvalidBudgetValue):
        PriceTable(
            effective_until=date(2026, 8, 24),
            prices={"known": object()},
        )


@pytest.mark.parametrize(
    ("soft", "hard"),
    [
        (Decimal("-1"), Decimal("5")),
        (Decimal("4"), Decimal("4")),
        (Decimal("5"), Decimal("4")),
        (Decimal("4"), Decimal("5.01")),
        (Decimal("NaN"), Decimal("5")),
        (Decimal("4"), Decimal("Infinity")),
        ("4", Decimal("5")),
    ],
)
def test_ledger_rejects_invalid_limits(migrated_store, soft, hard):
    with pytest.raises(InvalidBudgetValue):
        BudgetLedger(migrated_store, soft, hard)


def test_reservation_at_hard_limit_is_allowed_for_critical_role(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    ledger.reserve(
        "run-1",
        Decimal("5"),
        now=utc(2026, 8, 1),
        role=AgentRole.RESEARCH_EDITOR,
    )

    assert ledger.month_total(2026, 8) == Decimal("5")


def test_reservation_that_reaches_soft_limit_is_allowed(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    ledger.reserve("run-1", Decimal("4"), now=utc(2026, 8, 1))

    assert ledger.month_total(2026, 8) == Decimal("4")


@pytest.mark.parametrize("role", [None, AgentRole.FUNDAMENTAL_ANALYST])
def test_zero_cost_reservation_at_exact_soft_limit_is_allowed(
    migrated_store, role
):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    ledger.reserve("run-1", Decimal("4"), now=utc(2026, 8, 1))

    reservation = ledger.reserve(
        "run-2", Decimal("0"), now=utc(2026, 8, 1), role=role
    )

    assert reservation.amount == Decimal("0")
    assert ledger.month_total(2026, 8) == Decimal("4")


def test_subsequent_noncritical_reservation_at_soft_limit_is_rejected(
    migrated_store,
):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    ledger.reserve("run-1", Decimal("4"), now=utc(2026, 8, 1))

    with pytest.raises(SoftBudgetExceeded):
        ledger.reserve(
            "run-2",
            Decimal("0.01"),
            now=utc(2026, 8, 1),
            role=AgentRole.FUNDAMENTAL_ANALYST,
        )

    assert ledger.month_total(2026, 8) == Decimal("4")


def test_noncritical_reservation_cannot_cross_soft_limit(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    ledger.reserve("run-1", Decimal("3.90"), now=utc(2026, 8, 1))

    with pytest.raises(SoftBudgetExceeded):
        ledger.reserve(
            "run-2",
            Decimal("0.11"),
            now=utc(2026, 8, 1),
            role=AgentRole.FUNDAMENTAL_ANALYST,
        )

    assert ledger.month_total(2026, 8) == Decimal("3.90")


def test_unroled_reservation_cannot_cross_soft_limit(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    with pytest.raises(SoftBudgetExceeded):
        ledger.reserve("run-1", Decimal("4.01"), now=utc(2026, 8, 1))

    assert ledger.month_total(2026, 8) == Decimal("0")


@pytest.mark.parametrize(
    "role", [AgentRole.SKEPTICAL_REVIEWER, AgentRole.RESEARCH_EDITOR]
)
def test_exact_critical_roles_may_cross_soft_limit(migrated_store, role):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    ledger.reserve(
        "run-1", Decimal("4.50"), now=utc(2026, 8, 1), role=role
    )

    assert ledger.month_total(2026, 8) == Decimal("4.50")


def test_other_agent_role_cannot_cross_soft_limit(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    with pytest.raises(SoftBudgetExceeded):
        ledger.reserve(
            "run-1",
            Decimal("4.50"),
            now=utc(2026, 8, 1),
            role=AgentRole.FUNDAMENTAL_ANALYST,
        )


def test_reserve_rejects_invalid_role_type(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    with pytest.raises(InvalidBudgetValue):
        ledger.reserve(
            "run-1", Decimal("1"), now=utc(2026, 8, 1), role="research_editor"
        )


@pytest.mark.parametrize(
    "amount", [Decimal("-0.01"), Decimal("NaN"), Decimal("Infinity"), "1.00"]
)
def test_reserve_rejects_invalid_amount_without_creating_row(
    migrated_store, amount
):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    with pytest.raises(InvalidBudgetValue):
        ledger.reserve("run-1", amount, now=utc(2026, 8, 1))

    assert ledger.month_total(2026, 8) == Decimal("0")


def test_reserve_rejects_empty_owner_without_sql_error(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    with pytest.raises(BudgetError):
        ledger.reserve("  ", Decimal("1"), now=utc(2026, 8, 1))


def test_reserve_rejects_naive_time(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    with pytest.raises(InvalidBudgetValue, match="timezone-aware"):
        ledger.reserve("run-1", Decimal("1"), now=datetime(2026, 8, 1))


def test_accounting_month_is_reservation_created_month_in_utc(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    plus_two = timezone(timedelta(hours=2))
    local_september = datetime(2026, 9, 1, 0, 30, tzinfo=plus_two)
    ledger.reserve("august", Decimal("1.25"), now=local_september)
    ledger.reserve("september", Decimal("2.50"), now=utc(2026, 9, 1))

    assert ledger.month_total(2026, 8) == Decimal("1.25")
    assert ledger.month_total(2026, 9) == Decimal("2.50")


def test_reconciliation_remains_in_reservation_month(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = ledger.reserve(
        "run-1", Decimal("1"), now=utc(2026, 8, 31)
    )

    ledger.reconcile(
        reservation.id, Decimal("0.40"), now=utc(2026, 9, 1)
    )

    assert ledger.month_total(2026, 8) == Decimal("0.40")
    assert ledger.month_total(2026, 9) == Decimal("0")


def test_reservation_requires_no_task_or_run_row(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    reservation = ledger.reserve(
        "fresh-owner", Decimal("1"), now=utc(2026, 8, 1)
    )

    with migrated_store.connect() as connection:
        row = connection.execute(
            "SELECT owner_key, task_id, run_id FROM budget_reservations "
            "WHERE reservation_id = ?",
            (reservation.id,),
        ).fetchone()
    assert row == ("fresh-owner", None, None)


def test_same_owner_can_hold_multiple_collision_resistant_reservations(
    migrated_store,
):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    first = ledger.reserve("run-1", Decimal("1"), now=utc(2026, 8, 1))
    second = ledger.reserve("run-1", Decimal("1"), now=utc(2026, 8, 1))

    assert first.id != second.id
    assert ledger.month_total(2026, 8) == Decimal("2")


def test_release_counts_reserved_amount_as_zero_and_is_idempotent(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = ledger.reserve(
        "run-1", Decimal("1"), now=utc(2026, 8, 1)
    )

    released = ledger.release(reservation.id, now=utc(2026, 8, 2))
    released_again = ledger.release(reservation.id, now=utc(2026, 8, 3))

    assert released.state == "released"
    assert released_again == released
    assert ledger.month_total(2026, 8) == Decimal("0")


def test_usage_unknown_keeps_estimate_counted_and_is_idempotent(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = ledger.reserve(
        "run-1", Decimal("1.25"), now=utc(2026, 8, 1)
    )

    unknown = ledger.mark_usage_unknown(
        reservation.id, now=utc(2026, 8, 2)
    )
    unknown_again = ledger.mark_usage_unknown(
        reservation.id, now=utc(2026, 8, 3)
    )

    assert unknown.state == "usage_unknown"
    assert unknown_again == unknown
    assert ledger.month_total(2026, 8) == Decimal("1.25")


def test_usage_unknown_cannot_be_released(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = ledger.reserve(
        "run-1", Decimal("1"), now=utc(2026, 8, 1)
    )
    ledger.mark_usage_unknown(reservation.id, now=utc(2026, 8, 2))

    with pytest.raises(InvalidReservationState):
        ledger.release(reservation.id, now=utc(2026, 8, 3))


def test_reconciled_reservation_cannot_be_released(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = ledger.reserve(
        "run-1", Decimal("1"), now=utc(2026, 8, 1)
    )
    ledger.reconcile(reservation.id, Decimal("0.50"), now=utc(2026, 8, 2))

    with pytest.raises(InvalidReservationState):
        ledger.release(reservation.id, now=utc(2026, 8, 3))


def test_same_value_reconciliation_is_idempotent(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = ledger.reserve(
        "run-1", Decimal("1"), now=utc(2026, 8, 1)
    )

    first = ledger.reconcile(
        reservation.id, Decimal("0.42"), now=utc(2026, 8, 2)
    )
    second = ledger.reconcile(
        reservation.id, Decimal("0.420"), now=utc(2026, 8, 3)
    )

    assert second == first
    assert ledger.month_total(2026, 8) == Decimal("0.42")


def test_conflicting_second_reconciliation_fails(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = ledger.reserve(
        "run-1", Decimal("1"), now=utc(2026, 8, 1)
    )
    ledger.reconcile(reservation.id, Decimal("0.42"), now=utc(2026, 8, 2))

    with pytest.raises(InvalidReservationState):
        ledger.reconcile(
            reservation.id, Decimal("0.43"), now=utc(2026, 8, 3)
        )


@pytest.mark.parametrize(
    "amount", [Decimal("-0.01"), Decimal("NaN"), Decimal("Infinity"), "0.2"]
)
def test_reconcile_rejects_invalid_actual_cost_and_preserves_reservation(
    migrated_store, amount
):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = ledger.reserve(
        "run-1", Decimal("1"), now=utc(2026, 8, 1)
    )

    with pytest.raises(InvalidBudgetValue):
        ledger.reconcile(reservation.id, amount, now=utc(2026, 8, 2))

    assert ledger.get_reservation(reservation.id) == reservation


def test_provider_overage_is_recorded_before_hard_limit_error(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = ledger.reserve(
        "run-1", Decimal("1"), now=utc(2026, 8, 1)
    )

    with pytest.raises(BudgetExceeded, match="recorded provider overage"):
        ledger.reconcile(
            reservation.id, Decimal("5.10"), now=utc(2026, 8, 2)
        )

    stored = ledger.get_reservation(reservation.id)
    assert stored.state == "reconciled"
    assert stored.amount == Decimal("5.10")
    assert ledger.month_total(2026, 8) == Decimal("5.10")


@pytest.mark.parametrize("method_name", ["reconcile", "release", "mark_usage_unknown"])
def test_transitions_reject_missing_reservation(migrated_store, method_name):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    with pytest.raises(ReservationNotFound):
        if method_name == "reconcile":
            ledger.reconcile("missing", Decimal("0.1"), now=utc(2026, 8, 1))
        else:
            getattr(ledger, method_name)("missing", now=utc(2026, 8, 1))


def test_transition_methods_reject_naive_times(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = ledger.reserve(
        "run-1", Decimal("1"), now=utc(2026, 8, 1)
    )
    naive = datetime(2026, 8, 2)

    with pytest.raises(InvalidBudgetValue, match="timezone-aware"):
        ledger.reconcile(reservation.id, Decimal("0.2"), now=naive)
    with pytest.raises(InvalidBudgetValue, match="timezone-aware"):
        ledger.release(reservation.id, now=naive)
    with pytest.raises(InvalidBudgetValue, match="timezone-aware"):
        ledger.mark_usage_unknown(reservation.id, now=naive)


def test_reservation_persists_across_ledger_instances(migrated_store):
    first_ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = first_ledger.reserve(
        "run-1", Decimal("1.20"), now=utc(2026, 8, 1)
    )

    second_ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    assert second_ledger.get_reservation(reservation.id) == reservation
    assert second_ledger.month_total(2026, 8) == Decimal("1.20")


def test_concurrent_reservations_cannot_overspend(migrated_store):
    first = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    second = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    first.reserve("seed", Decimal("4"), now=utc(2026, 8, 1))
    barrier = Barrier(2)

    def attempt(ledger, owner):
        barrier.wait()
        try:
            ledger.reserve(
                owner,
                Decimal("1"),
                now=utc(2026, 8, 2),
                role=AgentRole.RESEARCH_EDITOR,
            )
        except BudgetExceeded:
            return "rejected"
        return "reserved"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(attempt, (first, second), ("run-1", "run-2"))
        )

    assert outcomes.count("reserved") == 1
    assert outcomes.count("rejected") == 1
    assert first.month_total(2026, 8) == Decimal("5")


def insert_persisted_reservation(
    store,
    *,
    reservation_id="persisted-1",
    state="reserved",
    amount="1.00",
    created_at="2026-08-01T00:00:00.000000Z",
):
    connection = store.connect()
    try:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "INSERT INTO budget_reservations ("
            "reservation_id, owner_key, amount_usd, state, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (
                reservation_id,
                "persisted-owner",
                amount,
                state,
                created_at,
                "2026-08-01T00:00:00.000000Z",
            ),
        )
    finally:
        connection.close()


@pytest.mark.parametrize("state", ["consumed", "expired"])
def test_all_nonreleased_recognized_states_count_conservatively(
    migrated_store, state
):
    insert_persisted_reservation(migrated_store, state=state, amount="1.2500")
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    assert ledger.month_total(2026, 8) == Decimal("1.2500")


@pytest.mark.parametrize(
    ("state", "amount", "created_at"),
    [
        ("reserved", "-0.01", "2026-08-01T00:00:00.000000Z"),
        ("reserved", "NaN", "2026-08-01T00:00:00.000000Z"),
        ("reserved", "Infinity", "2026-08-01T00:00:00.000000Z"),
        ("reserved", "not-a-decimal", "2026-08-01T00:00:00.000000Z"),
        ("reserved", "1.00", "2026-08-01T00:00:00.000000"),
        ("reserved", "1.00", "not-a-timestamp"),
        ("reserved", "1.00", "2026-08-01T01:00:00.000000+01:00"),
        ("unknown_state", "1.00", "2026-08-01T00:00:00.000000Z"),
    ],
)
@pytest.mark.parametrize("operation", ["month_total", "reserve"])
def test_corrupt_persisted_ledger_fails_closed_without_new_reservation(
    migrated_store, state, amount, created_at, operation
):
    insert_persisted_reservation(
        migrated_store, state=state, amount=amount, created_at=created_at
    )
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    with pytest.raises(BudgetIntegrityError):
        if operation == "month_total":
            ledger.month_total(2026, 8)
        else:
            ledger.reserve("new-owner", Decimal("0.10"), now=utc(2026, 8, 2))

    with migrated_store.connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM budget_reservations"
        ).fetchone()[0]
    assert count == 1


def test_usage_unknown_can_reconcile_to_lower_provider_actual(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = ledger.reserve(
        "run-1", Decimal("1.25"), now=utc(2026, 8, 1)
    )
    ledger.mark_usage_unknown(reservation.id, now=utc(2026, 8, 2))

    reconciled = ledger.reconcile(
        reservation.id, Decimal("0.40"), now=utc(2026, 8, 3)
    )

    assert reconciled.state == "reconciled"
    assert reconciled.amount == Decimal("0.40")
    assert ledger.month_total(2026, 8) == Decimal("0.40")


def test_usage_unknown_provider_overage_is_recorded_before_error(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = ledger.reserve(
        "run-1", Decimal("1"), now=utc(2026, 8, 1)
    )
    ledger.mark_usage_unknown(reservation.id, now=utc(2026, 8, 2))

    with pytest.raises(BudgetExceeded, match="recorded provider overage"):
        ledger.reconcile(
            reservation.id, Decimal("5.10"), now=utc(2026, 8, 3)
        )

    assert ledger.get_reservation(reservation.id).state == "reconciled"
    assert ledger.month_total(2026, 8) == Decimal("5.10")


def test_usage_unknown_reconciliation_retries_are_consistent(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    reservation = ledger.reserve(
        "run-1", Decimal("1"), now=utc(2026, 8, 1)
    )
    ledger.mark_usage_unknown(reservation.id, now=utc(2026, 8, 2))

    first = ledger.reconcile(
        reservation.id, Decimal("0.42"), now=utc(2026, 8, 3)
    )
    same = ledger.reconcile(
        reservation.id, Decimal("0.420"), now=utc(2026, 8, 4)
    )

    assert same == first
    with pytest.raises(InvalidReservationState):
        ledger.reconcile(
            reservation.id, Decimal("0.43"), now=utc(2026, 8, 4)
        )


def test_price_table_policy_and_nested_mapping_are_immutable():
    source_prices = {"known": ModelPrice(Decimal("1"), Decimal("1"))}
    table = PriceTable(date(2026, 8, 24), source_prices)

    with pytest.raises(FrozenInstanceError):
        table.effective_until = date(2099, 1, 1)
    with pytest.raises(FrozenInstanceError):
        table.prices = {}
    with pytest.raises(TypeError):
        table.prices["bypass"] = ModelPrice(Decimal("0"), Decimal("0"))
    source_prices["bypass"] = ModelPrice(Decimal("0"), Decimal("0"))
    with pytest.raises(UnknownModelPrice):
        table.estimate("bypass", 0, 0, as_of=date(2026, 8, 24))


def test_budget_ledger_policy_fields_are_immutable(migrated_store):
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))

    for name, value in (
        ("store", object()),
        ("soft_limit", Decimal("0")),
        ("hard_limit", Decimal("100")),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(ledger, name, value)


def test_reservation_id_collision_retries_with_injected_factory(migrated_store):
    ids = iter(("same-id", "same-id", "fresh-id"))
    ledger = BudgetLedger(
        migrated_store,
        Decimal("4"),
        Decimal("5"),
        id_factory=lambda: next(ids),
    )
    first = ledger.reserve("run-1", Decimal("1"), now=utc(2026, 8, 1))

    second = ledger.reserve("run-2", Decimal("1"), now=utc(2026, 8, 1))

    assert first.id == "same-id"
    assert second.id == "fresh-id"


def test_exhausted_reservation_id_collisions_raise_typed_error(migrated_store):
    ledger = BudgetLedger(
        migrated_store,
        Decimal("4"),
        Decimal("5"),
        id_factory=lambda: "same-id",
    )
    ledger.reserve("run-1", Decimal("1"), now=utc(2026, 8, 1))

    with pytest.raises(ReservationIdExhausted) as error:
        ledger.reserve("run-2", Decimal("1"), now=utc(2026, 8, 1))

    assert "sqlite" not in str(error.value).lower()
    assert "unique" not in str(error.value).lower()
    assert ledger.month_total(2026, 8) == Decimal("1")


def test_price_table_default_date_uses_injected_instant_in_utc():
    pacific = timezone(timedelta(hours=-7))
    local_august_24_but_utc_august_25 = datetime(
        2026, 8, 24, 17, 30, tzinfo=pacific
    )
    table = PriceTable(
        effective_until=date(2026, 8, 24),
        prices={},
        clock=lambda: local_august_24_but_utc_august_25,
    )

    with pytest.raises(ExpiredPricing):
        table.estimate("unknown", 0, 0)
