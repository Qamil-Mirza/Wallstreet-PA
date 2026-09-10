"""Deterministic builders for exhibits derived from normalized rows."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .models import DisplayExhibit, DisplayModel, _identifier, _text


DEFAULT_TOLERANCE = Decimal("0.000001")
Currency = Literal["USD", "EUR", "GBP", "JPY", "MYR"]
ValuationUnit = Literal["currency", "per_share", "multiple", "percent"]


def _plain_text(value: str) -> str:
    value = _text(value)
    if "<" in value or ">" in value:
        raise ValueError("model-authored HTML is not allowed in exhibit rows")
    return value


def _finite(value: Decimal) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError("must be a finite Decimal")
    return value


def _tolerance(value: Decimal) -> Decimal:
    value = _finite(value)
    if value < 0:
        raise ValueError("tolerance must be non-negative")
    return value


def _source_note(note: str, source_date: date) -> str:
    return f"{_plain_text(note)} ({source_date.isoformat()})"


class ExposureRow(DisplayModel):
    symbol: str
    label: str
    weight: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    currency: Currency
    source_note: str
    source_date: date

    _symbol = field_validator("symbol")(_identifier)
    _label = field_validator("label")(_plain_text)
    _weight = field_validator("weight")(_finite)
    _source = field_validator("source_note")(_plain_text)

    @property
    def display_weight(self) -> str:
        return f"{self.weight * Decimal('100'):.1f}%"


class ExposureExhibit(DisplayExhibit):
    rows: tuple[ExposureRow, ...]
    total_weight: Decimal


class ValuationRow(DisplayModel):
    label: str
    low: Decimal
    base: Decimal
    high: Decimal
    currency: Currency
    unit: ValuationUnit
    source_note: str
    source_date: date

    _label = field_validator("label")(_plain_text)
    _low = field_validator("low")(_finite)
    _base = field_validator("base")(_finite)
    _high = field_validator("high")(_finite)
    _source = field_validator("source_note")(_plain_text)

    @model_validator(mode="after")
    def _ordered(self) -> "ValuationRow":
        if not self.low <= self.base <= self.high:
            raise ValueError("valuation range must satisfy low <= base <= high")
        return self

    @property
    def display_range(self) -> str:
        return f"{self.low:g}–{self.high:g} {self.unit}"


class ValuationExhibit(DisplayExhibit):
    rows: tuple[ValuationRow, ...]


class ScenarioRow(DisplayModel):
    scenario: str
    probability: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    value: Decimal
    currency: Currency
    unit: ValuationUnit
    source_note: str
    source_date: date

    _scenario = field_validator("scenario")(_plain_text)
    _probability = field_validator("probability")(_finite)
    _value = field_validator("value")(_finite)
    _source = field_validator("source_note")(_plain_text)

    @property
    def display_probability(self) -> str:
        return f"{self.probability * Decimal('100'):.1f}%"


class ScenarioMatrix(DisplayExhibit):
    rows: tuple[ScenarioRow, ...]
    probability_total: Decimal


def _require_nonempty(rows: tuple[object, ...]) -> None:
    if not rows:
        raise ValueError("exhibit rows must not be empty")


def _require_one_currency(rows: tuple[object, ...]) -> None:
    currencies = {getattr(row, "currency") for row in rows}
    if len(currencies) != 1:
        raise ValueError("exhibit rows must use one currency")


def _notes(rows: tuple[object, ...]) -> tuple[str, ...]:
    return tuple(sorted({_source_note(row.source_note, row.source_date) for row in rows}))


def build_exposure_exhibit(
    rows: tuple[ExposureRow, ...],
    *,
    declared_total: Decimal = Decimal("1"),
    tolerance: Decimal = DEFAULT_TOLERANCE,
) -> ExposureExhibit:
    _require_nonempty(rows)
    declared_total, tolerance = _finite(declared_total), _tolerance(tolerance)
    _require_one_currency(rows)
    total = sum((row.weight for row in rows), Decimal("0"))
    if abs(total - declared_total) > tolerance:
        raise ValueError("exposure weights do not reconcile to declared total")
    ordered = tuple(sorted(rows, key=lambda row: (row.symbol, row.label)))
    return ExposureExhibit(
        title="Rounded exposure summary", rows=ordered, total_weight=total,
        source_notes=_notes(rows),
    )


def build_valuation_exhibit(
    rows: tuple[ValuationRow, ...],
) -> ValuationExhibit:
    _require_nonempty(rows)
    _require_one_currency(rows)
    units = {row.unit for row in rows}
    if len(units) != 1:
        raise ValueError("valuation rows must use one unit")
    return ValuationExhibit(
        title="Valuation ranges",
        rows=tuple(sorted(rows, key=lambda row: row.label)),
        source_notes=_notes(rows),
    )


def build_scenario_matrix(
    rows: tuple[ScenarioRow, ...],
    *,
    declared_probability: Decimal = Decimal("1"),
    tolerance: Decimal = DEFAULT_TOLERANCE,
) -> ScenarioMatrix:
    _require_nonempty(rows)
    declared_probability, tolerance = _finite(declared_probability), _tolerance(tolerance)
    _require_one_currency(rows)
    units = {row.unit for row in rows}
    if len(units) != 1:
        raise ValueError("scenario rows must use one unit")
    total = sum((row.probability for row in rows), Decimal("0"))
    if abs(total - declared_probability) > tolerance:
        raise ValueError("scenario probabilities do not reconcile to declared total")
    return ScenarioMatrix(
        title="Scenario matrix",
        rows=tuple(sorted(rows, key=lambda row: row.scenario)),
        probability_total=total,
        source_notes=_notes(rows),
    )
