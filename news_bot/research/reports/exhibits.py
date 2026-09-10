"""Deterministic builders for exhibits derived from authoritative normalized rows."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Literal, Mapping

from pydantic import Field, field_validator, model_validator

from ..quality import CalculatedExhibit
from .models import DisplayExhibit, DisplayModel, EventUpdate, _identifier, _text


Currency = Literal["USD", "EUR", "GBP", "JPY", "MYR"]
ValuationUnit = Literal["currency", "per_share", "multiple", "percent"]
_ONE = Decimal("1")
_ROW_ID_CHARACTER = re.compile(r"[^A-Za-z0-9]+")


def _plain_text(value: str) -> str:
    value = _text(value)
    if "<" in value or ">" in value:
        raise ValueError("model-authored HTML is not allowed in exhibit rows")
    return value


def _finite(value: Decimal) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError("must be a finite Decimal")
    return value


def _source_note(note: str, source_date: date) -> str:
    return f"{_plain_text(note)} ({source_date.isoformat()})"


def _row_id(value: str) -> str:
    identifier = _ROW_ID_CHARACTER.sub("-", value).strip("-").casefold()
    if not identifier:
        raise ValueError("exhibit row requires an authoritative row ID")
    return identifier


def _require_nonempty(rows: tuple[object, ...]) -> None:
    if not rows:
        raise ValueError("exhibit rows must not be empty")


def _require_one_currency(rows: tuple[object, ...]) -> None:
    currencies = {getattr(row, "currency") for row in rows}
    if len(currencies) != 1:
        raise ValueError("exhibit rows must use one currency")


def _notes(rows: tuple[object, ...]) -> tuple[str, ...]:
    return tuple(sorted({_source_note(row.source_note, row.source_date) for row in rows}))


def _require_notes(source_notes: tuple[str, ...], rows: tuple[object, ...]) -> None:
    if source_notes != _notes(rows):
        raise ValueError("exhibit source notes must be derived from its rows")


def _require_authority(
    calculation: CalculatedExhibit,
    *,
    exhibit_id: str,
    expected: Mapping[str, Mapping[str, Decimal]],
    row_count: int,
) -> None:
    if calculation.exhibit_id.casefold() != exhibit_id:
        raise ValueError("calculated exhibit uses the wrong authoritative exhibit ID")
    if not calculation.reconciles:
        raise ValueError("calculated exhibit does not reconcile")
    authoritative = {
        row.row_id.casefold(): dict(row.values) for row in calculation.normalized_rows
    }
    canonical_expected = {
        row_id.casefold(): dict(values) for row_id, values in expected.items()
    }
    if len(canonical_expected) != row_count:
        raise ValueError("display exhibit row IDs must be unique")
    if authoritative != canonical_expected:
        raise ValueError(
            "display rows do not exactly match authoritative calculated rows"
        )


class ExposureRow(DisplayModel):
    symbol: str
    label: str
    weight: Decimal = Field(ge=Decimal("0"), le=_ONE)
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
    rows: tuple[ExposureRow, ...] = Field(min_length=1)
    total_weight: Decimal

    _total = field_validator("total_weight")(_finite)

    @model_validator(mode="after")
    def _authoritative_rows(self) -> "ExposureExhibit":
        _require_one_currency(self.rows)
        if self.total_weight != _ONE or sum(
            (row.weight for row in self.rows), Decimal("0")
        ) != _ONE:
            raise ValueError("exposure weights do not reconcile to fixed total 1")
        _require_notes(self.source_notes, self.rows)
        _require_authority(
            self.calculation,
            exhibit_id="exposure",
            expected={row.symbol: {"weight": row.weight} for row in self.rows},
            row_count=len(self.rows),
        )
        return self


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
    rows: tuple[ValuationRow, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _authoritative_rows(self) -> "ValuationExhibit":
        _require_one_currency(self.rows)
        if len({row.unit for row in self.rows}) != 1:
            raise ValueError("valuation rows must use one unit")
        _require_notes(self.source_notes, self.rows)
        _require_authority(
            self.calculation,
            exhibit_id="valuation",
            expected={
                _row_id(row.label): {
                    "low": row.low,
                    "base": row.base,
                    "high": row.high,
                }
                for row in self.rows
            },
            row_count=len(self.rows),
        )
        return self


class ScenarioRow(DisplayModel):
    scenario: str
    probability: Decimal = Field(ge=Decimal("0"), le=_ONE)
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
    rows: tuple[ScenarioRow, ...] = Field(min_length=1)
    probability_total: Decimal

    _total = field_validator("probability_total")(_finite)

    @model_validator(mode="after")
    def _authoritative_rows(self) -> "ScenarioMatrix":
        _require_one_currency(self.rows)
        if len({row.unit for row in self.rows}) != 1:
            raise ValueError("scenario rows must use one unit")
        if self.probability_total != _ONE or sum(
            (row.probability for row in self.rows), Decimal("0")
        ) != _ONE:
            raise ValueError("scenario probabilities do not reconcile to fixed total 1")
        _require_notes(self.source_notes, self.rows)
        _require_authority(
            self.calculation,
            exhibit_id="scenario",
            expected={
                _row_id(row.scenario): {
                    "probability": row.probability,
                    "value": row.value,
                }
                for row in self.rows
            },
            row_count=len(self.rows),
        )
        return self


def build_exposure_exhibit(
    rows: tuple[ExposureRow, ...],
    *,
    calculation: CalculatedExhibit,
    evidence_ids: tuple[str, ...],
) -> ExposureExhibit:
    _require_nonempty(rows)
    ordered = tuple(sorted(rows, key=lambda row: (row.symbol, row.label)))
    return ExposureExhibit(
        title="Rounded exposure summary",
        rows=ordered,
        total_weight=sum((row.weight for row in ordered), Decimal("0")),
        source_notes=_notes(ordered),
        evidence_ids=evidence_ids,
        calculation=calculation,
        verified_reconciliation=True,
    )


def build_valuation_exhibit(
    rows: tuple[ValuationRow, ...],
    *,
    calculation: CalculatedExhibit,
    evidence_ids: tuple[str, ...],
) -> ValuationExhibit:
    _require_nonempty(rows)
    ordered = tuple(sorted(rows, key=lambda row: row.label))
    return ValuationExhibit(
        title="Valuation ranges",
        rows=ordered,
        source_notes=_notes(ordered),
        evidence_ids=evidence_ids,
        calculation=calculation,
        verified_reconciliation=True,
    )


def build_scenario_matrix(
    rows: tuple[ScenarioRow, ...],
    *,
    calculation: CalculatedExhibit,
    evidence_ids: tuple[str, ...],
) -> ScenarioMatrix:
    _require_nonempty(rows)
    ordered = tuple(sorted(rows, key=lambda row: row.scenario))
    return ScenarioMatrix(
        title="Scenario matrix",
        rows=ordered,
        probability_total=sum((row.probability for row in ordered), Decimal("0")),
        source_notes=_notes(ordered),
        evidence_ids=evidence_ids,
        calculation=calculation,
        verified_reconciliation=True,
    )


# Resolve EventUpdate's concrete exhibit union after the circular-safe model module
# has finished loading. This retains subtype rows during render-boundary validation.
EventUpdate.model_rebuild(
    _types_namespace={
        "ExposureExhibit": ExposureExhibit,
        "ValuationExhibit": ValuationExhibit,
        "ScenarioMatrix": ScenarioMatrix,
    }
)
