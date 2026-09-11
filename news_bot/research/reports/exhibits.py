"""Deterministic builders for exhibits derived from authoritative normalized rows."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal, Mapping

from pydantic import Field, field_validator, model_validator

from ..quality import CalculatedExhibit
from .models import (
    DisplayExhibit,
    DisplayModel,
    EventUpdate,
    _identifier,
    _identity_label,
    _text,
)


Currency = Literal["USD", "EUR", "GBP", "JPY", "MYR"]
ValuationUnit = Literal["currency", "per_share", "multiple", "percent"]
_ONE = Decimal("1")


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


def _require_stored_provenance(
    authority_rows: tuple[DisplayModel, ...], rendered_rows: tuple[DisplayModel, ...]
) -> None:
    def keyed(rows: tuple[DisplayModel, ...]) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for row in rows:
            row_id = str(getattr(row, "row_id")).casefold()
            if row_id in result:
                raise ValueError("stored and rendered exhibit row IDs must be unique")
            result[row_id] = row.model_dump(mode="python")
        return result

    if keyed(authority_rows) != keyed(rendered_rows):
        raise ValueError(
            "rendered exhibit fields do not exactly match authoritative stored provenance"
        )


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


class StoredExposureRow(DisplayModel):
    """Authoritative normalized exposure row loaded before presentation."""

    row_id: str
    symbol: str
    label: str
    weight: Decimal = Field(ge=Decimal("0"), le=_ONE)
    currency: Currency
    source_note: str
    source_date: date

    _row_id = field_validator("row_id")(_identifier)
    _symbol = field_validator("symbol")(_identifier)
    _label = field_validator("label")(_identity_label)
    _weight = field_validator("weight")(_finite)
    _source = field_validator("source_note")(_plain_text)


class ExposureRow(DisplayModel):
    row_id: str
    symbol: str
    label: str
    weight: Decimal = Field(ge=Decimal("0"), le=_ONE)
    currency: Currency
    source_note: str
    source_date: date

    _row_id = field_validator("row_id")(_identifier)
    _symbol = field_validator("symbol")(_identifier)
    _label = field_validator("label")(_identity_label)
    _weight = field_validator("weight")(_finite)
    _source = field_validator("source_note")(_plain_text)

    @property
    def display_weight(self) -> str:
        return f"{self.weight * Decimal('100'):.1f}%"


class ExposureExhibit(DisplayExhibit):
    authority_rows: tuple[StoredExposureRow, ...] = Field(min_length=1)
    rows: tuple[ExposureRow, ...] = Field(min_length=1)
    total_weight: Decimal

    _total = field_validator("total_weight")(_finite)

    @model_validator(mode="after")
    def _authoritative_rows(self) -> "ExposureExhibit":
        _require_stored_provenance(self.authority_rows, self.rows)
        _require_one_currency(self.rows)
        if self.total_weight != _ONE or sum(
            (row.weight for row in self.rows), Decimal("0")
        ) != _ONE:
            raise ValueError("exposure weights do not reconcile to fixed total 1")
        _require_notes(self.source_notes, self.authority_rows)
        _require_authority(
            self.calculation,
            exhibit_id="exposure",
            expected={row.row_id: {"weight": row.weight} for row in self.authority_rows},
            row_count=len(self.authority_rows),
        )
        return self


class StoredValuationRow(DisplayModel):
    """Authoritative normalized valuation row loaded before presentation."""

    row_id: str
    label: str
    low: Decimal
    base: Decimal
    high: Decimal
    currency: Currency
    unit: ValuationUnit
    source_note: str
    source_date: date

    _row_id = field_validator("row_id")(_identifier)
    _label = field_validator("label")(_plain_text)
    _low = field_validator("low")(_finite)
    _base = field_validator("base")(_finite)
    _high = field_validator("high")(_finite)
    _source = field_validator("source_note")(_plain_text)

    @model_validator(mode="after")
    def _ordered(self) -> "StoredValuationRow":
        if not self.low <= self.base <= self.high:
            raise ValueError("valuation range must satisfy low <= base <= high")
        return self


class ValuationRow(DisplayModel):
    row_id: str
    label: str
    low: Decimal
    base: Decimal
    high: Decimal
    currency: Currency
    unit: ValuationUnit
    source_note: str
    source_date: date

    _row_id = field_validator("row_id")(_identifier)
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
    authority_rows: tuple[StoredValuationRow, ...] = Field(min_length=1)
    rows: tuple[ValuationRow, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _authoritative_rows(self) -> "ValuationExhibit":
        _require_stored_provenance(self.authority_rows, self.rows)
        _require_one_currency(self.rows)
        if len({row.unit for row in self.rows}) != 1:
            raise ValueError("valuation rows must use one unit")
        _require_notes(self.source_notes, self.authority_rows)
        _require_authority(
            self.calculation,
            exhibit_id="valuation",
            expected={
                row.row_id: {
                    "low": row.low,
                    "base": row.base,
                    "high": row.high,
                }
                for row in self.authority_rows
            },
            row_count=len(self.authority_rows),
        )
        return self


class StoredScenarioRow(DisplayModel):
    """Authoritative normalized scenario row loaded before presentation."""

    row_id: str
    scenario: str
    probability: Decimal = Field(ge=Decimal("0"), le=_ONE)
    value: Decimal
    currency: Currency
    unit: ValuationUnit
    source_note: str
    source_date: date

    _row_id = field_validator("row_id")(_identifier)
    _scenario = field_validator("scenario")(_plain_text)
    _probability = field_validator("probability")(_finite)
    _value = field_validator("value")(_finite)
    _source = field_validator("source_note")(_plain_text)


class ScenarioRow(DisplayModel):
    row_id: str
    scenario: str
    probability: Decimal = Field(ge=Decimal("0"), le=_ONE)
    value: Decimal
    currency: Currency
    unit: ValuationUnit
    source_note: str
    source_date: date

    _row_id = field_validator("row_id")(_identifier)
    _scenario = field_validator("scenario")(_plain_text)
    _probability = field_validator("probability")(_finite)
    _value = field_validator("value")(_finite)
    _source = field_validator("source_note")(_plain_text)

    @property
    def display_probability(self) -> str:
        return f"{self.probability * Decimal('100'):.1f}%"


class ScenarioMatrix(DisplayExhibit):
    authority_rows: tuple[StoredScenarioRow, ...] = Field(min_length=1)
    rows: tuple[ScenarioRow, ...] = Field(min_length=1)
    probability_total: Decimal

    _total = field_validator("probability_total")(_finite)

    @model_validator(mode="after")
    def _authoritative_rows(self) -> "ScenarioMatrix":
        _require_stored_provenance(self.authority_rows, self.rows)
        _require_one_currency(self.rows)
        if len({row.unit for row in self.rows}) != 1:
            raise ValueError("scenario rows must use one unit")
        if self.probability_total != _ONE or sum(
            (row.probability for row in self.rows), Decimal("0")
        ) != _ONE:
            raise ValueError("scenario probabilities do not reconcile to fixed total 1")
        _require_notes(self.source_notes, self.authority_rows)
        _require_authority(
            self.calculation,
            exhibit_id="scenario",
            expected={
                row.row_id: {
                    "probability": row.probability,
                    "value": row.value,
                }
                for row in self.authority_rows
            },
            row_count=len(self.authority_rows),
        )
        return self


def build_exposure_exhibit(
    authority_rows: tuple[StoredExposureRow, ...],
    *,
    calculation: CalculatedExhibit,
    evidence_ids: tuple[str, ...],
) -> ExposureExhibit:
    _require_nonempty(authority_rows)
    ordered_authority = tuple(
        sorted(authority_rows, key=lambda row: (row.symbol, row.label, row.row_id))
    )
    rows = tuple(
        ExposureRow.model_validate(row.model_dump(mode="python"))
        for row in ordered_authority
    )
    return ExposureExhibit(
        title="Rounded exposure summary",
        rows=rows,
        total_weight=sum((row.weight for row in rows), Decimal("0")),
        authority_rows=ordered_authority,
        source_notes=_notes(ordered_authority),
        evidence_ids=evidence_ids,
        calculation=calculation,
        verified_reconciliation=True,
    )


def build_valuation_exhibit(
    authority_rows: tuple[StoredValuationRow, ...],
    *,
    calculation: CalculatedExhibit,
    evidence_ids: tuple[str, ...],
) -> ValuationExhibit:
    _require_nonempty(authority_rows)
    ordered_authority = tuple(
        sorted(authority_rows, key=lambda row: (row.label, row.row_id))
    )
    rows = tuple(
        ValuationRow.model_validate(row.model_dump(mode="python"))
        for row in ordered_authority
    )
    return ValuationExhibit(
        title="Valuation ranges",
        rows=rows,
        authority_rows=ordered_authority,
        source_notes=_notes(ordered_authority),
        evidence_ids=evidence_ids,
        calculation=calculation,
        verified_reconciliation=True,
    )


def build_scenario_matrix(
    authority_rows: tuple[StoredScenarioRow, ...],
    *,
    calculation: CalculatedExhibit,
    evidence_ids: tuple[str, ...],
) -> ScenarioMatrix:
    _require_nonempty(authority_rows)
    ordered_authority = tuple(
        sorted(authority_rows, key=lambda row: (row.scenario, row.row_id))
    )
    rows = tuple(
        ScenarioRow.model_validate(row.model_dump(mode="python"))
        for row in ordered_authority
    )
    return ScenarioMatrix(
        title="Scenario matrix",
        rows=rows,
        probability_total=sum((row.probability for row in rows), Decimal("0")),
        authority_rows=ordered_authority,
        source_notes=_notes(ordered_authority),
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
