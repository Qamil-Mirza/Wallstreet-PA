"""Deterministic entity resolution and evidence-backed exposure mapping."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .store import ResearchStore


_ENTITY_ID = re.compile(r"entity_[0-9a-f]{64}")
_RECORD_ID = re.compile(r"(?:alias|provenance|task|relationship)_[0-9a-f]{64}")
_CURRENCY = re.compile(r"[A-Z]{3}")
_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,31}")
_MARKET = re.compile(r"[A-Z0-9][A-Z0-9._\-]{0,31}")
_FIGI_CONSONANT = "BCDFGHJKLMNPQRSTVWXYZ"
# Current currency/fund/metal codes from the ISO 4217 Maintenance Agency's List One.
_ISO_4217_CODES = frozenset(
    """AED AFN ALL AMD AOA ARS AUD AWG AZN BAM BBD BDT BHD BIF BMD BND
    BOB BOV BRL BSD BTN BWP BYN BZD CAD CDF CHE CHF CHW CLF CLP CNY COP COU CRC
    CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF
    GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR
    KMF KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT
    MOP MRU MUR MVR MWK MXN MXV MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN
    PGK PHP PKR PLN PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SOS
    SRD SSP STN SVC SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX USD
    USN UYI UYU UYW UZS VED VES VND VUV WST XAD XAF XAG XAU XBA XBB XBC XBD
    XCD XCG XDR XOF XPD XPF XPT XSU XTS XUA
    XXX YER ZAR ZMW ZWG""".split()
)
_RELATIONSHIP_KINDS = {
    "supplier",
    "customer",
    "competitor",
    "complement",
    "substitute",
}
_LEGAL_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "plc",
}


class EntityError(RuntimeError):
    """Base error for entity and exposure boundary failures."""


class EntityValidationError(EntityError, ValueError):
    """Raised when identity, relationship, or exposure input is invalid."""


class MissingEvidence(EntityError, ValueError):
    """Raised when a relationship lacks verified evidence lineage."""


def _text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise EntityValidationError(f"{field_name} must be text")
    normalized = " ".join(value.split())
    if not normalized or any(unicodedata.category(char) == "Cc" for char in normalized):
        raise EntityValidationError(f"{field_name} must be safe nonblank text")
    return normalized


def _optional_text(value: str | None, field_name: str) -> str | None:
    return None if value is None else _text(value, field_name)


def _decimal(value: Decimal, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite():
        raise EntityValidationError(f"{field_name} must be finite")
    return value


def _aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise EntityValidationError(f"{field_name} must be timezone-aware")


def _utc(value: datetime, field_name: str) -> datetime:
    _aware(value, field_name)
    return value.astimezone(timezone.utc)


def _stable_id(prefix: str, *parts: str) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()}"


def _validate_entity_id(value: str, field_name: str = "entity_id") -> None:
    if not isinstance(value, str) or _ENTITY_ID.fullmatch(value) is None:
        raise EntityValidationError(f"{field_name} is invalid")


def _normalize_symbol(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _text(value, "symbol").upper()
    if _SYMBOL.fullmatch(normalized) is None:
        raise EntityValidationError("symbol is invalid")
    return normalized


def _normalize_market(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _text(value, "market").upper()
    if _MARKET.fullmatch(normalized) is None:
        raise EntityValidationError("market is invalid")
    return normalized


def _normalize_currency(value: str, field_name: str) -> str:
    normalized = _text(value, field_name).upper()
    if _CURRENCY.fullmatch(normalized) is None or normalized not in _ISO_4217_CODES:
        raise EntityValidationError(f"{field_name} is invalid")
    return normalized


def _normalize_exposure_symbol(value: str) -> str:
    if isinstance(value, str) and value.upper().startswith("CASH:"):
        prefix, separator, currency = value.upper().partition(":")
        if prefix != "CASH" or separator != ":":
            raise EntityValidationError("exposure symbol is invalid")
        return f"CASH:{_normalize_currency(currency, 'cash currency')}"
    normalized = _normalize_symbol(value)
    if normalized is None:
        raise EntityValidationError("exposure symbol is invalid")
    return normalized


def _deterministic_decimal_sum(terms: Sequence[tuple[str, Decimal]]) -> Decimal:
    ordered = sorted(
        terms,
        key=lambda item: (-abs(item[1]), item[0], item[1].as_tuple()),
    )
    return sum((value for _, value in ordered), Decimal("0"))


def _normalize_cik(value: str | int | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if re.fullmatch(r"[0-9]{1,10}", text) is None:
        raise EntityValidationError("cik is invalid")
    return text.zfill(10)


def _normalize_conid(value: str | int | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if re.fullmatch(r"[1-9][0-9]{0,19}", text) is None:
        raise EntityValidationError("conid is invalid")
    return text


def _luhn_valid(value: str) -> bool:
    digits = "".join(str(ord(char) - 55) if char.isalpha() else char for char in value)
    total = 0
    parity = len(digits) % 2
    for index, character in enumerate(digits):
        number = int(character)
        if index % 2 == parity:
            number *= 2
        total += number // 10 + number % 10
    return total % 10 == 0


def _normalize_isin(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _text(value, "isin").upper()
    if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", normalized) is None or not _luhn_valid(
        normalized
    ):
        raise EntityValidationError("isin is invalid")
    return normalized


def _cusip_value(character: str) -> int:
    if character.isdigit():
        return int(character)
    if character.isalpha():
        return ord(character) - 55
    return {"*": 36, "@": 37, "#": 38}[character]


def _normalize_cusip(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _text(value, "cusip").upper()
    if re.fullmatch(r"[A-Z0-9*@#]{8}[0-9]", normalized) is None:
        raise EntityValidationError("cusip is invalid")
    total = 0
    for index, character in enumerate(normalized[:8]):
        number = _cusip_value(character) * (2 if index % 2 else 1)
        total += number // 10 + number % 10
    if (10 - total % 10) % 10 != int(normalized[-1]):
        raise EntityValidationError("cusip is invalid")
    return normalized


def _normalize_figi(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _text(value, "figi").upper()
    body_pattern = rf"[{_FIGI_CONSONANT}]{{2}}G[{_FIGI_CONSONANT}0-9]{{8}}[0-9]"
    if re.fullmatch(body_pattern, normalized) is None:
        raise EntityValidationError("figi is invalid")
    if normalized[:2] in {"BS", "BM", "GG", "GB", "GH", "KY", "VG"}:
        raise EntityValidationError("figi is invalid")
    total = 0
    for index, character in enumerate(normalized[:11]):
        number = int(character) if character.isdigit() else ord(character) - 55
        if index % 2:
            number *= 2
        total += number // 10 + number % 10
    if (10 - total % 10) % 10 != int(normalized[-1]):
        raise EntityValidationError("figi is invalid")
    return normalized


def _normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", _text(value, "name")).casefold()
    tokens = re.findall(r"[a-z0-9]+", normalized)
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


@dataclass(frozen=True)
class SecurityIdentity:
    """Normalized identifiers presented for deterministic resolution."""

    symbol: str | None = None
    market: str | None = None
    name: str | None = None
    conid: str | int | None = None
    cik: str | int | None = None
    isin: str | None = None
    figi: str | None = None
    cusip: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _normalize_symbol(self.symbol))
        object.__setattr__(self, "market", _normalize_market(self.market))
        object.__setattr__(self, "name", _optional_text(self.name, "name"))
        object.__setattr__(self, "conid", _normalize_conid(self.conid))
        object.__setattr__(self, "cik", _normalize_cik(self.cik))
        object.__setattr__(self, "isin", _normalize_isin(self.isin))
        object.__setattr__(self, "figi", _normalize_figi(self.figi))
        object.__setattr__(self, "cusip", _normalize_cusip(self.cusip))
        identifiers = (self.conid, self.cik, self.isin, self.figi, self.cusip)
        if not any(identifiers) and not self.name and not self.symbol:
            raise EntityValidationError("at least one identity field is required")
        if self.symbol and not self.market and not any(identifiers) and not self.name:
            raise EntityValidationError("symbol-only identities require a market")
        if self.market and not self.symbol:
            raise EntityValidationError("market requires symbol")


@dataclass(frozen=True)
class ResolvedEntity:
    """Canonical entity selected by one explicit resolution method."""

    entity_id: str
    canonical_name: str
    resolution_method: str
    matched_identifier: str

    def __post_init__(self) -> None:
        _validate_entity_id(self.entity_id)
        object.__setattr__(self, "canonical_name", _text(self.canonical_name, "canonical_name"))
        if self.resolution_method not in {
            "conid",
            "cik",
            "isin",
            "figi",
            "cusip",
            "symbol_market",
            "normalized_name",
        }:
            raise EntityValidationError("resolution_method is invalid")
        object.__setattr__(
            self, "matched_identifier", _text(self.matched_identifier, "matched_identifier")
        )


@dataclass(frozen=True)
class Alias:
    """Immutable alternate label for a canonical entity."""

    alias_id: str
    entity_id: str
    value: str
    alias_type: str
    market: str | None

    def __post_init__(self) -> None:
        if _RECORD_ID.fullmatch(self.alias_id) is None or not self.alias_id.startswith("alias_"):
            raise EntityValidationError("alias_id is invalid")
        _validate_entity_id(self.entity_id)
        object.__setattr__(self, "value", _text(self.value, "alias value"))
        if self.alias_type not in {"name", "symbol_market"}:
            raise EntityValidationError("alias_type is invalid")
        object.__setattr__(self, "market", _normalize_market(self.market))


@dataclass(frozen=True)
class ResolutionProvenance:
    """Deterministic explanation of a successful entity match."""

    provenance_id: str
    entity_id: str
    method: str
    source: str
    matched_identifier: str

    def __post_init__(self) -> None:
        if _RECORD_ID.fullmatch(self.provenance_id) is None or not self.provenance_id.startswith(
            "provenance_"
        ):
            raise EntityValidationError("provenance_id is invalid")
        _validate_entity_id(self.entity_id)
        for field_name in ("method", "source", "matched_identifier"):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))


@dataclass(frozen=True)
class UnresolvedResearchTask:
    """Deduplicated research work emitted instead of guessing an identity."""

    task_id: str
    reason: str
    normalized_query: str
    candidate_entity_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if _RECORD_ID.fullmatch(self.task_id) is None or not self.task_id.startswith("task_"):
            raise EntityValidationError("task_id is invalid")
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(self, "normalized_query", _text(self.normalized_query, "normalized_query"))
        if not isinstance(self.candidate_entity_ids, tuple):
            raise EntityValidationError("candidate_entity_ids must be a tuple")
        for item in self.candidate_entity_ids:
            _validate_entity_id(item, "candidate_entity_id")
        if tuple(sorted(set(self.candidate_entity_ids))) != self.candidate_entity_ids:
            raise EntityValidationError("candidate_entity_ids must be sorted and unique")


@dataclass(frozen=True)
class Relationship:
    """One evidence-backed, directed value-chain assertion."""

    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    kind: str
    as_of: datetime
    confidence: Decimal
    stance: str
    evidence_ids: tuple[str, ...]
    supporting_claim_ids: tuple[str, ...]
    provenance: str

    def __post_init__(self) -> None:
        if _RECORD_ID.fullmatch(self.relationship_id) is None or not self.relationship_id.startswith(
            "relationship_"
        ):
            raise EntityValidationError("relationship_id is invalid")
        _validate_entity_id(self.source_entity_id, "source_entity_id")
        _validate_entity_id(self.target_entity_id, "target_entity_id")
        if self.source_entity_id == self.target_entity_id:
            raise EntityValidationError("relationship cannot be a self-edge")
        if self.kind not in _RELATIONSHIP_KINDS:
            raise EntityValidationError("relationship kind is invalid")
        object.__setattr__(self, "as_of", _utc(self.as_of, "relationship as_of"))
        _decimal(self.confidence, "relationship confidence")
        if not Decimal("0") <= self.confidence <= Decimal("1"):
            raise EntityValidationError("relationship confidence must be between 0 and 1")
        if self.stance not in {"supports", "contradicts"}:
            raise EntityValidationError("relationship stance is invalid")
        for field_name in ("evidence_ids", "supporting_claim_ids"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or tuple(sorted(set(values))) != values:
                raise EntityValidationError(f"{field_name} must be a sorted unique tuple")
            for value in values:
                _text(value, field_name)
        if not self.evidence_ids and not self.supporting_claim_ids:
            raise MissingEvidence("relationship requires evidence")
        object.__setattr__(self, "provenance", _text(self.provenance, "provenance"))


@dataclass(frozen=True)
class ETFConstituent:
    """One disclosed ETF constituent and its unit-interval portfolio weight."""

    symbol: str
    weight: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _normalize_symbol(self.symbol))
        _decimal(self.weight, "ETF constituent weight")
        if not Decimal("0") <= self.weight <= Decimal("1"):
            raise EntityValidationError("ETF constituent weight must be between 0 and 1")


@dataclass(frozen=True)
class ETFHoldings:
    """Disclosed ETF constituents with an explicit optional source date."""

    etf_symbol: str
    as_of: datetime | None
    constituents: tuple[ETFConstituent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "etf_symbol", _normalize_symbol(self.etf_symbol))
        if self.as_of is not None:
            _aware(self.as_of, "ETF holdings as_of")
        if not isinstance(self.constituents, tuple) or not self.constituents:
            raise EntityValidationError("ETF constituents must be a nonempty tuple")
        symbols = tuple(item.symbol for item in self.constituents)
        if len(symbols) != len(set(symbols)):
            raise EntityValidationError("ETF constituents must be unique")
        if sum((item.weight for item in self.constituents), Decimal("0")) > Decimal("1"):
            raise EntityValidationError("ETF constituent weights cannot exceed 1 in total")


@dataclass(frozen=True)
class ExposurePosition:
    """Minimal, privacy-safe position input for exposure arithmetic."""

    symbol: str
    market_value: Decimal
    currency: str
    asset_class: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _normalize_symbol(self.symbol))
        _decimal(self.market_value, "position market_value")
        object.__setattr__(self, "currency", _normalize_currency(self.currency, "currency"))
        object.__setattr__(self, "asset_class", _text(self.asset_class, "asset_class").upper())


@dataclass(frozen=True)
class PortfolioExposure:
    """Separated numeric and qualitative exposure for one symbol."""

    symbol: str
    direct_weight: Decimal
    lookthrough_weight: Decimal
    qualitative_relationships: tuple[Relationship, ...] = ()
    etf_holdings_status: str = "not_applicable"

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _normalize_exposure_symbol(self.symbol))
        _decimal(self.direct_weight, "direct_weight")
        _decimal(self.lookthrough_weight, "lookthrough_weight")
        if not isinstance(self.qualitative_relationships, tuple):
            raise EntityValidationError("qualitative_relationships must be a tuple")
        if self.etf_holdings_status not in {
            "not_applicable",
            "current",
            "missing",
            "stale",
            "unknown_date",
            "unevaluated",
            "future",
        }:
            raise EntityValidationError("etf_holdings_status is invalid")

    @property
    def etf_lookthrough_weight(self) -> Decimal:
        return self.lookthrough_weight

    @property
    def total_numeric_weight(self) -> Decimal:
        return self.direct_weight + self.lookthrough_weight


@dataclass(frozen=True)
class _CatalogEntity:
    entity_id: str
    canonical_name: str
    identifiers: tuple[tuple[str, str], ...]
    aliases: tuple[Alias, ...]


class EntityResolver:
    """Identifier-first resolver that emits research tasks instead of guessing."""

    def __init__(
        self,
        entities: Sequence[_CatalogEntity],
        *,
        store: ResearchStore | None = None,
        reference_verifier: Callable[[tuple[str, ...], tuple[str, ...]], bool] | None = None,
    ) -> None:
        self._entities = tuple(sorted(entities, key=lambda item: item.entity_id))
        self._store = store
        self._reference_verifier = reference_verifier
        self._indexes: dict[str, dict[str, tuple[_CatalogEntity, ...]]] = {}
        for method in ("conid", "cik", "isin", "figi", "cusip", "symbol_market", "normalized_name"):
            pending: dict[str, list[_CatalogEntity]] = defaultdict(list)
            for entity in self._entities:
                for identifier_method, value in entity.identifiers:
                    if identifier_method == method:
                        pending[value].append(entity)
            self._indexes[method] = {
                key: tuple(sorted(values, key=lambda item: item.entity_id))
                for key, values in pending.items()
            }
        self._tasks: dict[str, UnresolvedResearchTask] = {}

    @classmethod
    def from_sec_company_tickers(
        cls,
        payload: Mapping[str, Any],
        *,
        store: ResearchStore | None = None,
        reference_verifier: Callable[[tuple[str, ...], tuple[str, ...]], bool] | None = None,
    ) -> EntityResolver:
        """Validate an SEC ticker dataset and build an order-independent catalog."""
        if not isinstance(payload, Mapping):
            raise EntityValidationError("SEC ticker dataset must be a mapping")
        rows: Sequence[Any]
        if set(payload) >= {"fields", "data"}:
            fields = payload["fields"]
            data = payload["data"]
            if not isinstance(fields, list) or not isinstance(data, list):
                raise EntityValidationError("SEC ticker dataset shape is invalid")
            rows = tuple(dict(zip(fields, row, strict=True)) for row in data)
        else:
            rows = tuple(payload.values())
        grouped: dict[str, dict[str, object]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise EntityValidationError("SEC ticker record must be a mapping")
            try:
                canonical_name = _text(row["title"], "SEC title").upper()
                cik = _normalize_cik(row["cik_str"])
                symbol = _normalize_symbol(row["ticker"])
            except (KeyError, TypeError):
                raise EntityValidationError("SEC ticker record is incomplete") from None
            market = _normalize_market(row.get("exchange"))
            optional = {
                "conid": _normalize_conid(row.get("conid")),
                "isin": _normalize_isin(row.get("isin")),
                "figi": _normalize_figi(row.get("figi")),
                "cusip": _normalize_cusip(row.get("cusip")),
            }
            normalized_name = _normalize_name(canonical_name)
            group = grouped.setdefault(
                cik,
                {
                    "canonical_names": set(),
                    "normalized_names": set(),
                    "identifiers": {("cik", cik)},
                    "aliases": set(),
                },
            )
            group["canonical_names"].add(canonical_name)
            group["normalized_names"].add(normalized_name)
            group["identifiers"].update(
                (method, value) for method, value in optional.items() if value
            )
            group["identifiers"].add(("normalized_name", normalized_name))
            group["aliases"].add(("name", canonical_name, None))
            if market is not None:
                symbol_market = f"{symbol}@{market}"
                group["identifiers"].add(("symbol_market", symbol_market))
                group["aliases"].add(("symbol_market", symbol_market, market))
        entities: list[_CatalogEntity] = []
        for cik, group in sorted(grouped.items()):
            if len(group["normalized_names"]) != 1:
                raise EntityValidationError(
                    "SEC ticker records for one CIK have conflicting names"
                )
            canonical_name = sorted(group["canonical_names"])[0]
            entity_id = _stable_id("entity", "cik", cik)
            aliases = tuple(
                Alias(
                    _stable_id("alias", entity_id, alias_type, value, market or ""),
                    entity_id,
                    value,
                    alias_type,
                    market,
                )
                for alias_type, value, market in sorted(
                    group["aliases"], key=lambda item: (item[0], item[1], item[2] or "")
                )
            )
            entities.append(
                _CatalogEntity(
                    entity_id,
                    canonical_name,
                    tuple(sorted(group["identifiers"])),
                    tuple(sorted(aliases, key=lambda item: item.alias_id)),
                )
            )
        return cls(entities, store=store, reference_verifier=reference_verifier)

    @property
    def unresolved_tasks(self) -> tuple[UnresolvedResearchTask, ...]:
        return tuple(sorted(self._tasks.values(), key=lambda item: item.task_id))

    def _emit_task(
        self, reason: str, normalized_query: str, candidates: Sequence[_CatalogEntity]
    ) -> None:
        candidate_ids = tuple(sorted({item.entity_id for item in candidates}))
        task_id = _stable_id("task", reason, normalized_query, *candidate_ids)
        self._tasks.setdefault(
            task_id,
            UnresolvedResearchTask(task_id, reason, normalized_query, candidate_ids),
        )
        if self._store is not None:
            self._store.upsert_unresolved_research_task(self._tasks[task_id])

    def resolve(self, identity: SecurityIdentity) -> ResolvedEntity | None:
        if not isinstance(identity, SecurityIdentity):
            raise TypeError("identity must be SecurityIdentity")
        queries = (
            ("conid", identity.conid),
            ("cik", identity.cik),
            ("isin", identity.isin),
            ("figi", identity.figi),
            ("cusip", identity.cusip),
            (
                "symbol_market",
                f"{identity.symbol}@{identity.market}"
                if identity.symbol and identity.market
                else None,
            ),
            ("normalized_name", _normalize_name(identity.name) if identity.name else None),
        )
        last_query = "unresolved"
        for method, query in queries:
            if query is None:
                continue
            last_query = query
            candidates = self._indexes[method].get(query, ())
            if len(candidates) > 1:
                reason = (
                    "normalized_name_ambiguous"
                    if method == "normalized_name"
                    else "identifier_ambiguous"
                )
                self._emit_task(reason, query, candidates)
                return None
            if len(candidates) == 1:
                catalog = candidates[0]
                resolved = ResolvedEntity(
                    catalog.entity_id, catalog.canonical_name, method, query
                )
                if self._store is not None:
                    provenance = ResolutionProvenance(
                        _stable_id("provenance", catalog.entity_id, method, query),
                        catalog.entity_id,
                        method,
                        "sec_company_tickers",
                        query,
                    )
                    self._store.upsert_resolved_entity(resolved, catalog.aliases, provenance)
                return resolved
        self._emit_task("no_exact_identifier_or_name_match", last_query, ())
        return None

    def add_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        kind: str,
        *,
        as_of: datetime,
        confidence: Decimal,
        evidence_ids: Sequence[str] = (),
        supporting_claim_ids: Sequence[str] = (),
        stance: str = "supports",
        provenance: str = "analyst",
    ) -> Relationship:
        evidence = tuple(sorted(set(evidence_ids)))
        claims = tuple(sorted(set(supporting_claim_ids)))
        if not evidence and not claims:
            raise MissingEvidence("relationship requires evidence")
        verified = False
        if self._store is not None:
            verified = self._store.evidence_references_exist(evidence, claims)
        elif self._reference_verifier is not None:
            verified = self._reference_verifier(evidence, claims)
        if not verified:
            raise MissingEvidence("relationship evidence does not exist")
        observed_at = _utc(as_of, "relationship as_of")
        relationship_id = _stable_id(
            "relationship",
            source_entity_id,
            target_entity_id,
            kind,
            observed_at.isoformat(),
            format(confidence, "f") if isinstance(confidence, Decimal) else "invalid",
            stance,
            *(f"passage:{item}" for item in evidence),
            *(f"claim:{item}" for item in claims),
            provenance,
        )
        relationship = Relationship(
            relationship_id,
            source_entity_id,
            target_entity_id,
            kind,
            observed_at,
            confidence,
            stance,
            evidence,
            claims,
            provenance,
        )
        if self._store is not None:
            self._store.insert_relationship(relationship)
        return relationship

    def add_relationship_from_similarity(
        self,
        source_entity_id: str,
        target_entity_id: str,
        kind: str,
        similarity: Decimal,
    ) -> Relationship:
        """Reject similarity-only graph writes at the public boundary."""
        raise MissingEvidence("semantic similarity is not relationship evidence")


class PortfolioExposureMapper:
    """Compute deterministic exposure; future ETF snapshots have zero tolerance."""

    def __init__(self, max_etf_holdings_age: timedelta = timedelta(days=45)) -> None:
        if not isinstance(max_etf_holdings_age, timedelta) or max_etf_holdings_age < timedelta(0):
            raise EntityValidationError("max_etf_holdings_age must be non-negative")
        self.max_etf_holdings_age = max_etf_holdings_age

    def map_positions(
        self,
        positions: Sequence[ExposurePosition],
        *,
        nav: Decimal,
        base_currency: str,
        etf_holdings: Sequence[ETFHoldings] = (),
        as_of: datetime | None = None,
        fx_to_base: Mapping[str, Decimal] | None = None,
        relationship_exposures: Mapping[str, Sequence[Relationship]] | None = None,
    ) -> dict[str, PortfolioExposure]:
        _decimal(nav, "nav")
        if nav <= 0:
            raise EntityValidationError("nav must be positive")
        base = _normalize_currency(base_currency, "base_currency")
        if as_of is not None:
            _aware(as_of, "as_of")
        fx_rates: dict[str, Decimal] = {}
        for raw_currency, rate in (fx_to_base or {}).items():
            currency = _normalize_currency(raw_currency, "FX currency")
            if currency in fx_rates:
                raise EntityValidationError("duplicate FX currency")
            fx_rates[currency] = rate
        for currency, rate in fx_rates.items():
            _decimal(rate, "FX rate")
            if rate <= 0:
                raise EntityValidationError("FX rate must be positive")
        holdings_by_symbol: dict[str, ETFHoldings] = {}
        for item in etf_holdings:
            if item.etf_symbol in holdings_by_symbol:
                raise EntityValidationError("duplicate ETF holdings snapshots")
            holdings_by_symbol[item.etf_symbol] = item
        direct_terms: dict[str, list[tuple[str, Decimal]]] = defaultdict(list)
        lookthrough_terms: dict[str, list[tuple[str, Decimal]]] = defaultdict(list)
        statuses: dict[str, str] = {}
        relationships: dict[str, dict[str, Relationship]] = defaultdict(dict)
        for raw_symbol, assertions in (relationship_exposures or {}).items():
            symbol = _normalize_exposure_symbol(raw_symbol)
            if not isinstance(assertions, Sequence) or isinstance(assertions, (str, bytes)):
                raise EntityValidationError("relationship exposures must be a sequence")
            for assertion in assertions:
                if not isinstance(assertion, Relationship):
                    raise EntityValidationError(
                        "relationship exposures must contain Relationship records"
                    )
                existing = relationships[symbol].get(assertion.relationship_id)
                if existing is not None and existing != assertion:
                    raise EntityValidationError("conflicting relationship identity")
                relationships[symbol][assertion.relationship_id] = assertion
        cash_input_symbols: set[str] = set()
        for item in positions:
            if not isinstance(item, ExposurePosition):
                raise TypeError("positions must contain ExposurePosition records")
            if item.currency == base:
                base_value = item.market_value
            else:
                rate = fx_rates.get(item.currency)
                if rate is None:
                    raise EntityValidationError("foreign-currency position requires FX rate")
                base_value = item.market_value * rate
            position_term_id = _stable_id(
                "term",
                item.symbol,
                item.currency,
                item.asset_class,
                format(item.market_value, "f"),
                format(base_value, "f"),
            )
            if item.asset_class == "CASH":
                direct_terms[f"CASH:{item.currency}"].append(
                    (position_term_id, base_value / nav)
                )
                cash_input_symbols.add(item.symbol)
                continue
            if item.asset_class in {"STK", "EQUITY", "ETF", "FUND"}:
                direct_terms[item.symbol].append((position_term_id, base_value / nav))
            else:
                continue
            if item.asset_class not in {"ETF", "FUND"}:
                continue
            holdings = holdings_by_symbol.get(item.symbol)
            if holdings is None:
                statuses[item.symbol] = "missing"
                continue
            if holdings.as_of is None:
                statuses[item.symbol] = "unknown_date"
                continue
            if as_of is None:
                statuses[item.symbol] = "unevaluated"
                continue
            age = as_of - holdings.as_of
            if age < timedelta(0):
                statuses[item.symbol] = "future"
                continue
            if age > self.max_etf_holdings_age:
                statuses[item.symbol] = "stale"
                continue
            statuses[item.symbol] = "current"
            etf_weight = base_value / nav
            for constituent in holdings.constituents:
                constituent_term_id = _stable_id(
                    "term",
                    position_term_id,
                    constituent.symbol,
                    format(constituent.weight, "f"),
                )
                lookthrough_terms[constituent.symbol].append(
                    (constituent_term_id, etf_weight * constituent.weight)
                )
        direct = {
            symbol: _deterministic_decimal_sum(terms)
            for symbol, terms in direct_terms.items()
        }
        lookthrough = {
            symbol: _deterministic_decimal_sum(terms)
            for symbol, terms in lookthrough_terms.items()
        }
        relationship_symbols = {
            symbol
            for symbol in relationships
            if symbol not in cash_input_symbols and not symbol.upper().startswith("CASH:")
        }
        symbols = set(direct) | set(lookthrough) | relationship_symbols
        result = {}
        for symbol in sorted(symbols):
            qualitative = (
                ()
                if symbol.startswith("CASH:")
                else tuple(
                    sorted(
                        relationships.get(symbol, {}).values(),
                        key=lambda item: item.relationship_id,
                    )
                )
            )
            result[symbol] = PortfolioExposure(
                symbol=symbol,
                direct_weight=direct.get(symbol, Decimal("0")),
                lookthrough_weight=lookthrough.get(symbol, Decimal("0")),
                qualitative_relationships=qualitative,
                etf_holdings_status=statuses.get(symbol, "not_applicable"),
            )
        return result
