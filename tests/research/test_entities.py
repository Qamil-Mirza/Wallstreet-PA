"""Deterministic entity resolution and portfolio exposure contracts."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from news_bot.research.entities import (
    Alias,
    EntityResolver,
    EntityValidationError,
    ETFConstituent,
    ETFHoldings,
    ExposurePosition,
    MissingEvidence,
    PortfolioExposureMapper,
    Relationship,
    ResolutionProvenance,
    ResolvedEntity,
    SecurityIdentity,
    UnresolvedResearchTask,
)
from news_bot.research.models import SourceDocument


FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 8, 24, tzinfo=timezone.utc)


def ticker_data():
    return json.loads((FIXTURES / "sec_company_tickers.json").read_text())


def resolver(**kwargs):
    return EntityResolver.from_sec_company_tickers(ticker_data(), **kwargs)


def seed_passage(store, passage_id="passage-nvda"):
    store.insert_source_document(
        SourceDocument(
            document_id="document-nvda",
            source_type="filing",
            canonical_url="https://example.test/nvda",
            publisher="Example",
            published_at=NOW,
            retrieved_at=NOW,
            content_hash="sha256:nvda-relationship",
            raw_content_path=None,
            extraction_status="complete",
        )
    )
    store.insert_document_passage(
        passage_id=passage_id,
        document_id="document-nvda",
        ordinal=0,
        text="Company A supplies Company B.",
    )


def test_records_are_frozen_and_validate_their_boundaries():
    identity = SecurityIdentity(symbol=" nvda ", market=" nasdaq ", cik="1045810")

    assert identity.symbol == "NVDA"
    assert identity.market == "NASDAQ"
    assert identity.cik == "0001045810"
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.symbol = "INTC"
    with pytest.raises(EntityValidationError):
        SecurityIdentity(symbol="NV\nDA", market="NASDAQ")
    with pytest.raises(EntityValidationError):
        SecurityIdentity()

    records = (
        ResolvedEntity("entity_" + "a" * 64, "NVIDIA CORP", "cik", "0001045810"),
        Alias("alias_" + "a" * 64, "entity_" + "a" * 64, "NVIDIA", "name", None),
        ResolutionProvenance(
            "provenance_" + "a" * 64,
            "entity_" + "a" * 64,
            "cik",
            "sec_company_tickers",
            "0001045810",
        ),
        UnresolvedResearchTask(
            "task_" + "a" * 64, "normalized_name_ambiguous", "acme holdings", ()
        ),
        Relationship(
            "relationship_" + "a" * 64,
            "entity_" + "a" * 64,
            "entity_" + "b" * 64,
            "supplier",
            NOW,
            Decimal("0.80"),
            "supports",
            ("passage-1",),
            (),
            "analyst",
        ),
    )
    for record in records:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, "changed", True)


@pytest.mark.parametrize(
    "changes",
    [
        {"conid": "abc"},
        {"cik": "12345678901"},
        {"isin": "US67066G1041"},
        {"figi": "BBG000BBJQV1"},
        {"cusip": "458140101"},
        {"symbol": "NVDA", "market": None, "name": None, "cik": None},
    ],
)
def test_security_identity_rejects_invalid_or_unqualified_identifiers(changes):
    values = dict(
        symbol=None,
        market=None,
        name=None,
        conid=None,
        cik="0001045810",
        isin=None,
        figi=None,
        cusip=None,
    )
    values.update(changes)

    with pytest.raises(EntityValidationError):
        SecurityIdentity(**values)


def test_identifier_precedence_beats_adversarial_name_and_lower_priority_id():
    entity = resolver().resolve(
        SecurityIdentity(
            conid="4815747",
            cik="0000050863",
            name="Intel Corporation",
        )
    )

    assert entity is not None
    assert entity.canonical_name == "NVIDIA CORP"
    assert entity.resolution_method == "conid"


@pytest.mark.parametrize(
    ("identity", "method"),
    [
        (SecurityIdentity(cik="1045810"), "cik"),
        (SecurityIdentity(isin="us67066g1040"), "isin"),
        (SecurityIdentity(figi="bbg000bbjqv0"), "figi"),
        (SecurityIdentity(cusip="458140100"), "cusip"),
        (SecurityIdentity(symbol="nvda", market="nasdaq"), "symbol_market"),
    ],
)
def test_identifier_resolution_is_normalized_and_deterministic(identity, method):
    first = resolver().resolve(identity)
    second = resolver().resolve(identity)

    assert first is not None
    assert first == second
    assert first.resolution_method == method
    assert len(first.entity_id.removeprefix("entity_")) == 64


def test_cross_market_symbol_is_never_resolved_without_market_qualification():
    with pytest.raises(EntityValidationError):
        SecurityIdentity(symbol="SHOP")

    nyse = resolver().resolve(SecurityIdentity(symbol="SHOP", market="NYSE"))
    tsx = resolver().resolve(SecurityIdentity(symbol="SHOP", market="TSX"))

    assert nyse is not None and tsx is not None
    assert nyse.entity_id != tsx.entity_id


def test_ambiguous_normalized_name_emits_one_deduplicated_research_task():
    subject = resolver()
    identity = SecurityIdentity(name="  Acme  Holdings, LLC ")

    assert subject.resolve(identity) is None
    assert subject.resolve(identity) is None
    tasks = subject.unresolved_tasks

    assert len(tasks) == 1
    assert tasks[0].reason == "normalized_name_ambiguous"
    assert len(tasks[0].candidate_entity_ids) == 2


def test_unknown_name_stays_unresolved_without_fuzzy_guessing():
    subject = resolver()

    assert subject.resolve(SecurityIdentity(name="NVIDA Corporation")) is None
    assert subject.unresolved_tasks[0].reason == "no_exact_identifier_or_name_match"


def test_resolution_is_order_independent_and_entity_ids_are_collision_safe():
    forward = EntityResolver.from_sec_company_tickers(ticker_data())
    reversed_data = dict(reversed(tuple(ticker_data().items())))
    backward = EntityResolver.from_sec_company_tickers(reversed_data)
    identities = (
        SecurityIdentity(cik="1045810"),
        SecurityIdentity(cik="50863"),
        SecurityIdentity(symbol="GOOG", market="NASDAQ"),
    )

    assert [forward.resolve(item) for item in identities] == [
        backward.resolve(item) for item in identities
    ]
    assert len({forward.resolve(item).entity_id for item in identities}) == 3


def test_multiple_sec_tickers_for_one_cik_merge_into_one_entity():
    payload = ticker_data()
    payload["7"] = {
        "cik_str": 1652044,
        "ticker": "GOOGL",
        "title": "ALPHABET INC",
        "exchange": "NASDAQ",
    }
    subject = EntityResolver.from_sec_company_tickers(payload)

    by_cik = subject.resolve(SecurityIdentity(cik="1652044"))
    by_first_ticker = subject.resolve(SecurityIdentity(symbol="GOOG", market="NASDAQ"))
    by_second_ticker = subject.resolve(SecurityIdentity(symbol="GOOGL", market="NASDAQ"))

    assert by_cik.entity_id == by_first_ticker.entity_id == by_second_ticker.entity_id


def test_aliases_and_resolution_provenance_are_persisted_idempotently(migrated_store):
    subject = resolver(store=migrated_store)

    first = subject.resolve(SecurityIdentity(cik="1045810"))
    second = subject.resolve(SecurityIdentity(cik="0001045810"))

    assert first == second
    aliases = migrated_store.list_entity_aliases(first.entity_id)
    provenance = migrated_store.list_resolution_provenance(first.entity_id)
    assert {(item.alias_type, item.value) for item in aliases} >= {
        ("name", "NVIDIA CORP"),
        ("symbol_market", "NVDA@NASDAQ"),
    }
    assert [item.method for item in provenance] == ["cik"]


def test_unresolved_ambiguity_persists_one_deduplicated_research_task(migrated_store):
    subject = resolver(store=migrated_store)
    identity = SecurityIdentity(name="Acme Holdings LLC")

    subject.resolve(identity)
    subject.resolve(identity)

    with migrated_store.connect() as connection:
        tasks = connection.execute(
            "SELECT task_id, task_kind, state FROM research_tasks"
        ).fetchall()
    assert tasks == [
        (subject.unresolved_tasks[0].task_id, "entity_resolution", "pending")
    ]


def test_relationship_requires_at_least_one_existing_evidence_reference(migrated_store):
    subject = resolver(store=migrated_store)
    source = subject.resolve(SecurityIdentity(cik="1045810"))
    target = subject.resolve(SecurityIdentity(cik="50863"))

    with pytest.raises(MissingEvidence):
        subject.add_relationship(
            source.entity_id,
            target.entity_id,
            "supplier",
            as_of=NOW,
            confidence=Decimal("0.8"),
            evidence_ids=(),
        )
    with pytest.raises(MissingEvidence):
        subject.add_relationship(
            source.entity_id,
            target.entity_id,
            "supplier",
            as_of=NOW,
            confidence=Decimal("0.8"),
            evidence_ids=("missing-passage",),
        )


def test_relationship_is_directed_idempotent_and_evidence_backed(migrated_store):
    seed_passage(migrated_store)
    subject = resolver(store=migrated_store)
    source = subject.resolve(SecurityIdentity(cik="1045810"))
    target = subject.resolve(SecurityIdentity(cik="50863"))

    first = subject.add_relationship(
        source.entity_id,
        target.entity_id,
        "supplier",
        as_of=NOW,
        confidence=Decimal("0.8"),
        evidence_ids=("passage-nvda",),
        provenance="issuer_filing",
    )
    second = subject.add_relationship(
        source.entity_id,
        target.entity_id,
        "supplier",
        as_of=NOW,
        confidence=Decimal("0.8"),
        evidence_ids=("passage-nvda",),
        provenance="issuer_filing",
    )

    assert first == second
    assert first.source_entity_id == source.entity_id
    assert first.target_entity_id == target.entity_id
    assert migrated_store.list_relationships(source.entity_id) == (first,)


def test_relationship_rejects_self_edges_and_semantic_similarity(migrated_store):
    seed_passage(migrated_store)
    subject = resolver(store=migrated_store)
    source = subject.resolve(SecurityIdentity(cik="1045810"))

    with pytest.raises(EntityValidationError):
        subject.add_relationship(
            source.entity_id,
            source.entity_id,
            "competitor",
            as_of=NOW,
            confidence=Decimal("0.8"),
            evidence_ids=("passage-nvda",),
        )
    with pytest.raises(MissingEvidence):
        subject.add_relationship_from_similarity(
            source.entity_id, "entity_" + "b" * 64, "competitor", Decimal("0.99")
        )


def test_contradicting_relationship_assertion_is_retained_separately(migrated_store):
    seed_passage(migrated_store)
    migrated_store.insert_document_passage(
        passage_id="passage-contradiction",
        document_id="document-nvda",
        ordinal=1,
        text="Company A no longer supplies Company B.",
    )
    subject = resolver(store=migrated_store)
    source = subject.resolve(SecurityIdentity(cik="1045810"))
    target = subject.resolve(SecurityIdentity(cik="50863"))
    common = dict(
        source_entity_id=source.entity_id,
        target_entity_id=target.entity_id,
        kind="supplier",
        as_of=NOW,
        confidence=Decimal("0.8"),
        provenance="issuer_filing",
    )

    support = subject.add_relationship(
        **common, evidence_ids=("passage-nvda",), stance="supports"
    )
    contradiction = subject.add_relationship(
        **common,
        evidence_ids=("passage-contradiction",),
        stance="contradicts",
    )

    assert support.relationship_id != contradiction.relationship_id
    assert migrated_store.list_relationships(source.entity_id) == (
        contradiction,
        support,
    )


def position(symbol, value, *, asset_class="STK", currency="USD"):
    return ExposurePosition(
        symbol=symbol,
        market_value=Decimal(str(value)),
        currency=currency,
        asset_class=asset_class,
    )


def test_etf_overlap_rolls_up_underlying_without_double_counting():
    mapper = PortfolioExposureMapper()
    holdings = ETFHoldings(
        etf_symbol="SMH",
        as_of=NOW,
        constituents=(ETFConstituent("NVDA", Decimal("0.18")),),
    )

    exposure = mapper.map_positions(
        (position("NVDA", 20), position("SMH", 30, asset_class="ETF")),
        nav=Decimal("100"),
        base_currency="USD",
        etf_holdings=(holdings,),
    )

    assert exposure["NVDA"].direct_weight == Decimal("0.20")
    assert exposure["NVDA"].lookthrough_weight == Decimal("0.054")
    assert exposure["NVDA"].total_numeric_weight == Decimal("0.254")


def test_exposure_aggregation_is_decimal_only_and_order_independent():
    mapper = PortfolioExposureMapper()
    positions = (
        position("NVDA", Decimal("10.00")),
        position("NVDA", Decimal("5.00")),
        position("CASH", Decimal("5.00"), asset_class="CASH"),
        position("BOND", Decimal("10.00"), asset_class="BOND"),
    )

    forward = mapper.map_positions(positions, nav=Decimal("100"), base_currency="USD")
    backward = mapper.map_positions(
        tuple(reversed(positions)), nav=Decimal("100"), base_currency="USD"
    )

    assert forward == backward
    assert forward["NVDA"].direct_weight == Decimal("0.15")
    assert "CASH" not in forward
    assert "BOND" not in forward


def test_missing_and_stale_etf_holdings_are_explicit_not_fabricated():
    mapper = PortfolioExposureMapper(max_etf_holdings_age=timedelta(days=7))
    stale = ETFHoldings(
        etf_symbol="OLD",
        as_of=NOW - timedelta(days=8),
        constituents=(ETFConstituent("NVDA", Decimal("0.18")),),
    )

    exposure = mapper.map_positions(
        (
            position("SMH", 30, asset_class="ETF"),
            position("OLD", 20, asset_class="ETF"),
        ),
        nav=Decimal("100"),
        base_currency="USD",
        etf_holdings=(stale,),
        as_of=NOW,
    )

    assert exposure["SMH"].etf_holdings_status == "missing"
    assert exposure["OLD"].etf_holdings_status == "stale"
    assert "NVDA" not in exposure


@pytest.mark.parametrize(
    "holding_specs",
    [
        (("NVDA", Decimal("-0.01")),),
        (("NVDA", Decimal("1.01")),),
        (
            ("NVDA", Decimal("0.6")),
            ("INTC", Decimal("0.5")),
        ),
        (
            ("NVDA", Decimal("0.2")),
            ("NVDA", Decimal("0.3")),
        ),
    ],
)
def test_etf_holdings_validate_bounded_unique_sum_safe_weights(holding_specs):
    with pytest.raises(EntityValidationError):
        holdings = tuple(ETFConstituent(*spec) for spec in holding_specs)
        ETFHoldings(etf_symbol="SMH", as_of=NOW, constituents=holdings)


def test_exposure_rejects_float_nav_invalid_currency_and_missing_fx():
    mapper = PortfolioExposureMapper()

    with pytest.raises((TypeError, EntityValidationError)):
        mapper.map_positions((position("NVDA", 20),), nav=100.0, base_currency="USD")
    with pytest.raises(EntityValidationError):
        ExposurePosition("NVDA", Decimal("20"), "US", "STK")
    with pytest.raises(EntityValidationError):
        mapper.map_positions(
            (position("SHOP", 20, currency="CAD"),),
            nav=Decimal("100"),
            base_currency="USD",
        )


def test_foreign_currency_exposure_uses_explicit_decimal_fx_rate():
    mapper = PortfolioExposureMapper()

    exposure = mapper.map_positions(
        (position("SHOP", 25, currency="CAD"),),
        nav=Decimal("100"),
        base_currency="USD",
        fx_to_base={"CAD": Decimal("0.80")},
    )

    assert exposure["SHOP"].direct_weight == Decimal("0.20")


def test_qualitative_relationship_exposure_never_becomes_numeric():
    mapper = PortfolioExposureMapper()
    relationship = Relationship(
        relationship_id="relationship_" + "a" * 64,
        source_entity_id="entity_" + "a" * 64,
        target_entity_id="entity_" + "b" * 64,
        kind="supplier",
        as_of=NOW,
        confidence=Decimal("0.8"),
        stance="supports",
        evidence_ids=("passage-1",),
        supporting_claim_ids=(),
        provenance="filing",
    )

    exposure = mapper.map_positions(
        (position("NVDA", 20),),
        nav=Decimal("100"),
        base_currency="USD",
        relationship_exposures={"NVDA": (relationship,)},
    )

    assert exposure["NVDA"].direct_weight == Decimal("0.20")
    assert exposure["NVDA"].lookthrough_weight == Decimal("0")
    assert exposure["NVDA"].qualitative_relationships == (relationship,)
