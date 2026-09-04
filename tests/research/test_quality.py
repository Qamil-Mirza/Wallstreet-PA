from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from news_bot.research.agents.contracts import SecurityEligibility
from news_bot.research.models import InferenceMode, RecommendationRating, ReviewVerdict
from news_bot.research.quality import (
    CalculatedExhibit,
    CalculatedRow,
    ClaimQualityInput,
    EvidenceReference,
    ExchangeCalendar,
    FilingAvailability,
    GateReasonCode,
    InferenceDisclosure,
    MarketPrice,
    PublicationKind,
    PublicationVerdict,
    QualityGateInput,
    QualityGatePolicy,
    QualityGate,
    RecommendationGate,
    SourceReference,
    evaluate_quality_gates,
    price_is_fresh,
)


UTC = timezone.utc
AS_OF = datetime(2026, 8, 24, 22, 0, tzinfo=UTC)


def security(**changes) -> SecurityEligibility:
    values = {
        "entity_id": "entity-nvda",
        "security_id": "security-nvda",
        "symbol": "NVDA",
        "currency": "USD",
        "asset_kind": "equity",
        "resolved": True,
        "public": True,
        "tradable": True,
    }
    values.update(changes)
    return SecurityEligibility(**values)


def source(source_id: str = "sec-10q", **changes) -> EvidenceReference:
    values = {
        "evidence_id": f"passage-{source_id}",
        "canonical_source_id": source_id,
        "source_family": source_id,
        "source_type": "sec_filing",
        "stored": True,
        "primary": True,
    }
    values.update(changes)
    return EvidenceReference(**values)


def claim(claim_id: str = "claim-1", **changes) -> ClaimQualityInput:
    values = {
        "claim_id": claim_id,
        "section": "recommendation",
        "material": True,
        "quantitative": False,
        "consequential": True,
        "evidence": (source(),),
    }
    values.update(changes)
    return ClaimQualityInput(**values)


def disclosure(**changes) -> InferenceDisclosure:
    values = {
        "mode": InferenceMode.EXTERNAL,
        "provider": "openai",
        "model": "gpt-test",
        "confidence": Decimal("0.90"),
    }
    values.update(changes)
    return InferenceDisclosure(**values)


def request(**changes) -> QualityGateInput:
    values = {
        "as_of": AS_OF,
        "publication_kind": PublicationKind.RECOMMENDATION,
        "requested_rating": RecommendationRating.BUY,
        "portfolio_snapshot_at": AS_OF - timedelta(hours=2),
        "price": MarketPrice(
            exchange="NASDAQ",
            session_date=date(2026, 8, 24),
            observed_at=datetime(2026, 8, 24, 20, 1, tzinfo=UTC),
            value=Decimal("175.50"),
            currency="USD",
        ),
        "filing": FilingAvailability(
            filing_due=True,
            available_as_of=True,
            filed_at=datetime(2026, 8, 20, tzinfo=UTC),
        ),
        "security": security(),
        "claims": (claim(),),
        "contradictory_claim_ids": (),
        "reviewer_verdict": ReviewVerdict.PASS,
        "reviewer_approved_claim_ids": ("claim-1",),
        "editor_claim_ids": ("claim-1",),
        "exhibits": (),
        "inference_disclosure": disclosure(),
    }
    values.update(changes)
    return QualityGateInput(**values)


def reason_values(result) -> tuple[str, ...]:
    return tuple(reason.value for reason in result.reason_codes)


def test_stale_portfolio_allows_event_report_but_blocks_sizing():
    result = evaluate_quality_gates(
        request(portfolio_snapshot_at=AS_OF - timedelta(hours=37))
    )

    assert result.allow_event_report is True
    assert result.allow_sizing is False
    assert result.portfolio_age_hours == Decimal("37")
    assert GateReasonCode.PORTFOLIO_STALE in result.reason_codes


@pytest.mark.parametrize(
    "age, allowed",
    [(timedelta(hours=36), True), (timedelta(hours=36, microseconds=1), False)],
)
def test_portfolio_age_boundary(age, allowed):
    result = evaluate_quality_gates(request(portfolio_snapshot_at=AS_OF - age))
    assert result.allow_sizing is allowed


def test_uncited_material_number_forces_block():
    uncited = claim(quantitative=True, evidence=())
    result = evaluate_quality_gates(request(claims=(uncited,)))

    assert result.review_verdict is ReviewVerdict.BLOCK
    assert result.publication_verdict is PublicationVerdict.DRAFT
    assert result.effective_rating is RecommendationRating.NO_RATING
    assert GateReasonCode.MISSING_CLAIM_LINEAGE in result.reason_codes
    assert "recommendation" not in result.allowed_sections


def test_friday_price_is_fresh_on_monday_for_nasdaq():
    friday = MarketPrice(
        exchange="NASDAQ",
        session_date=date(2026, 8, 21),
        observed_at=datetime(2026, 8, 21, 20, 1, tzinfo=UTC),
        value=Decimal("175.50"),
        currency="USD",
    )
    monday = AS_OF.replace(hour=18)
    result = evaluate_quality_gates(request(
        as_of=monday, portfolio_snapshot_at=monday - timedelta(hours=2), price=friday
    ))
    assert GateReasonCode.PRICE_STALE not in result.reason_codes
    assert GateReasonCode.PRICE_SESSION_UNKNOWN not in result.reason_codes
    assert result.effective_rating is RecommendationRating.BUY


def test_unknown_filing_due_forces_no_rating():
    result = evaluate_quality_gates(
        request(filing=FilingAvailability(filing_due=None, available_as_of=None))
    )
    assert result.effective_rating is RecommendationRating.NO_RATING
    assert GateReasonCode.FILING_STATUS_UNKNOWN in result.reason_codes


def test_known_filing_due_with_unknown_availability_forces_no_rating():
    result = evaluate_quality_gates(
        request(filing=FilingAvailability(filing_due=False, available_as_of=None))
    )
    assert result.effective_rating is RecommendationRating.NO_RATING
    assert GateReasonCode.FILING_STATUS_UNKNOWN in result.reason_codes


@pytest.mark.parametrize(
    "price_date,reason",
    [
        (date(2026, 8, 20), None),
        (date(2026, 8, 19), GateReasonCode.PRICE_STALE),
        (date(2026, 8, 23), GateReasonCode.PRICE_NOT_TRADING_SESSION),
    ],
)
def test_price_four_day_boundary_and_weekend_fail_closed(price_date, reason):
    calendar = ExchangeCalendar(
        exchange="TEST",
        timezone_name="UTC",
        close_time=time(0, 0),
        holidays=(),
    )
    price = MarketPrice(
        exchange="TEST", session_date=price_date,
        observed_at=datetime.combine(price_date, time(1), UTC),
        value=Decimal("1"), currency="USD",
    )
    result = evaluate_quality_gates(
        request(price=price), calendars={"TEST": calendar}
    )
    if reason is None:
        assert GateReasonCode.PRICE_STALE not in result.reason_codes
    else:
        assert reason in result.reason_codes
        assert result.effective_rating is RecommendationRating.NO_RATING


def test_price_calendar_age_uses_exchange_local_date():
    calendar = ExchangeCalendar(
        exchange="PACIFIC", timezone_name="America/Los_Angeles",
        close_time=time(16),
        holidays=(date(2026, 8, 21), date(2026, 8, 24)),
    )
    as_of = datetime(2026, 8, 25, 0, 30, tzinfo=UTC)
    thursday = MarketPrice(
        exchange="PACIFIC", session_date=date(2026, 8, 20),
        observed_at=datetime(2026, 8, 20, 23, 1, tzinfo=UTC),
        value=Decimal("1"), currency="USD",
    )
    result = evaluate_quality_gates(request(
        as_of=as_of,
        portfolio_snapshot_at=as_of - timedelta(hours=2),
        price=thursday,
    ), calendars={"PACIFIC": calendar})
    assert GateReasonCode.PRICE_STALE not in result.reason_codes
    assert GateReasonCode.PRICE_NOT_LATEST_SESSION not in result.reason_codes


def test_holiday_price_and_unknown_exchange_fail_closed():
    holiday_calendar = ExchangeCalendar(
        exchange="TEST", timezone_name="UTC", close_time=time(20),
        holidays=(date(2026, 8, 24),),
    )
    holiday_price = MarketPrice(
        exchange="TEST", session_date=date(2026, 8, 24),
        observed_at=AS_OF, value=Decimal("1"), currency="USD",
    )
    holiday = evaluate_quality_gates(
        request(price=holiday_price), calendars={"TEST": holiday_calendar}
    )
    unknown = evaluate_quality_gates(
        request(price=holiday_price.model_copy(update={"exchange": "UNKNOWN"}))
    )
    assert GateReasonCode.PRICE_NOT_TRADING_SESSION in holiday.reason_codes
    assert GateReasonCode.PRICE_SESSION_UNKNOWN in unknown.reason_codes


def test_exchange_close_time_must_be_naive_local_wall_time():
    with pytest.raises(ValidationError, match="local wall time"):
        ExchangeCalendar(
            exchange="BAD", timezone_name="UTC",
            close_time=time(16, tzinfo=UTC), holidays=(),
        )


def test_calendar_override_controls_latest_completed_session():
    calendar = ExchangeCalendar(
        exchange="X24", timezone_name="UTC", close_time=time(23, 59),
        weekend_days=(), holidays=(),
    )
    prior = MarketPrice(
        exchange="X24", session_date=date(2026, 8, 23),
        observed_at=datetime(2026, 8, 23, 23, 59, tzinfo=UTC),
        value=Decimal("1"), currency="USD",
    )
    result = evaluate_quality_gates(request(price=prior), calendars={"X24": calendar})
    assert GateReasonCode.PRICE_NOT_LATEST_SESSION not in result.reason_codes


def test_price_observation_before_exchange_close_fails_closed():
    premature = MarketPrice(
        exchange="NASDAQ", session_date=date(2026, 8, 24),
        observed_at=datetime(2026, 8, 24, 19, 59, tzinfo=UTC),
        value=Decimal("1"), currency="USD",
    )
    result = evaluate_quality_gates(request(price=premature))
    assert GateReasonCode.PRICE_SESSION_UNKNOWN in result.reason_codes
    assert result.effective_rating is RecommendationRating.NO_RATING


def test_malformed_calendar_override_fails_closed():
    malformed = ExchangeCalendar.model_construct(
        exchange="NASDAQ", timezone_name="Not/AZone", close_time=time(16),
        weekend_days=(5, 6), holidays=(),
    )
    result = evaluate_quality_gates(request(), calendars={"NASDAQ": malformed})
    assert GateReasonCode.PRICE_SESSION_UNKNOWN in result.reason_codes
    assert result.effective_rating is RecommendationRating.NO_RATING


def test_two_canonical_independent_sources_corroborate_conclusion():
    secondary_1 = source("wire-1", primary=False, source_type="news")
    secondary_2 = source("wire-2", primary=False, source_type="news")
    passed = evaluate_quality_gates(
        request(claims=(claim(evidence=(secondary_1, secondary_2)),))
    )
    duplicate = evaluate_quality_gates(
        request(claims=(claim(evidence=(secondary_1, secondary_1.model_copy(
            update={"evidence_id": "another-passage"}
        ))),))
    )
    assert GateReasonCode.INSUFFICIENT_CORROBORATION not in passed.reason_codes
    assert GateReasonCode.INSUFFICIENT_CORROBORATION in duplicate.reason_codes


def test_search_snippets_are_inadmissible_even_when_stored():
    snippet = source("search-1", source_type="search_snippet", primary=True)
    result = evaluate_quality_gates(request(claims=(claim(evidence=(snippet,)),)))
    assert GateReasonCode.SEARCH_SNIPPET_INADMISSIBLE in result.reason_codes
    assert GateReasonCode.MISSING_CLAIM_LINEAGE in result.reason_codes


def test_mixed_search_snippet_blocks_its_affected_section():
    result = evaluate_quality_gates(request(claims=(claim(evidence=(
        source(),
        source("search-1", source_type="search_snippet", primary=False),
    )),)))
    assert GateReasonCode.SEARCH_SNIPPET_INADMISSIBLE in result.reason_codes
    assert result.allowed_sections == ()
    assert result.effective_rating is RecommendationRating.NO_RATING


def test_unstored_evidence_does_not_establish_lineage():
    unstored = source(stored=False)
    result = evaluate_quality_gates(request(claims=(claim(evidence=(unstored,)),)))
    assert GateReasonCode.MISSING_CLAIM_LINEAGE in result.reason_codes


def test_unresolved_contradiction_blocks_only_affected_section_and_rating():
    facts = claim("claim-facts", section="industry")
    disputed = claim("claim-1", section="recommendation")
    result = evaluate_quality_gates(
        request(
            claims=(facts, disputed),
            contradictory_claim_ids=("claim-1",),
            reviewer_approved_claim_ids=("claim-1", "claim-facts"),
            editor_claim_ids=("claim-1", "claim-facts"),
        )
    )
    assert result.allowed_sections == ("industry",)
    assert result.effective_rating is RecommendationRating.NO_RATING
    assert GateReasonCode.UNRESOLVED_CONTRADICTION in result.reason_codes


@pytest.mark.parametrize("verdict", [ReviewVerdict.REVISE, ReviewVerdict.BLOCK])
def test_non_pass_reviewer_cannot_publish_rating(verdict):
    result = evaluate_quality_gates(request(reviewer_verdict=verdict))
    assert result.publication_verdict is PublicationVerdict.DRAFT
    assert result.effective_rating is RecommendationRating.NO_RATING
    assert GateReasonCode.REVIEWER_NOT_PASSED in result.reason_codes
    assert result.allowed_sections == ()


def test_editor_cannot_introduce_claims_outside_approved_set():
    result = evaluate_quality_gates(request(editor_claim_ids=("claim-1", "invented")))
    assert GateReasonCode.EDITOR_UNAPPROVED_CLAIM in result.reason_codes
    assert result.publication_verdict is PublicationVerdict.DRAFT


def test_calculated_exhibits_reconcile_exactly_with_decimal_rows():
    good = CalculatedExhibit(
        exhibit_id="position-table",
        normalized_rows=(CalculatedRow(row_id="NVDA", values={"value": Decimal("0.3")}),),
        rendered_rows=(CalculatedRow(row_id="NVDA", values={"value": Decimal("0.30")}),),
    )
    bad = good.model_copy(update={
        "rendered_rows": (CalculatedRow(row_id="NVDA", values={"value": Decimal("0.3001")}),)
    })
    assert GateReasonCode.EXHIBIT_MISMATCH not in evaluate_quality_gates(
        request(exhibits=(good,))
    ).reason_codes
    assert GateReasonCode.EXHIBIT_MISMATCH in evaluate_quality_gates(
        request(exhibits=(bad,))
    ).reason_codes
    with pytest.raises(ValidationError):
        CalculatedRow(row_id="bad", values={"value": 0.3})


@pytest.mark.parametrize(
    "changes",
    [
        {"asset_kind": "private", "public": False, "tradable": False},
        {"asset_kind": "etf"},
        {"asset_kind": "cash"},
        {"asset_kind": "nontradable", "tradable": False},
        {"resolved": False},
    ],
)
def test_ineligible_security_is_no_rating_and_no_sizing(changes):
    result = evaluate_quality_gates(request(security=security(**changes)))
    assert result.effective_rating is RecommendationRating.NO_RATING
    assert result.allow_sizing is False
    assert GateReasonCode.SECURITY_INELIGIBLE in result.reason_codes


def test_local_only_requires_exact_disclosure_and_confidence():
    missing = evaluate_quality_gates(request(inference_disclosure=None))
    weak = evaluate_quality_gates(request(inference_disclosure=disclosure(
        mode=InferenceMode.LOCAL_ONLY, provider="ollama", model="llama-test",
        confidence=Decimal("0.49"),
    )))
    strong = evaluate_quality_gates(request(inference_disclosure=disclosure(
        mode=InferenceMode.LOCAL_ONLY, provider="ollama", model="llama-test",
    )))
    assert GateReasonCode.INFERENCE_DISCLOSURE_MISSING in missing.reason_codes
    assert GateReasonCode.LOCAL_CONFIDENCE_WEAK in weak.reason_codes
    assert weak.effective_rating is RecommendationRating.NO_RATING
    assert weak.allow_sizing is False
    assert strong.effective_rating is RecommendationRating.BUY


def test_any_no_rating_result_forbids_position_sizing():
    result = evaluate_quality_gates(request(
        requested_rating=RecommendationRating.NO_RATING
    ))
    assert result.effective_rating is RecommendationRating.NO_RATING
    assert result.allow_sizing is False


def test_gate_mapping_inputs_are_defensively_immutable():
    row_values = {"value": Decimal("1")}
    row = CalculatedRow(row_id="row", values=row_values)
    overrides = {"sec_filing": 10}
    policy = QualityGatePolicy(event_window_days_by_source=overrides)
    row_values["value"] = Decimal("2")
    overrides["sec_filing"] = 1
    assert row.values["value"] == Decimal("1")
    assert policy.event_window_days_by_source["sec_filing"] == 10
    with pytest.raises(TypeError):
        row.values["value"] = Decimal("3")
    with pytest.raises(TypeError):
        policy.event_window_days_by_source["sec_filing"] = 2


def test_external_mode_also_requires_provider_and_model_disclosure():
    with pytest.raises(ValidationError):
        disclosure(provider="")


def test_future_and_naive_timestamps_are_rejected():
    with pytest.raises(ValidationError):
        request(as_of=AS_OF.replace(tzinfo=None))
    with pytest.raises(ValidationError):
        request(portfolio_snapshot_at=AS_OF + timedelta(seconds=1))
    with pytest.raises(ValidationError):
        request(price=request().price.model_copy(
            update={"observed_at": AS_OF + timedelta(seconds=1)}
        ))
    with pytest.raises(ValidationError):
        MarketPrice(
            exchange="NASDAQ", session_date=date(2026, 8, 24),
            observed_at=AS_OF.replace(tzinfo=None), value=Decimal("1"), currency="USD",
        )


def test_filing_unavailable_or_future_is_no_rating():
    unavailable = evaluate_quality_gates(request(filing=FilingAvailability(
        filing_due=True, available_as_of=False
    )))
    assert GateReasonCode.REQUIRED_FILING_UNAVAILABLE in unavailable.reason_codes
    with pytest.raises(ValidationError):
        request(filing=FilingAvailability(
            filing_due=True, available_as_of=True,
            filed_at=AS_OF + timedelta(seconds=1),
        ))


def test_event_window_defaults_to_seven_days_and_supports_source_override():
    recent = request(event_source_type="sec_filing", event_published_at=AS_OF - timedelta(days=7))
    old = recent.model_copy(update={"event_published_at": AS_OF - timedelta(days=8)})
    policy = QualityGatePolicy(event_window_days_by_source={"sec_filing": 10})
    assert evaluate_quality_gates(recent).allow_event_report is True
    assert evaluate_quality_gates(old).allow_event_report is False
    assert evaluate_quality_gates(old, policy=policy).allow_event_report is True


@pytest.mark.parametrize(
    "source_type,published_at",
    [(None, None), ("sec_filing", None), (None, AS_OF - timedelta(days=1))],
)
def test_event_report_missing_or_partial_freshness_fails_closed(
    source_type, published_at
):
    result = evaluate_quality_gates(request(
        publication_kind=PublicationKind.EVENT_REPORT,
        requested_rating=RecommendationRating.NO_RATING,
        event_source_type=source_type,
        event_published_at=published_at,
    ))
    assert result.publication_verdict is PublicationVerdict.DRAFT
    assert result.allow_event_report is False
    assert GateReasonCode.EVENT_FRESHNESS_UNKNOWN in result.reason_codes


def test_future_event_timestamp_is_rejected_and_window_boundary_is_exact():
    with pytest.raises(ValidationError, match="UTC|as_of"):
        request(
            publication_kind=PublicationKind.EVENT_REPORT,
            event_source_type="news",
            event_published_at=AS_OF + timedelta(microseconds=1),
        )
    exact = request(
        publication_kind=PublicationKind.EVENT_REPORT,
        requested_rating=RecommendationRating.NO_RATING,
        event_source_type="news",
        event_published_at=AS_OF - timedelta(days=7),
    )
    expired = exact.model_copy(update={
        "event_published_at": AS_OF - timedelta(days=7, microseconds=1)
    })
    assert evaluate_quality_gates(exact).allow_event_report is True
    assert evaluate_quality_gates(expired).allow_event_report is False


def test_event_report_with_stale_portfolio_can_still_be_final():
    result = evaluate_quality_gates(request(
        publication_kind=PublicationKind.EVENT_REPORT,
        requested_rating=RecommendationRating.NO_RATING,
        portfolio_snapshot_at=AS_OF - timedelta(hours=40),
        event_source_type="news",
        event_published_at=AS_OF - timedelta(days=1),
    ))
    assert result.allow_event_report is True
    assert result.allow_sizing is False
    assert result.publication_verdict is PublicationVerdict.FINAL


def test_event_report_with_bad_event_lineage_is_not_allowed():
    result = evaluate_quality_gates(request(
        publication_kind=PublicationKind.EVENT_REPORT,
        requested_rating=RecommendationRating.NO_RATING,
        claims=(claim(section="event", evidence=()),),
        event_source_type="news",
        event_published_at=AS_OF - timedelta(days=1),
    ))
    assert result.allow_event_report is False
    assert result.publication_verdict is PublicationVerdict.DRAFT


def test_deterministic_failures_override_model_rating_and_order_is_stable():
    bad_claim_1 = claim("z", section="recommendation", evidence=())
    bad_claim_2 = claim("a", section="recommendation", evidence=())
    left = evaluate_quality_gates(request(
        claims=(bad_claim_1, bad_claim_2),
        reviewer_approved_claim_ids=("a", "z"), editor_claim_ids=("a", "z"),
    ))
    right = evaluate_quality_gates(request(
        claims=(bad_claim_2, bad_claim_1),
        reviewer_approved_claim_ids=("z", "a"), editor_claim_ids=("z", "a"),
        requested_rating=RecommendationRating.SELL_REDUCE,
    ))
    assert reason_values(left) == tuple(sorted(set(reason_values(left))))
    assert reason_values(left) == reason_values(right)
    assert left.allowed_sections == right.allowed_sections == ()
    assert left.effective_rating is right.effective_rating is RecommendationRating.NO_RATING


@pytest.mark.parametrize(
    "rating",
    [RecommendationRating.BUY, RecommendationRating.HOLD,
     RecommendationRating.SELL_REDUCE],
)
def test_claimless_rated_recommendation_is_draft_no_rating(rating):
    result = evaluate_quality_gates(request(
        requested_rating=rating, claims=(),
        reviewer_approved_claim_ids=(), editor_claim_ids=(),
    ))
    assert GateReasonCode.RESEARCH_CLAIMS_MISSING in result.reason_codes
    assert result.publication_verdict is PublicationVerdict.DRAFT
    assert result.effective_rating is RecommendationRating.NO_RATING
    assert result.allow_sizing is False


def test_claimless_no_rating_research_is_retained_as_draft():
    result = evaluate_quality_gates(request(
        requested_rating=RecommendationRating.NO_RATING, claims=(),
        reviewer_approved_claim_ids=(), editor_claim_ids=(),
    ))
    assert result.publication_verdict is PublicationVerdict.DRAFT
    assert GateReasonCode.RESEARCH_CLAIMS_MISSING in result.reason_codes


def test_matching_fabricated_reviewer_and_editor_claim_ids_fail_closed():
    result = evaluate_quality_gates(request(
        reviewer_approved_claim_ids=("claim-1", "ghost"),
        editor_claim_ids=("claim-1", "ghost"),
    ))
    assert GateReasonCode.APPROVED_CLAIM_UNKNOWN in result.reason_codes
    assert GateReasonCode.EDITOR_CLAIM_UNKNOWN in result.reason_codes
    assert result.publication_verdict is PublicationVerdict.DRAFT
    assert result.effective_rating is RecommendationRating.NO_RATING


@pytest.mark.parametrize("offset", [timedelta(hours=-5), timedelta(minutes=17)])
def test_all_quality_datetimes_require_exact_utc(offset):
    hostile = timezone(offset)
    with pytest.raises(ValidationError, match="datetime must be UTC"):
        request(as_of=AS_OF.astimezone(hostile))
    with pytest.raises(ValidationError, match="datetime must be UTC"):
        request(portfolio_snapshot_at=(AS_OF - timedelta(hours=1)).astimezone(hostile))
    with pytest.raises(ValidationError, match="datetime must be UTC"):
        MarketPrice(
            exchange="NASDAQ", session_date=date(2026, 8, 24),
            observed_at=AS_OF.astimezone(hostile), value=Decimal("1"), currency="USD",
        )
    with pytest.raises(ValidationError, match="datetime must be UTC"):
        FilingAvailability(
            filing_due=True, available_as_of=True,
            filed_at=(AS_OF - timedelta(days=1)).astimezone(hostile),
        )
    with pytest.raises(ValidationError, match="datetime must be UTC"):
        request(
            event_source_type="news",
            event_published_at=(AS_OF - timedelta(days=1)).astimezone(hostile),
        )


def test_source_identifiers_are_nfc_normalized_and_family_defines_independence():
    composed = "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    decomposed = "cafe\N{COMBINING ACUTE ACCENT}"
    first = source("doc-1", primary=False, source_type="news", source_family=composed)
    second_same_origin = source(
        "doc-2", primary=False, source_type="NEWS", source_family=decomposed.upper()
    )
    same = evaluate_quality_gates(request(
        claims=(claim(evidence=(first, second_same_origin)),),
    ))
    distinct = evaluate_quality_gates(request(claims=(claim(evidence=(
        first,
        source("doc-3", primary=False, source_type="news", source_family="wire-2"),
    )),)))
    assert first.source_family == second_same_origin.source_family
    assert second_same_origin.source_type == "news"
    assert EvidenceReference(
        evidence_id="evidence",
        canonical_source_id=decomposed,
        source_family="family",
        source_type="news",
        stored=True,
        primary=False,
    ).canonical_source_id == composed
    assert GateReasonCode.INSUFFICIENT_CORROBORATION in same.reason_codes
    assert GateReasonCode.INSUFFICIENT_CORROBORATION not in distinct.reason_codes
    assert SourceReference is EvidenceReference


def test_primary_source_is_sufficient_regardless_of_family_count():
    result = evaluate_quality_gates(request(claims=(claim(evidence=(source(
        "primary", primary=True, source_family="issuer"
    ),)),)))
    assert GateReasonCode.INSUFFICIENT_CORROBORATION not in result.reason_codes


def test_unsupported_industry_claim_isolated_from_valid_recommendation():
    unsupported = claim("claim-industry", section="industry", evidence=())
    result = evaluate_quality_gates(request(
        claims=(claim(), unsupported),
        reviewer_approved_claim_ids=("claim-1", "claim-industry"),
        editor_claim_ids=("claim-1",),
    ))
    assert result.allowed_sections == ("recommendation",)
    assert result.publication_verdict is PublicationVerdict.FINAL
    assert result.effective_rating is RecommendationRating.BUY
    assert result.allow_sizing is True


def test_industry_contradiction_isolated_from_valid_recommendation():
    industry = claim("claim-industry", section="industry")
    result = evaluate_quality_gates(request(
        claims=(claim(), industry),
        contradictory_claim_ids=("claim-industry",),
        reviewer_approved_claim_ids=("claim-1", "claim-industry"),
        editor_claim_ids=("claim-1",),
    ))
    assert result.allowed_sections == ("recommendation",)
    assert result.publication_verdict is PublicationVerdict.FINAL
    assert result.effective_rating is RecommendationRating.BUY


def test_zero_admissible_claims_cannot_support_a_rating():
    empty = claim(material=False, quantitative=False, consequential=False, evidence=())
    result = evaluate_quality_gates(request(claims=(empty,)))
    assert GateReasonCode.RESEARCH_CLAIMS_MISSING in result.reason_codes
    assert result.publication_verdict is PublicationVerdict.DRAFT
    assert result.effective_rating is RecommendationRating.NO_RATING


def test_mismatched_calculated_exhibit_blocks_final_release():
    exhibit = CalculatedExhibit(
        exhibit_id="valuation",
        normalized_rows=(CalculatedRow(row_id="x", values={"v": Decimal("1")}),),
        rendered_rows=(CalculatedRow(row_id="x", values={"v": Decimal("2")}),),
    )
    result = evaluate_quality_gates(request(exhibits=(exhibit,)))
    assert GateReasonCode.EXHIBIT_MISMATCH in result.reason_codes
    assert result.publication_verdict is PublicationVerdict.DRAFT
    assert result.effective_rating is RecommendationRating.NO_RATING


@pytest.mark.parametrize(
    "kind",
    [PublicationKind.FOUNDATIONAL_REPORT, PublicationKind.PORTFOLIO_BRIEF],
)
def test_foundational_and_portfolio_artifacts_require_all_sections(kind):
    unsupported = claim("claim-industry", section="industry", evidence=())
    result = evaluate_quality_gates(request(
        publication_kind=kind,
        claims=(claim(), unsupported),
        reviewer_approved_claim_ids=("claim-1", "claim-industry"),
        editor_claim_ids=("claim-1",),
    ))
    assert result.publication_verdict is PublicationVerdict.DRAFT
    assert result.allowed_sections == ("recommendation",)


def test_event_report_can_publish_supported_event_section_without_bad_industry():
    event_claim = claim("claim-event", section="event")
    unsupported = claim("claim-industry", section="industry", evidence=())
    result = evaluate_quality_gates(request(
        publication_kind=PublicationKind.EVENT_REPORT,
        requested_rating=RecommendationRating.NO_RATING,
        claims=(event_claim, unsupported),
        reviewer_approved_claim_ids=("claim-event", "claim-industry"),
        editor_claim_ids=("claim-event",),
        event_source_type="news",
        event_published_at=AS_OF - timedelta(days=1),
    ))
    assert result.allowed_sections == ("event",)
    assert result.allow_event_report is True
    assert result.publication_verdict is PublicationVerdict.FINAL


def test_documented_gate_wrappers_delegate_to_one_engine():
    context = request()
    direct = evaluate_quality_gates(context)
    assert QualityGate().evaluate(context) == direct
    recommendation = RecommendationGate().evaluate(context)
    assert recommendation == direct
    assert recommendation.rating is RecommendationRating.BUY


def test_price_is_fresh_public_api_supports_default_and_override_calendars():
    assert price_is_fresh(
        date(2026, 8, 21), date(2026, 8, 24), exchange="NASDAQ"
    ) is True
    assert price_is_fresh(
        date(2026, 8, 23), date(2026, 8, 24), exchange="NASDAQ"
    ) is False
    override = ExchangeCalendar(
        exchange="ALWAYS", timezone_name="UTC", close_time=time(0),
        weekend_days=(), holidays=(),
    )
    assert price_is_fresh(
        date(2026, 8, 23), date(2026, 8, 24),
        exchange="ALWAYS", calendar=override,
    ) is True


@pytest.mark.parametrize(
    "kind",
    [PublicationKind.RECOMMENDATION, PublicationKind.FOUNDATIONAL_REPORT,
     PublicationKind.PORTFOLIO_BRIEF],
)
def test_foundational_outputs_require_reviewer_pass(kind):
    result = evaluate_quality_gates(request(
        publication_kind=kind, reviewer_verdict=ReviewVerdict.REVISE
    ))
    assert result.publication_verdict is PublicationVerdict.DRAFT
