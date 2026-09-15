import json

from tests.research.golden_harness import (
    SOURCE_PACKET,
    build_golden_run,
    load_expected_characteristics,
    run_golden_evaluation,
)


def test_golden_report_meets_expected_quality_characteristics(tmp_path):
    evaluation = run_golden_evaluation(tmp_path)
    assert evaluation.passes
    assert evaluation.failures == ()
    assert evaluation.scores == evaluation.minimum_scores


def test_golden_packet_is_explicitly_redistributable_and_test_authored():
    manifest = json.loads(
        (SOURCE_PACKET / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["created_for_test"] is True
    assert manifest["redistribution_license"] == "CC0-1.0"
    assert "paraphrases" in manifest["provenance"]
    for filename in manifest["documents"]:
        document = json.loads(
            (SOURCE_PACKET / filename).read_text(encoding="utf-8")
        )
        assert document["created_for_test"] is True
        assert document["redistribution_license"] == "CC0-1.0"
        assert "not copied" in document["provenance"]


def test_golden_report_contains_no_prohibited_future_fact(tmp_path):
    golden_run = build_golden_run(tmp_path)
    expected = load_expected_characteristics()
    report_text = json.dumps(golden_run.report.model_dump(mode="json")).casefold()
    assert all(
        fact.casefold() not in report_text
        for fact in expected["prohibited_future_facts"]
    )
