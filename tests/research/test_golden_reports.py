import json
import shutil
import subprocess

import pytest

from news_bot.research.agents.contracts import AgentContractError
from news_bot.research.models import AgentRole, RecommendationRating, ReviewVerdict
from tests.research import golden_harness

from tests.research.golden_harness import (
    DeterministicGoldenProvider,
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


def test_golden_pipeline_runs_specialists_with_audited_usage(tmp_path):
    golden_run = build_golden_run(tmp_path)
    assert tuple(audit.role for audit in golden_run.agent_audits) == (
        AgentRole.EVIDENCE_ANALYST,
        AgentRole.EVIDENCE_ANALYST,
        AgentRole.FUNDAMENTAL_ANALYST,
        AgentRole.SKEPTICAL_REVIEWER,
        AgentRole.RESEARCH_EDITOR,
    )
    assert all(audit.prompt_hash and audit.evidence_hash for audit in golden_run.agent_audits)
    assert all(audit.provider_attempt_count == 1 for audit in golden_run.agent_audits)
    assert all(
        audit.reservation_state == "reconciled" for audit in golden_run.agent_audits
    )
    assert golden_run.reviewer.verdict is ReviewVerdict.PASS
    assert golden_run.fundamental.rating is RecommendationRating.HOLD
    assert golden_run.paid_cost > 0
    assert golden_run.paid_cost == sum(
        audit.reserved_cost_usd for audit in golden_run.agent_audits
    )
    assert len({audit.input_tokens for audit in golden_run.agent_audits}) > 1


def test_golden_report_renders_real_reconciled_exhibit_and_pdf(tmp_path):
    artifact = build_golden_run(tmp_path, render=True).artifact
    assert artifact is not None
    assert "Rounded exposure summary" in artifact.html
    assert "100.0%" in artifact.html
    assert artifact.pdf_path is not None
    payload = artifact.pdf_path.read_bytes()
    assert len(payload) > 10_000
    assert payload.startswith(b"%PDF-")
    assert payload.rstrip().endswith(b"%%EOF")
    assert "<table>" in artifact.html
    if shutil.which("pdfinfo"):
        completed = subprocess.run(
            ["pdfinfo", str(artifact.pdf_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "Pages:" in completed.stdout
    if shutil.which("pdftotext"):
        extracted = subprocess.run(
            ["pdftotext", str(artifact.pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert "Synthetic NVIDIA exposure" in extracted
        assert "100.0%" in extracted


def test_provider_rejects_agent_request_with_fixture_evidence_omitted(
    tmp_path,
    monkeypatch,
):
    original = DeterministicGoldenProvider._payload

    def omit_capacity(self, request):
        evidence = json.loads(request.canonical_evidence)
        evidence["passages"] = [
            passage
            for passage in evidence["passages"]
            if passage["passage_id"] != "passage-tsmc-capacity"
        ]
        object.__setattr__(request, "canonical_evidence", json.dumps(evidence))
        return original(self, request)

    monkeypatch.setattr(DeterministicGoldenProvider, "_payload", omit_capacity)
    with pytest.raises(AssertionError, match="omitted canonical fixture evidence"):
        build_golden_run(tmp_path)


def test_editor_rejects_provider_output_with_unapproved_claim(tmp_path, monkeypatch):
    original = DeterministicGoldenProvider._payload

    def introduce_unapproved(self, request):
        payload = original(self, request)
        if request.role is AgentRole.RESEARCH_EDITOR:
            payload["sections"][0]["approved_claim_ids"] = ["claim_unapproved"]
        return payload

    monkeypatch.setattr(
        DeterministicGoldenProvider, "_payload", introduce_unapproved
    )
    with pytest.raises(AgentContractError, match="contract validation"):
        build_golden_run(tmp_path)


def test_report_claim_binding_rejects_unsupported_section_prose(tmp_path):
    golden_run = build_golden_run(tmp_path)
    unsupported = golden_run.report.thesis.model_copy(
        update={"body": golden_run.report.thesis.body + " Unsupported prose."}
    )
    report = golden_run.report.model_copy(update={"thesis": unsupported})
    with pytest.raises(AssertionError, match="approved claims"):
        golden_harness.assert_report_claim_binding(
            report, golden_run.store, golden_run.editor
        )


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
