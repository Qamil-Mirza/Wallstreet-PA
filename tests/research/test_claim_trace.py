from news_bot.research.models import ClaimKind

from tests.research.golden_harness import build_golden_run


def test_every_material_report_claim_resolves_to_passage(tmp_path):
    golden_run = build_golden_run(tmp_path)
    editor_claim_ids = {
        claim_id
        for section in golden_run.editor.sections
        for claim_id in section.approved_claim_ids
    }
    assert {claim.claim_id for claim in golden_run.material_claims} == editor_claim_ids
    for claim in golden_run.material_claims:
        lineage = golden_run.store.list_claim_lineage(claim.claim_id)
        assert lineage
        assert {item.passage.passage_id for item in lineage} == set(
            claim.evidence_ids
        )
        assert all(item.stance == "supports" for item in lineage)

    assert all(claim.text in str(golden_run.report) for claim in golden_run.material_claims)


def test_inferences_disclose_kind_and_retain_claim_dependencies(tmp_path):
    golden_run = build_golden_run(tmp_path)
    inference_claims = tuple(
        claim
        for claim in golden_run.material_claims
        if claim.kind is ClaimKind.INFERENCE
    )
    assert inference_claims
    assert all(
        golden_run.store.list_supporting_claim_ids(claim.claim_id)
        for claim in inference_claims
    )
