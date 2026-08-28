"""Skeptical evidence reviewer role."""

from ..models import AgentRole
from .base import BoundedAgent
from .contracts import AgentContractError, ReviewerInput, ReviewerOutput


class SkepticalReviewer(BoundedAgent[ReviewerInput, ReviewerOutput]):
    role = AgentRole.SKEPTICAL_REVIEWER
    prompt_name = "skeptical_reviewer"
    input_type = ReviewerInput
    output_type = ReviewerOutput

    def _validate_evidence_packet(self, task_input, packet) -> None:
        loaded = {claim.claim_id for claim in packet.claims if claim.status == "active"}
        if not set(task_input.target_claim_ids) <= loaded:
            raise AgentContractError("review target evidence is unavailable")

    def _validate_semantics(self, task_input, output) -> None:
        super()._validate_semantics(task_input, output)
        evidence = set(task_input.evidence_ids)
        if any(not set(issue.evidence_ids) <= evidence for issue in output.issues):
            raise AgentContractError("review issue cites unavailable evidence")
        targets = set(task_input.target_claim_ids)
        if any(not set(issue.target_claim_ids) <= targets for issue in output.issues):
            raise AgentContractError("review issue cites a claim outside its target set")
