"""Skeptical evidence reviewer role."""

from ..models import AgentRole
from .base import BoundedAgent
from .contracts import AgentContractError, ReviewerInput, ReviewerOutput


class SkepticalReviewer(BoundedAgent[ReviewerInput, ReviewerOutput]):
    role = AgentRole.SKEPTICAL_REVIEWER
    prompt_name = "skeptical_reviewer"
    input_type = ReviewerInput
    output_type = ReviewerOutput

    def _validate_semantics(self, task_input, output) -> None:
        super()._validate_semantics(task_input, output)
        evidence = set(task_input.evidence_ids)
        if any(not set(issue.evidence_ids) <= evidence for issue in output.issues):
            raise AgentContractError("review issue cites unavailable evidence")

