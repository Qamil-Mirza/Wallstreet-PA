"""Research editor role constrained to approved claims."""

from ..models import AgentRole
from .base import BoundedAgent
from .contracts import AgentContractError, EditorInput, ResearchEditorOutput


class ResearchEditor(BoundedAgent[EditorInput, ResearchEditorOutput]):
    role = AgentRole.RESEARCH_EDITOR
    prompt_name = "research_editor"
    input_type = EditorInput
    output_type = ResearchEditorOutput

    def _preflight(self, task_input) -> None:
        if not set(task_input.approved_claim_ids) <= set(task_input.claim_ids):
            raise AgentContractError("editor approved claims are not loaded")

    def _validate_semantics(self, task_input, output) -> None:
        super()._validate_semantics(task_input, output)
        approved = set(task_input.approved_claim_ids)
        if any(
            not set(section.approved_claim_ids) <= approved
            for section in output.sections
        ):
            raise AgentContractError("editor introduced an unapproved claim")
