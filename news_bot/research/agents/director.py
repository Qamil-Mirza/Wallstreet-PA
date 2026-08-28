"""Deterministic research-director role."""

import hashlib
import json

from ..models import AgentRole
from .base import BoundedAgent
from .contracts import AgentContractError, DirectorInput, DirectorOutput


class ResearchDirector(BoundedAgent[DirectorInput, DirectorOutput]):
    role = AgentRole.RESEARCH_DIRECTOR
    prompt_name = "director"
    input_type = DirectorInput
    output_type = DirectorOutput

    def _validate_response(self, task_input, response):
        output = super()._validate_response(task_input, response)
        tasks = []
        for item in output.tasks:
            canonical = json.dumps(
                {
                    "evidence_ids": item.evidence_ids,
                    "kind": item.kind,
                    "priority": item.priority,
                    "question": item.question,
                    "schema_version": "1",
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            identifier = "research_task_" + hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()
            tasks.append(item.model_copy(update={"task_id": identifier}))
        if len({item.task_id for item in tasks}) != len(tasks):
            raise AgentContractError("director produced duplicate task content")
        return output.model_copy(
            update={"tasks": tuple(sorted(tasks, key=lambda item: item.task_id))}
        )

    def _validate_semantics(self, task_input, output) -> None:
        super()._validate_semantics(task_input, output)
        available = set(task_input.evidence_ids)
        if any(not set(item.evidence_ids) <= available for item in output.tasks):
            raise AgentContractError("directed task cites unavailable evidence")
