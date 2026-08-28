"""Bounded, evidence-only execution shared by all specialist roles."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from importlib import resources
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ..models import AgentRole
from ..providers.base import (
    FallbackPolicy,
    ModelRequest,
    ReasoningEffort,
    ValidationIssue,
    validation_issues,
)
from ..store import AgentEvidencePacket, AgentRunAudit, ResearchStore
from .contracts import (
    AgentContractError,
    AgentTask,
    AnalyticalOutput,
    EvidenceInput,
    EvidenceUnavailable,
)


InputT = TypeVar("InputT", bound=EvidenceInput)
OutputT = TypeVar("OutputT", bound=AnalyticalOutput)


class BoundedAgent(Generic[InputT, OutputT]):
    """One logical role with no chat, tools, orders, or mutable prompt state."""

    role: AgentRole
    prompt_name: str
    input_type: type[InputT]
    output_type: type[OutputT]
    max_output_tokens = 1600
    reasoning_effort = ReasoningEffort.MEDIUM
    fallback_policy = FallbackPolicy.OUTAGES

    def __init__(self, router: object, store: ResearchStore, *, clock) -> None:
        if not hasattr(router, "generate"):
            raise TypeError("router must provide generate")
        if not isinstance(store, ResearchStore):
            raise TypeError("store must be ResearchStore")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.router = router
        self.store = store
        self.clock = clock
        self.system_prompt = self._load_prompt()
        self.prompt_hash = hashlib.sha256(
            self.system_prompt.encode("utf-8")
        ).hexdigest()

    def _load_prompt(self) -> str:
        try:
            prompt = (
                resources.files("news_bot.research")
                .joinpath("prompts", f"{self.prompt_name}.md")
                .read_text(encoding="utf-8")
            )
        except (FileNotFoundError, OSError, UnicodeError):
            raise AgentContractError("versioned agent prompt is unavailable") from None
        if "Prompt-Version: 1" not in prompt or "Schema-Version: 1" not in prompt:
            raise AgentContractError("versioned agent prompt is invalid")
        return prompt

    def _preflight(self, task_input: InputT) -> None:
        """Role-specific deterministic checks that run before model inference."""

    @staticmethod
    def _evidence_json(packet: AgentEvidencePacket) -> dict[str, object]:
        return {
            "claims": [
                {
                    "as_of": claim.as_of.isoformat(),
                    "claim_id": claim.claim_id,
                    "confidence": format(claim.confidence, "f"),
                    "kind": claim.kind.value,
                    "status": claim.status,
                    "text": claim.text,
                }
                for claim in packet.claims
            ],
            "passages": [
                {
                    "content_hash": passage.content_hash,
                    "document_id": passage.document_id,
                    "end_offset": passage.end_offset,
                    "ordinal": passage.ordinal,
                    "passage_id": passage.passage_id,
                    "start_offset": passage.start_offset,
                    "text": passage.text,
                }
                for passage in packet.passages
            ],
            "trust_boundary": "UNTRUSTED_EVIDENCE_DATA",
        }

    def _request(
        self, task: AgentTask[InputT], packet: AgentEvidencePacket
    ) -> ModelRequest:
        input_payload = task.input.model_dump(mode="json")
        evidence = self._evidence_json(packet)
        evidence["task_input"] = input_payload
        return ModelRequest(
            role=self.role,
            system_prompt=self.system_prompt,
            evidence_packet=evidence,
            output_schema=self.output_type,
            max_output_tokens=self.max_output_tokens,
            reasoning_effort=self.reasoning_effort,
            fallback_policy=self.fallback_policy,
            run_id=task.run_id,
        )

    def _validate_semantics(self, task_input: InputT, output: OutputT) -> None:
        if not set(output.evidence_ids) <= set(task_input.evidence_ids):
            raise AgentContractError("agent output cites evidence outside its packet")
        if output.as_of != task_input.as_of:
            raise AgentContractError("agent output as_of does not match its task")

    def _validate_response(
        self, task_input: InputT, response: object
    ) -> OutputT:
        try:
            output = self.output_type.model_validate(response.data)
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            if isinstance(error, ValidationError):
                raise error
            raise AgentContractError("agent output failed contract validation") from None
        if output.inference_mode is not response.inference_mode:
            raise AgentContractError("agent output inference mode does not match routing")
        self._validate_semantics(task_input, output)
        return output

    @staticmethod
    def _repair_issues(error: Exception) -> tuple[ValidationIssue, ...]:
        return validation_issues(error)

    def run(self, task: AgentTask[InputT]) -> OutputT:
        try:
            validated_task = AgentTask[self.input_type].model_validate(task)
        except (TypeError, ValueError, ValidationError):
            raise AgentContractError("agent task failed contract validation") from None
        started_at = self.clock()
        if (
            not isinstance(started_at, datetime)
            or started_at.tzinfo is None
            or started_at.utcoffset() is None
        ):
            raise AgentContractError("agent clock must return an aware datetime")
        self._preflight(validated_task.input)
        packet = self.store.load_agent_evidence(
            validated_task.input.evidence_ids, validated_task.input.claim_ids
        )
        if packet is None:
            raise EvidenceUnavailable("agent evidence is unavailable")
        request = self._request(validated_task, packet)
        response = self.router.generate(request)
        try:
            output = self._validate_response(validated_task.input, response)
        except (AgentContractError, ValidationError) as first_error:
            repair = request.for_validation_retry(self._repair_issues(first_error))
            response = self.router.generate(repair)
            try:
                output = self._validate_response(validated_task.input, response)
            except (AgentContractError, ValidationError):
                raise AgentContractError(
                    "agent output failed contract validation after one repair"
                ) from None
        completed_at = self.clock()
        if (
            not isinstance(completed_at, datetime)
            or completed_at.tzinfo is None
            or completed_at.utcoffset() is None
            or completed_at < started_at
        ):
            raise AgentContractError("agent completion time is invalid")
        canonical_output = json.dumps(
            output.model_dump(mode="json"), ensure_ascii=False, allow_nan=False,
            sort_keys=True, separators=(",", ":"),
        )
        self.store.record_agent_run_audit(AgentRunAudit(
            run_id=validated_task.run_id, task_id=validated_task.task_id,
            role=self.role, started_at=started_at, completed_at=completed_at,
            provider=response.provider, model=response.model,
            inference_mode=response.inference_mode, prompt_hash=self.prompt_hash,
            evidence_hash=request.evidence_hash,
            output_hash=hashlib.sha256(canonical_output.encode("utf-8")).hexdigest(),
            input_tokens=response.input_tokens, output_tokens=response.output_tokens,
            reasoning_tokens=response.reasoning_tokens,
            fallback_reason=response.fallback_reason,
        ))
        return output

