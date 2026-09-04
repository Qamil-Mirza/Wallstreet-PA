"""Bounded, evidence-only execution shared by all specialist roles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime
from importlib import resources
from typing import Generic, TypeVar

from pydantic import ValidationError

from ..models import AgentRole, InferenceMode
from ..providers.base import (
    FallbackPolicy,
    ModelRequest,
    ReasoningEffort,
    ValidationIssue,
    ProviderAttemptTrace,
    ProviderError,
    provider_failure_code,
    validation_issues,
)
from ..store import (
    AgentEvidencePacket,
    AgentExecutionConflict,
    ProviderAttemptAudit,
    ResearchStore,
)
from .contracts import (
    AgentContractError,
    AgentTask,
    AnalyticalOutput,
    EvidenceInput,
    EvidenceUnavailable,
    IneligibleSecurity,
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

    def _validate_evidence_packet(
        self, task_input: InputT, packet: AgentEvidencePacket
    ) -> None:
        """Apply role-specific checks after the store validates every reference."""

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
                    "published_at": passage.published_at.isoformat(),
                    "retrieved_at": passage.retrieved_at.isoformat(),
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
            canonical_task = json.dumps(
                task.model_dump(mode="json", warnings="error"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            validated_task = AgentTask[self.input_type].model_validate_json(
                canonical_task
            )
        except Exception:
            raise AgentContractError("agent task failed contract validation") from None
        started_at = self.clock()
        if (
            not isinstance(started_at, datetime)
            or started_at.tzinfo is None
            or started_at.utcoffset() is None
        ):
            raise AgentContractError("agent clock must return an aware datetime")
        try:
            canonical_input = json.dumps(
                validated_task.input.model_dump(mode="json", warnings="error"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            execution_claim = self.store.claim_agent_execution(
                workflow_run_id=validated_task.run_id,
                task_id=validated_task.task_id,
                role=self.role,
                schema_version=validated_task.schema_version,
                input_hash=hashlib.sha256(
                    canonical_input.encode("utf-8")
                ).hexdigest(),
                prompt_hash=self.prompt_hash,
                evidence_hash=None,
                started_at=started_at,
            )
        except AgentExecutionConflict as error:
            raise AgentContractError(str(error)) from None

        attempt_id = execution_claim.attempt_id
        lease_token = execution_claim.lease_token
        provider_called = False
        try:
            self._preflight(validated_task.input)
            packet = self.store.load_agent_evidence(
                validated_task.input.evidence_ids,
                validated_task.input.claim_ids,
                as_of=validated_task.input.as_of,
            )
            if packet is None:
                raise EvidenceUnavailable("agent evidence is unavailable")
            self._validate_evidence_packet(validated_task.input, packet)
            request = self._request(validated_task, packet)
            if execution_claim.replay_output_json is not None:
                if request.evidence_hash != execution_claim.replay_evidence_hash:
                    raise AgentContractError(
                        "terminal agent replay evidence hash conflicts"
                    )
                try:
                    output = self.output_type.model_validate_json(
                        execution_claim.replay_output_json
                    )
                    self._validate_semantics(validated_task.input, output)
                except (AgentContractError, ValidationError, ValueError):
                    raise AgentContractError(
                        "terminal agent replay failed current contract validation"
                    ) from None
                return output
            if lease_token is None:
                raise AgentContractError("agent execution lease is unavailable")
            self.store.set_agent_execution_evidence_hash(
                attempt_id, request.evidence_hash, lease_token=lease_token
            )

            def record_attempt(trace: ProviderAttemptTrace) -> None:
                self.store.record_provider_attempt(
                    attempt_id,
                    ProviderAttemptAudit(
                        status=trace.status, provider=trace.provider,
                        model=trace.model, latency_ms=trace.latency_ms,
                        input_tokens=trace.input_tokens,
                        output_tokens=trace.output_tokens,
                        reasoning_tokens=trace.reasoning_tokens,
                        inference_mode=trace.inference_mode,
                        recorded_at=trace.recorded_at,
                        fallback_reason=trace.fallback_reason,
                        response_hash=trace.response_hash,
                        failure_code=trace.failure_code,
                        usage_known=trace.usage_known,
                        reservation_id=trace.reservation_id,
                        reservation_state=trace.reservation_state,
                        reserved_cost_usd=trace.reserved_cost_usd,
                    ),
                    lease_token=lease_token,
                )

            request = replace(
                request, attempt_id=attempt_id, attempt_recorder=record_attempt
            )
            provider_called = True
            response = self.router.generate(request)
            if self.store.provider_attempt_count(attempt_id) == 0:
                self.store.record_provider_attempt(
                    attempt_id,
                    ProviderAttemptAudit(
                        status="succeeded", provider=response.provider,
                        model=response.model,
                        latency_ms=getattr(response, "latency_ms", 0),
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        reasoning_tokens=response.reasoning_tokens,
                        inference_mode=response.inference_mode,
                        recorded_at=self._completed_at(started_at),
                        fallback_reason=response.fallback_reason,
                        response_hash=response.raw_response_hash,
                    ),
                    lease_token=lease_token,
                )
            try:
                output = self._validate_response(validated_task.input, response)
            except (AgentContractError, ValidationError):
                raise AgentContractError(
                    "agent output failed contract validation"
                ) from None
            completed_at = self._completed_at(started_at)
            canonical_output = json.dumps(
                output.model_dump(mode="json"), ensure_ascii=False, allow_nan=False,
                sort_keys=True, separators=(",", ":"),
            )
            self.store.finalize_agent_execution(
                attempt_id, lease_token=lease_token, succeeded=True,
                completed_at=completed_at,
                output_hash=hashlib.sha256(
                    canonical_output.encode("utf-8")
                ).hexdigest(),
                output_json=canonical_output,
            )
            return output
        except Exception as error:
            if lease_token is None:
                raise
            if provider_called and self.store.provider_attempt_count(attempt_id) == 0:
                self.store.record_provider_attempt(
                    attempt_id,
                    ProviderAttemptAudit(
                        status="failed", provider="router", model="unknown",
                        latency_ms=0, input_tokens=0, output_tokens=0,
                        reasoning_tokens=0,
                        inference_mode=InferenceMode.EXTERNAL,
                        recorded_at=self._completed_at(started_at),
                        failure_code=(
                            provider_failure_code(error)
                            if isinstance(error, ProviderError)
                            else "provider_error"
                        ),
                    ),
                    lease_token=lease_token,
                )
            self.store.finalize_agent_execution(
                attempt_id, lease_token=lease_token, succeeded=False,
                completed_at=self._completed_at(started_at),
                failure_code=self._failure_code(error),
            )
            raise

    def _completed_at(self, started_at: datetime) -> datetime:
        completed_at = self.clock()
        if (
            not isinstance(completed_at, datetime)
            or completed_at.tzinfo is None
            or completed_at.utcoffset() is None
            or completed_at < started_at
        ):
            return started_at
        return completed_at

    @staticmethod
    def _failure_code(error: Exception) -> str:
        if isinstance(error, ProviderError):
            return provider_failure_code(error)
        if isinstance(error, EvidenceUnavailable):
            return "evidence_unavailable"
        if isinstance(error, IneligibleSecurity):
            return "ineligible_security"
        if isinstance(error, AgentContractError | ValidationError):
            return "agent_contract"
        return "agent_execution"
