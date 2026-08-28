"""Immutable provider contracts and redacted failure taxonomy."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from ..models import AgentRole, InferenceMode


class ProviderError(Exception):
    """Base class for redacted inference-provider failures."""


class ProviderConfigurationError(ProviderError, ValueError):
    """Raised when a provider cannot be configured safely."""


class ProviderAuthenticationError(ProviderError):
    """Raised when external credentials are rejected."""


class ProviderRateLimitError(ProviderError):
    """Raised when a provider rejects a request due to quota or rate."""


class ProviderUnavailable(ProviderError):
    """Raised for timeouts and retryable provider outages."""


class ProviderValidationError(ProviderError):
    """Raised when provider JSON violates the requested contract."""


class ProviderNonRetryableError(ProviderError):
    """Raised for a redacted provider rejection that must not be retried."""


class TaskDeferred(ProviderError):
    """Raised when quality policy forbids a provider downgrade."""


class ReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FallbackPolicy(str, Enum):
    NEVER = "never"
    OUTAGES = "outages"
    ANY_FAILURE = "any_failure"

    @property
    def permits_outage(self) -> bool:
        return self is not FallbackPolicy.NEVER

    @property
    def permits_validation_failure(self) -> bool:
        return self is FallbackPolicy.ANY_FAILURE


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("evidence_packet numbers must be finite")
        return value
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("evidence_packet object keys must be strings")
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            canonical_key = unicodedata.normalize("NFC", key)
            if canonical_key in normalized:
                raise ValueError("evidence_packet has duplicate normalized object keys")
            normalized[canonical_key] = _freeze_json(item)
        return MappingProxyType(normalized)
    raise TypeError("evidence_packet must contain only JSON values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _thaw_json(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("evidence_packet must be canonical JSON") from exc


@dataclass(frozen=True)
class ModelRequest:
    """One bounded structured-generation request without mutable prompt state."""

    role: AgentRole
    system_prompt: str = field(repr=False)
    evidence_packet: Any = field(repr=False)
    output_schema: type[BaseModel] = field(repr=False)
    max_output_tokens: int
    reasoning_effort: ReasoningEffort
    fallback_policy: FallbackPolicy
    run_id: str
    canonical_evidence: str = field(init=False, repr=False)
    evidence_hash: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            role = self.role if isinstance(self.role, AgentRole) else AgentRole(self.role)
            effort = (
                self.reasoning_effort
                if isinstance(self.reasoning_effort, ReasoningEffort)
                else ReasoningEffort(self.reasoning_effort)
            )
            policy = (
                self.fallback_policy
                if isinstance(self.fallback_policy, FallbackPolicy)
                else FallbackPolicy(self.fallback_policy)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("request enum value is invalid") from exc
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("system_prompt must be nonblank")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be nonblank")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if not isinstance(self.output_schema, type) or not issubclass(self.output_schema, BaseModel):
            raise TypeError("output_schema must be a Pydantic BaseModel class")
        frozen_evidence = _freeze_json(self.evidence_packet)
        canonical = _canonical_json(frozen_evidence)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "reasoning_effort", effort)
        object.__setattr__(self, "fallback_policy", policy)
        object.__setattr__(self, "evidence_packet", frozen_evidence)
        object.__setattr__(self, "canonical_evidence", canonical)
        object.__setattr__(
            self, "evidence_hash", hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        )


@dataclass(frozen=True)
class ModelResponse:
    """Validated data plus auditable metadata; raw provider text is never retained."""

    data: BaseModel
    raw_response_hash: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    model: str
    latency_ms: int
    provider: str
    inference_mode: InferenceMode
    run_id: str
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, BaseModel):
            raise TypeError("data must be a validated Pydantic model")
        if (
            not isinstance(self.raw_response_hash, str)
            or len(self.raw_response_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.raw_response_hash)
        ):
            raise ValueError("raw_response_hash must be a lowercase SHA-256 digest")
        for name in ("input_tokens", "output_tokens", "reasoning_tokens", "latency_ms"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("model", "provider", "run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonblank")
        if not isinstance(self.inference_mode, InferenceMode):
            raise TypeError("inference_mode must be InferenceMode")
        if self.fallback_reason is not None and (
            not isinstance(self.fallback_reason, str) or not self.fallback_reason.strip()
        ):
            raise ValueError("fallback_reason must be nonblank when provided")


@runtime_checkable
class ModelProvider(Protocol):
    name: str

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Return requested-schema data and provider-reported usage."""
