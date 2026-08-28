"""Immutable provider contracts and redacted failure taxonomy."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from ..models import AgentRole, InferenceMode


class ProviderError(Exception):
    """Base class for redacted inference-provider failures."""


class ProviderConfigurationError(ProviderError, ValueError):
    """Raised when a provider cannot be configured safely."""


class ProviderRequestError(ProviderError, ValueError):
    """Raised when public request data is unsafe or invalid."""


class ProviderAuthenticationError(ProviderError):
    """Raised when external credentials are rejected."""


class ProviderRateLimitError(ProviderError):
    """Raised when a provider rejects a request due to quota or rate."""


class ProviderUnavailable(ProviderError):
    """Raised for timeouts and retryable provider outages."""


_TERMINAL_REASON_CODES = frozenset(
    {
        "response_cancelled",
        "response_error",
        "response_failed",
        "response_incomplete",
        "response_not_terminal",
        "response_uninspectable",
    }
)


class ProviderTerminalUnavailable(ProviderUnavailable):
    """A paid response that cannot safely be treated as completed."""

    def __init__(self, reason_code: str) -> None:
        if reason_code not in _TERMINAL_REASON_CODES:
            raise ProviderConfigurationError("terminal reason code is invalid")
        self.reason_code = reason_code
        super().__init__("external provider response was not completed")


class ProviderValidationError(ProviderError):
    """Raised when provider JSON violates the requested contract."""

    def __init__(
        self,
        message: str = "provider response failed validation",
        *,
        issues: tuple[ValidationIssue, ...] = (),
    ) -> None:
        if not isinstance(issues, tuple) or not all(
            isinstance(issue, ValidationIssue) for issue in issues
        ):
            raise TypeError("issues must be a tuple of ValidationIssue")
        del message
        self._issues = issues
        super().__init__("provider response failed validation")

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        """Return immutable, schema-only repair feedback."""
        return self._issues


class ProviderNonRetryableError(ProviderError):
    """Raised for a redacted provider rejection that must not be retried."""


class ProviderRefusalError(ProviderNonRetryableError):
    """Raised when a completed provider response contains a refusal."""

    reason_code = "response_refusal"

    def __init__(self) -> None:
        super().__init__("external provider refused the structured request")


class TaskDeferred(ProviderError):
    """Raised when quality policy forbids a provider downgrade."""


class ReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ProviderAttemptTrace:
    """Redacted telemetry for one physical provider invocation."""

    status: str
    provider: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    inference_mode: InferenceMode
    recorded_at: datetime
    fallback_reason: str | None = None
    response_hash: str | None = None
    failure_code: str | None = None


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


_ISSUE_PATH = re.compile(r"^[A-Za-z0-9_$\.\[\]-]+$")
_ISSUE_CODE = re.compile(r"^[a-z0-9_.-]+$")
_SCHEMA_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _safe_text(
    field_name: str,
    value: Any,
    *,
    allow_multiline: bool,
    nonblank: bool,
    error_type: type[Exception],
) -> str:
    if not isinstance(value, str):
        raise error_type(f"{field_name} must be text")
    normalized = unicodedata.normalize("NFC", value)
    try:
        normalized.encode("utf-8", errors="strict")
    except UnicodeError:
        raise error_type(f"{field_name} contains invalid Unicode") from None
    allowed_controls = {"\t", "\n", "\r"} if allow_multiline else set()
    if any(
        unicodedata.category(character) == "Cc" and character not in allowed_controls
        for character in normalized
    ):
        raise error_type(f"{field_name} contains a forbidden control character")
    if nonblank and not normalized.strip():
        raise error_type(f"{field_name} must be nonblank")
    return normalized


@dataclass(frozen=True)
class ValidationIssue:
    """Schema-only retry feedback that cannot contain provider output."""

    path: str
    code: str

    def __post_init__(self) -> None:
        path = _safe_text(
            "validation path",
            self.path,
            allow_multiline=False,
            nonblank=True,
            error_type=ProviderValidationError,
        )
        code = _safe_text(
            "validation code",
            self.code,
            allow_multiline=False,
            nonblank=True,
            error_type=ProviderValidationError,
        )
        if not _ISSUE_PATH.fullmatch(path) or not _ISSUE_CODE.fullmatch(code):
            raise ProviderValidationError("validation feedback is not schema-only")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "code", code)


def validation_issues(error: Any) -> tuple[ValidationIssue, ...]:
    """Reduce Pydantic errors to redacted field paths and machine codes."""
    try:
        entries = error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    except Exception:
        return (ValidationIssue("$", "schema_validation"),)
    issues: list[ValidationIssue] = []
    for entry in entries:
        location = entry.get("loc", ()) if isinstance(entry, dict) else ()
        pieces = []
        for item in location:
            piece = str(item)
            pieces.append(piece if re.fullmatch(r"[A-Za-z0-9_-]+", piece) else "field")
        path = ".".join(pieces) or "$"
        raw_code = entry.get("type", "schema_validation") if isinstance(entry, dict) else "schema_validation"
        code = raw_code if isinstance(raw_code, str) and _ISSUE_CODE.fullmatch(raw_code) else "schema_validation"
        issue = ValidationIssue(path, code)
        if issue not in issues:
            issues.append(issue)
    return tuple(issues) or (ValidationIssue("$", "schema_validation"),)


def validated_provider_json(value: Any) -> Any:
    """Validate model JSON text recursively without retaining its contents."""
    try:
        return _thaw_json(_freeze_json(value))
    except (ProviderRequestError, TypeError, ValueError):
        raise ProviderValidationError(
            issues=(ValidationIssue("$", "invalid_text"),)
        ) from None


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return _safe_text(
            "evidence text",
            value,
            allow_multiline=True,
            nonblank=False,
            error_type=ProviderRequestError,
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProviderRequestError("JSON numbers must be finite")
        return value
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("evidence_packet object keys must be strings")
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            canonical_key = _safe_text(
                "evidence key",
                key,
                allow_multiline=False,
                nonblank=True,
                error_type=ProviderRequestError,
            )
            if canonical_key in normalized:
                raise ProviderRequestError(
                    "evidence_packet has duplicate normalized object keys"
                )
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
    except (TypeError, ValueError):
        raise ProviderRequestError("value must be canonical JSON") from None


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
    attempt_id: str | None = None
    validation_feedback: tuple[ValidationIssue, ...] = field(default=(), repr=False)
    attempt_recorder: Callable[[ProviderAttemptTrace], None] | None = field(
        default=None, repr=False, compare=False
    )
    canonical_evidence: str = field(init=False, repr=False)
    evidence_hash: str = field(init=False)
    schema_name: str = field(init=False)
    canonical_schema: str = field(init=False, repr=False)
    provider_input: str = field(init=False, repr=False)

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
        except (TypeError, ValueError):
            raise ValueError("request enum value is invalid") from None
        system_prompt = _safe_text(
            "system_prompt",
            self.system_prompt,
            allow_multiline=True,
            nonblank=True,
            error_type=ProviderRequestError,
        )
        run_id = _safe_text(
            "run_id",
            self.run_id,
            allow_multiline=False,
            nonblank=True,
            error_type=ProviderRequestError,
        )
        attempt_id = None
        if self.attempt_id is not None:
            attempt_id = _safe_text(
                "attempt_id", self.attempt_id, allow_multiline=False,
                nonblank=True, error_type=ProviderRequestError,
            )
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if not isinstance(self.output_schema, type) or not issubclass(self.output_schema, BaseModel):
            raise TypeError("output_schema must be a Pydantic BaseModel class")
        schema_name = _safe_text(
            "schema name",
            self.output_schema.__name__,
            allow_multiline=False,
            nonblank=True,
            error_type=ProviderRequestError,
        )
        if not _SCHEMA_NAME.fullmatch(schema_name):
            raise ProviderRequestError("schema name is not API-safe")
        try:
            schema = self.output_schema.model_json_schema()
        except Exception:
            raise ProviderRequestError("output schema could not be serialized") from None
        frozen_schema = _freeze_json(schema)
        canonical_schema = _canonical_json(frozen_schema)
        if not isinstance(self.validation_feedback, tuple) or not all(
            isinstance(issue, ValidationIssue) for issue in self.validation_feedback
        ):
            raise ProviderRequestError(
                "validation_feedback must be a tuple of ValidationIssue"
            )
        if self.attempt_recorder is not None and not callable(self.attempt_recorder):
            raise ProviderRequestError("attempt_recorder must be callable or None")
        frozen_evidence = _freeze_json(self.evidence_packet)
        canonical = _canonical_json(frozen_evidence)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "reasoning_effort", effort)
        object.__setattr__(self, "fallback_policy", policy)
        object.__setattr__(self, "system_prompt", system_prompt)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "attempt_id", attempt_id)
        object.__setattr__(self, "evidence_packet", frozen_evidence)
        object.__setattr__(self, "canonical_evidence", canonical)
        object.__setattr__(self, "schema_name", schema_name)
        object.__setattr__(self, "canonical_schema", canonical_schema)
        object.__setattr__(
            self, "evidence_hash", hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        )
        if self.validation_feedback:
            provider_input = json.dumps(
                {
                    "evidence": _thaw_json(frozen_evidence),
                    "validation_feedback": [
                        {"path": issue.path, "code": issue.code}
                        for issue in self.validation_feedback
                    ],
                },
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        else:
            provider_input = canonical
        object.__setattr__(self, "provider_input", provider_input)

    def for_validation_retry(
        self, issues: tuple[ValidationIssue, ...]
    ) -> "ModelRequest":
        """Return one new validated request containing only schema feedback."""
        safe_issues = issues or (ValidationIssue("$", "schema_validation"),)
        return replace(self, validation_feedback=safe_issues)

    def schema_payload(self) -> dict[str, Any]:
        """Return an isolated JSON-schema copy for a provider request."""
        value = json.loads(self.canonical_schema)
        if not isinstance(value, dict):  # defensive: Pydantic schemas are objects
            raise ProviderRequestError("output schema must be a JSON object")
        return value


@dataclass(frozen=True, init=False)
class ModelResponse:
    """Validated data plus auditable metadata; raw provider text is never retained."""

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
    _data_schema: type[BaseModel] = field(repr=False)
    _canonical_data: str = field(repr=False)

    def __init__(
        self,
        data: BaseModel,
        raw_response_hash: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        model: str,
        latency_ms: int,
        provider: str,
        inference_mode: InferenceMode,
        run_id: str,
        fallback_reason: str | None = None,
    ) -> None:
        if not isinstance(data, BaseModel):
            raise TypeError("data must be a validated Pydantic model")
        try:
            frozen_data = _freeze_json(
                data.model_dump(mode="json", by_alias=True)
            )
            canonical_data = _canonical_json(frozen_data)
            type(data).model_validate_json(canonical_data)
        except Exception:
            raise ProviderRequestError("data contains invalid JSON text") from None
        object.__setattr__(self, "_data_schema", type(data))
        object.__setattr__(self, "_canonical_data", canonical_data)
        if (
            not isinstance(raw_response_hash, str)
            or len(raw_response_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in raw_response_hash
            )
        ):
            raise ValueError("raw_response_hash must be a lowercase SHA-256 digest")
        object.__setattr__(self, "raw_response_hash", raw_response_hash)
        values = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "latency_ms": latency_ms,
        }
        for name in ("input_tokens", "output_tokens", "reasoning_tokens", "latency_ms"):
            value = values[name]
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, value)
        text_values = {"model": model, "provider": provider, "run_id": run_id}
        for name in ("model", "provider", "run_id"):
            value = _safe_text(
                name,
                text_values[name],
                allow_multiline=False,
                nonblank=True,
                error_type=ProviderRequestError,
            )
            object.__setattr__(self, name, value)
        if not isinstance(inference_mode, InferenceMode):
            raise TypeError("inference_mode must be InferenceMode")
        object.__setattr__(self, "inference_mode", inference_mode)
        if fallback_reason is not None:
            fallback_reason = _safe_text(
                "fallback_reason",
                fallback_reason,
                allow_multiline=False,
                nonblank=True,
                error_type=ProviderRequestError,
            )
        object.__setattr__(self, "fallback_reason", fallback_reason)

    @property
    def data(self) -> BaseModel:
        """Return a fresh validated copy of the canonical stored result."""
        try:
            return self._data_schema.model_validate_json(self._canonical_data)
        except Exception:
            raise ProviderRequestError(
                "stored response data could not be validated"
            ) from None

    def with_routing(
        self,
        *,
        inference_mode: InferenceMode,
        fallback_reason: str | None,
        run_id: str,
    ) -> "ModelResponse":
        """Return a defensive response copy with explicit routing metadata."""
        return ModelResponse(
            data=self.data,
            raw_response_hash=self.raw_response_hash,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            reasoning_tokens=self.reasoning_tokens,
            model=self.model,
            latency_ms=self.latency_ms,
            provider=self.provider,
            inference_mode=inference_mode,
            run_id=run_id,
            fallback_reason=fallback_reason,
        )


@runtime_checkable
class ModelProvider(Protocol):
    name: str

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Return requested-schema data and provider-reported usage."""


def provider_failure_code(error: Exception) -> str:
    """Map provider failures to a stable code without retaining messages."""
    if isinstance(error, ProviderTerminalUnavailable):
        return error.reason_code
    mapping = (
        (ProviderAuthenticationError, "provider_authentication"),
        (ProviderRateLimitError, "provider_rate_limit"),
        (ProviderValidationError, "provider_validation"),
        (ProviderNonRetryableError, "provider_nonretryable"),
        (ProviderUnavailable, "provider_unavailable"),
        (TaskDeferred, "task_deferred"),
        (ProviderRequestError, "provider_request"),
        (ProviderConfigurationError, "provider_configuration"),
    )
    for error_type, code in mapping:
        if isinstance(error, error_type):
            return code
    return "provider_error"
