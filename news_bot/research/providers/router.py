"""Run-isolated routing between external inference and Ollama fallback."""

from __future__ import annotations

from threading import Lock, RLock

from ..config import ResearchConfig
from ..models import AgentRole, InferenceMode
from .base import (
    FallbackPolicy,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderAuthenticationError,
    ProviderNonRetryableError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderUnavailable,
    ProviderValidationError,
    TaskDeferred,
    _safe_text,
)


class _ValidationRetriesExhausted(Exception):
    def __init__(self, error: ProviderValidationError) -> None:
        self.error = error
        super().__init__("structured validation retry exhausted")


class ProviderRouter:
    """Select providers without silently changing inference quality."""

    def __init__(
        self,
        config: ResearchConfig,
        external: ModelProvider | None,
        ollama: ModelProvider,
    ) -> None:
        if not isinstance(config, ResearchConfig):
            raise TypeError("config must be ResearchConfig")
        if not isinstance(ollama, ModelProvider):
            raise TypeError("ollama must implement ModelProvider")
        if external is not None and not isinstance(external, ModelProvider):
            raise TypeError("external must implement ModelProvider or be None")
        self.config = config
        self.external = external
        self.ollama = ollama
        self._state_lock = RLock()
        self._disabled: dict[str, str] = {}
        self._run_locks: dict[str, Lock] = {}

    def provider_for(self, role: AgentRole, *, run_id: str) -> ModelProvider:
        if not isinstance(role, AgentRole):
            raise TypeError("role must be AgentRole")
        safe_run_id = self._safe_run_id(run_id)
        with self._state_lock:
            if not self._external_is_configured() or safe_run_id in self._disabled:
                return self.ollama
            assert self.external is not None
            return self.external

    def external_disabled_reason(self, run_id: str) -> str | None:
        safe_run_id = self._safe_run_id(run_id)
        with self._state_lock:
            return self._disabled.get(safe_run_id)

    def clear_run(self, run_id: str) -> None:
        """Discard routing state once an orchestrated run has completed."""
        safe_run_id = self._safe_run_id(run_id)
        with self._state_lock:
            self._disabled.pop(safe_run_id, None)
            self._run_locks.pop(safe_run_id, None)

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be ModelRequest")
        lock = self._lock_for(request.run_id)
        with lock:
            disabled = self.external_disabled_reason(request.run_id)
            if not self._external_is_configured():
                return self._generate_local(request, "external_not_configured")
            if disabled is not None:
                if not request.fallback_policy.permits_outage:
                    raise TaskDeferred("external provider disabled for this run")
                return self._generate_local(request, disabled)

            assert self.external is not None
            try:
                return self._generate_with_validation_retry(self.external, request)
            except _ValidationRetriesExhausted as exhausted:
                if request.fallback_policy.permits_validation_failure:
                    fallback_request = request.for_validation_retry(
                        exhausted.error.issues
                    )
                    return self._generate_local(
                        fallback_request, "external_validation"
                    )
                if request.fallback_policy is FallbackPolicy.NEVER:
                    raise TaskDeferred(
                        "external validation failed twice; local fallback forbidden"
                    ) from None
                raise exhausted.error from None
            except ProviderAuthenticationError:
                with self._state_lock:
                    self._disabled[request.run_id] = "authentication"
                if not request.fallback_policy.permits_outage:
                    raise TaskDeferred("external authentication failed; local fallback forbidden") from None
                return self._generate_local(request, "authentication")
            except ProviderRateLimitError:
                if not request.fallback_policy.permits_outage:
                    raise TaskDeferred("external provider rate limited; local fallback forbidden") from None
                return self._generate_local(request, "external_rate_limit")
            except ProviderUnavailable:
                if not request.fallback_policy.permits_outage:
                    raise TaskDeferred("external provider unavailable; local fallback forbidden") from None
                return self._generate_local(request, "external_unavailable")
            except ProviderNonRetryableError:
                if request.fallback_policy is not FallbackPolicy.ANY_FAILURE:
                    raise
                return self._generate_local(request, "external_nonretryable")

    def _generate_local(self, request: ModelRequest, reason: str) -> ModelResponse:
        try:
            result = self._generate_with_validation_retry(self.ollama, request)
        except _ValidationRetriesExhausted as exhausted:
            raise exhausted.error from None
        except ProviderUnavailable:
            raise TaskDeferred("configured Ollama model is unavailable") from None
        return result.with_routing(
            inference_mode=InferenceMode.LOCAL_ONLY,
            fallback_reason=reason,
            run_id=request.run_id,
        )

    @staticmethod
    def _generate_with_validation_retry(
        provider: ModelProvider, request: ModelRequest
    ) -> ModelResponse:
        try:
            return provider.generate(request)
        except ProviderValidationError as first_error:
            retry_request = request.for_validation_retry(first_error.issues)
        try:
            return provider.generate(retry_request)
        except ProviderValidationError as second_error:
            raise _ValidationRetriesExhausted(second_error) from None

    @staticmethod
    def _safe_run_id(run_id: object) -> str:
        return _safe_text(
            "run_id",
            run_id,
            allow_multiline=False,
            nonblank=True,
            error_type=ProviderRequestError,
        )

    def _lock_for(self, run_id: str) -> Lock:
        with self._state_lock:
            return self._run_locks.setdefault(run_id, Lock())

    def _external_is_configured(self) -> bool:
        return (
            self.config.inference_mode is InferenceMode.EXTERNAL
            and self.config.openai_api_key is not None
            and self.external is not None
        )
