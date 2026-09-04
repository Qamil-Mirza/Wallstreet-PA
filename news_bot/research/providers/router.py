"""Run-isolated routing between external inference and Ollama fallback."""

from __future__ import annotations

import time
from datetime import datetime, timezone
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
    ProviderAttemptTrace,
    ProviderRequestError,
    ProviderUnavailable,
    ProviderUsageUnavailable,
    ProviderValidationError,
    TaskDeferred,
    _safe_text,
    provider_failure_code,
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
        *,
        clock=lambda: datetime.now(timezone.utc),
        monotonic=time.monotonic,
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
        if not callable(clock) or not callable(monotonic):
            raise TypeError("router clocks must be callable")
        self.clock = clock
        self.monotonic = monotonic
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
            attempt_budget = [0]
            disabled = self.external_disabled_reason(request.run_id)
            if not self._external_is_configured():
                return self._generate_local(
                    request, "external_not_configured", attempt_budget
                )
            if disabled is not None:
                if not request.fallback_policy.permits_outage:
                    raise TaskDeferred("external provider disabled for this run")
                return self._generate_local(request, disabled, attempt_budget)

            assert self.external is not None
            try:
                return self._generate_with_validation_retry(
                    self.external, request, attempt_budget,
                    inference_mode=InferenceMode.EXTERNAL,
                )
            except _ValidationRetriesExhausted as exhausted:
                if request.fallback_policy.permits_validation_failure:
                    if attempt_budget[0] >= 2:
                        raise exhausted.error from None
                    fallback_request = request.for_validation_retry(
                        exhausted.error.issues
                    )
                    return self._generate_local(
                        fallback_request, "external_validation", attempt_budget
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
                return self._generate_local(request, "authentication", attempt_budget)
            except ProviderRateLimitError:
                if not request.fallback_policy.permits_outage:
                    raise TaskDeferred("external provider rate limited; local fallback forbidden") from None
                return self._generate_local(
                    request, "external_rate_limit", attempt_budget
                )
            except ProviderUnavailable:
                if not request.fallback_policy.permits_outage:
                    raise TaskDeferred("external provider unavailable; local fallback forbidden") from None
                return self._generate_local(
                    request, "external_unavailable", attempt_budget
                )
            except ProviderUsageUnavailable:
                raise
            except ProviderNonRetryableError:
                if request.fallback_policy is not FallbackPolicy.ANY_FAILURE:
                    raise
                return self._generate_local(
                    request, "external_nonretryable", attempt_budget
                )

    def _generate_local(
        self, request: ModelRequest, reason: str, attempt_budget: list[int]
    ) -> ModelResponse:
        try:
            result = self._generate_with_validation_retry(
                self.ollama, request, attempt_budget,
                inference_mode=InferenceMode.LOCAL_ONLY,
                fallback_reason=reason,
            )
        except _ValidationRetriesExhausted as exhausted:
            raise exhausted.error from None
        except ProviderUnavailable:
            raise TaskDeferred("configured Ollama model is unavailable") from None
        return result.with_routing(
            inference_mode=InferenceMode.LOCAL_ONLY,
            fallback_reason=reason,
            run_id=request.run_id,
        )

    def _generate_with_validation_retry(
        self,
        provider: ModelProvider,
        request: ModelRequest,
        attempt_budget: list[int],
        *,
        inference_mode: InferenceMode,
        fallback_reason: str | None = None,
    ) -> ModelResponse:
        try:
            return self._call_provider(
                provider, request, attempt_budget,
                inference_mode=inference_mode, fallback_reason=fallback_reason,
            )
        except ProviderValidationError as first_error:
            retry_request = request.for_validation_retry(first_error.issues)
        if attempt_budget[0] >= 2:
            raise _ValidationRetriesExhausted(first_error) from None
        try:
            return self._call_provider(
                provider, retry_request, attempt_budget,
                inference_mode=inference_mode, fallback_reason=fallback_reason,
            )
        except ProviderValidationError as second_error:
            raise _ValidationRetriesExhausted(second_error) from None

    def _call_provider(
        self,
        provider: ModelProvider,
        request: ModelRequest,
        attempt_budget: list[int],
        *,
        inference_mode: InferenceMode,
        fallback_reason: str | None,
    ) -> ModelResponse:
        if attempt_budget[0] >= 2:
            raise ProviderValidationError()
        attempt_budget[0] += 1
        started = self.monotonic()
        try:
            response = provider.generate(request)
        except Exception as error:
            self._record_attempt(request, ProviderAttemptTrace(
                status="failed", provider=getattr(provider, "name", "provider"),
                model=self._provider_model(provider, request),
                latency_ms=max(0, round((self.monotonic() - started) * 1000)),
                input_tokens=getattr(error, "input_tokens", 0),
                output_tokens=getattr(error, "output_tokens", 0),
                reasoning_tokens=getattr(error, "reasoning_tokens", 0),
                inference_mode=inference_mode, fallback_reason=fallback_reason,
                response_hash=getattr(error, "response_hash", None),
                failure_code=provider_failure_code(error), recorded_at=self.clock(),
                usage_known=getattr(error, "usage_known", True),
                reservation_id=getattr(error, "reservation_id", None),
                reservation_state=getattr(error, "reservation_state", None),
                reserved_cost_usd=getattr(error, "reserved_cost_usd", None),
            ))
            raise
        self._record_attempt(request, ProviderAttemptTrace(
            status="succeeded", provider=response.provider, model=response.model,
            latency_ms=response.latency_ms, input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            reasoning_tokens=response.reasoning_tokens,
            inference_mode=inference_mode, fallback_reason=fallback_reason,
            response_hash=response.raw_response_hash, failure_code=None,
            recorded_at=self.clock(),
        ))
        return response

    @staticmethod
    def _record_attempt(
        request: ModelRequest, trace: ProviderAttemptTrace
    ) -> None:
        if request.attempt_recorder is not None:
            request.attempt_recorder(trace)

    @staticmethod
    def _provider_model(provider: ModelProvider, request: ModelRequest) -> str:
        model = getattr(provider, "model", None)
        if isinstance(model, str) and model:
            return model
        routes = getattr(provider, "routes", None)
        try:
            routed = routes[request.role].model
        except (AttributeError, KeyError, TypeError):
            routed = None
        return routed if isinstance(routed, str) and routed else getattr(
            provider, "name", "provider"
        )

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
