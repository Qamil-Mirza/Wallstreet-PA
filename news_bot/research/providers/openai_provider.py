"""OpenAI Responses structured-output adapter with fail-closed budgeting.

The injected client is called using the Responses API structured-output shape
documented at https://developers.openai.com/api/reference/cli/resources/responses/methods/create:
``responses.create(model=..., instructions=..., input=...,
max_output_tokens=..., reasoning={"effort": ...},
text={"format": {"type": "json_schema", "name": ..., "schema": ...,
"strict": True}})``.  Keeping that boundary injectable makes the contract
testable without a network connection, SDK installation, or API key.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

from pydantic import ValidationError

from ..budget import BudgetLedger, PriceTable
from ..models import AgentRole, InferenceMode
from .base import (
    ModelRequest,
    ModelResponse,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderNonRetryableError,
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderTerminalUnavailable,
    ProviderUnavailable,
    ProviderUsageUnavailable,
    ProviderValidationError,
    ReasoningEffort,
    ValidationIssue,
    _safe_text,
    validated_provider_json,
    validation_issues,
)


_REQUEST_OVERHEAD_TOKENS = 64


def _utf8_token_upper_bound(text: str) -> int:
    """Use one token per UTF-8 byte as a deterministic conservative bound."""
    return len(text.encode("utf-8", errors="strict"))


@dataclass(frozen=True)
class ModelRoute:
    model: str
    reasoning_effort: ReasoningEffort

    def __post_init__(self) -> None:
        model = _safe_text(
            "route model",
            self.model,
            allow_multiline=False,
            nonblank=True,
            error_type=ProviderConfigurationError,
        ).strip()
        try:
            effort = (
                self.reasoning_effort
                if isinstance(self.reasoning_effort, ReasoningEffort)
                else ReasoningEffort(self.reasoning_effort)
            )
        except (TypeError, ValueError):
            raise ProviderConfigurationError("route reasoning effort is invalid") from None
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "reasoning_effort", effort)


def _default_client_factory(api_key: str | None = None) -> Any:
    try:
        from openai import OpenAI
    except ImportError:
        raise ProviderConfigurationError("OpenAI SDK is not installed") from None
    if not api_key:
        raise ProviderConfigurationError("OpenAI API key is not configured")
    return OpenAI(api_key=api_key)


class OpenAIProvider:
    """Paid structured inference guarded by pessimistic atomic reservations."""

    name = "openai"

    def __init__(
        self,
        *,
        client: Any | None,
        routes: Mapping[AgentRole, ModelRoute],
        budget: BudgetLedger,
        prices: PriceTable,
        api_key: str | None = None,
        client_factory: Callable[[], Any] | None = None,
        token_estimator: Callable[[str], int] | None = None,
        clock: Callable[[], datetime],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(routes, Mapping) or not routes:
            raise ProviderConfigurationError("at least one OpenAI role route is required")
        normalized: dict[AgentRole, ModelRoute] = {}
        for role, route in routes.items():
            if not isinstance(role, AgentRole) or not isinstance(route, ModelRoute):
                raise ProviderConfigurationError("OpenAI routes must map AgentRole to ModelRoute")
            normalized[role] = route
        if not callable(clock) or not callable(monotonic):
            raise ProviderConfigurationError("provider clocks must be callable")
        self._client = client
        self._api_key = api_key
        self._client_factory = client_factory or (lambda: _default_client_factory(api_key))
        self.routes = MappingProxyType(normalized)
        self.budget = budget
        self.prices = prices
        estimator = token_estimator or _utf8_token_upper_bound
        if not callable(estimator):
            raise ProviderConfigurationError("token estimator must be callable")
        self.token_estimator = estimator
        self.clock = clock
        self.monotonic = monotonic

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be ModelRequest")
        try:
            route = self.routes[request.role]
        except KeyError:
            raise ProviderConfigurationError("no approved OpenAI route for role") from None
        client = self._get_client()
        api_request = {
            "model": route.model,
            "instructions": request.system_prompt,
            "input": request.provider_input,
            "max_output_tokens": request.max_output_tokens,
            "reasoning": {"effort": request.reasoning_effort.value},
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": request.schema_name,
                    "schema": request.schema_payload(),
                    "strict": True,
                }
            },
        }
        estimation_envelope = dict(api_request)
        estimation_envelope["role"] = request.role.value
        canonical_request = json.dumps(
            estimation_envelope,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        estimated_input = self.token_estimator(canonical_request)
        if type(estimated_input) is not int or estimated_input < 0:
            raise ProviderConfigurationError("token estimator returned an invalid count")
        estimated_input += _REQUEST_OVERHEAD_TOKENS
        instant = self.clock()
        # Responses max_output_tokens covers both visible output and internal
        # reasoning tokens, all billed in the output category.
        estimate = self.prices.estimate(
            route.model,
            estimated_input,
            request.max_output_tokens,
            as_of=instant.date(),
        )
        reservation = self.budget.reserve(
            f"model:{request.attempt_id or request.run_id}:{request.role.value}",
            estimate,
            now=instant,
            role=request.role,
        )
        started = self.monotonic()
        try:
            provider_result = client.responses.create(**api_request)
        except Exception as exc:
            self._handle_call_failure(reservation.id, exc)
            raise AssertionError("unreachable")

        try:
            input_tokens, output_tokens, reasoning_tokens = _extract_usage(
                provider_result
            )
        except ProviderValidationError:
            unknown = self.budget.mark_usage_unknown(
                reservation.id, now=self.clock()
            )
            error = ProviderUsageUnavailable(
                "paid provider usage metadata is unavailable"
            )
            error.usage_known = False
            error.input_tokens = None
            error.output_tokens = None
            error.reasoning_tokens = None
            error.reservation_id = unknown.id
            error.reservation_state = unknown.state
            error.reserved_cost_usd = unknown.amount
            raw_candidate = _response_field(
                provider_result, "output_text", None
            )
            if isinstance(raw_candidate, str):
                try:
                    error.response_hash = hashlib.sha256(
                        raw_candidate.encode("utf-8", errors="strict")
                    ).hexdigest()
                except UnicodeError:
                    error.response_hash = None
            raise error from None

        actual = self.prices.estimate(
            route.model,
            input_tokens,
            output_tokens,
            as_of=self.clock().date(),
        )
        self.budget.reconcile(reservation.id, actual, now=self.clock())
        try:
            response_hash = None
            raw_candidate = _response_field(provider_result, "output_text", None)
            if isinstance(raw_candidate, str):
                try:
                    response_hash = hashlib.sha256(
                        raw_candidate.encode("utf-8", errors="strict")
                    ).hexdigest()
                except UnicodeError:
                    pass
            _enforce_completed_response(provider_result)
            raw = _extract_output_text(provider_result)
            try:
                raw_bytes = raw.encode("utf-8", errors="strict")
            except UnicodeError:
                raise ProviderValidationError(
                    "OpenAI response failed the requested schema",
                    issues=(ValidationIssue("$", "invalid_utf8"),),
                ) from None
            response_hash = hashlib.sha256(raw_bytes).hexdigest()
            try:
                parsed_json = json.loads(raw)
            except json.JSONDecodeError:
                raise ProviderValidationError(
                    "OpenAI response failed the requested schema",
                    issues=(ValidationIssue("$", "json_invalid"),),
                ) from None
            parsed_json = validated_provider_json(parsed_json)
            try:
                parsed = request.output_schema.model_validate(parsed_json)
            except ValidationError as exc:
                raise ProviderValidationError(
                    "OpenAI response failed the requested schema",
                    issues=validation_issues(exc),
                ) from None
        except ProviderError as error:
            error.input_tokens = input_tokens
            error.output_tokens = output_tokens
            error.reasoning_tokens = reasoning_tokens
            error.response_hash = locals().get("response_hash")
            raise
        return ModelResponse(
            data=parsed,
            raw_response_hash=response_hash,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            model=route.model,
            latency_ms=max(0, round((self.monotonic() - started) * 1000)),
            provider=self.name,
            inference_mode=InferenceMode.EXTERNAL,
            run_id=request.run_id,
        )

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                self._client = self._client_factory()
            except ProviderConfigurationError:
                raise
            except ImportError:
                raise ProviderConfigurationError("OpenAI SDK is not installed") from None
            except Exception:
                raise ProviderConfigurationError("OpenAI client could not be configured") from None
        responses = getattr(self._client, "responses", None)
        if responses is None or not callable(getattr(responses, "create", None)):
            raise ProviderConfigurationError("OpenAI client does not expose Responses API")
        return self._client

    def _handle_call_failure(self, reservation_id: str, exc: Exception) -> None:
        status = _status_code(exc)
        if status in {401, 403}:
            self.budget.release(reservation_id, now=self.clock())
            raise ProviderAuthenticationError("OpenAI authentication failed") from None
        if status == 429:
            self.budget.release(reservation_id, now=self.clock())
            raise ProviderRateLimitError("OpenAI rate limit reached") from None
        if (
            isinstance(status, int)
            and 400 <= status < 500
            and status not in {408, 409}
        ):
            self.budget.release(reservation_id, now=self.clock())
            raise ProviderNonRetryableError("OpenAI rejected the request") from None
        unknown = self.budget.mark_usage_unknown(
            reservation_id, now=self.clock()
        )
        error = ProviderUnavailable("OpenAI request outcome is unavailable")
        error.usage_known = False
        error.input_tokens = None
        error.output_tokens = None
        error.reasoning_tokens = None
        error.reservation_id = unknown.id
        error.reservation_state = unknown.state
        error.reserved_cost_usd = unknown.amount
        raise error from None


def _status_code(exc: Exception) -> int | None:
    direct = getattr(exc, "status_code", None)
    if type(direct) is int:
        return direct
    nested = getattr(getattr(exc, "response", None), "status_code", None)
    return nested if type(nested) is int else None


def _extract_usage(result: Any) -> tuple[int, int, int]:
    usage = getattr(result, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", 0)
    for value in (input_tokens, output_tokens, reasoning_tokens):
        if type(value) is not int or value < 0:
            raise ProviderValidationError(
                "OpenAI response omitted valid usage metadata",
                issues=(ValidationIssue("usage", "invalid"),),
            )
    return input_tokens, output_tokens, reasoning_tokens


def _extract_output_text(result: Any) -> str:
    raw = getattr(result, "output_text", None)
    if not isinstance(raw, str):
        raise ProviderValidationError(
            issues=(ValidationIssue("$", "output_missing"),),
        )
    return raw


_ABSENT = object()


def _response_field(value: Any, name: str, default: Any = _ABSENT) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _contains_refusal(result: Any) -> bool:
    direct = _response_field(result, "refusal", None)
    if direct is not None:
        return True
    output = _response_field(result, "output", ())
    if not isinstance(output, (list, tuple)):
        return False
    for item in output:
        if _response_field(item, "type", None) == "refusal":
            return True
        if _response_field(item, "refusal", None) is not None:
            return True
        content = _response_field(item, "content", ())
        if not isinstance(content, (list, tuple)):
            continue
        for part in content:
            if _response_field(part, "type", None) == "refusal":
                return True
            if _response_field(part, "refusal", None) is not None:
                return True
    return False


def _enforce_completed_response(result: Any) -> None:
    """Reject non-completed terminal states after known usage is accounted."""
    try:
        if _contains_refusal(result):
            raise ProviderRefusalError() from None
        if _response_field(result, "error", None) is not None:
            raise ProviderTerminalUnavailable("response_error") from None
        if _response_field(result, "incomplete_details", None) is not None:
            raise ProviderTerminalUnavailable("response_incomplete") from None
        status = _response_field(result, "status")
    except (ProviderRefusalError, ProviderTerminalUnavailable):
        raise
    except Exception:
        raise ProviderTerminalUnavailable("response_uninspectable") from None

    # Compatibility with injected clients and older SDK fixtures that predate
    # the public status property. Explicit status values must be completed.
    if status is _ABSENT:
        return
    reason_code = {
        "incomplete": "response_incomplete",
        "failed": "response_failed",
        "cancelled": "response_cancelled",
        "error": "response_error",
    }.get(status, "response_not_terminal")
    if status != "completed":
        raise ProviderTerminalUnavailable(reason_code) from None
