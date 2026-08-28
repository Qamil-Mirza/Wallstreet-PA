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
    ProviderNonRetryableError,
    ProviderRateLimitError,
    ProviderUnavailable,
    ProviderValidationError,
    ReasoningEffort,
)


@dataclass(frozen=True)
class ModelRoute:
    model: str
    reasoning_effort: ReasoningEffort

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ProviderConfigurationError("route model must be nonblank")
        try:
            effort = (
                self.reasoning_effort
                if isinstance(self.reasoning_effort, ReasoningEffort)
                else ReasoningEffort(self.reasoning_effort)
            )
        except (TypeError, ValueError) as exc:
            raise ProviderConfigurationError("route reasoning effort is invalid") from exc
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
        self.token_estimator = token_estimator or (lambda text: len(text.encode("utf-8")))
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
        prompt_for_estimate = f"{request.system_prompt}\n{request.canonical_evidence}"
        estimated_input = self.token_estimator(prompt_for_estimate)
        if type(estimated_input) is not int or estimated_input < 0:
            raise ProviderConfigurationError("token estimator returned an invalid count")
        instant = self.clock()
        estimate = self.prices.estimate(
            route.model,
            estimated_input,
            request.max_output_tokens,
            as_of=instant.date(),
        )
        reservation = self.budget.reserve(
            f"model:{request.run_id}:{request.role.value}",
            estimate,
            now=instant,
            role=request.role,
        )
        started = self.monotonic()
        try:
            provider_result = client.responses.create(
                model=route.model,
                instructions=request.system_prompt,
                input=request.canonical_evidence,
                max_output_tokens=request.max_output_tokens,
                reasoning={"effort": route.reasoning_effort.value},
                store=False,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": request.output_schema.__name__,
                        "schema": request.output_schema.model_json_schema(),
                        "strict": True,
                    }
                },
            )
        except Exception as exc:
            self._handle_call_failure(reservation.id, exc)
            raise AssertionError("unreachable")

        try:
            raw, input_tokens, output_tokens, reasoning_tokens = _extract_response(provider_result)
        except ProviderValidationError:
            self.budget.mark_usage_unknown(reservation.id, now=self.clock())
            raise

        actual = self.prices.estimate(
            route.model,
            input_tokens,
            output_tokens,
            as_of=self.clock().date(),
        )
        self.budget.reconcile(reservation.id, actual, now=self.clock())
        try:
            parsed_json = json.loads(raw)
            parsed = request.output_schema.model_validate(parsed_json)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
            raise ProviderValidationError("OpenAI response failed the requested schema") from None
        return ModelResponse(
            data=parsed,
            raw_response_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
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
        if isinstance(exc, (TimeoutError, ConnectionError)) or status in {408, 409} or (
            isinstance(status, int) and status >= 500
        ):
            self.budget.mark_usage_unknown(reservation_id, now=self.clock())
            raise ProviderUnavailable("OpenAI request outcome is unavailable") from None
        if isinstance(status, int) and 400 <= status < 500:
            self.budget.release(reservation_id, now=self.clock())
            raise ProviderNonRetryableError("OpenAI rejected the request") from None
        self.budget.mark_usage_unknown(reservation_id, now=self.clock())
        raise ProviderUnavailable("OpenAI request outcome is unavailable") from None


def _status_code(exc: Exception) -> int | None:
    direct = getattr(exc, "status_code", None)
    if type(direct) is int:
        return direct
    nested = getattr(getattr(exc, "response", None), "status_code", None)
    return nested if type(nested) is int else None


def _extract_response(result: Any) -> tuple[str, int, int, int]:
    raw = getattr(result, "output_text", None)
    usage = getattr(result, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", 0)
    if not isinstance(raw, str):
        raise ProviderValidationError("OpenAI response omitted structured output")
    for value in (input_tokens, output_tokens, reasoning_tokens):
        if type(value) is not int or value < 0:
            raise ProviderValidationError("OpenAI response omitted valid usage metadata")
    return raw, input_tokens, output_tokens, reasoning_tokens
