"""Contract tests for provider-neutral structured model generation."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from pydantic import BaseModel, ConfigDict

from news_bot.research.budget import BudgetLedger, ModelPrice, PriceTable
from news_bot.research.config import ResearchConfig
from news_bot.research.models import AgentRole, InferenceMode
from news_bot.research.providers.base import (
    FallbackPolicy,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderUnavailable,
    ProviderValidationError,
    ReasoningEffort,
    TaskDeferred,
    ValidationIssue,
)
from news_bot.research.providers.ollama_provider import OllamaProvider
from news_bot.research.providers.openai_provider import ModelRoute, OpenAIProvider
from news_bot.research.providers.router import ProviderRouter

from .conftest import utc


class ResearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    answer: str


def make_request(
    *,
    role: AgentRole = AgentRole.EVENT_SCOUT,
    run_id: str = "run-1",
    policy: FallbackPolicy = FallbackPolicy.OUTAGES,
    effort: ReasoningEffort = ReasoningEffort.LOW,
) -> ModelRequest:
    return ModelRequest(
        role=role,
        system_prompt="Cite the supplied evidence.",
        evidence_packet={"passages": [{"id": "passage-1", "text": "Revenue rose."}]},
        output_schema=ResearchOutput,
        max_output_tokens=80,
        reasoning_effort=effort,
        fallback_policy=policy,
        run_id=run_id,
    )


def make_config(*, external: bool) -> ResearchConfig:
    return ResearchConfig(
        enabled=True,
        data_dir=Path("research-data"),
        openai_api_key="configured-key" if external else None,
        inference_mode=(InferenceMode.EXTERNAL if external else InferenceMode.LOCAL_ONLY),
        ollama_base_url="http://localhost:11434",
        ollama_model="research:latest",
        budget_soft_usd=Decimal("4"),
        budget_hard_usd=Decimal("5"),
    )


def response(provider: str, *, run_id: str = "run-1") -> ModelResponse:
    return ModelResponse(
        data=ResearchOutput(answer="supported"),
        raw_response_hash="a" * 64,
        input_tokens=3,
        output_tokens=2,
        reasoning_tokens=1,
        model="research:latest",
        latency_ms=5,
        provider=provider,
        inference_mode=(InferenceMode.EXTERNAL if provider == "openai" else InferenceMode.LOCAL_ONLY),
        run_id=run_id,
    )


class StubProvider:
    def __init__(self, name: str, outcomes):
        self.name = name
        self.outcomes = list(outcomes)
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        outcome = self.outcomes.pop(0) if self.outcomes else response(self.name, run_id=request.run_id)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeHttpResponse:
    def __init__(self, payload, *, status_code=200, chunks=None):
        import json
        self.status_code = status_code
        self._body = json.dumps(payload).encode("utf-8")
        self._chunks = chunks
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError("remote request failed")
            error.response = self
            raise error

    def iter_content(self, chunk_size):
        del chunk_size
        yield from (self._chunks if self._chunks is not None else [self._body])

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False
        self.trust_env = True

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.responses.pop(0)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def test_model_request_is_frozen_validated_and_hashes_canonical_evidence():
    first = make_request()
    second = ModelRequest(
        role=first.role,
        system_prompt=first.system_prompt,
        evidence_packet={"passages": [{"text": "Revenue rose.", "id": "passage-1"}]},
        output_schema=ResearchOutput,
        max_output_tokens=80,
        reasoning_effort="low",
        fallback_policy="outages",
        run_id="run-1",
    )
    assert first.evidence_hash == second.evidence_hash
    assert first.canonical_evidence == second.canonical_evidence
    with pytest.raises(FrozenInstanceError):
        first.run_id = "changed"


def test_model_request_normalizes_equivalent_unicode_before_hashing():
    composed = make_request()
    decomposed = ModelRequest(
        role=AgentRole.EVENT_SCOUT,
        system_prompt="system",
        evidence_packet={"name": "caf\u0065\u0301"},
        output_schema=ResearchOutput,
        max_output_tokens=10,
        reasoning_effort=ReasoningEffort.LOW,
        fallback_policy=FallbackPolicy.OUTAGES,
        run_id="unicode-1",
    )
    normalized = ModelRequest(
        role=AgentRole.EVENT_SCOUT,
        system_prompt="system",
        evidence_packet={"name": "caf\u00e9"},
        output_schema=ResearchOutput,
        max_output_tokens=10,
        reasoning_effort=ReasoningEffort.LOW,
        fallback_policy=FallbackPolicy.OUTAGES,
        run_id="unicode-2",
    )
    assert composed.evidence_hash != decomposed.evidence_hash
    assert decomposed.evidence_hash == normalized.evidence_hash


@pytest.mark.parametrize("changes", [
    {"system_prompt": "   "},
    {"evidence_packet": {"value": float("nan")}},
    {"output_schema": dict},
    {"max_output_tokens": 0},
    {"reasoning_effort": "extreme"},
    {"run_id": ""},
])
def test_model_request_rejects_invalid_boundaries(changes):
    values = dict(
        role=AgentRole.EVENT_SCOUT, system_prompt="system", evidence_packet={},
        output_schema=ResearchOutput, max_output_tokens=10,
        reasoning_effort=ReasoningEffort.LOW,
        fallback_policy=FallbackPolicy.OUTAGES, run_id="run-1",
    )
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        ModelRequest(**values)


def test_model_request_repr_does_not_disclose_prompt_or_evidence():
    request = make_request()
    rendered = repr(request)
    assert "Cite the supplied evidence" not in rendered
    assert "Revenue rose" not in rendered
    assert request.evidence_hash in rendered


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("system_prompt", "marker\ud800"),
        ("system_prompt", "marker\x00"),
        ("run_id", "marker\ud800"),
        ("run_id", "marker\nvalue"),
        ("evidence_packet", {"text": "marker\ud800"}),
        ("evidence_packet", {"marker\x00": "value"}),
    ],
)
def test_model_request_rejects_invalid_utf8_or_control_text_without_leak(field, value):
    values = dict(
        role=AgentRole.EVENT_SCOUT,
        system_prompt="system",
        evidence_packet={},
        output_schema=ResearchOutput,
        max_output_tokens=10,
        reasoning_effort=ReasoningEffort.LOW,
        fallback_policy=FallbackPolicy.OUTAGES,
        run_id="run-1",
    )
    values[field] = value
    with pytest.raises(ProviderRequestError) as raised:
        ModelRequest(**values)
    assert "marker" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_validation_issue_rejects_unsafe_feedback_without_leak():
    with pytest.raises(ProviderValidationError) as raised:
        ValidationIssue(path="marker\ud800", code="missing")
    assert "marker" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_provider_validation_error_never_retains_caller_message():
    issue = ValidationIssue(path="answer", code="missing")
    error = ProviderValidationError("marker\ud800", issues=(issue,))
    assert "marker" not in str(error)
    assert error.issues == (issue,)
    assert error.__cause__ is None
    with pytest.raises(FrozenInstanceError):
        issue.path = "changed"
    with pytest.raises(AttributeError):
        error.issues = ()


def test_model_provider_protocol_is_runtime_checkable():
    assert isinstance(StubProvider("ollama", []), ModelProvider)


def test_model_response_rejects_raw_or_invalid_usage_metadata():
    with pytest.raises(ValueError):
        ModelResponse(
            data=ResearchOutput(answer="ok"), raw_response_hash="raw text",
            input_tokens=-1, output_tokens=0, reasoning_tokens=0, model="m",
            latency_ms=0, provider="ollama", inference_mode=InferenceMode.LOCAL_ONLY,
            run_id="run-1",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "marker\ud800"),
        ("provider", "marker\x00"),
        ("run_id", "marker\nvalue"),
        ("fallback_reason", "marker\ud800"),
    ],
)
def test_model_response_rejects_unsafe_metadata_without_leak(field, value):
    values = dict(
        data=ResearchOutput(answer="supported"),
        raw_response_hash="a" * 64,
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        model="research:latest",
        latency_ms=1,
        provider="ollama",
        inference_mode=InferenceMode.LOCAL_ONLY,
        run_id="run-1",
        fallback_reason=None,
    )
    values[field] = value
    with pytest.raises(ProviderRequestError) as raised:
        ModelResponse(**values)
    assert "marker" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_model_response_rejects_unsafe_validated_model_text_without_leak():
    with pytest.raises(ProviderRequestError) as raised:
        ModelResponse(
            data=ResearchOutput(answer="marker\ud800"),
            raw_response_hash="a" * 64,
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            model="research:latest",
            latency_ms=1,
            provider="ollama",
            inference_mode=InferenceMode.LOCAL_ONLY,
            run_id="run-1",
        )
    assert "marker" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_ollama_checks_model_then_generates_with_json_schema():
    session = FakeSession([
        FakeHttpResponse({"models": [{"name": "research:latest"}]}),
        FakeHttpResponse({"response": '{"answer":"supported"}', "prompt_eval_count": 9, "eval_count": 4}),
    ])
    provider = OllamaProvider(
        base_url="http://localhost:11434", model="research:latest",
        session=session, clock=lambda: 10.0,
    )
    result = provider.generate(make_request())
    assert [call[0] for call in session.calls] == ["GET", "POST"]
    payload = session.calls[1][2]["json"]
    assert payload["stream"] is False
    assert payload["format"] == ResearchOutput.model_json_schema()
    assert payload["options"]["num_predict"] == 80
    assert payload["prompt"] == make_request().canonical_evidence
    assert session.calls[0][2]["allow_redirects"] is False
    assert session.calls[1][2]["allow_redirects"] is False
    assert result.data == ResearchOutput(answer="supported")
    assert result.raw_response_hash != '{"answer":"supported"}'
    assert result.input_tokens == 9
    assert result.output_tokens == 4


def test_ollama_rejects_unavailable_model_before_generation():
    session = FakeSession([FakeHttpResponse({"models": [{"name": "other"}]})])
    provider = OllamaProvider("http://localhost:11434", "research:latest", session=session)
    with pytest.raises(ProviderUnavailable, match="configured model"):
        provider.generate(make_request())
    assert [call[0] for call in session.calls] == ["GET"]


@pytest.mark.parametrize("url", [
    "https://example.com:11434", "http://user@localhost:11434",
    "http://localhost:11434/path?key=secret",
])
def test_ollama_rejects_public_or_credential_bearing_base_urls(url):
    with pytest.raises(ProviderConfigurationError):
        OllamaProvider(url, "research:latest", session=FakeSession([]))


def test_ollama_accepts_private_literal_and_explicit_private_hostname():
    OllamaProvider("http://10.0.0.2:11434", "research:latest", session=FakeSession([]))
    OllamaProvider("http://ollama:11434", "research:latest", session=FakeSession([]), allowed_private_hosts={"ollama"})


def test_ollama_rejects_unsafe_model_identifier_without_leak():
    with pytest.raises(ProviderConfigurationError) as raised:
        OllamaProvider("http://localhost:11434", "marker\ud800", session=FakeSession([]))
    assert "marker" not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("timeout", [(True, 1), (float("nan"), 1), (1, float("inf"))])
def test_ollama_rejects_nonfinite_or_boolean_timeout_bounds(timeout):
    with pytest.raises(ProviderConfigurationError):
        OllamaProvider(
            "http://localhost:11434",
            "research:latest",
            session=FakeSession([]),
            timeout=timeout,
        )


def test_ollama_closes_owned_session_and_bounds_response_bytes():
    http_response = FakeHttpResponse({}, chunks=[b"x" * 40])
    session = FakeSession([http_response])
    provider = OllamaProvider(
        "http://localhost:11434", "research:latest",
        session_factory=lambda: session, max_response_bytes=32,
    )
    with pytest.raises(ProviderValidationError) as raised:
        provider.generate(make_request())
    assert raised.value.issues == (ValidationIssue("$", "response_too_large"),)
    assert http_response.closed is True
    provider.close()
    assert session.closed is True


def test_ollama_schema_errors_are_typed_and_redacted():
    secret = "portfolio-secret-value"
    session = FakeSession([
        FakeHttpResponse({"models": [{"name": "research:latest"}]}),
        FakeHttpResponse({"response": '{"wrong":"' + secret + '"}'}),
    ])
    provider = OllamaProvider("http://localhost:11434", "research:latest", session=session)
    with pytest.raises(ProviderValidationError) as raised:
        provider.generate(make_request())
    assert secret not in str(raised.value)


@pytest.mark.parametrize("unsafe_json", ['{"answer":"marker\\ud800"}', '{"answer":"marker\\u0000"}'])
def test_ollama_rejects_escaped_unsafe_model_text_without_leak(unsafe_json):
    session = FakeSession([
        FakeHttpResponse({"models": [{"name": "research:latest"}]}),
        FakeHttpResponse({"response": unsafe_json}),
    ])
    provider = OllamaProvider("http://localhost:11434", "research:latest", session=session)
    with pytest.raises(ProviderValidationError) as raised:
        provider.generate(make_request())
    assert raised.value.issues == (ValidationIssue("$", "invalid_text"),)
    assert "marker" not in str(raised.value)
    assert raised.value.__cause__ is None


class FakeResponses:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcome.pop(0) if isinstance(self.outcome, list) else self.outcome
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeOpenAIClient:
    def __init__(self, outcome):
        self.responses = FakeResponses(outcome)


def openai_result(*, output='{"answer":"supported"}', input_tokens=4, output_tokens=3, reasoning_tokens=1):
    return SimpleNamespace(
        output_text=output,
        usage=SimpleNamespace(
            input_tokens=input_tokens, output_tokens=output_tokens,
            output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        ),
    )


def make_openai_provider(migrated_store, client, *, clock=lambda: utc(2026, 8, 24)):
    prices = PriceTable(
        effective_until=date(2026, 12, 31),
        prices={"gpt-test": ModelPrice(Decimal("1"), Decimal("2"))},
        clock=clock,
    )
    ledger = BudgetLedger(migrated_store, Decimal("4"), Decimal("5"))
    return OpenAIProvider(
        client=client,
        routes={AgentRole.EVENT_SCOUT: ModelRoute("gpt-test", ReasoningEffort.LOW)},
        budget=ledger, prices=prices, token_estimator=lambda _: 10,
        clock=clock, monotonic=lambda: 1.0,
    ), ledger


def test_openai_route_rejects_unsafe_model_identifier_without_leak():
    with pytest.raises(ProviderConfigurationError) as raised:
        ModelRoute("marker\ud800", ReasoningEffort.LOW)
    assert "marker" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_openai_uses_request_reasoning_effort_and_reconciles(migrated_store):
    client = FakeOpenAIClient(openai_result())
    provider, ledger = make_openai_provider(migrated_store, client)
    request = make_request(effort=ReasoningEffort.HIGH)
    result = provider.generate(request)
    assert client.responses.calls[0] == {
        "model": "gpt-test",
        "instructions": "Cite the supplied evidence.",
        "input": request.provider_input,
        "max_output_tokens": 80,
        "reasoning": {"effort": "high"},
        "store": False,
        "text": {"format": {
            "type": "json_schema", "name": "ResearchOutput",
            "schema": ResearchOutput.model_json_schema(), "strict": True,
        }},
    }
    assert result.data == ResearchOutput(answer="supported")
    assert result.reasoning_tokens == 1
    assert ledger.month_total(2026, 8) == Decimal("0.000010")


class FakeRemoteError(Exception):
    def __init__(self, status_code, detail="remote failure"):
        super().__init__(detail)
        self.status_code = status_code


def test_openai_auth_failure_releases_reservation_and_redacts(migrated_store):
    secret = "configured-key-secret"
    provider, ledger = make_openai_provider(migrated_store, FakeOpenAIClient(FakeRemoteError(401, secret)))
    with pytest.raises(ProviderAuthenticationError) as raised:
        provider.generate(make_request())
    assert secret not in str(raised.value)
    assert ledger.month_total(2026, 8) == Decimal("0")


def test_openai_timeout_keeps_pessimistic_reservation_counted(migrated_store):
    provider, ledger = make_openai_provider(migrated_store, FakeOpenAIClient(TimeoutError("prompt secret")))
    with pytest.raises(ProviderUnavailable) as raised:
        provider.generate(make_request())
    assert "prompt secret" not in str(raised.value)
    assert ledger.month_total(2026, 8) == Decimal("0.000170")


def test_openai_schema_failure_reconciles_reported_usage(migrated_store):
    provider, ledger = make_openai_provider(migrated_store, FakeOpenAIClient(openai_result(output='{"wrong":1}')))
    with pytest.raises(ProviderValidationError):
        provider.generate(make_request())
    assert ledger.month_total(2026, 8) == Decimal("0.000010")


@pytest.mark.parametrize("unsafe_json", ['{"answer":"marker\\ud800"}', '{"answer":"marker\\u0000"}'])
def test_openai_rejects_escaped_unsafe_model_text_and_reconciles(
    migrated_store, unsafe_json
):
    provider, ledger = make_openai_provider(
        migrated_store, FakeOpenAIClient(openai_result(output=unsafe_json))
    )
    with pytest.raises(ProviderValidationError) as raised:
        provider.generate(make_request())
    assert raised.value.issues == (ValidationIssue("$", "invalid_text"),)
    assert "marker" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert ledger.month_total(2026, 8) == Decimal("0.000010")


def test_router_budgets_each_openai_validation_attempt_and_redacts_feedback(migrated_store):
    raw_secret = "raw-provider-secret"
    client = FakeOpenAIClient(
        [
            openai_result(output='{"wrong":"' + raw_secret + '"}'),
            openai_result(output='{"answer":"supported"}'),
        ]
    )
    external, ledger = make_openai_provider(migrated_store, client)
    router = ProviderRouter(
        make_config(external=True), external, StubProvider("ollama", [])
    )

    result = router.generate(make_request())

    assert result.provider == "openai"
    assert len(client.responses.calls) == 2
    assert ledger.month_total(2026, 8) == Decimal("0.000020")
    assert raw_secret not in client.responses.calls[1]["input"]
    assert "validation_feedback" in client.responses.calls[1]["input"]


def test_missing_optional_openai_sdk_fails_with_typed_error(migrated_store):
    provider, _ = make_openai_provider(migrated_store, None)
    provider._client_factory = lambda: (_ for _ in ()).throw(ImportError())
    with pytest.raises(ProviderConfigurationError, match="SDK"):
        provider.generate(make_request())


def test_invalid_injected_openai_client_fails_before_budget_reservation(migrated_store):
    provider, ledger = make_openai_provider(migrated_store, object())

    with pytest.raises(ProviderConfigurationError, match="Responses API"):
        provider.generate(make_request())

    assert ledger.month_total(2026, 8) == Decimal("0")


def test_no_api_key_routes_every_role_to_ollama():
    local = StubProvider("ollama", [])
    router = ProviderRouter(make_config(external=False), external=None, ollama=local)
    for role in AgentRole:
        assert router.provider_for(role, run_id="run-1").name == "ollama"


@pytest.mark.parametrize(
    "operation",
    [
        lambda router, value: router.provider_for(AgentRole.EVENT_SCOUT, run_id=value),
        lambda router, value: router.external_disabled_reason(value),
        lambda router, value: router.clear_run(value),
    ],
)
def test_router_rejects_unsafe_run_identifiers_without_leak(operation):
    router = ProviderRouter(
        make_config(external=False), None, StubProvider("ollama", [])
    )
    with pytest.raises(ProviderRequestError) as raised:
        operation(router, "marker\ud800")
    assert "marker" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_authentication_failure_disables_external_for_that_run_only():
    external = StubProvider("openai", [
        ProviderAuthenticationError("redacted"), response("openai", run_id="run-2")
    ])
    local = StubProvider("ollama", [response("ollama"), response("ollama")])
    router = ProviderRouter(make_config(external=True), external, local)
    fallback = router.generate(make_request(run_id="run-1"))
    again = router.generate(make_request(run_id="run-1"))
    other_run = router.generate(make_request(run_id="run-2"))
    assert fallback.fallback_reason == "authentication"
    assert again.fallback_reason == "authentication"
    assert other_run.provider == "openai"
    assert router.external_disabled_reason("run-1") == "authentication"
    assert router.external_disabled_reason("run-2") is None
    assert len(external.calls) == 2


def test_external_outage_falls_back_only_when_policy_permits():
    external = StubProvider("openai", [ProviderUnavailable("timeout")])
    router = ProviderRouter(make_config(external=True), external, StubProvider("ollama", [response("ollama")]))
    result = router.generate(make_request(policy=FallbackPolicy.OUTAGES))
    assert result.provider == "ollama"
    assert result.fallback_reason == "external_unavailable"


def test_frontier_review_defers_when_fallback_is_not_allowed():
    router = ProviderRouter(
        make_config(external=True),
        StubProvider("openai", [ProviderUnavailable("timeout")]),
        StubProvider("ollama", []),
    )
    with pytest.raises(TaskDeferred, match="external provider unavailable"):
        router.generate(make_request(role=AgentRole.SKEPTICAL_REVIEWER, policy=FallbackPolicy.NEVER))


def test_validation_failure_does_not_fallback_without_explicit_policy():
    local = StubProvider("ollama", [])
    router = ProviderRouter(
        make_config(external=True),
        StubProvider("openai", [ProviderValidationError(), ProviderValidationError()]),
        local,
    )
    with pytest.raises(ProviderValidationError):
        router.generate(make_request(policy=FallbackPolicy.OUTAGES))
    assert local.calls == []


def test_explicit_any_failure_policy_allows_validation_fallback():
    router = ProviderRouter(
        make_config(external=True),
        StubProvider("openai", [ProviderValidationError(), ProviderValidationError()]),
        StubProvider("ollama", [response("ollama")]),
    )
    assert router.generate(make_request(policy=FallbackPolicy.ANY_FAILURE)).fallback_reason == "external_validation"


def test_external_validation_retries_same_provider_once_with_safe_feedback():
    external = StubProvider(
        "openai",
        [
            ProviderValidationError(issues=(ValidationIssue("answer", "missing"),)),
            response("openai"),
        ],
    )
    router = ProviderRouter(make_config(external=True), external, StubProvider("ollama", []))

    result = router.generate(make_request())

    assert result.provider == "openai"
    assert len(external.calls) == 2
    assert external.calls[0].validation_feedback == ()
    assert external.calls[1].validation_feedback == (ValidationIssue("answer", "missing"),)
    assert "Revenue rose" in external.calls[1].provider_input
    assert '"code":"missing"' in external.calls[1].provider_input


def test_two_external_validation_failures_then_explicitly_fall_back_to_ollama():
    external = StubProvider(
        "openai",
        [ProviderValidationError(), ProviderValidationError()],
    )
    local = StubProvider("ollama", [response("ollama")])
    result = ProviderRouter(make_config(external=True), external, local).generate(
        make_request(policy=FallbackPolicy.ANY_FAILURE)
    )
    assert len(external.calls) == 2
    assert len(local.calls) == 1
    assert result.fallback_reason == "external_validation"


def test_two_external_validation_failures_defer_when_local_is_forbidden():
    external = StubProvider(
        "openai",
        [ProviderValidationError(), ProviderValidationError()],
    )
    router = ProviderRouter(make_config(external=True), external, StubProvider("ollama", []))
    with pytest.raises(TaskDeferred, match="validation"):
        router.generate(make_request(policy=FallbackPolicy.NEVER))
    assert len(external.calls) == 2


def test_local_validation_retries_same_ollama_provider_once():
    local = StubProvider(
        "ollama",
        [ProviderValidationError(), response("ollama")],
    )
    result = ProviderRouter(make_config(external=False), None, local).generate(make_request())
    assert result.provider == "ollama"
    assert len(local.calls) == 2
    assert local.calls[1].validation_feedback


def test_second_local_validation_failure_stops_without_another_fallback_chain():
    local = StubProvider(
        "ollama",
        [
            ProviderValidationError(
                issues=(ValidationIssue("answer", "missing"),)
            ),
            ProviderValidationError(
                issues=(ValidationIssue("answer", "string_type"),)
            ),
        ],
    )
    router = ProviderRouter(make_config(external=False), None, local)

    with pytest.raises(ProviderValidationError) as raised:
        router.generate(make_request(policy=FallbackPolicy.ANY_FAILURE))

    assert raised.value.issues == (ValidationIssue("answer", "string_type"),)
    assert len(local.calls) == 2
    assert local.calls[1].validation_feedback == (
        ValidationIssue("answer", "missing"),
    )


@pytest.mark.parametrize("failure", [ProviderAuthenticationError(), ProviderUnavailable()])
def test_auth_and_outage_failures_do_not_trigger_validation_retry(failure):
    external = StubProvider("openai", [failure])
    router = ProviderRouter(
        make_config(external=True), external, StubProvider("ollama", [response("ollama")])
    )
    router.generate(make_request())
    assert len(external.calls) == 1


def test_same_run_auth_disable_is_atomic_under_concurrency():
    external = StubProvider("openai", [ProviderAuthenticationError("invalid")])
    router = ProviderRouter(make_config(external=True), external, StubProvider("ollama", []))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: router.generate(make_request()), range(8)))
    assert len(external.calls) == 1
    assert all(result.provider == "ollama" for result in results)
    assert all(result.fallback_reason == "authentication" for result in results)


def test_same_run_validation_retries_do_not_leak_feedback_between_concurrent_calls():
    external = StubProvider(
        "openai",
        [ProviderValidationError(), ProviderValidationError(), response("openai")],
    )
    local = StubProvider("ollama", [response("ollama")])
    router = ProviderRouter(make_config(external=True), external, local)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: router.generate(
                    make_request(policy=FallbackPolicy.ANY_FAILURE)
                ),
                range(2),
            )
        )

    assert {result.provider for result in results} == {"openai", "ollama"}
    assert len(external.calls) == 3
    assert external.calls[0].validation_feedback == ()
    assert external.calls[1].validation_feedback
    assert external.calls[2].validation_feedback == ()
    assert len(local.calls) == 1
