"""Bounded Ollama structured-generation adapter for local/private runtimes."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import time
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import urlsplit

import requests
from pydantic import ValidationError

from ..models import InferenceMode
from .base import (
    ModelRequest,
    ModelResponse,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderNonRetryableError,
    ProviderRateLimitError,
    ProviderUnavailable,
    ProviderValidationError,
    ValidationIssue,
    _safe_text,
    validated_provider_json,
    validation_issues,
)


_DEFAULT_TIMEOUT = (3.05, 120.0)
_DEFAULT_MAX_RESPONSE_BYTES = 4 * 1024 * 1024


def _safe_private_base_url(base_url: str, allowed_private_hosts: Iterable[str]) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ProviderConfigurationError("Ollama base URL must be nonblank")
    try:
        parsed = urlsplit(base_url.strip())
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        raise ProviderConfigurationError("Ollama base URL is invalid") from None
    if parsed.scheme not in {"http", "https"} or not host or port is None:
        raise ProviderConfigurationError("Ollama base URL requires HTTP(S), host, and port")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ProviderConfigurationError("Ollama base URL cannot contain credentials or parameters")
    if parsed.path not in {"", "/"}:
        raise ProviderConfigurationError("Ollama base URL cannot contain an API path")

    allowed = {item.strip().lower() for item in allowed_private_hosts if item.strip()}
    is_private = host == "localhost" or host in allowed
    if not is_private:
        try:
            address = ipaddress.ip_address(host)
            is_private = address.is_private or address.is_loopback or address.is_link_local
        except ValueError:
            is_private = False
    if not is_private:
        raise ProviderConfigurationError("Ollama base URL must use a loopback or allowed private host")
    bracketed_host = f"[{host}]" if ":" in host else host
    return f"{parsed.scheme}://{bracketed_host}:{port}"


class OllamaProvider:
    """Generate schema-constrained local output without proxy or redirect escape."""

    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        session: Any | None = None,
        session_factory: Callable[[], Any] = requests.Session,
        timeout: tuple[float, float] = _DEFAULT_TIMEOUT,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
        allowed_private_hosts: Iterable[str] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = _safe_private_base_url(base_url, allowed_private_hosts)
        safe_model = _safe_text(
            "Ollama model",
            model,
            allow_multiline=False,
            nonblank=True,
            error_type=ProviderConfigurationError,
        ).strip()
        if (
            not isinstance(timeout, tuple)
            or len(timeout) != 2
            or any(
                type(value) not in {int, float}
                or not math.isfinite(value)
                or value <= 0
                for value in timeout
            )
        ):
            raise ProviderConfigurationError("Ollama timeout must contain positive bounds")
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ProviderConfigurationError("Ollama response bound must be positive")
        if not callable(clock):
            raise ProviderConfigurationError("Ollama clock must be callable")
        self.model = safe_model
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.clock = clock
        self._owns_session = session is None
        self._session = session if session is not None else session_factory()
        if self._owns_session and hasattr(self._session, "trust_env"):
            self._session.trust_env = False

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "OllamaProvider":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be ModelRequest")
        started = self.clock()
        tags = self._request_json("GET", "/api/tags")
        models = tags.get("models")
        if not isinstance(models, list) or self.model not in {
            entry.get("name") for entry in models if isinstance(entry, dict)
        }:
            raise ProviderUnavailable("configured model is unavailable in Ollama")
        payload = {
            "model": self.model,
            "system": request.system_prompt,
            "prompt": request.provider_input,
            "stream": False,
            "format": request.schema_payload(),
            "options": {"num_predict": request.max_output_tokens},
        }
        result = self._request_json("POST", "/api/generate", json_body=payload)
        raw = result.get("response")
        if not isinstance(raw, str):
            raise ProviderValidationError(
                "Ollama response did not contain structured text",
                issues=(ValidationIssue("$", "output_missing"),),
            )
        try:
            raw_bytes = raw.encode("utf-8", errors="strict")
        except UnicodeError:
            raise ProviderValidationError(
                "Ollama response failed the requested schema",
                issues=(ValidationIssue("$", "invalid_utf8"),),
            ) from None
        try:
            parsed_json = json.loads(raw)
        except json.JSONDecodeError:
            raise ProviderValidationError(
                "Ollama response failed the requested schema",
                issues=(ValidationIssue("$", "json_invalid"),),
            ) from None
        parsed_json = validated_provider_json(parsed_json)
        try:
            parsed = request.output_schema.model_validate(parsed_json)
        except ValidationError as exc:
            raise ProviderValidationError(
                "Ollama response failed the requested schema",
                issues=validation_issues(exc),
            ) from None
        return ModelResponse(
            data=parsed,
            raw_response_hash=hashlib.sha256(raw_bytes).hexdigest(),
            input_tokens=_nonnegative_count(result.get("prompt_eval_count")),
            output_tokens=_nonnegative_count(result.get("eval_count")),
            reasoning_tokens=_nonnegative_count(result.get("reasoning_eval_count", 0)),
            model=self.model,
            latency_ms=max(0, round((self.clock() - started) * 1000)),
            provider=self.name,
            inference_mode=InferenceMode.LOCAL_ONLY,
            run_id=request.run_id,
        )

    def _request_json(
        self, method: str, path: str, *, json_body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "allow_redirects": False,
            "stream": True,
        }
        if json_body is not None:
            kwargs["json"] = json_body
        response = None
        try:
            function = self._session.get if method == "GET" else self._session.post
            response = function(f"{self.base_url}{path}", **kwargs)
            response.raise_for_status()
            raw = bytearray()
            for chunk in response.iter_content(chunk_size=16 * 1024):
                if not chunk:
                    continue
                raw.extend(chunk)
                if len(raw) > self.max_response_bytes:
                    raise ProviderValidationError(
                        issues=(ValidationIssue("$", "response_too_large"),)
                    )
            decoded = bytes(raw).decode("utf-8", errors="strict")
            value = json.loads(decoded)
            if not isinstance(value, dict):
                raise ProviderValidationError(
                    issues=(ValidationIssue("$", "object_required"),)
                )
            return value
        except ProviderValidationError:
            raise
        except requests.Timeout:
            raise ProviderUnavailable("Ollama request timed out") from None
        except requests.ConnectionError:
            raise ProviderUnavailable("Ollama is unavailable") from None
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {401, 403}:
                raise ProviderAuthenticationError("Ollama authentication failed") from None
            if status == 429:
                raise ProviderRateLimitError("Ollama rate limit reached") from None
            if isinstance(status, int) and status >= 500:
                raise ProviderUnavailable("Ollama is unavailable") from None
            raise ProviderNonRetryableError("Ollama rejected the request") from None
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise ProviderValidationError(
                issues=(ValidationIssue("$", "json_invalid"),)
            ) from None
        except requests.RequestException:
            raise ProviderUnavailable("Ollama request failed") from None
        finally:
            if response is not None:
                response.close()


def _nonnegative_count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ProviderValidationError(
            issues=(ValidationIssue("usage", "invalid"),)
        )
    return value
