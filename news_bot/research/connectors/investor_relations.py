"""Strictly allowlisted investor-relations release adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from urllib.parse import unquote, urlsplit

from ...article_extractor import _fetch_and_extract
from ..evidence import DocumentInput, canonicalize_url
from .base import ConnectorError, NormalizedResearchDocument


@dataclass(frozen=True)
class InvestorRelationsConfig:
    """Issuer-scoped HTTPS URL prefixes approved for release ingestion."""

    allowed_prefixes: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_prefixes, Mapping) or not self.allowed_prefixes:
            raise ValueError("allowed_prefixes must map issuers to URL prefixes")
        normalized: dict[str, tuple[str, ...]] = {}
        for issuer, prefixes in self.allowed_prefixes.items():
            if not isinstance(issuer, str) or not issuer.strip():
                raise ValueError("issuer must be nonblank")
            if not isinstance(prefixes, tuple) or not prefixes:
                raise ValueError("issuer allowlist must be a nonempty tuple")
            checked: list[str] = []
            for prefix in prefixes:
                parsed = urlsplit(prefix)
                canonical = canonicalize_url(prefix)
                canonical_parts = urlsplit(canonical)
                if (
                    parsed.username is not None
                    or parsed.password is not None
                    or canonical_parts.scheme != "https"
                    or canonical_parts.port not in (None, 443)
                    or canonical_parts.query
                    or not canonical_parts.path.endswith("/")
                ):
                    raise ValueError("IR prefixes must be credential-free HTTPS directories")
                checked.append(canonical)
            normalized[issuer.strip().upper()] = tuple(checked)
        object.__setattr__(self, "allowed_prefixes", MappingProxyType(normalized))


class InvestorRelationsConnector:
    """Adapt the established article extractor behind issuer URL policy."""

    name = "investor_relations"

    def __init__(
        self,
        config: InvestorRelationsConfig,
        *,
        extractor: Callable[[str], tuple[str | None, bool]] = _fetch_and_extract,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(config, InvestorRelationsConfig):
            raise TypeError("config must be InvestorRelationsConfig")
        self.config = config
        self._extractor = extractor
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _approved_url(self, issuer: str, url: str) -> str:
        issuer_key = issuer.strip().upper() if isinstance(issuer, str) else ""
        prefixes = self.config.allowed_prefixes.get(issuer_key, ())
        parsed_input = urlsplit(url)
        canonical = canonicalize_url(url)
        candidate = urlsplit(canonical)
        if parsed_input.username is not None or parsed_input.password is not None:
            raise ValueError("investor-relations URL is not allowlisted")
        decoded_path = candidate.path
        for _ in range(3):
            further_decoded = unquote(decoded_path)
            if further_decoded == decoded_path:
                break
            decoded_path = further_decoded
        if "\\" in decoded_path or any(
            segment in {".", ".."} for segment in decoded_path.split("/")
        ):
            raise ValueError("investor-relations URL is not allowlisted")
        for prefix in prefixes:
            approved = urlsplit(prefix)
            if (
                candidate.scheme == approved.scheme
                and candidate.hostname == approved.hostname
                and candidate.port == approved.port
                and candidate.path.startswith(approved.path)
            ):
                return canonical
        raise ValueError("investor-relations URL is not allowlisted")

    def fetch_release(
        self, issuer: str, url: str, *, published_at: datetime
    ) -> NormalizedResearchDocument:
        canonical = self._approved_url(issuer, url)
        try:
            content, blocked = self._extractor(canonical)
        except Exception:
            raise ConnectorError(
                self.name, retryable=True, diagnostic_code="extraction_failed"
            ) from None
        if blocked:
            raise ConnectorError(
                self.name, retryable=False, diagnostic_code="content_blocked"
            )
        if not content or not content.strip():
            raise ConnectorError(
                self.name, retryable=True, diagnostic_code="content_unavailable"
            )
        issuer_key = issuer.strip().upper()
        return NormalizedResearchDocument(
            evidence=DocumentInput(
                source_type="investor_relations",
                url=canonical,
                publisher=f"{issuer_key} Investor Relations",
                published_at=published_at,
                retrieved_at=self._clock(),
                content=content,
            ),
            tags=("investor-relations", issuer_key),
        )
