"""Environment-driven configuration for the investment research pipeline."""

import os
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .models import InferenceMode


class ResearchConfigError(ValueError):
    """Raised when research configuration cannot be parsed or validated."""


def _read_secret(name: str) -> str | None:
    """Read a secret file when configured, otherwise use the direct value."""
    file_value = os.getenv(f"{name}_FILE")
    if file_value:
        secret_path = Path(file_value)
        try:
            value = secret_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise ResearchConfigError(
                f"Unable to read {name} secret file {secret_path}: {exc}"
            ) from exc
        return value or None

    value = os.getenv(name)
    if value is None:
        return None
    return value.strip() or None


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ResearchConfigError(
        f"{name} must be a boolean (true/false, yes/no, on/off, or 1/0); got {value!r}"
    )


def _get_decimal(name: str, default: str) -> Decimal:
    value = os.getenv(name, default)
    try:
        parsed = Decimal(value.strip())
    except (AttributeError, InvalidOperation) as exc:
        raise ResearchConfigError(f"{name} must be a decimal; got {value!r}") from exc
    if not parsed.is_finite():
        raise ResearchConfigError(f"{name} must be a finite decimal; got {value!r}")
    return parsed


def _get_text(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() or default


@dataclass(frozen=True)
class ResearchConfig:
    """Typed settings required by research foundation components."""

    enabled: bool
    data_dir: Path
    openai_api_key: str | None = field(repr=False)
    inference_mode: InferenceMode
    ollama_base_url: str
    ollama_model: str
    budget_soft_usd: Decimal
    budget_hard_usd: Decimal

    def __post_init__(self) -> None:
        self.validate()

    @property
    def database_path(self) -> Path:
        return self.data_dir / "research.db"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def report_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def backup_dir(self) -> Path:
        return self.data_dir / "backups"

    @classmethod
    def from_env(cls) -> "ResearchConfig":
        """Build configuration from the process environment without loading `.env`."""
        openai_key = _read_secret("OPENAI_API_KEY")
        return cls(
            enabled=_get_bool("RESEARCH_ENABLED", False),
            data_dir=Path(_get_text("RESEARCH_DATA_DIR", "research_data")),
            openai_api_key=openai_key,
            inference_mode=(
                InferenceMode.EXTERNAL if openai_key else InferenceMode.LOCAL_ONLY
            ),
            ollama_base_url=_get_text(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ),
            ollama_model=_get_text("OLLAMA_RESEARCH_MODEL", "llama3.1:8b"),
            budget_soft_usd=_get_decimal("MODEL_BUDGET_SOFT_USD", "4.00"),
            budget_hard_usd=_get_decimal("MODEL_BUDGET_HARD_USD", "5.00"),
        )

    def validate(self) -> None:
        """Reject unsafe budget thresholds and inconsistent inference routing."""
        if not isinstance(self.data_dir, Path):
            raise ResearchConfigError("data_dir must be Path")
        if not str(self.data_dir).strip():
            raise ResearchConfigError("data_dir must be a non-empty path")
        for field_name, value in (
            ("ollama_base_url", self.ollama_base_url),
            ("ollama_model", self.ollama_model),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ResearchConfigError(f"{field_name} must be non-empty")

        for field_name, value in (
            ("budget_soft_usd", self.budget_soft_usd),
            ("budget_hard_usd", self.budget_hard_usd),
        ):
            if not isinstance(value, Decimal):
                raise ResearchConfigError(f"{field_name} must be Decimal")
            if not value.is_finite():
                raise ResearchConfigError(f"{field_name} must be finite")

        if self.budget_soft_usd < 0 or self.budget_hard_usd < 0:
            raise ResearchConfigError("Model budget limits must be non-negative")
        if self.budget_soft_usd >= self.budget_hard_usd:
            raise ResearchConfigError(
                "MODEL_BUDGET_SOFT_USD must be less than MODEL_BUDGET_HARD_USD"
            )
        if self.budget_hard_usd > Decimal("5.00"):
            raise ResearchConfigError("MODEL_BUDGET_HARD_USD must not exceed 5.00")

        if not isinstance(self.inference_mode, InferenceMode):
            raise ResearchConfigError("inference_mode must be InferenceMode")
        if self.openai_api_key is not None:
            if not isinstance(self.openai_api_key, str):
                raise ResearchConfigError("openai_api_key must be str or None")
            if not self.openai_api_key.strip():
                raise ResearchConfigError(
                    "openai_api_key must be non-empty when provided"
                )
        if self.openai_api_key is None and self.inference_mode is not InferenceMode.LOCAL_ONLY:
            raise ResearchConfigError(
                "inference_mode must be local_only when OPENAI_API_KEY is absent"
            )
        if self.openai_api_key is not None and self.inference_mode is not InferenceMode.EXTERNAL:
            raise ResearchConfigError(
                "inference_mode must be external when OPENAI_API_KEY is configured"
            )
