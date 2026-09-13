"""Environment-driven configuration for the investment research pipeline."""

import os
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .models import InferenceMode


class ResearchConfigError(ValueError):
    """Raised when research configuration cannot be parsed or validated."""


_CRON_ATOM = re.compile(r"[A-Za-z0-9*/?,\-]+")


def _validate_cron_part(
    expression: str,
    *,
    minimum: int,
    maximum: int,
    names: frozenset[str] = frozenset(),
) -> None:
    def value(token: str) -> None:
        if token.lower() in names:
            return
        try:
            number = int(token)
        except ValueError as exc:
            raise ResearchConfigError("cron field contains an invalid value") from exc
        if not minimum <= number <= maximum:
            raise ResearchConfigError("cron field value is out of range")

    for item in expression.split(","):
        pieces = item.split("/")
        if len(pieces) > 2:
            raise ResearchConfigError("cron field contains too many steps")
        base = pieces[0]
        if len(pieces) == 2:
            try:
                step = int(pieces[1])
            except ValueError as exc:
                raise ResearchConfigError("cron step must be an integer") from exc
            if step <= 0:
                raise ResearchConfigError("cron step must be positive")
        if base == "*":
            continue
        endpoints = base.split("-")
        if len(endpoints) > 2 or any(not endpoint for endpoint in endpoints):
            raise ResearchConfigError("cron range is invalid")
        for endpoint in endpoints:
            value(endpoint)


def _validate_cron_fields(parts: tuple[str, ...]) -> None:
    month_names = frozenset(
        {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}
    )
    weekday_names = frozenset({"mon", "tue", "wed", "thu", "fri", "sat", "sun"})
    for expression, minimum, maximum, names in (
        (parts[0], 0, 59, frozenset()),
        (parts[1], 0, 23, frozenset()),
        (parts[2], 1, 31, frozenset()),
        (parts[3], 1, 12, month_names),
        (parts[4], 0, 6, weekday_names),
    ):
        _validate_cron_part(
            expression, minimum=minimum, maximum=maximum, names=names
        )


@dataclass(frozen=True)
class CronSchedule:
    """Validated five-field cron cadence in UTC."""

    minute: str
    hour: str
    day: str
    month: str
    day_of_week: str

    @classmethod
    def from_crontab(
        cls, value: str, *, setting: str = "cron setting"
    ) -> "CronSchedule":
        if not isinstance(value, str):
            raise ResearchConfigError(f"{setting} must be a five-field cron expression")
        parts = value.split()
        if len(parts) != 5 or any(
            _CRON_ATOM.fullmatch(part) is None for part in parts
        ):
            raise ResearchConfigError(f"{setting} must be a five-field cron expression")
        try:
            _validate_cron_fields(tuple(parts))
        except ResearchConfigError as exc:
            raise ResearchConfigError(f"{setting} is invalid") from exc
        return cls(*parts)

    def as_kwargs(self) -> dict[str, str]:
        """Return APScheduler-compatible cron trigger keyword arguments."""
        return {
            "minute": self.minute,
            "hour": self.hour,
            "day": self.day,
            "month": self.month,
            "day_of_week": self.day_of_week,
        }


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


def research_enabled_from_env() -> bool:
    """Read only the opt-in switch, leaving disabled legacy startup untouched."""
    return _get_bool("RESEARCH_ENABLED", False)


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
    daily_schedule: CronSchedule = CronSchedule.from_crontab("0 7 * * 1-5")
    weekly_schedule: CronSchedule = CronSchedule.from_crontab("0 8 * * 1")
    monthly_schedule: CronSchedule = CronSchedule.from_crontab("0 9 1 * *")
    ibkr_flex_token: str | None = field(default=None, repr=False)
    ibkr_flex_query_id: str | None = field(default=None, repr=False)
    ibkr_flex_account_salt: str | None = field(default=None, repr=False)

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
            daily_schedule=CronSchedule.from_crontab(
                _get_text("RESEARCH_DAILY_SCHEDULE", "0 7 * * 1-5"),
                setting="RESEARCH_DAILY_SCHEDULE",
            ),
            weekly_schedule=CronSchedule.from_crontab(
                _get_text("RESEARCH_WEEKLY_SCHEDULE", "0 8 * * 1"),
                setting="RESEARCH_WEEKLY_SCHEDULE",
            ),
            monthly_schedule=CronSchedule.from_crontab(
                _get_text("RESEARCH_MONTHLY_SCHEDULE", "0 9 1 * *"),
                setting="RESEARCH_MONTHLY_SCHEDULE",
            ),
            ibkr_flex_token=_read_secret("IBKR_FLEX_TOKEN"),
            ibkr_flex_query_id=_read_secret("IBKR_FLEX_QUERY_ID"),
            ibkr_flex_account_salt=_read_secret("IBKR_FLEX_ACCOUNT_SALT"),
        )

    def validate(self) -> None:
        """Reject unsafe budget thresholds and inconsistent inference routing."""
        if not isinstance(self.data_dir, Path):
            raise ResearchConfigError("data_dir must be Path")
        if not str(self.data_dir).strip():
            raise ResearchConfigError("data_dir must be a non-empty path")
        for field_name, value in (
            ("daily_schedule", self.daily_schedule),
            ("weekly_schedule", self.weekly_schedule),
            ("monthly_schedule", self.monthly_schedule),
        ):
            if not isinstance(value, CronSchedule):
                raise ResearchConfigError(f"{field_name} must be CronSchedule")
        flex_credentials = (
            self.ibkr_flex_token,
            self.ibkr_flex_query_id,
            self.ibkr_flex_account_salt,
        )
        if any(value is not None for value in flex_credentials) and not all(
            isinstance(value, str) and bool(value.strip())
            for value in flex_credentials
        ):
            raise ResearchConfigError(
                "IBKR Flex token, query ID, and account salt must be configured together"
            )
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
