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
_INDUSTRY_KEY = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def _validate_cron_part(
    expression: str,
    *,
    minimum: int,
    maximum: int,
    names: tuple[str, ...] = (),
    allow_numbers: bool = True,
) -> None:
    def value(token: str) -> int:
        lowered = token.lower()
        if lowered in names:
            return minimum + names.index(lowered)
        if not allow_numbers:
            raise ResearchConfigError("cron field requires named values")
        try:
            number = int(token)
        except ValueError as exc:
            raise ResearchConfigError("cron field contains an invalid value") from exc
        if not minimum <= number <= maximum:
            raise ResearchConfigError("cron field value is out of range")
        return number

    for item in expression.split(","):
        pieces = item.split("/")
        if len(pieces) > 2:
            raise ResearchConfigError("cron field contains too many steps")
        base = pieces[0]
        step: int | None = None
        if len(pieces) == 2:
            try:
                step = int(pieces[1])
            except ValueError as exc:
                raise ResearchConfigError("cron step must be an integer") from exc
            if step <= 0:
                raise ResearchConfigError("cron step must be positive")
        if base == "*":
            if step is not None and step > maximum - minimum:
                raise ResearchConfigError("cron step is too large")
            continue
        endpoints = base.split("-")
        if len(endpoints) > 2 or any(not endpoint for endpoint in endpoints):
            raise ResearchConfigError("cron range is invalid")
        values = tuple(value(endpoint) for endpoint in endpoints)
        if len(values) == 2 and values[0] > values[1]:
            raise ResearchConfigError("cron range is reversed")
        available_span = (
            maximum - values[0]
            if len(values) == 1
            else values[1] - values[0]
        )
        if step is not None and step > available_span:
            raise ResearchConfigError("cron step is too large")


def _validate_cron_fields(parts: tuple[str, ...]) -> None:
    month_names = (
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    )
    weekday_names = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
    for expression, minimum, maximum, names, allow_numbers in (
        (parts[0], 0, 59, (), True),
        (parts[1], 0, 23, (), True),
        (parts[2], 1, 31, (), True),
        (parts[3], 1, 12, month_names, True),
        (parts[4], 0, 6, weekday_names, False),
    ):
        _validate_cron_part(
            expression,
            minimum=minimum,
            maximum=maximum,
            names=names,
            allow_numbers=allow_numbers,
        )


@dataclass(frozen=True)
class CronSchedule:
    """Validated five-field cron cadence in UTC."""

    minute: str
    hour: str
    day: str
    month: str
    day_of_week: str

    def __post_init__(self) -> None:
        parts = (
            self.minute,
            self.hour,
            self.day,
            self.month,
            self.day_of_week,
        )
        if any(
            not isinstance(part, str) or _CRON_ATOM.fullmatch(part) is None
            for part in parts
        ):
            raise ResearchConfigError("cron schedule contains an invalid field")
        _validate_cron_fields(parts)

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


def _get_optional_text(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip() or None


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
    daily_schedule: CronSchedule = CronSchedule.from_crontab("0 7 * * mon-fri")
    weekly_schedule: CronSchedule = CronSchedule.from_crontab("0 8 * * mon")
    monthly_schedule: CronSchedule = CronSchedule.from_crontab("0 9 1 * *")
    monthly_industry: str | None = None
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
    def from_env(
        cls,
        *,
        include_flex: bool = True,
        include_model_secret: bool = True,
    ) -> "ResearchConfig":
        """Build configuration from the process environment without loading `.env`."""
        if not isinstance(include_flex, bool) or not isinstance(
            include_model_secret, bool
        ):
            raise TypeError("configuration scope flags must be bool")
        openai_key = (
            _read_secret("OPENAI_API_KEY") if include_model_secret else None
        )
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
                _get_text("RESEARCH_DAILY_SCHEDULE", "0 7 * * mon-fri"),
                setting="RESEARCH_DAILY_SCHEDULE",
            ),
            weekly_schedule=CronSchedule.from_crontab(
                _get_text("RESEARCH_WEEKLY_SCHEDULE", "0 8 * * mon"),
                setting="RESEARCH_WEEKLY_SCHEDULE",
            ),
            monthly_schedule=CronSchedule.from_crontab(
                _get_text("RESEARCH_MONTHLY_SCHEDULE", "0 9 1 * *"),
                setting="RESEARCH_MONTHLY_SCHEDULE",
            ),
            monthly_industry=_get_optional_text("RESEARCH_MONTHLY_INDUSTRY"),
            ibkr_flex_token=(
                _read_secret("IBKR_FLEX_TOKEN") if include_flex else None
            ),
            ibkr_flex_query_id=(
                _read_secret("IBKR_FLEX_QUERY_ID") if include_flex else None
            ),
            ibkr_flex_account_salt=(
                _read_secret("IBKR_FLEX_ACCOUNT_SALT") if include_flex else None
            ),
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
        if self.ibkr_flex_account_salt is not None:
            try:
                encoded_salt = self.ibkr_flex_account_salt.encode(
                    "utf-8", errors="strict"
                )
            except UnicodeEncodeError:
                raise ResearchConfigError(
                    "IBKR Flex account salt must be valid UTF-8"
                ) from None
            if len(encoded_salt) < 16:
                raise ResearchConfigError(
                    "IBKR Flex account salt must be at least 16 UTF-8 bytes"
                )
        if self.monthly_industry is not None and (
            not isinstance(self.monthly_industry, str)
            or _INDUSTRY_KEY.fullmatch(self.monthly_industry) is None
        ):
            raise ResearchConfigError(
                "RESEARCH_MONTHLY_INDUSTRY must be a safe industry key"
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
