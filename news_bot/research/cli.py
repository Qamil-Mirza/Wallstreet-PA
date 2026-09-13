"""Safe, injectable command line boundary for portfolio research workflows."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import IntEnum
from typing import Protocol

from .config import ResearchConfig, ResearchConfigError
from .orchestrator import WorkflowBusy
from .scheduler import RUN_LEASE_NAME, RunAlreadyActive, RunLease
from .store import ResearchStore


class ExitCode(IntEnum):
    """Stable process outcomes for automation and scheduler monitoring."""

    OK = 0
    OPERATIONAL_FAILURE = 1
    INVALID_INPUT = 2
    BUSY = 3
    BLOCKED = 4
    INVALID_CONFIG = 5


class CliInputError(ValueError):
    """Raised for command syntax or strict date validation failures."""


class WorkflowService(Protocol):
    def run_daily(self, **kwargs: object) -> object: ...
    def run_weekly(self, **kwargs: object) -> object: ...
    def run_monthly(self, **kwargs: object) -> object: ...
    def run_backfill(self, **kwargs: object) -> object: ...
    def regenerate(self, **kwargs: object) -> object: ...


@dataclass(frozen=True)
class CliServices:
    """Factories injected at the CLI boundary for network-free testing."""

    load_config: Callable[[], ResearchConfig]
    open_store: Callable[[ResearchConfig], ResearchStore]
    build_workflows: Callable[[ResearchConfig, ResearchStore], WorkflowService]
    validate_credentials: Callable[[str, ResearchConfig, bool], None]
    lease_ttl_seconds: float = 3600


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliInputError("invalid command input")


def _strict_date(value: str) -> datetime:
    if (
        not isinstance(value, str)
        or len(value) != 10
        or value[4:5] != "-"
        or value[7:8] != "-"
        or not (value[:4] + value[5:7] + value[8:]).isdigit()
    ):
        raise CliInputError("as-of must use YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise CliInputError("as-of must be a valid calendar date") from exc
    return datetime.combine(parsed, datetime.min.time(), tzinfo=timezone.utc)


def _bounded_documents(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CliInputError("max-documents must be an integer") from exc
    if not 1 <= parsed <= 1000:
        raise CliInputError("max-documents must be between 1 and 1000")
    return parsed


def _industry_key(value: str) -> str:
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value) is None:
        raise CliInputError("industry must be a safe key")
    return value


def _runtime_identifier(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", value) is None:
        raise CliInputError("run/report identifier is invalid")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(prog="portfolio-research", add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("daily", "weekly"):
        child = subparsers.add_parser(command)
        child.add_argument("--as-of", required=True, type=_strict_date)
    monthly = subparsers.add_parser("monthly")
    monthly.add_argument("--as-of", required=True, type=_strict_date)
    monthly.add_argument("--industry", required=True, type=_industry_key)
    dry_run = subparsers.add_parser("dry-run")
    dry_run.add_argument("--as-of", required=True, type=_strict_date)
    dry_run.add_argument("--synthetic-portfolio", action="store_true")
    regenerate = subparsers.add_parser("regenerate")
    regenerate.add_argument("--as-of", required=True, type=_strict_date)
    identity = regenerate.add_mutually_exclusive_group(required=True)
    identity.add_argument("--run-id", type=_runtime_identifier)
    identity.add_argument("--report-id", type=_runtime_identifier)
    backfill = subparsers.add_parser("backfill")
    backfill.add_argument("--as-of", required=True, type=_strict_date)
    backfill.add_argument("--max-documents", type=_bounded_documents, default=100)
    backfill.add_argument("--authorize-analysis", action="store_true")
    return parser


def _open_store(config: ResearchConfig) -> ResearchStore:
    store = ResearchStore(config.database_path)
    store.migrate()
    return store


def _validate_credentials(
    command: str, config: ResearchConfig, requires_portfolio: bool
) -> None:
    config.validate()
    if not requires_portfolio:
        return
    credentials = (
        config.ibkr_flex_token,
        config.ibkr_flex_query_id,
        config.ibkr_flex_account_salt,
    )
    if not all(isinstance(value, str) and bool(value.strip()) for value in credentials):
        raise ResearchConfigError(
            "required read-only IBKR Flex credentials are missing"
        )


def default_services(
    *,
    config_loader: Callable[[], ResearchConfig] = ResearchConfig.from_env,
    store_factory: Callable[[ResearchConfig], ResearchStore] = _open_store,
    runtime_factories: object | None = None,
) -> CliServices:
    from .runtime import RuntimeFactories, RuntimeWorkflowService

    factories = (
        RuntimeFactories()
        if runtime_factories is None
        else runtime_factories
    )
    if not isinstance(factories, RuntimeFactories):
        raise TypeError("runtime_factories must be RuntimeFactories")
    return CliServices(
        load_config=config_loader,
        open_store=store_factory,
        build_workflows=lambda config, store: RuntimeWorkflowService(
            config, store, factories
        ),
        validate_credentials=_validate_credentials,
    )


def _dispatch(
    command: str, arguments: argparse.Namespace, workflows: WorkflowService
) -> object:
    as_of = arguments.as_of
    if command == "daily":
        return workflows.run_daily(as_of=as_of)
    if command == "weekly":
        return workflows.run_weekly(as_of=as_of)
    if command == "monthly":
        return workflows.run_monthly(
            as_of=as_of, industry_key=arguments.industry
        )
    if command == "dry-run":
        return workflows.run_daily(
            as_of=as_of,
            dry_run=True,
            synthetic_portfolio=arguments.synthetic_portfolio,
        )
    if command == "backfill":
        return workflows.run_backfill(
            as_of=as_of,
            max_documents=arguments.max_documents,
            authorize_analysis=arguments.authorize_analysis,
        )
    if command == "regenerate":
        # The as-of date is the deterministic existing-run lookup key.  This
        # separate service method must render stored data without refetching.
        return workflows.regenerate(
            as_of=as_of,
            run_id=arguments.run_id,
            report_id=arguments.report_id,
        )
    raise CliInputError("unsupported command")


def _status_value(result: object) -> str:
    status = getattr(result, "status", None)
    value = getattr(status, "value", status)
    if not isinstance(value, str) or value not in {
        "running",
        "completed",
        "partial",
        "blocked",
        "failed",
        "deferred",
    }:
        raise RuntimeError("workflow returned an invalid status")
    return value


def _count_field(result: object, name: str) -> int:
    value = getattr(result, name, ())
    if isinstance(value, (str, bytes)):
        raise RuntimeError("workflow returned an invalid summary field")
    try:
        return len(value)
    except TypeError as exc:
        raise RuntimeError("workflow returned an invalid summary field") from exc


def _print_summary(command: str, result: object, status: str) -> None:
    # Only fixed command/status vocabularies and aggregate counts cross stdout.
    print(
        f"research command={command} status={status} "
        f"reports={_count_field(result, 'report_ids')} "
        f"omissions={_count_field(result, 'omissions')}"
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    services: CliServices | None = None,
) -> int:
    """Run exactly one workflow and return a stable, testable exit code."""
    try:
        arguments = _parser().parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code)
    except (CliInputError, argparse.ArgumentError):
        print("research error: invalid input", file=sys.stderr)
        return ExitCode.INVALID_INPUT

    dependencies = services or default_services()
    lease: RunLease | None = None
    try:
        config = dependencies.load_config()
        store = dependencies.open_store(config)
        requires_portfolio = arguments.command in {"daily", "weekly"} or (
            arguments.command == "dry-run"
            and not arguments.synthetic_portfolio
        )
        dependencies.validate_credentials(
            arguments.command, config, requires_portfolio
        )
        lease = RunLease.acquire(
            store,
            RUN_LEASE_NAME,
            ttl_seconds=dependencies.lease_ttl_seconds,
        )
        workflows = dependencies.build_workflows(config, store)
        result = _dispatch(arguments.command, arguments, workflows)
        lease.assert_current()
        status = _status_value(result)
        _print_summary(arguments.command, result, status)
        if status == "blocked":
            return ExitCode.BLOCKED
        if status == "failed":
            return ExitCode.OPERATIONAL_FAILURE
        return ExitCode.OK
    except ResearchConfigError:
        print("research error: invalid configuration", file=sys.stderr)
        return ExitCode.INVALID_CONFIG
    except (RunAlreadyActive, WorkflowBusy):
        print("research error: run already active", file=sys.stderr)
        return ExitCode.BUSY
    except Exception:
        print("research error: operational failure", file=sys.stderr)
        return ExitCode.OPERATIONAL_FAILURE
    finally:
        if lease is not None:
            lease.release()


if __name__ == "__main__":  # pragma: no cover - thin process boundary
    raise SystemExit(main())


__all__ = [
    "CliInputError",
    "CliServices",
    "ExitCode",
    "WorkflowService",
    "default_services",
    "main",
]
