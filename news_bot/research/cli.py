"""Safe, injectable command line boundary for portfolio research workflows."""

from __future__ import annotations

import argparse
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
    validate_credentials: Callable[[str, ResearchConfig], None]


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


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(prog="portfolio-research", add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("daily", "weekly", "monthly", "dry-run", "regenerate"):
        child = subparsers.add_parser(command)
        child.add_argument("--as-of", required=True, type=_strict_date)
    backfill = subparsers.add_parser("backfill")
    backfill.add_argument("--as-of", required=True, type=_strict_date)
    backfill.add_argument("--max-documents", type=_bounded_documents, default=100)
    backfill.add_argument("--authorize-analysis", action="store_true")
    return parser


class _UnconfiguredWorkflows:
    def _unavailable(self, **_kwargs: object) -> object:
        raise RuntimeError("research workflow runtime is not configured")

    run_daily = _unavailable
    run_weekly = _unavailable
    run_monthly = _unavailable
    run_backfill = _unavailable
    regenerate = _unavailable


def _open_store(config: ResearchConfig) -> ResearchStore:
    store = ResearchStore(config.database_path)
    store.migrate()
    return store


def _build_workflows(
    _config: ResearchConfig, _store: ResearchStore
) -> WorkflowService:
    # Concrete connectors/stage adapters are injected by the runtime composition
    # root.  Keeping this fallback inert guarantees a bare command makes no paid
    # call and cannot trade or publish accidentally.
    return _UnconfiguredWorkflows()


def _validate_credentials(command: str, config: ResearchConfig) -> None:
    config.validate()
    if command == "regenerate":
        return
    credentials = (
        config.ibkr_flex_token,
        config.ibkr_flex_query_id,
        config.ibkr_flex_account_salt,
    )
    if not all(isinstance(value, str) and bool(value.strip()) for value in credentials):
        raise ResearchConfigError("required read-only IBKR Flex credentials are missing")


def default_services() -> CliServices:
    return CliServices(
        load_config=ResearchConfig.from_env,
        open_store=_open_store,
        build_workflows=_build_workflows,
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
        return workflows.run_monthly(as_of=as_of)
    if command == "dry-run":
        return workflows.run_daily(as_of=as_of, dry_run=True)
    if command == "backfill":
        return workflows.run_backfill(
            as_of=as_of,
            max_documents=arguments.max_documents,
            authorize_analysis=arguments.authorize_analysis,
        )
    if command == "regenerate":
        # The as-of date is the deterministic existing-run lookup key.  This
        # separate service method must render stored data without refetching.
        return workflows.regenerate(as_of=as_of)
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
        dependencies.validate_credentials(arguments.command, config)
        lease = RunLease.acquire(store, RUN_LEASE_NAME, ttl_seconds=3600)
        workflows = dependencies.build_workflows(config, store)
        result = _dispatch(arguments.command, arguments, workflows)
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
