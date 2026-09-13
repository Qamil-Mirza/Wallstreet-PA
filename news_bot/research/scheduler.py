"""APScheduler cadence and shared SQLite-backed research run lease."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from .config import ResearchConfig
from .orchestrator import (
    ResearchOrchestrator,
    StageContext,
    StageExecutionControl,
    StageOutcome,
    WorkflowBusy,
    WorkflowConflict,
    WorkflowLease,
)
from .store import ResearchStore

if TYPE_CHECKING:
    from .cli import CliServices


RUN_LEASE_NAME = "research:runner"


class RunAlreadyActive(RuntimeError):
    """Raised when a one-shot or scheduled research run owns the shared lease."""


class RunLeaseLost(RuntimeError):
    """Raised when a runner no longer owns its renewable lease fence."""


class SchedulerClockError(RuntimeError):
    """Raised when a scheduled job is not given a strict UTC clock."""


class SchedulerRunError(RuntimeError):
    """Raised so APScheduler records a non-successful CLI job execution."""


class _LeaseOnlyRunner:
    def run(
        self, context: StageContext, control: StageExecutionControl
    ) -> StageOutcome:  # pragma: no cover - leases never execute a stage
        raise RuntimeError("lease-only orchestrator cannot execute stages")


@dataclass
class RunLease:
    """Fenced handle for the global scheduler/one-shot execution lease."""

    _orchestrator: ResearchOrchestrator
    _lease: WorkflowLease
    _released: bool = field(default=False, init=False)
    _stop: Event = field(default_factory=Event, init=False, repr=False)
    _lost: Event = field(default_factory=Event, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _thread: Thread | None = field(default=None, init=False, repr=False)
    _ttl_seconds: float = field(default=0, init=False, repr=False)

    @classmethod
    def acquire(
        cls,
        store: ResearchStore,
        name: str = RUN_LEASE_NAME,
        *,
        ttl_seconds: float = 3600,
    ) -> "RunLease":
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
            raise TypeError("ttl_seconds must be numeric")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        orchestrator = ResearchOrchestrator(
            store,
            _LeaseOnlyRunner(),
            owner_id=f"runner-{uuid4().hex}",
            lease_duration=timedelta(seconds=ttl_seconds),
        )
        try:
            lease = orchestrator.acquire_lease(
                name, duration=timedelta(seconds=ttl_seconds)
            )
        except WorkflowBusy as exc:
            raise RunAlreadyActive("research run is already active") from exc
        result = cls(orchestrator, lease)
        result._ttl_seconds = float(ttl_seconds)
        result._start_heartbeat()
        return result

    def _start_heartbeat(self) -> None:
        interval = max(0.01, min(self._ttl_seconds / 3, 30.0))

        def heartbeat() -> None:
            while not self._stop.wait(interval):
                with self._lock:
                    try:
                        if self._released:
                            return
                        self._lease = self._orchestrator.renew_lease(
                            self._lease,
                            duration=timedelta(seconds=self._ttl_seconds),
                        )
                    except Exception:
                        # Record loss before unlocking so assert_current cannot
                        # race a failed renewal and accept a stale result.
                        self._lost.set()
                        self._stop.set()
                        return

        self._thread = Thread(
            target=heartbeat,
            name="research-run-lease-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def assert_current(self) -> None:
        """Renew synchronously and reject results after ownership is lost."""
        try:
            with self._lock:
                if self._lost.is_set():
                    raise RunLeaseLost("research run lease was lost")
                if self._released:
                    raise RunLeaseLost("research run lease was released")
                self._lease = self._orchestrator.renew_lease(
                    self._lease,
                    duration=timedelta(seconds=self._ttl_seconds),
                )
        except WorkflowConflict as exc:
            self._lost.set()
            raise RunLeaseLost("research run lease was lost") from exc

    def release(self) -> None:
        """Release only this token; stale/idempotent releases are harmless."""
        if self._released:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        try:
            with self._lock:
                if self._released:
                    return
                self._released = True
                self._orchestrator.release_lease(self._lease)
        except WorkflowConflict:
            # The lease expired or was fenced by a newer owner.  The token-aware
            # DELETE guarantees this stale handle cannot release that owner.
            return

    def __enter__(self) -> "RunLease":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class _Scheduler(Protocol):
    def add_job(self, function: Callable[..., object], **kwargs: object) -> object:
        """Register one scheduled callable."""


def _scheduled_job(
    command: str,
    services: CliServices | None,
    clock: Callable[[], datetime],
    extra_arguments: tuple[str, ...] = (),
) -> None:
    from .cli import ExitCode, main

    now = clock()
    if (
        not isinstance(now, datetime)
        or now.tzinfo is None
        or now.utcoffset() != timedelta(0)
    ):
        raise SchedulerClockError("scheduler clock must return an aware UTC datetime")
    code = main(
        [command, "--as-of", now.astimezone(timezone.utc).date().isoformat(),
         *extra_arguments],
        services=services,
    )
    if code is not ExitCode.OK and code != int(ExitCode.OK):
        raise SchedulerRunError(
            f"scheduled research job failed with exit code {int(code)}"
        )


def build_scheduler(
    config: ResearchConfig,
    *,
    services: CliServices | None = None,
    scheduler_factory: Callable[[], _Scheduler] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> _Scheduler:
    """Register UTC research jobs without initializing workflow services."""
    if not isinstance(config, ResearchConfig):
        raise TypeError("config must be ResearchConfig")
    if config.monthly_industry is None:
        from .config import ResearchConfigError

        raise ResearchConfigError("RESEARCH_MONTHLY_INDUSTRY is required")
    if scheduler_factory is None:
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
        except ImportError as exc:  # pragma: no cover - packaging/runtime boundary
            raise RuntimeError("APScheduler is not installed") from exc

        scheduler: _Scheduler = BlockingScheduler(timezone=timezone.utc)
    else:
        scheduler = scheduler_factory()

    cadences = (
        ("daily", config.daily_schedule, ()),
        ("weekly", config.weekly_schedule, ()),
        (
            "monthly",
            config.monthly_schedule,
            ("--industry", config.monthly_industry),
        ),
    )
    for command, cadence, extra_arguments in cadences:
        scheduler.add_job(
            _scheduled_job,
            trigger="cron",
            **cadence.as_kwargs(),
            timezone=timezone.utc,
            id=f"research-{command}",
            args=(command, services, clock, extra_arguments),
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
    return scheduler


def main() -> int:
    """Start the blocking scheduler; startup performs no paid/API calls."""
    from .config import ResearchConfigError

    try:
        config = ResearchConfig.from_env()
        scheduler = build_scheduler(config)
        start = getattr(scheduler, "start")
        start()
    except (ResearchConfigError, RuntimeError):
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover - thin process boundary
    raise SystemExit(main())


__all__ = [
    "RUN_LEASE_NAME",
    "RunAlreadyActive",
    "RunLease",
    "RunLeaseLost",
    "SchedulerClockError",
    "SchedulerRunError",
    "build_scheduler",
    "main",
]
