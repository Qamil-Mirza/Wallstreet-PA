"""Production composition for durable portfolio-research workflows."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .config import ResearchConfig
from .ibkr_flex import FlexClient, FlexConfig, FlexError
from .models import PortfolioSnapshot
from .orchestrator import (
    ResearchOrchestrator,
    RequiredStageUnavailable,
    StageContext,
    StageExecutionControl,
    StageOutcome,
    WorkflowConflict,
    WorkflowRunResult,
)
from .store import ResearchStore


_RUNTIME_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")


class FlexSyncClient(Protocol):
    def sync(self, store: ResearchStore) -> object: ...
    def close(self) -> None: ...


def _default_owner_id() -> str:
    return f"runtime-{uuid4().hex}"


@dataclass(frozen=True)
class RuntimeFactories:
    """Network-facing construction seams for the production composition root."""

    flex_client_factory: Callable[[FlexConfig], FlexSyncClient] = FlexClient
    owner_id_factory: Callable[[], str] = _default_owner_id


class RuntimeStageRunner:
    """Run implemented stages and durably block unavailable required adapters."""

    def __init__(
        self,
        config: ResearchConfig,
        store: ResearchStore,
        factories: RuntimeFactories,
        *,
        synthetic_portfolio: bool = False,
    ) -> None:
        self.config = config
        self.store = store
        self.factories = factories
        self.synthetic_portfolio = synthetic_portfolio

    def _synthetic_snapshot(self, as_of: datetime) -> PortfolioSnapshot:
        digest = hashlib.sha256(
            f"synthetic-portfolio:{as_of.date().isoformat()}".encode("utf-8")
        ).hexdigest()
        snapshot = PortfolioSnapshot(
            snapshot_id=f"snapshot-synthetic-{digest[:24]}",
            as_of=as_of,
            base_currency="USD",
            nav=Decimal("0"),
            cash=Decimal("0"),
            is_stale=False,
        )
        self.store.insert_portfolio_snapshot(
            snapshot, (), f"acct_{digest[:24]}"
        )
        return snapshot

    def _portfolio(
        self, context: StageContext, control: StageExecutionControl
    ) -> StageOutcome:
        if self.synthetic_portfolio:
            snapshot = self._synthetic_snapshot(context.as_of)
            return StageOutcome(result_ref=snapshot.snapshot_id)

        flex_config = FlexConfig(
            token=self.config.ibkr_flex_token or "",
            query_id=self.config.ibkr_flex_query_id or "",
            account_salt=self.config.ibkr_flex_account_salt or "",
        )
        client: FlexSyncClient | None = None
        try:
            control.checkpoint()
            client = self.factories.flex_client_factory(flex_config)
            sync_result = client.sync(self.store)
            control.checkpoint()
        except FlexError:
            raise RequiredStageUnavailable(
                "required portfolio source is unavailable"
            ) from None
        finally:
            if client is not None:
                client.close()
        snapshot = getattr(sync_result, "snapshot", None)
        if not isinstance(snapshot, PortfolioSnapshot):
            raise TypeError("Flex sync must return a portfolio snapshot")
        return StageOutcome(result_ref=snapshot.snapshot_id)

    def run(
        self, context: StageContext, control: StageExecutionControl
    ) -> StageOutcome:
        control.checkpoint()
        if context.stage == "portfolio":
            result = self._portfolio(context, control)
        else:
            # Connectors, agents, and renderers have strict standalone contracts,
            # but no repository-wide stage adapter yet. Persist a typed durable
            # block instead of bypassing the orchestrator or reporting success.
            raise RequiredStageUnavailable(
                "required production stage adapter is unavailable"
            )
        control.checkpoint()
        return result


class RuntimeWorkflowService:
    """High-level workflows backed by a fresh durable orchestrator per command."""

    def __init__(
        self,
        config: ResearchConfig,
        store: ResearchStore,
        factories: RuntimeFactories | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.factories = factories or RuntimeFactories()

    def _orchestrator(
        self, *, synthetic_portfolio: bool = False
    ) -> ResearchOrchestrator:
        runner = RuntimeStageRunner(
            self.config,
            self.store,
            self.factories,
            synthetic_portfolio=synthetic_portfolio,
        )
        return ResearchOrchestrator(
            self.store,
            runner,
            owner_id=self.factories.owner_id_factory(),
        )

    def run_daily(
        self,
        *,
        as_of: datetime,
        dry_run: bool = False,
        synthetic_portfolio: bool = False,
    ) -> WorkflowRunResult:
        return self._orchestrator(
            synthetic_portfolio=synthetic_portfolio
        ).run_daily(as_of=as_of, dry_run=dry_run)

    def run_weekly(self, *, as_of: datetime) -> WorkflowRunResult:
        return self._orchestrator().run_weekly(as_of=as_of)

    def run_monthly(
        self, *, as_of: datetime, industry_key: str
    ) -> WorkflowRunResult:
        return self._orchestrator().run_monthly(
            as_of=as_of, industry_key=industry_key
        )

    def run_backfill(
        self,
        *,
        as_of: datetime,
        max_documents: int,
        authorize_analysis: bool,
    ) -> WorkflowRunResult:
        return self._orchestrator().run_backfill(
            as_of=as_of,
            max_documents=max_documents,
            authorize_analysis=authorize_analysis,
        )

    def regenerate(
        self,
        *,
        as_of: datetime,
        run_id: str | None,
        report_id: str | None,
    ) -> WorkflowRunResult:
        """Replay one stored run without invoking any stage or external adapter."""
        for value in (run_id, report_id):
            if value is not None and _RUNTIME_ID.fullmatch(value) is None:
                raise ValueError("regeneration identifier is invalid")
        if (run_id is None) == (report_id is None):
            raise ValueError("regeneration requires exactly one identifier")
        with self.store.connect() as connection:
            if run_id is not None:
                rows = connection.execute(
                    "SELECT workflow_id FROM workflow_runs "
                    "WHERE workflow_id = ? AND substr(as_of, 1, 10) = ?",
                    (run_id, as_of.date().isoformat()),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT workflow_id FROM workflow_runs "
                    "WHERE substr(as_of, 1, 10) = ? "
                    "AND EXISTS (SELECT 1 FROM json_each(report_ids_json) "
                    "WHERE value = ?) ORDER BY workflow_id",
                    (as_of.date().isoformat(), report_id),
                ).fetchall()
        if len(rows) != 1:
            raise WorkflowConflict("stored report or run was not found")
        result = self._orchestrator().replay_run(rows[0][0])
        selected_reports = (
            (report_id,) if report_id is not None else result.report_ids
        )
        if selected_reports:
            placeholders = ",".join("?" for _ in selected_reports)
            with self.store.connect() as connection:
                report_rows = connection.execute(
                    "SELECT report_id, body FROM reports WHERE report_id IN "
                    f"({placeholders}) AND status IN ('reviewed', 'published') "
                    "ORDER BY report_id",
                    selected_reports,
                ).fetchall()
            if tuple(row[0] for row in report_rows) != tuple(
                sorted(selected_reports)
            ):
                raise WorkflowConflict("stored report body was not found")
            for stored_report_id, body in report_rows:
                self._write_stored_report(stored_report_id, as_of, body)
        return result

    def _write_stored_report(
        self, report_id: str, as_of: datetime, body: str
    ) -> Path:
        if not isinstance(body, str):
            raise TypeError("stored report body must be text")
        payload = body.encode("utf-8", errors="strict")
        digest = hashlib.sha256(
            f"{report_id}:{as_of.date().isoformat()}".encode("utf-8")
        ).hexdigest()
        self.config.report_dir.mkdir(parents=True, exist_ok=True)
        destination = self.config.report_dir / f"regenerated-{digest[:24]}.html"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.config.report_dir,
                prefix=".regenerate-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return destination


__all__ = [
    "FlexSyncClient",
    "RuntimeFactories",
    "RuntimeStageRunner",
    "RuntimeWorkflowService",
]
