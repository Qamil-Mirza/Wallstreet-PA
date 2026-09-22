"""Production composition for durable portfolio-research workflows."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import ResearchConfig
from .ibkr_flex import FlexClient, FlexConfig, FlexError, PortfolioSyncResult
from .models import (
    InferenceMode,
    PortfolioSnapshot,
    RecommendationRating,
    ReviewVerdict,
)
from .orchestrator import (
    ResearchOrchestrator,
    RequiredStageUnavailable,
    StageContext,
    StageExecutionControl,
    StageOutcome,
    WorkflowConflict,
    WorkflowRunResult,
)
from .quality import GateReasonCode, PublicationVerdict, QualityGateResult
from .reports import (
    Citation,
    EmergingCompanyMonitor,
    EventUpdate,
    IndustryLandscape,
    PortfolioBrief,
    PublicationPrivacyContext,
    ReportMetadata,
    ReportRenderer,
    ReportSection,
)
from .store import (
    ResearchStore,
    _canonical_json,
    _decimal_text,
    _parse_utc,
    _utc_text,
)


_RUNTIME_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ACTIVE_MARKUP = re.compile(
    r"(?is)<\s*/?\s*(?:script|iframe|object|embed|svg|math|style|link|meta)\b|"
    r"\bon[a-z]+\s*=|\bjavascript\s*:"
)
_RUNTIME_STAGES = frozenset(
    {
        "ingestion",
        "resolve",
        "materiality",
        "event_analysis",
        "event_update",
        "changed_theses",
        "selected_recommendations",
        "public_signals",
        "emerging_map",
        "industry_refresh",
        "historical_documents",
        "evidence",
        "analysis",
        "review",
        "revision",
        "publish",
    }
)
_REPORT_ORIGIN_STAGES = MappingProxyType(
    {
        "event_update": "event_update",
        "portfolio_brief": "selected_recommendations",
        "industry_landscape": "industry_refresh",
        "emerging_monitor": "emerging_map",
    }
)


class FlexSyncClient(Protocol):
    def sync(self, store: ResearchStore) -> object: ...
    def close(self) -> None: ...


class RuntimeStageAdapter(Protocol):
    """Strict adapter contract for one configured orchestration stage."""

    def run(
        self, context: StageContext, control: StageExecutionControl
    ) -> StageOutcome: ...


class RuntimeCancellationSource(Protocol):
    """Propagate global runner-lease loss into cooperative stage controls."""

    def register_cancellation_event(self, event: object) -> None: ...
    def unregister_cancellation_event(self, event: object) -> None: ...


class _StoredReportProvenance(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        hide_input_in_errors=True,
    )

    schema_version: Literal["trusted_report_payload/v1"] = Field(alias="schema")
    workflow_id: str
    task_id: str
    report_type: Literal[
        "event_update",
        "portfolio_brief",
        "industry_landscape",
        "emerging_monitor",
    ]
    portfolio_snapshot_ref: str | None
    payload_sha256: str
    html_filename: str
    pdf_status: Literal[
        "rendered", "pdf_backend_unavailable", "pdf_render_failed"
    ]

    @field_validator("workflow_id", "task_id", "portfolio_snapshot_ref")
    @classmethod
    def _identifier(cls, value: str | None) -> str | None:
        if value is not None and _RUNTIME_ID.fullmatch(value) is None:
            raise ValueError("stored report provenance identifier is invalid")
        return value

    @field_validator("payload_sha256")
    @classmethod
    def _digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("stored report provenance digest is invalid")
        return value

    @field_validator("html_filename")
    @classmethod
    def _filename(cls, value: str) -> str:
        if (
            not isinstance(value, str)
            or Path(value).name != value
            or not value.endswith(".html")
        ):
            raise ValueError("stored report provenance filename is invalid")
        return value


def _default_owner_id() -> str:
    return f"runtime-{uuid4().hex}"


@dataclass(frozen=True)
class RuntimeFactories:
    """Network-facing construction seams for the production composition root."""

    flex_client_factory: Callable[[FlexConfig], FlexSyncClient] = FlexClient
    owner_id_factory: Callable[[], str] = _default_owner_id
    stage_adapters: Mapping[str, RuntimeStageAdapter] = field(default_factory=dict)

    def __post_init__(self) -> None:
        adapters = dict(self.stage_adapters)
        if any(
            not isinstance(stage, str)
            or stage not in _RUNTIME_STAGES
            or not callable(getattr(adapter, "run", None))
            for stage, adapter in adapters.items()
        ):
            raise TypeError("stage_adapters must map known stages to adapters")
        object.__setattr__(self, "stage_adapters", MappingProxyType(adapters))


class RuntimeStageRunner:
    """Run implemented stages and durably block unavailable required adapters."""

    def __init__(
        self,
        config: ResearchConfig,
        store: ResearchStore,
        factories: RuntimeFactories,
        *,
        synthetic_portfolio: bool = False,
        cancellation_source: RuntimeCancellationSource | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.factories = factories
        self.synthetic_portfolio = synthetic_portfolio
        self.cancellation_source = cancellation_source

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
        return snapshot

    def _validated_portfolio_result(
        self, result: object, requested_as_of: datetime
    ) -> PortfolioSyncResult:
        if not isinstance(result, PortfolioSyncResult):
            raise RequiredStageUnavailable(
                "portfolio source returned an invalid result"
            )
        freshness = result.freshness
        snapshot = result.snapshot
        if (
            freshness is None
            or snapshot.as_of > requested_as_of
            or snapshot.as_of > result.generated_at
            or result.generated_at > freshness.evaluated_at
        ):
            raise RequiredStageUnavailable(
                "portfolio source is inconsistent"
            )
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT as_of, base_currency, nav, cash, is_stale, account_ref "
                "FROM portfolio_snapshots WHERE snapshot_id = ?",
                (snapshot.snapshot_id,),
            ).fetchone()
            stored_positions = tuple(
                tuple(item)
                for item in connection.execute(
                    "SELECT position_id, snapshot_id, symbol, quantity, "
                    "market_value, currency, cost_basis, security_id "
                    "FROM positions WHERE snapshot_id = ? "
                    "ORDER BY position_id",
                    (snapshot.snapshot_id,),
                ).fetchall()
            )
            stored_securities = tuple(
                tuple(item)
                for item in connection.execute(
                    "SELECT position.position_id, security.security_id, "
                    "security.symbol, security.security_type, security.exchange, "
                    "security.currency, security.identifiers_json "
                    "FROM positions AS position JOIN securities AS security "
                    "ON security.security_id = position.security_id "
                    "WHERE position.snapshot_id = ? ORDER BY position.position_id",
                    (snapshot.snapshot_id,),
                ).fetchall()
            )
        expected_positions = tuple(
            sorted(
                (
                    position.position_id,
                    position.snapshot_id,
                    position.symbol,
                    _decimal_text(position.quantity),
                    _decimal_text(position.market_value),
                    position.currency,
                    (
                        None
                        if position.cost_basis is None
                        else _decimal_text(position.cost_basis)
                    ),
                    position.security_id,
                )
                for position in result.positions
            )
        )
        expected_securities = tuple(
            sorted(
                (
                    position.position_id,
                    position.security_id,
                    position.symbol,
                    position.asset_class or "UNKNOWN",
                    None,
                    position.currency,
                    _canonical_json(
                        {
                            key: value
                            for key, value in (
                                ("conid", position.conid),
                                ("isin", position.isin),
                                ("cusip", position.cusip),
                                ("figi", position.figi),
                                (
                                    "external_security_id",
                                    position.external_security_id,
                                ),
                                ("security_id_type", position.security_id_type),
                            )
                            if value is not None
                        }
                    ),
                )
                for position in result.positions
            )
        )
        if (
            row is None
            or _parse_utc(row[0]) != snapshot.as_of
            or row[1] != snapshot.base_currency
            or Decimal(row[2]) != snapshot.nav
            or Decimal(row[3]) != snapshot.cash
            or row[4] != (
                None if snapshot.is_stale is None else int(snapshot.is_stale)
            )
            or row[5] != result.account_ref
            or stored_positions != expected_positions
            or stored_securities != expected_securities
        ):
            raise RequiredStageUnavailable(
                "portfolio source result was not durably persisted"
            )
        return result

    def _portfolio_source_key(self) -> str:
        """Derive a non-reversible identity for one configured Flex source."""
        query_id = self.config.ibkr_flex_query_id or ""
        account_salt = self.config.ibkr_flex_account_salt or ""
        return hmac.new(
            account_salt.encode("utf-8"),
            f"ibkr-flex-query:{query_id}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _last_snapshot_ref(
        self, source_key: str, requested_as_of: datetime
    ) -> str | None:
        """Return the latest durable snapshot for exactly one Flex source."""
        return self.store.latest_portfolio_snapshot_for_source(
            source_key, requested_as_of
        )

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
            send_url=f"{self.config.ibkr_flex_base_url.rstrip('/')}/SendRequest",
            statement_url=(
                f"{self.config.ibkr_flex_base_url.rstrip('/')}/GetStatement"
            ),
            poll_timeout_seconds=self.config.ibkr_flex_poll_timeout_seconds,
            max_staleness_hours=self.config.portfolio_max_staleness_hours,
        )
        client: FlexSyncClient | None = None
        source_key = self._portfolio_source_key()
        try:
            control.checkpoint()
            client = self.factories.flex_client_factory(flex_config)
            sync_result = client.sync(self.store)
            control.checkpoint()
        except FlexError:
            fallback_ref = self._last_snapshot_ref(source_key, context.as_of)
            if fallback_ref is None:
                raise RequiredStageUnavailable(
                    "required portfolio source is unavailable"
                ) from None
            return StageOutcome(
                result_ref=fallback_ref,
                omissions=("portfolio_stale",),
            )
        finally:
            if client is not None:
                client.close()
        validated = self._validated_portfolio_result(
            sync_result, context.as_of
        )
        self.store.bind_portfolio_snapshot_source(
            validated.snapshot.snapshot_id, source_key
        )
        is_stale = bool(
            validated.freshness.is_stale
            or validated.snapshot.is_stale is True
        )
        return StageOutcome(
            result_ref=validated.snapshot.snapshot_id,
            omissions=("portfolio_stale",) if is_stale else (),
        )

    @staticmethod
    def _synthetic_ref(context: StageContext) -> str:
        digest = hashlib.sha256(
            f"{context.workflow_id}:{context.stage}:{context.as_of.isoformat()}".encode(
                "utf-8"
            )
        ).hexdigest()
        return f"synthetic-{context.stage}-{digest[:24]}"

    @staticmethod
    def _synthetic_gate() -> QualityGateResult:
        return QualityGateResult(
            reason_codes=(),
            allowed_sections=("synthetic-analysis",),
            allowed_claim_ids=("synthetic-claim",),
            allow_event_report=True,
            allow_sizing=False,
            publication_verdict=PublicationVerdict.FINAL,
            review_verdict=ReviewVerdict.PASS,
            effective_rating=RecommendationRating.HOLD,
            inference_mode=InferenceMode.LOCAL_ONLY,
            inference_provider="offline",
            inference_model="deterministic-fixture",
        )

    def _workflow_portfolio_ref(self, workflow_id: str) -> str:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT outcome_json FROM workflow_tasks "
                "WHERE workflow_id = ? AND stage = 'portfolio' "
                "AND state = 'completed'",
                (workflow_id,),
            ).fetchone()
        if row is None or row[0] is None:
            raise RequiredStageUnavailable(
                "required portfolio workflow result is unavailable"
            )
        outcome = StageOutcome.model_validate_json(row[0])
        if outcome.result_ref is None:
            raise RequiredStageUnavailable(
                "required portfolio workflow result is unavailable"
            )
        return outcome.result_ref

    def _workflow_portfolio_stale(self, workflow_id: str) -> bool:
        """Return whether the durable portfolio stage declared stale inputs."""
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT outcome_json FROM workflow_tasks "
                "WHERE workflow_id = ? AND stage = 'portfolio' "
                "AND state = 'completed'",
                (workflow_id,),
            ).fetchone()
        if row is None or row[0] is None:
            raise RequiredStageUnavailable(
                "required portfolio workflow result is unavailable"
            )
        outcome = StageOutcome.model_validate_json(row[0])
        return "portfolio_stale" in outcome.omissions

    def _enforce_stale_portfolio_gate(
        self, context: StageContext, outcome: StageOutcome
    ) -> StageOutcome:
        """Prevent stale holdings from authorizing ratings or position sizing."""
        gate = outcome.quality_gate
        if (
            context.stage != "review"
            or gate is None
            or not self._workflow_portfolio_stale(context.workflow_id)
        ):
            return outcome
        verdict = gate.publication_verdict
        if verdict is PublicationVerdict.FINAL:
            verdict = PublicationVerdict.PARTIAL
        safe_gate = QualityGateResult.model_validate(
            {
                **gate.model_dump(),
                "reason_codes": tuple(
                    sorted(
                        {*gate.reason_codes, GateReasonCode.PORTFOLIO_STALE},
                        key=lambda reason: reason.value,
                    )
                ),
                "allow_sizing": False,
                "publication_verdict": verdict,
                "effective_rating": RecommendationRating.NO_RATING,
            }
        )
        return outcome.model_copy(update={"quality_gate": safe_gate})

    def _synthetic_event_report(self, context: StageContext) -> EventUpdate:
        report_digest = hashlib.sha256(
            f"synthetic-report:{context.workflow_id}".encode("utf-8")
        ).hexdigest()
        evidence_hash = hashlib.sha256(
            f"synthetic-evidence:{context.as_of.date().isoformat()}".encode("utf-8")
        ).hexdigest()
        report_id = f"report-synthetic-{report_digest[:24]}"
        citation = Citation(
            evidence_id="synthetic-evidence",
            source="Deterministic offline fixture",
            url="https://example.invalid/synthetic-research-fixture",
            source_date=context.as_of.date(),
            data_date=context.as_of.date(),
            content_hash=evidence_hash,
        )

        def section(title: str, body: str) -> ReportSection:
            return ReportSection(
                title=title,
                body=body,
                evidence_ids=(citation.evidence_id,),
            )

        return EventUpdate(
            metadata=ReportMetadata(
                report_id=report_id,
                report_type="event_update",
                title="Offline synthetic portfolio research validation",
                as_of=context.as_of,
                inference_mode=InferenceMode.LOCAL_ONLY,
                provider="offline",
                model="deterministic-fixture",
                freshness="Deterministic fixture for the requested research date.",
                citations=(citation,),
                methodology=(
                    "Static synthetic evidence exercises the durable workflow and "
                    "publication boundary without external services."
                ),
                omissions=(
                    "Live portfolio, source, model, delivery, and trading services "
                    "were intentionally omitted.",
                ),
                disclosure="All content is synthetic and has no trading authority.",
            ),
            thesis=section(
                "Validation thesis",
                "The offline fixture validates deterministic orchestration behavior.",
            ),
            event_decomposition=(
                section(
                    "Synthetic event",
                    "A static fixture supplies evidence for runtime validation.",
                ),
            ),
            causal_decomposition=(
                section(
                    "Synthetic causal chain",
                    "Deterministic inputs produce reproducible reviewed artifacts.",
                ),
            ),
            read_through=(
                section(
                    "Synthetic read-through",
                    "No live portfolio holdings or account data are consulted.",
                ),
            ),
            thesis_changes=(
                section(
                    "Synthetic change record",
                    "The fixture records no live investment thesis change.",
                ),
            ),
            unchanged_assumptions=(
                section(
                    "Synthetic assumptions",
                    "The validation remains isolated from external state.",
                ),
            ),
            questions=(
                section(
                    "Validation question",
                    "Can the complete durable path run without external services?",
                ),
            ),
            signposts=(
                section(
                    "Validation signpost",
                    "A reviewed escaped artifact confirms the offline path.",
                ),
            ),
        )

    @staticmethod
    def _canonical_report_body(report: EventUpdate) -> str:
        return json.dumps(
            report.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def _persist_synthetic_report(
        self,
        context: StageContext,
        control: StageExecutionControl,
    ) -> StageOutcome:
        report = self._synthetic_event_report(context)
        body = self._canonical_report_body(report)
        payload_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
        control.checkpoint()
        artifact = ReportRenderer(
            self.config.report_dir,
            privacy_context=PublicationPrivacyContext(
                mode="no_sensitive_data",
                sensitive_literals=(),
                account_identifiers=(),
                portfolio_values=(),
            ),
        ).render_event_update(report)
        control.checkpoint()
        metadata = json.dumps(
            {
                "html_filename": artifact.html_path.name,
                "payload_sha256": payload_sha256,
                "pdf_status": (
                    "rendered" if artifact.pdf_path is not None else artifact.pdf_error
                ),
                "portfolio_snapshot_ref": self._workflow_portfolio_ref(
                    context.workflow_id
                ),
                "report_type": report.metadata.report_type,
                "schema": "trusted_report_payload/v1",
                "task_id": context.task_id,
                "workflow_id": context.workflow_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.store.transaction() as connection:
            existing = connection.execute(
                "SELECT report_key, version, title, body, metadata_json "
                "FROM reports WHERE report_id = ?",
                (report.metadata.report_id,),
            ).fetchone()
            expected = (
                f"synthetic-{context.workflow_id}",
                1,
                report.metadata.title,
                body,
                metadata,
            )
            if existing is None:
                connection.execute(
                    "INSERT INTO reports (report_id, report_key, version, title, "
                    "body, status, created_by_run_id, created_at, published_at, "
                    "metadata_json) VALUES (?, ?, ?, ?, ?, 'draft', NULL, ?, NULL, ?)",
                    (
                        report.metadata.report_id,
                        *expected[:4],
                        _utc_text(context.as_of),
                        expected[4],
                    ),
                )
            elif tuple(existing) != expected:
                raise RequiredStageUnavailable(
                    "stored report conflicts with deterministic output"
                )
        control.checkpoint()
        return StageOutcome(
            result_ref=report.metadata.report_id,
            result_hash=payload_sha256,
            report_id=report.metadata.report_id,
        )

    def _synthetic_stage(
        self, context: StageContext, control: StageExecutionControl
    ) -> StageOutcome:
        if context.stage == "materiality":
            return StageOutcome(
                result_ref=self._synthetic_ref(context), material_event=True
            )
        if context.stage == "event_update":
            return self._persist_synthetic_report(context, control)
        if context.stage == "review":
            report_ids = context.dependency_result_refs
            if len(report_ids) != 1:
                raise RequiredStageUnavailable(
                    "synthetic report review input is unavailable"
                )
            with self.store.transaction() as connection:
                cursor = connection.execute(
                    "UPDATE reports SET status = 'reviewed' "
                    "WHERE report_id = ? AND status IN ('draft', 'reviewed')",
                    (report_ids[0],),
                )
                if cursor.rowcount != 1:
                    raise RequiredStageUnavailable(
                        "synthetic report review input is unavailable"
                    )
            return StageOutcome(
                result_ref=self._synthetic_ref(context),
                reviewer_verdict=ReviewVerdict.PASS,
                quality_gate=self._synthetic_gate(),
                published_claim_ids=("synthetic-claim",),
            )
        if context.stage == "publish":
            raise RequiredStageUnavailable(
                "synthetic workflows cannot invoke publication"
            )
        return StageOutcome(result_ref=self._synthetic_ref(context))

    def run(
        self, context: StageContext, control: StageExecutionControl
    ) -> StageOutcome:
        cancellation_event = control.lost_lease_event
        registered = False
        if self.cancellation_source is not None:
            self.cancellation_source.register_cancellation_event(cancellation_event)
            registered = True
        try:
            control.checkpoint()
            if context.stage == "portfolio":
                result = self._portfolio(context, control)
            elif self.synthetic_portfolio:
                result = self._synthetic_stage(context, control)
            else:
                adapter = self.factories.stage_adapters.get(context.stage)
                if adapter is None:
                    raise RequiredStageUnavailable(
                        "required production stage adapter is unavailable"
                    )
                result = adapter.run(context, control)
                if not isinstance(result, StageOutcome):
                    raise RequiredStageUnavailable(
                        "required production stage adapter returned an invalid result"
                    )
            result = self._enforce_stale_portfolio_gate(context, result)
            control.checkpoint()
            return result
        finally:
            if registered and self.cancellation_source is not None:
                self.cancellation_source.unregister_cancellation_event(
                    cancellation_event
                )


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
        configured = factories or RuntimeFactories()
        if not configured.stage_adapters:
            from .production import build_production_stage_adapters

            configured = RuntimeFactories(
                flex_client_factory=configured.flex_client_factory,
                owner_id_factory=configured.owner_id_factory,
                stage_adapters=build_production_stage_adapters(config, store),
            )
        self.factories = configured
        self._cancellation_source: RuntimeCancellationSource | None = None

    def bind_run_lease(self, source: RuntimeCancellationSource) -> None:
        if not callable(
            getattr(source, "register_cancellation_event", None)
        ) or not callable(
            getattr(source, "unregister_cancellation_event", None)
        ):
            raise TypeError("run lease must provide cooperative cancellation")
        self._cancellation_source = source

    def _orchestrator(
        self, *, synthetic_portfolio: bool = False
    ) -> ResearchOrchestrator:
        runner = RuntimeStageRunner(
            self.config,
            self.store,
            self.factories,
            synthetic_portfolio=synthetic_portfolio,
            cancellation_source=self._cancellation_source,
        )
        return ResearchOrchestrator(
            self.store,
            runner,
            owner_id=self.factories.owner_id_factory(),
        )

    def _source_hashes(self, as_of: datetime) -> tuple[str, ...]:
        hashes = {
            document.content_hash
            for document in self.store.list_documents_as_of(as_of)
        }
        ingestion = self.factories.stage_adapters.get("ingestion")
        preview = getattr(ingestion, "preview_source_hashes", None)
        if callable(preview):
            current = preview(as_of)
            if not isinstance(current, tuple) or any(
                not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
                for digest in current
            ):
                raise RequiredStageUnavailable(
                    "required production source preview is invalid"
                )
            hashes.update(current)
        return tuple(sorted(hashes))

    def run_daily(
        self,
        *,
        as_of: datetime,
        dry_run: bool = False,
        synthetic_portfolio: bool = False,
    ) -> WorkflowRunResult:
        return self._orchestrator(
            synthetic_portfolio=synthetic_portfolio
        ).run_daily(
            as_of=as_of,
            source_hashes=self._source_hashes(as_of),
            dry_run=dry_run,
        )

    def run_weekly(self, *, as_of: datetime) -> WorkflowRunResult:
        return self._orchestrator().run_weekly(
            as_of=as_of, source_hashes=self._source_hashes(as_of)
        )

    def run_monthly(
        self, *, as_of: datetime, industry_key: str
    ) -> WorkflowRunResult:
        return self._orchestrator().run_monthly(
            as_of=as_of,
            industry_key=industry_key,
            source_hashes=self._source_hashes(as_of),
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
            source_hashes=self._source_hashes(as_of),
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
        workflow_id = rows[0][0]
        result = self._orchestrator().replay_run(workflow_id)
        selected_reports = (
            (report_id,) if report_id is not None else result.report_ids
        )
        if selected_reports:
            placeholders = ",".join("?" for _ in selected_reports)
            with self.store.connect() as connection:
                report_rows = connection.execute(
                    "SELECT report_id, body, metadata_json FROM reports "
                    "WHERE report_id IN "
                    f"({placeholders}) AND status IN ('reviewed', 'published') "
                    "ORDER BY report_id",
                    selected_reports,
                ).fetchall()
            if tuple(row[0] for row in report_rows) != tuple(
                sorted(selected_reports)
            ):
                raise WorkflowConflict("stored report body was not found")
            for stored_report_id, body, metadata_json in report_rows:
                self._regenerate_stored_report(
                    workflow_id,
                    stored_report_id,
                    as_of,
                    body,
                    metadata_json,
                )
        return result

    @staticmethod
    def _stored_report_model(
        report_type: str,
    ) -> type[EventUpdate | PortfolioBrief | IndustryLandscape | EmergingCompanyMonitor]:
        models = {
            "event_update": EventUpdate,
            "portfolio_brief": PortfolioBrief,
            "industry_landscape": IndustryLandscape,
            "emerging_monitor": EmergingCompanyMonitor,
        }
        try:
            return models[report_type]
        except KeyError:
            raise WorkflowConflict("stored report type is unsupported") from None

    @staticmethod
    def _contains_active_markup(value: object) -> bool:
        if isinstance(value, BaseModel):
            return RuntimeWorkflowService._contains_active_markup(
                value.model_dump(mode="python")
            )
        if isinstance(value, Mapping):
            return any(
                RuntimeWorkflowService._contains_active_markup(item)
                for item in value.values()
            )
        if isinstance(value, (tuple, list)):
            return any(
                RuntimeWorkflowService._contains_active_markup(item)
                for item in value
            )
        return isinstance(value, str) and _ACTIVE_MARKUP.search(value) is not None

    def _trusted_report(
        self,
        workflow_id: str,
        report_id: str,
        body: str,
        metadata_json: str,
    ) -> tuple[
        EventUpdate | PortfolioBrief | IndustryLandscape | EmergingCompanyMonitor,
        _StoredReportProvenance,
    ]:
        if not isinstance(body, str) or not isinstance(metadata_json, str):
            raise WorkflowConflict("stored report payload is invalid")
        try:
            provenance = _StoredReportProvenance.model_validate_json(
                metadata_json, strict=True
            )
        except Exception:
            raise WorkflowConflict("stored report provenance is invalid") from None
        if provenance.workflow_id != workflow_id:
            raise WorkflowConflict("stored report provenance is invalid")
        canonical_metadata = json.dumps(
            provenance.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        if metadata_json != canonical_metadata:
            raise WorkflowConflict("stored report provenance is not canonical")
        digest = hashlib.sha256(body.encode("utf-8", errors="strict")).hexdigest()
        if digest != provenance.payload_sha256:
            raise WorkflowConflict("stored report attestation is invalid")
        model_type = self._stored_report_model(provenance.report_type)
        try:
            report = model_type.model_validate_json(body, strict=True)
        except Exception:
            raise WorkflowConflict("stored report payload is invalid") from None
        canonical_body = json.dumps(
            report.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        if canonical_body != body:
            raise WorkflowConflict("stored report payload is not canonical")
        if (
            report.metadata.report_id != report_id
            or report.metadata.report_type != provenance.report_type
            or provenance.html_filename
            != ReportRenderer._filename(
                report.metadata.report_type,
                report.metadata.report_id,
                report.metadata.as_of,
            )
            + ".html"
            or self._contains_active_markup(report)
        ):
            raise WorkflowConflict("stored report payload is unsafe")
        with self.store.connect() as connection:
            task = connection.execute(
                "SELECT task.stage, task.outcome_json, run.as_of "
                "FROM workflow_tasks AS task JOIN workflow_runs AS run "
                "ON run.workflow_id = task.workflow_id "
                "WHERE task.workflow_id = ? AND task.task_id = ? "
                "AND task.state = 'completed'",
                (workflow_id, provenance.task_id),
            ).fetchone()
        if task is None or task[1] is None:
            raise WorkflowConflict("stored report provenance is invalid")
        try:
            outcome = StageOutcome.model_validate_json(task[1])
        except Exception:
            raise WorkflowConflict("stored report provenance is invalid") from None
        if (
            task[0] != _REPORT_ORIGIN_STAGES[provenance.report_type]
            or _parse_utc(task[2]) != report.metadata.as_of
            or outcome.report_id != report_id
            or outcome.result_ref != report_id
            or outcome.result_hash != provenance.payload_sha256
        ):
            raise WorkflowConflict("stored report provenance is invalid")
        return report, provenance

    def _publication_privacy_context(
        self,
        workflow_id: str,
        provenance: _StoredReportProvenance,
    ) -> PublicationPrivacyContext:
        with self.store.connect() as connection:
            task = connection.execute(
                "SELECT outcome_json FROM workflow_tasks "
                "WHERE workflow_id = ? AND stage = 'portfolio' "
                "AND state = 'completed'",
                (workflow_id,),
            ).fetchone()
            portfolio_ref = None
            if task is not None and task[0] is not None:
                portfolio_ref = StageOutcome.model_validate_json(task[0]).result_ref
            if provenance.portfolio_snapshot_ref != portfolio_ref:
                raise WorkflowConflict("stored report portfolio provenance is invalid")
            if portfolio_ref is None:
                return PublicationPrivacyContext(
                    mode="no_sensitive_data",
                    sensitive_literals=(),
                    account_identifiers=(),
                    portfolio_values=(),
                )
            snapshot = connection.execute(
                "SELECT nav, cash, account_ref FROM portfolio_snapshots "
                "WHERE snapshot_id = ?",
                (portfolio_ref,),
            ).fetchone()
            positions = connection.execute(
                "SELECT quantity, market_value, cost_basis FROM positions "
                "WHERE snapshot_id = ? ORDER BY position_id",
                (portfolio_ref,),
            ).fetchall()
        if snapshot is None:
            if portfolio_ref.startswith("snapshot-synthetic-"):
                return PublicationPrivacyContext(
                    mode="no_sensitive_data",
                    sensitive_literals=(),
                    account_identifiers=(),
                    portfolio_values=(),
                )
            raise WorkflowConflict("stored report portfolio source is unavailable")
        values = [Decimal(snapshot[0]), Decimal(snapshot[1])]
        for quantity, market_value, cost_basis in positions:
            values.extend((Decimal(quantity), Decimal(market_value)))
            if cost_basis is not None:
                values.append(Decimal(cost_basis))
        portfolio_values = tuple(
            sorted({abs(value) for value in values if value != 0})
        )
        return PublicationPrivacyContext(
            mode="enforced",
            sensitive_literals=(),
            account_identifiers=(snapshot[2],),
            portfolio_values=portfolio_values,
        )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _regenerate_stored_report(
        self,
        workflow_id: str,
        report_id: str,
        as_of: datetime,
        body: str,
        metadata_json: str,
    ) -> Path:
        report, provenance = self._trusted_report(
            workflow_id, report_id, body, metadata_json
        )
        privacy_context = self._publication_privacy_context(workflow_id, provenance)
        report_root = self.config.report_dir.resolve()
        report_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(
            (
                f"{report_id}:{as_of.date().isoformat()}:"
                f"{provenance.payload_sha256}"
            ).encode("utf-8")
        ).hexdigest()
        destination = report_root / f"regenerated-{digest[:24]}"
        with tempfile.TemporaryDirectory(
            dir=report_root, prefix=".regenerate-"
        ) as staging_name:
            staging = Path(staging_name)
            renderer = ReportRenderer(staging, privacy_context=privacy_context)
            render_method = getattr(
                renderer, f"render_{provenance.report_type}"
            )
            artifact = render_method(report)
            if artifact.pdf_path is None:
                status_path = staging / "pdf-status.txt"
                with status_path.open("wb") as handle:
                    handle.write(
                        (artifact.pdf_error or "pdf_render_failed").encode("ascii")
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
            self._fsync_directory(staging)
            if destination.exists():
                html_path = destination / artifact.html_path.name
                staged_files = tuple(sorted(staging.iterdir()))
                destination_files = tuple(sorted(destination.iterdir()))
                if (
                    not destination.is_dir()
                    or destination.is_symlink()
                    or any(
                        path.is_symlink() or not path.is_file()
                        for path in destination_files
                    )
                    or tuple(path.name for path in destination_files)
                    != tuple(path.name for path in staged_files)
                    or not html_path.is_file()
                    or html_path.read_text(encoding="utf-8") != artifact.html
                    or any(
                        destination_path.read_bytes()
                        != staged_path.read_bytes()
                        for destination_path, staged_path in zip(
                            destination_files, staged_files, strict=True
                        )
                    )
                ):
                    raise WorkflowConflict("regenerated report output conflicts")
                return html_path
            os.replace(staging, destination)
            self._fsync_directory(report_root)
        return destination / artifact.html_path.name


__all__ = [
    "FlexSyncClient",
    "RuntimeFactories",
    "RuntimeCancellationSource",
    "RuntimeStageAdapter",
    "RuntimeStageRunner",
    "RuntimeWorkflowService",
]
