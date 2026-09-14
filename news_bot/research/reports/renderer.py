"""Sandboxed HTML and failure-safe PDF rendering for research reports."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterator
from urllib.parse import unquote, urlsplit

from jinja2 import PackageLoader, StrictUndefined, select_autoescape
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel, ValidationError

from .models import (
    EmergingCompanyMonitor,
    EventUpdate,
    IndustryLandscape,
    PortfolioBrief,
    PublicationPrivacyContext,
    RenderedReportArtifact,
    ReportPublicationError,
)


_FILENAME_CHARACTER = re.compile(r"[^A-Za-z0-9_.-]+")
_PDF_DIAGNOSTIC_LOCK = Lock()


@contextmanager
def _captured_pdf_diagnostics() -> Iterator[None]:
    """Keep backend/native diagnostics off the process stderr boundary."""
    with _PDF_DIAGNOSTIC_LOCK, tempfile.TemporaryFile(mode="w+b") as sink:
        saved_stderr = os.dup(2)
        try:
            sys.stderr.flush()
            os.dup2(sink.fileno(), 2)
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)


class ReportRenderer:
    """Render validated display models without granting templates I/O access."""

    def __init__(
        self,
        output_dir: Path,
        *,
        privacy_context: PublicationPrivacyContext,
    ) -> None:
        if not isinstance(privacy_context, PublicationPrivacyContext):
            raise TypeError("privacy_context must be a PublicationPrivacyContext")
        privacy_context = PublicationPrivacyContext.model_validate(
            privacy_context, strict=True
        )
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.privacy_context = privacy_context
        self.template_root = Path(__file__).with_name("templates").resolve()
        self.environment = SandboxedEnvironment(
            loader=PackageLoader("news_bot.research.reports", "templates"),
            undefined=StrictUndefined,
            autoescape=select_autoescape(("html", "xml"), default=True),
            enable_async=False,
        )
        # No user-controlled values become template names or filters.
        self.environment.globals.clear()

    def url_fetcher(self, url: str) -> dict[str, object]:
        """Allow WeasyPrint to load only files beneath the packaged template root."""
        parsed = urlsplit(url)
        if parsed.scheme != "file":
            raise ValueError("remote and non-local report resources are blocked")
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("network file resources are blocked")
        candidate = Path(unquote(parsed.path)).resolve()
        if not candidate.is_relative_to(self.template_root):
            raise ValueError("local report resource is outside the trusted package root")
        from weasyprint import default_url_fetcher

        return default_url_fetcher(url)

    def render_event_update(self, report: EventUpdate) -> RenderedReportArtifact:
        return self._render("event_update.html", self._revalidate(EventUpdate, report))

    def render_portfolio_brief(self, report: PortfolioBrief) -> RenderedReportArtifact:
        return self._render(
            "portfolio_brief.html", self._revalidate(PortfolioBrief, report)
        )

    def render_industry_landscape(
        self, report: IndustryLandscape
    ) -> RenderedReportArtifact:
        return self._render(
            "industry_landscape.html", self._revalidate(IndustryLandscape, report)
        )

    def render_emerging_monitor(
        self, report: EmergingCompanyMonitor
    ) -> RenderedReportArtifact:
        return self._render(
            "emerging_monitor.html", self._revalidate(EmergingCompanyMonitor, report)
        )

    def _revalidate(
        self, model_type: type[BaseModel], report: BaseModel
    ) -> BaseModel:
        """Re-run every validator because model_copy(update=...) is not validated."""
        try:
            validated = model_type.model_validate(report)
        except ValidationError:
            # Registered values still receive the stable publication-boundary error,
            # while unrelated model-copy bypasses retain their validation details.
            self.privacy_context.assert_safe(report)
            raise
        self.privacy_context.assert_safe(validated)
        return validated

    def _render(self, template_name: str, report: object) -> RenderedReportArtifact:
        metadata = report.metadata
        html = self.environment.get_template(template_name).render(report=report)
        stem = self._filename(metadata.report_type, metadata.report_id, metadata.as_of)
        html_path = self._contained_path(f"{stem}.html")
        pdf_path = self._contained_path(f"{stem}.pdf")
        self._atomic_write(html_path, html.encode("utf-8"))

        if not self._remove_existing_pdf(pdf_path):
            raise ReportPublicationError("report output conflict")

        pdf_error = None
        try:
            with _captured_pdf_diagnostics():
                self._atomic_pdf(pdf_path, html)
        except Exception as exc:  # HTML is independently useful.
            pdf_error = (
                "pdf_backend_unavailable"
                if isinstance(exc, ImportError)
                else "pdf_render_failed"
            )
            if not self._remove_existing_pdf(pdf_path):
                raise ReportPublicationError("report output conflict") from None

        current_pdf_path = None if pdf_error is not None else pdf_path

        return RenderedReportArtifact(
            report_id=metadata.report_id,
            report_type=metadata.report_type,
            as_of=metadata.as_of,
            html=html,
            html_path=html_path,
            pdf_path=current_pdf_path,
            pdf_error=pdf_error,
        )

    @classmethod
    def _remove_existing_pdf(cls, path: Path) -> bool:
        try:
            if path.is_symlink() or path.exists():
                path.unlink()
        except Exception:
            return False
        return True

    @staticmethod
    def _filename(report_type: str, report_id: str, as_of: object) -> str:
        raw = f"{report_type}-{report_id}-{as_of:%Y%m%d}"
        return _FILENAME_CHARACTER.sub("-", raw).strip(".-_")

    def _contained_path(self, filename: str) -> Path:
        candidate = (self.output_dir / filename).resolve()
        if candidate.parent != self.output_dir:
            raise ValueError("report output escaped its configured directory")
        return candidate

    @staticmethod
    def _atomic_write(destination: Path, payload: bytes) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _atomic_pdf(self, destination: Path, html: str) -> None:
        from weasyprint import HTML

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
            HTML(
                string=html,
                base_url=self.template_root.as_uri() + "/",
                url_fetcher=self.url_fetcher,
            ).write_pdf(str(temporary))
            if not temporary.read_bytes().startswith(b"%PDF"):
                raise ValueError("PDF renderer returned an invalid document")
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
