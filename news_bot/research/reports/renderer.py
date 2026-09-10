"""Sandboxed HTML and failure-safe PDF rendering for research reports."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

from jinja2 import PackageLoader, StrictUndefined, select_autoescape
from jinja2.sandbox import SandboxedEnvironment
from weasyprint import HTML, default_url_fetcher

from .models import (
    EmergingCompanyMonitor,
    EventUpdate,
    IndustryLandscape,
    PortfolioBrief,
    RenderedReportArtifact,
)


_FILENAME_CHARACTER = re.compile(r"[^A-Za-z0-9_.-]+")


class ReportRenderer:
    """Render validated display models without granting templates I/O access."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
        return default_url_fetcher(url)

    def render_event_update(self, report: EventUpdate) -> RenderedReportArtifact:
        return self._render("event_update.html", report)

    def render_portfolio_brief(self, report: PortfolioBrief) -> RenderedReportArtifact:
        return self._render("portfolio_brief.html", report)

    def render_industry_landscape(
        self, report: IndustryLandscape
    ) -> RenderedReportArtifact:
        return self._render("industry_landscape.html", report)

    def render_emerging_monitor(
        self, report: EmergingCompanyMonitor
    ) -> RenderedReportArtifact:
        return self._render("emerging_monitor.html", report)

    def _render(self, template_name: str, report: object) -> RenderedReportArtifact:
        metadata = report.metadata
        html = self.environment.get_template(template_name).render(report=report)
        stem = self._filename(metadata.report_type, metadata.report_id, metadata.as_of)
        html_path = self._contained_path(f"{stem}.html")
        pdf_path = self._contained_path(f"{stem}.pdf")
        self._atomic_write(html_path, html.encode("utf-8"))

        pdf_error: str | None = None
        try:
            self._atomic_pdf(pdf_path, html)
        except Exception as exc:  # HTML is an independently useful approved artifact.
            pdf_error = f"{type(exc).__name__}: {exc}"
            pdf_path = None

        return RenderedReportArtifact(
            report_id=metadata.report_id,
            report_type=metadata.report_type,
            as_of=metadata.as_of,
            html=html,
            html_path=html_path,
            pdf_path=pdf_path,
            pdf_error=pdf_error,
        )

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
