"""Institutional research report contracts, exhibit builders, and renderer."""

from .exhibits import (
    ExposureRow,
    ScenarioRow,
    ValuationRow,
    build_exposure_exhibit,
    build_scenario_matrix,
    build_valuation_exhibit,
)
from .models import (
    Citation,
    EmergingCompanyMonitor,
    EventUpdate,
    IndustryLandscape,
    PortfolioBrief,
    RenderedReportArtifact,
    ReportMetadata,
    ReportSection,
)
from .renderer import ReportRenderer

__all__ = [
    "Citation", "EmergingCompanyMonitor", "EventUpdate", "ExposureRow",
    "IndustryLandscape", "PortfolioBrief", "RenderedReportArtifact",
    "ReportMetadata", "ReportRenderer", "ReportSection", "ScenarioRow",
    "ValuationRow", "build_exposure_exhibit", "build_scenario_matrix",
    "build_valuation_exhibit",
]
