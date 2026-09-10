"""Institutional research report contracts, exhibit builders, and renderer."""

from .exhibits import (
    ExposureExhibit,
    ExposureRow,
    ScenarioMatrix,
    ScenarioRow,
    ValuationExhibit,
    ValuationRow,
    build_exposure_exhibit,
    build_scenario_matrix,
    build_valuation_exhibit,
)
from .models import (
    Citation,
    ConcentrationCorrelation,
    EmergingCompanyMonitor,
    EventUpdate,
    IndustryLandscape,
    NonClaimSection,
    PortfolioBrief,
    PortfolioNewsItem,
    RenderedReportArtifact,
    ReportMetadata,
    ReportSection,
    ResearchView,
    ResearchViewChange,
    RoundedExposure,
)
from .renderer import ReportRenderer

__all__ = [
    "Citation", "ConcentrationCorrelation", "EmergingCompanyMonitor",
    "EventUpdate", "ExposureExhibit", "ExposureRow", "IndustryLandscape",
    "NonClaimSection", "PortfolioBrief", "PortfolioNewsItem",
    "RenderedReportArtifact", "ReportMetadata", "ReportRenderer",
    "ReportSection", "ResearchView", "ResearchViewChange", "RoundedExposure",
    "ScenarioMatrix", "ScenarioRow", "ValuationExhibit", "ValuationRow",
    "build_exposure_exhibit", "build_scenario_matrix", "build_valuation_exhibit",
]
