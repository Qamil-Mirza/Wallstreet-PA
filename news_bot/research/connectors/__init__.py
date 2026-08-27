"""Public research connector contracts and supported source adapters."""

from .base import (
    ConnectorBatch,
    ConnectorCheckpoint,
    ConnectorError,
    NormalizedResearchDocument,
    ResearchConnector,
    commit_connector_batch,
)
from .fmp import DatedPrice, FMPConfig, FMPConnector, FundamentalPacket
from .investor_relations import InvestorRelationsConfig, InvestorRelationsConnector
from .news import MarketAuxResearchConnector, RSSResearchConnector
from .sec import SECConfig, SECConnector, SECResult

__all__ = [
    "ConnectorBatch",
    "ConnectorCheckpoint",
    "ConnectorError",
    "DatedPrice",
    "FMPConfig",
    "FMPConnector",
    "FundamentalPacket",
    "InvestorRelationsConfig",
    "InvestorRelationsConnector",
    "MarketAuxResearchConnector",
    "NormalizedResearchDocument",
    "RSSResearchConnector",
    "ResearchConnector",
    "SECConfig",
    "SECConnector",
    "SECResult",
    "commit_connector_batch",
]
