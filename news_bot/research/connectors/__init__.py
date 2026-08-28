"""Public research connector contracts and supported source adapters."""

from .base import (
    ConnectorBatch,
    ConnectorCheckpoint,
    ConnectorError,
    ConnectorStatus,
    EmergingSignal,
    NormalizedResearchDocument,
    ResearchConnector,
    SignalConnectorBatch,
    SignalResearchConnector,
    commit_connector_batch,
)
from .awards import SBIRConfig, SBIRConnector, USASpendingConfig, USASpendingConnector
from .clinical_trials import ClinicalTrialsConfig, ClinicalTrialsConnector
from .fmp import DatedPrice, FMPConfig, FMPConnector, FundamentalPacket
from .form_d import FormDConfig, FormDConnector
from .investor_relations import InvestorRelationsConfig, InvestorRelationsConnector
from .manual_import import ManualImportConnector
from .news import MarketAuxResearchConnector, RSSResearchConnector
from .sec import SECConfig, SECConnector, SECResult
from .uspto import USPTOConfig, USPTOConnector

__all__ = [
    "ConnectorBatch",
    "ConnectorCheckpoint",
    "ConnectorError",
    "ConnectorStatus",
    "EmergingSignal",
    "ClinicalTrialsConfig",
    "ClinicalTrialsConnector",
    "DatedPrice",
    "FMPConfig",
    "FMPConnector",
    "FundamentalPacket",
    "FormDConfig",
    "FormDConnector",
    "InvestorRelationsConfig",
    "InvestorRelationsConnector",
    "MarketAuxResearchConnector",
    "ManualImportConnector",
    "NormalizedResearchDocument",
    "RSSResearchConnector",
    "ResearchConnector",
    "SBIRConfig",
    "SBIRConnector",
    "SECConfig",
    "SECConnector",
    "SECResult",
    "SignalConnectorBatch",
    "SignalResearchConnector",
    "USASpendingConfig",
    "USASpendingConnector",
    "USPTOConfig",
    "USPTOConnector",
    "commit_connector_batch",
]
