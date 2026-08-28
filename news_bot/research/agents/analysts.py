"""Evidence, long-horizon industry, and fundamental analyst roles."""

import hashlib
import json

from ..models import AgentRole
from .base import BoundedAgent
from .contracts import (
    AgentContractError,
    EvidenceAnalystInput,
    EvidenceAnalystOutput,
    FundamentalAnalystInput,
    FundamentalAnalystOutput,
    IndustryStrategistInput,
    IndustryStrategistOutput,
    IneligibleSecurity,
)


class EvidenceAnalyst(BoundedAgent[EvidenceAnalystInput, EvidenceAnalystOutput]):
    role = AgentRole.EVIDENCE_ANALYST
    prompt_name = "evidence_analyst"
    input_type = EvidenceAnalystInput
    output_type = EvidenceAnalystOutput

    def _validate_response(self, task_input, response):
        output = super()._validate_response(task_input, response)
        claims = []
        for item in output.claims:
            canonical = json.dumps(
                {
                    "evidence_ids": item.evidence_ids,
                    "kind": item.kind.value,
                    "schema_version": "1",
                    "supporting_claim_ids": item.supporting_claim_ids,
                    "text": item.text,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            identifier = "claim_" + hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()
            claims.append(item.model_copy(update={"claim_id": identifier}))
        if len({item.claim_id for item in claims}) != len(claims):
            raise AgentContractError("evidence analyst produced duplicate claim content")
        return output.model_copy(
            update={"claims": tuple(sorted(claims, key=lambda item: item.claim_id))}
        )

    def _validate_semantics(self, task_input, output) -> None:
        super()._validate_semantics(task_input, output)
        evidence = set(task_input.evidence_ids)
        claims = set(task_input.claim_ids)
        for draft in output.claims:
            if not set(draft.evidence_ids) <= evidence:
                raise AgentContractError("claim cites unavailable passage evidence")
            if not set(draft.supporting_claim_ids) <= claims:
                raise AgentContractError("inference cites an unavailable supporting claim")


class IndustryStrategist(
    BoundedAgent[IndustryStrategistInput, IndustryStrategistOutput]
):
    role = AgentRole.INDUSTRY_STRATEGIST
    prompt_name = "industry_strategist"
    input_type = IndustryStrategistInput
    output_type = IndustryStrategistOutput

    def _validate_semantics(self, task_input, output) -> None:
        super()._validate_semantics(task_input, output)
        if any(item.horizon_years != task_input.horizon_years for item in output.scenarios):
            raise AgentContractError("scenario horizon does not match strategy task")


class FundamentalAnalyst(
    BoundedAgent[FundamentalAnalystInput, FundamentalAnalystOutput]
):
    role = AgentRole.FUNDAMENTAL_ANALYST
    prompt_name = "fundamental_analyst"
    input_type = FundamentalAnalystInput
    output_type = FundamentalAnalystOutput

    def _preflight(self, task_input) -> None:
        if not task_input.security.rating_eligible:
            raise IneligibleSecurity("security is not eligible for a research rating")

    def _validate_semantics(self, task_input, output) -> None:
        super()._validate_semantics(task_input, output)
        if (
            output.security_id != task_input.security.security_id
            or output.horizon_months != task_input.horizon_months
            or output.valuation.currency != task_input.security.currency
            or output.valuation.as_of > task_input.as_of
        ):
            raise AgentContractError("recommendation does not match eligible security task")
