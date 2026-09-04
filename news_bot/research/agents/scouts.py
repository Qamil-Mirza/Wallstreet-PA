"""Event and emerging-company scout roles."""

from ..models import AgentRole
from .base import BoundedAgent
from .contracts import (
    AgentContractError,
    EmergingScoutInput,
    EmergingScoutOutput,
    EventScoutInput,
    EventScoutOutput,
)


def _validate_ranking(candidate_ids, ranked) -> None:
    ranked_ids = tuple(item[0] for item in ranked)
    ranks = tuple(item[1] for item in ranked)
    if len(ranked_ids) != len(set(ranked_ids)) or not set(ranked_ids) <= set(candidate_ids):
        raise AgentContractError("scout ranked a candidate outside its task")
    if tuple(sorted(ranks)) != tuple(range(1, len(ranks) + 1)):
        raise AgentContractError("scout ranks must be unique and contiguous")


class EventScout(BoundedAgent[EventScoutInput, EventScoutOutput]):
    role = AgentRole.EVENT_SCOUT
    prompt_name = "event_scout"
    input_type = EventScoutInput
    output_type = EventScoutOutput

    def _validate_semantics(self, task_input, output) -> None:
        super()._validate_semantics(task_input, output)
        _validate_ranking(
            (item.event_id for item in task_input.events),
            ((item.event_id, item.rank) for item in output.ranked_events),
        )


class EmergingCompanyScout(BoundedAgent[EmergingScoutInput, EmergingScoutOutput]):
    role = AgentRole.EMERGING_COMPANY_SCOUT
    prompt_name = "emerging_scout"
    input_type = EmergingScoutInput
    output_type = EmergingScoutOutput

    def _validate_semantics(self, task_input, output) -> None:
        super()._validate_semantics(task_input, output)
        _validate_ranking(
            (item.signal_id for item in task_input.signals),
            ((item.signal_id, item.rank) for item in output.ranked_signals),
        )
