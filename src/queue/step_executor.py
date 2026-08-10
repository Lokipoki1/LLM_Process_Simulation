"""
step_executor.py
----------------
Executes ONE agent step at a time so the DES engine can interleave
multiple cases across shared workers.

Agents and routing rules come from a ProcessDefinition, so this module
contains no domain knowledge: it does not know what a loan is, only that
some role acts and the case then moves somewhere.

Responsibilities
  - call the agent registered for a role
  - merge the agent's partial update into the case state
  - maintain the rework counter
  - resolve which role handles the next step

Not its responsibility
  - advancing simulation time
  - emitting XES entries
Both belong to the engine, the single owner of the timeline.

LangGraph was evaluated as the orchestrator and removed: its
run-to-completion model (`graph.invoke()` runs a whole case start to
finish) is incompatible with discrete event simulation, where cases
advance concurrently while competing for a limited pool of resources.
"""

from __future__ import annotations
import logging

from ..state import ProcessState
from ..process.definition import ProcessDefinition

logger = logging.getLogger("bps.executor")

# Fields that accumulate rather than overwrite
_APPEND_FIELDS = {"messages", "agent_history", "event_log"}


def merge_state(current: ProcessState, partial: dict) -> ProcessState:
    """Merge a partial update into the state; list fields are appended."""
    merged = dict(current)
    for key, value in partial.items():
        if key in _APPEND_FIELDS and isinstance(value, list):
            merged[key] = list(merged.get(key, [])) + value
        else:
            merged[key] = value
    return ProcessState(**merged)


class StepExecutor:
    """
    Holds one agent instance per role and runs a single step on demand.

    Agents are shared across all workers of the same role: junior_clerk_1
    and junior_clerk_2 are two resources drawing on the same persona, the
    way two clerks follow the same job description.
    """

    def __init__(self, llm, process: ProcessDefinition):
        self.process = process
        self._agents = {
            role: agent_cls(llm)
            for role, agent_cls in process.agent_classes.items()
        }

    def resolve_next_role(self, state: ProcessState) -> str | None:
        """
        Which role handles the next step, or None if the case is finished.

        An agent that returns next_agent=None on an open case is
        continuing its own sequence.
        """
        if state["status"] in ("approved", "rejected"):
            return None

        roles = self.process.agent_classes

        next_agent = state.get("next_agent")
        if next_agent in roles:
            return next_agent

        current = state.get("current_agent")
        if current in roles:
            return current

        return None

    def execute_step(
        self, state: ProcessState, role: str
    ) -> tuple[ProcessState, str | None]:
        """
        Run one step: the agent acts, the state is merged.

        Returns (updated_state, next_role); next_role is None when the
        case is complete.
        """
        agent = self._agents.get(role)
        if agent is None:
            logger.error("Unknown role: %s", role)
            return state, None

        partial_update = agent(state)
        state = merge_state(state, partial_update)

        # Rework: a case handed back to the entry role by a later role
        entry = self.process.entry_role
        if (partial_update.get("next_agent") == entry
                and partial_update.get("current_agent") != entry):
            state = merge_state(state, {"rework_count": state["rework_count"] + 1})

        return state, self.resolve_next_role(state)
