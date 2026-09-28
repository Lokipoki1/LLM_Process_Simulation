"""
step_executor.py
----------------
Executes ONE agent step at a time so the engine can interleave cases.

Calls the role's agent, merges its update into the case state, counts
rework, and resolves the next role. Time and XES output belong to the
engine. Replaces LangGraph, whose run-to-completion model does not fit
discrete event simulation.
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
    """Holds one agent per role (shared by its workers) and runs single steps."""

    def __init__(self, llm, process: ProcessDefinition):
        self.process = process
        self._agents = {
            role: agent_cls(llm)
            for role, agent_cls in process.agent_classes.items()
        }

    def resolve_next_role(self, state: ProcessState) -> str | None:
        """Next role, or None if the case is finished. next_agent=None means the same role continues."""
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
