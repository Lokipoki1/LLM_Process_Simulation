"""
step_executor.py
----------------
Executes ONE agent step at a time so the DES engine can interleave
multiple cases across shared workers.

Replaces the earlier process_graph.py. LangGraph was evaluated as the
process orchestrator and removed: its run-to-completion execution model
(`graph.invoke()` runs an entire case from start to finish) is
incompatible with discrete event simulation, where several cases advance
concurrently while competing for a limited pool of agent resources.
Wrapping it in checkpoints and interrupts would have added a dependency
and a serialisation layer without changing a single simulation outcome.

The process topology it used to express now lives in two explicit places:
  - each agent's `_resolve_next_agent()`  -> where a case goes after a tool
  - VALID_TRANSITIONS in the engine        -> which of those moves are legal

Responsibilities of this module
  - call the right agent for a role
  - merge the agent's partial update into the case state
  - maintain the rework counter
  - resolve which role handles the next step

NOT its responsibility
  - advancing simulation time
  - emitting XES entries
Both belong to the DES engine, which is the single owner of the timeline.
"""

from __future__ import annotations
import logging

from ..state import ProcessState
from ..agents.junior_clerk import JuniorClerk
from ..agents.senior_clerk import SeniorClerk
from ..agents.credit_officer import CreditOfficer

logger = logging.getLogger("bps.executor")

ROLES = ("junior_clerk", "senior_clerk", "credit_officer")

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


def resolve_next_role(state: ProcessState) -> str | None:
    """
    Which role handles the next step, or None if the case is finished.

    An agent that returns next_agent=None while the case is still open is
    continuing its own sequence (e.g. the JC between intake and forward).
    """
    if state["status"] in ("approved", "rejected"):
        return None

    next_agent = state.get("next_agent")
    if next_agent in ROLES:
        return next_agent

    current = state.get("current_agent")
    if current in ROLES:
        return current

    return None


class StepExecutor:
    """
    Holds one instance of each agent and runs a single step on demand.

    Agents are shared across all workers of the same role: junior_clerk_1
    and junior_clerk_2 are two resources drawing on the same persona, the
    way two clerks follow the same job description.
    """

    def __init__(self, llm):
        self._agents = {
            "junior_clerk":   JuniorClerk(llm),
            "senior_clerk":   SeniorClerk(llm),
            "credit_officer": CreditOfficer(llm),
        }

    def execute_step(
        self, state: ProcessState, role: str
    ) -> tuple[ProcessState, str | None]:
        """
        Run one step: the agent acts, the state is merged.

        Args:
            state: current ProcessState for this case
            role:  which agent role acts now

        Returns:
            (updated_state, next_role); next_role is None when the case
            is complete.
        """
        agent = self._agents.get(role)
        if agent is None:
            logger.error("Unknown role: %s", role)
            return state, None

        partial_update = agent(state)
        state = merge_state(state, partial_update)

        # Rework: the SC handed the case back to the JC
        if (partial_update.get("next_agent") == "junior_clerk"
                and partial_update.get("current_agent") != "junior_clerk"):
            state = merge_state(state, {"rework_count": state["rework_count"] + 1})

        return state, resolve_next_role(state)


def make_initial_state(application: dict, credit_data: dict | None = None) -> ProcessState:
    """
    Build the initial state for a case.

    application: LoanApplication fields
                 (case_id, amount_requested, loan_goal, application_type)
    credit_data: CreditBureauData fields, hidden until the SC checks;
                 None for cases the bank never took to the offer stage.
    """
    return ProcessState(
        application=application,
        credit_bureau_data=credit_data,
        credit_checked=False,
        status="pending",
        current_agent="junior_clerk",
        messages=[],
        agent_history=[],
        event_log=[],
        sim_clock=0.0,
        rework_count=0,
        rejection_reason=None,
        next_agent=None,
    )
