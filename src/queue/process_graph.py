"""
process_graph.py
----------------
LEGACY - not imported by the engine, and does not import as it stands
(it references `observer.observer_node`, which no longer exists).

Superseded by:
  - src/process/loan_application.py  the process definition
  - src/queue/step_executor.py       single-step execution

Kept as a record of the loan process expressed as a LangGraph
StateGraph (the BPMN-equivalent directed graph), whose routing
semantics the DES engine replicates.
"""

from __future__ import annotations
import logging
from functools import partial
from langgraph.graph import StateGraph, START, END

from ..state import ProcessState
from ..agents.junior_clerk import JuniorClerk
from ..agents.senior_clerk import SeniorClerk
from ..agents.credit_officer import CreditOfficer
from ..observer.observer import observer_node
from ..clock.simulation_clock import SimulationClock

logger = logging.getLogger("bps.process_graph")

MAX_STEPS = 20


# -------------------------------------------------
# Routing logic (the conditional edges / BPMN gateways)
# -------------------------------------------------

def route(state: ProcessState) -> str:
    """
    BPMN gateway logic: determines the next node from the current state.
    This IS the process definition — the control flow of the loan application.
    """
    if state["status"] in ("approved", "rejected"):
        return END

    if len(state["agent_history"]) >= MAX_STEPS:
        logger.warning("MAX_STEPS (%d) reached — forcing END", MAX_STEPS)
        return END

    next_agent = state.get("next_agent")
    current = state.get("current_agent", "junior_clerk")

    if next_agent == "junior_clerk":
        return "rework_counter"

    if next_agent in ("senior_clerk", "credit_officer"):
        return next_agent

    if current in ("junior_clerk", "senior_clerk", "credit_officer"):
        return current

    return END


def increment_rework(state: ProcessState) -> dict:
    """Rework counter — incremented when the SC sends a case back to the JC."""
    return {"rework_count": state["rework_count"] + 1}


# -------------------------------------------------
# Graph builder (documentation / visualisation artifact)
# -------------------------------------------------

def build_process_graph(
    llm,
    clock: SimulationClock | None = None,
) -> "CompiledGraph":
    """Build the LangGraph StateGraph of the loan process (not used at runtime)."""
    if clock is None:
        clock = SimulationClock()

    junior = JuniorClerk(llm)
    senior = SeniorClerk(llm)
    officer = CreditOfficer(llm)
    obs = partial(observer_node, clock=clock)

    builder = StateGraph(ProcessState)

    # Nodes (the roles/activities in the process)
    builder.add_node("junior_clerk", junior)
    builder.add_node("senior_clerk", senior)
    builder.add_node("credit_officer", officer)
    builder.add_node("observer", obs)
    builder.add_node("rework_counter", increment_rework)

    # Edges (the control flow)
    builder.add_edge(START, "junior_clerk")
    builder.add_edge("junior_clerk", "observer")
    builder.add_edge("senior_clerk", "observer")
    builder.add_edge("credit_officer", "observer")

    # Conditional edges (the BPMN gateways)
    builder.add_conditional_edges(
        "observer", route,
        {
            "junior_clerk":   "junior_clerk",
            "rework_counter": "rework_counter",
            "senior_clerk":   "senior_clerk",
            "credit_officer": "credit_officer",
            END:              END,
        },
    )
    builder.add_edge("rework_counter", "junior_clerk")

    compiled = builder.compile()
    logger.info("Process graph compiled: %d nodes", len(builder.nodes))
    return compiled


# -------------------------------------------------
# Single-step executor (used by the DES engine)
# -------------------------------------------------

class ProcessStepExecutor:
    """Legacy single-step executor; see step_executor.StepExecutor."""

    def __init__(self, llm):
        self._agents = {
            "junior_clerk":   JuniorClerk(llm),
            "senior_clerk":   SeniorClerk(llm),
            "credit_officer": CreditOfficer(llm),
        }

    def execute_step(
        self, state: ProcessState, role: str
    ) -> tuple[ProcessState, str | None]:
        """Execute one step. Returns (updated_state, next_role)."""
        agent = self._agents.get(role)
        if not agent:
            logger.error("Unknown role: %s", role)
            return state, None

        # 1. Agent acts (LLM call)
        partial_update = agent(state)

        # 2. Merge into state (replicating LangGraph's operator.add)
        state = _merge_state(state, partial_update)

        # 3. Rework counter — SC sent the case back to the JC
        if (partial_update.get("next_agent") == "junior_clerk"
                and partial_update.get("current_agent") != "junior_clerk"):
            state = _merge_state(state, {
                "rework_count": state["rework_count"] + 1
            })

        # 4. Determine next role (replicating route())
        next_role = _resolve_next_role(state)

        return state, next_role


# -- State merge (replicates LangGraph's operator.add) ---

_APPEND_FIELDS = {"messages", "agent_history", "event_log"}


def _merge_state(current: ProcessState, partial: dict) -> ProcessState:
    """Merge a partial update into the state, appending list fields."""
    merged = dict(current)
    for key, value in partial.items():
        if key in _APPEND_FIELDS and isinstance(value, list):
            merged[key] = list(merged.get(key, [])) + value
        else:
            merged[key] = value
    return ProcessState(**merged)


def _resolve_next_role(state: ProcessState) -> str | None:
    """Determine which role handles the next step."""
    if state["status"] in ("approved", "rejected"):
        return None
    next_agent = state.get("next_agent")
    if next_agent in ("junior_clerk", "senior_clerk", "credit_officer"):
        return next_agent
    current = state.get("current_agent")
    if current in ("junior_clerk", "senior_clerk", "credit_officer"):
        return current
    return None


# -------------------------------------------------
# Initial state factory
# -------------------------------------------------

def make_initial_state(application: dict, credit_data: dict | None = None) -> ProcessState:
    """Create the initial state for a case."""
    return ProcessState(
        application=application,
        credit_bureau_data=credit_data,
        credit_checked=False,
        status="pending", current_agent="junior_clerk",
        messages=[], agent_history=[], sim_clock=0.0, event_log=[],
        rework_count=0, rejection_reason=None, next_agent=None,
    )
