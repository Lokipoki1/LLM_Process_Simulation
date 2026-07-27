"""
process_graph.py
----------------
LangGraph process definition + single-step execution.

This module defines the loan application process as a LangGraph
StateGraph (the BPMN-equivalent directed graph), but exposes a
step-by-step interface that the DES engine can call.

Architecture:
  - LangGraph:  owns the process TOPOLOGY (nodes, edges, routing)
  - DES engine: owns the TEMPORAL orchestration (queues, schedules, clock)

This separation is a core contribution of the thesis: LangGraph
provides the structural semantics of the business process, while
the DES engine provides the temporal realism needed for valid
simulation.
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


# ─────────────────────────────────────────────
# Routing logic (the conditional edges of BPMN)
# ─────────────────────────────────────────────

def route(state: ProcessState) -> str:
    """
    BPMN gateway logic: determines the next node based on
    the current state. This IS the process definition —
    the control flow of the loan application.
    """
    if state["status"] in ("approved", "rejected"):
        return END

    if len(state["agent_history"]) >= MAX_STEPS:
        logger.warning("MAX_STEPS (%d) reached — forcing END", MAX_STEPS)
        return END

    next_agent = state.get("next_agent")
    current    = state.get("current_agent", "junior_clerk")

    if next_agent == "junior_clerk":
        return "rework_counter"

    if next_agent in ("senior_clerk", "credit_officer"):
        return next_agent

    if current in ("junior_clerk", "senior_clerk", "credit_officer"):
        return current

    return END


def increment_rework(state: ProcessState) -> dict:
    """Rework counter — incremented when SC sends case back to JC."""
    return {"revision_count": state["revision_count"] + 1}


# ─────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────

def build_process_graph(
    llm,
    clock: SimulationClock | None = None,
) -> "CompiledGraph":
    """
    Build the LangGraph StateGraph that defines the loan process.

    The graph can be used in two modes:
      1. Full execution:  graph.invoke(state)     — runs the entire case
      2. Step execution:  graph.stream(state)     — yields one step at a time

    The DES engine uses mode 2 for temporal interleaving.

    Args:
        llm:   the LangChain LLM instance (shared by all agents)
        clock: SimulationClock for the observer node

    Returns:
        Compiled LangGraph ready for invoke() or stream()
    """
    if clock is None:
        clock = SimulationClock()

    # Instantiate agents with the shared LLM
    junior  = JuniorClerk(llm)
    senior  = SeniorClerk(llm)
    officer = CreditOfficer(llm)
    obs     = partial(observer_node, clock=clock)

    # Define the process topology
    builder = StateGraph(ProcessState)

    # Nodes (the activities/roles in the process)
    builder.add_node("junior_clerk",    junior)
    builder.add_node("senior_clerk",    senior)
    builder.add_node("credit_officer",  officer)
    builder.add_node("observer",        obs)
    builder.add_node("rework_counter",  increment_rework)

    # Edges (the control flow)
    builder.add_edge(START,             "junior_clerk")
    builder.add_edge("junior_clerk",    "observer")
    builder.add_edge("senior_clerk",    "observer")
    builder.add_edge("credit_officer",  "observer")

    # Conditional edges (the BPMN gateways)
    builder.add_conditional_edges(
        "observer", route,
        {
            "junior_clerk":    "junior_clerk",
            "rework_counter":  "rework_counter",
            "senior_clerk":    "senior_clerk",
            "credit_officer":  "credit_officer",
            END:               END,
        },
    )
    builder.add_edge("rework_counter", "junior_clerk")

    compiled = builder.compile()
    logger.info("Process graph compiled: %d nodes", len(builder.nodes))
    return compiled


# ─────────────────────────────────────────────
# Single-step executor (used by the DES engine)
# ─────────────────────────────────────────────

class ProcessStepExecutor:
    """
    Wraps a compiled LangGraph to execute ONE agent step at a time.

    The DES engine calls execute_step() for each case when an agent
    becomes available. The executor runs the agent node + observer,
    then returns the updated state and the next role needed.

    This preserves LangGraph's state management (operator.add for
    append-only fields) while giving the DES engine per-step control.
    """

    def __init__(self, llm, clock: SimulationClock):
        self._agents = {
            "junior_clerk":   JuniorClerk(llm),
            "senior_clerk":   SeniorClerk(llm),
            "credit_officer": CreditOfficer(llm),
        }
        self._clock = clock

    def execute_step(
        self, state: ProcessState, role: str
    ) -> tuple[ProcessState, str | None]:
        """
        Execute one step: agent acts, observer records.

        Args:
            state: current ProcessState for this case
            role:  which agent role should act ("junior_clerk", etc.)

        Returns:
            (updated_state, next_role)
            next_role is None if the case is complete.
        """
        agent = self._agents.get(role)
        if not agent:
            logger.error("Unknown role: %s", role)
            return state, None

        # 1. Agent acts (LLM call)
        partial_update = agent(state)

        # 2. Merge into state (replicating LangGraph's operator.add)
        state = _merge_state(state, partial_update)

        # 3. Observer records (clock advance + XES entry)
        obs_update = observer_node(state, self._clock)
        state = _merge_state(state, obs_update)

        # 4. Determine next role (replicating route())
        next_role = _resolve_next_role(state)

        # 5. Handle rework counter
        if (partial_update.get("next_agent") == "junior_clerk"
                and partial_update.get("current_agent") != "junior_clerk"):
            state = _merge_state(state, {
                "revision_count": state["revision_count"] + 1
            })

        return state, next_role


# ── State merge (replicates LangGraph's operator.add) ──

_APPEND_FIELDS = {"messages", "agent_history", "event_log"}

def _merge_state(current: ProcessState, partial: dict) -> ProcessState:
    """Merge partial update into state, appending list fields."""
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


# ─────────────────────────────────────────────
# Initial state factory
# ─────────────────────────────────────────────

def make_initial_state(case: dict) -> ProcessState:
    return ProcessState(
        case=case, status="pending", current_agent="junior_clerk",
        messages=[], agent_history=[], sim_clock=0.0, event_log=[],
        revision_count=0, rejection_reason=None, next_agent=None,
    )
