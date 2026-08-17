"""
process_graph.py
----------------
LangGraph process definition + single-step execution.

This module defines the loan application process as a LangGraph
StateGraph (the BPMN-equivalent directed graph), and exposes a
step-by-step interface that the DES engine can call.

Architecture split:
  - LangGraph:  owns the process TOPOLOGY (nodes, edges, routing)
  - DES engine: owns the TEMPORAL orchestration (queues, schedules, clock)

IMPORTANT — clock ownership:
  `execute_step()` performs the agent's LLM call and merges the resulting
  state, but it does NOT advance any clock and does NOT emit XES entries.
  The DES engine is the single source of truth for simulation time: it
  samples the activity duration, schedules the completion event, and
  stamps the resulting XES entry with the real DES timestamp.

  Previously the observer was called here as well, which meant every step
  sampled its duration twice (once here, once in the engine) and the XES
  timestamps came from a global monotonic counter that knew nothing about
  queues or parallel workers. That made AED/CTD meaningless.
"""

from __future__ import annotations
import logging
from functools import partial
from langgraph.graph import StateGraph, START, END

from ..state import ProcessState
from ..agents.junior_clerk_ import JuniorClerk
from ..agents.senior_clerk import SeniorClerk
from ..agents.credit_officer import CreditOfficer
from ..observer.observer import observer_node
from ..clock.simulation_clock import SimulationClock

logger = logging.getLogger("bps.process_graph")

MAX_STEPS = 20


# ─────────────────────────────────────────────
# Routing logic (the conditional edges / BPMN gateways)
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Graph builder (documentation / visualisation artifact)
# ─────────────────────────────────────────────

def build_process_graph(
    llm,
    clock: SimulationClock | None = None,
) -> "CompiledGraph":
    """
    Build the LangGraph StateGraph that defines the loan process.

    This compiled graph is NOT used by the DES engine at runtime — its
    run-to-completion execution model (`invoke()` runs a whole case) is
    incompatible with discrete event simulation, where several cases run
    concurrently sharing the same agent resources.

    It is kept as the formal, inspectable definition of the process
    topology: it can be rendered to a diagram and it documents the
    routing semantics that the DES engine replicates.

    Args:
        llm:   the LangChain LLM instance (shared by all agents)
        clock: SimulationClock for the observer node

    Returns:
        Compiled LangGraph
    """
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


# ─────────────────────────────────────────────
# Single-step executor (used by the DES engine)
# ─────────────────────────────────────────────

class ProcessStepExecutor:
    """
    Executes ONE agent step at a time, so the DES engine can interleave
    multiple cases across shared workers.

    Responsibilities:
      - call the right agent for the role
      - merge the agent's partial update into the case state
        (replicating LangGraph's operator.add semantics)
      - resolve which role handles the next step

    NOT its responsibility:
      - advancing simulation time
      - emitting XES entries
    Both belong to the DES engine.
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
        Execute one step: the agent acts, the state is merged.

        Args:
            state: current ProcessState for this case
            role:  which agent role should act

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

        # 3. Rework counter — SC sent the case back to the JC
        if (partial_update.get("next_agent") == "junior_clerk"
                and partial_update.get("current_agent") != "junior_clerk"):
            state = _merge_state(state, {
                "rework_count": state["rework_count"] + 1
            })

        # 4. Determine next role (replicating route())
        next_role = _resolve_next_role(state)

        # NOTE: no clock advance and no XES entry here — the DES engine
        # stamps the timestamp once it knows when the activity finished.

        return state, next_role


# ── State merge (replicates LangGraph's operator.add) ──

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


# ─────────────────────────────────────────────
# Initial state factory
# ─────────────────────────────────────────────

def make_initial_state(application: dict, credit_data: dict | None = None) -> ProcessState:
    """
    Create the initial state for a case.

    application: LoanApplication fields
                 (case_id, amount_requested, loan_goal, application_type)
    credit_data: CreditBureauData fields — hidden until the SC checks;
                 None for cases the bank never took to the offer stage.
    """
    return ProcessState(
        application=application,
        credit_bureau_data=credit_data,
        credit_checked=False,
        status="pending", current_agent="junior_clerk",
        messages=[], agent_history=[], sim_clock=0.0, event_log=[],
        rework_count=0, rejection_reason=None, next_agent=None,
    )
