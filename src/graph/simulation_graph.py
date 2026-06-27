"""
simulation_graph.py — v4 (autonomous agents)
─────────────────────────────────────────────
Grafo simplificado: un solo LLM, routing basado en next_agent.
Los agentes deciden su propia secuencia via tool calls.
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

logger = logging.getLogger("bps.graph")

MAX_STEPS = 20  # safety net


def _build_llm(model: str, base_url: str, temperature: float):
    if model.startswith("gpt-"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)


def route(state: ProcessState) -> str:
    # 1. Caso cerrado
    if state["status"] in ("approved", "rejected"):
        return END

    # 2. Safety net
    steps = len(state["agent_history"])
    if steps >= MAX_STEPS:
        logger.warning("MAX_STEPS (%d) reached — forcing END", MAX_STEPS)
        return END

    next_agent = state.get("next_agent")
    current    = state.get("current_agent", "junior_clerk")

    # 3. SC envía caso de vuelta al JC → incrementar revisión
    if next_agent == "junior_clerk":
        return "jc_revision"

    # 4. Handoff explícito a otro agente
    if next_agent in ("senior_clerk", "credit_officer"):
        return next_agent

    # 5. Agente continúa su propia secuencia (next_agent is None, status not terminal)
    if current in ("junior_clerk", "senior_clerk", "credit_officer"):
        return current

    return END


def increment_revision(state: ProcessState) -> dict:
    return {"revision_count": state["revision_count"] + 1}


def build_graph(
    model: str           = "llama3.1",
    ollama_base_url: str = "http://localhost:11434",
    clock: SimulationClock | None = None,
    temp_tools: float    = 0.3,
) -> "CompiledGraph":
    if clock is None:
        clock = SimulationClock()

    llm = _build_llm(model, ollama_base_url, temp_tools)

    junior  = JuniorClerk(llm)
    senior  = SeniorClerk(llm)
    officer = CreditOfficer(llm)
    obs     = partial(observer_node, clock=clock)

    builder = StateGraph(ProcessState)
    builder.add_node("junior_clerk",       junior)
    builder.add_node("senior_clerk",       senior)
    builder.add_node("credit_officer",     officer)
    builder.add_node("observer",           obs)
    builder.add_node("increment_revision", increment_revision)

    builder.add_edge(START,            "junior_clerk")
    builder.add_edge("junior_clerk",   "observer")
    builder.add_edge("senior_clerk",   "observer")
    builder.add_edge("credit_officer", "observer")

    builder.add_conditional_edges(
        "observer", route,
        {
            "junior_clerk":    "junior_clerk",
            "jc_revision":     "increment_revision",
            "senior_clerk":    "senior_clerk",
            "credit_officer":  "credit_officer",
            END:               END,
        },
    )
    builder.add_edge("increment_revision", "junior_clerk")
    return builder.compile()


def make_initial_state(case: dict) -> ProcessState:
    return ProcessState(
        case=case, status="pending", current_agent="junior_clerk",
        messages=[], agent_history=[], sim_clock=0.0, event_log=[],
        revision_count=0, rejection_reason=None, next_agent=None,
    )
