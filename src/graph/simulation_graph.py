"""
simulation_graph.py — v3
Soporta OpenAI (gpt-4o-mini) y Ollama segun variables de entorno.
"""

from __future__ import annotations
import os
from functools import partial
from langgraph.graph import StateGraph, START, END

from ..state import ProcessState
from ..agents.junior_clerk import JuniorClerk
from ..agents.senior_clerk import SeniorClerk
from ..agents.credit_officer import CreditOfficer
from ..observer.observer import observer_node
from ..clock.simulation_clock import SimulationClock

MAX_STEPS = 15


def _build_llm(model: str, base_url: str, temperature: float):
    """Instancia el LLM correcto segun el modelo configurado."""
    if model.startswith("gpt-"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)


def route(state: ProcessState) -> str:
    if state["status"] in ("approved", "rejected"):
        return END
    if len(state["agent_history"]) >= MAX_STEPS:
        print(f"  [WARN] MAX_STEPS ({MAX_STEPS}) alcanzado — forzando END")
        return END

    next_agent = state.get("next_agent")
    current    = state.get("current_agent", "junior_clerk")

    # SC explicitly sent the case back to JC for a revision cycle → increment counter
    if next_agent == "junior_clerk":
        return "jc_revision"

    # Explicit handoff to other agents
    if next_agent in ("senior_clerk", "credit_officer"):
        return next_agent

    # next_agent is None: agent is continuing its own sequence (or LLM skipped a tool call)
    if current == "junior_clerk":
        return "jc_step"    # stay in JC without touching revision_count
    if current in ("senior_clerk", "credit_officer"):
        return current      # retry the same agent
    return END


def increment_revision(state: ProcessState) -> dict:
    return {"revision_count": state["revision_count"] + 1}


def build_graph(
    model: str           = "gpt-4o-mini",
    ollama_base_url: str = "http://localhost:11434",
    clock: SimulationClock | None = None,
    temp_tools: float    = 0.1,
    temp_cognitive: float = 0.7,
) -> "CompiledGraph":
    if clock is None:
        clock = SimulationClock()

    llm_proc   = _build_llm(model, ollama_base_url, temp_tools)
    llm_cog    = _build_llm(model, ollama_base_url, temp_cognitive)
    llm_sc_cog = _build_llm(model, ollama_base_url, 0.4)   # SC: lower temp → more selective
    junior  = JuniorClerk(llm_proc, llm_cog)
    senior  = SeniorClerk(llm_proc, llm_sc_cog)
    officer = CreditOfficer(llm_proc, llm_cog)
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
            "jc_revision":    "increment_revision",  # SC sent case back → count the revision
            "jc_step":        "junior_clerk",         # JC own sequence → no increment
            "senior_clerk":   "senior_clerk",
            "credit_officer": "credit_officer",
            END:              END,
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
