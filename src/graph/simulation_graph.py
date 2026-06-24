"""
simulation_graph.py — v2
Fixes:
  - route() tiene limite de pasos (agent_history) para evitar loops infinitos
  - next_agent None se maneja correctamente
  - logica de routing mas defensiva
"""

from __future__ import annotations
from functools import partial
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from ..state import ProcessState
from ..agents.junior_clerk import JuniorClerk
from ..agents.senior_clerk import SeniorClerk
from ..agents.credit_officer import CreditOfficer
from ..observer.observer import observer_node
from ..clock.simulation_clock import SimulationClock

# Maximo de acciones antes de forzar END (safety net)
MAX_STEPS = 12


def route(state: ProcessState) -> str:
    """
    Routing desde el observer hacia el siguiente nodo.

    Orden de prioridad:
    1. Caso cerrado (approved / rejected) → END
    2. Limite de pasos alcanzado          → END
    3. next_agent explícito del agente    → ese agente
    4. Agente actual sin next_agent       → reintentar mismo agente
    5. Fallback                           → END
    """
    # 1. Caso cerrado
    if state["status"] in ("approved", "rejected"):
        return END

    # 2. Safety net: demasiados pasos sin resolver
    steps = len(state["agent_history"])
    if steps >= MAX_STEPS:
        print(f"  [WARN] MAX_STEPS ({MAX_STEPS}) alcanzado — forzando END")
        return END

    next_agent = state.get("next_agent")
    current    = state.get("current_agent", "junior_clerk")

    # 3. next_agent explícito y válido
    if next_agent in ("junior_clerk", "senior_clerk", "credit_officer"):
        return next_agent

    # 4. Sin next_agent: reintentar el agente actual
    #    (ocurre cuando el LLM no llamó una tool — el prompt lo fuerza a reintentar)
    if current in ("junior_clerk", "senior_clerk", "credit_officer"):
        print(f"  [INFO] Sin tool call — reintentando {current} (paso {steps}/{MAX_STEPS})")
        return current

    # 5. Fallback
    return END


def increment_revision(state: ProcessState) -> dict:
    return {"revision_count": state["revision_count"] + 1}


def build_graph(
    model: str = "llama3.1",
    ollama_base_url: str = "http://localhost:11434",
    clock: SimulationClock | None = None,
    temp_tools: float = 0.1,
) -> "CompiledGraph":
    if clock is None:
        clock = SimulationClock()

    llm = ChatOllama(
        model=model,
        base_url=ollama_base_url,
        temperature=temp_tools,
    )

    junior  = JuniorClerk(llm)
    senior  = SeniorClerk(llm)
    officer = CreditOfficer(llm)
    obs_node = partial(observer_node, clock=clock)

    builder = StateGraph(ProcessState)

    builder.add_node("junior_clerk",        junior)
    builder.add_node("senior_clerk",        senior)
    builder.add_node("credit_officer",      officer)
    builder.add_node("observer",            obs_node)
    builder.add_node("increment_revision",  increment_revision)

    # Todos los agentes pasan por el observer
    builder.add_edge(START,            "junior_clerk")
    builder.add_edge("junior_clerk",   "observer")
    builder.add_edge("senior_clerk",   "observer")
    builder.add_edge("credit_officer", "observer")

    # El observer decide el routing
    builder.add_conditional_edges(
        "observer",
        route,
        {
            "junior_clerk":    "increment_revision",
            "senior_clerk":    "senior_clerk",
            "credit_officer":  "credit_officer",
            END:               END,
        },
    )

    builder.add_edge("increment_revision", "junior_clerk")

    return builder.compile()


def make_initial_state(case: dict) -> ProcessState:
    return ProcessState(
        case=case,
        status="pending",
        current_agent="junior_clerk",
        messages=[],
        agent_history=[],
        sim_clock=0.0,
        event_log=[],
        revision_count=0,
        rejection_reason=None,
        next_agent=None,
    )
