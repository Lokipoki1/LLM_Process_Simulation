"""
state.py
--------
Define el contrato de datos central del framework.
Todo nodo de LangGraph lee y escribe sobre este TypedDict.
"""

from __future__ import annotations
import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


# ─────────────────────────────────────────────
# 1. Datos del caso de préstamo
# ─────────────────────────────────────────────

class LoanCase(TypedDict):
    """
    Atributos de un caso de préstamo.
    Los campos reales (amount_requested, etc.) vienen del BPIC 2012/2017.
    Los campos sintéticos (credit_score, monthly_income) se samplearán
    con distribuciones estadísticas derivadas del mismo dataset.
    """
    case_id: str
    amount_requested: float
    loan_goal: str                   # e.g. "car", "home_improvement"
    number_of_terms: int             # meses
    monthly_cost: float

    # Campos sintéticos (generados estadísticamente)
    credit_score: int                # 300–850
    monthly_income: float
    applicant_id: str


# ─────────────────────────────────────────────
# 2. Registro de acciones de agentes
# ─────────────────────────────────────────────

class AgentAction(TypedDict):
    """
    Captura cada tool call que ejecuta un agente.
    El ObserverModule convierte esto en una entrada XES.
    """
    agent_name: str                  # "junior_clerk" | "senior_clerk" | "credit_officer"
    tool_name: str                   # nombre de la tool Pydantic ejecutada
    tool_input: dict                 # parámetros con los que se llamó
    tool_output: dict                # resultado estructurado devuelto
    sim_timestamp: float             # tiempo lógico de simulación (segundos)
    real_duration: float             # duración samplea de BPIC (segundos)


# ─────────────────────────────────────────────
# 3. Entrada del event log (formato XES)
# ─────────────────────────────────────────────

class XESEntry(TypedDict):
    """
    Estructura mínima compatible con PM4Py / IEEE XES.
    Nombres de campo exactos que PM4Py espera para importación.
    """
    case_concept_name: str           # case:concept:name  → case_id
    concept_name: str                # concept:name       → nombre de la actividad
    time_timestamp: str              # time:timestamp     → ISO 8601
    org_resource: str                # org:resource       → nombre del agente
    lifecycle_transition: str        # "complete" por defecto


# ─────────────────────────────────────────────
# 4. Estado global del proceso (ProcessState)
# ─────────────────────────────────────────────

class ProcessState(TypedDict):
    """
    Estado compartido que fluye a través del grafo LangGraph.

    Convención de anotaciones:
    - Annotated[list, operator.add] → LangGraph hace append automático
      cuando un nodo devuelve una lista parcial (no reemplaza la lista completa).
    - Campos sin Annotated → el nodo devuelve el valor nuevo completo.
    """

    # ── Caso activo ──────────────────────────
    case: LoanCase
    status: str                      # "pending" | "in_review" | "approved" | "rejected"
    current_agent: str               # nombre del agente que tiene el turno

    # ── Memoria conversacional ───────────────
    # Lista de mensajes LangChain (HumanMessage, AIMessage, ToolMessage).
    # operator.add hace que cada nodo solo devuelva los mensajes nuevos
    # y LangGraph los concatena al historial global automáticamente.
    messages: Annotated[list, operator.add]

    # ── Historial de acciones (append-only) ──
    # Fuente de verdad para el ObserverModule.
    agent_history: Annotated[list[AgentAction], operator.add]

    # ── Reloj lógico de simulación ───────────
    # Tiempo en segundos desde el inicio de la simulación.
    # El ObserverModule lo avanza usando distribuciones del BPIC.
    sim_clock: float

    # ── Event log (append-only) ──────────────
    # Se construye en paralelo al agent_history.
    # Al final de la simulación se exporta a .xes vía PM4Py.
    event_log: Annotated[list[XESEntry], operator.add]

    # ── Control de flujo ─────────────────────
    revision_count: int              # cuántas veces volvió al Junior Clerk
    rejection_reason: Optional[str]  # si status == "rejected"
    next_agent: Optional[str]        # hint de routing para los conditional edges
