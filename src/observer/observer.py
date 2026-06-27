"""
observer.py
-----------
ObserverModule: captura cada tool call y produce entradas XES válidas.

Se integra en el grafo como un nodo que corre DESPUÉS de cada agente.
Lee agent_history[-1] (la acción más reciente) y produce una XESEntry
que se añade automáticamente a event_log via operator.add.
"""

from __future__ import annotations
from datetime import datetime, timezone
from ..state import ProcessState, XESEntry, AgentAction
from ..clock.simulation_clock import SimulationClock


# Mapa de tool_name → nombre de actividad legible (para el event log)
TOOL_TO_ACTIVITY: dict[str, str] = {
    "IntakeApplication":       "A_INTAKE",
    "CheckDocuments":          "A_CHECK_DOCS",
    "ForwardCase":             "A_FORWARD",
    "ReturnApplicationEarly":  "O_RETURNED",
    "ValidateApplication":     "A_VALIDATE",
    "RequestAdditionalInfo":   "A_REQUEST_INFO",
    "EscalateCase":            "A_ESCALATE",
    "AssessRisk":              "A_ASSESS_RISK",
    "ApproveLoan":             "O_APPROVED",
    "RejectLoan":              "O_DECLINED",
}

# Mapa de agent_name → org:resource legible
AGENT_TO_RESOURCE: dict[str, str] = {
    "junior_clerk":   "Junior Clerk",
    "senior_clerk":   "Senior Clerk",
    "credit_officer": "Credit Officer",
}


def build_xes_entry(action: AgentAction) -> XESEntry:
    """
    Convierte un AgentAction en una XESEntry con los campos exactos
    que PM4Py necesita para importar el log sin transformaciones.
    """
    activity = TOOL_TO_ACTIVITY.get(action["tool_name"], action["tool_name"])
    resource = AGENT_TO_RESOURCE.get(action["agent_name"], action["agent_name"])

    timestamp_iso = datetime.fromtimestamp(
        action["sim_timestamp"], tz=timezone.utc
    ).isoformat()

    return XESEntry(
        case_concept_name=action["tool_input"].get("case_id", "UNKNOWN"),
        concept_name=activity,
        time_timestamp=timestamp_iso,
        org_resource=resource,
        lifecycle_transition="complete",
    )


def observer_node(state: ProcessState, clock: SimulationClock) -> dict:
    """
    Nodo LangGraph que corre después de cada agente.

    Lee la última acción del agent_history, avanza el reloj,
    y devuelve la nueva XESEntry para que operator.add la añada al log.

    Args:
        state:  estado actual del grafo
        clock:  instancia compartida del SimulationClock

    Returns:
        dict con las keys a actualizar en ProcessState
    """
    if not state["agent_history"]:
        return {}

    last_action: AgentAction = state["agent_history"][-1]

    # Avanzar reloj con la duración real de la actividad
    duration, new_ts = clock.advance_for_activity(last_action["tool_name"])

    # Actualizar el timestamp y duración en la acción
    last_action["sim_timestamp"] = new_ts
    last_action["real_duration"] = duration

    # Construir entrada XES
    xes_entry = build_xes_entry(last_action)

    return {
        "sim_clock": new_ts,
        "event_log": [xes_entry],   # operator.add lo concatena automáticamente
    }
