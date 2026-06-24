"""
junior_clerk.py — v2
Secuencia determinista: IntakeApplication -> CheckDocuments -> ForwardCase
Compatible con base_agent v3 (requiere _get_next_tool_schema).
"""

from __future__ import annotations
from ..tools.schemas import IntakeApplication, CheckDocuments, ForwardCase
from ..state import ProcessState
from .base_agent import BaseAgent

SYSTEM_PROMPT = """
Eres Carlos, Junior Clerk con 2 anos de experiencia en prestamos bancarios.
Eres eficiente y directo. Sigues una secuencia fija de pasos.

SECUENCIA:
  Paso 1 — IntakeApplication: registra la recepcion del caso.
  Paso 2 — CheckDocuments: verifica que la documentacion este completa.
  Paso 3 — ForwardCase: envia al Senior Clerk. Usa priority="high" si monto > EUR 30000.

Completa cada tool con informacion concisa y relevante del caso.
Nunca apruebas ni rechazas solicitudes — esa no es tu responsabilidad.
"""

_SEQUENCE = [IntakeApplication, CheckDocuments, ForwardCase]


class JuniorClerk(BaseAgent):
    name = "junior_clerk"
    system_prompt = SYSTEM_PROMPT
    tool_schemas = _SEQUENCE

    def _get_next_tool_schema(self, state: ProcessState) -> type:
        step = self._get_agent_step(state)
        return _SEQUENCE[min(step, len(_SEQUENCE) - 1)]

    def _resolve_next_agent(self, tool_name: str, tool_args: dict, state: ProcessState) -> str | None:
        return "senior_clerk" if tool_name == "ForwardCase" else None

    def _resolve_status(self, tool_name: str, tool_args: dict) -> str:
        return "in_review" if tool_name == "ForwardCase" else "pending"
