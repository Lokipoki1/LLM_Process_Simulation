"""
junior_clerk.py
---------------
Agente: Carlos — Junior Clerk
Responsabilidad: recepcion inicial y verificacion de documentos.
"""

from __future__ import annotations
from ..tools.schemas import IntakeApplication, CheckDocuments, ForwardCase
from ..state import ProcessState
from .base_agent import BaseAgent

SYSTEM_PROMPT = """
Eres Carlos, un Junior Clerk con 2 años de experiencia en el departamento
de préstamos de un banco. Tu personalidad es eficiente pero a veces apresurada.

TUS RESPONSABILIDADES:
1. Registrar la recepción de solicitudes de préstamo (IntakeApplication).
2. Verificar que los documentos estén completos (CheckDocuments).
3. Reenviar los casos completos al Senior Clerk (ForwardCase).

REGLAS DE COMPORTAMIENTO:
- Si los documentos están completos: siempre usa ForwardCase con priority="normal".
- Si el monto supera EUR 30,000: usa ForwardCase con priority="high".
- Si los documentos están incompletos: usa CheckDocuments con document_status="incomplete"
  y lista los documentos faltantes. NO reenvíes el caso.
- Nunca apruebas ni rechazas solicitudes — esa no es tu responsabilidad.
- Ejecuta exactamente UNA tool por turno.

ESTILO: Profesional y directo. Tus notas son breves y al punto.
"""


class JuniorClerk(BaseAgent):
    name = "junior_clerk"
    system_prompt = SYSTEM_PROMPT
    tool_schemas = [IntakeApplication, CheckDocuments, ForwardCase]

    def _resolve_next_agent(
        self, tool_name: str, tool_args: dict, state: ProcessState
    ) -> str | None:
        if tool_name == "ForwardCase":
            return "senior_clerk"
        # Si hay documentos incompletos, el caso se queda en espera
        return None

    def _resolve_status(self, tool_name: str) -> str:
        if tool_name == "ForwardCase":
            return "in_review"
        if tool_name == "CheckDocuments":
            return "pending"
        return "pending"
