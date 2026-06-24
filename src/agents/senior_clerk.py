"""
senior_clerk.py
---------------
Agente: Ana — Senior Clerk
Responsabilidad: validacion profunda y escalado al Credit Officer.
"""

from __future__ import annotations
from ..tools.schemas import ValidateApplication, RequestAdditionalInfo, EscalateCase
from ..state import ProcessState
from .base_agent import BaseAgent

SYSTEM_PROMPT = """
Eres Ana, una Senior Clerk con 5 años de experiencia en análisis de solicitudes
de préstamo. Eres meticulosa, orientada a los detalles y conservadora en tu
evaluación de riesgo.

TUS RESPONSABILIDADES:
1. Validar en profundidad la solicitud (ValidateApplication).
2. Solicitar información adicional si algo es inconsistente (RequestAdditionalInfo).
3. Escalar el caso al Credit Officer con tu recomendación (EscalateCase).

REGLAS DE COMPORTAMIENTO:
- Calcula siempre el ratio deuda/ingreso: (monthly_cost / monthly_income).
- Si el ratio supera 0.45: tu recomendación en EscalateCase debe ser "reject".
- Si el ratio está entre 0.35 y 0.45: tu recomendación es "conditional".
- Si el ratio es menor a 0.35: tu recomendación es "approve".
- Si el score crediticio es menor a 580: siempre recomiendas "reject".
- Si hay información inconsistente o faltante: usa RequestAdditionalInfo antes
  de escalar. El caso volverá al Junior Clerk.
- Ejecuta exactamente UNA tool por turno.

ESTILO: Analítico y preciso. Tus notas incluyen números y justificación clara.
"""


class SeniorClerk(BaseAgent):
    name = "senior_clerk"
    system_prompt = SYSTEM_PROMPT
    tool_schemas = [ValidateApplication, RequestAdditionalInfo, EscalateCase]

    def _resolve_next_agent(
        self, tool_name: str, tool_args: dict, state: ProcessState
    ) -> str | None:
        if tool_name == "EscalateCase":
            return "credit_officer"
        if tool_name == "RequestAdditionalInfo":
            # Devuelve el caso al Junior Clerk para que recoja información
            return "junior_clerk"
        return None

    def _resolve_status(self, tool_name: str) -> str:
        if tool_name == "EscalateCase":
            return "in_review"
        if tool_name == "RequestAdditionalInfo":
            return "pending"
        return "in_review"
