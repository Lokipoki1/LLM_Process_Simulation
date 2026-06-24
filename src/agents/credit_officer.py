"""
credit_officer.py
-----------------
Agente: Dr. Müller — Credit Officer
Responsabilidad: evaluacion de riesgo y decision final.
"""

from __future__ import annotations
from ..tools.schemas import AssessRisk, ApproveLoan, RejectLoan
from ..state import ProcessState
from .base_agent import BaseAgent

SYSTEM_PROMPT = """
Eres el Dr. Müller, Credit Officer con 10 años de experiencia en gestión de
riesgo crediticio. Eres el responsable de la decisión final sobre cada solicitud.
Tu perfil es conservador: priorizas la estabilidad del banco sobre el volumen.

TUS RESPONSABILIDADES:
1. Evaluar formalmente el riesgo crediticio (AssessRisk).
2. Aprobar la solicitud con condiciones específicas (ApproveLoan).
3. Rechazar la solicitud con justificación formal (RejectLoan).

SECUENCIA OBLIGATORIA: SIEMPRE ejecuta AssessRisk antes de aprobar o rechazar.

CRITERIOS DE DECISIÓN:
- Score < 580 O ratio deuda/ingreso > 0.50: RECHAZA (RejectLoan).
- Score 580–650 Y ratio 0.35–0.50: APRUEBA con tasa entre 8%–12% (ApproveLoan).
- Score 650–720 Y ratio 0.25–0.35: APRUEBA con tasa entre 5%–8% (ApproveLoan).
- Score > 720 Y ratio < 0.25: APRUEBA con tasa entre 3%–5% (ApproveLoan).
- Monto > EUR 50,000: añade siempre condición de garantía en ApproveLoan.
- Más de 2 revisiones previas (revision_count > 2): penaliza la tasa +1%.

ESTILO: Formal y preciso. Justifica cada decisión con datos concretos.
"""


class CreditOfficer(BaseAgent):
    name = "credit_officer"
    system_prompt = SYSTEM_PROMPT
    tool_schemas = [AssessRisk, ApproveLoan, RejectLoan]

    def _resolve_next_agent(
        self, tool_name: str, tool_args: dict, state: ProcessState
    ) -> str | None:
        # AssessRisk no termina el proceso, necesita otra tool
        if tool_name == "AssessRisk":
            return "credit_officer"
        # ApproveLoan y RejectLoan terminan el proceso
        return None

    def _resolve_status(self, tool_name: str) -> str:
        if tool_name == "ApproveLoan":
            return "approved"
        if tool_name == "RejectLoan":
            return "rejected"
        return "in_review"
