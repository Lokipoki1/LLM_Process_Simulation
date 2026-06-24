"""
credit_officer.py — v4
Paso 1: AssessRisk       (procedimental)
Paso 2: ApproveLoan o RejectLoan (decision cognitiva del LLM)
Fix: si el LLM alucina AssessRisk en paso 2, se ignora y se fuerza decision.
"""

from __future__ import annotations
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from ..tools.schemas import AssessRisk, ApproveLoan, RejectLoan
from ..state import ProcessState, AgentAction
from .base_agent import BaseAgent, _make_tool_func

SYSTEM_PROMPT = """
Eres el Dr. Müller, Credit Officer con 10 años de experiencia. Perfil conservador.

PASO 1 — AssessRisk: evalúa score, ratio y factores cualitativos. Clasifica riesgo.

PASO 2 — Decisión final basada en tu evaluación:

  ApproveLoan cuando el riesgo es manejable:
    - Fija monto aprobado, tasa de interés y condiciones específicas.
    - Score > 720 y ratio < 0.25 → tasa 3-5%
    - Score 650-720 y ratio < 0.35 → tasa 5-8%
    - Score 580-649 y ratio < 0.45 → tasa 8-12%

  RejectLoan cuando el riesgo es inaceptable:
    - Score < 580 O ratio > 0.50 → SIEMPRE rechazar
    - Lista razones cuantificadas y formales.

Tu decisión refleja criterio profesional, no solo algoritmos.
Un caso borderline puede ir en cualquier dirección según tu lectura del caso.
"""


class CreditOfficer(BaseAgent):
    name = "credit_officer"
    system_prompt = SYSTEM_PROMPT
    tool_schemas = [AssessRisk, ApproveLoan, RejectLoan]

    def _get_next_tool_schema(self, state: ProcessState) -> type:
        return AssessRisk if self._get_agent_step(state) == 0 else ApproveLoan

    def __call__(self, state: ProcessState) -> dict:
        if self._get_agent_step(state) == 0:
            return super().__call__(state)
        return self._cognitive_decision(state)

    def _cognitive_decision(self, state: ProcessState) -> dict:
        case  = state["case"]
        ratio = case["monthly_cost"] / case["monthly_income"]

        # Contexto de evaluacion previa
        risk_context = ""
        assessments = [a for a in state["agent_history"]
                       if a["agent_name"] == "credit_officer" and a["tool_name"] == "AssessRisk"]
        if assessments:
            last = assessments[-1]["tool_output"]
            risk_context = (
                f"Tu evaluación previa: riesgo={last.get('risk_category','?')} | "
                f"factores: {last.get('risk_factors', [])}\n"
            )

        senior_rec = ""
        escalations = [a for a in state["agent_history"]
                       if a["agent_name"] == "senior_clerk" and a["tool_name"] == "EscalateCase"]
        if escalations:
            rec = escalations[-1]["tool_output"].get("recommendation", "")
            senior_rec = f"Recomendación del Senior Clerk: {rec}\n"

        decision_tools = [ApproveLoan, RejectLoan]
        lc_tools = [
            StructuredTool.from_function(
                func=_make_tool_func(s), name=s.__name__,
                description=s.__doc__ or s.__name__, args_schema=s,
            )
            for s in decision_tools
        ]

        messages = [
            SystemMessage(content=self.system_prompt),
            *state["messages"][-4:],
            {
                "role": "user",
                "content": (
                    f"CASO: {case['case_id']} | Monto: EUR {case['amount_requested']:,.0f}\n"
                    f"Score: {case['credit_score']} | Ingreso: EUR {case['monthly_income']:,.0f}/mes\n"
                    f"Ratio deuda/ingreso: {ratio:.2f} | Plazo: {case['number_of_terms']} meses\n"
                    f"{risk_context}{senior_rec}"
                    f"Revisiones: {state['revision_count']}\n\n"
                    f"IMPORTANTE: Llama ApproveLoan O RejectLoan. No llames AssessRisk."
                ),
            },
        ]

        response: AIMessage = self._base_llm.bind_tools(lc_tools).invoke(messages)

        # ── Sin tool call o tool incorrecta → fallback basado en criterios ──
        tool_call = None
        if response.tool_calls:
            tc = response.tool_calls[0]
            # Ignorar si el LLM alucina AssessRisk en este paso
            if tc["name"] in ("ApproveLoan", "RejectLoan"):
                tool_call = tc

        if tool_call is None:
            # Fallback determinista: aplicar criterios del sistema prompt
            if case["credit_score"] < 580 or ratio > 0.50:
                tool_name = "RejectLoan"
                tool_args = {
                    "case_id": case["case_id"],
                    "rejection_reasons": [
                        f"Score crediticio insuficiente: {case['credit_score']}",
                        f"Ratio deuda/ingreso excesivo: {ratio:.2f}",
                    ],
                    "rejection_notes": "Rechazado por criterios de riesgo mínimos.",
                }
            else:
                rate = 0.04 if case["credit_score"] > 720 else \
                       0.065 if case["credit_score"] > 650 else 0.10
                tool_name = "ApproveLoan"
                tool_args = {
                    "case_id":         case["case_id"],
                    "approved_amount": case["amount_requested"],
                    "interest_rate":   rate,
                    "conditions":      [],
                    "approval_notes":  "Aprobado por criterios de riesgo estándar (fallback).",
                }
        else:
            tool_name = tool_call["name"]
            tool_args  = tool_call["args"]

        schema_cls  = {"ApproveLoan": ApproveLoan, "RejectLoan": RejectLoan}[tool_name]
        tool_output = {}
        try:
            tool_output = schema_cls(**tool_args).model_dump()
        except Exception as e:
            tool_output = {"error": str(e)}

        action: AgentAction = {
            "agent_name":    self.name,
            "tool_name":     tool_name,
            "tool_input":    tool_args,
            "tool_output":   tool_output,
            "sim_timestamp": state["sim_clock"],
            "real_duration": 0.0,
        }
        tool_msg_content = str(tool_output)
        tc_id = tool_call["id"] if tool_call else "fallback"
        tool_message = ToolMessage(content=tool_msg_content, tool_call_id=tc_id)

        return {
            "messages":      [response, tool_message],
            "agent_history": [action],
            "current_agent": self.name,
            "next_agent":    None,
            "status":        "approved" if tool_name == "ApproveLoan" else "rejected",
        }

    def _resolve_next_agent(self, tool_name: str, tool_args: dict, state: ProcessState) -> str | None:
        return "credit_officer" if tool_name == "AssessRisk" else None

    def _resolve_status(self, tool_name: str, tool_args: dict) -> str:
        if tool_name == "ApproveLoan": return "approved"
        if tool_name == "RejectLoan":  return "rejected"
        return "in_review"
