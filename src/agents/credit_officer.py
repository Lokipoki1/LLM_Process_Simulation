"""
credit_officer.py — v3
Paso 1: AssessRisk       (procedimental — siempre primero)
Paso 2: LLM elige ApproveLoan o RejectLoan con sus condiciones (decision cognitiva)
"""

from __future__ import annotations
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from ..tools.schemas import AssessRisk, ApproveLoan, RejectLoan
from ..state import ProcessState
from .base_agent import BaseAgent, _make_tool_func

SYSTEM_PROMPT = """
Eres el Dr. Muller, Credit Officer con 10 anos de experiencia. Perfil conservador.

PASO 1 — AssessRisk:
  Evalua el score crediticio, ratio deuda/ingreso y factores cualitativos del caso.
  Clasifica el riesgo como "low", "medium" o "high" y lista los factores determinantes.

PASO 2 — Decision final (tu criterio profesional):
  Con base en tu evaluacion de riesgo, decides:

  ApproveLoan — cuando el riesgo es manejable:
    - Fija el monto aprobado (puede ser menor al solicitado si el riesgo lo justifica)
    - Establece la tasa de interes segun el riesgo evaluado
    - Incluye condiciones si el caso es borderline (garantias, seguros, etc.)

  RejectLoan — cuando el riesgo es inaceptable:
    - Lista las razones especificas y cuantificadas
    - Se claro y formal

Tu decision NO es solo algoritmica — considera el contexto completo del caso,
la recomendacion del Senior Clerk y tu experiencia profesional.
Un caso borderline puede ir en cualquier direccion segun tu criterio.
"""


class CreditOfficer(BaseAgent):
    name = "credit_officer"
    system_prompt = SYSTEM_PROMPT
    tool_schemas = [AssessRisk, ApproveLoan, RejectLoan]

    def _get_next_tool_schema(self, state: ProcessState) -> type:
        step = self._get_agent_step(state)
        return AssessRisk if step == 0 else ApproveLoan

    def __call__(self, state: ProcessState) -> dict:
        step = self._get_agent_step(state)
        if step == 0:
            return super().__call__(state)
        return self._cognitive_decision(state)

    def _cognitive_decision(self, state: ProcessState) -> dict:
        """
        El Credit Officer toma la decision final con criterio propio.
        Esta es la decision cognitiva mas importante del proceso.
        """
        decision_tools = [ApproveLoan, RejectLoan]
        lc_tools = [
            StructuredTool.from_function(
                func=_make_tool_func(s),
                name=s.__name__,
                description=s.__doc__ or s.__name__,
                args_schema=s,
            )
            for s in decision_tools
        ]
        llm_with_choice = self._base_llm.bind_tools(lc_tools)

        case  = state["case"]
        ratio = case["monthly_cost"] / case["monthly_income"]

        # Extraer evaluacion de riesgo del paso anterior si existe
        risk_context = ""
        recent_assess = [
            a for a in state["agent_history"]
            if a["agent_name"] == "credit_officer" and a["tool_name"] == "AssessRisk"
        ]
        if recent_assess:
            last = recent_assess[-1]["tool_output"]
            risk_context = (
                f"Tu evaluacion previa: riesgo={last.get('risk_category','?')} | "
                f"factores: {last.get('risk_factors', [])}\n"
            )

        # Extraer recomendacion del Senior Clerk si existe
        senior_rec = ""
        escalations = [
            a for a in state["agent_history"]
            if a["agent_name"] == "senior_clerk" and a["tool_name"] == "EscalateCase"
        ]
        if escalations:
            rec = escalations[-1]["tool_output"].get("recommendation", "")
            senior_rec = f"Recomendacion del Senior Clerk: {rec}\n"

        messages = [
            SystemMessage(content=self.system_prompt),
            *state["messages"][-4:],
            {
                "role": "user",
                "content": (
                    f"CASO: {case['case_id']} | Monto solicitado: EUR {case['amount_requested']:,.0f}\n"
                    f"Score crediticio: {case['credit_score']} | "
                    f"Ingreso mensual: EUR {case['monthly_income']:,.0f}\n"
                    f"Ratio deuda/ingreso: {ratio:.2f} | "
                    f"Plazo: {case['number_of_terms']} meses\n"
                    f"{risk_context}"
                    f"{senior_rec}"
                    f"Revisiones del caso: {state['revision_count']}\n\n"
                    f"Toma tu decision final: ApproveLoan o RejectLoan. "
                    f"Justifica con los datos del caso y tu criterio profesional."
                ),
            },
        ]

        response: AIMessage = llm_with_choice.invoke(messages)

        if not response.tool_calls:
            return {"messages": [response], "current_agent": self.name, "next_agent": None}

        tool_call  = response.tool_calls[0]
        tool_name  = tool_call["name"]
        tool_args  = tool_call["args"]
        schema_cls = {s.__name__: s for s in decision_tools}.get(tool_name, RejectLoan)

        tool_output = {}
        try:
            tool_output = schema_cls(**tool_args).model_dump()
        except Exception as e:
            tool_output = {"error": str(e)}

        from ..state import AgentAction
        action: AgentAction = {
            "agent_name":    self.name,
            "tool_name":     tool_name,
            "tool_input":    tool_args,
            "tool_output":   tool_output,
            "sim_timestamp": state["sim_clock"],
            "real_duration": 0.0,
        }
        tool_message = ToolMessage(
            content=str(tool_output),
            tool_call_id=tool_call["id"],
        )

        status = "approved" if tool_name == "ApproveLoan" else "rejected"

        return {
            "messages":      [response, tool_message],
            "agent_history": [action],
            "current_agent": self.name,
            "next_agent":    None,
            "status":        status,
        }

    def _resolve_next_agent(self, tool_name: str, tool_args: dict, state: ProcessState) -> str | None:
        return "credit_officer" if tool_name == "AssessRisk" else None

    def _resolve_status(self, tool_name: str, tool_args: dict) -> str:
        if tool_name == "ApproveLoan":  return "approved"
        if tool_name == "RejectLoan":   return "rejected"
        return "in_review"
