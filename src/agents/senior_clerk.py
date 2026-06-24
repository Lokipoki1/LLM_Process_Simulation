"""
senior_clerk.py — v3
Paso 1: ValidateApplication  (procedimental — siempre primero)
Paso 2: LLM elige entre EscalateCase o RequestAdditionalInfo (decision cognitiva)
"""

from __future__ import annotations
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from ..tools.schemas import ValidateApplication, RequestAdditionalInfo, EscalateCase
from ..state import ProcessState
from .base_agent import BaseAgent, _make_tool_func

SYSTEM_PROMPT = """
Eres Ana, Senior Clerk con 5 anos de experiencia en analisis crediticio.
Eres metodica, conservadora y orientada al detalle.

PASO 1 — ValidateApplication:
  Verifica ingresos, calcula el ratio deuda/ingreso y documenta hallazgos.

PASO 2 — Decision cognitiva (tu aportas el juicio profesional):
  Analiza el caso y decide:

  EscalateCase — cuando el caso esta listo para decision final:
    recommendation "approve"     si ratio < 0.35 y score >= 650
    recommendation "conditional" si ratio 0.35-0.45 o score 580-649
    recommendation "reject"      si ratio > 0.45 o score < 580
    Escribe un risk_summary detallado con los numeros del caso.

  RequestAdditionalInfo — SOLO si hay una inconsistencia real:
    - Ingresos declarados no coinciden con el monto solicitado
    - Falta documentacion especifica del proposito del prestamo
    - Hay senales de informacion contradictoria
    No pidas informacion adicional solo por precaucion — eso genera demoras.

Tu decision refleja tu experiencia y criterio profesional.
"""


class SeniorClerk(BaseAgent):
    name = "senior_clerk"
    system_prompt = SYSTEM_PROMPT
    tool_schemas = [ValidateApplication, RequestAdditionalInfo, EscalateCase]

    def _get_next_tool_schema(self, state: ProcessState) -> type:
        """Paso 1 siempre es ValidateApplication. Paso 2 lo decide el LLM."""
        step = self._get_agent_step(state)
        if step == 0:
            return ValidateApplication
        # En paso 2 devolvemos EscalateCase como "hint",
        # pero __call__ es sobreescrito para ofrecer ambas opciones
        return EscalateCase

    def __call__(self, state: ProcessState) -> dict:
        step = self._get_agent_step(state)

        if step == 0:
            # Paso 1: procedimental, solo ValidateApplication
            return super().__call__(state)

        # Paso 2: decision cognitiva — LLM elige entre dos tools
        return self._cognitive_decision(state)

    def _cognitive_decision(self, state: ProcessState) -> dict:
        """
        Ofrece al LLM exactamente DOS opciones y deja que decida.
        Esto es la simulacion cognitiva central del Senior Clerk.
        """
        decision_tools = [EscalateCase, RequestAdditionalInfo]
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
        messages = [
            SystemMessage(content=self.system_prompt),
            *state["messages"][-4:],
            {
                "role": "user",
                "content": (
                    f"CASO: {case['case_id']} | Monto: EUR {case['amount_requested']:,.0f}\n"
                    f"Score: {case['credit_score']} | Ingreso: EUR {case['monthly_income']:,.0f}\n"
                    f"Ratio deuda/ingreso calculado: {ratio:.2f}\n"
                    f"Revisiones previas: {state['revision_count']}\n\n"
                    f"Has completado la validacion. Ahora toma tu decision profesional: "
                    f"llama EscalateCase con tu recomendacion, o RequestAdditionalInfo "
                    f"si hay una inconsistencia especifica que requiere aclaracion."
                ),
            },
        ]

        response: AIMessage = llm_with_choice.invoke(messages)

        if not response.tool_calls:
            return {"messages": [response], "current_agent": self.name, "next_agent": None}

        tool_call  = response.tool_calls[0]
        tool_name  = tool_call["name"]
        tool_args  = tool_call["args"]
        schema_cls = {s.__name__: s for s in decision_tools}.get(tool_name, EscalateCase)

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

        next_agent = "credit_officer" if tool_name == "EscalateCase" else "junior_clerk"
        status     = "in_review" if tool_name == "EscalateCase" else "pending"

        return {
            "messages":      [response, tool_message],
            "agent_history": [action],
            "current_agent": self.name,
            "next_agent":    next_agent,
            "status":        status,
        }

    def _resolve_next_agent(self, tool_name: str, tool_args: dict, state: ProcessState) -> str | None:
        if tool_name == "EscalateCase":       return "credit_officer"
        if tool_name == "RequestAdditionalInfo": return "junior_clerk"
        return None

    def _resolve_status(self, tool_name: str, tool_args: dict) -> str:
        return "pending" if tool_name == "RequestAdditionalInfo" else "in_review"
