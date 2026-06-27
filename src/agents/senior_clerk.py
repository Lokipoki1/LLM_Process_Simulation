"""
senior_clerk.py — v5
Paso 1: ValidateApplication  (procedimental)
Paso 2: EscalateCase o RequestAdditionalInfo  (decision cognitiva)

Paths:
  A: VALIDATE -> ESCALATE(approve)      caso limpio
  B: VALIDATE -> ESCALATE(reject)       caso claramente malo
  C: VALIDATE -> REQUEST_INFO           info faltante -> vuelve al JC
  D: VALIDATE -> ESCALATE(conditional)  borderline -> Credit Officer decide
"""

from __future__ import annotations
import logging
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from ..tools.schemas import ValidateApplication, RequestAdditionalInfo, EscalateCase
from ..state import ProcessState, AgentAction
from .base_agent import BaseAgent, _make_tool_func

logger = logging.getLogger("bps.senior_clerk")

SYSTEM_PROMPT = """
Eres Ana, Senior Clerk con 5 anos de experiencia en analisis crediticio.
Eres metodica y conservadora.

PASO 1 — ValidateApplication: verifica ingresos y calcula ratio deuda/ingreso.

PASO 2 — TU DECISION PROFESIONAL (elige UNA de las dos opciones):

  EscalateCase — en la mayoria de los casos:
    recommendation "approve"     si score >= 680 Y ratio < 0.30
    recommendation "conditional" si score 600-679 O ratio 0.30-0.45
    recommendation "reject"      si score < 580 O ratio > 0.45

  RequestAdditionalInfo — SOLO en casos borderline especificos (max 1 vez por caso):
    Usa esta opcion UNICAMENTE si se cumplen AMBAS condiciones:
      1. Es la primera revision (revision_count == 0)
      2. El score esta entre 580-649 O el ratio entre 0.30-0.45
    Pide documentacion especifica: justificante de ingresos, contrato laboral.
    NO lo uses para casos claramente aprobables (score > 680, ratio < 0.25).
    NO lo uses para casos claramente rechazables (score < 580, ratio > 0.45).

La mayoria de los casos tienen perfil claro — escalalos directamente.
"""


class SeniorClerk(BaseAgent):
    name = "senior_clerk"
    system_prompt = SYSTEM_PROMPT
    tool_schemas = [ValidateApplication, RequestAdditionalInfo, EscalateCase]

    def _get_next_tool_schema(self, state: ProcessState) -> type:
        # Even steps (0, 2, 4…) = ValidateApplication; odd = cognitive hint
        return ValidateApplication if self._get_agent_step(state) % 2 == 0 else EscalateCase

    def __call__(self, state: ProcessState) -> dict:
        # Each SC visit: step%2==0 → ValidateApplication, step%2==1 → cognitive
        if self._get_agent_step(state) % 2 == 0:
            return super().__call__(state)
        return self._cognitive_decision(state)

    def _cognitive_decision(self, state: ProcessState) -> dict:
        case  = state["case"]
        ratio = case["monthly_cost"] / case["monthly_income"]

        decision_tools = [EscalateCase, RequestAdditionalInfo]
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
                    f"Ratio deuda/ingreso: {ratio:.2f} | Proposito: {case['loan_goal']}\n"
                    f"Plazo: {case['number_of_terms']} meses | Cuota: EUR {case['monthly_cost']:,.0f}/mes\n"
                    f"Revisiones previas: {state['revision_count']}\n\n"
                    f"Has validado la solicitud. Ahora decide:\n"
                    f"- EscalateCase con tu recomendacion si el caso esta claro\n"
                    f"- RequestAdditionalInfo si necesitas mas datos antes de escalar\n"
                ),
            },
        ]

        response: AIMessage = self._cognitive_llm.bind_tools(lc_tools).invoke(messages)

        # Extract valid tool call
        tool_call = None
        if response.tool_calls:
            tc = response.tool_calls[0]
            if tc["name"] in {"EscalateCase", "RequestAdditionalInfo"}:
                tool_call = tc

        if tool_call is None:
            tool_name, tool_args = self._fallback_decision(case, ratio, state)
            tc_id = "fallback"
            logger.warning(
                "%s | [FALLBACK] no tool call → %s | ratio=%.2f | score=%d | revision=%d",
                case["case_id"], tool_name, ratio, case["credit_score"], state["revision_count"],
            )
        elif tool_call["name"] == "RequestAdditionalInfo" and state["revision_count"] >= 1:
            # Hard guard: allow only 1 revision cycle regardless of LLM choice
            logger.warning(
                "%s | [GUARD] RequestAdditionalInfo rejected (revision=%d >= 1) → forcing EscalateCase",
                case["case_id"], state["revision_count"],
            )
            tool_name, tool_args = self._fallback_decision(case, ratio, state)
            tc_id = "guard"
        else:
            tool_name = tool_call["name"]
            tool_args  = tool_call["args"]
            tc_id      = tool_call["id"]

        schema_cls  = {"EscalateCase": EscalateCase, "RequestAdditionalInfo": RequestAdditionalInfo}[tool_name]
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
        tool_message = ToolMessage(content=str(tool_output), tool_call_id=tc_id)

        if tool_name == "EscalateCase":
            logger.info(
                "%s | [COGNITIVE] EscalateCase | rec=%s | ratio=%.2f | score=%d",
                case["case_id"], tool_args.get("recommendation", "?"), ratio, case["credit_score"],
            )
            next_agent, status = "credit_officer", "in_review"
        else:
            logger.info(
                "%s | [COGNITIVE] RequestAdditionalInfo | from=%s | needed=%s",
                case["case_id"],
                tool_args.get("requested_from", "?"),
                tool_args.get("information_needed", []),
            )
            next_agent, status = "junior_clerk", "pending"

        return {
            "messages":      [response, tool_message],
            "agent_history": [action],
            "current_agent": self.name,
            "next_agent":    next_agent,
            "status":        status,
        }

    def _fallback_decision(self, case: dict, ratio: float, state: ProcessState) -> tuple[str, dict]:
        """Deterministic safety net when the LLM produces no valid tool call."""
        if state["revision_count"] >= 2:
            # Too many revision cycles → escalate regardless
            rec = "reject" if ratio > 0.45 or case["credit_score"] < 580 else "conditional"
            return "EscalateCase", {
                "case_id":        case["case_id"],
                "risk_summary":   f"Ratio: {ratio:.2f} | Score: {case['credit_score']} | Revisiones: {state['revision_count']}",
                "recommendation": rec,
            }

        if 0.35 <= ratio <= 0.50 and state["revision_count"] == 0:
            # Borderline ratio, first visit → request info
            return "RequestAdditionalInfo", {
                "case_id":          case["case_id"],
                "requested_from":   "junior_clerk",
                "information_needed": ["Verificacion de ingresos actualizada", "Justificacion del proposito del prestamo"],
                "reason":           f"Ratio {ratio:.2f} en zona borderline (0.35-0.50). Se requiere informacion adicional.",
            }

        # Default: escalate with standard recommendation
        if ratio < 0.30 and case["credit_score"] >= 680:
            rec = "approve"
        elif ratio > 0.45 or case["credit_score"] < 580:
            rec = "reject"
        else:
            rec = "conditional"
        return "EscalateCase", {
            "case_id":        case["case_id"],
            "risk_summary":   f"Ratio: {ratio:.2f} | Score: {case['credit_score']}",
            "recommendation": rec,
        }

    def _resolve_next_agent(self, tool_name: str, _tool_args: dict, _state: ProcessState) -> str | None:
        if tool_name == "EscalateCase":          return "credit_officer"
        if tool_name == "RequestAdditionalInfo": return "junior_clerk"
        return None  # ValidateApplication: stay in SC for cognitive step

    def _resolve_status(self, tool_name: str, _tool_args: dict) -> str:
        return "pending" if tool_name == "RequestAdditionalInfo" else "in_review"
