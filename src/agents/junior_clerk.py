"""
junior_clerk.py — v3
Pasos 1-2 procedurales: IntakeApplication -> CheckDocuments.
Paso 3 cognitivo: ForwardCase (al Senior Clerk) o ReturnApplicationEarly.
"""

from __future__ import annotations
import logging
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from ..tools.schemas import IntakeApplication, CheckDocuments, ForwardCase, ReturnApplicationEarly
from ..state import ProcessState, AgentAction
from .base_agent import BaseAgent, _make_tool_func

logger = logging.getLogger("bps.junior_clerk")

SYSTEM_PROMPT = """
Eres Carlos, Junior Clerk con 2 anos de experiencia en prestamos bancarios.
Eres eficiente y directo.

PASO 1 — IntakeApplication: registra la recepcion del caso.
PASO 2 — CheckDocuments: verifica que la documentacion este completa.
PASO 3 — Decision profesional (tu juicio):

  ForwardCase — cuando el caso cumple requisitos basicos:
    - Documentacion aceptable (completa o con deficiencias subsanables)
    - Score crediticio >= 480 y ratio deuda/ingreso <= 0.70
    - Usa priority="high" si monto > EUR 30.000, si no "normal"

  ReturnApplicationEarly — cuando el caso claramente no procede:
    - Documentacion fraudulenta
    - Score crediticio < 480 (perfil manifiestamente inviable)
    - Ratio deuda/ingreso > 0.70 (cuota economicamente insostenible)

Nunca apruebas ni rechazas creditos — esa es responsabilidad del Credit Officer.
Completa cada campo con informacion concisa y relevante del caso.
"""

_PROCEDURAL = [IntakeApplication, CheckDocuments]
_DECISION_TOOLS = [ForwardCase, ReturnApplicationEarly]


class JuniorClerk(BaseAgent):
    name = "junior_clerk"
    system_prompt = SYSTEM_PROMPT
    tool_schemas = [IntakeApplication, CheckDocuments, ForwardCase, ReturnApplicationEarly]

    def _get_next_tool_schema(self, state: ProcessState) -> type:
        step = self._get_agent_step(state)
        visit_step = step % 3
        return _PROCEDURAL[visit_step] if visit_step < 2 else ForwardCase

    def __call__(self, state: ProcessState) -> dict:
        step = self._get_agent_step(state)
        if step % 3 == 2:
            return self._cognitive_decision(state)
        return super().__call__(state)

    def _cognitive_decision(self, state: ProcessState) -> dict:
        case  = state["case"]
        ratio = case["monthly_cost"] / case["monthly_income"]

        lc_tools = [
            StructuredTool.from_function(
                func=_make_tool_func(s),
                name=s.__name__,
                description=s.__doc__ or s.__name__,
                args_schema=s,
            )
            for s in _DECISION_TOOLS
        ]

        messages = [
            SystemMessage(content=self.system_prompt),
            *state["messages"][-4:],
            {
                "role": "user",
                "content": (
                    f"CASO: {case['case_id']} | Monto: EUR {case['amount_requested']:,.0f}\n"
                    f"Score: {case['credit_score']} | Ingreso: EUR {case['monthly_income']:,.0f}/mes\n"
                    f"Ratio deuda/ingreso: {ratio:.2f} | Objetivo: {case['loan_goal']}\n"
                    f"Revisiones previas: {state['revision_count']}\n\n"
                    f"Has completado intake y verificacion de documentos. "
                    f"Decide: ForwardCase al Senior Clerk si el caso es viable "
                    f"(score >= 480 y ratio <= 0.70), o ReturnApplicationEarly si "
                    f"claramente no procede (docs fraudulentos, score < 480, ratio > 0.70)."
                ),
            },
        ]

        response: AIMessage = self._cognitive_llm.bind_tools(lc_tools).invoke(messages)

        # Fallback: forward por defecto si el LLM no hace tool call
        if not response.tool_calls:
            tool_name = "ForwardCase"
            tool_args  = {
                "case_id":      case["case_id"],
                "forwarded_to": "senior_clerk",
                "priority":     "high" if case["amount_requested"] > 30000 else "normal",
            }
            tc_id = "fallback"
        else:
            tc        = response.tool_calls[0]
            tc_id     = tc["id"]
            tool_name = tc["name"]
            tool_args  = tc["args"]
            if tool_name not in ("ForwardCase", "ReturnApplicationEarly"):
                tool_name = "ForwardCase"
                tool_args  = {
                    "case_id":      case["case_id"],
                    "forwarded_to": "senior_clerk",
                    "priority":     "high" if case["amount_requested"] > 30000 else "normal",
                }
            # Sanity guard: ReturnApplicationEarly is only valid for genuinely disqualified cases.
            # If the LLM hallucinates this for a viable case, override to ForwardCase.
            elif tool_name == "ReturnApplicationEarly":
                actually_disqualified = case["credit_score"] < 480 or ratio > 0.70
                if not actually_disqualified:
                    logger.warning(
                        "%s | [JC GUARD] LLM called ReturnApplicationEarly for viable case "
                        "(score=%d, ratio=%.2f) — overriding to ForwardCase",
                        case["case_id"], case["credit_score"], ratio,
                    )
                    tool_name = "ForwardCase"
                    tool_args  = {
                        "case_id":      case["case_id"],
                        "forwarded_to": "senior_clerk",
                        "priority":     "high" if case["amount_requested"] > 30000 else "normal",
                    }
                    tc_id = "guard"

        schema_cls  = {"ForwardCase": ForwardCase, "ReturnApplicationEarly": ReturnApplicationEarly}[tool_name]
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

        is_return  = tool_name == "ReturnApplicationEarly"
        next_agent = None if is_return else "senior_clerk"
        status     = "rejected" if is_return else "in_review"
        rejection  = tool_args.get("details") if is_return else None

        if is_return:
            logger.info(
                "%s | [COGNITIVE] ReturnApplicationEarly | reason=%s | ratio=%.2f | score=%d",
                case["case_id"], tool_args.get("return_reason", "?"), ratio, case["credit_score"],
            )
        else:
            logger.info(
                "%s | [COGNITIVE] ForwardCase | priority=%s",
                case["case_id"], tool_args.get("priority", "?"),
            )

        result = {
            "messages":      [response, tool_message],
            "agent_history": [action],
            "current_agent": self.name,
            "next_agent":    next_agent,
            "status":        status,
        }
        if rejection:
            result["rejection_reason"] = rejection
        return result

    def _resolve_next_agent(self, tool_name: str, _tool_args: dict, _state: ProcessState) -> str | None:
        return "senior_clerk" if tool_name == "ForwardCase" else None

    def _resolve_status(self, tool_name: str, _tool_args: dict) -> str:
        if tool_name == "ReturnApplicationEarly": return "rejected"
        return "in_review" if tool_name == "ForwardCase" else "pending"
