"""
base_agent.py — v4 (autonomous)
───────────────────────────────
Cada agente recibe TODAS sus tools y decide la secuencia.
El SOP vive en el system prompt, no en código.
El código solo garantiza: Pydantic validation, tool_choice="any", handoff narrativo.
"""

from __future__ import annotations
import logging
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from ..state import ProcessState, AgentAction

logger = logging.getLogger("bps.base_agent")


def _make_tool_func(schema_cls: type[BaseModel]):
    def tool_func(**kwargs):
        return schema_cls(**kwargs).model_dump()
    tool_func.__name__ = schema_cls.__name__
    return tool_func


class BaseAgent:
    name: str = "base_agent"
    system_prompt: str = "Eres un agente de procesamiento."
    tool_schemas: list[type[BaseModel]] = []

    def __init__(self, llm):
        self._all_tools = {s.__name__: s for s in self.tool_schemas}
        self._base_llm = llm
        self._all_lc_tools = [
            StructuredTool.from_function(
                func=_make_tool_func(s),
                name=s.__name__,
                description=s.__doc__ or s.__name__,
                args_schema=s,
            )
            for s in self.tool_schemas
        ]

    def _bind_available_tools(self, state: ProcessState):
        """
        Bind tools excluyendo la que se acaba de ejecutar.
        Esto NO es un guard — no decide QUÉ tool usar, solo impide
        que el agente repita la misma tool dos veces consecutivas.
        Equivale al SOP real: un clerk no re-verifica el mismo
        documento inmediatamente después de verificarlo.
        """
        my_actions = [a for a in state["agent_history"] if a["agent_name"] == self.name]
        last_tool = my_actions[-1]["tool_name"] if my_actions else None

        available = [t for t in self._all_lc_tools if t.name != last_tool]
        if not available:
            available = self._all_lc_tools  # fallback: all tools if only one exists

        return self._base_llm.bind_tools(available, tool_choice="any")

    # ── Handoff narrativo ─────────────────────────────────────

    def _build_handoff_context(self, state: ProcessState) -> str:
        """
        Construye el resumen narrativo de lo que hicieron los agentes anteriores.
        Esto es la "interacción entre agentes" de la tesis:
        cada agente lee el RAZONAMIENTO del anterior, no solo un dict.
        """
        lines = []
        for action in state["agent_history"]:
            agent   = action["agent_name"].replace("_", " ").title()
            tool    = action["tool_name"]
            output  = action["tool_output"]

            # Extraer el razonamiento narrativo de cada acción
            notes = (
                output.get("initial_notes")
                or output.get("notes")
                or output.get("validation_notes")
                or output.get("risk_summary")
                or output.get("assessment_notes")
                or output.get("approval_notes")
                or output.get("rejection_notes")
                or output.get("reason")
                or output.get("details")
                or ""
            )
            summary = f"{agent} ejecutó {tool}"
            if notes:
                summary += f": \"{notes[:200]}\""

            # Añadir datos clave del output
            if "recommendation" in output:
                summary += f" [recomendación: {output['recommendation']}]"
            if "document_status" in output:
                summary += f" [docs: {output['document_status']}]"
            if "risk_category" in output:
                summary += f" [riesgo: {output['risk_category']}]"
            if "debt_to_income_ratio" in output:
                summary += f" [ratio verificado: {output['debt_to_income_ratio']:.2f}]"

            lines.append(summary)

        return "\n".join(lines) if lines else "Ninguna acción previa."

    def _build_case_context(self, state: ProcessState) -> str:
        """Construye el contexto completo del caso para el LLM."""
        case = state["case"]
        ratio = case["monthly_cost"] / case["monthly_income"]
        handoff = self._build_handoff_context(state)

        return (
            f"══ CASO ACTIVO ══\n"
            f"ID: {case['case_id']}\n"
            f"Monto solicitado: EUR {case['amount_requested']:,.0f}\n"
            f"Propósito: {case['loan_goal']}\n"
            f"Plazo: {case['number_of_terms']} meses | Cuota mensual: EUR {case['monthly_cost']:,.0f}\n"
            f"Score crediticio: {case['credit_score']}\n"
            f"Ingreso mensual declarado: EUR {case['monthly_income']:,.0f}\n"
            f"Ratio deuda/ingreso estimado: {ratio:.2f}\n"
            f"Estado actual: {state['status']}\n"
            f"Revisiones previas del caso: {state['revision_count']}\n\n"
            f"══ HISTORIAL DE ACCIONES ══\n"
            f"{handoff}\n\n"
            f"══ INSTRUCCIÓN ══\n"
            f"Elige y ejecuta la tool más apropiada para tu siguiente paso."
        )

    def __call__(self, state: ProcessState) -> dict:
        _log = logging.getLogger(f"bps.{self.name}")
        case_id = state["case"]["case_id"]

        user_content = self._build_case_context(state)
        messages = [
            SystemMessage(content=self.system_prompt),
            {"role": "user", "content": user_content},
        ]

        _log.debug(
            "%s | ── PROMPT ──────────────────────────────\n"
            "SYSTEM:\n%s\n\nUSER:\n%s",
            case_id, self.system_prompt, user_content,
        )

        response: AIMessage = self._bind_available_tools(state).invoke(messages)

        usage = getattr(response, "usage_metadata", None) or (
            response.response_metadata.get("token_usage") if response.response_metadata else None
        )
        _log.debug(
            "%s | ── LLM RESPONSE ────────────────────────\n"
            "content: %s\ntool_calls: %s\nusage: %s",
            case_id,
            response.content or "(empty)",
            response.tool_calls,
            usage,
        )

        if not response.tool_calls:
            _log.warning("%s | no tool call — retrying", case_id)
            return {
                "messages": [response],
                "current_agent": self.name,
                "next_agent": None,
            }

        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        _log.debug("%s | ── TOOL INPUT ──────────────────────────\n%s(%s)", case_id, tool_name, tool_args)

        # Validar con Pydantic — el único control hard que mantenemos
        schema_cls = self._all_tools.get(tool_name)
        tool_output = {}
        if schema_cls:
            try:
                tool_output = schema_cls(**tool_args).model_dump()
            except Exception as e:
                tool_output = {"error": str(e)}
                _log.error("%s | validation error in %s: %s", case_id, tool_name, e)
        else:
            _log.warning("%s | unknown tool %s", case_id, tool_name)
            tool_output = tool_args

        _log.debug("%s | ── TOOL OUTPUT ─────────────────────────\n%s", case_id, tool_output)

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

        next_agent = self._resolve_next_agent(tool_name, tool_args, state)
        status = self._resolve_status(tool_name, tool_args)

        _log.info(
            "%s | %s | next=%s | status=%s",
            case_id, tool_name, next_agent or "self", status,
        )

        return {
            "messages":      [response, tool_message],
            "agent_history": [action],
            "current_agent": self.name,
            "next_agent":    next_agent,
            "status":        status,
        }

    def _resolve_next_agent(self, tool_name: str, tool_args: dict, state: ProcessState) -> str | None:
        return None

    def _resolve_status(self, tool_name: str, tool_args: dict) -> str:
        return "in_review"
