"""
base_agent.py — v2
Fixes: next_agent siempre se resetea explicitamente en cada llamada.
"""

from __future__ import annotations
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from ..state import ProcessState, AgentAction


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
        self._tools = {s.__name__: s for s in self.tool_schemas}
        lc_tools = [
            StructuredTool.from_function(
                func=_make_tool_func(schema),
                name=schema.__name__,
                description=schema.__doc__ or schema.__name__,
                args_schema=schema,
            )
            for schema in self.tool_schemas
        ]
        self._llm = llm.bind_tools(lc_tools)

    def _build_context_message(self, state: ProcessState) -> str:
        case = state["case"]
        history_summary = ""
        if state["agent_history"]:
            lines = [
                f"  - {a['agent_name']} ejecuto {a['tool_name']}"
                for a in state["agent_history"][-5:]
            ]
            history_summary = "\nAcciones previas:\n" + "\n".join(lines)

        tool_names = ", ".join(self._tools.keys())
        return (
            f"CASO: {case['case_id']}\n"
            f"Monto: EUR {case['amount_requested']:,.0f}\n"
            f"Proposito: {case['loan_goal']}\n"
            f"Plazo: {case['number_of_terms']} meses | Cuota: EUR {case['monthly_cost']:,.0f}\n"
            f"Score crediticio: {case['credit_score']} | Ingreso: EUR {case['monthly_income']:,.0f}\n"
            f"Estado: {state['status']} | Revisiones: {state['revision_count']}"
            f"{history_summary}\n\n"
            f"IMPORTANTE: Debes llamar exactamente una de estas tools: {tool_names}\n"
            f"No respondas con texto — usa siempre una tool call."
        )

    def __call__(self, state: ProcessState) -> dict:
        messages = [
            SystemMessage(content=self.system_prompt),
            *state["messages"],
            {"role": "user", "content": self._build_context_message(state)},
        ]

        response: AIMessage = self._llm.invoke(messages)

        # ── Sin tool call: resetear next_agent para evitar loops ──
        if not response.tool_calls:
            return {
                "messages":     [response],
                "current_agent": self.name,
                "next_agent":   None,   # reset explícito — clave para evitar loops
            }

        tool_call  = response.tool_calls[0]
        tool_name  = tool_call["name"]
        tool_args  = tool_call["args"]

        schema_cls  = self._tools.get(tool_name)
        tool_output = {}
        if schema_cls:
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

        tool_message = ToolMessage(
            content=str(tool_output),
            tool_call_id=tool_call["id"],
        )

        return {
            "messages":      [response, tool_message],
            "agent_history": [action],
            "current_agent": self.name,
            "next_agent":    self._resolve_next_agent(tool_name, tool_args, state),
            "status":        self._resolve_status(tool_name),
        }

    def _resolve_next_agent(self, tool_name: str, tool_args: dict, state: ProcessState) -> str | None:
        return None

    def _resolve_status(self, tool_name: str) -> str:
        return "in_review"
