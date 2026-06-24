"""
base_agent.py — v3
Cada agente determina deterministicamente cual tool viene a continuacion.
El LLM solo aporta el CONTENIDO (notas, decisiones, justificaciones).
La SECUENCIA la controla el codigo — igual que ChatDev / MetaGPT.
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
        self._base_llm = llm
        self._all_tools = {s.__name__: s for s in self.tool_schemas}

    def _bind_single_tool(self, schema: type[BaseModel]):
        """Vincula solo UNA tool al LLM — elimina ambiguedad de eleccion."""
        lc_tool = StructuredTool.from_function(
            func=_make_tool_func(schema),
            name=schema.__name__,
            description=schema.__doc__ or schema.__name__,
            args_schema=schema,
        )
        return self._base_llm.bind_tools([lc_tool])

    def _get_agent_step(self, state: ProcessState) -> int:
        """Cuantas veces ha actuado ESTE agente en el caso actual."""
        return sum(1 for a in state["agent_history"] if a["agent_name"] == self.name)

    def _get_next_tool_schema(self, state: ProcessState) -> type[BaseModel]:
        """
        Subclases DEBEN sobrescribir esto.
        Devuelve el schema Pydantic de la tool que corresponde en este paso.
        """
        raise NotImplementedError

    def _build_context_message(self, state: ProcessState, tool_schema: type[BaseModel]) -> str:
        case = state["case"]
        step = self._get_agent_step(state)
        history_summary = ""
        if state["agent_history"]:
            lines = [f"  - {a['agent_name']} ejecuto {a['tool_name']}"
                     for a in state["agent_history"][-6:]]
            history_summary = "\nAcciones previas:\n" + "\n".join(lines)

        return (
            f"CASO: {case['case_id']}\n"
            f"Monto: EUR {case['amount_requested']:,.0f} | Plazo: {case['number_of_terms']} meses\n"
            f"Cuota mensual: EUR {case['monthly_cost']:,.0f}\n"
            f"Score crediticio: {case['credit_score']} | Ingreso mensual: EUR {case['monthly_income']:,.0f}\n"
            f"Ratio deuda/ingreso: {case['monthly_cost']/case['monthly_income']:.2f}\n"
            f"Estado: {state['status']} | Revisiones: {state['revision_count']}"
            f"{history_summary}\n\n"
            f"INSTRUCCION (paso {step+1}): Debes llamar la tool '{tool_schema.__name__}'. "
            f"Completa todos sus campos con informacion relevante del caso."
        )

    def __call__(self, state: ProcessState) -> dict:
        tool_schema = self._get_next_tool_schema(state)
        llm_with_tool = self._bind_single_tool(tool_schema)

        messages = [
            SystemMessage(content=self.system_prompt),
            *state["messages"][-6:],  # ventana de contexto limitada
            {"role": "user", "content": self._build_context_message(state, tool_schema)},
        ]

        response: AIMessage = llm_with_tool.invoke(messages)

        if not response.tool_calls:
            # LLM no uso la tool — devolver sin cambios para reintento
            return {
                "messages":      [response],
                "current_agent": self.name,
                "next_agent":    None,
            }

        tool_call  = response.tool_calls[0]
        tool_name  = tool_call["name"]
        tool_args  = tool_call["args"]

        schema_cls  = self._all_tools.get(tool_name, tool_schema)
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

        tool_message = ToolMessage(
            content=str(tool_output),
            tool_call_id=tool_call["id"],
        )

        return {
            "messages":      [response, tool_message],
            "agent_history": [action],
            "current_agent": self.name,
            "next_agent":    self._resolve_next_agent(tool_name, tool_args, state),
            "status":        self._resolve_status(tool_name, tool_args),
        }

    def _resolve_next_agent(self, tool_name: str, tool_args: dict, state: ProcessState) -> str | None:
        return None

    def _resolve_status(self, tool_name: str, tool_args: dict) -> str:
        return "in_review"
