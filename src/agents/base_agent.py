"""
base_agent.py — v5 (information asymmetry)
──────────────────────────────────────────
Context builder enforces information asymmetry:
  - JC: sees application data only
  - SC: sees application + credit bureau data (after CheckCreditScore)
  - CO: sees everything + all prior reasoning
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
    system_prompt: str = "You are a processing agent."
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
        """Bind tools excluding the one just called (prevents repetition loops)."""
        my_actions = [a for a in state["agent_history"] if a["agent_name"] == self.name]
        last_tool = my_actions[-1]["tool_name"] if my_actions else None
        available = [t for t in self._all_lc_tools if t.name != last_tool]
        if not available:
            available = self._all_lc_tools
        return self._base_llm.bind_tools(available)

    # ── Handoff narrative ─────────────────────

    def _build_handoff_context(self, state: ProcessState) -> str:
        """Build narrative summary of all prior agent actions."""
        lines = []
        for action in state["agent_history"]:
            agent  = action["agent_name"].replace("_", " ").title()
            tool   = action["tool_name"]
            output = action["tool_output"]

            notes = (
                output.get("initial_notes")
                or output.get("notes")
                or output.get("check_notes")
                or output.get("risk_assessment")
                or output.get("risk_summary")
                or output.get("assessment_notes")
                or output.get("approval_notes")
                or output.get("rejection_notes")
                or output.get("reason")
                or output.get("details")
                or ""
            )
            summary = f"{agent} executed {tool}"
            if notes:
                summary += f': "{notes[:200]}"'
            if "recommendation" in output:
                summary += f" [recommendation: {output['recommendation']}]"
            if "document_status" in output:
                summary += f" [docs: {output['document_status']}]"
            if "risk_category" in output:
                summary += f" [risk: {output['risk_category']}]"
            if "credit_score_acceptable" in output:
                summary += f" [score acceptable: {output['credit_score_acceptable']}]"

            lines.append(summary)

        return "\n".join(lines) if lines else "No prior actions."

    # ── Information-asymmetric context ────────

    def _build_case_context(self, state: ProcessState) -> str:
        """
        Build context with INFORMATION ASYMMETRY.
        Each agent sees only what they would see in real life.
        """
        app = state["application"]

        # Application data — everyone sees this
        context = (
            f"== ACTIVE CASE ==\n"
            f"ID: {app['case_id']}\n"
            f"Requested amount: EUR {app['amount_requested']:,.0f}\n"
            f"Loan goal: {app['loan_goal']}\n"
            f"Application type: {app['application_type']}\n"
        )

        # Credit bureau data — only visible after CheckCreditScore
        if state.get("credit_checked") and state.get("credit_bureau_data"):
            cbd = state["credit_bureau_data"]
            context += (
                f"\n== CREDIT BUREAU DATA (checked) ==\n"
                f"Credit score: {cbd.get('credit_score', 'N/A')}\n"
                f"Monthly cost: EUR {cbd.get('monthly_cost', 'N/A')}\n"
                f"Number of terms: {cbd.get('number_of_terms', 'N/A')}\n"
                f"Offered amount: EUR {cbd.get('offered_amount', 'N/A')}\n"
            )
        elif self.name != "junior_clerk" and not state.get("credit_checked"):
            context += (
                f"\n== CREDIT BUREAU DATA ==\n"
                f"NOT YET CHECKED — call CheckCreditScore first.\n"
            )

        # Process state
        context += (
            f"\nStatus: {state['status']}\n"
            f"Rework count: {state['rework_count']}\n"
        )

        # Handoff narrative
        handoff = self._build_handoff_context(state)
        context += (
            f"\n== ACTION HISTORY ==\n"
            f"{handoff}\n\n"
            f"== INSTRUCTION ==\n"
            f"Choose and execute the most appropriate tool for your next step."
        )

        return context

    # ── Main execution ────────────────────────

    def __call__(self, state: ProcessState) -> dict:
        _log = logging.getLogger(f"bps.{self.name}")
        case_id = state["application"]["case_id"]

        messages = [
            SystemMessage(content=self.system_prompt),
            {"role": "user", "content": self._build_case_context(state)},
        ]

        response: AIMessage = self._bind_available_tools(state).invoke(messages)

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

        schema_cls = self._all_tools.get(tool_name)
        tool_output = {}
        if schema_cls:
            try:
                tool_output = schema_cls(**tool_args).model_dump()
            except Exception as e:
                tool_output = {"error": str(e)}
                _log.error("%s | validation error in %s: %s", case_id, tool_name, e)
        else:
            tool_output = tool_args

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

        _log.info("%s | %s | next=%s | status=%s", case_id, tool_name, next_agent or "self", status)

        result = {
            "messages":      [response, tool_message],
            "agent_history": [action],
            "current_agent": self.name,
            "next_agent":    next_agent,
            "status":        status,
        }

        # CheckCreditScore reveals the credit bureau data
        if tool_name == "CheckCreditScore":
            result["credit_checked"] = True

        return result

    def _resolve_next_agent(self, tool_name, tool_args, state) -> str | None:
        return None

    def _resolve_status(self, tool_name, tool_args) -> str:
        return "in_review"
