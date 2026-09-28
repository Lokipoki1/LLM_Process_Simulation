"""
rule_based.py
-------------
Deterministic baseline agents: same tools, same routing, no LLM.

Drop-in replacements for the LLM agents, used to test what the LLM
contributes over a decision table. They cover every criterion the
prompts describe, and replace the prompts' judgement calls with the
explicit thresholds below.
"""

from __future__ import annotations
import logging

from ..state import ProcessState, AgentAction
from ..tools.schemas import (
    IntakeApplication, CheckDocuments, ForwardCase, ReturnApplicationEarly,
    CheckCreditScore, ValidateApplication, RequestAdditionalInfo, EscalateCase,
    AssessRisk, ApproveLoan, RejectLoan,
)

logger = logging.getLogger("bps.rule_agent")


# -------------------------------------------------
# Thresholds - explicit stand-ins for the prompts' judgement calls
# -------------------------------------------------

HIGH_AMOUNT = 50_000        # priority="high" above this
IMPLAUSIBLE_CAR = 100_000   # a car loan this large is not credible
GOOD_SCORE = 700
POOR_SCORE = 550

# LoanGoal values that carry no actual purpose. In BPIC 2017 these are
# literal labels meaning "the reason is not recorded here".
VAGUE_GOALS = {"other, see explanation", "unknown", "other", ""}


def is_vague_goal(goal: str) -> bool:
    return (goal or "").strip().lower() in VAGUE_GOALS


class RuleBasedAgent:
    """Base for deterministic agents, with the same interface as BaseAgent."""

    name: str = "rule_agent"
    tool_schemas: list = []

    def __init__(self, llm=None):
        # Signature matches BaseAgent so the executor can swap them.
        self._tools = {s.__name__: s for s in self.tool_schemas}

    # -- helpers -------------------------------

    def _my_tools_used(self, state: ProcessState) -> list[str]:
        return [
            a["tool_name"] for a in state["agent_history"]
            if a["agent_name"] == self.name
        ]

    def _tools_since_rework(self, state: ProcessState) -> list[str]:
        """Tools this agent used since the most recent RequestAdditionalInfo."""
        history = state["agent_history"]
        start = 0
        for i, a in enumerate(history):
            if a["tool_name"] == "RequestAdditionalInfo":
                start = i + 1
        return [
            a["tool_name"] for a in history[start:]
            if a["agent_name"] == self.name
        ]

    def _emit(
        self,
        state: ProcessState,
        tool_name: str,
        args: dict,
        next_agent: str | None,
        status: str,
    ) -> dict:
        """Build the same partial update an LLM agent would return."""
        schema = self._tools[tool_name]
        try:
            output = schema(**args).model_dump()
        except Exception as e:
            logger.error("%s | rule agent produced invalid %s: %s",
                         state["application"]["case_id"], tool_name, e)
            output = dict(args)

        action: AgentAction = {
            "agent_name":    self.name,
            "tool_name":     tool_name,
            "tool_input":    args,
            "tool_output":   output,
            "sim_timestamp": state["sim_clock"],
            "real_duration": 0.0,
            "context_sent":  "",   # rule agents receive no prompt
            "raw_content":   "",
        }

        result = {
            "messages":      [],
            "agent_history": [action],
            "current_agent": self.name,
            "next_agent":    next_agent,
            "status":        status,
        }
        if tool_name == "CheckCreditScore":
            result["credit_checked"] = True
        return result

    def __call__(self, state: ProcessState) -> dict:
        raise NotImplementedError


# -------------------------------------------------
# Junior Clerk
# -------------------------------------------------

class RuleJuniorClerk(RuleBasedAgent):
    """
    Sequence: IntakeApplication -> CheckDocuments -> forward or return.
    Returns cases with a vague goal or an implausible amount.
    """

    name = "junior_clerk"
    tool_schemas = [
        IntakeApplication, CheckDocuments, ForwardCase, ReturnApplicationEarly,
    ]

    def __call__(self, state: ProcessState) -> dict:
        app = state["application"]
        case_id = app["case_id"]
        amount = app["amount_requested"]
        goal = app["loan_goal"]
        used = self._tools_since_rework(state)
        is_rework = state["rework_count"] > 0

        # -- rework visit: re-check, then forward --
        if is_rework:
            if "CheckDocuments" not in used:
                return self._emit(state, "CheckDocuments", {
                    "case_id": case_id,
                    "document_status": "complete",
                    "missing_documents": [],
                    "notes": "Re-checked following the senior clerk's request.",
                    "typical_duration_minutes": 40,
                    "complexity_rationale": "Rework visit: the file already raised a question.",
                    "case_complexity": 4,
                }, None, "pending")
            return self._emit(state, "ForwardCase", {
                "case_id": case_id,
                "forwarded_to": "senior_clerk",
                "priority": "high" if amount > HIGH_AMOUNT else "normal",
                "typical_duration_minutes": 8,
                "complexity_rationale": "Returning a reworked file to the senior clerk.",
                "case_complexity": 3,
            }, "senior_clerk", "in_review")

        # -- first visit --
        if "IntakeApplication" not in used:
            return self._emit(state, "IntakeApplication", {
                "case_id": case_id,
                "applicant_acknowledged": True,
                "initial_notes": f"Received: EUR {amount:,.0f} for {goal}.",
                "typical_duration_minutes": 30,
                "complexity_rationale": (
                    "Stated purpose is not specific." if is_vague_goal(goal)
                    else "Standard intake of a new application."
                ),
                "case_complexity": 3 if is_vague_goal(goal) else 2,
            }, None, "pending")

        vague = is_vague_goal(goal)
        implausible = goal.lower().startswith("car") and amount > IMPLAUSIBLE_CAR
        blocked = vague or implausible

        if "CheckDocuments" not in used:
            if vague:
                notes = (
                    f"Loan goal recorded as '{goal}' with no explanation attached; "
                    f"the purpose of EUR {amount:,.0f} cannot be established."
                )
                missing = ["Written explanation of the loan purpose"]
                rationale = "No usable statement of purpose in the file."
            elif implausible:
                notes = f"Amount of EUR {amount:,.0f} is not credible for goal '{goal}'."
                missing = ["Justification for requested amount"]
                rationale = "Amount does not fit the stated purpose."
            else:
                notes = f"Documentation consistent with a {goal} request."
                missing = []
                rationale = "Application is internally consistent."

            return self._emit(state, "CheckDocuments", {
                "case_id": case_id,
                "document_status": "incomplete" if blocked else "complete",
                "missing_documents": missing,
                "notes": notes,
                "typical_duration_minutes": 40,
                "complexity_rationale": rationale,
                "case_complexity": 4 if blocked else 2,
            }, None, "pending")

        # -- decision --
        if blocked:
            if vague:
                reason = "incomplete_docs"
                details = (
                    f"Loan goal '{goal}' carries no explanation; the file cannot "
                    f"be assessed until the purpose is stated."
                )
            else:
                reason = "unreasonable_request"
                details = f"EUR {amount:,.0f} for '{goal}' is outside any plausible range."

            return self._emit(state, "ReturnApplicationEarly", {
                "case_id": case_id,
                "return_reason": reason,
                "details": details,
                "typical_duration_minutes": 7,
                "complexity_rationale": "Clear-cut return, no judgement needed.",
                "case_complexity": 3,
            }, None, "rejected")

        return self._emit(state, "ForwardCase", {
            "case_id": case_id,
            "forwarded_to": "senior_clerk",
            "priority": "high" if amount > HIGH_AMOUNT else "normal",
            "typical_duration_minutes": 8,
            "complexity_rationale": "Routine handover to the senior clerk.",
            "case_complexity": 2,
        }, "senior_clerk", "in_review")


# -------------------------------------------------
# Senior Clerk
# -------------------------------------------------

class RuleSeniorClerk(RuleBasedAgent):
    """
    Sequence: CheckCreditScore -> ValidateApplication -> escalate or rework.

    Requests rework once when the bureau returned no score, then escalates.
    """

    name = "senior_clerk"
    tool_schemas = [
        CheckCreditScore, ValidateApplication, RequestAdditionalInfo, EscalateCase,
    ]

    def __call__(self, state: ProcessState) -> dict:
        app = state["application"]
        case_id = app["case_id"]
        cbd = state.get("credit_bureau_data") or {}
        score = cbd.get("credit_score")
        used = self._tools_since_rework(state)
        first_pass = state["rework_count"] == 0

        # Bureau query happens on every visit, including after rework:
        # the clerk re-runs it to see whether anything came back.
        if "CheckCreditScore" not in used:
            return self._emit(state, "CheckCreditScore", {
                "case_id": case_id,
                "bureau_consulted": "national_credit_bureau",
                "check_notes": (
                    "Re-running the bureau query after rework."
                    if not first_pass else
                    "Standard bureau query before validation."
                ),
                "typical_duration_minutes": 50,
                "complexity_rationale": "Routine lookup.",
                "case_complexity": 2,
            }, None, "in_review")

        if "ValidateApplication" not in used:
            acceptable = score is not None and score >= POOR_SCORE
            return self._emit(state, "ValidateApplication", {
                "case_id": case_id,
                "credit_score_acceptable": bool(acceptable),
                "risk_assessment": (
                    f"Credit score {score}; amount EUR {app['amount_requested']:,.0f}."
                    if score is not None else
                    "No credit score returned by the bureau."
                ),
                "typical_duration_minutes": 80,
                "complexity_rationale": (
                    "Bureau returned no score." if score is None
                    else "Full credit data available."
                ),
                "case_complexity": 4 if score is None else 2,
            }, None, "in_review")

        # Missing bureau data, first pass only: send it back once.
        if score is None and first_pass:
            return self._emit(state, "RequestAdditionalInfo", {
                "case_id": case_id,
                "requested_from": "junior_clerk",
                "information_needed": ["Applicant identification for the bureau query"],
                "reason": "The credit bureau returned no record; the file cannot be assessed as it stands.",
                "typical_duration_minutes": 165,
                "complexity_rationale": "Case cannot proceed without bureau data.",
                "case_complexity": 4,
            }, "junior_clerk", "pending")

        # Escalate with a recommendation.
        if score is None:
            rec = "reject"
        elif score >= GOOD_SCORE:
            rec = "approve"
        elif score < POOR_SCORE:
            rec = "reject"
        else:
            rec = "conditional"

        return self._emit(state, "EscalateCase", {
            "case_id": case_id,
            "risk_summary": (
                f"Score {score if score is not None else 'unavailable'}, "
                f"EUR {app['amount_requested']:,.0f} for {app['loan_goal']}. "
                f"Recommendation: {rec}."
            ),
            "recommendation": rec,
            "typical_duration_minutes": 11,
            "complexity_rationale": "Handover with a formed recommendation.",
            "case_complexity": 2,
        }, "credit_officer", "in_review")


# -------------------------------------------------
# Credit Officer
# -------------------------------------------------

class RuleCreditOfficer(RuleBasedAgent):
    """
    Sequence: AssessRisk -> approve or reject.

    Decides on fixed credit score bands (GOOD_SCORE / POOR_SCORE), and
    follows the senior clerk's recommendation on borderline files.
    """

    name = "credit_officer"
    tool_schemas = [AssessRisk, ApproveLoan, RejectLoan]

    def _recommendation(self, state: ProcessState) -> str:
        for a in reversed(state["agent_history"]):
            if a["tool_name"] == "EscalateCase":
                return a["tool_output"].get("recommendation", "conditional")
        return "conditional"

    def __call__(self, state: ProcessState) -> dict:
        app = state["application"]
        case_id = app["case_id"]
        amount = app["amount_requested"]
        cbd = state.get("credit_bureau_data") or {}
        score = cbd.get("credit_score")
        used = self._my_tools_used(state)

        if "AssessRisk" not in used:
            if score is None:
                category, factors = "high", ["No credit score available"]
            elif score >= GOOD_SCORE:
                category, factors = "low", []
            elif score < POOR_SCORE:
                category, factors = "high", [f"Credit score {score} below threshold"]
            else:
                category, factors = "medium", [f"Credit score {score} in middle band"]
            if amount > HIGH_AMOUNT:
                factors = factors + [f"Exposure of EUR {amount:,.0f}"]

            return self._emit(state, "AssessRisk", {
                "case_id": case_id,
                "risk_category": category,
                "risk_factors": factors,
                "assessment_notes": f"Formal assessment: {category} risk.",
                "typical_duration_minutes": 135,
                "complexity_rationale": (
                    "No score to work from." if score is None
                    else "Standard assessment against the score bands."
                ),
                "case_complexity": 4 if score is None else 2,
            }, "credit_officer", "in_review")

        rec = self._recommendation(state)
        approve = (
            score is not None
            and score >= POOR_SCORE
            and (score >= GOOD_SCORE or rec != "reject")
        )

        if approve:
            rate = 0.04 if score >= GOOD_SCORE else 0.075
            conditions = ["Collateral required"] if amount > HIGH_AMOUNT else []
            return self._emit(state, "ApproveLoan", {
                "case_id": case_id,
                "approved_amount": amount,
                "interest_rate": rate,
                "conditions": conditions,
                "approval_notes": f"Approved on a score of {score}.",
                "typical_duration_minutes": 22,
                "complexity_rationale": "Decision follows directly from the score band.",
                "case_complexity": 2,
            }, None, "approved")

        reasons = (
            ["No credit score available"] if score is None
            else [f"Credit score {score} below the acceptable threshold"]
        )
        return self._emit(state, "RejectLoan", {
            "case_id": case_id,
            "rejection_reasons": reasons,
            "rejection_notes": "Rejected on credit risk grounds.",
            "typical_duration_minutes": 18,
            "complexity_rationale": "Decision follows directly from the score band.",
            "case_complexity": 2,
        }, None, "rejected")
