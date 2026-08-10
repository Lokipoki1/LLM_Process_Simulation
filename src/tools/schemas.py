"""
schemas.py
----------
Pydantic tools for each agent role.

Information asymmetry
  - JC tools: only application data (amount, goal, type)
  - SC tools: include CheckCreditScore, which reveals bureau data
  - CO tools: full information (application + credit + handoff history)

Duration estimation
  Every tool carries `duration_minutes`: the agent's own estimate of how
  much hands-on working time the activity took for THIS case. This makes
  time estimation part of the agent's cognition rather than a statistical
  module bolted on beside it.

  The estimate is HANDS-ON WORK ONLY. Queue waiting, overnight gaps and
  weekends are added by the simulation engine, which knows the schedule
  and the queue state. An agent that says 45 minutes may well produce an
  event 19 hours later in the log.

  The engine can ignore these estimates and sample from BPIC-fitted
  distributions instead (`--durations distribution`), so both approaches
  can be compared on identical cases.
"""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


# Shared constraint for every duration estimate: 1 minute to 40 hours.
_DURATION = Field(
    ge=1, le=2400,
    description=(
        "Hands-on working minutes this activity took for THIS case. "
        "Exclude any waiting, queueing or overnight gaps. "
        "Scale it to the complexity of the case in front of you."
    ),
)


class DocumentStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FRAUDULENT = "fraudulent"


# -- Junior Clerk tools -------------------------
# Sees only: amount_requested, loan_goal, application_type

class IntakeApplication(BaseModel):
    """
    Register initial receipt of a loan application.
    First mandatory action for the Junior Clerk.
    """
    case_id: str = Field(description="Loan case ID")
    applicant_acknowledged: bool = Field(
        description="Has the applicant been notified of receipt?"
    )
    initial_notes: str = Field(
        description=(
            "Initial observations based ONLY on requested amount, "
            "loan goal, and application type"
        ),
        max_length=500,
    )
    duration_minutes: int = _DURATION


class CheckDocuments(BaseModel):
    """
    Verify that submitted documentation is complete and valid.
    """
    case_id: str
    document_status: DocumentStatus
    missing_documents: list[str] = Field(
        default_factory=list,
        description="Missing documents if status is 'incomplete'",
    )
    notes: str = Field(default="", max_length=500)
    duration_minutes: int = _DURATION


class ForwardCase(BaseModel):
    """
    Forward the case to the Senior Clerk for deep validation.
    Only valid when documentation is acceptable.
    """
    case_id: str
    forwarded_to: str = Field(default="senior_clerk")
    priority: str = Field(
        description="'normal' or 'high' based on amount and case complexity",
        pattern="^(normal|high)$",
    )
    duration_minutes: int = _DURATION


class ReturnApplicationEarly(BaseModel):
    """
    Return the application to the applicant without further processing.
    Used when the case clearly fails minimum requirements given the
    LIMITED information available to the Junior Clerk.
    """
    case_id: str
    return_reason: str = Field(
        description="'incomplete_docs' | 'fraudulent_docs' | 'unreasonable_request'",
        pattern="^(incomplete_docs|fraudulent_docs|unreasonable_request)$",
    )
    details: str = Field(max_length=400)
    duration_minutes: int = _DURATION


# -- Senior Clerk tools -------------------------
# Starts with the same info as the JC, but can query the credit bureau

class CheckCreditScore(BaseModel):
    """
    Consult the credit bureau for the applicant's credit data.
    Reveals: credit_score, monthly_cost, number_of_terms, offered_amount.
    MUST be called before ValidateApplication or EscalateCase.
    """
    case_id: str
    bureau_consulted: str = Field(
        default="national_credit_bureau",
        description="Which credit bureau was consulted",
    )
    check_notes: str = Field(
        default="",
        description="Reason for checking, or preliminary observations",
        max_length=300,
    )
    duration_minutes: int = _DURATION


class ValidateApplication(BaseModel):
    """
    Deep validation of the application using credit bureau data.
    Must be called AFTER CheckCreditScore.
    """
    case_id: str
    credit_score_acceptable: bool = Field(
        description="Is the credit score within acceptable range?"
    )
    risk_assessment: str = Field(
        description="Assessment based on credit score, monthly cost and loan terms",
        max_length=800,
    )
    duration_minutes: int = _DURATION


class RequestAdditionalInfo(BaseModel):
    """
    Request additional information from the applicant or Junior Clerk.
    Triggers a rework loop - the case goes back to the JC.
    """
    case_id: str
    requested_from: str = Field(
        description="'applicant' | 'junior_clerk'",
        pattern="^(applicant|junior_clerk)$",
    )
    information_needed: list[str] = Field(
        description="Specific list of required information"
    )
    reason: str = Field(max_length=500)
    duration_minutes: int = _DURATION


class EscalateCase(BaseModel):
    """
    Escalate the case to the Credit Officer for the final decision.
    """
    case_id: str
    risk_summary: str = Field(
        description="Detailed risk profile from bureau data and validation",
        max_length=800,
    )
    recommendation: str = Field(
        description="'approve' | 'reject' | 'conditional'",
        pattern="^(approve|reject|conditional)$",
    )
    duration_minutes: int = _DURATION


# -- Credit Officer tools -----------------------
# Sees everything: application + credit data + all prior reasoning

class AssessRisk(BaseModel):
    """
    Formal credit risk evaluation considering all available data.
    """
    case_id: str
    risk_category: str = Field(
        description="'low' | 'medium' | 'high'",
        pattern="^(low|medium|high)$",
    )
    risk_factors: list[str] = Field(
        description="Specific risk factors identified"
    )
    assessment_notes: str = Field(max_length=800)
    duration_minutes: int = _DURATION


class ApproveLoan(BaseModel):
    """
    Approve the loan application with specific conditions.
    """
    case_id: str
    approved_amount: float = Field(gt=0)
    interest_rate: float = Field(
        ge=0.0, le=1.0,
        description="Annual interest rate (e.g. 0.045 for 4.5 percent)",
    )
    conditions: list[str] = Field(
        default_factory=list,
        description="Additional approval conditions",
    )
    approval_notes: str = Field(default="", max_length=500)
    duration_minutes: int = _DURATION


class RejectLoan(BaseModel):
    """
    Reject the loan application with formal justification.
    """
    case_id: str
    rejection_reasons: list[str] = Field(
        min_length=1,
        description="At least one formal rejection reason",
    )
    rejection_notes: str = Field(max_length=500)
    duration_minutes: int = _DURATION


# -- Tool registry per agent role ---------------

AGENT_TOOLS: dict[str, list[type[BaseModel]]] = {
    "junior_clerk": [
        IntakeApplication, CheckDocuments, ForwardCase, ReturnApplicationEarly,
    ],
    "senior_clerk": [
        CheckCreditScore, ValidateApplication, RequestAdditionalInfo, EscalateCase,
    ],
    "credit_officer": [
        AssessRisk, ApproveLoan, RejectLoan,
    ],
}
