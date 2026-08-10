"""
loan_application.py
-------------------
The loan application process, as a single ProcessDefinition.

Everything domain-specific about this simulation is either in this file
or reachable from it: the three roles, who may hand a case to whom, the
activity vocabulary, and how a raw case dict becomes a ProcessState.

This is the reference instantiation. A second process would be a sibling
module of the same shape - the engine in src/queue/ does not change.

Reference log
    BPI Challenge 2017, a Dutch bank's loan application process.
    Case-level attributes are visible from the start; offer-level
    attributes (credit score, monthly cost, terms) are only revealed
    once the Senior Clerk queries the credit bureau, matching the point
    in the real process where that data first appears.
"""

from __future__ import annotations

from ..state import ProcessState
from ..queue.agent_pool import WorkSchedule
from ..agents.junior_clerk import JuniorClerk
from ..agents.senior_clerk import SeniorClerk
from ..agents.credit_officer import CreditOfficer
from .definition import ProcessDefinition


# -------------------------------------------------
# Initial state
# -------------------------------------------------

def make_loan_state(case_data: dict) -> ProcessState:
    """
    Build the starting state for a loan case.

    Splits the raw case into what the applicant submitted (visible
    immediately) and what the bank later obtains from the credit bureau
    (hidden until CheckCreditScore). That split is the information
    asymmetry the simulation depends on: the Junior Clerk decides with an
    incomplete picture, exactly as a real intake clerk does.
    """
    application = {
        "case_id":          case_data["case_id"],
        "amount_requested": case_data["amount_requested"],
        "loan_goal":        case_data["loan_goal"],
        "application_type": case_data.get("application_type", "New credit"),
    }
    return ProcessState(
        application=application,
        credit_bureau_data=case_data.get("credit_bureau_data"),
        credit_checked=False,
        status="pending",
        current_agent="junior_clerk",
        messages=[],
        agent_history=[],
        event_log=[],
        sim_clock=0.0,
        rework_count=0,
        rejection_reason=None,
        next_agent=None,
    )


# -------------------------------------------------
# Activity vocabulary
# -------------------------------------------------
# tool name -> XES activity label.
#
# The labels below are the framework's own vocabulary. To compute NGD or
# a directly-follows comparison against BPIC 2017, these need to be
# aligned with the reference log's labels - that alignment is a research
# decision, not an implementation detail, so it is left explicit here
# rather than hidden in the observer.

ACTIVITY_MAP: dict[str, str] = {
    "IntakeApplication":      "A_INTAKE",
    "CheckDocuments":         "A_CHECK_DOCS",
    "ForwardCase":            "A_FORWARD",
    "ReturnApplicationEarly": "O_RETURNED",
    "CheckCreditScore":       "W_CHECK_CREDIT",
    "ValidateApplication":    "A_VALIDATE",
    "RequestAdditionalInfo":  "A_REQUEST_INFO",
    "EscalateCase":           "A_ESCALATE",
    "AssessRisk":             "A_ASSESS_RISK",
    "ApproveLoan":            "O_APPROVED",
    "RejectLoan":             "O_DECLINED",
}

RESOURCE_MAP: dict[str, str] = {
    "junior_clerk":   "Junior Clerk",
    "senior_clerk":   "Senior Clerk",
    "credit_officer": "Credit Officer",
}


# -------------------------------------------------
# Silent tools
# -------------------------------------------------
# Tools that move the case without producing a log event.
#
# ForwardCase and EscalateCase are internal handovers: they change who
# holds the file, but a bank's information system records activities, not
# handoffs. AssessRisk is deliberation rather than a recorded step.
# Emitting all three lengthens every trace by two to three events
# relative to the reference log, which inflates any label-based distance.
#
# Left empty for now so the current results stay comparable with earlier
# runs. Enable by replacing the empty frozenset with the commented one,
# then re-run and compare trace lengths against BPIC 2017.

SILENT_TOOLS: frozenset[str] = frozenset()
# SILENT_TOOLS = frozenset({"ForwardCase", "EscalateCase", "AssessRisk"})


# -------------------------------------------------
# Routing
# -------------------------------------------------
# Which roles a case may move to from each role. A role that continues
# its own multi-step sequence must list itself.
#
#   junior_clerk   -> itself (intake, docs) or the senior clerk (forward)
#   senior_clerk   -> itself (bureau, validate), the credit officer
#                     (escalate), or back to the junior clerk (rework)
#   credit_officer -> itself (assess, then decide); the decision ends it

VALID_TRANSITIONS: dict[str, set[str]] = {
    "junior_clerk":   {"junior_clerk", "senior_clerk"},
    "senior_clerk":   {"senior_clerk", "credit_officer", "junior_clerk"},
    "credit_officer": {"credit_officer"},
}


# -------------------------------------------------
# The definition
# -------------------------------------------------

LOAN_PROCESS = ProcessDefinition(
    name="Loan application (BPIC 2017)",
    entry_role="junior_clerk",
    agent_classes={
        "junior_clerk":   JuniorClerk,
        "senior_clerk":   SeniorClerk,
        "credit_officer": CreditOfficer,
    },
    valid_transitions=VALID_TRANSITIONS,
    activity_map=ACTIVITY_MAP,
    resource_map=RESOURCE_MAP,
    silent_tools=SILENT_TOOLS,
    initial_state=make_loan_state,
    default_schedules={
        "junior_clerk":   WorkSchedule(8.0, 17.0),
        "senior_clerk":   WorkSchedule(8.0, 17.0),
        "credit_officer": WorkSchedule(9.0, 17.0),
    },
)
