"""
loan_application.py
-------------------
The loan application process (BPI Challenge 2017) as a ProcessDefinition:
roles, routing, activity vocabulary and initial state.
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
    Build the starting state for a loan case. Credit bureau data stays
    hidden until CheckCreditScore.
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
# tool name -> BPIC 2017 activity label, so both logs share an alphabet.
# A_ = application state, O_ = offer state (customer-driven), W_ = work
# item. There is no customer actor, so no tool maps to an O_ label.

ACTIVITY_MAP: dict[str, str] = {
    # Junior Clerk
    "IntakeApplication":      "A_Create Application",
    "CheckDocuments":         "W_Complete application",
    "ForwardCase":            "_HANDOVER_TO_SENIOR",       # silent
    "ReturnApplicationEarly": "A_Cancelled",

    # Senior Clerk
    "CheckCreditScore":       "_CREDIT_BUREAU_LOOKUP",     # silent
    "ValidateApplication":    "W_Validate application",
    "RequestAdditionalInfo":  "W_Call incomplete files",
    "EscalateCase":           "_HANDOVER_TO_OFFICER",      # silent

    # Credit Officer
    "AssessRisk":             "A_Validating",
    "ApproveLoan":            "A_Pending",
    "RejectLoan":             "A_Denied",
}

RESOURCE_MAP: dict[str, str] = {
    "junior_clerk":   "Junior Clerk",
    "senior_clerk":   "Senior Clerk",
    "credit_officer": "Credit Officer",
}


# -------------------------------------------------
# Silent tools
# -------------------------------------------------
# Tools that advance the case without producing a log event: the two
# handovers have no BPIC 2017 activity, and the credit score only
# appears there as an attribute of the later O_Create Offer.

SILENT_TOOLS: frozenset[str] = frozenset({
    "ForwardCase",
    "EscalateCase",
    "CheckCreditScore",
})


# -------------------------------------------------
# Routing
# -------------------------------------------------
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


# -------------------------------------------------
# Reference: the full BPIC 2017 vocabulary
# -------------------------------------------------
# All 26 activities in the log, for checking label coverage. The
# simulation produces 8 of them.

BPIC_2017_ACTIVITIES: frozenset[str] = frozenset({
    # Application state
    "A_Create Application", "A_Submitted", "A_Concept", "A_Accepted",
    "A_Complete", "A_Validating", "A_Incomplete", "A_Pending",
    "A_Denied", "A_Cancelled",
    # Offer state (customer-driven; the simulation has no customer actor)
    "O_Create Offer", "O_Created", "O_Sent (mail and online)",
    "O_Sent (online only)", "O_Returned", "O_Accepted", "O_Refused",
    "O_Cancelled",
    # Work items
    "W_Handle leads", "W_Complete application", "W_Call after offers",
    "W_Validate application", "W_Call incomplete files",
    "W_Assess potential fraud", "W_Shortened completion",
    "W_Personal Loan collection",
})


def check_mapping_coverage() -> dict:
    """Which emitted labels exist in the reference log. Run after changing ACTIVITY_MAP."""
    emitted = {
        label for tool, label in ACTIVITY_MAP.items()
        if tool not in SILENT_TOOLS
    }
    return {
        "emitted": sorted(emitted),
        "valid": sorted(emitted & BPIC_2017_ACTIVITIES),
        "unknown": sorted(emitted - BPIC_2017_ACTIVITIES),
        "unused_in_log": sorted(BPIC_2017_ACTIVITIES - emitted),
    }


if __name__ == "__main__":
    result = check_mapping_coverage()
    print(f"Emitted labels: {len(result['emitted'])}")
    for label in result["emitted"]:
        mark = "ok " if label in BPIC_2017_ACTIVITIES else "NOT IN LOG"
        print(f"  {mark:<11}{label}")
    if result["unknown"]:
        print(f"\nNot present in BPIC 2017: {result['unknown']}")
    else:
        print("\nEvery emitted label exists in BPIC 2017.")
    print(f"\nLog activities the simulation does not produce: "
          f"{len(result['unused_in_log'])}")
