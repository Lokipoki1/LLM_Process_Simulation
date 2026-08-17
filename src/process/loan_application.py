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
# tool name -> XES activity label, using BPIC 2017's own vocabulary so
# the synthetic log and the reference log share an alphabet. Without
# that, any label-based distance (n-gram, directly-follows) compares
# disjoint sets and reports maximum distance regardless of how faithful
# the simulation is.
#
# The labels were inherited from BPIC 2012 (A_INTAKE, O_APPROVED, ...)
# and never updated when the project moved datasets. These are the
# BPIC 2017 equivalents.
#
# Reading the BPIC 2017 prefixes:
#   A_  the state of the APPLICATION, as the bank sees it
#   O_  the state of the OFFER, driven by what the CUSTOMER does with it
#   W_  work items: what an employee actually performs
#
# The simulation has no customer actor, so no tool maps to an O_ label:
# the Credit Officer decides whether the BANK lends, which is A_Pending
# or A_Denied, not the customer accepting or refusing an offer.

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
# Tools that advance the case without producing a log event.
#
# ForwardCase and EscalateCase are internal handovers. They change who
# holds the file, but a bank's information system records activities,
# not handoffs - there is no BPIC 2017 activity for "the junior clerk
# passed this to the senior clerk".
#
# CheckCreditScore is information gathering. The credit score does exist
# in BPIC 2017, but as an ATTRIBUTE of O_Create Offer, and offer
# creation happens after the bank has decided to lend - later in the
# process than the point where the Senior Clerk looks the score up.
# Mapping it to O_Create Offer would put that activity before
# validation, inverting the real order.
#
# Emitting all three inflates every trace by three events relative to
# the reference log, which distorts trace-length and n-gram comparisons.

SILENT_TOOLS: frozenset[str] = frozenset({
    "ForwardCase",
    "EscalateCase",
    "CheckCreditScore",
})


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


# -------------------------------------------------
# Reference: the full BPIC 2017 vocabulary
# -------------------------------------------------
# All 26 activities in the log, for checking coverage and for deciding
# what a future, finer-grained instantiation might model. The simulation
# currently produces 7 of these; the rest belong to steps it abstracts
# away (lead handling, offer dispatch, customer response, fraud checks,
# collection).

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
    """
    Which emitted labels exist in the reference log, and which do not.

    Run this after changing ACTIVITY_MAP. Any label reported as unknown
    will have no counterpart in the reference log and will contribute
    maximum distance to every label-based metric.
    """
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
