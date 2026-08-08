"""
observer.py — updated for BPIC 2017 tools
"""

from __future__ import annotations
from datetime import datetime, timezone
from ..state import ProcessState, XESEntry, AgentAction
from ..clock.simulation_clock import SimulationClock

TOOL_TO_ACTIVITY: dict[str, str] = {
    "IntakeApplication":       "A_INTAKE",
    "CheckDocuments":          "A_CHECK_DOCS",
    "ForwardCase":             "A_FORWARD",
    "ReturnApplicationEarly":  "O_RETURNED",
    "CheckCreditScore":        "W_CHECK_CREDIT",
    "ValidateApplication":     "A_VALIDATE",
    "RequestAdditionalInfo":   "A_REQUEST_INFO",
    "EscalateCase":            "A_ESCALATE",
    "AssessRisk":              "A_ASSESS_RISK",
    "ApproveLoan":             "O_APPROVED",
    "RejectLoan":              "O_DECLINED",
}

AGENT_TO_RESOURCE: dict[str, str] = {
    "junior_clerk":   "Junior Clerk",
    "senior_clerk":   "Senior Clerk",
    "credit_officer": "Credit Officer",
}


def build_xes_entry(action: AgentAction) -> XESEntry:
    activity = TOOL_TO_ACTIVITY.get(action["tool_name"], action["tool_name"])
    resource = AGENT_TO_RESOURCE.get(action["agent_name"], action["agent_name"])
    timestamp_iso = datetime.fromtimestamp(
        action["sim_timestamp"], tz=timezone.utc
    ).isoformat()

    return XESEntry(
        case_concept_name=action["tool_input"].get("case_id", "UNKNOWN"),
        concept_name=activity,
        time_timestamp=timestamp_iso,
        org_resource=resource,
        lifecycle_transition="complete",
    )


def observer_node(state: ProcessState, clock: SimulationClock) -> dict:
    if not state["agent_history"]:
        return {}
    last_action: AgentAction = state["agent_history"][-1]
    duration, new_ts = clock.advance_for_activity(last_action["tool_name"])
    last_action["sim_timestamp"] = new_ts
    last_action["real_duration"] = duration
    xes_entry = build_xes_entry(last_action)
    return {
        "sim_clock": new_ts,
        "event_log": [xes_entry],
    }
