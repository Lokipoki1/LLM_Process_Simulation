"""
observer.py
-----------
Turns a recorded agent action into an XES event log entry.

The mapping from tool name to activity label and from role to resource
label comes from the ProcessDefinition, so this module holds no domain
vocabulary of its own.

Two timestamps per event
    Every entry carries both the moment the activity STARTED and the
    moment it FINISHED. The gap between them is the hands-on work plus
    any shift suspension; the gap between one event's end and the next
    event's start is queue waiting.

    This is not cosmetic. The temporal log-distance measures (AED, CED,
    RED) are computed over start AND end timestamps, and BPI17W - the
    reference subset AgentSimulator evaluates on - is built precisely
    from the BPIC activities that carry both. With a single timestamp
    per event, half the temporal evaluation cannot run, and the queueing
    behaviour the engine exists to model is invisible in the output.

Timestamps are supplied by the engine, which owns the simulation
timeline. This module only formats.
"""

from __future__ import annotations
from datetime import datetime, timezone

from ..state import XESEntry, AgentAction


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def build_xes_entry(
    action: AgentAction,
    activity_map: dict[str, str],
    resource_map: dict[str, str],
    start_timestamp: float | None = None,
) -> XESEntry:
    """
    Format one agent action as an XES entry.

    Args:
        action:          the recorded action, with sim_timestamp already
                         set by the engine to the activity's finish time
        activity_map:    tool name -> activity label
        resource_map:    role -> org:resource label
        start_timestamp: when the activity began. Falls back to the
                         finish time minus the recorded duration, and
                         then to the finish time itself.
    """
    end = action["sim_timestamp"]

    if start_timestamp is None:
        duration = action.get("real_duration") or 0.0
        start_timestamp = end - duration

    return XESEntry(
        case_concept_name=action["tool_input"].get("case_id", "UNKNOWN"),
        concept_name=activity_map.get(action["tool_name"], action["tool_name"]),
        start_timestamp=_iso(start_timestamp),
        time_timestamp=_iso(end),
        org_resource=resource_map.get(action["agent_name"], action["agent_name"]),
        lifecycle_transition="complete",
    )
