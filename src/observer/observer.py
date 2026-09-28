"""
observer.py
-----------
Turns a recorded agent action into an XES event log entry.

Each entry carries a start and an end timestamp, both supplied by the
engine: the temporal measures (AED, CED, RED) need both, and the gap
between events is what shows queue waiting.
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

    action["sim_timestamp"] is the finish time. Without start_timestamp,
    the start is the finish time minus the recorded duration.
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
