"""
observer.py
-----------
Turns a recorded agent action into an XES event log entry.

The mapping from tool name to activity label and from role to resource
label comes from the ProcessDefinition, so this module holds no domain
vocabulary of its own.

Timestamps are supplied by the engine, which owns the simulation
timeline. This module only formats.
"""

from __future__ import annotations
from datetime import datetime, timezone

from ..state import XESEntry, AgentAction


def build_xes_entry(
    action: AgentAction,
    activity_map: dict[str, str],
    resource_map: dict[str, str],
) -> XESEntry:
    """
    Format one agent action as an XES entry.

    Args:
        action:       the recorded action, with sim_timestamp already
                      set by the engine to the activity's finish time
        activity_map: tool name -> activity label
        resource_map: role -> org:resource label
    """
    return XESEntry(
        case_concept_name=action["tool_input"].get("case_id", "UNKNOWN"),
        concept_name=activity_map.get(action["tool_name"], action["tool_name"]),
        time_timestamp=datetime.fromtimestamp(
            action["sim_timestamp"], tz=timezone.utc
        ).isoformat(),
        org_resource=resource_map.get(action["agent_name"], action["agent_name"]),
        lifecycle_transition="complete",
    )
