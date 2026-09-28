"""
definition.py
-------------
ProcessDefinition: everything the engine needs to know about a specific
business process. The engine in src/queue/ is domain-agnostic; porting
to another process means writing a new ProcessDefinition.

See loan_application.py for the reference instantiation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

from ..state import ProcessState
from ..queue.agent_pool import WorkSchedule


@dataclass(frozen=True)
class ProcessDefinition:
    """
    A complete, self-contained description of one business process.

    Attributes:
        name               human-readable process name
        entry_role         role a newly arrived case is queued for
        agent_classes      role -> agent class (one instance shared per role)
        valid_transitions  role -> roles a case may move to; a role that
                           continues its own sequence must include itself
        activity_map       tool name -> XES activity label
        resource_map       role -> org:resource label
        silent_tools       tools that advance the case but emit no event
        initial_state      raw case dict -> ProcessState
        default_schedules  role -> WorkSchedule, unless the config overrides
    """

    name: str
    entry_role: str
    agent_classes: dict[str, type]
    valid_transitions: dict[str, set[str]]
    activity_map: dict[str, str]
    resource_map: dict[str, str]
    initial_state: Callable[[dict], ProcessState]
    silent_tools: frozenset[str] = frozenset()
    default_schedules: dict[str, WorkSchedule] = field(default_factory=dict)

    # -- Derived views -------------------------

    @property
    def roles(self) -> list[str]:
        return list(self.agent_classes.keys())

    def schedule_for(self, role: str) -> WorkSchedule:
        return self.default_schedules.get(role, WorkSchedule())

    def is_silent(self, tool_name: str | None) -> bool:
        """True if this tool advances the case without producing an event."""
        return tool_name in self.silent_tools

    def activity_label(self, tool_name: str) -> str:
        return self.activity_map.get(tool_name, tool_name)

    def resource_label(self, role: str) -> str:
        return self.resource_map.get(role, role)

    def validate(self) -> list[str]:
        """Structural check of the definition; returns a list of problems."""
        problems: list[str] = []
        roles = set(self.agent_classes)

        if self.entry_role not in roles:
            problems.append(
                f"entry_role '{self.entry_role}' is not in agent_classes"
            )

        for role, targets in self.valid_transitions.items():
            if role not in roles:
                problems.append(f"valid_transitions has unknown role '{role}'")
            for target in targets:
                if target not in roles:
                    problems.append(
                        f"valid_transitions['{role}'] points at unknown role '{target}'"
                    )

        for role in roles:
            if role not in self.valid_transitions:
                problems.append(f"role '{role}' has no entry in valid_transitions")
            if role not in self.resource_map:
                problems.append(f"role '{role}' has no entry in resource_map")

        for role, agent_cls in self.agent_classes.items():
            for schema in getattr(agent_cls, "tool_schemas", []):
                if schema.__name__ not in self.activity_map:
                    problems.append(
                        f"tool '{schema.__name__}' ({role}) is missing from activity_map"
                    )

        for tool in self.silent_tools:
            if tool not in self.activity_map:
                problems.append(f"silent tool '{tool}' is not a known tool")

        return problems
