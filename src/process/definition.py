"""
definition.py
-------------
ProcessDefinition: everything the simulation engine needs to know about
a specific business process, and nothing it needs to know about how to
simulate one.

This type is the boundary of the framework. Above it, the engine
(events, queues, schedules, clock, XES export) is domain-agnostic:
nothing in src/queue/ mentions loans, credit scores or clerks. Below it,
a process instantiation supplies its own roles, tools, agent personas and
routing rules.

Porting the framework to another process means writing one new module
that constructs a ProcessDefinition, plus the state fields, Pydantic
tools and agent personas it points at. The engine is not touched.

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
        name
            Human-readable process name, used in logs and reports.

        entry_role
            The role a newly arrived case is queued for.

        agent_classes
            role -> BaseAgent subclass. The executor instantiates one
            agent per role and shares it across every worker of that role,
            the way two clerks follow the same job description.

        valid_transitions
            role -> the set of roles a case may move to from there. The
            engine rejects any other move, which catches an agent that
            hallucinates a routing target. A role that can continue its
            own sequence must include itself.

        activity_map
            tool name -> XES activity label. This is where the process
            vocabulary is defined, and the place to align labels with a
            reference log when computing NGD or a directly-follows graph.

        resource_map
            role -> org:resource label in the exported log.

        silent_tools
            Tools that advance the case but emit NO event in the log.
            Use this for internal handovers that the real system does not
            record: a clerk passing a file to a colleague changes who
            holds it, but a bank's information system logs activities,
            not handoffs. Emitting them inflates trace length and
            distorts any label-based distance metric.

        initial_state
            Builds the ProcessState for a case, given the raw case dict
            from the data loader. This is where the domain schema
            (which fields exist, what starts hidden) is fixed.

        default_schedules
            role -> WorkSchedule. Used when the run configuration does
            not override a role's hours.
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
        """
        Structural check of the definition. Returns a list of problems;
        an empty list means the definition is internally consistent.

        Run this once at startup - a malformed definition otherwise shows
        up as a stalled case halfway through a simulation.
        """
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
