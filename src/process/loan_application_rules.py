"""
loan_application_rules.py
-------------------------
The same loan process with deterministic agents instead of LLM agents.
Everything else is identical to LOAN_PROCESS, so the two runs form a
controlled comparison.
"""

from __future__ import annotations

from ..agents.rule_based import RuleJuniorClerk, RuleSeniorClerk, RuleCreditOfficer
from .definition import ProcessDefinition
from .loan_application import (
    ACTIVITY_MAP, RESOURCE_MAP, SILENT_TOOLS, VALID_TRANSITIONS,
    make_loan_state, LOAN_PROCESS,
)

LOAN_PROCESS_RULES = ProcessDefinition(
    name="Loan application (BPIC 2017) - rule-based baseline",
    entry_role="junior_clerk",
    agent_classes={
        "junior_clerk":   RuleJuniorClerk,
        "senior_clerk":   RuleSeniorClerk,
        "credit_officer": RuleCreditOfficer,
    },
    valid_transitions=VALID_TRANSITIONS,
    activity_map=ACTIVITY_MAP,
    resource_map=RESOURCE_MAP,
    silent_tools=SILENT_TOOLS,
    initial_state=make_loan_state,
    default_schedules=LOAN_PROCESS.default_schedules,
)
