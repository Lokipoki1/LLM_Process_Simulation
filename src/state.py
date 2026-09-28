"""
state.py
--------
Data contracts for the simulation: the case data, the agent action
record, the XES entry and the per-case process state.
"""

from __future__ import annotations
import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


# -------------------------------------------------
# 1. Initial application data (client submits this)
#    BPIC 2017 case-level attributes
# -------------------------------------------------

class LoanApplication(TypedDict):
    """What the client submits (BPIC 2017 case attributes). All the JC sees."""
    case_id: str
    amount_requested: float
    loan_goal: str               # "Car", "Existing loan takeover", etc.
    application_type: str        # "New credit", "Limit raise", etc.


# -------------------------------------------------
# 2. Credit bureau data (bank discovers this)
#    BPIC 2017 offer-level attributes
# -------------------------------------------------

class CreditBureauData(TypedDict):
    """
    Revealed by CheckCreditScore (BPIC 2017 O_Create Offer attributes).
    None for cases that never reached the offer stage.
    """
    credit_score: Optional[int]
    monthly_cost: Optional[float]
    number_of_terms: Optional[int]
    offered_amount: Optional[float]


# -------------------------------------------------
# 3. Agent action record
# -------------------------------------------------

class AgentAction(TypedDict):
    agent_name: str
    tool_name: str
    tool_input: dict
    tool_output: dict
    sim_timestamp: float
    real_duration: float
    context_sent: str      # the exact prompt text the agent received
    raw_content: str       # prose the model emitted alongside the tool call


# -------------------------------------------------
# 4. XES event log entry
# -------------------------------------------------

class XESEntry(TypedDict):
    case_concept_name: str
    concept_name: str
    start_timestamp: str
    time_timestamp: str
    org_resource: str
    lifecycle_transition: str


# -------------------------------------------------
# 5. Process state
# -------------------------------------------------

class ProcessState(TypedDict):
    """Shared state for a single case. List fields are append-only."""
    # Case data
    application: LoanApplication
    credit_bureau_data: Optional[dict]    # CreditBureauData, None until loaded
    credit_checked: bool                  # False until SC calls CheckCreditScore

    # Process control
    status: str
    current_agent: str
    rework_count: int
    rejection_reason: Optional[str]
    next_agent: Optional[str]

    # Append-only collections
    messages: Annotated[list, operator.add]
    agent_history: Annotated[list[AgentAction], operator.add]
    event_log: Annotated[list[XESEntry], operator.add]

    # Simulation clock
    sim_clock: float
