"""
state.py
--------
Data contracts for the multi-LLM-agent BPS framework.

Information flow mirrors the real BPIC 2017 loan process:
  - LoanApplication: what the client submits (visible from the start)
  - CreditBureauData: what the bank discovers during processing
    (hidden until Senior Clerk calls CheckCreditScore)

This information asymmetry is key to the thesis: agents make
decisions with different levels of information, just like real
bank workers.
"""

from __future__ import annotations
import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


# ─────────────────────────────────────────────
# 1. Initial application data (client submits this)
#    BPIC 2017 case-level attributes
# ─────────────────────────────────────────────

class LoanApplication(TypedDict):
    """
    What the client fills in when applying for a loan.
    These are the ONLY fields visible to the Junior Clerk.

    All fields come directly from BPIC 2017 case attributes:
      - case:concept:name     -> case_id
      - case:RequestedAmount  -> amount_requested
      - case:LoanGoal         -> loan_goal
      - case:ApplicationType  -> application_type
    """
    case_id: str
    amount_requested: float
    loan_goal: str               # "Car", "Existing loan takeover", etc.
    application_type: str        # "New credit", "Limit raise", etc.


# ─────────────────────────────────────────────
# 2. Credit bureau data (bank discovers this)
#    BPIC 2017 offer-level attributes
# ─────────────────────────────────────────────

class CreditBureauData(TypedDict):
    """
    Data revealed when the Senior Clerk consults the credit bureau.
    Hidden from the Junior Clerk — only visible after CheckCreditScore.

    In BPIC 2017 these appear on O_Create Offer events:
      - CreditScore    -> credit_score
      - MonthlyCost    -> monthly_cost
      - NumberOfTerms  -> number_of_terms
      - OfferedAmount  -> offered_amount

    For cases rejected before the offer stage,
    credit_score will be None (the bank never checked).
    """
    credit_score: Optional[int]
    monthly_cost: Optional[float]
    number_of_terms: Optional[int]
    offered_amount: Optional[float]


# ─────────────────────────────────────────────
# 3. Agent action record
# ─────────────────────────────────────────────

class AgentAction(TypedDict):
    agent_name: str
    tool_name: str
    tool_input: dict
    tool_output: dict
    sim_timestamp: float
    real_duration: float
    context_sent: str      # the exact prompt text the agent received
    raw_content: str       # prose the model emitted alongside the tool call


# ─────────────────────────────────────────────
# 4. XES event log entry
# ─────────────────────────────────────────────

class XESEntry(TypedDict):
    case_concept_name: str
    concept_name: str
    start_timestamp: str
    time_timestamp: str
    org_resource: str
    lifecycle_transition: str


# ─────────────────────────────────────────────
# 5. Process state
# ─────────────────────────────────────────────

class ProcessState(TypedDict):
    """
    Shared state for a single loan application case.

    Information asymmetry enforced by the context builder:
      - Junior Clerk:   sees application only
      - Senior Clerk:   sees application + credit_bureau_data (after check)
      - Credit Officer: sees everything + all prior reasoning
    """
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
