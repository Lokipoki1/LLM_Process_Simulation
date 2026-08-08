"""
simulation_controller.py — updated for BPIC 2017 structure
"""

from __future__ import annotations
import logging
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

from .logger import setup_logging
from .state import LoanApplication

logger = logging.getLogger("bps.controller")


LOAN_GOALS = ["Car", "Home improvement", "Existing loan takeover", "Unknown"]
APP_TYPES = ["New credit", "Limit raise"]


def generate_synthetic_case(case_id: str, rng: np.random.Generator) -> dict:
    """
    Generate a synthetic case with BPIC 2017 structure.
    Returns a dict with application data + hidden credit bureau data.
    """
    profile = rng.choice(["good", "borderline", "bad"], p=[0.50, 0.25, 0.25])

    loan_goal = str(rng.choice(LOAN_GOALS))
    app_type = str(rng.choice(APP_TYPES, p=[0.85, 0.15]))

    if profile == "good":
        amount = float(rng.lognormal(mean=9.5, sigma=0.5))     # ~EUR 13000
        credit_score = int(np.clip(rng.normal(720, 40), 650, 850))
        terms = int(rng.choice([36, 48, 60, 72]))
    elif profile == "borderline":
        amount = float(rng.lognormal(mean=10.0, sigma=0.6))    # ~EUR 22000
        credit_score = int(np.clip(rng.normal(620, 50), 500, 700))
        terms = int(rng.choice([24, 36, 48]))
    else:  # bad
        amount = float(rng.lognormal(mean=10.5, sigma=0.7))    # ~EUR 36000
        credit_score = int(np.clip(rng.normal(480, 70), 300, 600))
        terms = int(rng.choice([12, 24, 36]))

    monthly_cost = round(amount / terms * 1.06, 2)

    return {
        # Application data (visible from start)
        "case_id": case_id,
        "amount_requested": round(amount, 2),
        "loan_goal": loan_goal,
        "application_type": app_type,
        # Credit bureau data (hidden until SC checks)
        "credit_bureau_data": {
            "credit_score": credit_score,
            "monthly_cost": monthly_cost,
            "number_of_terms": terms,
            "offered_amount": round(amount, 2),
        },
    }


def load_bpic_cases(xes_path: str, max_cases: int | None = None) -> list[dict]:
    """
    Load real cases from BPIC 2017 XES file.

    Case-level attributes (application data):
      - case:concept:name  -> case_id
      - RequestedAmount    -> amount_requested
      - LoanGoal           -> loan_goal
      - ApplicationType    -> application_type

    Offer-level attributes (credit bureau data):
      - CreditScore        -> credit_score
      - MonthlyCost        -> monthly_cost
      - NumberOfTerms      -> number_of_terms
      - OfferedAmount      -> offered_amount

    Cases without offers (early rejections) have credit_bureau_data=None.
    """
    import pm4py

    log = pm4py.read_xes(xes_path)
    df  = pm4py.convert_to_dataframe(log)

    # Get case-level attributes (one row per case)
    case_attrs = df.groupby("case:concept:name").first().reset_index()
    if max_cases:
        case_attrs = case_attrs.head(max_cases)

    # Get offer-level attributes (from O_Create Offer events)
    offer_events = df[df["concept:name"].str.startswith("O_Create", na=False)]
    offer_by_case = offer_events.groupby("case:concept:name").first()

    cases = []
    for _, row in case_attrs.iterrows():
        case_id = str(row["case:concept:name"])

        # Application data (always available)
        application = {
            "case_id": case_id,
            "amount_requested": float(row.get("RequestedAmount", row.get("case:RequestedAmount", 0))),
            "loan_goal": str(row.get("LoanGoal", row.get("case:LoanGoal", "Unknown"))),
            "application_type": str(row.get("ApplicationType", row.get("case:ApplicationType", "New credit"))),
        }

        # Credit bureau data (from offer, may not exist)
        credit_data = None
        if case_id in offer_by_case.index:
            offer = offer_by_case.loc[case_id]
            cs = offer.get("CreditScore", offer.get("case:CreditScore", None))
            credit_data = {
                "credit_score": int(cs) if pd.notna(cs) and cs else None,
                "monthly_cost": float(offer.get("MonthlyCost", 0)) if pd.notna(offer.get("MonthlyCost")) else None,
                "number_of_terms": int(offer.get("NumberOfTerms", 0)) if pd.notna(offer.get("NumberOfTerms")) else None,
                "offered_amount": float(offer.get("OfferedAmount", 0)) if pd.notna(offer.get("OfferedAmount")) else None,
            }

        cases.append({
            **application,
            "credit_bureau_data": credit_data,
        })

    logger.info("Loaded %d cases from BPIC 2017 (%d with credit data)",
                len(cases), sum(1 for c in cases if c["credit_bureau_data"]))
    return cases
