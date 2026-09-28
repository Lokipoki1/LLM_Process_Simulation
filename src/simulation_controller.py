"""
simulation_controller.py
------------------------
Case loading (BPIC 2017) and synthetic case generation.

Both return dicts with the same shape:
    {
      "case_id", "amount_requested", "loan_goal", "application_type",
      "arrival_time": float | None,
      "credit_bureau_data": {...} | None
    }
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("bps.controller")


LOAN_GOALS = ["Car", "Home improvement", "Existing loan takeover", "Unknown"]
APP_TYPES = ["New credit", "Limit raise"]


def generate_synthetic_case(case_id: str, rng: np.random.Generator) -> dict:
    """
    Generate a synthetic case (50% good / 25% borderline / 25% bad
    profile). No arrival time - the engine generates those.
    """
    profile = rng.choice(["good", "borderline", "bad"], p=[0.50, 0.25, 0.25])

    loan_goal = str(rng.choice(LOAN_GOALS))
    app_type = str(rng.choice(APP_TYPES, p=[0.85, 0.15]))

    if profile == "good":
        amount = float(rng.lognormal(mean=9.5, sigma=0.5))
        credit_score = int(np.clip(rng.normal(720, 40), 650, 850))
        terms = int(rng.choice([36, 48, 60, 72]))
    elif profile == "borderline":
        amount = float(rng.lognormal(mean=10.0, sigma=0.6))
        credit_score = int(np.clip(rng.normal(620, 50), 500, 700))
        terms = int(rng.choice([24, 36, 48]))
    else:
        amount = float(rng.lognormal(mean=10.5, sigma=0.7))
        credit_score = int(np.clip(rng.normal(480, 70), 300, 600))
        terms = int(rng.choice([12, 24, 36]))

    monthly_cost = round(amount / terms * 1.06, 2)

    return {
        "case_id":          case_id,
        "amount_requested": round(amount, 2),
        "loan_goal":        loan_goal,
        "application_type": app_type,
        "arrival_time":     None,
        "credit_bureau_data": {
            "credit_score":    credit_score,
            "monthly_cost":    monthly_cost,
            "number_of_terms": terms,
            "offered_amount":  round(amount, 2),
        },
    }


def load_bpic_cases(
    xes_path: str,
    max_cases: int | None = None,
    offset: int = 0,
) -> list[dict]:
    """
    Load real cases from a BPIC 2017 XES file.

    Takes a contiguous block of `max_cases` cases in arrival order,
    starting at `offset`, so the block covers a compact period of time.
    arrival_time is each case's first event. Case attributes become the
    application; O_Create Offer attributes become the credit bureau data
    (None when the case never reached an offer).
    """
    import pm4py

    log = pm4py.read_xes(xes_path)
    df = pm4py.convert_to_dataframe(log)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)

    # Case-level attributes plus the arrival time
    grouped = df.sort_values("time:timestamp").groupby("case:concept:name")
    case_attrs = grouped.first().reset_index()
    case_attrs["_arrival"] = grouped["time:timestamp"].min().values

    # Arrival order, then a contiguous block
    case_attrs = case_attrs.sort_values("_arrival").reset_index(drop=True)
    total = len(case_attrs)
    if offset:
        case_attrs = case_attrs.iloc[offset:]
    if max_cases:
        case_attrs = case_attrs.head(max_cases)

    # Offer-level attributes
    offer_events = df[df["concept:name"].str.startswith("O_Create", na=False)]
    offer_by_case = offer_events.groupby("case:concept:name").first()

    cases = []
    for _, row in case_attrs.iterrows():
        case_id = str(row["case:concept:name"])

        application = {
            "case_id": case_id,
            "amount_requested": float(
                row.get("RequestedAmount", row.get("case:RequestedAmount", 0))
            ),
            "loan_goal": str(
                row.get("LoanGoal", row.get("case:LoanGoal", "Unknown"))
            ),
            "application_type": str(
                row.get("ApplicationType", row.get("case:ApplicationType", "New credit"))
            ),
            "arrival_time": float(row["_arrival"].timestamp()),
        }

        credit_data = None
        if case_id in offer_by_case.index:
            offer = offer_by_case.loc[case_id]
            cs = offer.get("CreditScore", offer.get("case:CreditScore", None))
            credit_data = {
                "credit_score": int(cs) if pd.notna(cs) and cs else None,
                "monthly_cost": (
                    float(offer.get("MonthlyCost", 0))
                    if pd.notna(offer.get("MonthlyCost")) else None
                ),
                "number_of_terms": (
                    int(offer.get("NumberOfTerms", 0))
                    if pd.notna(offer.get("NumberOfTerms")) else None
                ),
                "offered_amount": (
                    float(offer.get("OfferedAmount", 0))
                    if pd.notna(offer.get("OfferedAmount")) else None
                ),
            }

        cases.append({**application, "credit_bureau_data": credit_data})

    if cases:
        span_start = pd.Timestamp(cases[0]["arrival_time"], unit="s", tz="UTC")
        span_end = pd.Timestamp(cases[-1]["arrival_time"], unit="s", tz="UTC")
        logger.info(
            "Loaded %d of %d cases (offset %d), arrivals %s to %s, %d with credit data",
            len(cases), total, offset,
            span_start.strftime("%Y-%m-%d"), span_end.strftime("%Y-%m-%d"),
            sum(1 for c in cases if c["credit_bureau_data"]),
        )
    return cases
