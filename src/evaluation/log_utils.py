"""
log_utils.py
------------
Loading, filtering and splitting event logs for evaluation.

  - vocabulary filtering: restrict the reference log to the simulated
    activities (8 of the 26 in BPIC 2017), reporting before/after counts
  - temporal splitting: calibration / evaluation hold-out by case arrival
  - case matching and alignment, for measures that need equal case counts
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


# Column names used throughout. These match the defaults of the
# log-distance-measures package so no remapping is needed downstream.
CASE_ID = "case_id"
ACTIVITY = "activity"
START = "start_time"
END = "end_time"
RESOURCE = "resource"


@dataclass
class LogStats:
    """Descriptive counts, for the before/after table in the write-up."""
    name: str
    n_cases: int
    n_events: int
    n_activities: int
    n_variants: int
    mean_trace_length: float
    activities: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "log": self.name,
            "cases": self.n_cases,
            "events": self.n_events,
            "activities": self.n_activities,
            "variants": self.n_variants,
            "mean_trace_len": round(self.mean_trace_length, 1),
        }


# -------------------------------------------------
# Loading
# -------------------------------------------------

def load_xes(path: str | Path, collapse_lifecycle: bool = True) -> pd.DataFrame:
    """
    Read an XES file into the flat frame the measures expect.

    With collapse_lifecycle, each schedule/start/complete group becomes
    one activity instance with a start and end timestamp. Logs without a
    start timestamp get start = end.
    """
    import pm4py

    log = pm4py.read_xes(str(path))
    df = pm4py.convert_to_dataframe(log)

    out = pd.DataFrame({
        CASE_ID: df["case:concept:name"].astype(str),
        ACTIVITY: df["concept:name"].astype(str),
        END: pd.to_datetime(df["time:timestamp"], utc=True, format="ISO8601"),
    })

    start_col = next(
        (c for c in ("start_timestamp", "start:timestamp", "time:start")
         if c in df.columns),
        None,
    )
    out[START] = (
        pd.to_datetime(df[start_col], utc=True, format="ISO8601")
        if start_col else out[END]
    )

    out[RESOURCE] = (
        df["org:resource"].astype(str) if "org:resource" in df.columns else "unknown"
    )

    lifecycle_col = next(
        (c for c in ("lifecycle:transition", "lifecycle_transition")
         if c in df.columns),
        None,
    )
    if collapse_lifecycle and lifecycle_col is not None:
        out["_lifecycle"] = df[lifecycle_col].astype(str).str.lower()
        if out["_lifecycle"].nunique() > 1:
            out = _collapse_lifecycle(out)
        else:
            out = out.drop(columns="_lifecycle")

    return out.sort_values([CASE_ID, END]).reset_index(drop=True)


def _collapse_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fold schedule/start/complete groups into single activity instances.
    A group opens on the first non-complete transition and closes on the
    next complete of the same activity.
    """
    rows = []

    for case_id, case_df in df.sort_values([CASE_ID, END]).groupby(CASE_ID, sort=False):
        open_groups: dict[str, dict] = {}

        for _, ev in case_df.iterrows():
            activity = ev[ACTIVITY]
            phase = ev["_lifecycle"]

            if phase in ("complete", "completed"):
                started = open_groups.pop(activity, None)
                rows.append({
                    CASE_ID: case_id,
                    ACTIVITY: activity,
                    START: started[START] if started else ev[END],
                    END: ev[END],
                    RESOURCE: (started or ev)[RESOURCE],
                })
            else:
                # schedule, start, assign, withdraw, suspend, resume...
                # keep the earliest one as the instance's start
                if activity not in open_groups:
                    open_groups[activity] = {START: ev[END], RESOURCE: ev[RESOURCE]}

        # activity instances that never completed
        for activity, started in open_groups.items():
            rows.append({
                CASE_ID: case_id,
                ACTIVITY: activity,
                START: started[START],
                END: started[START],
                RESOURCE: started[RESOURCE],
            })

    return pd.DataFrame(rows, columns=[CASE_ID, ACTIVITY, START, END, RESOURCE])


def describe(df: pd.DataFrame, name: str) -> LogStats:
    """Descriptive statistics of a log."""
    variants = df.groupby(CASE_ID)[ACTIVITY].apply(tuple)
    return LogStats(
        name=name,
        n_cases=df[CASE_ID].nunique(),
        n_events=len(df),
        n_activities=df[ACTIVITY].nunique(),
        n_variants=variants.nunique(),
        mean_trace_length=df.groupby(CASE_ID).size().mean(),
        activities=sorted(df[ACTIVITY].unique()),
    )


# -------------------------------------------------
# Filtering
# -------------------------------------------------

def filter_to_vocabulary(
    df: pd.DataFrame,
    vocabulary: set[str] | frozenset[str],
    drop_empty_cases: bool = True,
) -> pd.DataFrame:
    """Keep only events whose activity is in `vocabulary`; drop emptied cases."""
    out = df[df[ACTIVITY].isin(vocabulary)].copy()
    if drop_empty_cases:
        keep = out.groupby(CASE_ID).size()
        out = out[out[CASE_ID].isin(keep[keep > 0].index)]
    return out.reset_index(drop=True)


def prepare_reference_log(
    path: str | Path,
    vocabulary: set[str] | frozenset[str],
    collapse_lifecycle: bool = True,
) -> tuple[pd.DataFrame, LogStats, LogStats]:
    """
    Load a reference log and restrict it to the simulated vocabulary.
    Returns (filtered, stats_before, stats_after).
    """
    raw = load_xes(path, collapse_lifecycle=collapse_lifecycle)
    before = describe(raw, "reference (raw)")
    filtered = filter_to_vocabulary(raw, vocabulary)
    after = describe(filtered, "reference (filtered)")
    return filtered, before, after


def vocabulary_overlap(
    simulated_vocabulary: set[str] | frozenset[str],
    reference_activities: list[str],
) -> dict[str, list[str]]:
    """Matched, simulated-only and reference-only activity labels."""
    sim = set(simulated_vocabulary)
    ref = set(reference_activities)
    return {
        "matched": sorted(sim & ref),
        "simulated_only": sorted(sim - ref),
        "reference_only": sorted(ref - sim),
    }


# -------------------------------------------------
# Splitting
# -------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by case arrival time, dropping cases that straddle the cut."""
    if df.empty:
        raise ValueError(
            "Cannot split an empty log. If this is the reference log after "
            "filtering, the simulated vocabulary shares no activity with it - "
            "check the activity mapping."
        )

    arrivals = df.groupby(CASE_ID)[START].min().sort_values()
    ends = df.groupby(CASE_ID)[END].max()

    if len(arrivals) < 5:
        raise ValueError(
            f"Only {len(arrivals)} case(s) in the log - too few to split. "
            "Pass --no-split to evaluate against the whole log instead."
        )

    # Clamp so a fraction of 1.0, or rounding on a small log, cannot run
    # past the end of the series.
    cut_index = min(int(len(arrivals) * train_fraction), len(arrivals) - 1)
    cut_time = arrivals.iloc[cut_index]

    train_ids = arrivals[(arrivals < cut_time) & (ends < cut_time)].index
    test_ids = arrivals[arrivals >= cut_time].index

    return (
        df[df[CASE_ID].isin(train_ids)].reset_index(drop=True),
        df[df[CASE_ID].isin(test_ids)].reset_index(drop=True),
    )


def sample_cases(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Take a random sample of `n` complete cases."""
    ids = pd.Series(df[CASE_ID].unique())
    if len(ids) <= n:
        return df.reset_index(drop=True)
    chosen = ids.sample(n=n, random_state=seed)
    return df[df[CASE_ID].isin(chosen)].reset_index(drop=True)


def align_case_counts(
    a: pd.DataFrame, b: pd.DataFrame, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trim whichever log has more cases so both hold the same number."""
    n = min(a[CASE_ID].nunique(), b[CASE_ID].nunique())
    return sample_cases(a, n, seed), sample_cases(b, n, seed)


def reference_window(df: pd.DataFrame, n_cases: int, seed: int = 42) -> pd.DataFrame:
    """
    Take a block of `n_cases` consecutive arrivals, so the reference
    covers a time span comparable to the simulation (for AED and CAR).
    """
    arrivals = df.groupby(CASE_ID)[START].min().sort_values()
    if len(arrivals) <= n_cases:
        return df.reset_index(drop=True)

    rng = pd.Series(range(len(arrivals) - n_cases + 1)).sample(1, random_state=seed)
    offset = int(rng.iloc[0])
    chosen = arrivals.index[offset:offset + n_cases]
    return df[df[CASE_ID].isin(chosen)].reset_index(drop=True)


def shift_to_match(simulated: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Shift the simulated log so its first arrival matches the reference's."""
    sim_start = simulated[START].min()
    ref_start = reference[START].min()
    delta = ref_start - sim_start

    out = simulated.copy()
    out[START] = out[START] + delta
    out[END] = out[END] + delta
    return out


def select_cases(df: pd.DataFrame, case_ids: set[str]) -> pd.DataFrame:
    """Keep only the given cases, preserving their order."""
    return df[df[CASE_ID].isin(case_ids)].reset_index(drop=True)


def match_by_case_id(
    reference: pd.DataFrame, simulated: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Restrict both logs to their shared case IDs (a paired comparison of
    replayed cases). Returns both logs and an overlap report.
    """
    ref_ids = set(reference[CASE_ID].unique())
    sim_ids = set(simulated[CASE_ID].unique())
    shared = ref_ids & sim_ids

    info = {
        "reference_cases": len(ref_ids),
        "simulated_cases": len(sim_ids),
        "shared": len(shared),
        "simulated_only": len(sim_ids - ref_ids),
    }

    if not shared:
        return reference, simulated, info

    return select_cases(reference, shared), select_cases(simulated, shared), info
