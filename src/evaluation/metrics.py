"""
metrics.py
----------
Log-distance measures between a synthetic log and a reference log.

These are the metrics established for business process simulation
quality by Chapela-Campa et al., "A framework for measuring the quality
of business process simulation models" (Information Systems 127, 2025;
BPM 2023). They compare EVENT LOGS rather than models, which is what
makes them usable here: the reference process has no published model,
and two simulators rarely share a representation.

Three perspectives, and what each one tests in this engine:

    control-flow    n-gram distance over activity sequences.
                    Tests whether the agents route cases the way the
                    real process routes them.

    temporal        absolute, circadian and relative event
                    distributions. Relative is the sharpest test of the
                    engine: it measures when events happen RELATIVE to
                    case arrival, so it is sensitive to activity
                    durations, queue waiting and shift suspension all at
                    once.

    congestion      case arrival rate and cycle time distribution.
                    Cycle time aggregates processing, waiting and
                    resource availability, making it the single best
                    summary of whether the queueing model behaves.

On interpreting the numbers
    N-gram distance is normalised to [0, 1] and is comparable across
    studies. The distribution measures are Earth Mover's Distance in
    units of the discretisation bin (hours here) divided by the number
    of observations, so a value of 10 means "on average each observation
    sat 10 hours away from where the reference log put it". Those are
    NOT comparable across papers unless the binning, the log size and
    the normalisation all match.

On running more than once
    A simulation is stochastic and the reference log is a single
    realisation, so one run gives one draw from a distribution of
    possible distances. The convention is to simulate K times and report
    the mean with a confidence interval; aggregate_runs() does that.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import pandas as pd

from .log_utils import CASE_ID, ACTIVITY, START, END, align_case_counts


# -------------------------------------------------
# Package availability
# -------------------------------------------------

try:
    from log_distance_measures.config import (
        EventLogIDs, AbsoluteTimestampType, discretize_to_hour,
    )
    from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
    from log_distance_measures.absolute_event_distribution import (
        absolute_event_distribution_distance,
    )
    from log_distance_measures.circadian_event_distribution import (
        circadian_event_distribution_distance,
    )
    from log_distance_measures.relative_event_distribution import (
        relative_event_distribution_distance,
    )
    from log_distance_measures.case_arrival_distribution import (
        case_arrival_distribution_distance,
    )
    from log_distance_measures.cycle_time_distribution import (
        cycle_time_distribution_distance,
    )
    from log_distance_measures.control_flow_log_distance import (
        control_flow_log_distance,
    )
    MEASURES_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    MEASURES_AVAILABLE = False
    _IMPORT_ERROR = exc


LOG_IDS = None
if MEASURES_AVAILABLE:
    LOG_IDS = EventLogIDs(
        case=CASE_ID,
        activity=ACTIVITY,
        start_time=START,
        end_time=END,
    )


def require_package() -> None:
    if not MEASURES_AVAILABLE:
        raise ImportError(
            "log-distance-measures is not installed.\n"
            "  pip install log-distance-measures\n"
            f"Original error: {_IMPORT_ERROR}"
        )


# -------------------------------------------------
# Result container
# -------------------------------------------------

@dataclass
class MetricResult:
    name: str
    value: float | None
    perspective: str
    note: str = ""

    @property
    def failed(self) -> bool:
        return self.value is None


def _safe(name: str, perspective: str, fn) -> MetricResult:
    """Run one measure, turning a failure into a reported result."""
    try:
        return MetricResult(name, float(fn()), perspective)
    except Exception as exc:
        return MetricResult(name, None, perspective, f"{type(exc).__name__}: {exc}")


# -------------------------------------------------
# Individual measures
# -------------------------------------------------

def control_flow_metrics(
    reference: pd.DataFrame, simulated: pd.DataFrame, n: int = 2,
) -> list[MetricResult]:
    """
    N-gram distance over activity sequences.

    n=2 is the value used in the literature: the bigram histogram of a
    log is its directly-follows graph, which the authors found carries
    enough control-flow information for models without duplicate
    activities. An n-gram present in only one log contributes its full
    frequency, so differing vocabularies are handled without special
    casing - which also means an unfiltered vocabulary mismatch will
    dominate the result.
    """
    require_package()
    return [
        _safe(
            f"NGD (n={n})", "control-flow",
            lambda: n_gram_distribution_distance(
                reference, LOG_IDS, simulated, LOG_IDS, n=n,
            ),
        )
    ]


def control_flow_log_distance_metric(
    reference: pd.DataFrame, simulated: pd.DataFrame, seed: int = 42,
) -> MetricResult:
    """
    Optimal-matching distance between traces.

    Requires equal case counts and is expensive - cubic in the number of
    cases. The framework authors report a rank correlation of 1.0 with
    n-gram distance and recommend the latter, so this is optional and
    off by default.
    """
    require_package()
    ref, sim = align_case_counts(reference, simulated, seed)
    return _safe(
        "CFLD", "control-flow",
        lambda: control_flow_log_distance(ref, LOG_IDS, sim, LOG_IDS),
    )


def temporal_metrics(
    reference: pd.DataFrame, simulated: pd.DataFrame,
) -> list[MetricResult]:
    """
    Absolute, circadian and relative event distributions.

    All three use both timestamps of every event. A log carrying only
    completion times makes them measure half of what they should.
    """
    require_package()
    both = AbsoluteTimestampType.BOTH

    return [
        _safe(
            "AED", "temporal",
            lambda: absolute_event_distribution_distance(
                reference, LOG_IDS, simulated, LOG_IDS,
                discretize_type=both, discretize_event=discretize_to_hour,
            ),
        ),
        _safe(
            "CED", "temporal",
            lambda: circadian_event_distribution_distance(
                reference, LOG_IDS, simulated, LOG_IDS,
                discretize_type=both,
            ),
        ),
        _safe(
            "RED", "temporal",
            lambda: relative_event_distribution_distance(
                reference, LOG_IDS, simulated, LOG_IDS,
                discretize_type=both, discretize_event=discretize_to_hour,
            ),
        ),
    ]


def congestion_metrics(
    reference: pd.DataFrame, simulated: pd.DataFrame,
) -> list[MetricResult]:
    """Case arrival rate and cycle time distribution."""
    require_package()
    return [
        _safe(
            "CAR", "congestion",
            lambda: case_arrival_distribution_distance(
                reference, LOG_IDS, simulated, LOG_IDS,
                discretize_event=discretize_to_hour,
            ),
        ),
        _safe(
            "CTD", "congestion",
            lambda: cycle_time_distribution_distance(
                reference, LOG_IDS, simulated, LOG_IDS,
                bin_size=pd.Timedelta(hours=1),
            ),
        ),
    ]


# -------------------------------------------------
# Full evaluation
# -------------------------------------------------

def evaluate_log(
    reference: pd.DataFrame,
    simulated: pd.DataFrame,
    n_gram: int = 2,
    include_cfld: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute the full metric set for one simulated log.

    Returns a frame with one row per measure: name, perspective, value,
    and a note when a measure could not be computed.
    """
    require_package()

    results: list[MetricResult] = []
    results += control_flow_metrics(reference, simulated, n=n_gram)
    if include_cfld:
        results.append(control_flow_log_distance_metric(reference, simulated, seed))
    results += temporal_metrics(reference, simulated)
    results += congestion_metrics(reference, simulated)

    return pd.DataFrame([
        {
            "metric": r.name,
            "perspective": r.perspective,
            "value": round(r.value, 4) if r.value is not None else None,
            "note": r.note,
        }
        for r in results
    ])


def aggregate_runs(run_results: list[pd.DataFrame], confidence: float = 0.95) -> pd.DataFrame:
    """
    Mean and confidence interval across K simulation runs.

    A single run is one draw from a distribution: the simulation is
    stochastic, and with LLM agents it is stochastic even at temperature
    zero. Reporting one number hides that. The convention in this
    literature is K=10 runs with a confidence interval.
    """
    if not run_results:
        return pd.DataFrame()

    combined = pd.concat(
        [df.assign(run=i) for i, df in enumerate(run_results)],
        ignore_index=True,
    )

    rows = []
    for (metric, perspective), grp in combined.groupby(["metric", "perspective"], sort=False):
        values = grp["value"].dropna()
        n = len(values)
        if n == 0:
            rows.append({
                "metric": metric, "perspective": perspective,
                "n_runs": 0, "mean": None, "std": None,
                "ci_low": None, "ci_high": None,
            })
            continue

        mean = values.mean()
        std = values.std(ddof=1) if n > 1 else 0.0
        # Normal approximation; with K=10 a t-interval would be slightly
        # wider, so this is the optimistic reading of the spread.
        z = 1.96 if confidence == 0.95 else 2.576
        half = z * std / math.sqrt(n) if n > 1 else 0.0

        rows.append({
            "metric": metric,
            "perspective": perspective,
            "n_runs": n,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci_low": round(mean - half, 4),
            "ci_high": round(mean + half, 4),
        })

    return pd.DataFrame(rows)


# -------------------------------------------------
# Descriptive comparison (no package needed)
# -------------------------------------------------

def variant_comparison(
    reference: pd.DataFrame, simulated: pd.DataFrame, top_k: int = 10,
) -> pd.DataFrame:
    """
    Most frequent trace variants in each log, side by side.

    Distances say how far apart two logs are; this says where. A variant
    the simulation produces often and the reference never produces is
    visible here and invisible in a single scalar.
    """
    def variants(df, label):
        v = df.groupby(CASE_ID)[ACTIVITY].apply(lambda s: " -> ".join(s))
        counts = v.value_counts()
        return pd.DataFrame({
            "variant": counts.index,
            f"{label}_n": counts.values,
            f"{label}_share": (counts.values / len(v) * 100).round(1),
        })

    ref_v = variants(reference, "ref")
    sim_v = variants(simulated, "sim")

    merged = ref_v.merge(sim_v, on="variant", how="outer").fillna(0)
    merged["total"] = merged["ref_n"] + merged["sim_n"]
    return merged.sort_values("total", ascending=False).head(top_k).drop(columns="total")


def activity_comparison(
    reference: pd.DataFrame, simulated: pd.DataFrame,
) -> pd.DataFrame:
    """Relative frequency of each activity in both logs."""
    def shares(df, label):
        c = df[ACTIVITY].value_counts()
        return pd.DataFrame({
            "activity": c.index,
            f"{label}_n": c.values,
            f"{label}_%": (c.values / len(df) * 100).round(1),
        })

    merged = shares(reference, "ref").merge(
        shares(simulated, "sim"), on="activity", how="outer",
    ).fillna(0)
    merged["diff_pp"] = (merged["sim_%"] - merged["ref_%"]).round(1)
    return merged.sort_values("ref_n", ascending=False)


def trace_length_comparison(
    reference: pd.DataFrame, simulated: pd.DataFrame,
) -> pd.DataFrame:
    """Trace length distribution in both logs."""
    def stats(df, label):
        lengths = df.groupby(CASE_ID).size()
        return {
            "log": label,
            "cases": len(lengths),
            "mean": round(lengths.mean(), 1),
            "median": int(lengths.median()),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
        }

    return pd.DataFrame([stats(reference, "reference"), stats(simulated, "simulated")])
