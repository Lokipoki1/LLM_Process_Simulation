"""
simulation_clock.py
--------------------
Logical simulation clock.

Single responsibility: given an activity name, return a duration in
seconds sampled from a log-normal distribution estimated from the BPIC.

This is CRITICAL for AED and CTD to be valid:
the LLM decides WHAT activity to execute, the clock decides HOW LONG it takes.

NOTE: this class only SAMPLES durations. The DES engine owns the
simulation timeline — it is the only component that decides when an
activity starts and finishes. `advance()` / `advance_for_activity()`
are kept for the LangGraph observer path only.
"""

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────
# Log-normal parameters per activity, keyed by TOOL CLASS NAME.
# Format: { tool_name: (mu, sigma) } of the underlying normal
# distribution. Replace with extract_bpic_distributions() output
# once the BPIC calibration phase is done.
# ─────────────────────────────────────────────────────────────────

DEFAULT_DISTRIBUTIONS: dict[str, tuple[float, float]] = {
    # Junior Clerk
    "IntakeApplication":      (7.5, 0.8),   # ~1800s ≈ 30 min
    "CheckDocuments":         (7.8, 0.7),   # ~2400s ≈ 40 min
    "ForwardCase":            (6.2, 0.5),   # ~490s  ≈ 8 min
    "ReturnApplicationEarly": (6.0, 0.4),   # ~400s  ≈ 7 min

    # Senior Clerk
    "CheckCreditScore":       (8.0, 0.7),   # ~3000s ≈ 50 min (bureau query + wait)
    "ValidateApplication":    (8.5, 0.9),   # ~4900s ≈ 80 min
    "RequestAdditionalInfo":  (9.2, 1.0),   # ~9900s ≈ 165 min (waiting for a reply)
    "EscalateCase":           (6.5, 0.5),   # ~665s  ≈ 11 min

    # Credit Officer
    "AssessRisk":             (9.0, 0.8),   # ~8100s ≈ 135 min
    "ApproveLoan":            (7.2, 0.6),   # ~1300s ≈ 22 min
    "RejectLoan":             (7.0, 0.6),   # ~1100s ≈ 18 min

    # Fallback for unrecognised activities
    "_default":               (7.5, 1.0),
}


class SimulationClock:
    """
    Samples activity durations and (optionally) tracks a logical time cursor.

    In the DES engine only `sample_duration()` is used — the engine owns
    the timeline. `advance()` and `advance_for_activity()` exist for the
    LangGraph observer path (documentation artifact).

    Usage:
        clock = SimulationClock(start_timestamp="2017-01-02T08:00:00", seed=42)
        seconds = clock.sample_duration("ValidateApplication")
    """

    def __init__(
        self,
        start_timestamp: str = "2017-01-02T08:00:00",
        distributions: dict[str, tuple[float, float]] | None = None,
        seed: int | None = None,
    ):
        from datetime import datetime
        self._current: float = datetime.fromisoformat(start_timestamp).timestamp()
        self._distributions = distributions or DEFAULT_DISTRIBUTIONS
        self._rng = np.random.default_rng(seed)

    # ── Current time ─────────────────────────

    @property
    def current(self) -> float:
        """Current logical time as a Unix timestamp (float)."""
        return self._current

    def current_iso(self) -> str:
        """Current logical time as an ISO 8601 string."""
        from datetime import datetime, timezone
        return datetime.fromtimestamp(self._current, tz=timezone.utc).isoformat()

    # ── Sampling ─────────────────────────────

    def sample_duration(self, activity_name: str) -> float:
        """
        Sample a duration in seconds for the given activity.
        Falls back to '_default' if the activity is not configured.
        """
        mu, sigma = self._distributions.get(
            activity_name,
            self._distributions["_default"],
        )
        duration = float(self._rng.lognormal(mean=mu, sigma=sigma))
        # Cap at 7 days to avoid extreme outliers
        return min(duration, 7 * 24 * 3600)

    # ── Time cursor (LangGraph observer path only) ──

    def advance(self, seconds: float) -> float:
        """Advance the cursor by `seconds`. Returns the new timestamp."""
        self._current += seconds
        return self._current

    def advance_for_activity(self, activity_name: str) -> tuple[float, float]:
        """
        Sample a duration and advance the cursor.
        Returns (duration_seconds, new_timestamp).

        NOT used by the DES engine — the engine samples durations directly
        and owns the timeline.
        """
        duration = self.sample_duration(activity_name)
        new_ts = self.advance(duration)
        return duration, new_ts


# ─────────────────────────────────────────────────────────────────
# Extract real distributions from a BPIC event log.
# Run once during the dataset analysis phase, then pass the result
# to SimulationClock(distributions=...).
# ─────────────────────────────────────────────────────────────────

def extract_bpic_distributions(xes_path: str) -> dict[str, tuple[float, float]]:
    """
    Read a BPIC XES log and estimate log-normal parameters per activity
    from the observed durations between consecutive events.

    NOTE: the returned keys are BPIC ACTIVITY names (e.g. "A_Submitted"),
    while SimulationClock is keyed by TOOL names (e.g. "IntakeApplication").
    A mapping between the two vocabularies is required before these
    distributions can be plugged into the clock.

    Args:
        xes_path: path to the BPIC .xes file

    Returns:
        dict {activity_name: (mu, sigma)}
    """
    import pm4py

    log = pm4py.read_xes(xes_path)
    df = pm4py.convert_to_dataframe(log)

    # Duration of each event = gap to the next event in the same case
    df = df.sort_values(["case:concept:name", "time:timestamp"])
    df["duration"] = (
        df.groupby("case:concept:name")["time:timestamp"]
        .diff()
        .shift(-1)
        .dt.total_seconds()
    )
    df = df.dropna(subset=["duration"])
    df = df[df["duration"] > 0]

    distributions: dict[str, tuple[float, float]] = {}
    for activity, group in df.groupby("concept:name"):
        durations = group["duration"].values
        if len(durations) < 5:
            continue
        log_durations = np.log(durations)
        distributions[str(activity)] = (
            float(np.mean(log_durations)),
            float(np.std(log_durations)),
        )

    distributions["_default"] = DEFAULT_DISTRIBUTIONS["_default"]
    return distributions
