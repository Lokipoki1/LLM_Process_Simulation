"""
duration_reference.py
---------------------
Reference durations per activity, and the complexity scale agents use
to place a case against them.

  the AGENT   rates how complex this case was, on a 1-5 scale
  the ENGINE  turns that rating into minutes: anchor x multiplier

LLMs estimate absolute durations poorly (far too short unanchored,
copied verbatim when anchored), but judging complexity is a reading
task. Agents also give a free estimate of the typical duration, which
is recorded for analysis but never advances the clock.
"""

from __future__ import annotations

# tool_name -> typical hands-on minutes for an ordinary case.
# Medians of the log-normal parameters in clock/simulation_clock.py.
# build_anchors_from_distributions() regenerates this table from fitted
# distributions keyed by tool name.
DURATION_ANCHOR: dict[str, int] = {
    # Junior Clerk
    "IntakeApplication":      30,
    "CheckDocuments":         40,
    "ForwardCase":             8,
    "ReturnApplicationEarly":  7,

    # Senior Clerk
    "CheckCreditScore":       50,
    "ValidateApplication":    80,
    "RequestAdditionalInfo": 165,
    "EscalateCase":           11,

    # Credit Officer
    "AssessRisk":            135,
    "ApproveLoan":            22,
    "RejectLoan":             18,
}


# Complexity rating -> multiplier applied to the anchor.
# 3 is the ordinary case; the extremes span roughly a threefold range,
# which is the spread the fitted distributions show between their 10th
# and 90th percentiles.
COMPLEXITY_MULTIPLIER: dict[int, float] = {
    1: 0.7,   # unusually clean, nothing to check twice
    2: 1.0,   # straightforward
    3: 1.3,   # ordinary
    4: 1.8,   # awkward: something did not line up
    5: 2.5,   # difficult: conflicting signals, had to work for it
}

COMPLEXITY_SCALE_TEXT = """\
    1  unusually clean - everything lined up, nothing to check twice
    2  straightforward - ordinary file, no surprises
    3  ordinary - the usual amount of work for this task
    4  awkward - something did not line up and cost you extra effort
    5  difficult - conflicting signals, or you had to dig for an answer"""


def resolve_minutes(tool_name: str, complexity: int) -> float | None:
    """
    Convert a complexity rating into hands-on minutes for a tool.
    Returns None when the tool has no anchor, so the caller can fall back.
    """
    anchor = DURATION_ANCHOR.get(tool_name)
    if anchor is None:
        return None
    multiplier = COMPLEXITY_MULTIPLIER.get(int(complexity), 1.0)
    return anchor * multiplier


def complexity_prompt_block() -> str:
    """
    The duration section of an agent's system prompt. Deliberately
    contains no reference durations, so the free estimate stays unaided.
    """
    return f"""
=== TIME AND COMPLEXITY ===

Every tool asks you two things about the work you just did.

typical_duration_minutes
    From your experience, how long does this kind of task take on an
    ORDINARY case? Not this case in particular - the usual one. Count
    only hands-on working minutes: time with the file actually open.
    Queueing, overnight gaps and weekends are handled elsewhere.

complexity_rationale, then case_complexity
    First write, in one sentence, what made THIS case easy or hard.
    Then rate it:

{COMPLEXITY_SCALE_TEXT}

    Rate the file in front of you, not the task in general. Most cases
    are a 3. Reserve 1 and 5 for files that genuinely stand out, and be
    willing to use them when they do - a rating that never moves off 3
    is not a judgement.
"""


def build_anchors_from_distributions(
    distributions: dict[str, tuple[float, float]],
) -> dict[str, int]:
    """Rebuild the anchor table from {tool_name: (mu, sigma)}: exp(mu) / 60."""
    import math
    return {
        name: max(1, round(math.exp(mu) / 60))
        for name, (mu, _sigma) in distributions.items()
        if not name.startswith("_")
    }
