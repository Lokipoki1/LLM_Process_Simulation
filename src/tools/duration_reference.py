"""
duration_reference.py
---------------------
Reference durations per activity, and the complexity scale that agents
use to place a specific case against them.

Why this split exists
---------------------
Two earlier configurations both failed, in opposite directions:

  Uncalibrated absolute estimates
      gpt-4o-mini answered 5 minutes for a credit bureau query and 10
      for a rework request: 10x to 16x below the fitted distributions.
      With every step taking 5-30 minutes the workers never saturate,
      no queue forms, nothing crosses a shift boundary, and every case
      closes the same day. The queueing dynamics the engine exists to
      model disappear.

  Anchored absolute estimates
      Given a reference band, the model copied the typical figure
      verbatim. Eight of ten tools came back with zero variance and a
      ratio of exactly 1.00 against their anchor. The scale was right
      and the agent contributed nothing beyond the anchor.

The reading of those two results is that the model is poor at absolute
temporal magnitude, but is being asked the wrong question. Judging
whether a file is straightforward or messy is a reading task, which is
what it is good at. So the question is split:

  the AGENT   judges how complex this case was, on a 1-5 scale
  the ENGINE  turns that judgement into minutes, against an anchor
              derived from the reference log

Variance now comes from the agent's reading of each case rather than
from its arithmetic, and the scale is guaranteed by construction.

Agents are ALSO asked for a free, unanchored estimate of how long the
task usually takes. That number never advances the clock - it is
recorded so the two abilities can be reported separately: calibration in
absolute magnitude, and discrimination of relative complexity.
"""

from __future__ import annotations

# tool_name -> typical hands-on minutes for an ordinary case.
# Medians of the log-normal parameters in clock/simulation_clock.py.
# Regenerate from the reference log with build_anchors_from_distributions()
# once the activity-label mapping is settled.
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
    The duration section of an agent's system prompt.

    Deliberately contains NO reference durations. The agent is asked for
    a free estimate of the typical time - recorded but never used to
    advance the clock - and for a complexity judgement, which is.
    Withholding the anchor is what keeps the free estimate usable as
    evidence about the model's unaided calibration.
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
    """
    Rebuild the anchor table from fitted log-normal parameters.
    For parameters (mu, sigma) in seconds, the anchor is exp(mu) / 60.

    Args:
        distributions: {tool_name: (mu, sigma)}, as produced by
                       extract_bpic_distributions()
    """
    import math
    return {
        name: max(1, round(math.exp(mu) / 60))
        for name, (mu, _sigma) in distributions.items()
        if not name.startswith("_")
    }
