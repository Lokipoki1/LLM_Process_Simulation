"""
duration_reference.py
---------------------
Reference durations per activity, injected into each agent's system
prompt so their estimates land on a realistic scale.

Why this exists
    Uncalibrated, gpt-4o-mini estimates 5 minutes for a credit bureau
    query and 10 minutes for a rework request. Measured against the
    BPIC-fitted distributions that is an underestimate of 10x to 16x,
    and the consequence is not just wrong timestamps: with every step
    taking 5-30 minutes the workers never saturate, no queue forms, no
    activity crosses the end of a shift, and every case closes the same
    day. The queueing dynamics the engine exists to model disappear.

    Giving the agent a reference range restores the scale while leaving
    the judgement with the agent: it still decides whether THIS case sits
    below, inside, or above the typical band.

Calibration source
    The anchors below are the medians and interquartile-style bands of
    the log-normal parameters in clock/simulation_clock.py. Once the
    activity-label mapping to BPIC 2017 is settled, regenerate them from
    the real log with extract_bpic_distributions() and replace this table
    (see build_reference_from_distributions).
"""

from __future__ import annotations

# tool_name -> (typical_minutes, low_minutes, high_minutes)
DURATION_REFERENCE: dict[str, tuple[int, int, int]] = {
    # Junior Clerk
    "IntakeApplication":      (30, 15, 60),
    "CheckDocuments":         (40, 20, 80),
    "ForwardCase":            (8, 5, 15),
    "ReturnApplicationEarly": (7, 5, 15),

    # Senior Clerk
    "CheckCreditScore":       (50, 25, 100),
    "ValidateApplication":    (80, 40, 160),
    "RequestAdditionalInfo":  (165, 60, 300),
    "EscalateCase":           (11, 6, 20),

    # Credit Officer
    "AssessRisk":             (135, 60, 260),
    "ApproveLoan":            (22, 12, 45),
    "RejectLoan":             (18, 10, 35),
}


def format_reference(tool_names: list[str]) -> str:
    """
    Render the reference band for a set of tools as a prompt fragment.

    Only the tools the agent actually owns are included, so each persona
    sees its own workload and nothing else.
    """
    lines = []
    for name in tool_names:
        ref = DURATION_REFERENCE.get(name)
        if not ref:
            continue
        typical, low, high = ref
        lines.append(f"    {name}: typically {typical} min (usual range {low}-{high})")
    return "\n".join(lines)


def duration_prompt_block(tool_names: list[str]) -> str:
    """
    The full duration-estimation section for an agent's system prompt.

    The reference is framed as departmental experience rather than as an
    instruction to copy: the agent is told what these tasks usually take
    and asked to place THIS case against that baseline.
    """
    return f"""
=== ESTIMATING DURATION ===

Every tool asks for duration_minutes: the hands-on working minutes this
activity took you for THIS case. Count only time with the file actually
open in front of you. Queueing, overnight gaps and weekends are added by
the system separately, so never include them.

From your years in this department, these are the times these tasks
usually take:

{format_reference(tool_names)}

Those are baselines, not targets. Place this case against them:
  - a clean, small, routine file lands below the typical figure
  - an unusual amount, a vague loan goal, contradictory numbers, or a
    file you had to re-open after rework lands above it
  - a genuinely difficult case can exceed the usual range

Report the number this case actually cost you. Two different cases should
rarely take exactly the same time.
"""


def build_reference_from_distributions(
    distributions: dict[str, tuple[float, float]],
) -> dict[str, tuple[int, int, int]]:
    """
    Rebuild the reference table from fitted log-normal parameters.

    For a log-normal with parameters (mu, sigma):
        median = exp(mu)
        low    = exp(mu - 0.675 * sigma)   ~ 25th percentile
        high   = exp(mu + 0.675 * sigma)   ~ 75th percentile

    Args:
        distributions: {tool_name: (mu, sigma)} in seconds, as produced
                       by extract_bpic_distributions()

    Returns:
        {tool_name: (typical_min, low_min, high_min)}
    """
    import math

    out: dict[str, tuple[int, int, int]] = {}
    for name, (mu, sigma) in distributions.items():
        if name.startswith("_"):
            continue
        median = math.exp(mu) / 60
        low = math.exp(mu - 0.675 * sigma) / 60
        high = math.exp(mu + 0.675 * sigma) / 60
        out[name] = (max(1, round(median)), max(1, round(low)), max(1, round(high)))
    return out
