"""
compare_logs.py
---------------
Compare two event logs produced by the same engine over the same cases.

Intended use is the LLM-versus-rules comparison: run the simulation twice
with the same seed and the same cases, changing only --agents, then point
this script at the two XES files. Because everything else is held fixed,
every difference it reports is attributable to how the next action was
chosen.

    python run_des.py --cases 10 --bpic <log> --agents llm   --output out_llm
    python run_des.py --cases 10 --bpic <log> --agents rules --output out_rules
    python compare_logs.py out_llm/simulation.xes out_rules/simulation.xes

What it reports
    trace variants   the distinct activity sequences each log contains,
                     and how many cases each arm routed identically
    outcomes         approval and rejection counts
    per case         the cases where the two arms disagreed, and how
    activities       frequency of each activity in each arm
    directly-follows the transition pairs that differ between arms
    timing           cycle time distribution per arm

Reading the result
    Near-identical variants and outcomes mean the LLM's reasoning is not
    changing the process for these cases: a decision table reproduces it.
    That is a legitimate finding and worth reporting as one.
    Divergence is evidence that the cognition does something a fixed
    policy does not, and the per-case section shows exactly where.
"""

from __future__ import annotations
import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

NS = {"x": "http://www.xes-standard.org/"}


# -------------------------------------------------
# Loading
# -------------------------------------------------

def load_log(path: str) -> dict[str, list[tuple[str, datetime, str]]]:
    """Read an XES file into {case_id: [(activity, timestamp, resource), ...]}."""
    root = ET.parse(path).getroot()
    cases: dict[str, list[tuple[str, datetime, str]]] = {}

    for trace in root.findall(".//x:trace", NS):
        name_el = trace.find('x:string[@key="concept:name"]', NS)
        if name_el is None:
            continue
        case_id = name_el.get("value")
        events = []
        for ev in trace.findall("x:event", NS):
            act = ev.find('x:string[@key="concept:name"]', NS)
            ts = ev.find('x:date[@key="time:timestamp"]', NS)
            res = ev.find('x:string[@key="org:resource"]', NS)
            if act is None or ts is None:
                continue
            events.append((
                act.get("value"),
                datetime.fromisoformat(ts.get("value").replace("Z", "+00:00")),
                res.get("value") if res is not None else "",
            ))
        events.sort(key=lambda e: e[1])
        cases[case_id] = events

    return cases


def variant(events) -> tuple[str, ...]:
    return tuple(a for a, _, _ in events)


def outcome(events) -> str:
    """Final activity, used as the case outcome."""
    return events[-1][0] if events else "EMPTY"


def cycle_hours(events) -> float:
    if len(events) < 2:
        return 0.0
    return (events[-1][1] - events[0][1]).total_seconds() / 3600


def directly_follows(events) -> set[tuple[str, str]]:
    acts = [a for a, _, _ in events]
    return set(zip(acts, acts[1:]))


# -------------------------------------------------
# Reporting
# -------------------------------------------------

def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def report(log_a, log_b, name_a: str, name_b: str) -> None:
    shared = sorted(set(log_a) & set(log_b))
    only_a = sorted(set(log_a) - set(log_b))
    only_b = sorted(set(log_b) - set(log_a))

    print(f"\n{'=' * 68}")
    print(f"  {name_a}  vs  {name_b}")
    print(f"{'=' * 68}")
    print(f"  Cases: {len(log_a)} and {len(log_b)}, {len(shared)} in common")
    if only_a:
        print(f"  Only in {name_a}: {len(only_a)}")
    if only_b:
        print(f"  Only in {name_b}: {len(only_b)}")

    if not shared:
        print("\n  No cases in common - nothing to compare.")
        return

    # -- trace variants ------------------------
    section("Trace variants")
    va = Counter(variant(log_a[c]) for c in shared)
    vb = Counter(variant(log_b[c]) for c in shared)
    print(f"  {name_a}: {len(va)} distinct variants")
    print(f"  {name_b}: {len(vb)} distinct variants")
    print(f"  Shared variants: {len(set(va) & set(vb))}")

    identical = [c for c in shared if variant(log_a[c]) == variant(log_b[c])]
    print(f"\n  Cases with an IDENTICAL sequence: {len(identical)}/{len(shared)}"
          f"  ({len(identical) / len(shared) * 100:.0f}%)")

    # -- outcomes ------------------------------
    section("Outcomes")
    oa = Counter(outcome(log_a[c]) for c in shared)
    ob = Counter(outcome(log_b[c]) for c in shared)
    for key in sorted(set(oa) | set(ob)):
        print(f"  {key:<18} {name_a}: {oa.get(key, 0):>3}   {name_b}: {ob.get(key, 0):>3}")

    same_outcome = [c for c in shared if outcome(log_a[c]) == outcome(log_b[c])]
    print(f"\n  Cases with the SAME outcome: {len(same_outcome)}/{len(shared)}"
          f"  ({len(same_outcome) / len(shared) * 100:.0f}%)")

    # -- disagreements -------------------------
    differing = [c for c in shared if variant(log_a[c]) != variant(log_b[c])]
    if differing:
        section(f"Where they diverged ({len(differing)} cases)")
        for c in differing:
            print(f"\n  {c}")
            print(f"    {name_a:<8} {' -> '.join(variant(log_a[c]))}")
            print(f"    {name_b:<8} {' -> '.join(variant(log_b[c]))}")
    else:
        section("Where they diverged")
        print("  Nowhere: every shared case produced the same activity sequence.")

    # -- activity frequency --------------------
    section("Activity frequency")
    fa = Counter(a for c in shared for a, _, _ in log_a[c])
    fb = Counter(a for c in shared for a, _, _ in log_b[c])
    print(f"  {'activity':<20}{name_a:>10}{name_b:>10}{'diff':>8}")
    for act in sorted(set(fa) | set(fb)):
        x, y = fa.get(act, 0), fb.get(act, 0)
        flag = "" if x == y else "  <-"
        print(f"  {act:<20}{x:>10}{y:>10}{y - x:>+8}{flag}")

    # -- directly-follows ----------------------
    section("Directly-follows relations")
    dfa = set().union(*(directly_follows(log_a[c]) for c in shared)) if shared else set()
    dfb = set().union(*(directly_follows(log_b[c]) for c in shared)) if shared else set()
    print(f"  {name_a}: {len(dfa)} distinct pairs")
    print(f"  {name_b}: {len(dfb)} distinct pairs")
    print(f"  Shared: {len(dfa & dfb)}")
    for label, pairs in ((name_a, dfa - dfb), (name_b, dfb - dfa)):
        if pairs:
            print(f"\n  Only in {label}:")
            for a, b in sorted(pairs):
                print(f"    {a} -> {b}")

    # -- timing --------------------------------
    section("Cycle time (hours)")
    ca = [cycle_hours(log_a[c]) for c in shared]
    cb = [cycle_hours(log_b[c]) for c in shared]

    def stats(v):
        v = sorted(v)
        n = len(v)
        mean = sum(v) / n
        median = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
        return mean, median, v[0], v[-1]

    ma, mda, mina, maxa = stats(ca)
    mb, mdb, minb, maxb = stats(cb)
    print(f"  {'':<10}{'mean':>9}{'median':>9}{'min':>9}{'max':>9}")
    print(f"  {name_a:<10}{ma:>9.1f}{mda:>9.1f}{mina:>9.1f}{maxa:>9.1f}")
    print(f"  {name_b:<10}{mb:>9.1f}{mdb:>9.1f}{minb:>9.1f}{maxb:>9.1f}")

    # -- verdict -------------------------------
    section("Reading")
    pct_same = len(identical) / len(shared) * 100
    if pct_same == 100:
        print("  Every shared case produced an identical sequence. For these")
        print("  cases the LLM's reasoning is not changing the process: a")
        print("  decision table reproduces it exactly. Worth reporting as a")
        print("  finding, and worth checking on a larger and more varied set")
        print("  before generalising.")
    elif pct_same >= 80:
        print(f"  {pct_same:.0f}% of cases matched. The LLM follows the same policy")
        print("  on routine files and diverges on the rest. The divergent")
        print("  cases listed above are the evidence of what it contributes.")
    else:
        print(f"  Only {pct_same:.0f}% of cases matched. The two arms behave")
        print("  substantially differently. The next question is which of the")
        print("  two is closer to the reference log.")
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare two event logs produced over the same cases",
    )
    p.add_argument("log_a", help="First XES file (e.g. the LLM arm)")
    p.add_argument("log_b", help="Second XES file (e.g. the rules arm)")
    p.add_argument("--name-a", default=None, help="Label for the first log")
    p.add_argument("--name-b", default=None, help="Label for the second log")
    args = p.parse_args()

    for path in (args.log_a, args.log_b):
        if not Path(path).exists():
            print(f"Not found: {path}", file=sys.stderr)
            sys.exit(1)

    name_a = args.name_a or Path(args.log_a).parent.name or "log A"
    name_b = args.name_b or Path(args.log_b).parent.name or "log B"

    report(load_log(args.log_a), load_log(args.log_b), name_a, name_b)


if __name__ == "__main__":
    main()
