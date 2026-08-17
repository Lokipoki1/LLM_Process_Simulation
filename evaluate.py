"""
evaluate.py
-----------
Compare a simulated event log against the reference log.

    python evaluate.py output/simulation.xes data/bpic/BPI_Challenge_2017.xes

    # several runs of the same configuration, reported with a CI
    python evaluate.py "out_run*/simulation.xes" data/bpic/BPI_Challenge_2017.xes

    # compare two configurations against the same reference
    python evaluate.py out_llm/simulation.xes data/bpic/BPI_Challenge_2017.xes \\
        --compare out_rules/simulation.xes --name-a llm --name-b rules

What it does
    1. loads both logs
    2. restricts the reference log to the activities the simulation
       produces, and reports the before/after counts
    3. takes the test portion of the reference log by case arrival
    4. computes the control-flow, temporal and congestion distances
    5. prints descriptive comparisons that show WHERE the logs differ

Reading the output
    N-gram distance is in [0, 1] and comparable across studies. The
    distribution measures are in hours per observation and are only
    comparable within this setup. The descriptive tables at the end are
    usually more informative than any single scalar.
"""

from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

from src.evaluation.log_utils import (
    load_xes, describe, prepare_reference_log, temporal_split,
    reference_window, shift_to_match, vocabulary_overlap,
    ACTIVITY, CASE_ID,
)
from src.evaluation.metrics import (
    MEASURES_AVAILABLE, evaluate_log, aggregate_runs,
    variant_comparison, activity_comparison, trace_length_comparison,
)


def heading(text: str) -> None:
    print(f"\n{text}")
    print("-" * len(text))


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare a simulated event log against a reference log",
    )
    p.add_argument("simulated", help="Simulated XES file, or a glob for several runs")
    p.add_argument("reference", help="Reference XES file (e.g. BPIC 2017)")
    p.add_argument("--compare", default=None,
                   help="A second simulated log, evaluated against the same reference")
    p.add_argument("--name-a", default="simulated", help="Label for the first log")
    p.add_argument("--name-b", default="second", help="Label for the second log")
    p.add_argument("--n-gram", type=int, default=2, help="n for the n-gram distance")
    p.add_argument("--cfld", action="store_true",
                   help="Also compute control-flow log distance (slow)")
    p.add_argument("--no-split", action="store_true",
                   help="Use the whole reference log instead of its test portion")
    p.add_argument("--train-fraction", type=float, default=0.8,
                   help="Share of the reference log used for calibration")
    p.add_argument("--raw-lifecycle", action="store_true",
                   help="Do not fold schedule/start/complete into single activity instances")
    p.add_argument("--no-align", action="store_true",
                   help="Do not shift the simulated log onto the reference window")
    p.add_argument("--csv", default=None, help="Write the metric table here")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not MEASURES_AVAILABLE:
        print("log-distance-measures is not installed.\n"
              "  pip install log-distance-measures", file=sys.stderr)
        sys.exit(1)

    sim_paths = sorted(glob.glob(args.simulated)) or [args.simulated]
    for path in sim_paths + [args.reference]:
        if not Path(path).exists():
            print(f"Not found: {path}", file=sys.stderr)
            sys.exit(1)

    # -- simulated logs ------------------------
    print(f"\n{'=' * 70}")
    print(f"  Evaluating {len(sim_paths)} simulated log(s) against {Path(args.reference).name}")
    print(f"{'=' * 70}")

    sim_logs = [load_xes(p) for p in sim_paths]
    vocabulary = set().union(*(set(df[ACTIVITY].unique()) for df in sim_logs))

    heading("Simulated vocabulary")
    print(f"  {len(vocabulary)} activities: {', '.join(sorted(vocabulary))}")

    # -- reference log -------------------------
    reference, before, after = prepare_reference_log(
        args.reference, vocabulary, collapse_lifecycle=not args.raw_lifecycle,
    )

    heading("Reference log preprocessing")
    print(pd.DataFrame([before.as_row(), after.as_row()]).to_string(index=False))

    # -- vocabulary check ----------------------
    overlap = vocabulary_overlap(vocabulary, before.activities)

    if not overlap["matched"]:
        heading("No shared activities")
        print("  The simulated log and the reference log have no activity label")
        print("  in common, so filtering the reference log leaves nothing to")
        print("  compare against.\n")
        print(f"  Simulated ({len(vocabulary)}):")
        for a in sorted(vocabulary):
            print(f"    {a}")
        print(f"\n  Reference ({len(before.activities)}):")
        for a in before.activities:
            print(f"    {a}")
        print("\n  The activity map in src/process/loan_application.py has to")
        print("  translate each tool to a label the reference log actually uses.")
        print("  If the map is already correct, this simulated log predates the")
        print("  change - re-run the simulation before evaluating.\n")
        sys.exit(1)

    if overlap["simulated_only"]:
        heading("Simulated activities absent from the reference log")
        for a in overlap["simulated_only"]:
            print(f"  {a}")
        print("\n  These can never be matched: they add distance to every")
        print("  label-based measure without a counterpart to compare to.")

    if not args.no_split:
        try:
            _, reference = temporal_split(reference, args.train_fraction)
        except ValueError as exc:
            heading("Could not split the reference log")
            print(f"  {exc}")
            sys.exit(1)
        print(f"\n  Test portion (last {(1 - args.train_fraction) * 100:.0f}% by arrival): "
              f"{reference[CASE_ID].nunique()} cases, {len(reference)} events")

    # -- metrics -------------------------------
    # -- comparison window ---------------------
    n_sim_cases = max(df[CASE_ID].nunique() for df in sim_logs)
    if n_sim_cases < 100:
        heading("Sample size")
        print(f"  The simulated log holds {n_sim_cases} cases against "
              f"{reference[CASE_ID].nunique()} in the reference.")
        print("  Distribution distances over a few dozen cases are dominated by")
        print("  sampling noise. Treat these numbers as a pipeline check, not")
        print("  as a result - a few hundred cases is the point where they")
        print("  start to settle.")

    heading("Distance measures")
    run_tables = []
    for path, sim in zip(sim_paths, sim_logs):
        # A contiguous block of arrivals, so the reference covers a span
        # comparable to the simulation's rather than a year-wide scatter.
        ref_block = reference_window(reference, sim[CASE_ID].nunique())
        sim_aligned = sim if args.no_align else shift_to_match(sim, ref_block)

        table = evaluate_log(
            ref_block, sim_aligned,
            n_gram=args.n_gram, include_cfld=args.cfld,
        )
        run_tables.append(table)
        if len(sim_paths) > 1:
            print(f"\n  {Path(path).parent.name}")
            print(table.to_string(index=False))

    if len(sim_paths) == 1:
        print(run_tables[0].to_string(index=False))
        final = run_tables[0]
    else:
        heading(f"Across {len(sim_paths)} runs")
        final = aggregate_runs(run_tables)
        print(final.to_string(index=False))

    failed = [r for t in run_tables for _, r in t.iterrows() if r["note"]]
    if failed:
        heading("Measures that could not be computed")
        for r in failed:
            print(f"  {r['metric']}: {r['note']}")

    # -- descriptive ---------------------------
    primary = sim_logs[0]
    reference = reference_window(reference, primary[CASE_ID].nunique())

    if not args.no_align:
        print(f"\n  Simulated logs shifted onto the reference window "
              f"(first arrival {reference[CASE_ID].nunique()} cases).")

    heading("Trace length")
    print(trace_length_comparison(reference, primary).to_string(index=False))

    heading("Activity frequency")
    print(activity_comparison(reference, primary).to_string(index=False))

    heading("Most frequent variants")
    variants = variant_comparison(reference, primary, top_k=8)
    for _, row in variants.iterrows():
        v = row["variant"]
        shown = v if len(v) <= 96 else v[:93] + "..."
        print(f"  ref {row['ref_share']:>5.1f}%   sim {row['sim_share']:>5.1f}%   {shown}")

    # -- optional second configuration ---------
    if args.compare:
        if not Path(args.compare).exists():
            print(f"\nNot found: {args.compare}", file=sys.stderr)
        else:
            heading(f"Second configuration: {args.name_b}")
            second = load_xes(args.compare)
            ref_b = reference_window(reference, second[CASE_ID].nunique())
            sim_b = second if args.no_align else shift_to_match(second, ref_b)
            table_b = evaluate_log(ref_b, sim_b, n_gram=args.n_gram, include_cfld=args.cfld)

            merged = final[["metric", "perspective"]].copy()
            col_a = "mean" if "mean" in final.columns else "value"
            merged[args.name_a] = final[col_a].values
            merged[args.name_b] = table_b["value"].values
            merged["better"] = [
                args.name_a if (a is not None and b is not None and a < b)
                else args.name_b if (a is not None and b is not None)
                else "-"
                for a, b in zip(merged[args.name_a], merged[args.name_b])
            ]
            print(merged.to_string(index=False))
            print("\n  Lower is better for every measure.")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(args.csv, index=False)
        print(f"\n  Metrics -> {args.csv}")

    print()


if __name__ == "__main__":
    main()
