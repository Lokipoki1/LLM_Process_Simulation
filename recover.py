"""
recover.py
----------
Turn a checkpoint from an interrupted run into a usable event log.

A run that was killed halfway still produced complete traces for every
case that finished. Those cases are not damaged: the engine writes a
checkpoint after every N completions, so what is on disk is a valid
event log of the work already paid for.

    python recover.py out_llm/checkpoint.json
    python recover.py out_llm/checkpoint.json --output out_llm_partial

The result is an XES file that evaluate.py reads like any other.

What it does not do
    It cannot resume the simulation. Recovering the in-flight cases
    would mean restoring agent state, queue positions and the event
    queue, which the checkpoint deliberately does not carry - a partial
    trace is worse than no trace, because it looks like a case that
    ended early. Cases still in progress when the run died are dropped.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def load_checkpoint(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def to_dataframe(events: list[dict]) -> pd.DataFrame:
    """Rename the stored fields to the XES column names."""
    df = pd.DataFrame(events).rename(columns={
        "case_concept_name":    "case:concept:name",
        "concept_name":         "concept:name",
        "start_timestamp":      "start_timestamp",
        "time_timestamp":       "time:timestamp",
        "org_resource":         "org:resource",
        "lifecycle_transition": "lifecycle:transition",
    })
    for col in ("start_timestamp", "time:timestamp"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="ISO8601", utc=True)
    return df.sort_values(["case:concept:name", "time:timestamp"])


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rebuild an event log from an interrupted run's checkpoint",
    )
    p.add_argument("checkpoint", help="Path to checkpoint.json")
    p.add_argument("--output", default=None,
                   help="Directory for the recovered files (default: alongside the checkpoint)")
    p.add_argument("--name", default="recovered",
                   help="Base name for the output files")
    args = p.parse_args()

    cp_path = Path(args.checkpoint)
    if not cp_path.exists():
        print(f"Not found: {cp_path}", file=sys.stderr)
        sys.exit(1)

    data = load_checkpoint(cp_path)
    events = data.get("events", [])

    if not events:
        print("The checkpoint holds no events.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output) if args.output else cp_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = to_dataframe(events)
    n_cases = df["case:concept:name"].nunique()

    print(f"\n  Checkpoint: {cp_path}")
    print(f"  Completed when written: {data.get('completed_cases', '?')} "
          f"of {data.get('total_cases', '?')} cases")
    print(f"  Recovered: {n_cases} cases, {len(df)} events")

    failed = data.get("failed_cases") or []
    if failed:
        print(f"  Cases that failed to run: {len(failed)}")
        print("    Excluded already - they carry no events.")

    span = df["time:timestamp"].max() - df["time:timestamp"].min()
    print(f"  Span: {span.days} days")

    # -- export --------------------------------
    json_path = out_dir / f"{args.name}.json"
    with open(json_path, "w") as f:
        json.dump(events, f, indent=2, default=str)
    print(f"\n  JSON -> {json_path}")

    try:
        import pm4py
        xes_path = out_dir / f"{args.name}.xes"
        pm4py.write_xes(pm4py.convert_to_event_log(df), str(xes_path))
        print(f"  XES  -> {xes_path}")
        print(f"\n  python evaluate.py {xes_path} data/bpic/BPI_Challenge_2017.xes\n")
    except ImportError:
        print("\n  pm4py not available - only the JSON was written.\n")


if __name__ == "__main__":
    main()
