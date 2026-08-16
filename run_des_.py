"""
run_des.py
----------
Run the Discrete Event Simulation engine on the loan application process.

The engine itself is process-agnostic. This script picks a process
definition, builds a run configuration for it, and reports the results.
Simulating a different process means importing a different definition
here - the engine is not touched.

Usage:
    # Synthetic cases
    python run_des.py --cases 20

    # Real BPIC 2017 cases
    python run_des.py --cases 100 --bpic data/bpic/BPI_Challenge_2017.xes

    # Agents estimate their own durations (default)
    python run_des.py --cases 20 --durations llm

    # Baseline: durations sampled from fitted distributions
    python run_des.py --cases 20 --durations distribution

    # Custom workforce and arrival rate
    python run_des.py --cases 50 --jc 3 --sc 2 --co 1 --arrival 30
"""

import argparse
import os
import numpy as np
from dotenv import load_dotenv

from src.queue.simulation_engine import SimulationEngine, EngineConfig
from src.clock.simulation_clock import SimulationClock
from src.process.loan_application import LOAN_PROCESS
from src.simulation_controller import generate_synthetic_case, load_bpic_cases
from src.logger import setup_logging

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description="Multi-LLM-Agent BPS - DES Engine")
    p.add_argument("--cases",   type=int,   default=20,       help="Number of cases to simulate")
    p.add_argument("--bpic",    type=str,   default=None,     help="Path to BPIC XES file")
    p.add_argument("--model",   type=str,   default=None,     help="LLM model name")
    p.add_argument("--seed",    type=int,   default=42,       help="Random seed")
    p.add_argument("--output",  type=str,   default="output", help="Output directory")
    p.add_argument("--jc",      type=int,   default=2,        help="Number of Junior Clerks")
    p.add_argument("--sc",      type=int,   default=1,        help="Number of Senior Clerks")
    p.add_argument("--co",      type=int,   default=1,        help="Number of Credit Officers")
    p.add_argument("--arrival", type=float, default=60.0,     help="Mean inter-arrival time (minutes)")
    p.add_argument(
        "--durations", type=str, default="complexity",
        choices=["complexity", "llm", "distribution"],
        help=(
            "Where hands-on work time comes from. "
            "complexity: agent rates the case 1-5, engine scales an anchor. "
            "llm: agent's own absolute estimate. "
            "distribution: sample from the fitted log-normal. "
            "(default: complexity)"
        ),
    )
    p.add_argument(
        "--noise", type=float, default=0.0,
        help="Lognormal noise on work time as a coefficient of variation (0 = off)",
    )
    p.add_argument("--verbose", action="store_true",          help="Print debug logs to console")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(output_dir=args.output, verbose=args.verbose)

    model    = args.model or os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Cases
    if args.bpic:
        cases = load_bpic_cases(args.bpic, max_cases=args.cases)
        print(f"  Loaded {len(cases)} cases from BPIC")
    else:
        rng = np.random.default_rng(args.seed)
        cases = [
            generate_synthetic_case(f"LOAN-{i:04d}", rng)
            for i in range(1, args.cases + 1)
        ]
        print(f"  Generated {len(cases)} synthetic cases (seed={args.seed})")

    config = EngineConfig(
        workforce={
            "junior_clerk":   args.jc,
            "senior_clerk":   args.sc,
            "credit_officer": args.co,
        },
        mean_interarrival_s=args.arrival * 60,
        duration_source=args.durations,
        duration_noise_cv=args.noise,
        model=model,
        ollama_base_url=base_url,
    )

    clock = SimulationClock(start_timestamp="2017-01-02T08:00:00", seed=args.seed)

    engine = SimulationEngine(
        cases=cases,
        process=LOAN_PROCESS,
        config=config,
        clock=clock,
        output_dir=args.output,
    )

    engine.run(seed=args.seed)

    # Export
    engine.export_json("simulation.json")
    if engine._global_event_log:
        engine.export_xes("simulation.xes")

    # Case summary
    df = engine.summary_dataframe()
    if not df.empty:
        print("\n  Case summary:")
        print(df.to_string(index=False))

        print(f"\n  Approval rate: {(df['status'] == 'approved').mean() * 100:.1f}%")
        print(f"  Avg steps/case: {df['n_steps'].mean():.1f}")
        print(f"  Avg cycle time: {df['cycle_time_h'].mean():.1f}h")
        print(f"  Avg work time:  {df['work_time_h'].mean():.1f}h")
        print(f"  Avg queue time: {df['queue_time_h'].mean():.1f}h")

    # What the agents said about time and complexity
    dur = engine.duration_summary()
    if not dur.empty:
        print("\n  Timing per tool:")
        print("    anchor = reference minutes | typical_mean = agent's free estimate")
        print("    cplx_* = complexity rating | used_* = minutes the engine applied")
        print(dur.to_string(index=False))

        cplx = engine.complexity_distribution()
        if not cplx.empty:
            print("\n  Complexity ratings used:")
            print(cplx.to_string(index=False))

        engine.duration_dataframe().to_csv(f"{args.output}/durations.csv", index=False)
        print(f"\n  Per-activity detail -> {args.output}/durations.csv")

    print()


if __name__ == "__main__":
    main()
