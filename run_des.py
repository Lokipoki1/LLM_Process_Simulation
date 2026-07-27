"""
run_des.py
----------
Run the Discrete Event Simulation engine.

Usage:
    # Synthetic cases, default config:
    python run_des.py --cases 20

    # With BPIC data:
    python run_des.py --cases 100 --bpic data/bpic/BPI_Challenge_2012.xes

    # Custom workforce:
    python run_des.py --cases 50 --jc 3 --sc 2 --co 1

    # With verbose logging:
    python run_des.py --cases 20 --verbose
"""

import argparse
import os
import numpy as np
from dotenv import load_dotenv

from src.engine.simulation_engine import SimulationEngine, EngineConfig
from src.engine.agent_pool import WorkSchedule
from src.clock.simulation_clock import SimulationClock
from src.simulation_controller import generate_synthetic_case, load_bpic_cases
from src.logger import setup_logging

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description="Multi-LLM-Agent BPS — DES Engine")
    p.add_argument("--cases",     type=int,   default=20,     help="Number of cases to simulate")
    p.add_argument("--bpic",      type=str,   default=None,   help="Path to BPIC XES file")
    p.add_argument("--model",     type=str,   default=None,   help="LLM model name")
    p.add_argument("--seed",      type=int,   default=42,     help="Random seed")
    p.add_argument("--output",    type=str,   default="output", help="Output directory")
    p.add_argument("--jc",        type=int,   default=2,      help="Number of Junior Clerks")
    p.add_argument("--sc",        type=int,   default=1,      help="Number of Senior Clerks")
    p.add_argument("--co",        type=int,   default=1,      help="Number of Credit Officers")
    p.add_argument("--arrival",   type=float, default=60.0,   help="Mean inter-arrival time (minutes)")
    p.add_argument("--verbose",   action="store_true",        help="Print debug logs to console")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(output_dir=args.output, verbose=args.verbose)

    model    = args.model or os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Prepare cases
    if args.bpic:
        cases = load_bpic_cases(args.bpic, max_cases=args.cases)
        print(f"  Loaded {len(cases)} cases from BPIC")
    else:
        rng = np.random.default_rng(args.seed)
        cases = [generate_synthetic_case(f"LOAN-{i:04d}", rng) for i in range(1, args.cases + 1)]
        print(f"  Generated {len(cases)} synthetic cases (seed={args.seed})")

    # Build engine config
    config = EngineConfig(
        n_junior_clerks=args.jc,
        n_senior_clerks=args.sc,
        n_credit_officers=args.co,
        mean_interarrival_s=args.arrival * 60,  # convert minutes to seconds
        model=model,
        ollama_base_url=base_url,
    )

    # Build clock
    clock = SimulationClock(start_timestamp="2012-01-02T08:00:00", seed=args.seed)

    # Build and run engine
    engine = SimulationEngine(
        cases=cases,
        config=config,
        clock=clock,
        output_dir=args.output,
    )

    results = engine.run(seed=args.seed)

    # Export
    engine.export_json("simulation.json")
    if engine._global_event_log:
        engine.export_xes("simulation.xes")

    # Summary table
    df = engine.summary_dataframe()
    if not df.empty:
        print("\n  Case summary:")
        print(df.to_string(index=False))

        approved_pct = (df["status"] == "approved").mean() * 100
        print(f"\n  Approval rate: {approved_pct:.1f}%")
        print(f"  Avg steps/case: {df['n_steps'].mean():.1f}")
        print(f"  Avg cycle time: {df['cycle_time_h'].mean():.1f}h")
        print(f"  Avg queue time: {df['queue_time_h'].mean():.1f}h\n")


if __name__ == "__main__":
    main()
