"""
simulation_engine.py
--------------------
Discrete Event Simulation (DES) engine for multi-LLM-agent BPS.

Event-based simulator where:
  - multiple cases run concurrently
  - agents have working hours (busyness clock)
  - cases wait in queues when agents are busy
  - time advances by jumping between events, not step by step

CLOCK OWNERSHIP
  The engine is the single source of truth for simulation time.
  For each step it samples the activity duration exactly once, schedules
  the completion event at `dispatch_time + duration`, and stamps the XES
  entry with that completion time. The ProcessStepExecutor performs the
  LLM call only — it never advances a clock and never emits XES entries.

  This matters: XES timestamps are what AED and CTD are computed from,
  so they must reflect the engine's queues and parallel workers.
"""

from __future__ import annotations
import json
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .events import EventQueue, EventType, SimEvent
from .agent_pool import (
    AgentPool, AgentWorker, WorkSchedule, compute_next_shift_start,
)
from .process_graph import ProcessStepExecutor
from ..observer.observer import build_xes_entry
from ..clock.simulation_clock import SimulationClock
from ..state import ProcessState

logger = logging.getLogger("bps.engine")


# ─────────────────────────────────────────────
# State merge (replicates LangGraph's operator.add)
# ─────────────────────────────────────────────

_APPEND_FIELDS = {"messages", "agent_history", "event_log"}


def merge_state(current: ProcessState, partial: dict) -> ProcessState:
    """Merge a partial update into the state; append-only fields are extended."""
    merged = dict(current)
    for key, value in partial.items():
        if key in _APPEND_FIELDS and isinstance(value, list):
            merged[key] = list(merged.get(key, [])) + value
        else:
            merged[key] = value
    return ProcessState(**merged)


# ─────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────

def resolve_next_role(state: ProcessState) -> str | None:
    """Which role handles the next step, or None if the case is complete."""
    if state["status"] in ("approved", "rejected"):
        return None

    next_agent = state.get("next_agent")
    if next_agent in ("junior_clerk", "senior_clerk", "credit_officer"):
        return next_agent

    current = state.get("current_agent")
    if current in ("junior_clerk", "senior_clerk", "credit_officer"):
        return current

    return None


# Valid transitions — enforces the process flow.
# A case can only move to a role if there is a valid path to it.
VALID_TRANSITIONS: dict[str, set[str]] = {
    "junior_clerk":   {"junior_clerk", "senior_clerk"},
    "senior_clerk":   {"senior_clerk", "credit_officer", "junior_clerk"},
    "credit_officer": {"credit_officer"},
}


def validate_transition(current_role: str, next_role: str, case_id: str) -> bool:
    """Check a case transition against the process flow; logs if invalid."""
    valid = VALID_TRANSITIONS.get(current_role, set())
    if next_role not in valid:
        logger.error(
            "%s | INVALID TRANSITION: %s -> %s (allowed: %s)",
            case_id, current_role, next_role, valid,
        )
        return False
    return True


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class EngineConfig:
    """Configuration for the simulation engine."""
    # Workforce
    n_junior_clerks: int = 2
    n_senior_clerks: int = 1
    n_credit_officers: int = 1

    # Working hours
    jc_schedule: WorkSchedule = field(default_factory=lambda: WorkSchedule(8.0, 17.0))
    sc_schedule: WorkSchedule = field(default_factory=lambda: WorkSchedule(8.0, 17.0))
    co_schedule: WorkSchedule = field(default_factory=lambda: WorkSchedule(9.0, 17.0))

    # Case arrivals
    mean_interarrival_s: float = 3600.0   # mean seconds between arrivals

    # Safety limits
    max_steps_per_case: int = 20
    max_events: int = 10_000

    # LLM
    model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.3


# ─────────────────────────────────────────────
# Case result
# ─────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    status: str
    rework_count: int
    n_steps: int
    n_log_entries: int
    wall_time_s: float      # real elapsed LLM time for this case
    sim_start: float        # sim time the case arrived
    sim_end: float          # sim time the case completed
    queue_time_s: float     # total time waiting in queues
    event_log: list = field(default_factory=list)


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

class SimulationEngine:
    """
    Discrete event simulation engine for multi-LLM-agent BPS.

    Usage:
        engine = SimulationEngine(cases, config, clock)
        results = engine.run()
        engine.export_xes("simulation.xes")
    """

    def __init__(
        self,
        cases: list[dict],
        config: EngineConfig | None = None,
        clock: SimulationClock | None = None,
        output_dir: str = "output",
    ):
        self.config = config or EngineConfig()
        self.cases = cases
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Simulation state
        self._event_queue = EventQueue()
        self._clock = clock or SimulationClock()
        self._agent_pool = AgentPool()
        self._case_states: dict[str, ProcessState] = {}
        self._case_arrivals: dict[str, float] = {}
        self._case_queue_time: dict[str, float] = {}
        self._case_queue_start: dict[str, float] = {}
        self._case_wall_time: dict[str, float] = {}
        self._step_counts: dict[str, int] = {}
        self._results: list[CaseResult] = []
        self._global_event_log: list[dict] = []
        self._events_processed = 0
        self._scheduled_shifts: set[str] = set()
        self._last_printed = 0

        self._build_agent_pool()

    # ── Setup ─────────────────────────────────────

    def _build_llm(self):
        """Instantiate the LLM based on config."""
        model = self.config.model
        if model.startswith("gpt-"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=self.config.temperature)
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            base_url=self.config.ollama_base_url,
            temperature=self.config.temperature,
        )

    def _build_agent_pool(self):
        """Create workers for each role plus the shared step executor."""
        llm = self._build_llm()

        # The executor holds the agents; workers are resources with queues.
        self._executor = ProcessStepExecutor(llm)

        roles = [
            ("junior_clerk", self.config.n_junior_clerks, self.config.jc_schedule),
            ("senior_clerk", self.config.n_senior_clerks, self.config.sc_schedule),
            ("credit_officer", self.config.n_credit_officers, self.config.co_schedule),
        ]
        for role, count, schedule in roles:
            for i in range(count):
                self._agent_pool.add_worker(AgentWorker(
                    worker_id=f"{role}_{i + 1}",
                    role=role,
                    schedule=schedule,
                ))

    # ── Case arrivals ─────────────────────────────

    def _schedule_arrivals(self, rng: np.random.Generator):
        """Schedule all case arrivals as a Poisson process, snapped to office hours."""
        office = WorkSchedule(8.0, 17.0)
        current_time = self._clock.current

        for case in self.cases:
            current_time += rng.exponential(self.config.mean_interarrival_s)

            # Snap arrivals outside office hours to the next shift start
            dt = datetime.fromtimestamp(current_time, tz=timezone.utc)
            hour = dt.hour + dt.minute / 60.0
            if hour >= office.shift_end or hour < office.shift_start or dt.weekday() >= 5:
                current_time = compute_next_shift_start(current_time, office)

            self._event_queue.schedule_arrival(
                time=current_time,
                case_id=case["case_id"],
                case_data=dict(case),
            )
            logger.debug(
                "Scheduled %s arrival at %s",
                case["case_id"],
                datetime.fromtimestamp(current_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
            )

    def _make_initial_state(self, case_data: dict) -> ProcessState:
        application = {
            "case_id":          case_data["case_id"],
            "amount_requested": case_data["amount_requested"],
            "loan_goal":        case_data["loan_goal"],
            "application_type": case_data.get("application_type", "New credit"),
        }
        return ProcessState(
            application=application,
            credit_bureau_data=case_data.get("credit_bureau_data"),
            credit_checked=False,
            status="pending", current_agent="junior_clerk",
            messages=[], agent_history=[], sim_clock=0.0, event_log=[],
            rework_count=0, rejection_reason=None, next_agent=None,
        )

    # ── Event handlers ────────────────────────────

    def _handle_arrival(self, event: SimEvent):
        """A new case enters the system → enqueue it for a junior clerk."""
        case_id = event.case_id

        self._case_states[case_id] = self._make_initial_state(event.data)
        self._case_arrivals[case_id] = event.time
        self._case_queue_time[case_id] = 0.0
        self._case_queue_start[case_id] = event.time
        self._case_wall_time[case_id] = 0.0
        self._step_counts[case_id] = 0

        self._agent_pool.enqueue_case("junior_clerk", case_id)
        logger.info(
            "%s | ARRIVED | EUR %.0f | goal=%s | type=%s",
            case_id,
            event.data.get("amount_requested", 0),
            event.data.get("loan_goal", "?"),
            event.data.get("application_type", "?"),
        )
        self._try_dispatch_all(event.time)

    def _handle_step_complete(self, event: SimEvent):
        """An agent finished a step → stamp the XES entry and route the case."""
        case_id = event.case_id
        worker = self._agent_pool.get_worker(event.worker_id)
        worker.current_case = None   # free the worker

        state = self._case_states[case_id]
        step_result = event.data
        next_role = step_result.get("next_role")

        # ── Stamp the XES entry with DES time ──
        # Only if this step actually produced a new action. Without this
        # guard, a step where the LLM returned text instead of a tool call
        # would re-log the previous action.
        if step_result.get("produced_action") and state["agent_history"]:
            last_action = state["agent_history"][-1]
            dispatch_time = step_result.get("_dispatch_time", event.time)
            last_action["sim_timestamp"] = event.time
            last_action["real_duration"] = event.time - dispatch_time

            xes_entry = dict(build_xes_entry(last_action))
            self._global_event_log.append(xes_entry)

            state = merge_state(state, {
                "event_log": [xes_entry],
                "sim_clock": event.time,
            })
            self._case_states[case_id] = state

        # ── Route the case ──
        if next_role is None:
            self._complete_case(case_id, event.time)
        else:
            current_role = state.get("current_agent", "unknown")
            if not validate_transition(current_role, next_role, case_id):
                logger.error("%s | Forcing case to rejected (invalid transition)", case_id)
                state = merge_state(state, {"status": "rejected"})
                self._case_states[case_id] = state
                self._complete_case(case_id, event.time)
            else:
                self._case_queue_start[case_id] = event.time
                self._agent_pool.enqueue_case(next_role, case_id)

        self._try_dispatch_all(event.time)

    # ── Dispatch ──────────────────────────────────

    def _try_dispatch_all(self, sim_time: float):
        """Dispatch work to every idle worker that has cases queued."""
        while True:
            worker = self._agent_pool.find_any_dispatchable(sim_time)
            if not worker:
                break
            self._dispatch_step(worker, sim_time)

        # Wake up off-shift workers that have work waiting, otherwise
        # their cases would be orphaned when everyone goes home.
        for w in self._agent_pool.get_all_workers():
            if w.queue and w.is_idle and not w.is_on_shift(sim_time):
                next_start = w.next_shift_start(sim_time)
                event_key = f"shift_{w.worker_id}_{next_start:.0f}"
                if event_key not in self._scheduled_shifts:
                    self._scheduled_shifts.add(event_key)
                    self._event_queue.schedule_shift(
                        time=next_start,
                        worker_id=w.worker_id,
                        event_type=EventType.SHIFT_START,
                    )
                    logger.debug(
                        "%s off-shift with %d queued — shift start scheduled at %s",
                        w.worker_id, len(w.queue),
                        datetime.fromtimestamp(next_start, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                    )

    def _dispatch_step(self, worker: AgentWorker, sim_time: float):
        """Have a worker process one step of the next case in its queue."""
        if not worker.queue:
            return

        case_id = worker.queue.popleft()
        state = self._case_states[case_id]
        self._step_counts[case_id] = self._step_counts.get(case_id, 0) + 1

        # Accumulate queue waiting time
        if case_id in self._case_queue_start:
            wait = sim_time - self._case_queue_start[case_id]
            self._case_queue_time[case_id] = self._case_queue_time.get(case_id, 0) + wait

        # Safety net
        if self._step_counts[case_id] > self.config.max_steps_per_case:
            logger.warning("%s | MAX_STEPS reached — forcing completion", case_id)
            self._case_states[case_id] = merge_state(state, {"status": "rejected"})
            self._complete_case(case_id, sim_time)
            return

        worker.current_case = case_id
        logger.debug(
            "%s | DISPATCH to %s (step %d)",
            case_id, worker.worker_id, self._step_counts[case_id],
        )

        # ── LLM call (no clock side effects inside) ──
        t0 = time.time()
        try:
            updated_state, next_role = self._executor.execute_step(state, worker.role)
        except Exception as e:
            logger.error("%s | EXECUTOR ERROR on %s: %s", case_id, worker.worker_id, e)
            updated_state = merge_state(state, {"status": "rejected"})
            next_role = None
        wall_time = time.time() - t0
        self._case_wall_time[case_id] = self._case_wall_time.get(case_id, 0.0) + wall_time

        # Did this step actually produce a new action?
        produced_action = (
            len(updated_state["agent_history"]) > len(state["agent_history"])
        )
        tool_name = (
            updated_state["agent_history"][-1]["tool_name"]
            if produced_action else None
        )

        # ── Sample the activity duration ONCE ──
        activity_duration = self._clock.sample_duration(tool_name or "_default")
        finish_time = sim_time + activity_duration

        logger.info(
            "%s | %s on %s | tool=%s | duration=%.0fs (%.0fmin) | wall=%.1fs | next=%s",
            case_id, worker.worker_id, worker.role,
            tool_name or "none", activity_duration, activity_duration / 60,
            wall_time, next_role or "END",
        )

        self._case_states[case_id] = updated_state

        self._event_queue.schedule_step_complete(
            time=finish_time,
            case_id=case_id,
            worker_id=worker.worker_id,
            step_result={
                "next_role":       next_role,
                "produced_action": produced_action,
                "_dispatch_time":  sim_time,
            },
        )

    # ── Completion ────────────────────────────────

    def _complete_case(self, case_id: str, sim_time: float):
        """Record the result for a finished case."""
        state = self._case_states[case_id]
        arrival = self._case_arrivals.get(case_id, sim_time)

        result = CaseResult(
            case_id=case_id,
            status=state["status"],
            rework_count=state["rework_count"],
            n_steps=self._step_counts.get(case_id, 0),
            n_log_entries=len(state["event_log"]),
            wall_time_s=round(self._case_wall_time.get(case_id, 0.0), 2),
            sim_start=arrival,
            sim_end=sim_time,
            queue_time_s=self._case_queue_time.get(case_id, 0.0),
            event_log=state["event_log"],
        )
        self._results.append(result)

        path = "->".join(e["concept_name"] for e in state["event_log"])
        logger.info(
            "%s | COMPLETE | %s | reworks=%d | steps=%d | cycle=%.1fh | queue=%.1fh | path: %s",
            case_id, state["status"].upper(), state["rework_count"],
            self._step_counts.get(case_id, 0),
            (sim_time - arrival) / 3600,
            self._case_queue_time.get(case_id, 0) / 3600,
            path,
        )

    # ── Main loop ─────────────────────────────────

    def run(self, seed: int = 42) -> list[CaseResult]:
        """Run the discrete event simulation. Returns all case results."""
        rng = np.random.default_rng(seed)

        print(f"\n{'=' * 60}")
        print("  DES Engine — Multi-LLM-Agent BPS")
        print(f"  Model: {self.config.model}")
        print(f"  Cases: {len(self.cases)} | Workers: "
              f"{self.config.n_junior_clerks} JC + "
              f"{self.config.n_senior_clerks} SC + "
              f"{self.config.n_credit_officers} CO")
        print(f"  Mean inter-arrival: {self.config.mean_interarrival_s / 60:.0f} min")
        print(f"{'=' * 60}\n")

        self._schedule_arrivals(rng)

        while not self._event_queue.empty:
            event = self._event_queue.pop()
            self._events_processed += 1

            if self._events_processed > self.config.max_events:
                logger.warning(
                    "Global event limit (%d) reached — stopping",
                    self.config.max_events,
                )
                break

            if event.event_type == EventType.CASE_ARRIVAL:
                self._handle_arrival(event)
            elif event.event_type == EventType.STEP_COMPLETE:
                self._handle_step_complete(event)
            elif event.event_type == EventType.SHIFT_START:
                logger.debug("SHIFT_START for %s", event.worker_id)
                self._try_dispatch_all(event.time)

            self._print_progress()

        self._print_summary()
        return self._results

    def _print_progress(self):
        """Print each completion, plus a heartbeat every 50 events."""
        completed = len(self._results)
        total = len(self.cases)

        if completed > self._last_printed:
            r = self._results[-1]
            print(
                f"  [{completed:3d}/{total}] {r.case_id} -> "
                f"{r.status.upper():8s} | {r.n_steps} steps | "
                f"cycle {(r.sim_end - r.sim_start) / 3600:.1f}h | "
                f"queue {r.queue_time_s / 3600:.1f}h"
            )
            self._last_printed = completed

        if self._events_processed % 50 == 0:
            print(
                f"    ... {self._events_processed} events | "
                f"{completed} done | {total - completed} in progress",
                flush=True,
            )

    def _print_summary(self):
        approved = sum(1 for r in self._results if r.status == "approved")
        rejected = sum(1 for r in self._results if r.status == "rejected")
        total = len(self._results)

        print(f"\n{'-' * 60}")
        print(f"  Completed: {total}/{len(self.cases)}")
        print(f"  Approved: {approved} | Rejected: {rejected}")
        if total:
            print(f"  Avg steps/case: {sum(r.n_steps for r in self._results) / total:.1f}")
            print(f"  Avg cycle time: {sum(r.sim_end - r.sim_start for r in self._results) / total / 3600:.1f}h")
            print(f"  Avg queue time: {sum(r.queue_time_s for r in self._results) / total / 3600:.1f}h")
        print(f"  Events processed: {self._events_processed}")
        print(f"  Event log entries: {len(self._global_event_log)}")
        print(f"{'=' * 60}\n")

    # ── Export ────────────────────────────────────

    def export_xes(self, filename: str = "simulation.xes") -> Path:
        """Export the accumulated event log to XES via PM4Py."""
        import pm4py
        if not self._global_event_log:
            raise RuntimeError("No data — run the simulation first.")

        df = pd.DataFrame(self._global_event_log).rename(columns={
            "case_concept_name":    "case:concept:name",
            "concept_name":         "concept:name",
            "time_timestamp":       "time:timestamp",
            "org_resource":         "org:resource",
            "lifecycle_transition": "lifecycle:transition",
        })
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
        df = df.sort_values(["case:concept:name", "time:timestamp"])

        out = self.output_dir / filename
        pm4py.write_xes(pm4py.convert_to_event_log(df), str(out))
        print(f"  XES exported -> {out}")
        return out

    def export_json(self, filename: str = "simulation.json") -> Path:
        """Export the event log as JSON for debugging."""
        out = self.output_dir / filename
        with open(out, "w") as f:
            json.dump(self._global_event_log, f, indent=2, default=str)
        print(f"  JSON exported -> {out}")
        return out

    def summary_dataframe(self) -> pd.DataFrame:
        """One row per case, for the console summary and later analysis."""
        return pd.DataFrame([{
            "case_id":       r.case_id,
            "status":        r.status,
            "rework_count":  r.rework_count,
            "n_steps":       r.n_steps,
            "n_log_entries": r.n_log_entries,
            "cycle_time_h":  round((r.sim_end - r.sim_start) / 3600, 2),
            "queue_time_h":  round(r.queue_time_s / 3600, 2),
            "wall_time_s":   r.wall_time_s,
        } for r in self._results])
