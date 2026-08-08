"""
simulation_engine.py
--------------------
Discrete Event Simulation (DES) engine for multi-LLM-agent BPS.

This replaces the sequential SimulationController with a proper
event-based simulator where:
  - Multiple cases run concurrently
  - Agents have working hours (busyness clock)
  - Cases wait in queues when agents are busy
  - Time advances by jumping between events (not step-by-step)

This is the core contribution of the thesis: the simulation ENGINE
that orchestrates LLM agents in a realistic temporal environment.
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
from .agent_pool import AgentPool, AgentWorker, WorkSchedule
from .process_graph import ProcessStepExecutor, make_initial_state
from ..observer.observer import build_xes_entry
from ..clock.simulation_clock import SimulationClock
from ..state import ProcessState, LoanApplication, AgentAction, XESEntry

logger = logging.getLogger("bps.engine")


# ─────────────────────────────────────────────
# State merge utility (replaces LangGraph's auto-merge)
# ─────────────────────────────────────────────

# Fields that use append semantics (operator.add in LangGraph)
_APPEND_FIELDS = {"messages", "agent_history", "event_log"}


def merge_state(current: ProcessState, partial: dict) -> ProcessState:
    """
    Merge a partial state update into the current ProcessState.
    Append-only fields are extended; all others are overwritten.
    """
    merged = dict(current)
    for key, value in partial.items():
        if key in _APPEND_FIELDS and isinstance(value, list):
            merged[key] = list(merged.get(key, [])) + value
        else:
            merged[key] = value
    return ProcessState(**merged)


# ─────────────────────────────────────────────
# Routing logic (extracted from simulation_graph.route)
# ─────────────────────────────────────────────

def resolve_next_role(state: ProcessState) -> str | None:
    """
    Determine which role should handle the next step of a case.
    Returns the role name or None if the case is complete.
    """
    if state["status"] in ("approved", "rejected"):
        return None  # case is done

    next_agent = state.get("next_agent")
    if next_agent in ("junior_clerk", "senior_clerk", "credit_officer"):
        return next_agent

    # Agent continues their own sequence
    current = state.get("current_agent")
    if current in ("junior_clerk", "senior_clerk", "credit_officer"):
        return current

    return None  # fallback: case is done


# Valid transitions — enforces the process flow
# A case can only move to a role if there's a valid path to it
VALID_TRANSITIONS: dict[str, set[str]] = {
    "junior_clerk":   {"junior_clerk", "senior_clerk"},     # JC → JC (own steps) or SC (forward)
    "senior_clerk":   {"senior_clerk", "credit_officer", "junior_clerk"},  # SC → SC, CO (escalate), JC (rework)
    "credit_officer": {"credit_officer"},                    # CO → CO (assess then decide), then END
}


def validate_transition(current_role: str, next_role: str, case_id: str) -> bool:
    """
    Check that a case transition is valid.
    Logs a warning if the transition violates the process flow.
    """
    valid = VALID_TRANSITIONS.get(current_role, set())
    if next_role not in valid:
        logger.error(
            "%s | INVALID TRANSITION: %s → %s (allowed: %s)",
            case_id, current_role, next_role, valid,
        )
        return False
    return True


# ─────────────────────────────────────────────
# Engine configuration
# ─────────────────────────────────────────────

@dataclass
class EngineConfig:
    """Configuration for the simulation engine."""
    # Workforce
    n_junior_clerks: int      = 2
    n_senior_clerks: int      = 1
    n_credit_officers: int    = 1

    # Working hours
    jc_schedule: WorkSchedule = field(default_factory=lambda: WorkSchedule(8.0, 17.0))
    sc_schedule: WorkSchedule = field(default_factory=lambda: WorkSchedule(8.0, 17.0))
    co_schedule: WorkSchedule = field(default_factory=lambda: WorkSchedule(9.0, 17.0))

    # Case arrivals
    mean_interarrival_s: float = 3600.0   # mean time between case arrivals (seconds)

    # Safety limits
    max_steps_per_case: int   = 20
    max_events: int           = 10_000    # global event limit

    # LLM
    model: str                = "gpt-4o-mini"
    ollama_base_url: str      = "http://localhost:11434"
    temperature: float        = 0.3


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
    wall_time_s: float          # real elapsed time for this case
    sim_start: float            # sim time when case arrived
    sim_end: float              # sim time when case completed
    queue_time_s: float         # total time spent waiting in queues
    event_log: list = field(default_factory=list)


# ─────────────────────────────────────────────
# The Engine
# ─────────────────────────────────────────────

class SimulationEngine:
    """
    Discrete event simulation engine for multi-LLM-agent BPS.

    Usage:
        engine = SimulationEngine(cases, config)
        results = engine.run()
        engine.export_xes("output/simulation.xes")
    """

    def __init__(
        self,
        cases: list[LoanCase],
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
        self._case_states: dict[str, ProcessState] = {}      # case_id → current state
        self._case_arrivals: dict[str, float] = {}            # case_id → arrival time
        self._case_queue_time: dict[str, float] = {}          # case_id → accumulated queue time
        self._case_queue_start: dict[str, float] = {}         # case_id → when current queue wait started
        self._step_counts: dict[str, int] = {}                # case_id → steps processed
        self._results: list[CaseResult] = []
        self._global_event_log: list[dict] = []
        self._events_processed = 0
        self._scheduled_shifts: set[str] = set()  # track scheduled shift events

        self._build_agent_pool()

    def _build_llm(self):
        """Instantiate the LLM based on config."""
        model = self.config.model
        if model.startswith("gpt-"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=self.config.temperature)
        else:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model, base_url=self.config.ollama_base_url,
                temperature=self.config.temperature,
            )

    def _build_agent_pool(self):
        """Create workers for each role and the shared step executor."""
        llm = self._build_llm()

        # The ProcessStepExecutor wraps LangGraph's process definition
        # and provides single-step execution for the DES engine.
        self._executor = ProcessStepExecutor(llm, self._clock)

        for i in range(self.config.n_junior_clerks):
            self._agent_pool.add_worker(AgentWorker(
                worker_id=f"junior_clerk_{i+1}",
                role="junior_clerk",
                agent=None,  # agents live in the executor now
                schedule=self.config.jc_schedule,
            ))

        for i in range(self.config.n_senior_clerks):
            self._agent_pool.add_worker(AgentWorker(
                worker_id=f"senior_clerk_{i+1}",
                role="senior_clerk",
                agent=None,
                schedule=self.config.sc_schedule,
            ))

        for i in range(self.config.n_credit_officers):
            self._agent_pool.add_worker(AgentWorker(
                worker_id=f"credit_officer_{i+1}",
                role="credit_officer",
                agent=None,
                schedule=self.config.co_schedule,
            ))

    # ── Case arrival scheduling ───────────────────

    def _schedule_arrivals(self, rng: np.random.Generator):
        """Schedule all case arrivals using an exponential inter-arrival process."""
        current_time = self._clock.current
        for case in self.cases:
            # Exponential inter-arrival time (Poisson process)
            gap = rng.exponential(self.config.mean_interarrival_s)
            current_time += gap

            # Snap to working hours: if arrival lands outside 8-17, move to next shift
            dt = datetime.fromtimestamp(current_time, tz=timezone.utc)
            hour = dt.hour + dt.minute / 60.0
            if hour >= 17.0 or hour < 8.0 or dt.weekday() >= 5:
                # Create a temp worker just to use next_shift_start logic
                temp = AgentWorker(
                    worker_id="_temp", role="_temp",
                    agent=None,  # type: ignore
                    schedule=WorkSchedule(8.0, 17.0),
                )
                current_time = temp.next_shift_start(current_time)

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

    # ── Initial state factory ─────────────────────

    def _make_initial_state(self, case_data: dict) -> ProcessState:
        application = {
            "case_id": case_data["case_id"],
            "amount_requested": case_data["amount_requested"],
            "loan_goal": case_data["loan_goal"],
            "application_type": case_data.get("application_type", "New credit"),
        }
        credit_data = case_data.get("credit_bureau_data")
        return ProcessState(
            application=application,
            credit_bureau_data=credit_data,
            credit_checked=False,
            status="pending", current_agent="junior_clerk",
            messages=[], agent_history=[], sim_clock=0.0, event_log=[],
            rework_count=0, rejection_reason=None, next_agent=None,
        )

    # ── Event handlers ────────────────────────────

    def _handle_arrival(self, event: SimEvent):
        """A new case enters the system → enqueue to junior_clerk."""
        case_id = event.case_id
        state = self._make_initial_state(event.data)
        self._case_states[case_id] = state
        self._case_arrivals[case_id] = event.time
        self._case_queue_time[case_id] = 0.0
        self._case_queue_start[case_id] = event.time
        self._step_counts[case_id] = 0

        self._agent_pool.enqueue_case("junior_clerk", case_id)
        logger.info(
            "%s | ARRIVED | EUR %.0f | goal=%s | type=%s",
            case_id, event.data.get("amount_requested", 0),
            event.data.get("loan_goal", "?"),
            event.data.get("application_type", "?"),
        )
        self._try_dispatch_all(event.time)

    def _handle_step_complete(self, event: SimEvent):
        """An agent finished one step → route the case to the next queue."""
        case_id = event.case_id
        worker_id = event.worker_id
        worker = self._agent_pool.get_worker(worker_id)

        # Free the worker
        worker.current_case = None
        worker.busy_until = event.time

        state = self._case_states[case_id]
        step_result = event.data
        next_role = step_result.get("next_role")

        # Record XES entry from the latest action
        if state["agent_history"]:
            last_action = state["agent_history"][-1]
            xes_entry = dict(build_xes_entry(last_action))
            self._global_event_log.append(xes_entry)

        if next_role is None:
            # Case is complete
            self._complete_case(case_id, event.time)
        else:
            # Validate the transition
            current_role = state.get("current_agent", "unknown")
            if not validate_transition(current_role, next_role, case_id):
                logger.error(
                    "%s | Forcing case to rejected due to invalid transition",
                    case_id,
                )
                state = merge_state(state, {"status": "rejected"})
                self._case_states[case_id] = state
                self._complete_case(case_id, event.time)
            else:
                # Enqueue for next role
                self._case_queue_start[case_id] = event.time
                self._agent_pool.enqueue_case(next_role, case_id)

        # Try to dispatch any waiting work
        self._try_dispatch_all(event.time)

    def _try_dispatch_all(self, sim_time: float):
        """Try to dispatch work to any idle worker that has cases queued."""
        dispatched = True
        while dispatched:
            dispatched = False
            worker = self._agent_pool.find_any_dispatchable(sim_time)
            if worker:
                self._dispatch_step(worker, sim_time)
                dispatched = True

        # Schedule shift-start events for workers with queued work who are off-shift.
        # Without this, cases get orphaned when all workers go home.
        for w in self._agent_pool.get_all_workers():
            if w.queue and w.is_idle and not w.is_on_shift(sim_time):
                next_start = w.next_shift_start(sim_time)
                # Avoid duplicate shift events
                event_key = f"shift_{w.worker_id}_{next_start:.0f}"
                if event_key not in self._scheduled_shifts:
                    self._scheduled_shifts.add(event_key)
                    self._event_queue.schedule_shift(
                        time=next_start,
                        worker_id=w.worker_id,
                        event_type=EventType.SHIFT_START,
                    )
                    logger.debug(
                        "%s off-shift with %d queued — scheduled shift start at %s",
                        w.worker_id, len(w.queue),
                        __import__('datetime').datetime.fromtimestamp(
                            next_start, tz=__import__('datetime').timezone.utc
                        ).strftime("%Y-%m-%d %H:%M"),
                    )

    def _dispatch_step(self, worker: AgentWorker, sim_time: float):
        """
        Have a worker process one step of the next case in their queue.

        Uses the ProcessStepExecutor which wraps LangGraph's process
        definition — the agent call + observer happen inside the executor.
        """
        if not worker.queue:
            return

        case_id = worker.queue.popleft()
        state = self._case_states[case_id]
        self._step_counts[case_id] = self._step_counts.get(case_id, 0) + 1

        # Track queue waiting time
        if case_id in self._case_queue_start:
            wait = sim_time - self._case_queue_start[case_id]
            self._case_queue_time[case_id] = self._case_queue_time.get(case_id, 0) + wait

        # Safety check
        if self._step_counts[case_id] > self.config.max_steps_per_case:
            logger.warning("%s | MAX_STEPS reached — forcing completion", case_id)
            state = merge_state(state, {"status": "rejected"})
            self._case_states[case_id] = state
            self._complete_case(case_id, sim_time)
            return

        # Mark worker as busy
        worker.current_case = case_id

        logger.debug(
            "%s | DISPATCH to %s (step %d)",
            case_id, worker.worker_id, self._step_counts[case_id],
        )

        # Execute one step through the ProcessStepExecutor
        # This calls the LLM agent + observer (LangGraph process semantics)
        t0 = time.time()
        try:
            updated_state, next_role = self._executor.execute_step(state, worker.role)
        except Exception as e:
            logger.error("%s | EXECUTOR ERROR on %s: %s", case_id, worker.worker_id, e)
            updated_state = merge_state(state, {"status": "rejected"})
            next_role = None
        wall_time = time.time() - t0

        # Determine activity duration from the clock
        tool_name = None
        if updated_state["agent_history"] and len(updated_state["agent_history"]) > len(state["agent_history"]):
            tool_name = updated_state["agent_history"][-1]["tool_name"]

        if tool_name:
            activity_duration = self._clock.sample_duration(tool_name)
        else:
            activity_duration = self._clock.sample_duration("_default")

        finish_time = sim_time + activity_duration

        logger.info(
            "%s | %s on %s | tool=%s | duration=%.0fs (%.0fmin) | wall=%.1fs | next=%s",
            case_id, worker.worker_id, worker.role,
            tool_name or "none", activity_duration, activity_duration / 60,
            wall_time, next_role or "END",
        )

        # Store the updated state and schedule completion
        self._case_states[case_id] = updated_state

        # Package the result for the completion event
        step_result = {
            "next_role": next_role,
            "_dispatch_time": sim_time,
        }

        self._event_queue.schedule_step_complete(
            time=finish_time,
            case_id=case_id,
            worker_id=worker.worker_id,
            step_result=step_result,
        )

    def _complete_case(self, case_id: str, sim_time: float):
        """Mark a case as complete and record the result."""
        state = self._case_states[case_id]
        arrival = self._case_arrivals.get(case_id, sim_time)

        result = CaseResult(
            case_id=case_id,
            status=state["status"],
            rework_count=state["rework_count"],
            n_steps=self._step_counts.get(case_id, 0),
            n_log_entries=len(state["event_log"]),
            wall_time_s=0.0,  # TODO: track real wall time per case
            sim_start=arrival,
            sim_end=sim_time,
            queue_time_s=self._case_queue_time.get(case_id, 0.0),
            event_log=state["event_log"],
        )
        self._results.append(result)

        path = "->".join(e["concept_name"] for e in state["event_log"])
        cycle_time = (sim_time - arrival) / 3600
        logger.info(
            "%s | COMPLETE | %s | reworks=%d | steps=%d | cycle=%.1fh | queue=%.1fh | path: %s",
            case_id, state["status"].upper(), state["rework_count"],
            self._step_counts.get(case_id, 0),
            cycle_time, self._case_queue_time.get(case_id, 0) / 3600,
            path,
        )

    # ── Main run loop ─────────────────────────────

    def run(self, seed: int = 42) -> list[CaseResult]:
        """
        Run the discrete event simulation.

        Returns a list of CaseResult for all completed cases.
        """
        rng = np.random.default_rng(seed)

        print(f"\n{'='*60}")
        print(f"  DES Engine — Multi-LLM-Agent BPS")
        print(f"  Model: {self.config.model}")
        print(f"  Cases: {len(self.cases)} | Workers: "
              f"{self.config.n_junior_clerks} JC + "
              f"{self.config.n_senior_clerks} SC + "
              f"{self.config.n_credit_officers} CO")
        print(f"  Mean inter-arrival: {self.config.mean_interarrival_s/60:.0f} min")
        print(f"{'='*60}\n")

        # Schedule all case arrivals
        self._schedule_arrivals(rng)

        # Main DES loop
        while not self._event_queue.empty:
            event = self._event_queue.pop()
            self._events_processed += 1

            if self._events_processed > self.config.max_events:
                logger.warning("Global event limit (%d) reached — stopping", self.config.max_events)
                break

            if event.event_type == EventType.CASE_ARRIVAL:
                self._handle_arrival(event)
            elif event.event_type == EventType.STEP_COMPLETE:
                self._handle_step_complete(event)
            elif event.event_type == EventType.SHIFT_START:
                # Worker's shift starts — try to dispatch queued work
                logger.debug("SHIFT_START for %s", event.worker_id)
                self._try_dispatch_all(event.time)

            # Progress: print every completion + periodic heartbeat
            completed = len(self._results)
            total = len(self.cases)

            if completed > 0 and completed != getattr(self, '_last_printed', 0):
                r = self._results[-1]
                print(
                    f"  [{completed:3d}/{total}] {r.case_id} → "
                    f"{r.status.upper():8s} | {r.n_steps} steps | "
                    f"cycle {(r.sim_end - r.sim_start)/3600:.1f}h | "
                    f"queue {r.queue_time_s/3600:.1f}h"
                )
                self._last_printed = completed

            # Heartbeat every 50 events so the user knows it's alive
            if self._events_processed % 50 == 0:
                active = total - completed
                print(
                    f"    ... {self._events_processed} events processed | "
                    f"{completed} done | {active} in progress",
                    flush=True,
                )

        # Print summary
        self._print_summary()
        return self._results

    def _print_summary(self):
        """Print simulation summary to console."""
        approved = sum(1 for r in self._results if r.status == "approved")
        rejected = sum(1 for r in self._results if r.status == "rejected")
        total = len(self._results)

        print(f"\n{'─'*60}")
        print(f"  Completed: {total}/{len(self.cases)}")
        print(f"  Approved: {approved} | Rejected: {rejected}")
        if total > 0:
            avg_steps = sum(r.n_steps for r in self._results) / total
            avg_cycle = sum(r.sim_end - r.sim_start for r in self._results) / total / 3600
            avg_queue = sum(r.queue_time_s for r in self._results) / total / 3600
            print(f"  Avg steps/case: {avg_steps:.1f}")
            print(f"  Avg cycle time: {avg_cycle:.1f}h")
            print(f"  Avg queue time: {avg_queue:.1f}h")
        print(f"  Events processed: {self._events_processed}")
        print(f"  Event log entries: {len(self._global_event_log)}")
        print(f"{'='*60}\n")

    # ── Export ────────────────────────────────────

    def export_xes(self, filename: str = "simulation.xes") -> Path:
        """Export the accumulated event log to XES format via PM4Py."""
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
        out = self.output_dir / filename
        pm4py.write_xes(pm4py.convert_to_event_log(df), str(out))
        print(f"  XES exported → {out}")
        return out

    def export_json(self, filename: str = "simulation.json") -> Path:
        """Export event log as JSON for debugging."""
        out = self.output_dir / filename
        with open(out, "w") as f:
            json.dump(self._global_event_log, f, indent=2, default=str)
        print(f"  JSON exported → {out}")
        return out

    def summary_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame summarizing all case results."""
        return pd.DataFrame([{
            "case_id": r.case_id,
            "status": r.status,
            "rework_count": r.rework_count,
            "n_steps": r.n_steps,
            "n_log_entries": r.n_log_entries,
            "cycle_time_h": round((r.sim_end - r.sim_start) / 3600, 2),
            "queue_time_h": round(r.queue_time_s / 3600, 2),
        } for r in self._results])
