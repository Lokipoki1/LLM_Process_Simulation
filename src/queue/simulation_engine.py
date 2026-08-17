"""
simulation_engine.py
--------------------
Discrete Event Simulation (DES) engine for multi-LLM-agent BPS.

Domain-agnostic by construction: this module names no role, no activity
and no case attribute. Everything process-specific arrives through a
ProcessDefinition, so the same engine simulates a loan application, a
hospital admission or a support workflow without modification.

What it does own:
  - the event queue and the simulation timeline
  - one shared FIFO queue per role feeding several workers
  - working hours, including suspending work across shift boundaries
  - deciding how long each activity takes and when it finishes
  - emitting the event log

CLOCK OWNERSHIP
    For each step the engine determines the hands-on work time exactly
    once, spreads it across the worker's shift windows, schedules the
    completion event, and stamps the XES entry with the completion time.
    The executor performs the LLM call and never touches a clock.

DURATION SOURCE
    config.duration_source selects where work time comes from:

      "llm"           the agent's own duration_minutes estimate.
                      Time estimation becomes part of the agent's
                      cognition rather than a statistical module.
      "distribution"  a sample from the fitted log-normal in
                      SimulationClock - the AgentSimulator-style baseline.

    Both run on identical cases, so the two are directly comparable.
    Under "llm" an unusable estimate falls back to the distribution and
    is counted in duration_fallbacks.
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
    AgentPool, AgentWorker, WorkSchedule,
    compute_next_shift_start, compute_finish_time,
)
from .step_executor import StepExecutor, merge_state
from ..observer.observer import build_xes_entry
from ..clock.simulation_clock import SimulationClock
from ..process.definition import ProcessDefinition
from ..tools.duration_reference import resolve_minutes, DURATION_ANCHOR
from ..state import ProcessState

logger = logging.getLogger("bps.engine")


# -------------------------------------------------
# Configuration
# -------------------------------------------------

@dataclass
class EngineConfig:
    """
    Run configuration. Nothing here names a specific role: the workforce
    is a role -> headcount map filled in from the ProcessDefinition's
    roles by the caller.
    """
    # role -> number of workers. Empty means one worker per role.
    workforce: dict[str, int] = field(default_factory=dict)

    # role -> WorkSchedule override. Falls back to the definition.
    schedules: dict[str, WorkSchedule] = field(default_factory=dict)

    # Case arrivals
    mean_interarrival_s: float = 3600.0

    # Office hours used to snap arrivals to plausible submission times
    arrival_window: WorkSchedule = field(default_factory=lambda: WorkSchedule(8.0, 17.0))

    # Where hands-on work time comes from:
    #   "complexity"    agent rates the case 1-5, engine scales an anchor
    #   "llm"           agent's own absolute estimate, used as given
    #   "distribution"  sample from the fitted log-normal
    duration_source: str = "complexity"

    # Multiplicative lognormal noise on the resolved work time, as a
    # coefficient of variation. 0.0 disables it. Covers the operational
    # scatter no judgement can produce: interruptions, a phone call
    # mid-task, a colleague stopping by.
    duration_noise_cv: float = 0.0

    # Safety limits
    max_steps_per_case: int = 20
    max_events: int = 10000

    # LLM
    model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.3


# -------------------------------------------------
# Case result
# -------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    status: str
    rework_count: int
    n_steps: int
    n_log_entries: int
    wall_time_s: float      # real LLM time spent on this case
    sim_start: float        # sim time the case arrived
    sim_end: float          # sim time the case completed
    queue_time_s: float     # total time waiting in queues
    work_time_s: float      # total hands-on work time
    event_log: list = field(default_factory=list)


# -------------------------------------------------
# Engine
# -------------------------------------------------

class SimulationEngine:
    """
    Discrete event simulation engine.

        engine = SimulationEngine(cases, LOAN_PROCESS, config, clock)
        results = engine.run()
        engine.export_xes("simulation.xes")
    """

    def __init__(
        self,
        cases: list[dict],
        process: ProcessDefinition,
        config: EngineConfig | None = None,
        clock: SimulationClock | None = None,
        output_dir: str = "output",
    ):
        problems = process.validate()
        if problems:
            raise ValueError(
                "Invalid process definition:\n  - " + "\n  - ".join(problems)
            )

        self.process = process
        self.config = config or EngineConfig()
        self.cases = cases
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self._event_queue = EventQueue()
        self._clock = clock or SimulationClock()
        self._agent_pool = AgentPool()

        self._case_states: dict[str, ProcessState] = {}
        self._case_arrivals: dict[str, float] = {}
        self._case_queue_time: dict[str, float] = {}
        self._case_queue_start: dict[str, float] = {}
        self._case_wall_time: dict[str, float] = {}
        self._case_work_time: dict[str, float] = {}
        self._step_counts: dict[str, int] = {}

        self._results: list[CaseResult] = []
        self._global_event_log: list[dict] = []
        self._events_processed = 0
        self._scheduled_shifts: set[str] = set()
        self._last_printed = 0
        self._duration_fallbacks = 0
        self._duration_samples: list[dict] = []
        self._noise_rng = np.random.default_rng(20250729)

        self._build_agent_pool()

    # -- Setup ---------------------------------

    def _build_llm(self):
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
        """One executor shared by everyone, plus N worker slots per role."""
        self._executor = StepExecutor(self._build_llm(), self.process)

        for role in self.process.roles:
            count = self.config.workforce.get(role, 1)
            schedule = self.config.schedules.get(role) or self.process.schedule_for(role)
            for i in range(count):
                self._agent_pool.add_worker(AgentWorker(
                    worker_id=f"{role}_{i + 1}",
                    role=role,
                    schedule=schedule,
                ))

    # -- Routing -------------------------------

    def _validate_transition(self, current_role: str, next_role: str, case_id: str) -> bool:
        valid = self.process.valid_transitions.get(current_role, set())
        if next_role not in valid:
            logger.error(
                "%s | INVALID TRANSITION: %s -> %s (allowed: %s)",
                case_id, current_role, next_role, valid,
            )
            return False
        return True

    # -- Case arrivals -------------------------

    def _schedule_arrivals(self, rng: np.random.Generator):
        """Poisson arrivals, snapped into the submission window."""
        window = self.config.arrival_window
        current_time = self._clock.current

        for case in self.cases:
            current_time += rng.exponential(self.config.mean_interarrival_s)

            dt = datetime.fromtimestamp(current_time, tz=timezone.utc)
            hour = dt.hour + dt.minute / 60.0
            if (hour >= window.shift_end
                    or hour < window.shift_start
                    or dt.weekday() in window.days_off):
                current_time = compute_next_shift_start(current_time, window)

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

    # -- Duration resolution -------------------

    def _resolve_work_seconds(
        self, tool_name: str | None, tool_output: dict
    ) -> float:
        """
        Hands-on work time for one activity, per config.duration_source.

        "complexity"   the agent rated this case 1-5; the engine
                       multiplies the tool's anchor by that rating's
                       factor. Judgement from the agent, scale from the
                       reference log.
        "llm"          the agent's own absolute estimate, used as given.
        "distribution" a sample from the fitted log-normal.

        Every branch records what the agent said, including the free
        estimate it is never asked to act on, so calibration and
        complexity discrimination can be reported separately.
        """
        source = self.config.duration_source
        seconds: float | None = None
        resolution = source

        typical = tool_output.get("typical_duration_minutes")
        complexity = tool_output.get("case_complexity")

        if source == "complexity" and tool_name:
            if isinstance(complexity, (int, float)) and 1 <= complexity <= 5:
                minutes = resolve_minutes(tool_name, int(complexity))
                if minutes is not None:
                    seconds = minutes * 60.0
            if seconds is None:
                self._duration_fallbacks += 1
                resolution = "fallback_distribution"
                logger.warning(
                    "No usable case_complexity for %s - falling back to distribution",
                    tool_name,
                )

        elif source == "llm":
            if isinstance(typical, (int, float)) and typical > 0:
                seconds = float(typical) * 60.0
            else:
                self._duration_fallbacks += 1
                resolution = "fallback_distribution"
                logger.warning(
                    "No usable typical_duration_minutes for %s - falling back",
                    tool_name or "unknown",
                )

        if seconds is None:
            seconds = self._clock.sample_duration(tool_name or "_default")

        # Operational scatter, off by default
        cv = self.config.duration_noise_cv
        if cv > 0:
            import math
            sigma = math.sqrt(math.log(1 + cv ** 2))
            seconds *= float(self._noise_rng.lognormal(-0.5 * sigma ** 2, sigma))

        if tool_name:
            self._duration_samples.append({
                "tool":            tool_name,
                "resolution":      resolution,
                "agent_typical":   typical if isinstance(typical, (int, float)) else None,
                "agent_complexity": int(complexity) if isinstance(complexity, (int, float)) else None,
                "anchor":          DURATION_ANCHOR.get(tool_name),
                "resolved_minutes": round(seconds / 60, 1),
                "rationale":       str(tool_output.get("complexity_rationale", ""))[:120],
            })

        return seconds

    # -- Event handlers ------------------------

    def _handle_arrival(self, event: SimEvent):
        """A new case enters the system and joins the entry role's queue."""
        case_id = event.case_id

        self._case_states[case_id] = self.process.initial_state(event.data)
        self._case_arrivals[case_id] = event.time
        self._case_queue_time[case_id] = 0.0
        self._case_queue_start[case_id] = event.time
        self._case_wall_time[case_id] = 0.0
        self._case_work_time[case_id] = 0.0
        self._step_counts[case_id] = 0

        self._agent_pool.enqueue_case(self.process.entry_role, case_id)
        logger.info("%s | ARRIVED", case_id)
        self._try_dispatch_all(event.time)

    def _handle_step_complete(self, event: SimEvent):
        """A worker finished a step: record the event and route the case."""
        case_id = event.case_id
        self._agent_pool.release(event.worker_id)

        state = self._case_states[case_id]
        step_result = event.data
        next_role = step_result.get("next_role")

        # Record the event, unless this step produced no new action or the
        # tool is a silent internal handover.
        if step_result.get("produced_action") and state["agent_history"]:
            last_action = state["agent_history"][-1]
            dispatch_time = step_result.get("_dispatch_time", event.time)
            last_action["sim_timestamp"] = event.time
            last_action["real_duration"] = event.time - dispatch_time

            if not self.process.is_silent(last_action["tool_name"]):
                xes_entry = dict(build_xes_entry(
                    last_action,
                    self.process.activity_map,
                    self.process.resource_map,
                    start_timestamp=dispatch_time,
                ))
                self._global_event_log.append(xes_entry)
                state = merge_state(state, {"event_log": [xes_entry]})

            state = merge_state(state, {"sim_clock": event.time})
            self._case_states[case_id] = state

        if next_role is None:
            self._complete_case(case_id, event.time)
        else:
            current_role = state.get("current_agent", "unknown")
            if not self._validate_transition(current_role, next_role, case_id):
                logger.error("%s | Forcing case to rejected (invalid transition)", case_id)
                self._case_states[case_id] = merge_state(state, {"status": "rejected"})
                self._complete_case(case_id, event.time)
            else:
                self._case_queue_start[case_id] = event.time
                self._agent_pool.enqueue_case(next_role, case_id)

        self._try_dispatch_all(event.time)

    # -- Dispatch ------------------------------

    def _try_dispatch_all(self, sim_time: float):
        """Hand queued work to every idle, on-shift worker."""
        while True:
            claim = self._agent_pool.claim_next(sim_time)
            if claim is None:
                break
            worker, case_id = claim
            self._dispatch_step(worker, case_id, sim_time)

        # Wake up off-shift workers whose role has work waiting, otherwise
        # those cases are orphaned when everyone goes home.
        for w in self._agent_pool.workers_needing_shift_wakeup(sim_time):
            next_start = w.next_shift_start(sim_time)
            event_key = f"shift_{w.worker_id}_{next_start:.0f}"
            if event_key in self._scheduled_shifts:
                continue
            self._scheduled_shifts.add(event_key)
            self._event_queue.schedule_shift(
                time=next_start,
                worker_id=w.worker_id,
                event_type=EventType.SHIFT_START,
            )
            logger.debug(
                "%s off-shift with work waiting - shift start scheduled at %s",
                w.worker_id,
                datetime.fromtimestamp(next_start, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
            )

    def _dispatch_step(self, worker: AgentWorker, case_id: str, sim_time: float):
        """Run one step of `case_id` on `worker`. The case is already claimed."""
        state = self._case_states[case_id]
        self._step_counts[case_id] = self._step_counts.get(case_id, 0) + 1

        if case_id in self._case_queue_start:
            wait = sim_time - self._case_queue_start[case_id]
            self._case_queue_time[case_id] = self._case_queue_time.get(case_id, 0) + wait

        if self._step_counts[case_id] > self.config.max_steps_per_case:
            logger.warning("%s | MAX_STEPS reached - forcing completion", case_id)
            self._case_states[case_id] = merge_state(state, {"status": "rejected"})
            self._agent_pool.release(worker.worker_id)
            self._complete_case(case_id, sim_time)
            return

        logger.debug(
            "%s | DISPATCH to %s (step %d)",
            case_id, worker.worker_id, self._step_counts[case_id],
        )

        t0 = time.time()
        try:
            updated_state, next_role = self._executor.execute_step(state, worker.role)
        except Exception as e:
            logger.error("%s | EXECUTOR ERROR on %s: %s", case_id, worker.worker_id, e)
            updated_state = merge_state(state, {"status": "rejected"})
            next_role = None
        wall_time = time.time() - t0
        self._case_wall_time[case_id] = self._case_wall_time.get(case_id, 0.0) + wall_time

        produced_action = (
            len(updated_state["agent_history"]) > len(state["agent_history"])
        )
        last_action = updated_state["agent_history"][-1] if produced_action else None
        tool_name = last_action["tool_name"] if last_action else None
        tool_output = last_action["tool_output"] if last_action else {}

        work_seconds = self._resolve_work_seconds(tool_name, tool_output)
        self._case_work_time[case_id] = self._case_work_time.get(case_id, 0.0) + work_seconds

        finish_time = compute_finish_time(sim_time, work_seconds, worker.schedule)
        elapsed = finish_time - sim_time
        suspended = elapsed > work_seconds + 60

        logger.info(
            "%s | %s on %s | tool=%s | work=%.0fmin | elapsed=%.0fmin%s | wall=%.1fs | next=%s",
            case_id, worker.worker_id, worker.role,
            tool_name or "none", work_seconds / 60, elapsed / 60,
            " (suspended overnight)" if suspended else "",
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

    # -- Completion ----------------------------

    def _complete_case(self, case_id: str, sim_time: float):
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
            work_time_s=self._case_work_time.get(case_id, 0.0),
            event_log=state["event_log"],
        )
        self._results.append(result)

        path = "->".join(e["concept_name"] for e in state["event_log"])
        logger.info(
            "%s | COMPLETE | %s | reworks=%d | steps=%d | cycle=%.1fh | "
            "work=%.1fh | queue=%.1fh | path: %s",
            case_id, state["status"].upper(), state["rework_count"],
            self._step_counts.get(case_id, 0),
            (sim_time - arrival) / 3600,
            result.work_time_s / 3600,
            result.queue_time_s / 3600,
            path,
        )

    # -- Main loop -----------------------------

    def run(self, seed: int = 42) -> list[CaseResult]:
        rng = np.random.default_rng(seed)

        workforce = ", ".join(
            f"{self.config.workforce.get(r, 1)} {r}" for r in self.process.roles
        )
        print(f"\n{'=' * 62}")
        print("  DES Engine - Multi-LLM-Agent BPS")
        print(f"  Process: {self.process.name}")
        print(f"  Model: {self.config.model}")
        print(f"  Cases: {len(self.cases)} | Workforce: {workforce}")
        print(f"  Mean inter-arrival: {self.config.mean_interarrival_s / 60:.0f} min")
        print(f"  Duration source: {self.config.duration_source}")
        if self.process.silent_tools:
            print(f"  Silent tools: {', '.join(sorted(self.process.silent_tools))}")
        print(f"{'=' * 62}\n")

        self._schedule_arrivals(rng)

        while not self._event_queue.empty:
            event = self._event_queue.pop()
            self._events_processed += 1

            if self._events_processed > self.config.max_events:
                logger.warning(
                    "Global event limit (%d) reached - stopping",
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
        completed = len(self._results)
        total = len(self.cases)

        if completed > self._last_printed:
            r = self._results[-1]
            print(
                f"  [{completed:3d}/{total}] {r.case_id} -> "
                f"{r.status.upper():8s} | {r.n_steps} steps | "
                f"cycle {(r.sim_end - r.sim_start) / 3600:.1f}h | "
                f"work {r.work_time_s / 3600:.1f}h | "
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
        total = len(self._results)
        by_status: dict[str, int] = {}
        for r in self._results:
            by_status[r.status] = by_status.get(r.status, 0) + 1

        print(f"\n{'-' * 62}")
        print(f"  Completed: {total}/{len(self.cases)}")
        print("  " + " | ".join(f"{k}: {v}" for k, v in sorted(by_status.items())))
        if total:
            print(f"  Avg steps/case: {sum(r.n_steps for r in self._results) / total:.1f}")
            print(f"  Avg cycle time: {sum(r.sim_end - r.sim_start for r in self._results) / total / 3600:.1f}h")
            print(f"  Avg work time:  {sum(r.work_time_s for r in self._results) / total / 3600:.1f}h")
            print(f"  Avg queue time: {sum(r.queue_time_s for r in self._results) / total / 3600:.1f}h")
        print(f"  Events processed: {self._events_processed}")
        print(f"  Event log entries: {len(self._global_event_log)}")
        if self.config.duration_source in ("llm", "complexity"):
            print(f"  Duration fallbacks: {self._duration_fallbacks}")
        print(f"{'=' * 62}\n")

    # -- Duration analysis ---------------------

    def duration_dataframe(self) -> pd.DataFrame:
        """
        One row per activity: what the agent said and what the engine
        used. Columns:
            tool, resolution, agent_typical, agent_complexity,
            anchor, resolved_minutes, rationale
        """
        if not self._duration_samples:
            return pd.DataFrame()
        return pd.DataFrame(self._duration_samples)

    def duration_summary(self) -> pd.DataFrame:
        """
        Per-tool view of the two abilities being measured:

            agent_typical vs anchor  absolute calibration
            complexity spread        discrimination between cases
        """
        df = self.duration_dataframe()
        if df.empty:
            return df
        out = df.groupby("tool").agg(
            n=("tool", "count"),
            anchor=("anchor", "first"),
            typical_mean=("agent_typical", "mean"),
            cplx_mean=("agent_complexity", "mean"),
            cplx_std=("agent_complexity", "std"),
            cplx_min=("agent_complexity", "min"),
            cplx_max=("agent_complexity", "max"),
            used_mean=("resolved_minutes", "mean"),
            used_std=("resolved_minutes", "std"),
        ).round(2).reset_index()
        return out

    def complexity_distribution(self) -> pd.DataFrame:
        """How often each 1-5 rating was used, across all activities."""
        df = self.duration_dataframe()
        if df.empty or "agent_complexity" not in df:
            return pd.DataFrame()
        counts = df["agent_complexity"].value_counts().sort_index()
        return pd.DataFrame({
            "complexity": counts.index,
            "n": counts.values,
            "share": (counts.values / len(df) * 100).round(1),
        })

    # -- Export --------------------------------

    def export_xes(self, filename: str = "simulation.xes") -> Path:
        import pm4py
        if not self._global_event_log:
            raise RuntimeError("No data - run the simulation first.")

        df = pd.DataFrame(self._global_event_log).rename(columns={
            "case_concept_name":    "case:concept:name",
            "concept_name":         "concept:name",
            "start_timestamp":      "start_timestamp",
            "time_timestamp":       "time:timestamp",
            "org_resource":         "org:resource",
            "lifecycle_transition": "lifecycle:transition",
        })
        for col in ("start_timestamp", "time:timestamp"):
            df[col] = pd.to_datetime(df[col], format="ISO8601", utc=True)
            
        df = df.sort_values(["case:concept:name", "time:timestamp"])

        out = self.output_dir / filename
        pm4py.write_xes(pm4py.convert_to_event_log(df), str(out))
        print(f"  XES exported -> {out}")
        return out

    def export_json(self, filename: str = "simulation.json") -> Path:
        out = self.output_dir / filename
        with open(out, "w") as f:
            json.dump(self._global_event_log, f, indent=2, default=str)
        print(f"  JSON exported -> {out}")
        return out

    def summary_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "case_id":       r.case_id,
            "status":        r.status,
            "rework_count":  r.rework_count,
            "n_steps":       r.n_steps,
            "n_log_entries": r.n_log_entries,
            "cycle_time_h":  round((r.sim_end - r.sim_start) / 3600, 2),
            "work_time_h":   round(r.work_time_s / 3600, 2),
            "queue_time_h":  round(r.queue_time_s / 3600, 2),
            "wall_time_s":   r.wall_time_s,
        } for r in self._results])
