"""
agent_pool.py
-------------
Agent pool that manages workers, their schedules, and case queues.

Each role (junior_clerk, senior_clerk, credit_officer) can have one or
more workers. Each worker has:
  - a work schedule (shift_start, shift_end in hours; days off)
  - a FIFO queue of pending cases
  - a busyness state (idle, or currently holding a case)

This models a real office: when every junior clerk is busy, new cases
wait in a queue - producing the waiting times that make cycle times
comparable to the BPIC data.

SHIFT-AWARE WORK
    compute_finish_time() spreads an activity's work time across the
    available shift windows. An activity started at 16:50 that needs
    2 hours of work does not finish at 18:50 - it consumes the 10
    minutes left today and resumes at 08:00 tomorrow, finishing at
    09:50. This keeps event timestamps inside office hours, which is
    what the BPIC logs show and what AED/CED are computed against.
"""

from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("bps.agent_pool")


@dataclass
class WorkSchedule:
    """Working hours for an agent. Times are in hours (0-24)."""
    shift_start: float = 8.0    # 08:00
    shift_end: float = 17.0     # 17:00
    days_off: set[int] = field(default_factory=lambda: {5, 6})  # Sat=5, Sun=6

    @property
    def daily_seconds(self) -> float:
        """Seconds of work available in one full shift."""
        return (self.shift_end - self.shift_start) * 3600


# -------------------------------------------------
# Schedule helpers (module level so the engine can use them
# without instantiating a throwaway worker)
# -------------------------------------------------

def is_on_shift(sim_time: float, schedule: WorkSchedule) -> bool:
    """True if sim_time falls inside the given schedule's working hours."""
    dt = datetime.fromtimestamp(sim_time, tz=timezone.utc)
    if dt.weekday() in schedule.days_off:
        return False
    hour = dt.hour + dt.minute / 60.0
    return schedule.shift_start <= hour < schedule.shift_end


def _at_hour(day: datetime, hour_float: float) -> float:
    """Timestamp of `hour_float` on the given day."""
    return day.replace(
        hour=int(hour_float),
        minute=int((hour_float % 1) * 60),
        second=0, microsecond=0,
    ).timestamp()


def compute_next_shift_start(sim_time: float, schedule: WorkSchedule) -> float:
    """
    Return the sim time of the next shift start at or after sim_time.
    Skips days off.
    """
    dt = datetime.fromtimestamp(sim_time, tz=timezone.utc)
    hour = dt.hour + dt.minute / 60.0

    # Before today's shift start on a working day -> starts today
    if dt.weekday() not in schedule.days_off and hour < schedule.shift_start:
        return _at_hour(dt, schedule.shift_start)

    # Otherwise find the next working day
    candidate = dt + timedelta(days=1)
    for _ in range(7):
        if candidate.weekday() not in schedule.days_off:
            return _at_hour(candidate, schedule.shift_start)
        candidate += timedelta(days=1)

    # Fallback (only reachable if every day is a day off)
    return sim_time + 86400


def shift_end_timestamp(sim_time: float, schedule: WorkSchedule) -> float:
    """Timestamp of the shift end on the same calendar day as sim_time."""
    dt = datetime.fromtimestamp(sim_time, tz=timezone.utc)
    return _at_hour(dt, schedule.shift_end)


def compute_finish_time(
    start: float,
    work_seconds: float,
    schedule: WorkSchedule,
    max_iterations: int = 60,
) -> float:
    """
    Spread `work_seconds` of actual work across the schedule's shift
    windows, beginning at `start`. Returns the wall-clock sim time at
    which the work is finished.

    Work does not happen outside shift hours or on days off: an activity
    that does not fit in the remaining shift is suspended at the end of
    the day and resumed at the next shift start.

    Args:
        start:          sim time when the work begins
        work_seconds:   actual working seconds the activity requires
        schedule:       the worker's shift definition
        max_iterations: safety bound on how many shifts to span

    Returns:
        sim time (unix timestamp) when the activity completes
    """
    if work_seconds <= 0:
        return start

    t = start
    remaining = work_seconds

    # If we start outside the shift, wait for the next one
    if not is_on_shift(t, schedule):
        t = compute_next_shift_start(t, schedule)

    for _ in range(max_iterations):
        end_of_shift = shift_end_timestamp(t, schedule)
        available = end_of_shift - t

        if remaining <= available:
            return t + remaining

        # Consume what is left today, resume next shift
        remaining -= available
        t = compute_next_shift_start(end_of_shift, schedule)

    # Safety valve: the activity is longer than max_iterations shifts
    logger.warning(
        "compute_finish_time hit the iteration bound with %.0fs of work left",
        remaining,
    )
    return t + remaining


# -------------------------------------------------
# Worker
# -------------------------------------------------

@dataclass
class AgentWorker:
    """
    A single worker instance. One role can have several workers
    (e.g. 2 junior clerks, 1 senior clerk, 1 credit officer).

    The agent behaviour itself lives in the ProcessStepExecutor -
    a worker is a resource with a queue and a schedule, not an LLM.
    """
    worker_id: str                   # "junior_clerk_1"
    role: str                        # "junior_clerk"
    schedule: WorkSchedule
    queue: deque = field(default_factory=deque)   # pending case_ids (FIFO)
    current_case: str | None = None  # case being processed, None if idle

    @property
    def is_idle(self) -> bool:
        return self.current_case is None

    def is_on_shift(self, sim_time: float) -> bool:
        return is_on_shift(sim_time, self.schedule)

    def next_shift_start(self, sim_time: float) -> float:
        return compute_next_shift_start(sim_time, self.schedule)

    def finish_time_for(self, start: float, work_seconds: float) -> float:
        """When this worker finishes `work_seconds` of work started at `start`."""
        return compute_finish_time(start, work_seconds, self.schedule)


# -------------------------------------------------
# Pool
# -------------------------------------------------

class AgentPool:
    """Manages all workers across all roles: enqueue, find idle, dispatch."""

    def __init__(self):
        self._workers: dict[str, AgentWorker] = {}       # worker_id -> worker
        self._role_workers: dict[str, list[str]] = {}    # role -> [worker_ids]

    def add_worker(self, worker: AgentWorker) -> None:
        self._workers[worker.worker_id] = worker
        self._role_workers.setdefault(worker.role, []).append(worker.worker_id)
        logger.debug(
            "Added worker %s (role=%s, shift=%s-%s)",
            worker.worker_id, worker.role,
            worker.schedule.shift_start, worker.schedule.shift_end,
        )

    def get_worker(self, worker_id: str) -> AgentWorker:
        return self._workers[worker_id]

    def get_all_workers(self) -> list[AgentWorker]:
        return list(self._workers.values())

    def enqueue_case(self, role: str, case_id: str) -> str | None:
        """
        Add a case to the shortest queue among workers of this role.
        Returns the worker_id that received it, or None if the role is empty.
        """
        worker_ids = self._role_workers.get(role, [])
        if not worker_ids:
            logger.error("No workers for role %s", role)
            return None

        best = min(worker_ids, key=lambda wid: len(self._workers[wid].queue))
        worker = self._workers[best]
        worker.queue.append(case_id)
        logger.debug(
            "%s queued in %s (queue depth: %d)",
            case_id, best, len(worker.queue),
        )
        return best

    def find_idle_worker(self, role: str, sim_time: float) -> AgentWorker | None:
        """Find an idle, on-shift worker of this role that has work queued."""
        for wid in self._role_workers.get(role, []):
            worker = self._workers[wid]
            if worker.is_idle and worker.is_on_shift(sim_time) and worker.queue:
                return worker
        return None

    def find_any_dispatchable(self, sim_time: float) -> AgentWorker | None:
        """Find any idle, on-shift worker across all roles that has work queued."""
        for worker in self._workers.values():
            if worker.is_idle and worker.is_on_shift(sim_time) and worker.queue:
                return worker
        return None

    def get_queue_stats(self) -> dict:
        """Queue depths and busyness, for monitoring."""
        return {
            wid: {
                "queue_depth": len(w.queue),
                "is_idle": w.is_idle,
                "current_case": w.current_case,
            }
            for wid, w in self._workers.items()
        }
