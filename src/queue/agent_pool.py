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
wait in a queue — producing the waiting times that make cycle times
comparable to the BPIC data.
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


# ─────────────────────────────────────────────
# Schedule helpers (module level so the engine can use them
# without instantiating a throwaway worker)
# ─────────────────────────────────────────────

def is_on_shift(sim_time: float, schedule: WorkSchedule) -> bool:
    """True if sim_time falls inside the given schedule's working hours."""
    dt = datetime.fromtimestamp(sim_time, tz=timezone.utc)
    if dt.weekday() in schedule.days_off:
        return False
    hour = dt.hour + dt.minute / 60.0
    return schedule.shift_start <= hour < schedule.shift_end


def compute_next_shift_start(sim_time: float, schedule: WorkSchedule) -> float:
    """
    Return the sim time of the next shift start at or after sim_time.
    Skips days off.
    """
    dt = datetime.fromtimestamp(sim_time, tz=timezone.utc)
    hour = dt.hour + dt.minute / 60.0

    def _at_shift_start(day: datetime) -> float:
        return day.replace(
            hour=int(schedule.shift_start),
            minute=int((schedule.shift_start % 1) * 60),
            second=0, microsecond=0,
        ).timestamp()

    # Before today's shift start on a working day → starts today
    if dt.weekday() not in schedule.days_off and hour < schedule.shift_start:
        return _at_shift_start(dt)

    # Otherwise find the next working day
    candidate = dt + timedelta(days=1)
    for _ in range(7):
        if candidate.weekday() not in schedule.days_off:
            return _at_shift_start(candidate)
        candidate += timedelta(days=1)

    # Fallback (only reachable if every day is a day off)
    return sim_time + 86400


# ─────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────

@dataclass
class AgentWorker:
    """
    A single worker instance. One role can have several workers
    (e.g. 2 junior clerks, 1 senior clerk, 1 credit officer).

    The agent behaviour itself lives in the ProcessStepExecutor —
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


# ─────────────────────────────────────────────
# Pool
# ─────────────────────────────────────────────

class AgentPool:
    """Manages all workers across all roles: enqueue, find idle, dispatch."""

    def __init__(self):
        self._workers: dict[str, AgentWorker] = {}       # worker_id → worker
        self._role_workers: dict[str, list[str]] = {}    # role → [worker_ids]

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
