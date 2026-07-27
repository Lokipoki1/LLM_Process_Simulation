"""
agent_pool.py
-------------
Agent pool that manages workers, their schedules, and case queues.

Each role (junior_clerk, senior_clerk, credit_officer) can have
one or more workers. Each worker has:
  - A work schedule (shift_start, shift_end in hours)
  - A FIFO queue of pending cases
  - A busyness state (idle or busy until time T)

This simulates a real office: when all junior clerks are busy,
new cases wait in their queue — creating realistic waiting times
that match the BPIC data.
"""

from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..agents.base_agent import BaseAgent

logger = logging.getLogger("bps.agent_pool")


@dataclass
class WorkSchedule:
    """Working hours for an agent. Times are in hours (0-24)."""
    shift_start: float = 8.0    # 08:00
    shift_end: float   = 17.0   # 17:00
    days_off: set[int] = field(default_factory=lambda: {5, 6})  # Sat=5, Sun=6


@dataclass
class AgentWorker:
    """
    A single worker instance. One role can have multiple workers
    (e.g., 2 junior clerks, 1 senior clerk, 1 credit officer).
    """
    worker_id: str                  # "junior_clerk_1"
    role: str                       # "junior_clerk"
    agent: BaseAgent                # the callable LLM agent
    schedule: WorkSchedule
    queue: deque = field(default_factory=deque)      # pending case_ids (FIFO)
    busy_until: float = 0.0         # sim time when current task finishes
    current_case: str | None = None # case_id being processed, or None if idle

    @property
    def is_idle(self) -> bool:
        return self.current_case is None

    def is_on_shift(self, sim_time: float) -> bool:
        """Check if the worker is within working hours at the given sim time."""
        dt = datetime.fromtimestamp(sim_time, tz=timezone.utc)
        if dt.weekday() in self.schedule.days_off:
            return False
        hour = dt.hour + dt.minute / 60.0
        return self.schedule.shift_start <= hour < self.schedule.shift_end

    def next_shift_start(self, sim_time: float) -> float:
        """
        Returns the sim time of the next shift start after sim_time.
        Skips weekends and holidays.
        """
        dt = datetime.fromtimestamp(sim_time, tz=timezone.utc)

        # If currently before shift_start today and it's a workday, shift starts today
        hour = dt.hour + dt.minute / 60.0
        if dt.weekday() not in self.schedule.days_off and hour < self.schedule.shift_start:
            start_dt = dt.replace(
                hour=int(self.schedule.shift_start),
                minute=int((self.schedule.shift_start % 1) * 60),
                second=0, microsecond=0,
            )
            return start_dt.timestamp()

        # Otherwise, find the next workday
        from datetime import timedelta
        candidate = dt + timedelta(days=1)
        for _ in range(7):  # max 7 days to find a workday
            if candidate.weekday() not in self.schedule.days_off:
                start_dt = candidate.replace(
                    hour=int(self.schedule.shift_start),
                    minute=int((self.schedule.shift_start % 1) * 60),
                    second=0, microsecond=0,
                )
                return start_dt.timestamp()
            candidate += timedelta(days=1)

        # Fallback (shouldn't happen unless all days are off)
        return sim_time + 86400

    def next_shift_end(self, sim_time: float) -> float:
        """Returns the sim time when the current shift ends."""
        dt = datetime.fromtimestamp(sim_time, tz=timezone.utc)
        end_dt = dt.replace(
            hour=int(self.schedule.shift_end),
            minute=int((self.schedule.shift_end % 1) * 60),
            second=0, microsecond=0,
        )
        if end_dt.timestamp() <= sim_time:
            # Already past shift end — return next day's end
            from datetime import timedelta
            end_dt += timedelta(days=1)
        return end_dt.timestamp()


class AgentPool:
    """
    Manages all workers across all roles.

    Provides methods to enqueue cases, find idle workers,
    and dispatch work.
    """

    def __init__(self):
        self._workers: dict[str, AgentWorker] = {}          # worker_id → worker
        self._role_workers: dict[str, list[str]] = {}        # role → [worker_ids]

    def add_worker(self, worker: AgentWorker) -> None:
        self._workers[worker.worker_id] = worker
        if worker.role not in self._role_workers:
            self._role_workers[worker.role] = []
        self._role_workers[worker.role].append(worker.worker_id)
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
        Returns the worker_id whose queue received the case, or None.
        """
        worker_ids = self._role_workers.get(role, [])
        if not worker_ids:
            logger.error("No workers for role %s", role)
            return None

        # Pick the worker with the shortest queue
        best = min(worker_ids, key=lambda wid: len(self._workers[wid].queue))
        worker = self._workers[best]
        worker.queue.append(case_id)
        logger.debug(
            "%s queued in %s (queue depth: %d)",
            case_id, best, len(worker.queue),
        )
        return best

    def find_idle_worker(self, role: str, sim_time: float) -> AgentWorker | None:
        """
        Find an idle worker of the given role who is on shift
        and has cases in their queue.
        """
        for wid in self._role_workers.get(role, []):
            worker = self._workers[wid]
            if worker.is_idle and worker.is_on_shift(sim_time) and worker.queue:
                return worker
        return None

    def find_any_dispatchable(self, sim_time: float) -> AgentWorker | None:
        """Find any idle worker across all roles who has work to do."""
        for worker in self._workers.values():
            if worker.is_idle and worker.is_on_shift(sim_time) and worker.queue:
                return worker
        return None

    def get_queue_stats(self) -> dict:
        """Return queue depths and busyness for monitoring."""
        stats = {}
        for wid, worker in self._workers.items():
            stats[wid] = {
                "queue_depth": len(worker.queue),
                "is_idle": worker.is_idle,
                "current_case": worker.current_case,
            }
        return stats
