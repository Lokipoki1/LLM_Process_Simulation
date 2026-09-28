"""
agent_pool.py
-------------
Resource pool: workers, their schedules, and one shared FIFO queue per
role (M/M/c style - any idle worker of the role takes the next case).

compute_finish_time() spreads work across shift windows: 2 hours started
at 16:50 uses the 10 minutes left today and finishes at 09:50 tomorrow.
"""

from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("bps.agent_pool")


@dataclass
class WorkSchedule:
    """Working hours for a role. Times are in hours (0-24)."""
    shift_start: float = 8.0
    shift_end: float = 17.0
    days_off: set[int] = field(default_factory=lambda: {5, 6})  # Sat=5, Sun=6

    @property
    def daily_seconds(self) -> float:
        return (self.shift_end - self.shift_start) * 3600


# -------------------------------------------------
# Schedule helpers
# -------------------------------------------------

def is_on_shift(sim_time: float, schedule: WorkSchedule) -> bool:
    """True if sim_time falls inside the schedule's working hours."""
    dt = datetime.fromtimestamp(sim_time, tz=timezone.utc)
    if dt.weekday() in schedule.days_off:
        return False
    hour = dt.hour + dt.minute / 60.0
    return schedule.shift_start <= hour < schedule.shift_end


def _at_hour(day: datetime, hour_float: float) -> float:
    return day.replace(
        hour=int(hour_float),
        minute=int((hour_float % 1) * 60),
        second=0, microsecond=0,
    ).timestamp()


def compute_next_shift_start(sim_time: float, schedule: WorkSchedule) -> float:
    """Sim time of the next shift start at or after sim_time. Skips days off."""
    dt = datetime.fromtimestamp(sim_time, tz=timezone.utc)
    hour = dt.hour + dt.minute / 60.0

    if dt.weekday() not in schedule.days_off and hour < schedule.shift_start:
        return _at_hour(dt, schedule.shift_start)

    candidate = dt + timedelta(days=1)
    for _ in range(7):
        if candidate.weekday() not in schedule.days_off:
            return _at_hour(candidate, schedule.shift_start)
        candidate += timedelta(days=1)

    return sim_time + 86400


def shift_end_timestamp(sim_time: float, schedule: WorkSchedule) -> float:
    """Timestamp of the shift end on the same calendar day as sim_time."""
    return _at_hour(
        datetime.fromtimestamp(sim_time, tz=timezone.utc),
        schedule.shift_end,
    )


def compute_finish_time(
    start: float,
    work_seconds: float,
    schedule: WorkSchedule,
    max_iterations: int = 60,
) -> float:
    """
    Spread `work_seconds` of work across shift windows starting at
    `start`, and return the finish time.
    """
    if work_seconds <= 0:
        return start

    t = start
    remaining = work_seconds

    if not is_on_shift(t, schedule):
        t = compute_next_shift_start(t, schedule)

    for _ in range(max_iterations):
        end_of_shift = shift_end_timestamp(t, schedule)
        available = end_of_shift - t
        if remaining <= available:
            return t + remaining
        remaining -= available
        t = compute_next_shift_start(end_of_shift, schedule)

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
    """A single resource: a role, a schedule and a busy flag."""
    worker_id: str
    role: str
    schedule: WorkSchedule
    current_case: str | None = None

    @property
    def is_idle(self) -> bool:
        return self.current_case is None

    def is_on_shift(self, sim_time: float) -> bool:
        return is_on_shift(sim_time, self.schedule)

    def next_shift_start(self, sim_time: float) -> float:
        return compute_next_shift_start(sim_time, self.schedule)

    def finish_time_for(self, start: float, work_seconds: float) -> float:
        return compute_finish_time(start, work_seconds, self.schedule)


# -------------------------------------------------
# Pool
# -------------------------------------------------

class AgentPool:
    """
    Manages workers and one FIFO queue per role.

        enqueue_case(role, case_id)  case joins the role's shared line
        claim_next(sim_time)         an idle on-shift worker takes the
                                     head of its role's line
    """

    def __init__(self):
        self._workers: dict[str, AgentWorker] = {}
        self._role_workers: dict[str, list[str]] = {}
        self._role_queues: dict[str, deque] = {}

    # -- Registration --------------------------

    def add_worker(self, worker: AgentWorker) -> None:
        self._workers[worker.worker_id] = worker
        self._role_workers.setdefault(worker.role, []).append(worker.worker_id)
        self._role_queues.setdefault(worker.role, deque())
        logger.debug(
            "Added worker %s (role=%s, shift=%s-%s)",
            worker.worker_id, worker.role,
            worker.schedule.shift_start, worker.schedule.shift_end,
        )

    def get_worker(self, worker_id: str) -> AgentWorker:
        return self._workers[worker_id]

    def get_all_workers(self) -> list[AgentWorker]:
        return list(self._workers.values())

    # -- Queueing ------------------------------

    def enqueue_case(self, role: str, case_id: str) -> bool:
        """Append a case to the role's shared FIFO queue."""
        if role not in self._role_queues:
            logger.error("No workers registered for role %s", role)
            return False
        self._role_queues[role].append(case_id)
        logger.debug(
            "%s queued for %s (role queue depth: %d)",
            case_id, role, len(self._role_queues[role]),
        )
        return True

    def claim_next(self, sim_time: float) -> tuple[AgentWorker, str] | None:
        """
        Give the head of a role's queue to an idle, on-shift worker of that
        role and mark it busy. Returns (worker, case_id) or None.
        """
        for role, queue in self._role_queues.items():
            if not queue:
                continue
            for wid in self._role_workers.get(role, []):
                worker = self._workers[wid]
                if worker.is_idle and worker.is_on_shift(sim_time):
                    case_id = queue.popleft()
                    worker.current_case = case_id
                    return worker, case_id
        return None

    def release(self, worker_id: str) -> None:
        """Mark a worker idle again."""
        self._workers[worker_id].current_case = None

    # -- Shift coordination --------------------

    def workers_needing_shift_wakeup(self, sim_time: float) -> list[AgentWorker]:
        """Idle, off-shift workers whose role has cases waiting."""
        out = []
        for role, queue in self._role_queues.items():
            if not queue:
                continue
            for wid in self._role_workers.get(role, []):
                worker = self._workers[wid]
                if worker.is_idle and not worker.is_on_shift(sim_time):
                    out.append(worker)
        return out

    # -- Monitoring ----------------------------

    def queue_depth(self, role: str) -> int:
        return len(self._role_queues.get(role, ()))

    def total_queued(self) -> int:
        return sum(len(q) for q in self._role_queues.values())

    def get_queue_stats(self) -> dict:
        return {
            "role_queues": {r: len(q) for r, q in self._role_queues.items()},
            "workers": {
                wid: {"is_idle": w.is_idle, "current_case": w.current_case}
                for wid, w in self._workers.items()
            },
        }
