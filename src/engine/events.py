"""
events.py
---------
Event types and priority queue for the discrete event simulation engine.

The simulation is driven by events ordered by simulation time.
Each event represents something that happens at a specific moment:
a case arriving, an agent finishing a task, a shift starting/ending.
"""

from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    CASE_ARRIVAL   = "CASE_ARRIVAL"     # New case enters the system
    STEP_COMPLETE  = "STEP_COMPLETE"    # Agent finished processing one step
    SHIFT_START    = "SHIFT_START"      # Agent's working day begins
    SHIFT_END      = "SHIFT_END"        # Agent's working day ends


@dataclass(order=True)
class SimEvent:
    """
    A single event in the discrete event simulation.

    Events are ordered by time (earliest first). When two events have
    the same time, priority breaks the tie (lower = higher priority).
    """
    time: float                                        # simulation time (unix timestamp)
    priority: int        = field(compare=True)         # lower = processed first at same time
    event_type: EventType = field(compare=False)
    case_id: str | None  = field(compare=False, default=None)
    worker_id: str | None = field(compare=False, default=None)
    data: dict           = field(compare=False, default_factory=dict)

    # Priority constants (lower number = higher priority)
    PRIORITY_SHIFT   = 0    # shift events processed first
    PRIORITY_COMPLETE = 1   # completions before arrivals (free up agents)
    PRIORITY_ARRIVAL  = 2   # arrivals last


class EventQueue:
    """
    Priority queue for simulation events, ordered by time.

    Wraps heapq for O(log n) push/pop. Events at the same time
    are processed in priority order (shift > complete > arrival).
    """

    def __init__(self):
        self._heap: list[SimEvent] = []

    def push(self, event: SimEvent) -> None:
        heapq.heappush(self._heap, event)

    def pop(self) -> SimEvent:
        return heapq.heappop(self._heap)

    def peek(self) -> SimEvent | None:
        return self._heap[0] if self._heap else None

    @property
    def empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)

    # ── Convenience factories ──────────────────────

    def schedule_arrival(self, time: float, case_id: str, case_data: dict) -> None:
        self.push(SimEvent(
            time=time,
            priority=SimEvent.PRIORITY_ARRIVAL,
            event_type=EventType.CASE_ARRIVAL,
            case_id=case_id,
            data=case_data,
        ))

    def schedule_step_complete(
        self, time: float, case_id: str, worker_id: str, step_result: dict
    ) -> None:
        self.push(SimEvent(
            time=time,
            priority=SimEvent.PRIORITY_COMPLETE,
            event_type=EventType.STEP_COMPLETE,
            case_id=case_id,
            worker_id=worker_id,
            data=step_result,
        ))

    def schedule_shift(
        self, time: float, worker_id: str, event_type: EventType
    ) -> None:
        self.push(SimEvent(
            time=time,
            priority=SimEvent.PRIORITY_SHIFT,
            event_type=event_type,
            worker_id=worker_id,
        ))
