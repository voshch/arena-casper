"""The run of an allocated CASPER goal: which task each member does next."""

from __future__ import annotations

import dataclasses
import enum

from casper.allocation import Allocation
from casper.goal import Goal


class Status(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"


@dataclasses.dataclass
class TaskState:
    owner: str
    fallbacks: tuple[str, ...]
    status: Status = Status.PENDING
    failed_by: set[str] = dataclasses.field(default_factory=set)


class Schedule:
    def __init__(self, goal: Goal, allocation: Allocation) -> None:
        self.goal = goal
        self.states = {task_id: TaskState(owner=assignment.member, fallbacks=assignment.fallbacks) for task_id, assignment in allocation.assignments.items()}

    @property
    def cleared(self) -> bool:
        """Whether the tasks awaiting clearance may run."""
        return all(self.states[task_id].status is Status.DONE for task_id, task in self.goal.tasks.items() if task.releases_clearance)

    def ready(self, task_id: str) -> bool:
        if self.states[task_id].status is not Status.PENDING:
            return False
        if self.goal.tasks[task_id].awaits_clearance and not self.cleared:
            return False
        return all(self.states[before].status is Status.DONE for before in self.goal.predecessors(task_id))

    def running(self, member: str) -> str | None:
        return next((task_id for task_id, state in self.states.items() if state.owner == member and state.status is Status.RUNNING), None)

    def next_for(self, member: str) -> str | None:
        """The member's ready task of the lowest priority number, None while it runs one or has none ready."""
        if self.running(member) is not None:
            return None
        ready = [task_id for task_id, state in self.states.items() if state.owner == member and self.ready(task_id)]
        return min(ready, key=lambda task_id: self.goal.tasks[task_id].priority, default=None)

    def start(self, task_id: str) -> None:
        if not self.ready(task_id):
            raise ValueError(f"task {task_id!r} is not ready")
        self.states[task_id].status = Status.RUNNING

    def finish(self, task_id: str) -> None:
        self._running(task_id).status = Status.DONE

    def interrupt(self, task_id: str) -> None:
        """Put a running task back to wait for its owner."""
        self._running(task_id).status = Status.PENDING

    def fail(self, task_id: str) -> str | None:
        """Hand a failed task to its first fallback that has not failed it, None when none is left."""
        state = self._running(task_id)
        state.failed_by.add(state.owner)
        state.status = Status.PENDING
        successor = next((name for name in state.fallbacks if name not in state.failed_by), None)
        if successor is not None:
            state.owner = successor
        return successor

    @property
    def done(self) -> bool:
        return all(state.status is Status.DONE for state in self.states.values())

    def _running(self, task_id: str) -> TaskState:
        state = self.states[task_id]
        if state.status is not Status.RUNNING:
            raise ValueError(f"task {task_id!r} is {state.status.value}, not running")
        return state
