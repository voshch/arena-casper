"""CASPER goal graphs, read from the goal JSON."""

from __future__ import annotations

import dataclasses
import graphlib
import json
import math
import pathlib
from collections.abc import Mapping

Constraint = bool | float

TASK_KEYS = frozenset({"id", "context", "required_skills", "required_constraints", "duration", "priority", "subgoal_level", "is_end_goal", "time_to_clean", "at_end"})
_GOAL_KEYS = frozenset({"tasks", "dependencies"})


@dataclasses.dataclass(frozen=True)
class GoalTask:
    id: str
    context: str
    required_skills: Mapping[str, float]
    constraints: Mapping[str, Constraint]
    workspaces: tuple[str, ...] | None
    duration: float
    priority: float
    subgoal_level: int | None = None
    end_goal: bool = False
    time_to_clean: bool | None = None
    at_end: bool = False

    @property
    def supervisor_only(self) -> bool:
        return self.end_goal or self.subgoal_level == 1

    @property
    def awaits_clearance(self) -> bool:
        """Held until every subgoal_level 1 task outside at_end is done."""
        return self.time_to_clean is False

    @property
    def releases_clearance(self) -> bool:
        return self.subgoal_level == 1 and not self.at_end


@dataclasses.dataclass(frozen=True)
class Goal:
    tasks: Mapping[str, GoalTask]
    dependencies: tuple[tuple[str, str], ...]

    def predecessors(self, task_id: str) -> tuple[str, ...]:
        return tuple(source for source, target in self.dependencies if target == task_id)


def _number(task_id: str, key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"task {task_id!r}: {key} must be a number, got {value!r}")
    return float(value)


def _flag(task_id: str, key: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"task {task_id!r}: {key} must be true or false, got {value!r}")
    return value


def _constraint(task_id: str, key: str, value: object) -> Constraint:
    if isinstance(value, bool):
        return value
    return _number(task_id, f"required_constraints.{key}", value)


def _workspaces(task_id: str, value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(name, str) for name in value):
        return tuple(value)
    raise ValueError(f"task {task_id!r}: required_constraints.workspace must be a workspace name or a list of them, got {value!r}")


def parse_task(raw: Mapping[str, object]) -> GoalTask:
    task_id = raw.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"every task needs a string id, got {raw!r}")
    if unknown := sorted(set(raw) - TASK_KEYS):
        raise ValueError(f"task {task_id!r}: unknown keys {unknown}, allowed: {sorted(TASK_KEYS)}")
    context = raw.get("context")
    if not isinstance(context, str):
        raise ValueError(f"task {task_id!r}: context must be a string, got {context!r}")
    skills = raw.get("required_skills", {})
    constraints = raw.get("required_constraints", {})
    if not isinstance(skills, dict) or not isinstance(constraints, dict):
        raise ValueError(f"task {task_id!r}: required_skills and required_constraints must be mappings")
    subgoal_level = raw.get("subgoal_level")
    if subgoal_level is not None and (isinstance(subgoal_level, bool) or not isinstance(subgoal_level, int)):
        raise ValueError(f"task {task_id!r}: subgoal_level must be an integer, got {subgoal_level!r}")
    time_to_clean = raw.get("time_to_clean")
    return GoalTask(
        id=task_id,
        context=context,
        required_skills={skill: _number(task_id, f"required_skills.{skill}", level) for skill, level in skills.items()},
        constraints={key: _constraint(task_id, key, value) for key, value in constraints.items() if key != "workspace"},
        workspaces=_workspaces(task_id, constraints.get("workspace")),
        duration=_number(task_id, "duration", raw.get("duration", 0.0)),
        priority=_number(task_id, "priority", raw.get("priority", math.inf)),
        subgoal_level=subgoal_level,
        end_goal=_flag(task_id, "is_end_goal", raw.get("is_end_goal", False)),
        time_to_clean=None if time_to_clean is None else _flag(task_id, "time_to_clean", time_to_clean),
        at_end=_flag(task_id, "at_end", raw.get("at_end", False)),
    )


def build_goal(tasks: list[GoalTask], dependencies: list[tuple[str, str]]) -> Goal:
    """A goal from parsed tasks and [before, after] pairs, ValueError on a duplicate, an unknown task, a cycle or clean tasks nothing releases."""
    by_id: dict[str, GoalTask] = {}
    for task in tasks:
        if task.id in by_id:
            raise ValueError(f"goal: task {task.id!r} listed twice")
        by_id[task.id] = task
    for pair in dependencies:
        for name in pair:
            if name not in by_id:
                raise ValueError(f"goal: dependency {list(pair)!r} names unknown task {name!r}")
    sorter = graphlib.TopologicalSorter({task_id: [] for task_id in by_id})
    for source, target in dependencies:
        sorter.add(target, source)
    try:
        sorter.prepare()
    except graphlib.CycleError as e:
        raise ValueError(f"goal: dependency cycle {e.args[1]}") from e
    if any(task.awaits_clearance for task in tasks) and not any(task.releases_clearance for task in tasks):
        raise ValueError("goal: tasks with time_to_clean false are never released without a subgoal_level 1 task outside at_end")
    return Goal(tasks=by_id, dependencies=tuple(dependencies))


def parse_goal(raw: Mapping[str, object]) -> Goal:
    if unknown := sorted(set(raw) - _GOAL_KEYS):
        raise ValueError(f"goal: unknown keys {unknown}, allowed: {sorted(_GOAL_KEYS)}")
    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("goal: tasks must be a non-empty list")
    tasks = []
    for entry in raw_tasks:
        if not isinstance(entry, dict):
            raise ValueError(f"goal: every task must be a mapping, got {entry!r}")
        tasks.append(parse_task(entry))
    raw_dependencies = raw.get("dependencies", [])
    if not isinstance(raw_dependencies, list):
        raise ValueError("goal: dependencies must be a list of [before, after] pairs")
    dependencies: list[tuple[str, str]] = []
    for pair in raw_dependencies:
        if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(name, str) for name in pair)):
            raise ValueError(f"goal: dependency {pair!r} must be a [before, after] pair of task ids")
        dependencies.append((pair[0], pair[1]))
    return build_goal(tasks, dependencies)


def load_goal(path: str | pathlib.Path) -> Goal:
    try:
        return parse_goal(json.loads(pathlib.Path(path).read_text()))
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e
