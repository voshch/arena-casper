"""CASPER++ task allocation: who may take each task of a goal, and who does."""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Mapping

from casper.goal import Goal, GoalTask
from casper.skills import Member, Team

SKILL_WEIGHT = 0.7
"""Share of skill against preference in balance_skill_preference."""

Ranking = dict[str, tuple[tuple[str, float], ...]]
"""Per task the qualifying members with their scores, best first, ties in team order."""


class Mode(enum.StrEnum):
    SKILL = "skill"
    PREFERENCE = "preference"
    BALANCE_SKILL_PREFERENCE = "balance_skill_preference"
    BALANCE_WORKLOAD = "balance_workload"


@dataclasses.dataclass
class Assignment:
    member: str
    score: float
    fallbacks: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Allocation:
    mode: Mode
    assignments: Mapping[str, Assignment]
    ranking: Ranking
    initial: Mapping[str, Assignment]


class AllocationError(ValueError):
    def __init__(self, unassigned: list[str]) -> None:
        super().__init__(f"no team member qualifies for tasks {unassigned}")
        self.unassigned = unassigned


def qualifies(task: GoalTask, member: Member, mode: Mode) -> float | None:
    """The member's score for the task, None when it may not take it."""
    if task.context not in member.contexts:
        return None
    if task.workspaces is not None:
        if member.workspaces is None or not {zone.lower() for zone in task.workspaces} <= {zone.lower() for zone in member.workspaces}:
            return None
    ranked = sum(1 for by_context in member.preference.values() if task.context in by_context)
    score = 0.0
    for skill, minimum in task.required_skills.items():
        level = member.level.get(skill, {}).get(task.context, 0.0)
        if level < minimum:
            return None
        rank = member.preference.get(skill, {}).get(task.context)
        preference = (ranked - rank + 1) / ranked if rank is not None and rank > 0 and ranked > 0 else 0.0
        if mode is Mode.SKILL:
            score += level
        elif mode is Mode.PREFERENCE:
            score += preference
        elif mode is Mode.BALANCE_SKILL_PREFERENCE:
            score += SKILL_WEIGHT * level + (1 - SKILL_WEIGHT) * preference
    for key, required in task.constraints.items():
        if key in member.constraints and member.constraints[key] < required:
            return None
    if task.required_skills:
        score /= len(task.required_skills)
    return score


def rank(goal: Goal, team: Team, mode: Mode) -> Ranking:
    ranking: Ranking = {}
    for task_id, task in goal.tasks.items():
        scored = [(member.name, score) for member in team.candidates(task.supervisor_only) if (score := qualifies(task, member, mode)) is not None]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        ranking[task_id] = tuple(scored)
    return ranking


def fallbacks(scored: tuple[tuple[str, float], ...], selected: str, top_k: int) -> tuple[str, ...]:
    """The top_k ranked members other than the selected one."""
    return tuple(name for name, _ in scored if name != selected)[:top_k]


def assign(ranking: Ranking, top_k: int, pinned: Mapping[str, str] | None = None) -> tuple[dict[str, Assignment], dict[str, int], list[str]]:
    """Each task to its pinned member, else its best ranked one, a top-score tie to the one with fewer tasks so far, with the workload of each assigned member and the tasks nobody qualifies for."""
    assignments: dict[str, Assignment] = {}
    workload: dict[str, int] = {}
    unassigned: list[str] = []
    for task_id, scored in ranking.items():
        if pinned is not None and task_id in pinned:
            name = pinned[task_id]
            assignments[task_id] = Assignment(member=name, score=_score_of(scored, name), fallbacks=fallbacks(scored, name, top_k))
            workload[name] = workload.get(name, 0) + 1
            continue
        if not scored:
            unassigned.append(task_id)
            continue
        top = scored[0][1]
        name, score = min((pair for pair in scored if pair[1] == top), key=lambda pair: workload.get(pair[0], 0))
        assignments[task_id] = Assignment(member=name, score=score, fallbacks=fallbacks(scored, name, top_k))
        workload[name] = workload.get(name, 0) + 1
    return assignments, workload, unassigned


def _score_of(scored: tuple[tuple[str, float], ...], member: str) -> float:
    return next((score for name, score in scored if name == member), -math.inf)


def rebalance(assignments: dict[str, Assignment], ranking: Ranking, workload: dict[str, int], top_k: int) -> None:
    """Move tasks off the busiest assigned member while that narrows the gap, preferring the best fitting receiver, in place."""
    for _ in range(len(assignments) + 1):
        if not workload:
            return
        busiest = max(workload, key=lambda name: workload[name])
        least = min(workload, key=lambda name: workload[name])
        if workload[busiest] - workload[least] <= 1:
            return
        best: tuple[float, str, str] | None = None
        for task_id, assignment in assignments.items():
            if assignment.member != busiest:
                continue
            for target, _ in ranking[task_id]:
                if target == busiest or workload.get(target, 0) >= workload[busiest] - 1:
                    continue
                fit = _score_of(ranking[task_id], target)
                if best is None or fit > best[0]:
                    best = (fit, task_id, target)
        if best is None:
            return
        fit, task_id, target = best
        assignments[task_id] = Assignment(member=target, score=fit, fallbacks=fallbacks(ranking[task_id], target, top_k))
        workload[busiest] -= 1
        workload[target] = workload.get(target, 0) + 1


def allocate(goal: Goal, team: Team, mode: Mode, top_k: int) -> Allocation:
    """rank, assign and, in balance_workload, rebalance, AllocationError naming every task nobody qualifies for."""
    ranking = rank(goal, team, mode)
    assignments, workload, unassigned = assign(ranking, top_k)
    if unassigned:
        raise AllocationError(unassigned)
    initial = {task_id: dataclasses.replace(assignment) for task_id, assignment in assignments.items()}
    if mode is Mode.BALANCE_WORKLOAD:
        rebalance(assignments, ranking, workload, top_k)
    return Allocation(mode=mode, assignments=assignments, ranking=ranking, initial=initial)
