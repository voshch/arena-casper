"""Team members as CASPER sees them: skill levels and preference ranks per context, constraints and workspaces."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

from casper.goal import Constraint

UNSTATED = 0.1
"""Level and preference rank of a skill in a context its member lists it under no other."""

SkillWeights = Mapping[str, Mapping[str, Sequence[float]]]
"""{context: {skill: [level, preference rank]}}, rank 1 the most preferred."""


@dataclasses.dataclass(frozen=True)
class Member:
    name: str
    level: Mapping[str, Mapping[str, float]]
    preference: Mapping[str, Mapping[str, float]]
    constraints: Mapping[str, Constraint]
    workspaces: tuple[str, ...] | None

    @property
    def contexts(self) -> frozenset[str]:
        return frozenset(context for by_context in self.level.values() for context in by_context)


@dataclasses.dataclass(frozen=True)
class Team:
    members: Mapping[str, Member]
    regular: tuple[str, ...]
    """The members that compete for regular tasks, in ranking tie order."""
    supervisors: tuple[str, ...]
    """The members that compete for supervisor-only tasks, in ranking tie order."""

    def candidates(self, supervisor_only: bool) -> list[Member]:
        return [self.members[name] for name in (self.supervisors if supervisor_only else self.regular)]


def fill(level: Mapping[str, Mapping[str, float]], preference: Mapping[str, Mapping[str, float]], contexts: Sequence[str]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Every known skill in every context, at UNSTATED where the member did not state it."""
    filled_level = {skill: dict(by_context) for skill, by_context in level.items()}
    filled_preference = {skill: dict(by_context) for skill, by_context in preference.items()}
    for skill in filled_level:
        filled_preference.setdefault(skill, {})
        for context in contexts:
            filled_level[skill].setdefault(context, UNSTATED)
            filled_preference[skill].setdefault(context, UNSTATED)
    return filled_level, filled_preference


def member(name: str, skill_weights: SkillWeights, constraints: Mapping[str, Constraint], workspaces: Sequence[str] | None) -> Member:
    """A member from skill weights in the robot config form, the unstated contexts filled."""
    level: dict[str, dict[str, float]] = {}
    preference: dict[str, dict[str, float]] = {}
    for context, skills in skill_weights.items():
        for skill, pair in skills.items():
            if len(pair) != 2:
                raise ValueError(f"member {name!r}: {context}.{skill} must be [level, preference rank], got {list(pair)!r}")
            level.setdefault(skill, {})[context] = float(pair[0])
            preference.setdefault(skill, {})[context] = float(pair[1])
    level, preference = fill(level, preference, list(skill_weights))
    return Member(name=name, level=level, preference=preference, constraints=dict(constraints), workspaces=None if workspaces is None else tuple(workspaces))
