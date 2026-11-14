"""The Arena team file: CASPER members keyed by robot or pedestrian name, can_move and can_manipulate taken from the robot or fixed for a pedestrian."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping

import yaml
from casper.skills import Team, member

DERIVED_CONSTRAINTS = frozenset({"can_move", "can_manipulate"})
PEDESTRIAN_CONSTRAINTS = {"can_move": True, "can_manipulate": False}

_TEAM_KEYS = frozenset({"supervisor", "supervisor_team", "members"})
_MEMBER_KEYS = frozenset({"kind", "workspace", "constraints", "skills"})
_KINDS = ("robot", "pedestrian")


@dataclasses.dataclass(frozen=True)
class TeamFile:
    """The CASPER team and the names of its members that are pedestrians."""

    team: Team
    pedestrians: frozenset[str]


def _skill_weights(name: str, raw: object) -> dict[str, dict[str, list[float]]]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"member {name!r}: skills must map a context to {{skill: [level, preference rank]}}")
    for context, skills in raw.items():
        if not isinstance(skills, dict):
            raise ValueError(f"member {name!r}: skills.{context} must map a skill to [level, preference rank]")
        for skill, pair in skills.items():
            if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(v, int | float) and not isinstance(v, bool) for v in pair)):
                raise ValueError(f"member {name!r}: skills.{context}.{skill} must be [level, preference rank], got {pair!r}")
    return raw


def _kind(name: str, raw: object) -> str:
    kind = raw.get("kind", "robot") if isinstance(raw, dict) else "robot"
    if kind not in _KINDS:
        raise ValueError(f"member {name!r}: kind must be one of {list(_KINDS)}, got {kind!r}")
    return kind


def _profile(name: str, raw: object, derived: Mapping[str, bool], source: str) -> tuple[dict[str, dict[str, list[float]]], dict[str, bool | float], list[str] | None]:
    if not isinstance(raw, dict):
        raise ValueError(f"member {name!r}: expected a mapping, got {raw!r}")
    if unknown := sorted(set(raw) - _MEMBER_KEYS):
        raise ValueError(f"member {name!r}: unknown keys {unknown}, allowed: {sorted(_MEMBER_KEYS)}")
    constraints = raw.get("constraints", {})
    if not isinstance(constraints, dict):
        raise ValueError(f"member {name!r}: constraints must be a mapping")
    if stated := sorted(DERIVED_CONSTRAINTS & set(constraints)):
        raise ValueError(f"member {name!r}: {stated} come from {source}, remove them from the team file")
    for key, value in constraints.items():
        if not isinstance(value, bool | int | float):
            raise ValueError(f"member {name!r}: constraint {key} must be a boolean or a number, got {value!r}")
    workspace = raw.get("workspace")
    if workspace is not None and not (isinstance(workspace, list) and all(isinstance(zone, str) for zone in workspace)):
        raise ValueError(f"member {name!r}: workspace must be a list of zone names, got {workspace!r}")
    return _skill_weights(name, raw.get("skills")), {**constraints, **derived}, workspace


def parse_team(raw: object, robots: Mapping[str, Mapping[str, bool]]) -> TeamFile:
    """`robots` maps each robot of the env to its derived constraints, members of kind `pedestrian` are no robots."""
    if not isinstance(raw, dict):
        raise ValueError(f"team: expected a mapping, got {raw!r}")
    if unknown := sorted(set(raw) - _TEAM_KEYS):
        raise ValueError(f"team: unknown keys {unknown}, allowed: {sorted(_TEAM_KEYS)}")
    members = raw.get("members")
    if not isinstance(members, dict) or not members:
        raise ValueError("team: members must map robot or pedestrian names to their profiles")
    kinds = {name: _kind(name, profile) for name, profile in members.items()}
    pedestrians = frozenset(name for name, kind in kinds.items() if kind == "pedestrian")
    if both := sorted(pedestrians & set(robots)):
        raise ValueError(f"team: members {both} are pedestrians in the team file and robots of this env")
    if missing := sorted(set(members) - pedestrians - set(robots)):
        raise ValueError(f"team: members {missing} are not robots of this env, robots: {sorted(robots)}")
    supervisor = raw.get("supervisor")
    if not isinstance(supervisor, str) or supervisor not in members:
        raise ValueError(f"team: supervisor must name a member, got {supervisor!r}, members: {sorted(members)}")
    supervisor_team = raw.get("supervisor_team", True)
    if not isinstance(supervisor_team, bool):
        raise ValueError(f"team: supervisor_team must be true or false, got {supervisor_team!r}")
    built = {}
    for name, profile in members.items():
        derived, source = (PEDESTRIAN_CONSTRAINTS, "the pedestrian kind") if name in pedestrians else (robots[name], "the robot's adapters")
        built[name] = member(name, *_profile(name, profile, derived, source))
    team = Team(members=built, regular=tuple(name for name in built if supervisor_team or name != supervisor), supervisors=(supervisor,))
    return TeamFile(team=team, pedestrians=pedestrians)


def load_team(path: str | pathlib.Path, robots: Mapping[str, Mapping[str, bool]]) -> TeamFile:
    try:
        return parse_team(yaml.safe_load(pathlib.Path(path).read_text()), robots)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e
