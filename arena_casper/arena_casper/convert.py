"""CASPER_MAS config folders turned into an Arena team file, a goal copy and the launch arguments."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import typing
from collections.abc import Mapping, Sequence

import attrs
import yaml
from casper.allocation import Mode
from casper.goal import parse_goal

from arena_casper.team import DERIVED_CONSTRAINTS

ROBOT_KEYS = frozenset({"skill_weights", "constraints", "workspace"})
ROBOT_SPAWN_KEYS = frozenset({"debug", "use_sim", "robot_model", "robot_id", "x_pos", "y_pos", "z_pos", "yaw", "spawn_delay", "actor_type", "color", "result_timeout"})
EXP_KEYS = frozenset({"agents", "supervisor", "supervisor_team", "optimizeMode", "top_k", "task"})
EXP_DROPPED_KEYS = frozenset({"debug", "use_sim", "party_duration", "world_name", "team_size", "guests"})


@attrs.frozen
class Conversion:
    team: dict[str, object]
    goal: dict[str, object]
    launch: dict[str, str]
    dropped: list[str]


def _mapping(path: pathlib.Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping, got {raw!r}")
    return raw


def _check_keys(path: pathlib.Path, raw: Mapping[str, object], known: frozenset[str], dropped: frozenset[str]) -> None:
    if unknown := sorted(set(raw) - known - dropped):
        raise ValueError(f"{path}: unknown keys {unknown}, known: {sorted(known | dropped)}")


def _robot_file(config: pathlib.Path, world: object, agent: str) -> pathlib.Path:
    candidates = [config / "robots" / str(world) / f"{agent}.yaml", config / "robots" / f"{agent}.yaml"] if world else [config / "robots" / f"{agent}.yaml"]
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise ValueError(f"agent {agent!r} has no robot file, looked for {[str(path) for path in candidates]}")
    return found


def _goal_file(config: pathlib.Path, task: object, override: str | pathlib.Path | None) -> pathlib.Path:
    if override is not None:
        return pathlib.Path(override)
    if not isinstance(task, str):
        raise ValueError(f"exps task must be a goal path, got {task!r}")
    relative = pathlib.Path(task)
    candidates = [config.parent / relative, config / pathlib.Path(*relative.parts[1:])]
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise ValueError(f"goal {task!r} not found, looked for {[str(path) for path in candidates]}, pass --goal")
    return found


def _rename(zones: Mapping[str, str], names: Sequence[str], used: set[str]) -> list[str]:
    used.update(name for name in names if name in zones)
    return [zones.get(name, name) for name in names]


def convert(
    config: str | pathlib.Path,
    exp: str,
    robots: Mapping[str, str],
    zones: Mapping[str, str] | None = None,
    goal: str | pathlib.Path | None = None,
) -> Conversion:
    """`robots` maps every agent of the experiment to its Arena robot name, `zones` renames workspaces."""
    zones = dict(zones or {})
    config = pathlib.Path(config)
    exp_path = config / "exps" / f"{exp}.yaml"
    if not exp_path.is_file():
        raise ValueError(f"experiment {exp!r} not found, looked for {exp_path}")
    raw = _mapping(exp_path)
    _check_keys(exp_path, raw, EXP_KEYS, EXP_DROPPED_KEYS)
    agents = raw.get("agents")
    if not isinstance(agents, list) or not agents or not all(isinstance(agent, str) for agent in agents):
        raise ValueError(f"{exp_path}: agents must be a list of robot config names")
    if unmapped := [agent for agent in agents if agent not in robots]:
        raise ValueError(f"agents {unmapped} have no --robot mapping, agents: {agents}")
    if extra := sorted(set(robots) - set(agents)):
        raise ValueError(f"--robot names {extra} are no agents of {exp!r}, agents: {agents}")
    if len(set(robots.values())) != len(robots):
        raise ValueError(f"--robot maps two agents to one Arena robot: {dict(robots)}")
    supervisor = raw.get("supervisor")
    if supervisor not in agents:
        raise ValueError(f"{exp_path}: supervisor {supervisor!r} is no agent, agents: {agents}")
    mode = raw.get("optimizeMode")
    if mode not in {m.value for m in Mode}:
        raise ValueError(f"{exp_path}: optimizeMode {mode!r} is none of {[m.value for m in Mode]}")
    dropped = [f"{exp_path.name}: {key}" for key in sorted(set(raw) & EXP_DROPPED_KEYS)]
    used: set[str] = set()
    members: dict[str, object] = {}
    for agent in agents:
        path = _robot_file(config, raw.get("world_name"), agent)
        profile = _mapping(path)
        _check_keys(path, profile, ROBOT_KEYS, ROBOT_SPAWN_KEYS)
        dropped += [f"{agent}: {key}" for key in sorted(set(profile) & ROBOT_SPAWN_KEYS)]
        if "skill_weights" not in profile:
            raise ValueError(f"{path}: skill_weights missing")
        member: dict[str, object] = {}
        workspace = profile.get("workspace")
        if workspace is not None:
            if not (isinstance(workspace, list) and all(isinstance(zone, str) for zone in workspace)):
                raise ValueError(f"{path}: workspace must be a list of names, got {workspace!r}")
            member["workspace"] = _rename(zones, workspace, used)
        constraints = profile.get("constraints") or {}
        if not isinstance(constraints, dict):
            raise ValueError(f"{path}: constraints must be a mapping")
        dropped += [f"{agent}: constraints.{key}" for key in sorted(set(constraints) & DERIVED_CONSTRAINTS)]
        if kept := {key: value for key, value in constraints.items() if key not in DERIVED_CONSTRAINTS}:
            member["constraints"] = kept
        member["skills"] = profile["skill_weights"]
        members[robots[agent]] = member
    goal_path = _goal_file(config, raw.get("task"), goal)
    renamed = copy.deepcopy(json.loads(goal_path.read_text()))
    for task in renamed.get("tasks", []):
        constraints = task.get("required_constraints", {})
        names = constraints.get("workspace")
        if isinstance(names, str):
            constraints["workspace"] = _rename(zones, [names], used)[0]
        elif isinstance(names, list):
            constraints["workspace"] = _rename(zones, names, used)
    if unused := sorted(set(zones) - used):
        raise ValueError(f"--zone names {unused} occur in no member workspace and no goal task")
    parse_goal(renamed)
    team: dict[str, object] = {"supervisor": robots[typing.cast("str", supervisor)], "supervisor_team": raw.get("supervisor_team", True), "members": members}
    launch = {"task.casper.mode": str(mode), "task.casper.top_k": str(raw.get("top_k", 3))}
    return Conversion(team=team, goal=renamed, launch=launch, dropped=dropped)


def write(conversion: Conversion, out_dir: str | pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, str]:
    """team.yaml and goal.json into out_dir, with the launch arguments line."""
    out = pathlib.Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    team, goal = out / "team.yaml", out / "goal.json"
    team.write_text(yaml.safe_dump(conversion.team, sort_keys=False))
    goal.write_text(json.dumps(conversion.goal, indent=2) + "\n")
    launch = {"task.casper.goal": str(goal), "task.casper.team": str(team), **conversion.launch}
    return team, goal, " ".join(f"{key}:={value}" for key, value in launch.items())


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> typing.NoReturn:
        self.print_usage(sys.stderr)
        sys.stderr.write(f"{self.prog}: error: {message}\nrun '{self.prog} --help' for the arguments and the files written\n")
        raise SystemExit(2)


def _pairs(parser: argparse.ArgumentParser, flag: str, values: Sequence[str]) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for value in values:
        left, sep, right = value.partition("=")
        if not sep or not left or not right:
            parser.error(f"{flag} {value!r} must look like <name in CASPER_MAS>=<name in Arena>")
        pairs[left] = right
    return pairs


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python -m arena_casper.convert",
        description=(
            "Turn a CASPER_MAS config folder into an Arena casper team: writes team.yaml (members keyed by Arena robot name) "
            "and goal.json (a copy with renamed zones) into --out, then prints the task.casper.* launch arguments. "
            "can_move and can_manipulate constraints and spawn settings (positions, models, delays) are dropped and listed, "
            "Arena derives the constraints from the robot. Agents without --robot, unknown keys and unused --zone names are errors."
        ),
    )
    parser.add_argument("config", help="CASPER_MAS configs folder holding robots/ and exps/")
    parser.add_argument("exp", help="experiment name, the file exps/<exp>.yaml without extension")
    parser.add_argument("--robot", action="append", default=[], metavar="HER_ID=ARENA_ROBOT", help="agent of the experiment and the Arena robot name that plays it, once per agent")
    parser.add_argument("--zone", action="append", default=[], metavar="HER_WORKSPACE=ZONE", help="rename a workspace to a zone of the Arena world, applied to members and goal")
    parser.add_argument("--goal", help="goal JSON to use instead of the exps task path")
    parser.add_argument("--out", default=".", help="directory for team.yaml and goal.json (default: current directory)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = sys.argv[1:] if argv is None else list(argv)
    if not arguments:
        parser.print_help()
        return 2
    args = parser.parse_args(arguments)
    robots = _pairs(parser, "--robot", args.robot)
    zones = _pairs(parser, "--zone", args.zone)
    if not pathlib.Path(args.config).is_dir():
        parser.error(f"config folder {args.config!r} does not exist")
    try:
        conversion = convert(args.config, args.exp, robots, zones, args.goal)
    except ValueError as e:
        parser.error(str(e))
    team, goal, launch = write(conversion, args.out)
    print(f"wrote {team} and {goal}")
    print("dropped:" + ("".join(f"\n  {entry}" for entry in conversion.dropped) or " nothing"))
    print(f"launch with: {launch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
