from __future__ import annotations

import pathlib

import pytest
from casper.allocation import Mode, allocate
from casper.goal import load_goal

from arena_casper.team import load_team, parse_team

MOBILE = {"can_move": True, "can_manipulate": False}
ARM = {"can_move": True, "can_manipulate": True}
CONFIGS = pathlib.Path(__file__).parents[5] / "arena_simulation_setup" / "worlds" / "office_1" / "scenarios" / "casper"
SCENARIO_FOLDER = pytest.mark.skipif(not CONFIGS.is_dir(), reason="the scenario folders live in the Arena metarepo")


def _member(level: float, constraints: dict[str, object] | None = None) -> dict[str, object]:
    profile: dict[str, object] = {"skills": {"party": {"cook": [level, 1]}}}
    if constraints is not None:
        profile["constraints"] = constraints
    return profile


def test_robot_constraints_join_the_stated_ones() -> None:
    team = parse_team({"supervisor": "arm", "members": {"arm": _member(0.5, {"max_payload_kg": 3})}}, {"arm": ARM}).team
    assert team.members["arm"].constraints == {"max_payload_kg": 3, "can_move": True, "can_manipulate": True}


def test_supervisor_team_decides_whether_the_supervisor_takes_regular_tasks() -> None:
    raw = {"supervisor": "boss", "members": {"a": _member(0.5), "boss": _member(0.9)}}
    robots = {"a": MOBILE, "boss": MOBILE}
    assert parse_team(raw, robots).team.regular == ("a", "boss")
    assert parse_team({**raw, "supervisor_team": False}, robots).team.regular == ("a",)
    assert parse_team(raw, robots).team.supervisors == ("boss",)


def test_team_refuses_constraints_the_robot_decides() -> None:
    with pytest.raises(ValueError, match=r"\['can_manipulate'\] come from the robot's adapters"):
        parse_team({"supervisor": "a", "members": {"a": _member(0.5, {"can_manipulate": True})}}, {"a": MOBILE})


def test_team_refuses_members_that_are_no_robots_of_the_env() -> None:
    with pytest.raises(ValueError, match=r"members \['ghost'\] are not robots of this env, robots: \['a'\]"):
        parse_team({"supervisor": "a", "members": {"a": _member(0.5), "ghost": _member(0.5)}}, {"a": MOBILE})


def test_team_supervisor_must_be_a_member() -> None:
    with pytest.raises(ValueError, match="supervisor must name a member"):
        parse_team({"supervisor": "boss", "members": {"a": _member(0.5)}}, {"a": MOBILE})


def test_team_refuses_skills_that_are_no_level_rank_pair() -> None:
    with pytest.raises(ValueError, match=r"skills.party.cook must be \[level, preference rank\]"):
        parse_team({"supervisor": "a", "members": {"a": {"skills": {"party": {"cook": [0.5]}}}}}, {"a": MOBILE})


@SCENARIO_FOLDER
def test_office_example_balances_over_the_three_jackals() -> None:
    goal = load_goal(CONFIGS / "goal.json")
    team = load_team(CONFIGS / "team.yaml", dict.fromkeys(("jackal_0", "jackal_1", "jackal_2"), MOBILE)).team
    owners = {task_id: assignment.member for task_id, assignment in allocate(goal, team, Mode.BALANCE_WORKLOAD, top_k=3).assignments.items()}
    assert owners == {
        "GreetVisitors": "jackal_0",
        "GuideToMeeting": "jackal_1",
        "FetchSupplies": "jackal_2",
        "DeliverSupplies": "jackal_1",
        "FetchCoffee": "jackal_2",
        "ServeCoffee": "jackal_1",
        "HostMeeting": "jackal_0",
        "TidyMeetingRoom": "jackal_2",
        "CloseUp": "jackal_0",
        "OfficeMeeting": "jackal_0",
    }


def test_pedestrian_members_get_fixed_constraints_and_are_named() -> None:
    raw = {"supervisor": "a", "members": {"a": _member(0.5), "walker": {**_member(0.7, {"max_payload_kg": 1}), "kind": "pedestrian"}}}
    parsed = parse_team(raw, {"a": ARM})
    assert parsed.pedestrians == frozenset({"walker"})
    assert parsed.team.members["walker"].constraints == {"max_payload_kg": 1, "can_move": True, "can_manipulate": False}
    assert parsed.team.members["a"].constraints == {"can_move": True, "can_manipulate": True}


def test_robot_kind_is_the_default_and_can_be_stated() -> None:
    raw = {"supervisor": "a", "members": {"a": {**_member(0.5), "kind": "robot"}}}
    assert parse_team(raw, {"a": MOBILE}).pedestrians == frozenset()


def test_team_refuses_a_pedestrian_named_like_a_robot() -> None:
    raw = {"supervisor": "a", "members": {"a": {**_member(0.5), "kind": "pedestrian"}}}
    with pytest.raises(ValueError, match=r"members \['a'\] are pedestrians in the team file and robots of this env"):
        parse_team(raw, {"a": MOBILE})


def test_team_refuses_an_unknown_member_kind() -> None:
    raw = {"supervisor": "a", "members": {"a": {**_member(0.5), "kind": "drone"}}}
    with pytest.raises(ValueError, match=r"member 'a': kind must be one of \['robot', 'pedestrian'\], got 'drone'"):
        parse_team(raw, {"a": MOBILE})


def test_team_refuses_constraints_the_pedestrian_kind_decides() -> None:
    raw = {"supervisor": "w", "members": {"w": {**_member(0.5, {"can_move": False}), "kind": "pedestrian"}}}
    with pytest.raises(ValueError, match=r"\['can_move'\] come from the pedestrian kind"):
        parse_team(raw, {})


def test_a_robot_member_still_needs_a_robot_beside_pedestrians() -> None:
    raw = {"supervisor": "w", "members": {"w": {**_member(0.5), "kind": "pedestrian"}, "ghost": _member(0.5)}}
    with pytest.raises(ValueError, match=r"members \['ghost'\] are not robots of this env, robots: \[\]"):
        parse_team(raw, {})
