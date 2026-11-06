from __future__ import annotations

import pathlib

import pytest
from casper.allocation import Mode, allocate
from casper.goal import load_goal

from arena_casper.team import load_team, parse_team

MOBILE = {"can_move": True, "can_manipulate": False}
ARM = {"can_move": True, "can_manipulate": True}
CONFIGS = pathlib.Path(__file__).parents[2] / "configs" / "office_1"


def _member(level: float, constraints: dict[str, object] | None = None) -> dict[str, object]:
    profile: dict[str, object] = {"skills": {"party": {"cook": [level, 1]}}}
    if constraints is not None:
        profile["constraints"] = constraints
    return profile


def test_robot_constraints_join_the_stated_ones() -> None:
    team = parse_team({"supervisor": "arm", "members": {"arm": _member(0.5, {"max_payload_kg": 3})}}, {"arm": ARM})
    assert team.members["arm"].constraints == {"max_payload_kg": 3, "can_move": True, "can_manipulate": True}


def test_supervisor_team_decides_whether_the_supervisor_takes_regular_tasks() -> None:
    raw = {"supervisor": "boss", "members": {"a": _member(0.5), "boss": _member(0.9)}}
    robots = {"a": MOBILE, "boss": MOBILE}
    assert parse_team(raw, robots).regular == ("a", "boss")
    assert parse_team({**raw, "supervisor_team": False}, robots).regular == ("a",)
    assert parse_team(raw, robots).supervisors == ("boss",)


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


def test_office_example_balances_over_the_three_jackals() -> None:
    goal = load_goal(CONFIGS / "goal.json")
    team = load_team(CONFIGS / "team.yaml", dict.fromkeys(("jackal_0", "jackal_1", "jackal_2"), MOBILE))
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
