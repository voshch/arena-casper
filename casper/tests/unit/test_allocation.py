from __future__ import annotations

import pytest

from casper.allocation import AllocationError, Mode, allocate, assign, rank, rebalance
from casper.goal import Goal, parse_goal
from casper.skills import UNSTATED, Team, member


def _goal(*tasks: dict[str, object]) -> Goal:
    return parse_goal({"tasks": [{"context": "party", **task} for task in tasks]})


def _team(profiles: dict[str, dict[str, object]], supervisor: str | None = None, supervisor_team: bool = True) -> Team:
    members = {name: member(name, profile.get("skills", {}), profile.get("constraints", {}), profile.get("workspace")) for name, profile in profiles.items()}
    boss = supervisor or next(iter(profiles))
    return Team(members=members, regular=tuple(name for name in members if supervisor_team or name != boss), supervisors=(boss,))


def _profile(skills: dict[str, list[float]], workspace: list[str] | None = None, constraints: dict[str, object] | None = None, context: str = "party") -> dict[str, object]:
    return {"skills": {context: skills}, "workspace": workspace, "constraints": constraints or {}}


def test_skill_mode_ranks_by_mean_level() -> None:
    goal = _goal({"id": "serve", "required_skills": {"transport": 0.5, "communication": 0.2}})
    team = _team({"a": _profile({"transport": [0.6, 1], "communication": [0.4, 2]}), "b": _profile({"transport": [1.0, 1], "communication": [0.6, 2]})})
    assert rank(goal, team, Mode.SKILL)["serve"] == (("b", pytest.approx(0.8)), ("a", pytest.approx(0.5)))
    assert allocate(goal, team, Mode.SKILL, top_k=3).assignments["serve"].member == "b"


def test_preference_counts_the_ranked_skills_of_every_context() -> None:
    goal = _goal({"id": "clean", "required_skills": {"clean": 0.4}})
    team = _team(
        {
            "a": {"skills": {"party": {"cook": [0.5, 1], "clean": [0.5, 2]}, "assembly": {"weld": [0.9, 1]}}},
            "b": _profile({"clean": [0.5, 1]}),
        }
    )
    assert rank(goal, team, Mode.PREFERENCE)["clean"] == (("b", pytest.approx(1.0)), ("a", pytest.approx((3 - 2 + 1) / 3)))


def test_unstated_context_skills_take_the_unstated_level() -> None:
    filled = member("a", {"party": {"cook": [0.5, 1]}, "assembly": {"weld": [0.9, 1]}}, {}, None)
    assert filled.level["weld"]["party"] == UNSTATED
    assert filled.preference["cook"]["assembly"] == UNSTATED


def test_balance_skill_preference_weighs_skill_seven_tenths() -> None:
    goal = _goal({"id": "cook", "required_skills": {"cook": 0.4}})
    team = _team({"a": _profile({"cook": [0.5, 2], "grill": [0.9, 1]})})
    assert rank(goal, team, Mode.BALANCE_SKILL_PREFERENCE)["cook"] == (("a", pytest.approx(0.7 * 0.5 + 0.3 * 0.5)),)


def test_top_score_ties_go_to_the_member_with_fewer_tasks() -> None:
    goal = _goal({"id": "t1", "required_skills": {"cook": 0.5}}, {"id": "t2", "required_skills": {"cook": 0.5}})
    team = _team({"a": _profile({"cook": [0.8, 1]}), "b": _profile({"cook": [0.8, 1]})})
    assignments = allocate(goal, team, Mode.SKILL, top_k=3).assignments
    assert (assignments["t1"].member, assignments["t2"].member) == ("a", "b")


def _rebalance_case() -> tuple[Goal, Team]:
    goal = _goal(
        {"id": "t1", "required_constraints": {"workspace": ["hall"]}},
        {"id": "t2", "required_constraints": {"workspace": ["kitchen"]}},
        {"id": "t3", "required_constraints": {"workspace": ["garden"]}},
        {"id": "t4", "required_constraints": {"workspace": ["kitchen"]}},
    )
    team = _team({"a": _profile({"cook": [0.5, 1]}, workspace=["hall", "kitchen"]), "b": _profile({"cook": [0.5, 1]}, workspace=["hall", "garden"])})
    return goal, team


def test_balance_workload_moves_a_task_off_the_busiest_member() -> None:
    goal, team = _rebalance_case()
    assignments = allocate(goal, team, Mode.BALANCE_WORKLOAD, top_k=3).assignments
    assert {task_id: assignment.member for task_id, assignment in assignments.items()} == {"t1": "b", "t2": "a", "t3": "b", "t4": "a"}
    assert assignments["t1"].fallbacks == ("a",)


def test_allocation_keeps_the_assignment_before_rebalancing() -> None:
    goal, team = _rebalance_case()
    allocation = allocate(goal, team, Mode.BALANCE_WORKLOAD, top_k=3)
    assert {task_id: assignment.member for task_id, assignment in allocation.initial.items()} == {"t1": "a", "t2": "a", "t3": "b", "t4": "a"}
    assert allocation.assignments["t1"].member == "b"


def test_without_rebalancing_the_initial_assignment_is_the_final_one() -> None:
    goal, team = _rebalance_case()
    allocation = allocate(goal, team, Mode.SKILL, top_k=3)
    assert {t: a.member for t, a in allocation.initial.items()} == {t: a.member for t, a in allocation.assignments.items()}


def test_assign_then_rebalance_are_the_steps_of_allocate() -> None:
    goal, team = _rebalance_case()
    ranking = rank(goal, team, Mode.BALANCE_WORKLOAD)
    assignments, workload, unassigned = assign(ranking, top_k=3)
    assert unassigned == []
    assert workload == {"a": 3, "b": 1}
    assert assignments["t1"].member == "a"
    rebalance(assignments, ranking, workload, top_k=3)
    assert workload == {"a": 2, "b": 2}
    assert assignments["t1"].member == "b"


def test_pinned_tasks_keep_their_member_and_count_toward_its_workload() -> None:
    goal = _goal({"id": "t1", "required_skills": {"cook": 0.5}}, {"id": "t2", "required_skills": {"cook": 0.5}}, {"id": "fly", "required_skills": {"fly": 0.5}})
    team = _team({"a": _profile({"cook": [0.8, 1]}), "b": _profile({"cook": [0.8, 1]})})
    assignments, workload, unassigned = assign(rank(goal, team, Mode.SKILL), top_k=3, pinned={"t1": "b", "fly": "a"})
    assert (assignments["t1"].member, assignments["t2"].member, assignments["fly"].member) == ("b", "a", "a")
    assert workload == {"b": 1, "a": 2}
    assert unassigned == []


def test_balance_workload_leaves_members_without_tasks_out_of_rebalancing() -> None:
    goal = _goal(
        {"id": "t1", "required_constraints": {"workspace": ["hall"]}},
        {"id": "t2", "required_constraints": {"workspace": ["kitchen"]}},
        {"id": "t3", "required_constraints": {"workspace": ["kitchen"]}},
    )
    team = _team({"a": _profile({"cook": [0.5, 1]}, workspace=["hall", "kitchen"]), "b": _profile({"cook": [0.5, 1]}, workspace=["hall"])})
    assignments = allocate(goal, team, Mode.BALANCE_WORKLOAD, top_k=3).assignments
    assert {assignment.member for assignment in assignments.values()} == {"a"}


def test_skill_mode_never_rebalances() -> None:
    goal = _goal(*({"id": f"t{i}", "required_skills": {"cook": 0.1}} for i in range(4)))
    team = _team({"a": _profile({"cook": [0.9, 1]}), "b": _profile({"cook": [0.5, 1]})})
    assert {assignment.member for assignment in allocate(goal, team, Mode.SKILL, top_k=3).assignments.values()} == {"a"}


def test_eligibility_checks_context_workspace_skills_and_constraints() -> None:
    goal = _goal(
        {"id": "grill", "required_skills": {"grill": 0.5}, "required_constraints": {"can_manipulate": True, "max_payload_kg": 2, "workspace": ["Grill"]}},
    )
    arm = {"can_manipulate": True}
    team = _team(
        {
            "fits": _profile({"grill": [0.6, 1]}, workspace=["grill", "garden"], constraints={**arm, "max_payload_kg": 3}),
            "no_arm": _profile({"grill": [0.9, 1]}, workspace=["Grill"], constraints={"can_manipulate": False}),
            "weak": _profile({"grill": [0.4, 1]}, workspace=["Grill"], constraints=arm),
            "light": _profile({"grill": [0.9, 1]}, workspace=["Grill"], constraints={**arm, "max_payload_kg": 1}),
            "elsewhere": _profile({"grill": [0.9, 1]}, workspace=["Garden"], constraints=arm),
            "nowhere": _profile({"grill": [0.9, 1]}, constraints=arm),
            "other_context": _profile({"grill": [0.9, 1]}, workspace=["Grill"], constraints=arm, context="assembly"),
            "unknown_payload": _profile({"grill": [0.7, 1]}, workspace=["Grill"], constraints=arm),
        }
    )
    assert [name for name, _ in rank(goal, team, Mode.SKILL)["grill"]] == ["unknown_payload", "fits"]


def test_false_constraint_does_not_exclude_a_member_that_can() -> None:
    goal = _goal({"id": "greet", "required_skills": {"talk": 0.3}, "required_constraints": {"can_manipulate": False, "can_transport_objects": False}})
    team = _team({"arm": _profile({"talk": [0.9, 1]}, constraints={"can_manipulate": True, "can_transport_objects": True}), "base": _profile({"talk": [0.5, 1]}, constraints={"can_manipulate": False})})
    assert [name for name, _ in rank(goal, team, Mode.SKILL)["greet"]] == ["arm", "base"]


def test_supervisor_only_tasks_go_to_the_supervisors() -> None:
    goal = _goal({"id": "supervise", "subgoal_level": 1, "required_skills": {"plan": 0.3}}, {"id": "party", "is_end_goal": True, "required_skills": {"plan": 0.3}})
    team = _team({"a": _profile({"plan": [0.5, 1]}), "boss": _profile({"plan": [0.4, 1]})}, supervisor="boss")
    assignments = allocate(goal, team, Mode.SKILL, top_k=3).assignments
    assert assignments["supervise"].member == "boss"
    assert assignments["party"].member == "boss"
    assert assignments["party"].fallbacks == ()


def test_supervisor_outside_the_team_takes_no_regular_task() -> None:
    goal = _goal({"id": "cook", "required_skills": {"cook": 0.3}})
    team = _team({"boss": _profile({"cook": [0.9, 1]}), "a": _profile({"cook": [0.4, 1]})}, supervisor="boss", supervisor_team=False)
    assert allocate(goal, team, Mode.SKILL, top_k=3).assignments["cook"].member == "a"


def test_allocation_names_every_task_nobody_qualifies_for() -> None:
    goal = _goal({"id": "fly", "required_skills": {"fly": 0.1}}, {"id": "cook", "required_skills": {"cook": 0.1}}, {"id": "dive", "required_skills": {"dive": 0.1}})
    team = _team({"a": _profile({"cook": [0.5, 1]})})
    with pytest.raises(AllocationError) as error:
        allocate(goal, team, Mode.SKILL, top_k=3)
    assert error.value.unassigned == ["fly", "dive"]


def test_fallbacks_hold_the_next_top_k_ranked_members() -> None:
    goal = _goal({"id": "cook", "required_skills": {"cook": 0.1}})
    team = _team({name: _profile({"cook": [level, 1]}) for name, level in (("a", 0.2), ("b", 0.9), ("c", 0.5), ("d", 0.4))})
    assignment = allocate(goal, team, Mode.SKILL, top_k=2).assignments["cook"]
    assert (assignment.member, assignment.fallbacks) == ("b", ("c", "d"))


def test_member_refuses_skill_weights_that_are_no_pair() -> None:
    with pytest.raises(ValueError, match=r"member 'a': party.cook must be \[level, preference rank\]"):
        member("a", {"party": {"cook": [0.5]}}, {}, None)
