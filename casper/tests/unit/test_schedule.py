from __future__ import annotations

import pytest

from casper.allocation import Mode, allocate
from casper.goal import parse_goal
from casper.schedule import Schedule, Status
from casper.skills import Team, member


def _schedule(tasks: list[dict[str, object]], dependencies: list[list[str]] | None = None, members: dict[str, float] | None = None) -> Schedule:
    goal = parse_goal({"tasks": [{"context": "party", "required_skills": {"work": 0.1}, **task} for task in tasks], "dependencies": dependencies or []})
    levels = members or {"a": 0.9}
    members = {name: member(name, {"party": {"work": [level, 1]}}, {}, None) for name, level in levels.items()}
    team = Team(members=members, regular=tuple(members), supervisors=(next(iter(members)),))
    return Schedule(goal, allocate(goal, team, Mode.SKILL, top_k=3))


def test_member_takes_its_ready_task_of_the_lowest_priority_number() -> None:
    schedule = _schedule([{"id": "late", "priority": 5}, {"id": "early", "priority": 2}, {"id": "blocked", "priority": 1}], dependencies=[["late", "blocked"]])
    assert schedule.next_for("a") == "early"


def test_member_runs_one_task_at_a_time() -> None:
    schedule = _schedule([{"id": "t1", "priority": 1}, {"id": "t2", "priority": 2}])
    schedule.start("t1")
    assert schedule.next_for("a") is None
    schedule.finish("t1")
    assert schedule.next_for("a") == "t2"


def test_task_waits_for_its_predecessors() -> None:
    schedule = _schedule([{"id": "cook", "priority": 1}, {"id": "serve", "priority": 2}], dependencies=[["cook", "serve"]], members={"a": 0.9, "b": 0.9})
    assert schedule.states["serve"].owner == "b"
    assert schedule.next_for("b") is None
    schedule.start("cook")
    assert schedule.next_for("b") is None
    schedule.finish("cook")
    assert schedule.next_for("b") == "serve"


def test_clean_tasks_wait_for_the_subgoals_outside_at_end() -> None:
    schedule = _schedule(
        [
            {"id": "party", "priority": 1, "subgoal_level": 1},
            {"id": "wash", "priority": 2, "time_to_clean": False},
            {"id": "close", "priority": 3, "subgoal_level": 1, "at_end": True},
        ],
        dependencies=[["wash", "close"]],
    )
    assert schedule.next_for("a") == "party"
    schedule.start("party")
    schedule.finish("party")
    assert schedule.next_for("a") == "wash"


def test_failed_task_moves_to_the_next_fallback_until_none_is_left() -> None:
    schedule = _schedule([{"id": "cook"}], members={"a": 0.9, "b": 0.5, "c": 0.3})
    schedule.start("cook")
    assert schedule.fail("cook") == "b"
    assert schedule.next_for("b") == "cook"
    schedule.start("cook")
    assert schedule.fail("cook") == "c"
    schedule.start("cook")
    assert schedule.fail("cook") is None
    assert schedule.states["cook"].failed_by == {"a", "b", "c"}


def test_interrupted_task_waits_for_its_owner_again() -> None:
    schedule = _schedule([{"id": "cook"}])
    schedule.start("cook")
    schedule.interrupt("cook")
    assert schedule.states["cook"].status is Status.PENDING
    assert schedule.next_for("a") == "cook"


def test_schedule_is_done_once_every_task_is() -> None:
    schedule = _schedule([{"id": "t1", "priority": 1}, {"id": "t2", "priority": 2}])
    for task_id in ("t1", "t2"):
        assert not schedule.done
        schedule.start(task_id)
        schedule.finish(task_id)
    assert schedule.done


def test_only_ready_tasks_start_and_only_running_ones_finish() -> None:
    schedule = _schedule([{"id": "cook"}, {"id": "serve"}], dependencies=[["cook", "serve"]])
    with pytest.raises(ValueError, match="'serve' is not ready"):
        schedule.start("serve")
    with pytest.raises(ValueError, match="'cook' is pending, not running"):
        schedule.finish("cook")
