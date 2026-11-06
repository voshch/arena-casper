from __future__ import annotations

import json
import math
import pathlib

import pytest

from casper.goal import load_goal, parse_goal


def _task(task_id: str, **fields: object) -> dict[str, object]:
    return {"id": task_id, "context": "party", **fields}


def test_goal_reads_requirements_workspaces_and_dependencies(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "goal.json"
    path.write_text(
        json.dumps(
            {
                "tasks": [
                    _task("pick", required_skills={"manipulation": 0.5}, required_constraints={"can_manipulate": True, "max_payload_kg": 1, "workspace": ["Table", "Grill"]}, duration=30, priority=2),
                    _task("serve", required_constraints={"workspace": "Garden"}, duration=5, priority=1, subgoal_level=1),
                ],
                "dependencies": [["pick", "serve"]],
            }
        )
    )
    goal = load_goal(path)
    pick, serve = goal.tasks["pick"], goal.tasks["serve"]
    assert pick.required_skills == {"manipulation": 0.5}
    assert pick.constraints == {"can_manipulate": True, "max_payload_kg": 1.0}
    assert pick.workspaces == ("Table", "Grill")
    assert serve.workspaces == ("Garden",)
    assert serve.supervisor_only and not pick.supervisor_only
    assert goal.predecessors("serve") == ("pick",)
    assert goal.predecessors("pick") == ()


def test_goal_defaults_match_casper() -> None:
    task = parse_goal({"tasks": [_task("idle")]}).tasks["idle"]
    assert task.priority == math.inf
    assert task.duration == 0.0
    assert task.workspaces is None
    assert task.required_skills == {}


def test_goal_rejects_unknown_task_keys() -> None:
    with pytest.raises(ValueError, match=r"task 'pick': unknown keys \['selected_agent'\]"):
        parse_goal({"tasks": [_task("pick", selected_agent="r0")]})


def test_goal_rejects_dependency_on_unknown_task() -> None:
    with pytest.raises(ValueError, match="unknown task 'ghost'"):
        parse_goal({"tasks": [_task("pick")], "dependencies": [["ghost", "pick"]]})


def test_goal_rejects_dependency_cycle() -> None:
    with pytest.raises(ValueError, match="dependency cycle"):
        parse_goal({"tasks": [_task("a"), _task("b")], "dependencies": [["a", "b"], ["b", "a"]]})


def test_goal_rejects_clean_tasks_nothing_releases() -> None:
    with pytest.raises(ValueError, match="never released"):
        parse_goal({"tasks": [_task("wash", time_to_clean=False), _task("close", subgoal_level=1, at_end=True)]})


def test_goal_names_the_file_of_a_bad_goal(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "goal.json"
    path.write_text(json.dumps({"tasks": [], "dependencies": []}))
    with pytest.raises(ValueError, match=f"{path}: goal: tasks must be a non-empty list"):
        load_goal(path)
