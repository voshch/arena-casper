from __future__ import annotations

import pathlib

import pytest
from casper.allocation import Mode, allocate
from casper.goal import load_goal

from arena_casper.scene import Binding, Place, load_scene, parse_scene
from arena_casper.team import load_team

CONFIGS = pathlib.Path(__file__).parents[2] / "configs" / "office_1_boxes"


def test_pick_and_place_bindings_parse() -> None:
    scene = parse_scene(
        {
            "tasks": {
                "Fetch": {"pick": "cup"},
                "Serve": {"place": {"object": "cup", "at": "garden", "z": 0.9}},
                "Carry": {"pick": "plate", "place": "garden"},
                "Shelf": {"pick": {"object": "box", "approach": 1.5}},
            }
        }
    )
    assert scene.tasks == {
        "Fetch": Binding(pick="cup"),
        "Serve": Binding(place=Place(at="garden", object="cup", z=0.9)),
        "Carry": Binding(pick="plate", place=Place(at="garden", object="plate")),
        "Shelf": Binding(pick="box", approach=1.5),
    }


def test_carried_object_is_the_picked_or_the_placed_one() -> None:
    assert Binding(pick="cup").carried == "cup"
    assert Binding(place=Place(at="garden", object="plate")).carried == "plate"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([], "scene: expected a mapping"),
        ({"task": {}}, r"scene: unknown keys \['task'\]"),
        ({"tasks": {}}, "scene: tasks must map CASPER task ids"),
        ({"tasks": {"A": "cup"}}, "task 'A': expected a mapping with pick and/or place"),
        ({"tasks": {"A": {}}}, "task 'A': needs pick, place or both"),
        ({"tasks": {"A": {"grab": "cup"}}}, r"task 'A': unknown keys \['grab'\]"),
        ({"tasks": {"A": {"pick": 3}}}, "task 'A': pick must be an object name or"),
        ({"tasks": {"A": {"pick": {"object": 3}}}}, "task 'A': pick.object must be a name"),
        ({"tasks": {"A": {"pick": {"approach": 0.0}}}}, "task 'A': pick needs the object"),
        ({"tasks": {"A": {"pick": {"object": "cup", "side": 0.0}}}}, r"unknown pick keys \['side'\]"),
        ({"tasks": {"A": {"pick": {"object": "cup", "approach": True}}}}, "pick.approach must be a number"),
        ({"tasks": {"A": {"place": {"at": "garden"}}}}, "task 'A': place without pick needs the object"),
        ({"tasks": {"A": {"place": {"object": "cup"}}}}, "task 'A': place needs the zone"),
        ({"tasks": {"A": {"place": {"object": "cup", "at": "garden", "height": 1}}}}, r"unknown place keys \['height'\]"),
        ({"tasks": {"A": {"pick": "cup", "place": {"object": "plate", "at": "garden"}}}}, "place.object 'plate' differs from the picked 'cup'"),
        ({"tasks": {"A": {"place": {"object": "cup", "at": "garden", "z": "high"}}}}, "place.z must be a number"),
        ({"tasks": {"A": {"place": ["garden"]}}}, "place must be a zone name or"),
    ],
)
def test_scene_refuses(raw: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_scene(raw)


def test_load_scene_names_the_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "scene.yaml"
    path.write_text("tasks:\n  A: {pick: cup, drop: garden}\n")
    with pytest.raises(ValueError, match=rf"{path}: scene task 'A': unknown keys \['drop'\]"):
        load_scene(path)


def test_boxes_example_binds_goal_tasks_the_team_can_take() -> None:
    goal = load_goal(CONFIGS / "goal.json")
    scene = load_scene(CONFIGS / "scene.yaml")
    team = load_team(CONFIGS / "team.yaml", {"tiago": {"can_move": True, "can_manipulate": True}}).team
    assert set(scene.tasks) <= set(goal.tasks)
    owners = {task_id: assignment.member for task_id, assignment in allocate(goal, team, Mode.SKILL, top_k=1).assignments.items()}
    assert owners == dict.fromkeys(goal.tasks, "tiago")


def test_place_on_a_support_parses_with_its_approach() -> None:
    scene = parse_scene({"tasks": {"Deliver": {"place": {"object": "cup", "at": "garden", "onto": "table", "approach": 1.5}}}})
    assert scene.tasks["Deliver"] == Binding(place=Place(at="garden", object="cup", onto="table", approach=1.5))


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ({"place": {"object": "cup", "at": "garden", "onto": "table", "z": 0.9}}, "place takes the height of what it is on, drop place.z"),
        ({"place": {"object": "cup", "at": "garden", "approach": 0.0}}, "place.approach needs place.onto"),
        ({"place": {"object": "cup", "at": "garden", "onto": ""}}, "place.onto must be a name"),
        ({"place": {"object": "cup", "at": "garden", "onto": "table", "approach": "north"}}, "place.approach must be a number"),
    ],
    ids=["height_and_support", "approach_without_support", "empty_support", "named_approach"],
)
def test_a_bad_place_on_a_support_is_refused(raw: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse_scene({"tasks": {"A": raw}})
