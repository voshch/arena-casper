from __future__ import annotations

from types import SimpleNamespace

import pytest
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position
from casper.goal import parse_goal

from arena_casper.scene import make_site, parse_scene, validate_scene

SQUARE = [Position(0.0, 0.0), Position(4.0, 0.0), Position(4.0, 2.0), Position(0.0, 2.0)]


@pytest.fixture(autouse=True)
def _ros_gate() -> None:
    pytest.importorskip("rclpy")


def _ctx(shift: float = 0.0) -> SimpleNamespace:
    cup = SimpleNamespace(name="world_cup", pose=Pose(Position(1.0, 1.0, 0.8), Orientation.identity()))
    level = SimpleNamespace(
        objects=[cup],
        zone_ref_names=lambda: ["kitchen", "garden"],
        lookup_zone_polygon=lambda name: SQUARE if name == "kitchen" else None,
    )
    return SimpleNamespace(
        world_manager=SimpleNamespace(world_compacted=lambda: level),
        environment_manager=SimpleNamespace(realize=lambda pose: Pose(Position(pose.position.x + shift, pose.position.y, pose.position.z), pose.orientation)),
    )


def _goal() -> object:
    task = {"context": "party", "required_constraints": {"workspace": ["kitchen"]}}
    return parse_goal({"tasks": [{"id": "Fetch", **task}, {"id": "Idle", "context": "party"}]})


def test_valid_scene_passes() -> None:
    scene = parse_scene({"tasks": {"Fetch": {"pick": "cup", "place": "garden"}}})
    validate_scene(scene, _goal(), _ctx())


def test_scene_naming_a_task_outside_the_goal_is_refused() -> None:
    with pytest.raises(ValueError, match=r"tasks \['Ghost'\] are not in the goal, tasks: \['Fetch', 'Idle'\]"):
        validate_scene(parse_scene({"tasks": {"Ghost": {"pick": "cup"}}}), _goal(), _ctx())


def test_scene_naming_an_object_outside_the_world_is_refused() -> None:
    with pytest.raises(ValueError, match="task 'Fetch': object 'mug' is no object of this world"):
        validate_scene(parse_scene({"tasks": {"Fetch": {"pick": "mug"}}}), _goal(), _ctx())


def test_scene_naming_a_zone_outside_the_world_is_refused() -> None:
    scene = parse_scene({"tasks": {"Fetch": {"pick": "cup", "place": "attic"}}})
    with pytest.raises(ValueError, match=r"task 'Fetch': place.at 'attic' is no zone of this world, zones: \['garden', 'kitchen'\]"):
        validate_scene(scene, _goal(), _ctx())


def test_pick_on_a_task_without_workspace_is_refused() -> None:
    with pytest.raises(ValueError, match="task 'Idle': pick needs the goal task to list a workspace"):
        validate_scene(parse_scene({"tasks": {"Idle": {"pick": "cup"}}}), _goal(), _ctx())


def test_site_finds_world_objects_under_their_authored_name() -> None:
    assert make_site(_ctx()).locate("cup").position == Position(1.0, 1.0, 0.8)


def test_site_finds_the_interior_of_a_zone() -> None:
    pose = make_site(_ctx()).locate("kitchen")
    assert (pose.position.x, pose.position.y, pose.position.z) == (2.0, 1.0, 0.0)


def test_site_locates_in_the_abstract_frame_and_realizes_into_the_simulator_frame() -> None:
    site = make_site(_ctx(shift=100.0))
    cup = site.locate("cup")
    assert cup.position.x == 1.0
    assert site.realize(cup).position.x == 101.0


def test_site_refuses_unknown_names() -> None:
    with pytest.raises(ValueError, match="'ghost' is neither an object nor a zone of this world"):
        make_site(_ctx()).locate("ghost")
