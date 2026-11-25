from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest
import yaml
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position
from casper.goal import parse_goal

from arena_casper.scene import make_site, parse_scene, validate_scene

SQUARE = [Position(0.0, 0.0), Position(4.0, 0.0), Position(4.0, 2.0), Position(0.0, 2.0)]


@pytest.fixture(autouse=True)
def _ros_gate() -> None:
    pytest.importorskip("rclpy")


def _ctx(model: object = None) -> SimpleNamespace:
    cup = SimpleNamespace(name="world_cup", pose=Pose(Position(1.0, 1.0, 0.8), Orientation.identity()), model=model)
    level = SimpleNamespace(
        objects=[cup],
        zone_ref_names=lambda: ["kitchen", "garden"],
        lookup_zone_polygon=lambda name: SQUARE if name == "kitchen" else None,
    )
    return SimpleNamespace(world_manager=SimpleNamespace(world_compacted=lambda: level))


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


def test_site_refuses_unknown_names() -> None:
    with pytest.raises(ValueError, match="'ghost' is neither an object nor a zone of this world"):
        make_site(_ctx()).locate("ghost")


def test_site_reads_an_objects_annotated_box(tmp_path: pathlib.Path) -> None:
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier, ObjectView

    bundle = tmp_path / "cup"
    bundle.mkdir()
    (bundle / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[-0.1, 0.1], [-0.05, 0.05], [0.0, 0.2]]}))
    site = make_site(_ctx(model=ObjectIdentifier.inline(ObjectView(bundle), name="cup")))
    size, center = site.extent("cup")
    assert size == pytest.approx((0.2, 0.1, 0.2))
    assert center == pytest.approx((0.0, 0.0, 0.1))
    assert site.extent("kitchen") is None


def _bundle(root: pathlib.Path, name: str, box: list[list[float]]) -> object:
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier, ObjectView

    bundle = root / name
    bundle.mkdir()
    (bundle / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": box}))
    return ObjectIdentifier.inline(ObjectView(bundle), name=name)


def test_site_names_the_smallest_object_whose_box_holds_another_objects_box_center(tmp_path: pathlib.Path) -> None:
    def place(name: str, x: float, z: float, box: list[list[float]]) -> SimpleNamespace:
        return SimpleNamespace(name=f"world_{name}", pose=Pose(Position(x, 0.0, z), Orientation.identity()), model=_bundle(tmp_path, name, box))

    objects = [
        place("box", 2.0, 0.5, [[-0.2, 0.2], [-0.1, 0.1], [0.0, 0.1]]),
        place("rack", 2.1, 0.0, [[-0.3, 0.3], [-0.6, 0.6], [0.0, 1.8]]),
        place("room", 2.5, 0.0, [[-2.0, 2.0], [-2.0, 2.0], [0.0, 3.0]]),
        place("desk", 0.0, 0.0, [[-0.5, 0.5], [-0.5, 0.5], [0.0, 0.7]]),
    ]
    level = SimpleNamespace(objects=objects, zone_ref_names=lambda: [], lookup_zone_polygon=lambda name: None)
    site = make_site(SimpleNamespace(world_manager=SimpleNamespace(world_compacted=lambda: level)))
    assert site.holder("box") == "rack"
    assert site.holder("desk") is None
    assert site.holder("ghost") is None


def test_scene_placing_on_an_object_outside_the_world_is_refused() -> None:
    scene = parse_scene({"tasks": {"Fetch": {"pick": "cup", "place": {"at": "garden", "onto": "desk"}}}})
    with pytest.raises(ValueError, match="task 'Fetch': place.onto 'desk' is no object of this world"):
        validate_scene(scene, _goal(), _ctx())
