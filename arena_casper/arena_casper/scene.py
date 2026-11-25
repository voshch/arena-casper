"""The scene binding of a CASPER goal: which world objects its tasks pick and where they place them."""

from __future__ import annotations

import dataclasses
import math
import pathlib
import typing
from collections.abc import Callable, Mapping

import yaml
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position
from casper.goal import Goal

if typing.TYPE_CHECKING:
    from arena_runtime.env.task_layer import TaskContext

_SCENE_KEYS = frozenset({"tasks"})
_BINDING_KEYS = frozenset({"pick", "place"})
_PICK_KEYS = frozenset({"object", "approach"})
_PLACE_KEYS = frozenset({"object", "at", "z", "onto", "approach"})
HOLDER_RADIUS = 3.0


@dataclasses.dataclass(frozen=True)
class Place:
    at: str
    object: str = ""
    z: float | None = None
    onto: str = ""
    approach: float | None = None


@dataclasses.dataclass(frozen=True)
class Binding:
    pick: str = ""
    place: Place | None = None
    approach: float | None = None

    @property
    def carried(self) -> str:
        """The object the task moves."""
        return self.pick or (self.place.object if self.place else "")


@dataclasses.dataclass(frozen=True)
class Scene:
    tasks: Mapping[str, Binding]


def _name(task_id: str, key: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"scene task {task_id!r}: {key} must be a name, got {value!r}")
    return value


def _number(task_id: str, key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"scene task {task_id!r}: {key} must be a number, got {value!r}")
    return float(value)


def _pick(task_id: str, raw: object) -> tuple[str, float | None]:
    if isinstance(raw, str):
        raw = {"object": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"scene task {task_id!r}: pick must be an object name or {{object, approach}}, got {raw!r}")
    if unknown := sorted(set(raw) - _PICK_KEYS):
        raise ValueError(f"scene task {task_id!r}: unknown pick keys {unknown}, allowed: {sorted(_PICK_KEYS)}")
    if "object" not in raw:
        raise ValueError(f"scene task {task_id!r}: pick needs the object to pick")
    approach = raw.get("approach")
    return _name(task_id, "pick.object", raw["object"]), None if approach is None else _number(task_id, "pick.approach", approach)


def _place(task_id: str, pick: str, raw: object) -> Place:
    if isinstance(raw, str):
        raw = {"at": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"scene task {task_id!r}: place must be a zone name or {{object, at, z, onto, approach}}, got {raw!r}")
    if unknown := sorted(set(raw) - _PLACE_KEYS):
        raise ValueError(f"scene task {task_id!r}: unknown place keys {unknown}, allowed: {sorted(_PLACE_KEYS)}")
    if "at" not in raw:
        raise ValueError(f"scene task {task_id!r}: place needs the zone to place at")
    at = _name(task_id, "place.at", raw["at"])
    obj = _name(task_id, "place.object", raw["object"]) if "object" in raw else pick
    if not obj:
        raise ValueError(f"scene task {task_id!r}: place without pick needs the object to place")
    if pick and obj != pick:
        raise ValueError(f"scene task {task_id!r}: place.object {obj!r} differs from the picked {pick!r}")
    if "onto" in raw and "z" in raw:
        raise ValueError(f"scene task {task_id!r}: place takes the height of what it is on, drop place.z")
    if "approach" in raw and "onto" not in raw:
        raise ValueError(f"scene task {task_id!r}: place.approach needs place.onto")
    z, approach = raw.get("z"), raw.get("approach")
    return Place(
        at=at,
        object=obj,
        z=None if z is None else _number(task_id, "place.z", z),
        onto=_name(task_id, "place.onto", raw["onto"]) if "onto" in raw else "",
        approach=None if approach is None else _number(task_id, "place.approach", approach),
    )


def _binding(task_id: str, raw: object) -> Binding:
    if not isinstance(raw, dict):
        raise ValueError(f"scene task {task_id!r}: expected a mapping with pick and/or place, got {raw!r}")
    if unknown := sorted(set(raw) - _BINDING_KEYS):
        raise ValueError(f"scene task {task_id!r}: unknown keys {unknown}, allowed: {sorted(_BINDING_KEYS)}")
    if not raw:
        raise ValueError(f"scene task {task_id!r}: needs pick, place or both")
    pick, approach = _pick(task_id, raw["pick"]) if "pick" in raw else ("", None)
    place = _place(task_id, pick, raw["place"]) if "place" in raw else None
    return Binding(pick=pick, place=place, approach=approach)


def parse_scene(raw: object) -> Scene:
    if not isinstance(raw, dict):
        raise ValueError(f"scene: expected a mapping, got {raw!r}")
    if unknown := sorted(set(raw) - _SCENE_KEYS):
        raise ValueError(f"scene: unknown keys {unknown}, allowed: {sorted(_SCENE_KEYS)}")
    tasks = raw.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("scene: tasks must map CASPER task ids to their bindings")
    return Scene(tasks={str(task_id): _binding(str(task_id), entry) for task_id, entry in tasks.items()})


def load_scene(path: str | pathlib.Path) -> Scene:
    try:
        return parse_scene(yaml.safe_load(pathlib.Path(path).read_text()))
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e


def validate_scene(scene: Scene, goal: Goal, ctx: TaskContext) -> None:
    """ValueError naming the task ids, objects and zones that are not in the goal or the world."""
    from arena_runtime.shared import WORLD_ENTITY_PREFIX

    level = ctx.world_manager.world_compacted()
    objects = {entity.name.removeprefix(WORLD_ENTITY_PREFIX) for entity in level.objects}
    zones = set(level.zone_ref_names())
    if unknown := sorted(set(scene.tasks) - set(goal.tasks)):
        raise ValueError(f"scene: tasks {unknown} are not in the goal, tasks: {sorted(goal.tasks)}")
    for task_id, binding in scene.tasks.items():
        if binding.pick and not goal.tasks[task_id].workspaces:
            raise ValueError(f"scene task {task_id!r}: pick needs the goal task to list a workspace to stand in")
        if binding.carried not in objects:
            raise ValueError(f"scene task {task_id!r}: object {binding.carried!r} is no object of this world")
        if binding.place is not None and binding.place.at not in zones:
            raise ValueError(f"scene task {task_id!r}: place.at {binding.place.at!r} is no zone of this world, zones: {sorted(zones)}")
        if binding.place is not None and binding.place.onto and binding.place.onto not in objects:
            raise ValueError(f"scene task {task_id!r}: place.onto {binding.place.onto!r} is no object of this world")


def _interior_pose(polygon: list[Position]) -> Pose:
    import shapely

    shape = shapely.Polygon([(corner.x, corner.y) for corner in polygon])
    point = shape.centroid if shape.contains(shape.centroid) else shape.representative_point()
    return Pose(Position(point.x, point.y, 0.0), Orientation.identity())


Extent = tuple[tuple[float, float, float], tuple[float, float, float]]


def _no_extent(name: str) -> Extent | None:
    return None


def _no_holder(name: str) -> str | None:
    return None


@dataclasses.dataclass(frozen=True)
class Site:
    """World object poses and zone interior points in the env's abstract frame, an object's annotated box as (size, center) in its own frame, and the object whose box holds another's box center."""

    locate: Callable[[str], Pose]
    extent: Callable[[str], Extent | None] = _no_extent
    holder: Callable[[str], str | None] = _no_holder


def make_site(ctx: TaskContext) -> Site:
    """The env's site, whose locate raises ValueError for a name that is neither an object nor a zone."""
    from arena_runtime.shared import WORLD_ENTITY_PREFIX

    def locate(name: str) -> Pose:
        level = ctx.world_manager.world_compacted()
        for entity in level.objects:
            if entity.name.removeprefix(WORLD_ENTITY_PREFIX) == name:
                return entity.pose
        if (polygon := level.lookup_zone_polygon(name)) is not None:
            return _interior_pose(polygon)
        raise ValueError(f"{name!r} is neither an object nor a zone of this world")

    def extent(name: str) -> Extent | None:
        for entity in ctx.world_manager.world_compacted().objects:
            if entity.name.removeprefix(WORLD_ENTITY_PREFIX) == name:
                try:
                    return entity.model.resolve_sync().bbox
                except FileNotFoundError:
                    return None
        return None

    def holder(name: str) -> str | None:
        objects = ctx.world_manager.world_compacted().objects
        held = next((entity for entity in objects if entity.name.removeprefix(WORLD_ENTITY_PREFIX) == name), None)
        if held is None:
            return None
        box = extent(name)
        center = held.pose.position if box is None else held.pose * Position(*box[1])
        best: tuple[float, str] | None = None
        for entity in objects:
            if entity is held or math.hypot(entity.pose.position.x - center.x, entity.pose.position.y - center.y) > HOLDER_RADIUS:
                continue
            try:
                bbox = entity.model.resolve_sync().bbox
            except FileNotFoundError:
                continue
            if bbox is None:
                continue
            size, middle = bbox
            local = entity.pose.inverse() * center
            if all(abs(value - mid) <= half / 2 for value, mid, half in zip(local, middle, size, strict=True)):
                volume = size[0] * size[1] * size[2]
                if best is None or volume < best[0]:
                    best = (volume, entity.name.removeprefix(WORLD_ENTITY_PREFIX))
        return None if best is None else best[1]

    return Site(locate=locate, extent=extent, holder=holder)
