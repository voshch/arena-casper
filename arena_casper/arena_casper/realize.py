"""The TaskRequest a CASPER task becomes for the robot that runs it."""

from __future__ import annotations

import dataclasses
import math

from arena_simulation_setup.shared.task import GoToPhase, GraspPhase, ReachPhase, TaskPhase, TaskRequest
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position
from casper.goal import GoalTask

from arena_casper.scene import Binding, Place, Scene, Site

PLACE_FRAME = "base_link"
GRASP_MARGIN = 0.01
APPROACH = 0.1
APPROACH_SPEED = 0.2
SETTLE = 1.0
PLACE_INSET = 0.03
PLACE_CLEARANCE = 0.01
LIFT = 0.03
RETREAT = 0.05
PULL_MARGIN = 0.15
STAND_TOLERANCE = 0.05
STAND_YAW_TOLERANCE = 0.05


@dataclasses.dataclass(frozen=True)
class ArmReach:
    """How far a robot stands from what it reaches, how far its stand may miss in distance and heading, its arm's tip link, the named poses it tucks its arm into after a grasp, `carry` when the grasp holds an object, and how far its hand reaches past the arm's tip."""

    standoff: float
    slack: float
    yaw_slack: float
    tip: str
    stow: str = ""
    carry: str = ""
    hand: float = 0.0


def _grasp(action: str, obj: str, arm: ArmReach, leave: list[TaskPhase]) -> list[TaskPhase]:
    grasp: list[TaskPhase] = [GraspPhase(action=action, object=obj), *leave]
    pose = arm.carry or arm.stow if action == "close" else arm.stow
    return [*grasp, ReachPhase(named_target=pose)] if pose else grasp


def _grasp_point(name: str, target: Pose, heading: float, site: Site, hand: float) -> Position:
    extent = site.extent(name)
    if extent is None:
        return Position(target.position.x - hand * math.cos(heading), target.position.y - hand * math.sin(heading), target.position.z)
    (sx, sy, _sz), center = extent
    middle = target * Position(*center)
    turn = heading - target.orientation.to_yaw()
    back = abs(math.cos(turn)) * sx / 2 + abs(math.sin(turn)) * sy / 2 + GRASP_MARGIN + hand
    return Position(middle.x - back * math.cos(heading), middle.y - back * math.sin(heading), middle.z)


def _pull_out(name: str, point: Position, heading: float, site: Site) -> float:
    """How far the fingers move back along the heading to leave the box of what holds the object, plus the margin, 0 when they lie outside."""
    holder = site.holder(name)
    held = None if holder is None else site.extent(holder)
    if holder is None or held is None:
        return 0.0
    (hx, hy, _hz), hc = held
    frame = site.locate(holder)
    yaw = heading - frame.orientation.to_yaw()
    back = (-math.cos(yaw), -math.sin(yaw))
    local = frame.inverse() * point
    offset = (local.x - hc[0], local.y - hc[1])
    if abs(offset[0]) >= hx / 2 or abs(offset[1]) >= hy / 2:
        return 0.0
    exits = [(math.copysign(half / 2, step) - at) / step for at, half, step in zip(offset, (hx, hy), back, strict=True) if abs(step) > 1e-9]
    return min(exits) + PULL_MARGIN


def _straight(at: Position, facing: Orientation) -> GoToPhase:
    return GoToPhase(pose=Pose(at, facing), tolerance_radius=STAND_TOLERANCE, tolerance_angle=STAND_YAW_TOLERANCE, hold_time=SETTLE)


def _stand_and_reach(name: str, zone: Pose | None, approach: float | None, arm: ArmReach, site: Site, can_move: bool) -> tuple[list[TaskPhase], list[TaskPhase]]:
    target = site.locate(name)
    if approach is not None:
        heading = approach
    else:
        heading = 0.0 if zone is None else math.atan2(target.position.y - zone.position.y, target.position.x - zone.position.x)
    facing = Orientation.from_yaw(heading)
    point = _grasp_point(name, target, heading, site, arm.hand)
    short = Position(point.x - APPROACH * math.cos(heading), point.y - APPROACH * math.sin(heading), point.z)
    reach: list[TaskPhase] = [
        ReachPhase(target=target.inverse() * Pose(short, facing), frame=name),
        ReachPhase(target=target.inverse() * Pose(point, facing), frame=name, motion="linear", speed=APPROACH_SPEED),
    ]
    retreat = ReachPhase(target=Pose(Position(-RETREAT, 0.0, LIFT), Orientation.identity()), frame=arm.tip, motion="linear", speed=APPROACH_SPEED)
    if not can_move:
        return reach, [retreat]
    stand = Position(point.x - arm.standoff * math.cos(heading), point.y - arm.standoff * math.sin(heading), 0.0)
    fingers = Position(point.x + arm.hand * math.cos(heading), point.y + arm.hand * math.sin(heading), point.z)
    pull = _pull_out(name, fingers, heading, site)
    if pull <= 0.0:
        return [GoToPhase(pose=Pose(stand, facing), tolerance_radius=arm.slack, tolerance_angle=arm.yaw_slack, hold_time=SETTLE), *reach], [retreat]
    back = Position(stand.x - pull * math.cos(heading), stand.y - pull * math.sin(heading), 0.0)
    outside = Position(short.x - pull * math.cos(heading), short.y - pull * math.sin(heading), short.z)
    enter = [_straight(stand, facing), _straight(back, facing), ReachPhase(target=target.inverse() * Pose(outside, facing), frame=name), _straight(stand, facing), reach[1]]
    return enter, [retreat, _straight(back, facing)]


def _set_down(place: Place, zone: Pose | None, arm: ArmReach, site: Site, can_move: bool) -> tuple[list[TaskPhase], list[TaskPhase]]:
    support = site.locate(place.onto)
    if place.approach is not None:
        heading = place.approach
    else:
        heading = 0.0 if zone is None else math.atan2(support.position.y - zone.position.y, support.position.x - zone.position.x)
    facing = Orientation.from_yaw(heading)
    extent = site.extent(place.onto)
    if extent is None:
        middle, front, top = support.position, 0.0, support.position.z
    else:
        (sx, sy, sz), center = extent
        middle = support * Position(*center)
        turn = heading - support.orientation.to_yaw()
        front = abs(math.cos(turn)) * sx / 2 + abs(math.sin(turn)) * sy / 2
        top = middle.z + sz / 2
    carried = site.extent(place.object)
    lift = 0.0 if carried is None else carried[0][2] / 2
    ahead = PLACE_INSET - front - GRASP_MARGIN - arm.hand
    point = Position(middle.x + ahead * math.cos(heading), middle.y + ahead * math.sin(heading), top + lift + PLACE_CLEARANCE)
    above = Position(point.x, point.y, point.z + APPROACH)
    reach: list[TaskPhase] = [
        ReachPhase(target=support.inverse() * Pose(above, facing), frame=place.onto),
        ReachPhase(target=support.inverse() * Pose(point, facing), frame=place.onto, motion="linear", speed=APPROACH_SPEED),
    ]
    retreat = ReachPhase(target=support.inverse() * Pose(above, facing), frame=place.onto, motion="linear", speed=APPROACH_SPEED)
    if not can_move:
        return reach, [retreat]
    stand = Position(point.x - arm.standoff * math.cos(heading), point.y - arm.standoff * math.sin(heading), 0.0)
    return [_straight(stand, facing), *reach], [retreat]


def _bound_phases(task: GoalTask, binding: Binding, site: Site, arm: ArmReach, can_move: bool) -> list[TaskPhase]:
    route = list(task.workspaces or ())
    phases: list[TaskPhase] = []
    standing = ""
    if binding.pick:
        standing = route.pop(0) if route or can_move else ""
        zone = site.locate(standing) if standing else None
        reach, leave = _stand_and_reach(binding.pick, zone, binding.approach, arm, site, can_move)
        phases += [*reach, *_grasp("close", binding.pick, arm, leave)]
    if binding.place is not None and binding.place.onto and route and route[-1] == binding.place.at:
        route.pop()
    for zone in route:
        standing = zone
        if can_move:
            phases.append(GoToPhase(target=zone))
    if (place := binding.place) is not None and place.onto:
        reach, leave = _set_down(place, site.locate(place.at), arm, site, can_move)
        phases += [*reach, *_grasp("open", place.object, arm, leave)]
    elif place is not None:
        if can_move and standing != place.at:
            phases.append(GoToPhase(target=place.at, hold_time=SETTLE))
        height = site.locate(place.object).position.z if place.z is None else place.z
        retreat = ReachPhase(target=Pose(Position(arm.standoff - APPROACH, 0.0, height + LIFT), Orientation.identity()), frame=PLACE_FRAME, motion="linear", speed=APPROACH_SPEED)
        phases += [ReachPhase(target=Pose(Position(arm.standoff, 0.0, height), Orientation.identity()), frame=PLACE_FRAME), *_grasp("open", place.object, arm, [retreat])]
    return phases


def request_for(task: GoalTask, duration_scale: float, scene: Scene | None = None, site: Site | None = None, arm: ArmReach | None = None, *, can_move: bool = True) -> TaskRequest:
    """A goto through the task's workspaces holding at the last one for its scaled duration, with a scene binding the pick and place phases around that route and the hold on its last goto, a robot that cannot move holds in place and reaches from where it stands."""
    hold = task.duration * duration_scale
    binding = None if scene is None else scene.tasks.get(task.id)
    if binding is None:
        if not can_move or not task.workspaces:
            return TaskRequest(phases=[GoToPhase(hold_time=hold)], task=task.id)
        *legs, last = task.workspaces
        return TaskRequest(phases=[*(GoToPhase(target=zone) for zone in legs), GoToPhase(target=last, hold_time=hold)], task=task.id)
    if site is None:
        raise ValueError(f"task {task.id!r} has a scene binding and needs a site")
    if arm is None:
        raise ValueError(f"task {task.id!r} has a scene binding and its robot declares no box arm workspace")
    phases = _bound_phases(task, binding, site, arm, can_move)
    last = next((phase for phase in reversed(phases) if isinstance(phase, GoToPhase)), None)
    if last is None:
        phases.append(GoToPhase(hold_time=hold))
    else:
        last.hold_time = max(hold, last.hold_time or 0.0)
    return TaskRequest(phases=phases, task=task.id)
