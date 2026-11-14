"""The TaskRequest a CASPER task becomes for the robot that runs it."""

from __future__ import annotations

import dataclasses
import math

from arena_simulation_setup.shared.task import GoToPhase, GraspPhase, ReachPhase, TaskPhase, TaskRequest
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position
from casper.goal import GoalTask

from arena_casper.scene import Binding, Scene, Site

REACH_FRAME = "odom"
PLACE_FRAME = "base_link"
SETTLE = 1.0


@dataclasses.dataclass(frozen=True)
class ArmReach:
    """How far a robot stands from what it reaches, how far its stand may miss in distance and heading, and the named pose it tucks its arm into after a grasp."""

    standoff: float
    slack: float
    yaw_slack: float
    stow: str = ""


def _grasp(action: str, obj: str, arm: ArmReach) -> list[TaskPhase]:
    grasp: list[TaskPhase] = [GraspPhase(action=action, object=obj)]
    return [*grasp, ReachPhase(named_target=arm.stow)] if arm.stow else grasp


def _stand_and_reach(target: Pose, zone: Pose, approach: float | None, arm: ArmReach, site: Site) -> list[TaskPhase]:
    heading = math.atan2(target.position.y - zone.position.y, target.position.x - zone.position.x) if approach is None else approach
    stand = Position(target.position.x - arm.standoff * math.cos(heading), target.position.y - arm.standoff * math.sin(heading), 0.0)
    facing = Orientation.from_yaw(heading)
    return [
        GoToPhase(pose=Pose(stand, facing), tolerance_radius=arm.slack, tolerance_angle=arm.yaw_slack, hold_time=SETTLE),
        ReachPhase(target=Pose(site.realize(target).position, facing), frame=REACH_FRAME),
    ]


def _bound_phases(task: GoalTask, binding: Binding, site: Site, arm: ArmReach) -> list[TaskPhase]:
    route = list(task.workspaces or ())
    phases: list[TaskPhase] = []
    standing = ""
    if binding.pick:
        standing = route.pop(0)
        phases += [*_stand_and_reach(site.locate(binding.pick), site.locate(standing), binding.approach, arm, site), *_grasp("close", binding.pick, arm)]
    for zone in route:
        standing = zone
        phases.append(GoToPhase(target=zone))
    if (place := binding.place) is not None:
        if standing != place.at:
            phases.append(GoToPhase(target=place.at, hold_time=SETTLE))
        height = site.locate(place.object).position.z if place.z is None else place.z
        phases += [ReachPhase(target=Pose(Position(arm.standoff, 0.0, height), Orientation.identity()), frame=PLACE_FRAME), *_grasp("open", place.object, arm)]
    return phases


def request_for(task: GoalTask, duration_scale: float, scene: Scene | None = None, site: Site | None = None, arm: ArmReach | None = None) -> TaskRequest:
    """A goto through the task's workspaces holding at the last one for its scaled duration, with a scene binding the pick and place phases around that route and the hold on its last goto."""
    hold = task.duration * duration_scale
    binding = None if scene is None else scene.tasks.get(task.id)
    if binding is None:
        if not task.workspaces:
            return TaskRequest(phases=[GoToPhase(hold_time=hold)], task=task.id)
        *legs, last = task.workspaces
        return TaskRequest(phases=[*(GoToPhase(target=zone) for zone in legs), GoToPhase(target=last, hold_time=hold)], task=task.id)
    if site is None:
        raise ValueError(f"task {task.id!r} has a scene binding and needs a site")
    if arm is None:
        raise ValueError(f"task {task.id!r} has a scene binding and its robot declares no box arm workspace")
    phases = _bound_phases(task, binding, site, arm)
    last = next(phase for phase in reversed(phases) if isinstance(phase, GoToPhase))
    last.hold_time = max(hold, last.hold_time or 0.0)
    return TaskRequest(phases=phases, task=task.id)
