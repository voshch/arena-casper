from __future__ import annotations

import math

import pytest
from arena_simulation_setup.shared.task import GoToPhase, GraspPhase, ReachPhase
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position
from casper.goal import GoalTask, parse_task

from arena_casper.realize import PLACE_FRAME, REACH_FRAME, SETTLE, ArmReach, request_for
from arena_casper.scene import Binding, Place, Scene, Site

POSES = {
    "cup": Pose(Position(4.0, 5.0, 0.8), Orientation.from_yaw(1.0)),
    "plate": Pose(Position(1.0, 2.0, 0.05), Orientation.identity()),
    "table": Pose(Position(4.0, 1.0, 0.0), Orientation.identity()),
    "garden": Pose(Position(10.0, 11.0, 0.0), Orientation.identity()),
}
SHIFT = 100.0
STANDOFF = 0.6
ARM = ArmReach(standoff=STANDOFF, slack=0.3, yaw_slack=0.35)


def _realize(pose: Pose) -> Pose:
    return Pose(Position(pose.position.x + SHIFT, pose.position.y, pose.position.z), pose.orientation)


SITE = Site(locate=POSES.__getitem__, realize=_realize)


def _task(workspaces: list[str] | None, duration: float = 10.0) -> GoalTask:
    constraints = {} if workspaces is None else {"workspace": workspaces}
    return parse_task({"id": "T", "context": "party", "required_constraints": constraints, "duration": duration})


def _request(task: GoalTask, scene: Scene) -> list:
    return request_for(task, 1.0, scene, SITE, ARM).phases


def _xyz(pose: Pose | None) -> tuple[float, float, float]:
    assert pose is not None
    return tuple(pose.position)


def test_without_a_binding_the_task_is_a_goto_through_its_workspaces_holding_at_the_last() -> None:
    request = request_for(_task(["a", "b"]), 0.5)
    assert request.task == "T"
    assert request.phases == [GoToPhase(target="a"), GoToPhase(target="b", hold_time=5.0)]


def test_without_a_binding_or_workspaces_the_task_holds_in_place() -> None:
    assert request_for(_task(None), 2.0).phases == [GoToPhase(hold_time=20.0)]


def test_a_scene_without_the_task_leaves_it_unbound() -> None:
    scene = Scene(tasks={"other": Binding(pick="cup")})
    assert _request(_task(["a"]), scene) == [GoToPhase(target="a", hold_time=10.0)]


def test_pick_stands_off_the_object_facing_it_from_the_zone() -> None:
    goto, reach, grasp = _request(_task(["table"]), Scene(tasks={"T": Binding(pick="cup")}))
    assert isinstance(goto, GoToPhase)
    assert goto.target is None
    assert _xyz(goto.pose) == pytest.approx((4.0, 5.0 - STANDOFF, 0.0))
    assert goto.pose.orientation.to_yaw() == pytest.approx(math.pi / 2)
    assert (goto.tolerance_radius, goto.tolerance_angle, goto.hold_time) == (ARM.slack, ARM.yaw_slack, 10.0)
    assert grasp == GraspPhase(action="close", object="cup")


def test_pick_reaches_the_object_in_the_simulator_frame_along_the_heading() -> None:
    _, reach, _ = _request(_task(["table"]), Scene(tasks={"T": Binding(pick="cup")}))
    assert isinstance(reach, ReachPhase)
    assert reach.frame == REACH_FRAME
    assert _xyz(reach.target) == (4.0 + SHIFT, 5.0, 0.8)
    assert reach.target.orientation.to_yaw() == pytest.approx(math.pi / 2)


def test_pick_approach_fixes_the_heading() -> None:
    goto, reach, _ = _request(_task(["table"]), Scene(tasks={"T": Binding(pick="cup", approach=math.pi)}))
    assert _xyz(goto.pose) == pytest.approx((4.0 + STANDOFF, 5.0, 0.0))
    assert abs(goto.pose.orientation.to_yaw()) == pytest.approx(math.pi)
    assert abs(reach.target.orientation.to_yaw()) == pytest.approx(math.pi)


def test_place_goes_to_the_zone_and_reaches_ahead_at_the_object_height() -> None:
    goto, reach, grasp = _request(_task(["garden"]), Scene(tasks={"T": Binding(place=Place(at="garden", object="cup"))}))
    assert goto == GoToPhase(target="garden", hold_time=10.0)
    assert reach == ReachPhase(target=Pose(Position(STANDOFF, 0.0, 0.8), Orientation.identity()), frame=PLACE_FRAME)
    assert grasp == GraspPhase(action="open", object="cup")


def test_place_height_override_replaces_the_object_height() -> None:
    scene = Scene(tasks={"T": Binding(place=Place(at="garden", object="cup", z=1.1))})
    assert _xyz(_request(_task(None), scene)[-2].target) == (STANDOFF, 0.0, 1.1)


def test_place_outside_the_workspaces_adds_a_goto_to_its_zone() -> None:
    phases = _request(_task(["table"]), Scene(tasks={"T": Binding(place=Place(at="garden", object="plate"))}))
    assert [type(phase).__name__ for phase in phases] == ["GoToPhase", "GoToPhase", "ReachPhase", "GraspPhase"]
    assert phases[0] == GoToPhase(target="table")
    assert phases[1] == GoToPhase(target="garden", hold_time=10.0)


def test_carry_picks_at_the_first_workspace_and_places_at_the_last() -> None:
    scene = Scene(tasks={"T": Binding(pick="cup", place=Place(at="garden", object="cup"))})
    phases = _request(_task(["table", "garden"]), scene)
    assert [type(phase).__name__ for phase in phases] == ["GoToPhase", "ReachPhase", "GraspPhase", "GoToPhase", "ReachPhase", "GraspPhase"]
    assert phases[0].hold_time == SETTLE
    assert phases[2] == GraspPhase(action="close", object="cup")
    assert phases[3] == GoToPhase(target="garden", hold_time=10.0)
    assert phases[4].frame == PLACE_FRAME
    assert phases[5] == GraspPhase(action="open", object="cup")


def test_a_binding_needs_a_site() -> None:
    with pytest.raises(ValueError, match="task 'T' has a scene binding and needs a site"):
        request_for(_task(["table"]), 1.0, Scene(tasks={"T": Binding(pick="cup")}))


def test_a_binding_needs_a_robot_with_an_arm_workspace() -> None:
    with pytest.raises(ValueError, match="declares no box arm workspace"):
        request_for(_task(["table"]), 1.0, Scene(tasks={"T": Binding(pick="cup")}), SITE)


def test_unknown_names_surface_from_the_locator() -> None:
    with pytest.raises(KeyError, match="ghost"):
        _request(_task(["table"]), Scene(tasks={"T": Binding(pick="ghost")}))


def test_an_arm_with_a_stow_pose_tucks_after_each_grasp() -> None:
    scene = Scene(tasks={"T": Binding(pick="cup", place=Place(at="garden", object="cup"))})
    phases = request_for(_task(["table", "garden"]), 1.0, scene, SITE, ArmReach(standoff=STANDOFF, slack=0.3, yaw_slack=0.35, stow="stow")).phases
    assert [type(phase).__name__ for phase in phases] == ["GoToPhase", "ReachPhase", "GraspPhase", "ReachPhase", "GoToPhase", "ReachPhase", "GraspPhase", "ReachPhase"]
    assert phases[3] == ReachPhase(named_target="stow")
    assert phases[7] == ReachPhase(named_target="stow")
    assert phases[4] == GoToPhase(target="garden", hold_time=10.0)


def test_a_goto_before_a_reach_holds_until_the_robot_settles() -> None:
    scene = Scene(tasks={"T": Binding(pick="cup", place=Place(at="garden", object="cup"))})
    phases = request_for(_task(["table"]), 0.001, scene, SITE, ARM).phases
    gotos = [phase for phase in phases if isinstance(phase, GoToPhase)]
    assert [goto.hold_time for goto in gotos] == [SETTLE, SETTLE]
