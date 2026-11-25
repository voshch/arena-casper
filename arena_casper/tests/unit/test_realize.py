from __future__ import annotations

import math

import pytest
from arena_simulation_setup.shared.task import GoToPhase, GraspPhase, ReachPhase
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position
from casper.goal import GoalTask, parse_task

from arena_casper.realize import APPROACH, APPROACH_SPEED, LIFT, PLACE_CLEARANCE, PLACE_FRAME, PLACE_INSET, PULL_MARGIN, RETREAT, SETTLE, STAND_TOLERANCE, STAND_YAW_TOLERANCE, ArmReach, request_for
from arena_casper.scene import Binding, Place, Scene, Site

POSES = {
    "cup": Pose(Position(4.0, 5.0, 0.8), Orientation.from_yaw(1.0)),
    "plate": Pose(Position(1.0, 2.0, 0.05), Orientation.identity()),
    "table": Pose(Position(4.0, 1.0, 0.0), Orientation.identity()),
    "garden": Pose(Position(10.0, 11.0, 0.0), Orientation.identity()),
}
STANDOFF = 0.6
TIP = "arm_tool_link"
ARM = ArmReach(standoff=STANDOFF, slack=0.3, yaw_slack=0.35, tip=TIP)


SITE = Site(locate=POSES.__getitem__)


def _task(workspaces: list[str] | None, duration: float = 10.0) -> GoalTask:
    constraints = {} if workspaces is None else {"workspace": workspaces}
    return parse_task({"id": "T", "context": "party", "required_constraints": constraints, "duration": duration})


def _request(task: GoalTask, scene: Scene) -> list:
    return request_for(task, 1.0, scene, SITE, ARM).phases


def _xyz(pose: Pose | None) -> tuple[float, float, float]:
    assert pose is not None
    return tuple(pose.position)


def _held(reach: ReachPhase, site: Site = SITE) -> Pose:
    assert reach.target is not None
    return site.locate(reach.frame) * reach.target


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
    goto, _, reach, grasp, _ = _request(_task(["table"]), Scene(tasks={"T": Binding(pick="cup")}))
    assert isinstance(goto, GoToPhase)
    assert goto.target is None
    assert _xyz(goto.pose) == pytest.approx((4.0, 5.0 - STANDOFF, 0.0))
    assert goto.pose.orientation.to_yaw() == pytest.approx(math.pi / 2)
    assert (goto.tolerance_radius, goto.tolerance_angle, goto.hold_time) == (ARM.slack, ARM.yaw_slack, 10.0)
    assert grasp == GraspPhase(action="close", object="cup")


def test_pick_reaches_the_object_in_its_own_frame_along_the_heading() -> None:
    _, _, reach, _, _ = _request(_task(["table"]), Scene(tasks={"T": Binding(pick="cup")}))
    assert isinstance(reach, ReachPhase)
    assert reach.frame == "cup"
    assert _xyz(_held(reach)) == pytest.approx((4.0, 5.0, 0.8))
    assert _held(reach).orientation.to_yaw() == pytest.approx(math.pi / 2)


def test_pick_approach_fixes_the_heading() -> None:
    goto, _, reach, _, _ = _request(_task(["table"]), Scene(tasks={"T": Binding(pick="cup", approach=math.pi)}))
    assert _xyz(goto.pose) == pytest.approx((4.0 + STANDOFF, 5.0, 0.0))
    assert abs(goto.pose.orientation.to_yaw()) == pytest.approx(math.pi)
    assert abs(_held(reach).orientation.to_yaw()) == pytest.approx(math.pi)


def test_place_goes_to_the_zone_and_reaches_ahead_at_the_object_height() -> None:
    goto, reach, grasp, retreat = _request(_task(["garden"]), Scene(tasks={"T": Binding(place=Place(at="garden", object="cup"))}))
    assert goto == GoToPhase(target="garden", hold_time=10.0)
    assert reach == ReachPhase(target=Pose(Position(STANDOFF, 0.0, 0.8), Orientation.identity()), frame=PLACE_FRAME)
    assert grasp == GraspPhase(action="open", object="cup")
    assert retreat == ReachPhase(target=Pose(Position(STANDOFF - APPROACH, 0.0, 0.8 + LIFT), Orientation.identity()), frame=PLACE_FRAME, motion="linear", speed=APPROACH_SPEED)


def test_place_height_override_replaces_the_object_height() -> None:
    scene = Scene(tasks={"T": Binding(place=Place(at="garden", object="cup", z=1.1))})
    assert _xyz(_request(_task(None), scene)[-3].target) == (STANDOFF, 0.0, 1.1)


def test_place_outside_the_workspaces_adds_a_goto_to_its_zone() -> None:
    phases = _request(_task(["table"]), Scene(tasks={"T": Binding(place=Place(at="garden", object="plate"))}))
    assert [type(phase).__name__ for phase in phases] == ["GoToPhase", "GoToPhase", "ReachPhase", "GraspPhase", "ReachPhase"]
    assert phases[0] == GoToPhase(target="table")
    assert phases[1] == GoToPhase(target="garden", hold_time=10.0)


def test_carry_picks_at_the_first_workspace_and_places_at_the_last() -> None:
    scene = Scene(tasks={"T": Binding(pick="cup", place=Place(at="garden", object="cup"))})
    phases = _request(_task(["table", "garden"]), scene)
    assert [type(phase).__name__ for phase in phases] == ["GoToPhase", "ReachPhase", "ReachPhase", "GraspPhase", "ReachPhase", "GoToPhase", "ReachPhase", "GraspPhase", "ReachPhase"]
    assert phases[0].hold_time == SETTLE
    assert phases[3] == GraspPhase(action="close", object="cup")
    assert phases[5] == GoToPhase(target="garden", hold_time=10.0)
    assert phases[6].frame == PLACE_FRAME
    assert phases[7] == GraspPhase(action="open", object="cup")


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
    phases = request_for(_task(["table", "garden"]), 1.0, scene, SITE, ArmReach(standoff=STANDOFF, slack=0.3, yaw_slack=0.35, tip=TIP, stow="stow")).phases
    assert [type(phase).__name__ for phase in phases] == ["GoToPhase", "ReachPhase", "ReachPhase", "GraspPhase", "ReachPhase", "ReachPhase", "GoToPhase", "ReachPhase", "GraspPhase", "ReachPhase", "ReachPhase"]
    assert phases[5] == ReachPhase(named_target="stow")
    assert phases[10] == ReachPhase(named_target="stow")
    assert phases[6] == GoToPhase(target="garden", hold_time=10.0)


def test_an_arm_with_a_carry_pose_carries_after_closing_and_stows_after_opening() -> None:
    scene = Scene(tasks={"T": Binding(pick="cup", place=Place(at="garden", object="cup"))})
    phases = request_for(_task(["table", "garden"]), 1.0, scene, SITE, ArmReach(standoff=STANDOFF, slack=0.3, yaw_slack=0.35, tip=TIP, stow="stow", carry="carry")).phases
    assert phases[5] == ReachPhase(named_target="carry")
    assert phases[10] == ReachPhase(named_target="stow")


def test_an_arm_with_only_a_carry_pose_does_not_tuck_after_opening() -> None:
    scene = Scene(tasks={"T": Binding(pick="cup", place=Place(at="garden", object="cup"))})
    phases = request_for(_task(["table", "garden"]), 1.0, scene, SITE, ArmReach(standoff=STANDOFF, slack=0.3, yaw_slack=0.35, tip=TIP, carry="carry")).phases
    assert phases[5] == ReachPhase(named_target="carry")
    assert [type(phase).__name__ for phase in phases][-2:] == ["GraspPhase", "ReachPhase"]
    assert phases[-1].motion == "linear"


def _immobile(task: GoalTask, scene: Scene | None = None, arm: ArmReach = ARM) -> list:
    return request_for(task, 1.0, scene, SITE, arm, can_move=False).phases


def test_an_immobile_robot_without_a_binding_holds_in_place_whatever_the_workspaces() -> None:
    assert request_for(_task(["a", "b"]), 0.5, can_move=False).phases == [GoToPhase(hold_time=5.0)]


def test_an_immobile_robot_with_an_unbound_task_in_the_scene_holds_in_place() -> None:
    scene = Scene(tasks={"other": Binding(pick="cup")})
    assert _immobile(_task(["table"]), scene) == [GoToPhase(hold_time=10.0)]


def test_an_immobile_pick_reaches_and_grasps_without_standing_or_moving() -> None:
    phases = _immobile(_task(["table"]), Scene(tasks={"T": Binding(pick="cup")}))
    assert [type(phase).__name__ for phase in phases] == ["ReachPhase", "ReachPhase", "GraspPhase", "ReachPhase", "GoToPhase"]
    assert phases[1].frame == "cup"
    assert _xyz(_held(phases[1])) == pytest.approx((4.0, 5.0, 0.8))
    assert phases[2] == GraspPhase(action="close", object="cup")
    assert phases[4] == GoToPhase(hold_time=10.0)


def test_an_immobile_pick_keeps_the_heading_from_its_zone() -> None:
    phases = _immobile(_task(["table"]), Scene(tasks={"T": Binding(pick="cup")}))
    assert _held(phases[0]).orientation.to_yaw() == pytest.approx(math.atan2(5.0 - 1.0, 4.0 - 4.0))


def test_an_immobile_pick_without_workspaces_faces_along_the_simulator_x_axis() -> None:
    phases = _immobile(_task(None), Scene(tasks={"T": Binding(pick="cup")}))
    assert _held(phases[0]).orientation.to_yaw() == pytest.approx(0.0)


def test_an_immobile_pick_approach_fixes_the_heading() -> None:
    phases = _immobile(_task(["table"]), Scene(tasks={"T": Binding(pick="cup", approach=math.pi / 2)}))
    assert _held(phases[0]).orientation.to_yaw() == pytest.approx(math.pi / 2)


def test_an_immobile_place_reaches_ahead_without_going_to_its_zone() -> None:
    phases = _immobile(_task(["table"]), Scene(tasks={"T": Binding(place=Place(at="garden", object="plate"))}))
    assert [type(phase).__name__ for phase in phases] == ["ReachPhase", "GraspPhase", "ReachPhase", "GoToPhase"]
    assert phases[0].frame == PLACE_FRAME
    assert _xyz(phases[0].target) == (STANDOFF, 0.0, 0.05)
    assert phases[1] == GraspPhase(action="open", object="plate")
    assert phases[3] == GoToPhase(hold_time=10.0)


def test_an_immobile_carry_skips_every_goto_but_the_closing_hold() -> None:
    scene = Scene(tasks={"T": Binding(pick="cup", place=Place(at="garden", object="cup"))})
    phases = _immobile(_task(["table", "garden"]), scene, ArmReach(standoff=STANDOFF, slack=0.3, yaw_slack=0.35, tip=TIP, stow="stow"))
    assert [type(phase).__name__ for phase in phases] == ["ReachPhase", "ReachPhase", "GraspPhase", "ReachPhase", "ReachPhase", "ReachPhase", "GraspPhase", "ReachPhase", "ReachPhase", "GoToPhase"]
    assert phases[4] == ReachPhase(named_target="stow")
    assert phases[8] == ReachPhase(named_target="stow")
    assert phases[9] == GoToPhase(hold_time=10.0)


def test_a_mobile_place_already_at_its_zone_still_holds_on_a_goto() -> None:
    scene = Scene(tasks={"T": Binding(place=Place(at="table", object="plate"))})
    phases = _request(_task(["table"]), scene)
    assert [type(phase).__name__ for phase in phases] == ["GoToPhase", "ReachPhase", "GraspPhase", "ReachPhase"]
    assert phases[0] == GoToPhase(target="table", hold_time=10.0)


BOXED = Site(
    locate={"box": Pose(Position(2.0, 0.0, 0.5), Orientation.identity()), "turned": Pose(Position(2.0, 0.0, 0.5), Orientation.from_yaw(math.pi / 2)), "table": Pose(Position(0.0, 0.0, 0.0), Orientation.identity())}.__getitem__,
    extent=lambda name: ((0.4, 0.2, 0.1), (0.0, 0.0, 0.05)) if name in ("box", "turned") else None,
)


def test_pick_reaches_the_near_face_of_an_annotated_box_at_mid_height() -> None:
    goto, _, reach, _, _ = request_for(_task(["table"]), 1.0, Scene(tasks={"T": Binding(pick="box", approach=0.0)}), BOXED, ARM).phases
    assert _xyz(_held(reach, BOXED)) == pytest.approx((2.0 - 0.2 - 0.01, 0.0, 0.55))
    assert _xyz(goto.pose) == pytest.approx((2.0 - 0.2 - 0.01 - STANDOFF, 0.0, 0.0))


def test_a_turned_box_is_reached_across_its_short_side() -> None:
    _, _, reach, _, _ = request_for(_task(["table"]), 1.0, Scene(tasks={"T": Binding(pick="turned", approach=0.0)}), BOXED, ARM).phases
    assert _xyz(_held(reach, BOXED)) == pytest.approx((2.0 - 0.1 - 0.01, 0.0, 0.55))


def test_pick_moves_straight_onto_the_grasp_point_from_the_approach_short_of_it() -> None:
    _, short, reach, _, _ = request_for(_task(["table"]), 1.0, Scene(tasks={"T": Binding(pick="box", approach=0.0)}), BOXED, ARM).phases
    assert (short.frame, short.motion, reach.frame, reach.motion, reach.speed) == ("box", "free", "box", "linear", APPROACH_SPEED)
    assert _xyz(_held(short, BOXED)) == pytest.approx((2.0 - 0.2 - 0.01 - APPROACH, 0.0, 0.55))
    assert _held(short, BOXED).orientation.to_yaw() == pytest.approx(_held(reach, BOXED).orientation.to_yaw())


def test_a_goto_before_a_reach_holds_until_the_robot_settles() -> None:
    scene = Scene(tasks={"T": Binding(pick="cup", place=Place(at="garden", object="cup"))})
    phases = request_for(_task(["table"]), 0.001, scene, SITE, ARM).phases
    gotos = [phase for phase in phases if isinstance(phase, GoToPhase)]
    assert [goto.hold_time for goto in gotos] == [SETTLE, SETTLE]


def test_the_arm_tip_stands_back_by_the_hand_so_the_fingers_stop_short_of_the_face() -> None:
    arm = ArmReach(standoff=STANDOFF, slack=0.3, yaw_slack=0.35, tip=TIP, hand=0.2)
    goto, _, reach, _, _ = request_for(_task(["table"]), 1.0, Scene(tasks={"T": Binding(pick="box", approach=0.0)}), BOXED, arm).phases
    assert _xyz(_held(reach, BOXED)) == pytest.approx((2.0 - 0.2 - 0.01 - 0.2, 0.0, 0.55))
    assert _xyz(goto.pose) == pytest.approx((2.0 - 0.2 - 0.01 - 0.2 - STANDOFF, 0.0, 0.0))


SUPPORTED = Site(
    locate={
        "box": Pose(Position(2.0, 0.0, 0.5), Orientation.identity()),
        "desk": Pose(Position(5.0, 3.0, 0.0), Orientation.from_yaw(math.pi / 2)),
        "lobby": Pose(Position(5.0, 0.0, 0.0), Orientation.identity()),
    }.__getitem__,
    extent=lambda name: {"box": ((0.4, 0.2, 0.1), (0.0, 0.0, 0.05)), "desk": ((1.6, 0.8, 0.7), (0.1, 0.0, 0.35))}.get(name),
)


def _set_down(place: Place, arm: ArmReach = ARM, can_move: bool = True) -> list:
    return request_for(_task(["lobby"]), 1.0, Scene(tasks={"T": Binding(place=place)}), SUPPORTED, arm, can_move=can_move).phases


def test_a_place_on_a_support_stands_precisely_facing_it_from_the_zone_and_lowers_the_object_inside_its_near_edge() -> None:
    goto, above, down, release, retreat = _set_down(Place(at="lobby", object="box", onto="desk"))
    edge_y = 3.1 - 0.8
    point = (5.0, edge_y + PLACE_INSET - 0.01, 0.7 + 0.05 + PLACE_CLEARANCE)
    assert _xyz(_held(down, SUPPORTED)) == pytest.approx(point)
    assert _held(down, SUPPORTED).orientation.to_yaw() == pytest.approx(math.pi / 2)
    assert (down.frame, down.motion, down.speed) == ("desk", "linear", APPROACH_SPEED)
    assert _xyz(_held(above, SUPPORTED)) == pytest.approx((5.0, point[1], point[2] + APPROACH))
    assert _xyz(goto.pose) == pytest.approx((5.0, point[1] - STANDOFF, 0.0))
    assert (goto.hold_time, goto.tolerance_radius, goto.tolerance_angle) == (10.0, STAND_TOLERANCE, STAND_YAW_TOLERANCE)
    assert release == GraspPhase(action="open", object="box")
    assert (retreat.frame, retreat.motion, retreat.speed, retreat.target) == ("desk", "linear", APPROACH_SPEED, above.target)


def test_a_place_on_a_support_along_its_approach_meets_the_face_that_heading_reaches() -> None:
    _, _, down, _, _ = _set_down(Place(at="lobby", object="box", onto="desk", approach=0.0))
    assert _xyz(_held(down, SUPPORTED)) == pytest.approx((5.0 - 0.4 + PLACE_INSET - 0.01, 3.1, 0.7 + 0.05 + PLACE_CLEARANCE))


def test_a_place_on_a_support_keeps_the_fingers_clear_of_the_object_by_the_hand() -> None:
    arm = ArmReach(standoff=STANDOFF, slack=0.3, yaw_slack=0.35, tip=TIP, hand=0.2)
    _, _, down, _, _ = _set_down(Place(at="lobby", object="box", onto="desk", approach=0.0), arm)
    assert _held(down, SUPPORTED).position.x == pytest.approx(5.0 - 0.4 + PLACE_INSET - 0.01 - 0.2)


def test_an_immobile_place_on_a_support_reaches_from_where_the_robot_stands() -> None:
    phases = _set_down(Place(at="lobby", object="box", onto="desk"), can_move=False)
    assert [type(phase).__name__ for phase in phases] == ["ReachPhase", "ReachPhase", "GraspPhase", "ReachPhase", "GoToPhase"]


def test_a_pick_backs_the_held_object_straight_out_of_the_tip_frame_and_lifts_it_clear() -> None:
    *_, retreat = request_for(_task(["table"]), 1.0, Scene(tasks={"T": Binding(pick="box", approach=0.0)}), BOXED, ARM).phases
    assert retreat == ReachPhase(target=Pose(Position(-RETREAT, 0.0, LIFT), Orientation.identity()), frame=TIP, motion="linear", speed=APPROACH_SPEED)


def test_the_carry_pose_comes_after_the_pick_retreat() -> None:
    arm = ArmReach(standoff=STANDOFF, slack=0.3, yaw_slack=0.35, tip=TIP, carry="carry")
    phases = request_for(_task(["table"]), 1.0, Scene(tasks={"T": Binding(pick="box", approach=0.0)}), BOXED, arm).phases
    assert [phase.motion for phase in phases[4:6]] == ["linear", "free"]
    assert phases[5] == ReachPhase(named_target="carry")


RACKED = Site(
    locate={
        "box": Pose(Position(2.0, 0.0, 0.5), Orientation.identity()),
        "rack": Pose(Position(2.0, 0.0, 0.0), Orientation.identity()),
        "plinth": Pose(Position(1.0, 0.0, 0.0), Orientation.identity()),
        "table": Pose(Position(0.0, 0.0, 0.0), Orientation.identity()),
    }.__getitem__,
    extent=lambda name: {"box": ((0.4, 0.2, 0.1), (0.0, 0.0, 0.05)), "rack": ((0.6, 1.2, 1.8), (0.0, 0.0, 0.9)), "plinth": ((0.2, 0.2, 1.0), (0.0, 0.0, 0.5))}.get(name),
    holder=lambda name: "rack" if name == "box" else None,
)


def _racked(site: Site = RACKED, can_move: bool = True) -> list:
    return request_for(_task(["table"]), 1.0, Scene(tasks={"T": Binding(pick="box", approach=0.0)}), site, ARM, can_move=can_move).phases


def test_a_pick_from_inside_a_holder_backs_out_drives_straight_in_and_backs_out_again_by_what_takes_the_hand_clear_of_it() -> None:
    stand, back, outside, enter, reach, close, retreat, out = _racked()
    grasp_x = 2.0 - 0.2 - 0.01
    pull = grasp_x - 1.7 + PULL_MARGIN
    assert _xyz(stand.pose) == pytest.approx((grasp_x - STANDOFF, 0.0, 0.0))
    assert _xyz(back.pose) == pytest.approx((grasp_x - STANDOFF - pull, 0.0, 0.0))
    assert (enter.pose, out.pose) == (stand.pose, back.pose)
    assert pull > STAND_TOLERANCE
    assert {(goto.tolerance_radius, goto.tolerance_angle) for goto in (stand, back, enter, out)} == {(STAND_TOLERANCE, STAND_YAW_TOLERANCE)}
    assert (outside.frame, outside.motion) == ("box", "free")
    assert _xyz(_held(outside, RACKED)) == pytest.approx((grasp_x - APPROACH - pull, 0.0, 0.55))
    assert (reach.motion, _xyz(_held(reach, RACKED))) == ("linear", pytest.approx((grasp_x, 0.0, 0.55)))
    assert close == GraspPhase(action="close", object="box")
    assert retreat.frame == TIP


def test_a_holder_the_footprint_already_clears_adds_no_straight_moves() -> None:
    site = Site(locate=RACKED.locate, extent=RACKED.extent, holder=lambda name: "plinth" if name == "box" else None)
    phases = _racked(site)
    assert [type(phase).__name__ for phase in phases] == ["GoToPhase", "ReachPhase", "ReachPhase", "GraspPhase", "ReachPhase"]
    assert phases[0].tolerance_radius == ARM.slack


def test_an_immobile_pick_from_inside_a_holder_reaches_from_where_the_robot_stands() -> None:
    phases = _racked(can_move=False)
    assert not any(isinstance(phase, GoToPhase) and phase.pose is not None for phase in phases)


def test_a_pick_from_inside_a_holder_pulls_out_by_the_fingers_whatever_the_hand_length() -> None:
    arm = ArmReach(standoff=STANDOFF, slack=0.3, yaw_slack=0.35, tip=TIP, hand=0.2)
    stand, back, *_ = request_for(_task(["table"]), 1.0, Scene(tasks={"T": Binding(pick="box", approach=0.0)}), RACKED, arm).phases
    assert stand.pose.position.x - back.pose.position.x == pytest.approx(2.0 - 0.2 - 0.01 - 1.7 + PULL_MARGIN)
