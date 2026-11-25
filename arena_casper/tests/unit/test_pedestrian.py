from __future__ import annotations

import math

import numpy as np
import pytest
from arena_simulation_setup.shared.pathing import OccupancyGrid, PathError, iter_points

from arena_casper.pedestrian import MAX_SPEED, YAW_RATE, PedRoute, PedState, plan_legs

DT = 0.1


def _run(route: PedRoute, seconds: float, can_hold: bool = True) -> list:
    return [route.step(DT, can_hold) for _ in range(round(seconds / DT))]


def test_route_walks_along_the_polyline_at_the_requested_speed() -> None:
    route = PedRoute([(0.0, 0.0), (4.0, 0.0)], yaw=0.0, speed=1.0)
    steps = _run(route, 2.0)
    assert steps[-1].x == pytest.approx(2.0)
    assert steps[-1].y == pytest.approx(0.0)
    assert {step.state for step in steps} == {PedState.WALKING}
    assert all(step.speed == pytest.approx(1.0) for step in steps)


def test_route_speed_stays_below_the_run_threshold() -> None:
    route = PedRoute([(0.0, 0.0), (40.0, 0.0)], yaw=0.0, speed=5.0)
    step = route.step(1.0)
    assert step.speed == pytest.approx(MAX_SPEED)
    assert step.x == pytest.approx(MAX_SPEED)
    assert MAX_SPEED < 1.8


def test_route_holds_at_its_end_for_the_hold_time_then_is_done() -> None:
    route = PedRoute([(0.0, 0.0), (1.0, 0.0)], yaw=0.0, speed=1.0, hold=1.0)
    _run(route, 1.0)
    assert route.walked
    held = _run(route, 0.9)
    assert {step.state for step in held} == {PedState.HOLDING}
    assert all(step.speed == 0.0 for step in held)
    assert route.step(DT).state is PedState.DONE
    assert route.step(DT).x == pytest.approx(1.0)


def test_route_hold_waits_while_arrival_is_not_confirmed() -> None:
    route = PedRoute([(0.0, 0.0), (1.0, 0.0)], yaw=0.0, speed=1.0, hold=0.5)
    _run(route, 1.0)
    assert {step.state for step in _run(route, 3.0, can_hold=False)} == {PedState.HOLDING}
    assert route.state is PedState.HOLDING
    assert route.step(DT, can_hold=True).state is PedState.HOLDING
    assert {step.state for step in _run(route, 0.5)} >= {PedState.DONE}


def test_route_turns_in_place_before_a_sharp_corner() -> None:
    route = PedRoute([(0.0, 0.0), (0.0, 4.0)], yaw=0.0, speed=1.0)
    first = route.step(DT)
    assert first.state is PedState.TURNING
    assert (first.x, first.y, first.speed) == (0.0, 0.0, 0.0)
    assert first.yaw == pytest.approx(YAW_RATE * DT)
    turning = [first]
    while (step := route.step(DT)).state is PedState.TURNING:
        turning.append(step)
    assert all((step.x, step.y) == (0.0, 0.0) for step in turning)
    assert step.state is PedState.WALKING
    assert step.y > 0.0
    assert step.x == pytest.approx(0.0)
    assert abs(step.yaw - math.pi / 2) < 0.6


def test_route_follows_gentle_bends_without_stopping() -> None:
    route = PedRoute([(0.0, 0.0), (2.0, 0.0), (4.0, 0.5)], yaw=0.0, speed=1.0)
    steps = _run(route, 4.0)
    assert {step.state for step in steps[:-1]} == {PedState.WALKING}


def test_route_turns_to_the_final_facing_before_it_is_done() -> None:
    route = PedRoute([(0.0, 0.0), (1.0, 0.0)], yaw=0.0, speed=1.0, hold=0.0, face=math.pi)
    _run(route, 1.0)
    first = route.step(DT)
    assert first.state is PedState.TURNING
    while (step := route.step(DT)).state is PedState.TURNING:
        pass
    assert step.state is PedState.DONE
    assert abs(math.remainder(step.yaw - math.pi, 2 * math.pi)) <= 0.05


def test_single_point_route_only_holds() -> None:
    route = PedRoute([(3.0, 2.0)], yaw=0.5, speed=1.0, hold=0.3)
    steps = _run(route, 0.5)
    assert steps[0].state is PedState.HOLDING
    assert (steps[0].x, steps[0].y) == (3.0, 2.0)
    assert steps[-1].state is PedState.DONE


def test_route_needs_its_start() -> None:
    with pytest.raises(ValueError, match="at least its start"):
        PedRoute([], yaw=0.0, speed=1.0)


def test_paused_clock_leaves_the_pedestrian_where_it_is() -> None:
    route = PedRoute([(0.0, 0.0), (4.0, 0.0)], yaw=0.0, speed=1.0)
    step = route.step(0.0)
    assert (step.x, step.y, step.speed) == (0.0, 0.0, 0.0)


def _walled_grid(gap_rows: slice | None) -> OccupancyGrid:
    occupied = np.zeros((100, 100), dtype=bool)
    occupied[:, 49:52] = True
    if gap_rows is not None:
        occupied[gap_rows, 49:52] = False
    return OccupancyGrid(occupied=occupied, res=0.1, ox=0.0, oy=0.0)


def test_plan_legs_detours_through_the_gap_of_a_wall() -> None:
    grid = _walled_grid(slice(0, 20))
    route = plan_legs(grid, [(2.0, 2.0), (8.0, 2.0)], clearance=0.04)
    assert route[0] == (2.0, 2.0)
    assert route[-1] == (8.0, 2.0)
    assert max(y for _, y in route) > 7.9
    assert all(y > 7.8 for x, y in iter_points(route, 0.05) if 4.9 < x < 5.4)


def test_plan_legs_chains_the_legs_in_order() -> None:
    grid = _walled_grid(slice(0, 20))
    route = plan_legs(grid, [(2.0, 2.0), (8.0, 2.0), (8.0, 6.0)], clearance=0.04)
    assert route[-1] == (8.0, 6.0)
    assert (8.0, 2.0) in route


def test_plan_legs_keeps_a_single_start_as_the_route() -> None:
    assert plan_legs(_walled_grid(None), [(2.0, 2.0)], clearance=0.04) == [(2.0, 2.0)]


def test_plan_legs_fails_when_the_wall_has_no_gap() -> None:
    with pytest.raises(PathError):
        plan_legs(_walled_grid(None), [(2.0, 2.0), (8.0, 2.0)], clearance=0.04)
