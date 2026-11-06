from __future__ import annotations

import asyncio
import json
import math
import pathlib
from types import SimpleNamespace

import pytest
import yaml

ZONES = ("reception", "kitchen", "garden")


@pytest.fixture(autouse=True)
def _ros_gate() -> None:
    pytest.importorskip("rclpy")


class _Manager:
    def __init__(self, name: str, kinds: tuple[str, ...] = ("goto_pose",)) -> None:
        from arena_runtime.env.robot_manager.task_runner import TaskRunner
        from arena_runtime.env.task_api import TaskKind

        self.name = name
        self.accepts = frozenset(TaskKind(kind) for kind in kinds)
        self.runner = TaskRunner(robot=name)
        self.requests: list[object] = []

    @property
    async def is_done(self) -> bool:
        return self.runner.done

    async def submit_task(self, request: object, start: object = None, pedestrians: object = ()) -> None:
        self.requests.append(request)
        self.runner.submit(request.phases, request.conditions)

    def end(self, outcome: str) -> None:
        for state in self.runner.phases:
            if state.outcome == "pending":
                state.outcome = outcome
        self.runner.active = None


class _Inner:
    def __init__(self, plan: object) -> None:
        self.plan = plan
        self.ticks: list[tuple[dict[str, object], list[str]]] = []

    async def reset_robots(self, record: object) -> object:
        return self.plan

    async def tick(self, ended: dict[str, object], assigned: list[str]) -> None:
        self.ticks.append((dict(ended), list(assigned)))


def _write(tmp_path: pathlib.Path, tasks: list[dict[str, object]], dependencies: list[list[str]], members: dict[str, dict[str, object]], supervisor: str) -> tuple[pathlib.Path, pathlib.Path]:
    goal = tmp_path / "goal.json"
    goal.write_text(json.dumps({"tasks": [{"context": "party", "required_skills": {"work": 0.1}, **task} for task in tasks], "dependencies": dependencies}))
    team = tmp_path / "team.yaml"
    team.write_text(yaml.safe_dump({"supervisor": supervisor, "members": members}))
    return goal, team


def _member(level: float, workspace: list[str] | None = None) -> dict[str, object]:
    return {"skills": {"party": {"work": [level, 1]}}, "workspace": workspace or list(ZONES)}


def _casper(tmp_path: pathlib.Path, tasks: list[dict[str, object]], dependencies: list[list[str]], members: dict[str, dict[str, object]], managers: list[_Manager], plan: object = None, duration_scale: float = 1.0) -> tuple[object, _Inner, list[str]]:
    import rclpy.logging
    from arena_runtime.env.task_layer import ResetPlan

    from casper.allocation import Mode
    from arena_casper.layer import CasperTask

    goal, team = _write(tmp_path, tasks, dependencies, members, supervisor=next(iter(members)))
    aborted: list[str] = []
    events: list[object] = []
    ctx = SimpleNamespace(
        robots={manager.name: manager for manager in managers},
        world_manager=SimpleNamespace(world_compacted=lambda: SimpleNamespace(zone_ref_names=lambda: list(ZONES))),
        episode=SimpleNamespace(abort_episode=aborted.append, task_event=events.append),
    )
    params = SimpleNamespace(
        GOAL=SimpleNamespace(value=str(goal)),
        TEAM=SimpleNamespace(value=str(team)),
        MODE=SimpleNamespace(value=Mode.SKILL),
        SCENE=SimpleNamespace(value=""),
        TOP_K=SimpleNamespace(value=3),
        DURATION_SCALE=SimpleNamespace(value=duration_scale),
    )
    inner = _Inner(plan if plan is not None else ResetPlan())
    node = SimpleNamespace(get_logger=lambda: rclpy.logging.get_logger("test_casper_layer"))
    task = CasperTask(node=node, ctx=ctx, params=params, inner=inner)
    task.events = events
    return task, inner, aborted


def _kinds(task: object, *, skip: tuple[str, ...] = ("allocated", "assigned")) -> list[tuple[str, str, str]]:
    return [(event.task, event.kind, event.robot) for event in task.events if event.kind not in skip]


def _targets(request: object) -> list[tuple[str | None, float | None]]:
    return [(phase.target, phase.hold_time) for phase in request.phases]


def _record() -> object:
    from arena_runtime.env.task_layer import EpisodeRecord

    return EpisodeRecord()


def test_casper_is_a_registered_task_layer() -> None:
    from arena_runtime.env.task_layer import task_layer

    layer = task_layer("casper")
    for name in ("declare_task_params", "create_task", "validate_modes", "declare_mode_schema", "mode_namespaces", "fleet_only_robots", "next_episode"):
        assert callable(getattr(layer, name))


def test_task_runs_through_its_workspaces_and_holds_at_the_last() -> None:
    from casper.goal import parse_task
    from arena_casper.realize import request_for

    task = parse_task({"id": "serve", "context": "party", "required_constraints": {"workspace": ["kitchen", "garden"]}, "duration": 40})
    assert _targets(request_for(task, 0.5)) == [("kitchen", None), ("garden", 20.0)]
    idle = parse_task({"id": "wait", "context": "party", "duration": 10})
    assert _targets(request_for(idle, 1.0)) == [(None, 10.0)]


def test_reset_keeps_start_poses_and_clears_the_team_robots_tasks(tmp_path: pathlib.Path) -> None:
    from arena_runtime.env.task_api import GoToPhase, TaskRequest
    from arena_runtime.env.task_layer import ResetPlan, RobotReset
    from arena_runtime.shared import Pose

    start = Pose()
    own = TaskRequest(phases=[GoToPhase(target="garden")])
    plan = ResetPlan(robots={"r0": RobotReset(start_pose=start, task=own), "visitor": RobotReset(start_pose=start, task=own)})
    task, _, _ = _casper(tmp_path, [{"id": "cook"}], [], {"r0": _member(0.9)}, [_Manager("r0"), _Manager("visitor")], plan=plan)
    plan = asyncio.run(task.reset_robots(_record()))
    assert plan.robots["r0"].task is None
    assert plan.robots["r0"].start_pose is start
    assert plan.robots["visitor"].task is own


def test_tick_dispatches_ready_tasks_and_follows_the_dependencies(tmp_path: pathlib.Path) -> None:
    r0, r1 = _Manager("r0"), _Manager("r1")
    tasks = [
        {"id": "cook", "priority": 1, "duration": 30, "required_constraints": {"workspace": ["kitchen"]}},
        {"id": "greet", "priority": 2, "duration": 10, "required_constraints": {"workspace": ["reception"]}},
        {"id": "serve", "priority": 3, "duration": 5, "required_constraints": {"workspace": ["kitchen", "garden"]}},
    ]
    task, inner, _ = _casper(tmp_path, tasks, [["cook", "serve"]], {"r0": _member(0.9), "r1": _member(0.9)}, [r0, r1], duration_scale=0.1)
    asyncio.run(task.reset_robots(_record()))
    asyncio.run(task.tick({}, []))
    assert [_targets(request) for request in r0.requests] == [[("kitchen", pytest.approx(3.0))]]
    assert [_targets(request) for request in r1.requests] == [[("reception", pytest.approx(1.0))]]
    r1.end("met")
    asyncio.run(task.tick({}, []))
    assert len(r1.requests) == 1
    r0.end("met")
    asyncio.run(task.tick({}, []))
    assert _targets(r0.requests[-1]) == [("kitchen", None), ("garden", pytest.approx(0.5))]
    assert not asyncio.run(task.done)
    r0.end("met")
    asyncio.run(task.tick({}, []))
    assert asyncio.run(task.done)
    assert len(inner.ticks) == 4


def test_failed_task_goes_to_its_fallback_robot(tmp_path: pathlib.Path) -> None:
    r0, r1 = _Manager("r0"), _Manager("r1")
    task, _, aborted = _casper(tmp_path, [{"id": "cook", "required_constraints": {"workspace": ["kitchen"]}}], [], {"r0": _member(0.9), "r1": _member(0.5)}, [r0, r1])
    asyncio.run(task.reset_robots(_record()))
    asyncio.run(task.tick({}, []))
    r0.end("failed")
    asyncio.run(task.tick({}, []))
    assert len(r0.requests) == 1
    assert _targets(r1.requests[0]) == [("kitchen", 0.0)]
    assert task.schedule.states["cook"].owner == "r1"
    assert aborted == []


def test_task_failed_on_every_candidate_aborts_the_episode(tmp_path: pathlib.Path) -> None:
    r0 = _Manager("r0")
    task, _, aborted = _casper(tmp_path, [{"id": "cook", "required_constraints": {"workspace": ["kitchen"]}}], [], {"r0": _member(0.9)}, [r0])
    asyncio.run(task.reset_robots(_record()))
    asyncio.run(task.tick({}, []))
    r0.end("failed")
    asyncio.run(task.tick({}, []))
    assert aborted == ["casper task cook failed on every candidate"]


def test_robot_assigned_from_outside_resumes_its_task_after(tmp_path: pathlib.Path) -> None:
    from arena_runtime.env.task_api import GoToPhase

    r0 = _Manager("r0")
    task, _, _ = _casper(tmp_path, [{"id": "cook", "required_constraints": {"workspace": ["kitchen"]}}], [], {"r0": _member(0.9)}, [r0])
    asyncio.run(task.reset_robots(_record()))
    asyncio.run(task.tick({}, []))
    r0.runner.submit([GoToPhase(target="garden")])
    asyncio.run(task.tick({}, ["r0"]))
    assert len(r0.requests) == 1
    r0.end("met")
    asyncio.run(task.tick({}, []))
    assert [_targets(request) for request in r0.requests] == [[("kitchen", 0.0)], [("kitchen", 0.0)]]


def test_reset_refuses_workspaces_the_world_lacks(tmp_path: pathlib.Path) -> None:
    task, _, _ = _casper(tmp_path, [{"id": "grill", "required_constraints": {"workspace": ["Grill"]}}], [], {"r0": _member(0.9, workspace=["Grill"])}, [_Manager("r0")])
    with pytest.raises(ValueError, match=r"workspaces \['Grill'\] are no zones of this world"):
        asyncio.run(task.reset_robots(_record()))


def test_team_constraints_come_from_the_robots_adapters(tmp_path: pathlib.Path) -> None:
    from casper.allocation import AllocationError

    arm, base = _Manager("arm", kinds=("goto_pose", "reach_pose")), _Manager("base")
    task, _, _ = _casper(tmp_path, [{"id": "chop", "required_constraints": {"can_manipulate": True, "workspace": ["kitchen"]}}], [], {"base": _member(0.9), "arm": _member(0.5)}, [arm, base])
    asyncio.run(task.reset_robots(_record()))
    assert task.schedule.states["chop"].owner == "arm"
    task, _, _ = _casper(tmp_path, [{"id": "chop", "required_constraints": {"can_manipulate": True}}], [], {"base": _member(0.9)}, [base])
    with pytest.raises(AllocationError):
        asyncio.run(task.reset_robots(_record()))


def test_reset_reports_the_allocation_and_each_assignment(tmp_path: pathlib.Path) -> None:
    tasks = [{"id": "cook", "required_constraints": {"workspace": ["kitchen"]}}, {"id": "greet", "required_constraints": {"workspace": ["reception"]}}]
    task, _, _ = _casper(tmp_path, tasks, [], {"r0": _member(0.9), "r1": _member(0.5, ["kitchen"])}, [_Manager("r0"), _Manager("r1")])
    asyncio.run(task.reset_robots(_record()))
    allocated, *assigned = task.events
    assert (allocated.task, allocated.kind, allocated.robot) == ("", "allocated", "")
    assert allocated.detail["mode"] == "skill"
    assert allocated.detail["top_k"] == 3
    assert allocated.detail["goal"].endswith("goal.json")
    assert allocated.detail["team"].endswith("team.yaml")
    assert allocated.detail["assignments"] == {"cook": {"member": "r0", "score": 0.9, "fallbacks": ["r1"]}, "greet": {"member": "r0", "score": 0.9, "fallbacks": []}}
    assert allocated.detail["ranking"] == {"cook": [["r0", 0.9], ["r1", 0.5]], "greet": [["r0", 0.9]]}
    assert allocated.detail["workload"] == {"r0": 2, "r1": 0}
    assert [(event.task, event.kind, event.robot, event.detail) for event in assigned] == [
        ("cook", "assigned", "r0", {"score": 0.9, "fallbacks": ["r1"]}),
        ("greet", "assigned", "r0", {"score": 0.9, "fallbacks": []}),
    ]


def test_tick_reports_start_and_outcome(tmp_path: pathlib.Path) -> None:
    r0 = _Manager("r0")
    tasks = [{"id": "cook", "priority": 1, "required_constraints": {"workspace": ["kitchen", "garden"]}}, {"id": "greet", "priority": 2}]
    task, _, _ = _casper(tmp_path, tasks, [["cook", "greet"]], {"r0": _member(0.9)}, [r0])
    asyncio.run(task.reset_robots(_record()))
    asyncio.run(task.tick({}, []))
    r0.end("met")
    asyncio.run(task.tick({}, []))
    assert _kinds(task) == [("cook", "started", "r0"), ("cook", "met", "r0"), ("greet", "started", "r0")]
    assert next(event.detail for event in task.events if event.kind == "started") == {"phases": 2}


def test_failure_reports_the_fallback_owner(tmp_path: pathlib.Path) -> None:
    r0, r1 = _Manager("r0"), _Manager("r1")
    task, _, _ = _casper(tmp_path, [{"id": "cook", "required_constraints": {"workspace": ["kitchen"]}}], [], {"r0": _member(0.9), "r1": _member(0.5)}, [r0, r1])
    asyncio.run(task.reset_robots(_record()))
    asyncio.run(task.tick({}, []))
    r0.end("failed")
    asyncio.run(task.tick({}, []))
    assert _kinds(task) == [("cook", "started", "r0"), ("cook", "failed", "r0"), ("cook", "reassigned", "r1"), ("cook", "started", "r1")]
    assert next(event.detail for event in task.events if event.kind == "reassigned") == {"from": "r0"}


def test_failure_without_a_fallback_reports_failed_only(tmp_path: pathlib.Path) -> None:
    r0 = _Manager("r0")
    task, _, _ = _casper(tmp_path, [{"id": "cook"}], [], {"r0": _member(0.9)}, [r0])
    asyncio.run(task.reset_robots(_record()))
    asyncio.run(task.tick({}, []))
    r0.end("failed")
    asyncio.run(task.tick({}, []))
    assert _kinds(task) == [("cook", "started", "r0"), ("cook", "failed", "r0")]


def test_task_dropped_when_its_request_ends_otherwise(tmp_path: pathlib.Path) -> None:
    r0 = _Manager("r0")
    task, _, _ = _casper(tmp_path, [{"id": "cook"}], [], {"r0": _member(0.9)}, [r0])
    asyncio.run(task.reset_robots(_record()))
    asyncio.run(task.tick({}, []))
    r0.runner.stop()
    asyncio.run(task.tick({}, []))
    assert _kinds(task)[:2] == [("cook", "started", "r0"), ("cook", "dropped", "r0")]


def test_task_dropped_when_the_robot_is_assigned_from_outside(tmp_path: pathlib.Path) -> None:
    from arena_runtime.env.task_api import GoToPhase

    r0 = _Manager("r0")
    task, _, _ = _casper(tmp_path, [{"id": "cook"}], [], {"r0": _member(0.9)}, [r0])
    asyncio.run(task.reset_robots(_record()))
    asyncio.run(task.tick({}, []))
    r0.runner.submit([GoToPhase(target="garden")])
    asyncio.run(task.tick({}, ["r0"]))
    assert _kinds(task) == [("cook", "started", "r0"), ("cook", "dropped", "r0")]


def test_clearance_opening_releases_the_waiting_task(tmp_path: pathlib.Path) -> None:
    r0 = _Manager("r0")
    tasks = [{"id": "welcome", "priority": 1, "subgoal_level": 1}, {"id": "tidy", "priority": 2, "time_to_clean": False}]
    task, _, _ = _casper(tmp_path, tasks, [], {"r0": _member(0.9)}, [r0])
    asyncio.run(task.reset_robots(_record()))
    asyncio.run(task.tick({}, []))
    asyncio.run(task.tick({}, []))
    assert _kinds(task) == [("welcome", "started", "r0")]
    r0.end("met")
    asyncio.run(task.tick({}, []))
    asyncio.run(task.tick({}, []))
    assert _kinds(task) == [("welcome", "started", "r0"), ("welcome", "met", "r0"), ("tidy", "released", "r0"), ("tidy", "started", "r0")]


def _view(model: str) -> SimpleNamespace:
    from arena_robots.Robot import RobotIdentifier

    return SimpleNamespace(robot_view=RobotIdentifier(model).resolve_sync(), robot=SimpleNamespace(resolved_request={}, frames=None))


def test_arm_reach_comes_from_the_arm_workspace_box() -> None:
    from arena_casper.layer import arm_reach

    reach = arm_reach(_view("tiago"))
    assert reach is not None
    assert (reach.standoff, reach.slack) == (pytest.approx(0.6), pytest.approx(0.3))
    assert reach.yaw_slack == pytest.approx(math.atan2(0.5, 0.6) / 2)
    assert reach.stow == "stow"


def test_a_robot_without_an_arm_has_no_reach() -> None:
    from arena_casper.layer import arm_reach

    assert arm_reach(_view("jackal")) is None
