"""The casper task layer: task_generator's modes, with CASPER++ allocating and dispatching the team robots' tasks."""

from __future__ import annotations

import math
import typing
from collections.abc import Collection, Mapping

import attrs
from arena_rclpy_mixins.params import Choices, Param, Range, declare
from arena_runtime.env.task_api import NodeInterface, PhaseState, RobotManager, TaskKind
from arena_runtime.env.task_layer import EpisodeRecord, ResetPlan, TaskContext, TaskEvent, TaskModes
from casper.allocation import Allocation, Mode, allocate
from casper.goal import Goal, load_goal
from casper.schedule import Schedule
from task_generator.tasks import layer as task_generator_layer
from task_generator.tasks.layer import declare_mode_schema, fleet_only_robots, mode_namespaces, next_episode, validate_modes

from arena_casper.realize import ArmReach, request_for
from arena_casper.scene import Scene, Site, load_scene, make_site, validate_scene
from arena_casper.team import load_team

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from arena_runtime.env.task_layer import TaskParams
    from task_generator.tasks.task import Task

__all__ = [
    "CasperParams",
    "CasperTask",
    "create_task",
    "declare_mode_schema",
    "declare_task_params",
    "fleet_only_robots",
    "mode_namespaces",
    "next_episode",
    "validate_modes",
]

PARAM_NAMESPACE = "task.casper"
STOW = "stow"


def _mode(value: object) -> Mode:
    try:
        return Mode(str(value))
    except ValueError:
        raise ValueError(f"{PARAM_NAMESPACE}.mode {value!r} is none of {[mode.value for mode in Mode]}") from None


class CasperParams:
    """task_generator's mode selection plus the task.casper.* parameters."""

    def __init__(self, server: ROSParamServer) -> None:
        self.modes = task_generator_layer.declare_task_params(server)
        self.GOAL = declare(server, Param[str](f"{PARAM_NAMESPACE}.goal", "", "CASPER goal JSON file."))
        self.TEAM = declare(server, Param[str](f"{PARAM_NAMESPACE}.team", "", "Team YAML file."))
        self.MODE = declare(server, Param[Mode](f"{PARAM_NAMESPACE}.mode", Mode.SKILL.value, "CASPER++ allocation mode.", constraint=Choices(*(mode.value for mode in Mode)), parse=_mode))
        self.SCENE = declare(server, Param[str](f"{PARAM_NAMESPACE}.scene", "", "Scene file binding tasks to entities and objects, empty for none."))
        self.TOP_K = declare(server, Param[int](f"{PARAM_NAMESPACE}.top_k", 3, "Fallback candidates kept per task.", constraint=Range(1, None)))
        self.DURATION_SCALE = declare(server, Param[float](f"{PARAM_NAMESPACE}.duration_scale", 1.0, "Factor from a task's duration to its hold in seconds.", constraint=Range(0.0, None)))

    def resolve(self, modes: TaskModes) -> TaskModes:
        return self.modes.resolve(modes)

    def select(self, modes: TaskModes) -> None:
        self.modes.select(modes)


def declare_task_params(node: ROSParamServer) -> CasperParams:
    return CasperParams(node)


def derived_constraints(manager: RobotManager) -> dict[str, bool]:
    return {"can_move": TaskKind.GOTO_POSE in manager.accepts, "can_manipulate": TaskKind.REACH_POSE in manager.accepts}


def arm_reach(manager: RobotManager) -> ArmReach | None:
    """The reach of the robot's first box arm workspace: its depth middle as standoff, half its depth as slack, half its lateral angle there as yaw slack, its `stow` named pose when it declares one and the TF reach of the hand past the arm's tip link."""
    from arena_runtime.tf_frames import reach_below

    caps = manager.robot_view.effective_caps(manager.robot.resolved_request, frames=manager.robot.frames)
    for arm in (caps.arm or {}).values():
        ws = arm.workspace
        if ws is not None and ws.get("type") == "box" and isinstance(lo := ws.get("min"), list) and isinstance(hi := ws.get("max"), list):
            standoff = (float(lo[0]) + float(hi[0])) / 2
            return ArmReach(
                standoff=standoff,
                slack=(float(hi[0]) - float(lo[0])) / 2,
                yaw_slack=math.atan2(min(-float(lo[1]), float(hi[1])), standoff) / 2,
                stow=STOW if STOW in arm.named_poses else "",
                hand=reach_below(manager.tf_buffer, manager.frame.tf(arm.tip_link)),
            )
    return None


@attrs.define
class _Dispatch:
    task: str
    request: int


class CasperTask(NodeInterface["ArenaMixinNode"]):
    """task_generator's Task for placement, obstacles and modules, CASPER++ for the tasks of the team robots."""

    def __init__(self, *, node: ArenaMixinNode, ctx: TaskContext, params: CasperParams, inner: Task) -> None:
        super().__init__(node=node)
        self._ctx = ctx
        self._params = params
        self._inner = inner
        self._goal: Goal | None = None
        self._schedule: Schedule | None = None
        self._members: tuple[str, ...] = ()
        self._dispatched: dict[str, _Dispatch] = {}
        self._outside: set[str] = set()
        self._scene: Scene | None = None
        self._site: Site | None = None
        self._cleared = False

    @property
    def schedule(self) -> Schedule | None:
        return self._schedule

    @property
    async def done(self) -> bool:
        return self._schedule is not None and self._schedule.done

    def _load(self) -> tuple[Goal, Schedule, Allocation]:
        goal_path, team_path = self._params.GOAL.value, self._params.TEAM.value
        if not goal_path or not team_path:
            raise ValueError(f"task_layer:=casper needs {PARAM_NAMESPACE}.goal and {PARAM_NAMESPACE}.team, got goal={goal_path!r} team={team_path!r}")
        goal = load_goal(goal_path)
        zones = set(self._ctx.world_manager.world_compacted().zone_ref_names())
        if unknown := sorted({zone for task in goal.tasks.values() for zone in task.workspaces or () if zone not in zones}):
            raise ValueError(f"{goal_path}: workspaces {unknown} are no zones of this world, zones: {sorted(zones)}")
        team = load_team(team_path, {name: derived_constraints(manager) for name, manager in self._ctx.robots.items()})
        allocation = allocate(goal, team, self._params.MODE.value, self._params.TOP_K.value)
        self._members = tuple(team.members)
        for task_id, assignment in allocation.assignments.items():
            self._logger.info(f"casper {allocation.mode.value}: {task_id} -> {assignment.member} ({assignment.score:.2f}), fallbacks {list(assignment.fallbacks)}")
        scene_path = self._params.SCENE.value
        self._scene = load_scene(scene_path) if scene_path else None
        self._site = None
        if self._scene is not None:
            validate_scene(self._scene, goal, self._ctx)
            self._site = make_site(self._ctx)
        return goal, Schedule(goal, allocation), allocation

    async def reset_robots(self, record: EpisodeRecord) -> ResetPlan:
        """The inner robot mode's plan without tasks for the team robots, which get theirs on the episode's ticks."""
        plan = await self._inner.reset_robots(record)
        self._goal, self._schedule, allocation = self._load()
        self._dispatched.clear()
        self._outside.clear()
        self._cleared = self._schedule.cleared
        self._report_allocation(allocation)
        for name in self._members:
            entry = plan.robots.get(name)
            if entry is not None and entry.task is not None:
                plan.robots[name] = attrs.evolve(entry, task=None)
        return plan

    def _report_allocation(self, allocation: Allocation) -> None:
        report = self._ctx.episode.task_event
        report(
            TaskEvent(
                task="",
                kind="allocated",
                detail={
                    "goal": self._params.GOAL.value,
                    "team": self._params.TEAM.value,
                    "mode": allocation.mode.value,
                    "top_k": self._params.TOP_K.value,
                    "assignments": {task_id: {"member": a.member, "score": a.score, "fallbacks": list(a.fallbacks)} for task_id, a in allocation.assignments.items()},
                    "ranking": {task_id: [[member, score] for member, score in scored] for task_id, scored in allocation.ranking.items()},
                    "workload": {member: sum(1 for a in allocation.assignments.values() if a.member == member) for member in self._members},
                    "initial": {task_id: {"member": a.member, "score": a.score} for task_id, a in allocation.initial.items()},
                },
            )
        )
        for task_id, assignment in allocation.assignments.items():
            report(TaskEvent(task=task_id, kind="assigned", robot=assignment.member, detail={"score": assignment.score, "fallbacks": list(assignment.fallbacks)}))

    async def reset_obstacles(self, plan: ResetPlan, record: EpisodeRecord) -> None:
        await self._inner.reset_obstacles(plan, record)

    def after_reset(self) -> None:
        self._inner.after_reset()

    async def teardown(self) -> None:
        await self._inner.teardown()

    def _fail(self, schedule: Schedule, task_id: str, member: str) -> bool:
        """Report the failure and the hand-over to the next candidate, False once none is left and the episode is aborted."""
        report = self._ctx.episode.task_event
        successor = schedule.fail(task_id)
        report(TaskEvent(task=task_id, kind="failed", robot=member))
        if successor is None:
            self._ctx.episode.abort_episode(f"casper task {task_id} failed on every candidate")
            return False
        report(TaskEvent(task=task_id, kind="reassigned", robot=successor, detail={"from": member}))
        self._logger.warning(f"casper task {task_id} failed on {member}, handed to {successor}")
        return True

    async def tick(self, ended: Mapping[str, PhaseState], assigned: Collection[str]) -> None:
        """Settle the tasks whose request ended, then start each idle team robot's next ready task."""
        await self._inner.tick(ended, assigned)
        goal, schedule = self._goal, self._schedule
        if goal is None or schedule is None:
            return
        report = self._ctx.episode.task_event
        for robot in set(assigned) & set(self._members):
            self._outside.add(robot)
            if (dispatch := self._dispatched.pop(robot, None)) is not None:
                schedule.interrupt(dispatch.task)
                report(TaskEvent(task=dispatch.task, kind="dropped", robot=robot))
        for robot, dispatch in list(self._dispatched.items()):
            outcome = self._ctx.robots[robot].runner.request_outcome(dispatch.request)
            if outcome is None:
                continue
            del self._dispatched[robot]
            if outcome == "met":
                schedule.finish(dispatch.task)
                report(TaskEvent(task=dispatch.task, kind="met", robot=robot))
            elif outcome == "failed":
                if not self._fail(schedule, dispatch.task, robot):
                    return
            else:
                schedule.interrupt(dispatch.task)
                report(TaskEvent(task=dispatch.task, kind="dropped", robot=robot))
        if schedule.cleared and not self._cleared:
            for task_id, task in goal.tasks.items():
                if task.awaits_clearance:
                    report(TaskEvent(task=task_id, kind="released", robot=schedule.states[task_id].owner))
        self._cleared = schedule.cleared
        for robot in list(self._outside):
            if await self._ctx.robots[robot].is_done:
                self._outside.discard(robot)
        for robot in self._members:
            if robot in self._outside or robot in self._dispatched:
                continue
            task_id = schedule.next_for(robot)
            if task_id is None:
                continue
            schedule.start(task_id)
            manager = self._ctx.robots[robot]
            request = request_for(goal.tasks[task_id], self._params.DURATION_SCALE.value, self._scene, self._site, arm_reach(manager) if self._scene is not None else None)
            report(TaskEvent(task=task_id, kind="started", robot=robot, detail={"phases": len(request.phases)}))
            try:
                await manager.submit_task(request)
            except ValueError as e:
                self._logger.warning(f"casper {robot} cannot run {task_id}: {e}")
                if not self._fail(schedule, task_id, robot):
                    return
                continue
            self._dispatched[robot] = _Dispatch(task=task_id, request=manager.runner.phases[-1].request)
            self._logger.info(f"casper {robot} starts {task_id}")


async def create_task(*, node: ArenaMixinNode, ctx: TaskContext, params: TaskParams) -> CasperTask:
    casper = typing.cast("CasperParams", params)
    inner = await task_generator_layer.create_task(node=node, ctx=ctx, params=casper.modes)
    return CasperTask(node=node, ctx=ctx, params=casper, inner=inner)
