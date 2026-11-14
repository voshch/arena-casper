from __future__ import annotations

import asyncio
import json
import math
import pathlib
from types import SimpleNamespace

import pytest
import yaml

SHIFT = 100.0
FPS = 30.0
ZONES = {"kitchen": ((1.0, 4.0), (3.0, 6.0)), "garden": ((7.0, 1.0), (9.0, 3.0))}


@pytest.fixture(autouse=True)
def _ros_gate() -> None:
    pytest.importorskip("rclpy")


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def to_seconds(self) -> float:
        return self.now

    def to_msg(self) -> object:
        from builtin_interfaces.msg import Time

        nanoseconds = round(self.now * 1e9)
        return Time(sec=nanoseconds // 10**9, nanosec=nanoseconds % 10**9)


class _Publisher:
    def __init__(self) -> None:
        self.batches: list[object] = []

    def publish(self, msg: object) -> None:
        self.batches.append(msg)


class _Node:
    def __init__(self) -> None:
        import rclpy.logging

        self.sim_time = _Clock()
        self.stream = _Publisher()
        self.callbacks: dict[str, object] = {}
        self.publishers: dict[str, object] = {}
        self._logger = rclpy.logging.get_logger("test_pedestrians")

    def get_namespace(self) -> str:
        return "/env"

    def get_logger(self) -> object:
        return self._logger

    def create_subscription(self, msg_type: object, topic: str, callback: object, qos: object) -> str:
        self.callbacks[topic] = callback
        return topic

    def create_publisher(self, msg_type: object, topic: str, qos: object) -> _Publisher:
        self.publishers[topic] = qos
        return self.stream

    def destroy_subscription(self, subscription: str) -> None:
        self.callbacks.pop(subscription, None)

    def destroy_publisher(self, publisher: object) -> None:
        self.publishers.clear()


class _Level:
    stairs: list[object] = []
    objects: list[object] = []

    def zone_ref_names(self) -> list[str]:
        return list(ZONES)

    def lookup_zone_polygon(self, name: str) -> list[object] | None:
        from arena_simulation_setup.utils.geometry import Position

        if name not in ZONES:
            return None
        (x0, y0), (x1, y1) = ZONES[name]
        return [Position(x0, y0), Position(x1, y0), Position(x1, y1), Position(x0, y1)]


def _realize(pose: object) -> object:
    from arena_simulation_setup.utils.geometry import Pose, Position

    return Pose(Position(pose.position.x + SHIFT, pose.position.y, pose.position.z), pose.orientation)


def _ezilear(pose: object) -> object:
    from arena_simulation_setup.utils.geometry import Pose, Position

    return Pose(Position(pose.position.x - SHIFT, pose.position.y, pose.position.z), pose.orientation)


def _prefix(*parts: str) -> str:
    return "/".join(("env", *(part for part in parts if part)))


class _World:
    def __init__(self) -> None:
        import numpy as np

        self.node = _Node()
        self._echoed = 0
        walls = np.full((100, 100), 255, dtype=np.uint8)
        self.ctx = SimpleNamespace(
            environment_manager=SimpleNamespace(realize=_realize, ezilear=_ezilear),
            realizer=SimpleNamespace(prefix=_prefix),
            world_manager=SimpleNamespace(
                map=SimpleNamespace(occupancy=SimpleNamespace(grid=walls), resolution=0.1, origin=SimpleNamespace(x=0.0, y=0.0)),
                world_compacted=_Level,
            ),
        )

    def appear(self, name: str, ident: int, x: float, y: float, yaw: float = 0.0) -> None:
        from arena_people_msgs.msg import Pedestrian, Pedestrians
        from arena_simulation_setup.utils.geometry import Orientation

        ped = Pedestrian()
        ped.name = name
        ped.id = ident
        ped.pose.position.x = x + SHIFT
        ped.pose.position.y = y
        q = Orientation.from_yaw(yaw)
        ped.pose.orientation.x, ped.pose.orientation.y, ped.pose.orientation.z, ped.pose.orientation.w = q.x, q.y, q.z, q.w
        msg = Pedestrians()
        msg.pedestrians = [ped]
        self.node.callbacks["/env/arena_peds"](msg)

    def echo(self) -> None:
        for batch in self.node.stream.batches[self._echoed :]:
            if batch.pedestrians:
                self.node.callbacks["/env/arena_peds"](batch)
        self._echoed = len(self.node.stream.batches)

    @property
    def batches(self) -> list[object]:
        return self.node.stream.batches

    def claimed(self) -> list[object]:
        return [batch for batch in self.batches if batch.pedestrians]


def _obstacle(name: str = "walker", level: str = "", velocity: float = 1.0, waypoints: bool = True) -> object:
    from arena_simulation_setup.shared.entities import DynamicObstacle
    from arena_simulation_setup.utils.geometry import Pose, Position

    return DynamicObstacle(
        name=name,
        model="arenian",
        pose=Pose(Position(5.0, 5.0)),
        level=level,
        velocity=velocity,
        waypoints=[Position(9.0, 9.0), Position(1.0, 9.0)] if waypoints else [],
    )


async def _run(world: _World, seconds: float, driver: object) -> tuple[list[str], list[str]]:
    met: list[str] = []
    failed: list[str] = []
    for _ in range(round(seconds * FPS)):
        world.node.sim_time.now += 1.0 / FPS
        done, bad = await driver.tick()
        met += done
        failed += bad
        world.echo()
        if met or failed:
            break
    return met, failed


def test_walking_pedestrian_has_a_filled_joint_state() -> None:
    from arena_people_msgs.msg import Pedestrian
    from arena_runtime.humans.gait import GaitGenerator

    gait = GaitGenerator()
    joints = gait.joint_state(gait.compute(3, Pedestrian.WALKING, 1.0, 0.05))
    assert list(joints.name) == list(GaitGenerator.JOINT_NAMES)
    assert len(joints.position) == len(GaitGenerator.JOINT_NAMES)
    assert any(abs(angle) > 1e-3 for angle in joints.position)


def test_driver_walks_the_pedestrian_to_its_zone_and_streams_the_claim() -> None:
    from arena_people_msgs.msg import Pedestrian
    from arena_runtime.humans.gait import GaitGenerator

    from arena_casper.pedestrian import PedDriver

    world = _World()
    driver = PedDriver(world.node, world.ctx)
    world.appear("env/walker", 7, 5.0, 5.0)
    driver.start(_obstacle(), ["kitchen"], 1.0)
    met, failed = asyncio.run(_run(world, 30.0, driver))
    assert (met, failed) == (["walker"], [])
    claimed = world.claimed()
    assert {ped.name for batch in claimed for ped in batch.pedestrians} == {"env/walker"}
    assert all(batch.header.frame_id == "map" for batch in claimed)
    stamps = [batch.header.stamp.sec + batch.header.stamp.nanosec * 1e-9 for batch in claimed]
    assert stamps == sorted(stamps)
    assert stamps[-1] == pytest.approx(world.node.sim_time.now, abs=1e-6)
    walking = [ped for batch in claimed for ped in batch.pedestrians if ped.animation_state == Pedestrian.WALKING]
    assert walking
    assert all(len(ped.joint_state.name) == len(GaitGenerator.JOINT_NAMES) for batch in claimed for ped in batch.pedestrians)
    assert any(abs(angle) > 1e-3 for ped in walking for angle in ped.joint_state.position)
    speeds = [math.hypot(ped.twist.linear.x, ped.twist.linear.y) for ped in walking]
    assert max(speeds) == pytest.approx(1.0)
    assert min(speeds) > 0.0
    final = claimed[-1].pedestrians[0]
    assert (final.pose.position.x - SHIFT, final.pose.position.y, final.pose.position.z) == pytest.approx((2.0, 5.0, 0.0), abs=0.05)
    assert final.animation_state == Pedestrian.IDLE
    assert final.twist.linear.x == 0.0
    assert len(final.joint_state.name) == len(GaitGenerator.JOINT_NAMES)


def test_driver_keeps_claiming_a_pedestrian_between_tasks() -> None:
    from arena_casper.pedestrian import PedDriver

    world = _World()
    driver = PedDriver(world.node, world.ctx)
    world.appear("env/walker", 7, 5.0, 5.0)
    driver.start(_obstacle(), [], 0.2)
    assert asyncio.run(_run(world, 5.0, driver)) == (["walker"], [])
    before = len(world.claimed())

    async def idle() -> None:
        for _ in range(10):
            world.node.sim_time.now += 1.0 / FPS
            await driver.tick()

    asyncio.run(idle())
    assert len(world.claimed()) == before + 10


def test_driver_release_all_publishes_an_empty_stamped_batch() -> None:
    from arena_casper.pedestrian import PedDriver

    world = _World()
    driver = PedDriver(world.node, world.ctx)
    world.appear("env/walker", 7, 5.0, 5.0)
    driver.start(_obstacle(), ["kitchen"], 0.0)
    asyncio.run(_run(world, 0.2, driver))
    world.node.sim_time.now += 0.5
    driver.release_all()
    last = world.batches[-1]
    assert list(last.pedestrians) == []
    assert last.header.stamp.sec + last.header.stamp.nanosec * 1e-9 == pytest.approx(world.node.sim_time.now, abs=1e-6)
    count = len(world.batches)
    driver.release_all()
    assert len(world.batches) == count


@pytest.mark.parametrize("arena_name", ["env/walker", "env/walker/L1", "walker"])
def test_driver_claims_the_pedestrian_under_the_name_arena_peds_carries(arena_name: str) -> None:
    from arena_casper.pedestrian import PedDriver

    world = _World()
    driver = PedDriver(world.node, world.ctx)
    world.appear(arena_name, 4, 5.0, 5.0)
    driver.start(_obstacle(level="L1"), [], 0.1)
    asyncio.run(_run(world, 5.0, driver))
    assert {ped.name for batch in world.claimed() for ped in batch.pedestrians} == {arena_name}


def test_driver_fails_a_pedestrian_that_never_appears() -> None:
    from arena_casper.pedestrian import SPAWN_WAIT, PedDriver

    world = _World()
    driver = PedDriver(world.node, world.ctx)
    driver.start(_obstacle(), ["kitchen"], 0.0)
    met, failed = asyncio.run(_run(world, SPAWN_WAIT + 1.0, driver))
    assert (met, failed) == ([], ["walker"])
    assert world.batches == []


def test_driver_fails_a_walk_to_a_zone_walled_off() -> None:
    import numpy as np

    from arena_casper.pedestrian import PedDriver

    world = _World()
    walls = world.ctx.world_manager.map.occupancy.grid
    walls[:, 40:60] = np.uint8(0)
    driver = PedDriver(world.node, world.ctx)
    world.appear("env/walker", 7, 8.0, 2.0)
    driver.start(_obstacle(), ["kitchen"], 0.0)
    assert asyncio.run(_run(world, 2.0, driver)) == ([], ["walker"])


def test_driver_walks_around_an_occupied_static_obstacle() -> None:
    from arena_casper.pedestrian import PedDriver

    world = _World()
    world.ctx.world_manager.map.occupancy.grid[30:70, 35:40] = 0
    driver = PedDriver(world.node, world.ctx)
    world.appear("env/walker", 7, 5.0, 5.0)
    driver.start(_obstacle(), ["kitchen"], 0.0)
    assert asyncio.run(_run(world, 40.0, driver)) == (["walker"], [])
    positions = [(ped.pose.position.x - SHIFT, ped.pose.position.y) for batch in world.claimed() for ped in batch.pedestrians]
    assert positions
    assert not any(3.5 <= x <= 4.0 and 3.0 <= y <= 7.0 for x, y in positions)
    assert any(y < 3.0 or y > 7.0 for _, y in positions)


def test_driver_reset_forgets_claims_and_pedestrians_seen() -> None:
    from arena_casper.pedestrian import PedDriver

    world = _World()
    driver = PedDriver(world.node, world.ctx)
    world.appear("env/walker", 7, 5.0, 5.0)
    driver.start(_obstacle(), [], 5.0)
    asyncio.run(_run(world, 0.5, driver))
    driver.reset()
    count = len(world.batches)
    asyncio.run(_run(world, 0.5, driver))
    assert len(world.batches) == count


class _Manager:
    def __init__(self, name: str) -> None:
        from arena_runtime.env.robot_manager.task_runner import TaskRunner
        from arena_runtime.env.task_api import TaskKind

        self.name = name
        self.accepts = frozenset({TaskKind.GOTO_POSE})
        self.runner = TaskRunner(robot=name)
        self.requests: list[object] = []

    @property
    async def is_done(self) -> bool:
        return self.runner.done

    async def submit_task(self, request: object, start: object = None, pedestrians: object = ()) -> None:
        self.requests.append(request)
        self.runner.submit(request.phases, request.conditions)


class _Inner:
    def __init__(self, plan: object) -> None:
        self.plan = plan
        self.torn_down = False

    async def reset_robots(self, record: object) -> object:
        return self.plan

    async def reset_obstacles(self, plan: object, record: object) -> None:
        return None

    async def tick(self, ended: dict[str, object], assigned: list[str]) -> None:
        return None

    async def teardown(self) -> None:
        self.torn_down = True


def _walker(level: float, workspace: list[str] | None = None) -> dict[str, object]:
    return {"kind": "pedestrian", "skills": {"party": {"work": [level, 1]}}, "workspace": workspace or list(ZONES)}


def _robot(level: float, workspace: list[str] | None = None) -> dict[str, object]:
    return {"skills": {"party": {"work": [level, 1]}}, "workspace": workspace or list(ZONES)}


class _Layer:
    def __init__(self, tmp_path: pathlib.Path, tasks: list[dict[str, object]], members: dict[str, dict[str, object]], managers: tuple[_Manager, ...] = (), obstacles: tuple[object, ...] = (), duration_scale: float = 1.0) -> None:
        from arena_runtime.env.task_layer import ResetPlan
        from casper.allocation import Mode

        from arena_casper.layer import CasperTask

        goal, team = tmp_path / "goal.json", tmp_path / "team.yaml"
        goal.write_text(json.dumps({"tasks": [{"context": "party", "required_skills": {"work": 0.1}, **task} for task in tasks], "dependencies": []}))
        team.write_text(yaml.safe_dump({"supervisor": next(iter(members)), "members": members}))
        self.world = _World()
        self.aborted: list[str] = []
        self.events: list[object] = []
        self.managers = managers
        self.world.ctx.robots = {manager.name: manager for manager in managers}
        self.world.ctx.episode = SimpleNamespace(abort_episode=self.aborted.append, task_event=self.events.append)
        params = SimpleNamespace(
            GOAL=SimpleNamespace(value=str(goal)),
            TEAM=SimpleNamespace(value=str(team)),
            MODE=SimpleNamespace(value=Mode.SKILL),
            SCENE=SimpleNamespace(value=""),
            TOP_K=SimpleNamespace(value=3),
            DURATION_SCALE=SimpleNamespace(value=duration_scale),
        )
        self.plan = ResetPlan(dynamic_obstacles=list(obstacles))
        self.inner = _Inner(self.plan)
        node = self.world.node
        self.task = CasperTask(node=node, ctx=self.world.ctx, params=params, inner=self.inner)

    async def reset(self) -> None:
        from arena_runtime.env.task_layer import EpisodeRecord

        await self.task.reset_robots(EpisodeRecord())
        await self.task.reset_obstacles(self.plan, EpisodeRecord())

    async def run(self, seconds: float) -> None:
        for _ in range(round(seconds * FPS)):
            self.world.node.sim_time.now += 1.0 / FPS
            await self.task.tick({}, [])
            self.world.echo()
            if await self.task.done:
                return

    def kinds(self) -> list[tuple[str, str, str]]:
        return [(event.task, event.kind, event.robot) for event in self.events if event.kind not in ("allocated", "assigned")]


def test_pedestrian_member_walks_its_task_and_the_claims_end_with_the_goal(tmp_path: pathlib.Path) -> None:
    from arena_people_msgs.msg import Pedestrian

    layer = _Layer(tmp_path, [{"id": "greet", "duration": 4, "required_constraints": {"workspace": ["kitchen"]}}], {"walker": _walker(0.9)}, obstacles=(_obstacle(),), duration_scale=0.25)

    async def scenario() -> None:
        await layer.reset()
        layer.world.appear("env/walker", 7, 5.0, 5.0)
        await layer.run(30.0)

    asyncio.run(scenario())
    assert layer.kinds() == [("greet", "started", "walker"), ("greet", "met", "walker")]
    assert next(event.detail for event in layer.events if event.kind == "started") == {"phases": 1}
    assert asyncio.run(layer.task.done)
    claimed = layer.world.claimed()
    assert any(ped.animation_state == Pedestrian.WALKING for batch in claimed for ped in batch.pedestrians)
    assert list(layer.world.batches[-1].pedestrians) == []
    assert layer.aborted == []


def test_reset_strips_the_waypoints_of_pedestrian_members_only(tmp_path: pathlib.Path) -> None:
    other = _obstacle(name="bystander")
    layer = _Layer(tmp_path, [{"id": "greet"}], {"walker": _walker(0.9)}, obstacles=(_obstacle(), other))
    asyncio.run(layer.reset())
    by_name = {obstacle.name: obstacle for obstacle in layer.plan.dynamic_obstacles}
    assert by_name["walker"].waypoints == []
    assert len(by_name["bystander"].waypoints) == 2


def test_reset_refuses_pedestrian_members_the_scenario_lacks(tmp_path: pathlib.Path) -> None:
    layer = _Layer(tmp_path, [{"id": "greet"}], {"walker": _walker(0.9), "guest": _walker(0.5)}, obstacles=(_obstacle(),))
    with pytest.raises(ValueError, match=r"casper pedestrian members \['guest'\] are no humans of the scenario, declared: \['walker'\]"):
        asyncio.run(layer.reset())


def test_robot_takes_over_a_task_the_pedestrian_cannot_walk(tmp_path: pathlib.Path) -> None:
    from arena_casper.pedestrian import SPAWN_WAIT

    r0 = _Manager("r0")
    layer = _Layer(tmp_path, [{"id": "cook", "required_constraints": {"workspace": ["kitchen"]}}], {"walker": _walker(0.9), "r0": _robot(0.5)}, managers=(r0,), obstacles=(_obstacle(),))

    async def scenario() -> None:
        await layer.reset()
        await layer.run(SPAWN_WAIT + 1.0)

    asyncio.run(scenario())
    assert layer.kinds() == [("cook", "started", "walker"), ("cook", "failed", "walker"), ("cook", "reassigned", "r0"), ("cook", "started", "r0")]
    assert next(event.detail for event in layer.events if event.kind == "reassigned") == {"from": "walker"}
    assert len(r0.requests) == 1
    assert layer.aborted == []


def test_episode_aborts_when_the_only_pedestrian_cannot_walk(tmp_path: pathlib.Path) -> None:
    from arena_casper.pedestrian import SPAWN_WAIT

    layer = _Layer(tmp_path, [{"id": "cook", "required_constraints": {"workspace": ["kitchen"]}}], {"walker": _walker(0.9)}, obstacles=(_obstacle(),))

    async def scenario() -> None:
        await layer.reset()
        await layer.run(SPAWN_WAIT + 1.0)

    asyncio.run(scenario())
    assert layer.aborted == ["casper task cook failed on every candidate"]


def test_pedestrian_and_robot_members_run_their_tasks_side_by_side(tmp_path: pathlib.Path) -> None:
    r0 = _Manager("r0")
    tasks = [
        {"id": "greet", "priority": 1, "required_constraints": {"workspace": ["kitchen"]}},
        {"id": "serve", "priority": 2, "required_constraints": {"workspace": ["garden"]}},
    ]
    layer = _Layer(tmp_path, tasks, {"walker": _walker(0.9, ["kitchen"]), "r0": _robot(0.9, ["garden"])}, managers=(r0,), obstacles=(_obstacle(),))

    async def scenario() -> None:
        await layer.reset()
        layer.world.appear("env/walker", 7, 5.0, 5.0)
        await layer.run(30.0)

    asyncio.run(scenario())
    assert ("greet", "started", "walker") in layer.kinds()
    assert ("greet", "met", "walker") in layer.kinds()
    assert ("serve", "started", "r0") in layer.kinds()
    assert len(r0.requests) == 1


def test_teardown_releases_the_pedestrian_claims(tmp_path: pathlib.Path) -> None:
    layer = _Layer(tmp_path, [{"id": "greet", "duration": 100, "required_constraints": {"workspace": ["kitchen"]}}], {"walker": _walker(0.9)}, obstacles=(_obstacle(),))

    async def scenario() -> None:
        await layer.reset()
        layer.world.appear("env/walker", 7, 5.0, 5.0)
        await layer.run(1.0)
        assert layer.world.claimed()
        await layer.task.teardown()

    asyncio.run(scenario())
    assert list(layer.world.batches[-1].pedestrians) == []
    assert layer.inner.torn_down
    assert layer.world.node.callbacks == {}
