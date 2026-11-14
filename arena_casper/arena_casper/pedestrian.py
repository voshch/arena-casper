"""Pedestrian team members: a ROS-free route follower and the driver that claims their pedestrians on human/stream."""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import math
import typing
from collections.abc import Sequence

from arena_simulation_setup.shared.conditions import MembershipAtom
from arena_simulation_setup.shared.judge import Sample, atom_holds, zone_polygons
from arena_simulation_setup.shared.pathing import OccupancyGrid, PathError, Point, plan_route, point_at_arc, route_length
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

from arena_casper.scene import make_site

if typing.TYPE_CHECKING:
    from arena_people_msgs.msg import Pedestrian, Pedestrians
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_runtime.env.task_layer import TaskContext
    from arena_runtime.humans.gait import GaitGenerator
    from arena_simulation_setup.shared.entities import DynamicObstacle

__all__ = ["CLEARANCE", "MAX_SPEED", "PedDriver", "PedRoute", "PedState", "PedStep", "plan_legs"]

MAX_SPEED = 1.7
MIN_SPEED = 0.1
YAW_RATE = 3.0
TURN_IN_PLACE = 0.6
FACING_TOLERANCE = 0.05
CLEARANCE = 0.3
SPAWN_WAIT = 5.0
_LOOKAHEAD = 1e-6
_EPS = 1e-9


class PedState(enum.Enum):
    WALKING = "walking"
    TURNING = "turning"
    HOLDING = "holding"
    DONE = "done"


@dataclasses.dataclass(frozen=True)
class PedStep:
    x: float
    y: float
    yaw: float
    speed: float
    state: PedState


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _turn(yaw: float, target: float, limit: float) -> float:
    return _wrap(yaw + max(-limit, min(limit, _wrap(target - yaw))))


class PedRoute:
    """A polyline walked at a speed clamped below the run threshold, with an in-place turn at sharp corners and a hold at its end."""

    def __init__(self, route: Sequence[Point], yaw: float, speed: float, hold: float = 0.0, face: float | None = None) -> None:
        if not route:
            raise ValueError("a pedestrian route needs at least its start")
        self._route = list(route)
        self._length = route_length(self._route)
        self._arc = 0.0
        self._yaw = yaw
        self._speed = min(max(speed, MIN_SPEED), MAX_SPEED)
        self._hold = hold
        self._face = face
        self._held = 0.0
        self._state = PedState.WALKING if self._length > 0.0 else PedState.HOLDING

    @property
    def state(self) -> PedState:
        return self._state

    @property
    def walked(self) -> bool:
        return self._length - self._arc <= _EPS

    def _position(self) -> Point:
        if self._length <= 0.0:
            return self._route[0]
        return point_at_arc(self._route, self._arc)[0]

    def _result(self, speed: float) -> PedStep:
        x, y = self._position()
        return PedStep(x=x, y=y, yaw=self._yaw, speed=speed, state=self._state)

    def step(self, dt: float, can_hold: bool = True) -> PedStep:
        """Advance by `dt` seconds, `can_hold` False keeps the hold timer from running at the end of the route."""
        dt = max(dt, 0.0)
        limit = YAW_RATE * dt
        if not self.walked:
            dx, dy = point_at_arc(self._route, self._arc + _LOOKAHEAD)[1]
            heading = math.atan2(dy, dx)
            sharp = abs(_wrap(heading - self._yaw)) > TURN_IN_PLACE
            self._yaw = _turn(self._yaw, heading, limit)
            if sharp:
                self._state = PedState.TURNING
                return self._result(0.0)
            moved = min(self._speed * dt, self._length - self._arc)
            self._arc += moved
            self._state = PedState.WALKING
            return self._result(moved / dt if dt > 0.0 else 0.0)
        if self._face is not None:
            self._yaw = _turn(self._yaw, self._face, limit)
        facing = self._face is None or abs(_wrap(self._face - self._yaw)) <= FACING_TOLERANCE
        if can_hold:
            self._held += dt
        if self._held + _EPS >= self._hold and facing:
            self._state = PedState.DONE
        else:
            self._state = PedState.HOLDING if facing else PedState.TURNING
        return self._result(0.0)


def plan_legs(grid: OccupancyGrid, points: Sequence[Point], clearance: float = CLEARANCE) -> list[Point]:
    """The free-space route through `points` in order, PathError when a leg has none."""
    valid = grid.clear_of(clearance)
    route = [points[0]]
    for goal in points[1:]:
        route.extend(plan_route(grid, route[-1], goal, is_valid=valid)[1:])
    return route


@dataclasses.dataclass
class _Job:
    obstacle: DynamicObstacle
    legs: tuple[str, ...]
    hold: float
    face: float | None
    waited: float = 0.0
    route: PedRoute | None = None
    zones: dict[str, typing.Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _Claim:
    member: str
    ped: str
    id: int
    x: float
    y: float
    yaw: float
    gait: GaitGenerator
    speed: float = 0.0


class PedDriver:
    """Claims the team's pedestrians through human/stream and walks each through the workspaces of its task."""

    def __init__(self, node: ArenaMixinNode, ctx: TaskContext) -> None:
        import arena_people_msgs.msg
        import rclpy.qos
        from arena_rclpy_mixins.shared import Namespace
        from arena_runtime.humans.gait import GaitGenerator

        self._node = node
        self._ctx = ctx
        self._people = arena_people_msgs.msg
        self._gait_type = GaitGenerator
        self._site = make_site(ctx)
        self._seen: dict[str, Pedestrian] = {}
        self._jobs: dict[str, _Job] = {}
        self._claims: dict[str, _Claim] = {}
        self._last: float | None = None
        namespace = Namespace(node.get_namespace())
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._sub = node.create_subscription(arena_people_msgs.msg.Pedestrians, str(namespace("arena_peds")), self._on_peds, 10)
        self._pub = node.create_publisher(arena_people_msgs.msg.Pedestrians, str(namespace("human", "stream")), qos)

    def _on_peds(self, msg: Pedestrians) -> None:
        for ped in msg.pedestrians:
            self._seen[ped.name] = ped

    def reset(self) -> None:
        """Forget claims, jobs and the pedestrians seen, for a new episode."""
        self._seen.clear()
        self._jobs.clear()
        self._claims.clear()
        self._last = None

    def close(self) -> None:
        self._node.destroy_subscription(self._sub)
        self._node.destroy_publisher(self._pub)

    def start(self, obstacle: DynamicObstacle, legs: Sequence[str], hold: float, face: float | None = None) -> None:
        """Queue the walk through the zones `legs` and a hold of `hold` seconds at the last for the pedestrian `obstacle`."""
        self._jobs[obstacle.name] = _Job(obstacle=obstacle, legs=tuple(legs), hold=hold, face=face)

    def release_all(self) -> None:
        """Publish an empty stamped batch, which releases every claim, and drop the jobs."""
        if self._claims:
            self._pub.publish(self._batch([]))
        self._claims.clear()
        self._jobs.clear()

    def _batch(self, peds: list[Pedestrian]) -> Pedestrians:
        batch = self._people.Pedestrians()
        batch.header.frame_id = "map"
        batch.header.stamp = self._node.sim_time.to_msg()
        batch.pedestrians = peds
        return batch

    def _abstract(self, ped: Pedestrian) -> tuple[float, float, float]:
        q = ped.pose.orientation
        local = self._ctx.environment_manager.ezilear(Pose(Position(ped.pose.position.x, ped.pose.position.y), Orientation(q.w, q.x, q.y, q.z)))
        return local.position.x, local.position.y, local.orientation.to_yaw()

    def _find(self, obstacle: DynamicObstacle) -> Pedestrian | None:
        prefix = self._ctx.realizer.prefix
        for name in (prefix(obstacle.name, obstacle.level), prefix(obstacle.name), obstacle.name):
            if (ped := self._seen.get(name)) is not None:
                return ped
        return None

    def _grid(self) -> OccupancyGrid:
        from arena_runtime.env.task_api import WorldOccupancy

        world = self._ctx.world_manager.map
        return OccupancyGrid(occupied=WorldOccupancy.full(world.occupancy.grid), res=world.resolution, ox=world.origin.x, oy=world.origin.y)

    async def _prepare(self, member: str, job: _Job, dt: float) -> bool:
        claim = self._claims.get(member)
        if claim is None:
            ped = self._find(job.obstacle)
            if ped is None:
                job.waited += dt
                if job.waited >= SPAWN_WAIT:
                    raise ValueError(f"pedestrian {member!r} is not on arena_peds")
                return False
            x, y, yaw = self._abstract(ped)
            claim = self._claims[member] = _Claim(member=member, ped=ped.name, id=ped.id, x=x, y=y, yaw=yaw, gait=self._gait_type())
        points = [(claim.x, claim.y), *((pose.position.x, pose.position.y) for pose in map(self._site.locate, job.legs))]
        route = await asyncio.to_thread(plan_legs, self._grid(), points)
        if job.legs:
            job.zones = zone_polygons(self._ctx.world_manager.world_compacted())
        job.route = PedRoute(route, claim.yaw, job.obstacle.velocity, job.hold, job.face)
        self._node.get_logger().get_child("PedDriver").info(f"pedestrian {member!r} walks {route_length(route):.1f} m through {list(job.legs)} and holds {job.hold:.1f} s")
        return True

    def _arrived(self, claim: _Claim, job: _Job, route: PedRoute) -> bool:
        if not job.legs or not route.walked:
            return True
        ped = self._seen.get(claim.ped)
        if ped is None:
            return False
        x, y, _ = self._abstract(ped)
        sample = Sample(t=self._last or 0.0, peds={claim.member: (x, y)})
        return bool(atom_holds(MembershipAtom(subject=claim.member, zone=job.legs[-1]), sample, job.zones))

    def _entry(self, claim: _Claim, dt: float) -> Pedestrian:
        ped = self._people.Pedestrian()
        realized = self._ctx.environment_manager.realize(Pose(Position(claim.x, claim.y, 0.0), Orientation.from_yaw(claim.yaw)))
        yaw = realized.orientation.to_yaw()
        ped.name = claim.ped
        ped.id = claim.id
        ped.pose.position.x = realized.position.x
        ped.pose.position.y = realized.position.y
        ped.pose.position.z = 0.0
        ped.pose.orientation.x = realized.orientation.x
        ped.pose.orientation.y = realized.orientation.y
        ped.pose.orientation.z = realized.orientation.z
        ped.pose.orientation.w = realized.orientation.w
        ped.twist.linear.x = claim.speed * math.cos(yaw)
        ped.twist.linear.y = claim.speed * math.sin(yaw)
        ped.animation_state = self._people.Pedestrian.WALKING if claim.speed > 0.0 else self._people.Pedestrian.IDLE
        ped.joint_state = claim.gait.joint_state(claim.gait.compute(claim.id, ped.animation_state, claim.speed, dt))
        ped.gait_phase = 0.0
        return ped

    async def tick(self) -> tuple[list[str], list[str]]:
        """Advance every claimed pedestrian and publish the claim set, returning the members whose walk and hold finished and the ones that cannot walk."""
        now = self._node.sim_time.to_seconds()
        dt = 0.0 if self._last is None else max(0.0, now - self._last)
        self._last = now
        met: list[str] = []
        failed: list[str] = []
        for member, job in list(self._jobs.items()):
            if job.route is not None:
                continue
            try:
                await self._prepare(member, job, dt)
            except (PathError, ValueError) as e:
                self._node.get_logger().get_child("PedDriver").warning(f"pedestrian {member!r}: {e}")
                failed.append(member)
                del self._jobs[member]
        for member, claim in self._claims.items():
            job = self._jobs.get(member)
            if job is None or job.route is None:
                claim.speed = 0.0
                continue
            step = job.route.step(dt, self._arrived(claim, job, job.route))
            claim.x, claim.y, claim.yaw, claim.speed = step.x, step.y, step.yaw, step.speed
            if step.state is PedState.DONE:
                self._node.get_logger().get_child("PedDriver").info(f"pedestrian {member!r} finished its walk and hold")
                met.append(member)
                del self._jobs[member]
        if self._claims:
            self._pub.publish(self._batch([self._entry(claim, dt) for claim in self._claims.values()]))
        return met, failed
