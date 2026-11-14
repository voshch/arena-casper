# arena_casper

The `casper` task layer, package `arena_casper`: CASPER++ multi-agent task allocation (Leticia
Berto, University of Manchester, [leticiaberto/CASPER_MAS](https://github.com/leticiaberto/CASPER_MAS))
as the scheduler of an Arena env. It allocates the tasks of a CASPER goal graph to the env's
robots and dispatches each robot's tasks as `TaskRequest`s in dependency and priority order.

```bash
arena feature tasks add casper
arena launch sim:=flatsim world:=office_1 robot:=jackal[3] \
    task_layer:=casper task.robots:=stationary \
    task.casper.goal:=$(ros2 pkg prefix arena_casper)/share/arena_casper/configs/office_1/goal.json \
    task.casper.team:=$(ros2 pkg prefix arena_casper)/share/arena_casper/configs/office_1/team.yaml \
    task.casper.mode:=balance_workload
```

Goal format and allocation rules: [casper](../casper/README.md).

## Layering

`casper` wraps task_generator's layer: the axes, modes, params, obstacles and modules are
task_generator's. The robots axis places the robots (`stationary`, `scenario`), and casper
clears the reset task of every team robot and drives the team itself on the episode's ticks.
A robot the team file does not list keeps what the robots axis gives it.

| Param | Default | Meaning |
| --- | --- | --- |
| `task.casper.goal` | | CASPER goal JSON |
| `task.casper.team` | | team YAML |
| `task.casper.scene` | | scene file binding tasks to entities and objects, empty for none |
| `task.casper.mode` | `skill` | `skill`, `preference`, `balance_skill_preference` or `balance_workload` |
| `task.casper.top_k` | `3` | fallback candidates kept per task |
| `task.casper.duration_scale` | `1.0` | factor from a task's `duration` to its hold in seconds |

Goal and team are read at every reset.

## Team

```yaml
supervisor: jackal_0
supervisor_team: true
members:
  jackal_1:
    workspace: [reception, storage]
    constraints: {can_transport_objects: true, max_payload_kg: 5}
    skills:
      OfficeMeeting:
        transport: [0.9, 1]
```

`members` are keyed by robot name. `skills` maps a context to `[level, preference rank]` per
skill, rank 1 the most preferred. A skill a member lists under one context counts in its other
contexts at level and rank 0.1. `can_move` and `can_manipulate` come from the robot's adapters
(goto_pose, reach_pose) and may not be stated. `supervisor_team: false` keeps the supervisor
out of the regular tasks.

## Execution

A task is ready once its predecessors are done. Tasks with `time_to_clean: false` also wait
until every `subgoal_level: 1` task outside `at_end` is done. Each team robot runs its ready
task of the lowest priority number, one at a time, as a `GoToPhase` per workspace in order,
holding at the last for `duration * duration_scale` seconds. A failed task, a task whose
request its robot refuses included, moves to its next fallback candidate that has not failed
it, with none left the episode aborts. A robot given a
task from outside (RunTask, SubmitTask) gets its interrupted task back once that one ends. The
episode is done when every task is.

## Scene

`task.casper.scene` names a YAML file next to the goal that binds tasks to world objects. A
bound task picks, carries and places instead of holding at its workspace.

```yaml
tasks:
  FetchBox:
    pick: {object: storage_box_20_2nd_2, approach: 0.0}
  DeliverBox:
    place: {object: storage_box_20_2nd_2, at: reception}
  RelocateBox:
    pick: {object: storage_box_30_2nd_2, approach: 0.0}
    place: meeting_room_big
```

`pick` names a world object, or is `{object, approach}` with `approach` the world yaw in radians
the robot faces while it picks (default: from the interior of its first workspace towards the
object). `place` names a zone (`object` defaults to the picked one, optional `z` is the placement
height in meters and defaults to the object's own). Task ids not in the goal, unknown keys,
objects or zones the world does not have, and a pick on a task without a workspace are errors at
reset.

A bound task's robot stands off what it picks by the depth midpoint of its box arm workspace,
within half that depth and half the workspace's lateral angle there. The pick is a `GoToPhase`
to the pose that standoff short of the grasp point along the approach, a `ReachPhase`
to the grasp point (in the frame of the picked object, so the arm may touch it) along the approach and a `GraspPhase` close. The grasp point
is where the arm's tip link goes so that its hand stops 3 cm in front of the face of the object's
annotated bounding box that the approach meets, at the box's middle height (at the object's pose
for an object without one): the hand's length, the TF distance from the tip link to the farthest
frame below it, comes off along the approach. Then come a
`GoToPhase` for each remaining workspace and one to the place zone unless the robot already
stands there, a `ReachPhase` the standoff ahead at the placement height (frame `base_link`) and a
`GraspPhase` open. An arm that declares a `stow` named pose tucks into it after each grasp. The
task's `duration * duration_scale` is the hold of the last `GoToPhase`.
Unbound tasks run as before. Objects are located at their authored world poses.

## Events

The layer reports each task step through the episode's task events, which the env stamps with sim time and records in the episode record's `tasks` JSON list.

| Kind | When | Robot | Detail |
| --- | --- | --- | --- |
| `allocated` | once per reset, task `""` | | `goal`, `team`, `mode`, `top_k`, `assignments` (`{task: {member, score, fallbacks}}`), `ranking` (`{task: [[member, score]]}`), `workload` (`{member: count}`) |
| `assigned` | per task at reset | owner | `score`, `fallbacks` |
| `started` | its request is submitted | runner | `phases` |
| `met`, `failed` | its request ends that way | runner | |
| `dropped` | its request is replaced or stopped, or the robot is given a task from outside | runner | |
| `reassigned` | a failed task moves to a fallback | new owner | `from` |
| `released` | the clearance opens a `time_to_clean: false` task | owner | |

## Export

```bash
python -m arena_casper.export episodes.json goal.json --run-id exp1 --out experiment_logs
```

Writes the CSV files of CASPER_MAS's `ExperimentLogger` with its column names and order from the recorded events: `task_execution.csv`, `task_allocation.csv`, `agent_workload.csv` and `allocation_summary.csv`. `episodes.json` holds a list of events, an episode record or a list of records (their `tasks` field, JSON string or list); with several records `--run-id` gets `_<index>` appended. `goal.json` is the goal the run used, the source of the predecessor and successor columns. Rows are appended to existing files, headers are written once.

- `task_execution.csv`: one row per `met` task in order of completion, `execution_order` counted per agent. `started_at` and `finished_at` are sim seconds since the episode start, `timestamp` equals `finished_at`. `successor_statuses` takes `done`, `running`, `ready` (all predecessors done) or `pending` as of the finish.
- `task_allocation.csv`: `final_*` columns are the allocation at reset, `final_status` the last of `done`, `running`, `failed`, `assigned` from the events (`not assigned` without an allocation). `top_candidates` is the task's fallbacks.
- `agent_workload.csv`: every member of the team, idle ones with 0.
- `allocation_summary.csv`: `timestamp` is the export's wall-clock time.
- The `allocated` event carries the allocation before rebalancing as `initial`, which fills `initial_agent`, `initial_score`, `was_rebalanced`, `initial_task_count`, `task_count_delta`, `num_rebalanced_tasks` and `pct_rebalanced`. In mode `balance_workload` a record without `initial` leaves these columns empty. In the other modes nothing is rebalanced and the initial columns equal the final ones.
- `idle_events.csv` and `shutdown_events.csv` are not written.

## Convert

```bash
python -m arena_casper.convert CASPER_MAS/configs exp1_b \
    --robot fr3_arm_1=franka --robot human_host_1=person --zone Grill_Table=grill --out my_team
```

Turns the config folder of CASPER_MAS into `team.yaml`, a goal copy `goal.json` and the launch arguments it prints (`task.casper.goal`, `task.casper.team`, `task.casper.mode`, `task.casper.top_k`).

- `exps/<exp>.yaml` gives `agents`, `supervisor`, `supervisor_team`, `optimizeMode` (the mode) and `top_k`. `task` locates the goal file relative to the folder above the config folder, `--goal` overrides it.
- Each agent's `robots/<world_name>/<agent>.yaml`, else `robots/<agent>.yaml`, gives `workspace`, `constraints` and `skill_weights`, which become the member's `workspace`, `constraints` and `skills` unchanged.
- `--robot <her id>=<Arena robot name>` is required for every agent and names the member. `--zone <her workspace>=<zone>` renames workspaces in the members and in the goal copy, a rename that occurs nowhere is an error.
- `can_move` and `can_manipulate` are dropped since Arena derives them from the robot, as are the spawn settings (`robot_model`, `robot_id`, positions, `spawn_delay`, `actor_type`, `color`, `result_timeout`, `debug`, `use_sim`) and the experiment's `party_duration`, `world_name`, `team_size`, `guests`. Every dropped key is printed.
- Unmapped agents, `--robot` names that are no agents, unknown keys, an unknown `optimizeMode` and a goal that `casper` refuses are errors naming the offender.

Both commands print their arguments on `--help` and with no arguments, and name the valid forms on a bad argument.

## Example

[configs/office_1](configs/office_1) runs a ten-task office meeting on world `office_1`
with three jackals, jackal_0 as supervisor.

[configs/office_1_boxes](configs/office_1_boxes) runs a three-task box demo on world
`office_1` with one tiago (`robot:=tiago`) as team and supervisor. FetchBox picks the shelf box
`storage_box_20_2nd_2` in the storage room, DeliverBox places it in reception, RelocateBox picks
`storage_box_30_2nd_2` from the same shelf and places it in the big meeting room, and BoxesMoved ends the
goal at reception. `scene.yaml` holds the bindings, `goal.json` is in CASPER's format.
