# casper

The CASPER++ allocation core: goal graphs, team skills, allocation and schedule, in plain Python (3.10 and later, standard library only). [arena_casper](../arena_casper) runs it as an Arena task layer.

| Module | Holds |
| --- | --- |
| `goal` | `Goal`, `GoalTask`, `load_goal`, `parse_goal`, `parse_task`, `build_goal` |
| `skills` | `Member`, `Team` (`regular` and `supervisors`, each in tie order), `member` from `{context: {skill: [level, rank]}}`, `fill`, `UNSTATED` |
| `allocation` | `Mode`, `qualifies`, `rank`, `assign` (with `pinned` tasks), `rebalance`, `allocate`, `AllocationError` |
| `schedule` | `Schedule`: readiness, priority, clearance, fallbacks on failure, done |

## Goal

The goal JSON: `tasks`, each with `id`, `context`, `required_skills` (`{skill: minimum
level}`), `required_constraints` (`can_move`, `can_manipulate`, numbers such as
`max_payload_kg`, and `workspace`, a workspace name or a list of them), `duration`, `priority`
(lower first), `subgoal_level`, `is_end_goal`, `time_to_clean` and `at_end`, plus
`dependencies` as `[before, after]` pairs. `load_goal` refuses unknown keys, unknown tasks in
dependencies, a task listed twice, cycles and `time_to_clean: false` tasks that no
`subgoal_level: 1` task outside `at_end` releases.

## Allocation

A member qualifies for a task when it knows the task's context, its workspaces cover the
task's (case-insensitive), it meets every skill minimum and every constraint it states as a
minimum, `false` counting as 0 and `true` as 1, so `can_manipulate: false` admits any member
(an unstated constraint passes). Tasks with `subgoal_level: 1` or `is_end_goal` are open to the
team's supervisors only. The score is the mean over the required skills of the level (`skill`), the
rank normalized as `(N - rank + 1) / N` over the member's N ranked skills of the context
(`preference`), or `0.7 * level + 0.3 * preference` (`balance_skill_preference`). Each task
goes to its best scoring member, a top-score tie to the member with fewer tasks so far.
`balance_workload` scores every qualifying member 0, so workload alone decides, then moves
tasks off the busiest member to a less loaded qualifying one while their gap exceeds one, only
members that hold a task counting as least loaded. A task with `pinned` member keeps it. `allocate`
raises `AllocationError` naming every task nobody qualifies for, `assign` returns them.

## Schedule

A task is ready once its predecessors are done. Tasks with `time_to_clean: false` also wait
until every `subgoal_level: 1` task outside `at_end` is done. A member runs its ready task of
the lowest priority number, one at a time. A failed task moves to its next fallback that has
not failed it. The schedule is done when every task is.

