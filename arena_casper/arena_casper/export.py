"""Recorded casper task events written as the CSV files of CASPER_MAS's ExperimentLogger."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
import time
import typing
from collections.abc import Mapping, Sequence

from casper.goal import Goal, load_goal

TASK_ALLOCATION_COLUMNS = (
    "run_id",
    "task_id",
    "mode",
    "is_end_goal",
    "context",
    "initial_agent",
    "initial_score",
    "final_agent",
    "final_score",
    "was_rebalanced",
    "final_status",
    "top_candidates",
    "best_possible_score",
    "skill_fit_gap",
    "num_predecessors",
    "num_successors",
    "predecessors",
    "successors",
    "is_bottleneck",
    "is_leaf",
    "is_source",
)
AGENT_WORKLOAD_COLUMNS = ("run_id", "agent_id", "mode", "initial_task_count", "final_task_count", "task_count_delta", "is_idle")
ALLOCATION_SUMMARY_COLUMNS = (
    "run_id",
    "mode",
    "top_k",
    "num_tasks",
    "num_agents",
    "num_unassigned_tasks",
    "num_rebalanced_tasks",
    "pct_rebalanced",
    "num_idle_agents",
    "pct_idle_agents",
    "max_tasks_per_agent",
    "min_tasks_per_agent",
    "workload_spread",
    "workload_stdev",
    "avg_tasks_per_agent",
    "avg_skill_fit_gap",
    "max_skill_fit_gap",
    "num_bottleneck_tasks",
    "num_source_tasks",
    "num_leaf_tasks",
    "timestamp",
)
TASK_EXECUTION_COLUMNS = (
    "run_id",
    "agent_id",
    "task_id",
    "execution_order",
    "num_predecessors",
    "num_successors",
    "predecessors",
    "successors",
    "is_bottleneck",
    "is_source",
    "is_leaf",
    "successor_statuses",
    "successors_unblocked_by_this",
    "timestamp",
    "started_at",
    "finished_at",
)
FILES = {
    "task_execution.csv": TASK_EXECUTION_COLUMNS,
    "task_allocation.csv": TASK_ALLOCATION_COLUMNS,
    "agent_workload.csv": AGENT_WORKLOAD_COLUMNS,
    "allocation_summary.csv": ALLOCATION_SUMMARY_COLUMNS,
}

Event = Mapping[str, typing.Any]
Row = dict[str, typing.Any]

_REBALANCING_MODE = "balance_workload"


def episodes(raw: object) -> list[list[Event]]:
    """The event lists of a JSON document: a list of events, an episode record, or a list of records."""
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError(f"expected a list of task events or of episode records, got {type(raw).__name__}")
    if all(isinstance(entry, dict) and "kind" in entry for entry in raw):
        return [list(raw)]
    runs = []
    for entry in raw:
        if not isinstance(entry, dict) or "tasks" not in entry:
            raise ValueError(f"entry {entry!r} is neither a task event nor an episode record with a tasks field")
        events = json.loads(entry["tasks"]) if isinstance(entry["tasks"], str) else entry["tasks"]
        runs.append(list(events))
    return runs


def _graph(goal: Goal, task_id: str) -> tuple[list[str], list[str]]:
    predecessors = list(goal.predecessors(task_id))
    successors = [target for source, target in goal.dependencies if source == task_id]
    return predecessors, successors


def _facts(goal: Goal, task_id: str) -> Row:
    predecessors, successors = _graph(goal, task_id)
    return {
        "num_predecessors": len(predecessors),
        "num_successors": len(successors),
        "predecessors": json.dumps(predecessors),
        "successors": json.dumps(successors),
        "is_bottleneck": len(successors) >= 2,
        "is_leaf": len(successors) == 0,
        "is_source": len(predecessors) == 0,
    }


def _allocated(events: Sequence[Event]) -> Mapping[str, typing.Any]:
    for event in events:
        if event["kind"] == "allocated":
            return event["detail"]
    raise ValueError("no allocated event among the task events")


def _status(events: Sequence[Event], task_id: str, until: float | None = None) -> str:
    status = "assigned"
    for event in events:
        if event["task"] != task_id or (until is not None and event["t"] > until):
            continue
        if event["kind"] == "started":
            status = "running"
        elif event["kind"] == "met":
            status = "done"
        elif event["kind"] in ("failed", "dropped"):
            status = "assigned"
    return status


def _final_status(events: Sequence[Event], task_id: str) -> str:
    last = [event["kind"] for event in events if event["task"] == task_id and event["kind"] in ("started", "met", "failed", "dropped")]
    if not last:
        return "assigned"
    return {"started": "running", "met": "done", "failed": "failed", "dropped": "assigned"}[last[-1]]


def _status_at(events: Sequence[Event], goal: Goal, task_id: str, t: float) -> str:
    status = _status(events, task_id, t)
    if status == "assigned" and all(_status(events, before, t) == "done" for before in goal.predecessors(task_id)):
        return "ready"
    if status == "assigned":
        return "pending"
    return status


def allocation_rows(run_id: str, events: Sequence[Event], goal: Goal, timestamp: float) -> tuple[list[Row], list[Row], Row]:
    """The task_allocation, agent_workload and allocation_summary rows of one run."""
    detail = _allocated(events)
    mode = str(detail["mode"])
    assignments: Mapping[str, Mapping[str, typing.Any]] = detail["assignments"]
    ranking: Mapping[str, list[list[typing.Any]]] = detail["ranking"]
    workload: Mapping[str, int] = detail["workload"]
    rebalancing = mode == _REBALANCING_MODE
    task_rows: list[Row] = []
    gaps: list[float] = []
    for task_id, task in goal.tasks.items():
        assignment = assignments.get(task_id)
        agent = assignment["member"] if assignment else None
        score = assignment["score"] if assignment else None
        scores = [pair[1] for pair in ranking.get(task_id, [])]
        best = max(scores) if scores else None
        gap = best - score if best is not None and score is not None else None
        if gap is not None:
            gaps.append(gap)
        task_rows.append(
            {
                "run_id": run_id,
                "task_id": task_id,
                "mode": mode,
                "is_end_goal": task.end_goal,
                "context": task.context,
                "initial_agent": None if rebalancing else agent,
                "initial_score": None if rebalancing else score,
                "final_agent": agent,
                "final_score": score,
                "was_rebalanced": None if rebalancing else False,
                "final_status": _final_status(events, task_id) if assignment else "not assigned",
                "top_candidates": json.dumps([str(name) for name in assignment["fallbacks"]]) if assignment else "[]",
                "best_possible_score": best,
                "skill_fit_gap": gap,
                **_facts(goal, task_id),
            }
        )
    agents = sorted(set(workload) | {pair[0] for pairs in ranking.values() for pair in pairs})
    agent_rows: list[Row] = []
    for agent_id in agents:
        final = workload.get(agent_id, 0)
        agent_rows.append(
            {
                "run_id": run_id,
                "agent_id": agent_id,
                "mode": mode,
                "initial_task_count": None if rebalancing else final,
                "final_task_count": final,
                "task_count_delta": None if rebalancing else 0,
                "is_idle": final == 0,
            }
        )
    counts = [row["final_task_count"] for row in agent_rows]
    idle = sum(1 for row in agent_rows if row["is_idle"])
    tasks = len(task_rows)
    summary: Row = {
        "run_id": run_id,
        "mode": mode,
        "top_k": detail["top_k"],
        "num_tasks": tasks,
        "num_agents": len(agent_rows),
        "num_unassigned_tasks": sum(1 for row in task_rows if row["final_agent"] is None),
        "num_rebalanced_tasks": None if rebalancing else 0,
        "pct_rebalanced": None if rebalancing else 0.0,
        "num_idle_agents": idle,
        "pct_idle_agents": idle / len(agent_rows) * 100.0 if agent_rows else 0.0,
        "max_tasks_per_agent": max(counts) if counts else 0,
        "min_tasks_per_agent": min(counts) if counts else 0,
        "workload_spread": max(counts) - min(counts) if counts else 0,
        "workload_stdev": statistics.pstdev(counts) if len(counts) > 1 else 0.0,
        "avg_tasks_per_agent": statistics.mean(counts) if counts else 0.0,
        "avg_skill_fit_gap": statistics.mean(gaps) if gaps else None,
        "max_skill_fit_gap": max(gaps) if gaps else None,
        "num_bottleneck_tasks": sum(1 for row in task_rows if row["is_bottleneck"]),
        "num_source_tasks": sum(1 for row in task_rows if row["is_source"]),
        "num_leaf_tasks": sum(1 for row in task_rows if row["is_leaf"]),
        "timestamp": timestamp,
    }
    return task_rows, agent_rows, summary


def execution_rows(run_id: str, events: Sequence[Event], goal: Goal) -> list[Row]:
    """One row per met task in order of completion, with its runner and the graph facts at that moment."""
    rows: list[Row] = []
    orders: dict[str, int] = {}
    starts: dict[tuple[str, str], float] = {}
    for event in events:
        key = (str(event["task"]), str(event["robot"]))
        if event["kind"] == "started":
            starts[key] = float(event["t"])
        elif event["kind"] == "met" and event["task"] in goal.tasks:
            task_id, agent_id, finished = key[0], key[1], float(event["t"])
            orders[agent_id] = orders.get(agent_id, 0) + 1
            _, successors = _graph(goal, task_id)
            statuses = {successor: _status_at(events, goal, successor, finished) for successor in successors}
            unblocked = sum(1 for successor in successors if all(_status(events, before, finished) == "done" for before in goal.predecessors(successor)))
            rows.append(
                {
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "execution_order": orders[agent_id],
                    **_facts(goal, task_id),
                    "successor_statuses": json.dumps(statuses),
                    "successors_unblocked_by_this": unblocked,
                    "timestamp": finished,
                    "started_at": starts.get(key),
                    "finished_at": finished,
                }
            )
    return rows


def _append(path: pathlib.Path, columns: Sequence[str], rows: Sequence[Row]) -> None:
    exists = path.is_file()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def export(events: Sequence[Event], goal: Goal, run_id: str, out_dir: str | pathlib.Path, timestamp: float | None = None) -> list[pathlib.Path]:
    """Append one run's rows to the four CSV files in out_dir, headers written when a file is new."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    task_rows, agent_rows, summary = allocation_rows(run_id, events, goal, time.time() if timestamp is None else timestamp)
    tables = {
        "task_execution.csv": execution_rows(run_id, events, goal),
        "task_allocation.csv": task_rows,
        "agent_workload.csv": agent_rows,
        "allocation_summary.csv": [summary],
    }
    for name, rows in tables.items():
        _append(out / name, FILES[name], rows)
    return [out / name for name in FILES]


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> typing.NoReturn:
        self.print_usage(sys.stderr)
        sys.stderr.write(f"{self.prog}: error: {message}\nrun '{self.prog} --help' for the arguments and the files written\n")
        raise SystemExit(2)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python -m arena_casper.export",
        description=(f"Write the CSV files of CASPER_MAS's ExperimentLogger from recorded casper task events: {', '.join(FILES)}. Rows are appended to existing files, headers are written once. Columns the events cannot give are left empty (see the arena_casper README)."),
    )
    parser.add_argument("events", help="JSON file: a list of task events, an episode record, or a list of episode records (their tasks field)")
    parser.add_argument("goal", help="the CASPER goal JSON the run used, source of the graph columns")
    parser.add_argument("--run-id", required=True, help="value of the run_id column; with several episode records '_<index>' is appended")
    parser.add_argument("--out", default="experiment_logs", help="directory the CSV files go to (default: experiment_logs)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = sys.argv[1:] if argv is None else list(argv)
    if not arguments:
        parser.print_help()
        return 2
    args = parser.parse_args(arguments)
    for label, path in (("events", args.events), ("goal", args.goal)):
        if not pathlib.Path(path).is_file():
            parser.error(f"{label} file {path!r} does not exist")
    written: list[pathlib.Path] = []
    try:
        runs = episodes(json.loads(pathlib.Path(args.events).read_text()))
        goal = load_goal(args.goal)
        for index, events in enumerate(runs):
            run_id = args.run_id if len(runs) == 1 else f"{args.run_id}_{index}"
            written = export(events, goal, run_id, args.out)
    except (ValueError, KeyError) as e:
        parser.error(f"{args.events}: {e!r}")
    print(f"appended {len(runs)} run(s) to " + ", ".join(str(path) for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
