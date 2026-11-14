from __future__ import annotations

import csv
import json
import pathlib

import pytest
from casper.goal import parse_goal

from arena_casper import export

GOAL = parse_goal(
    {
        "tasks": [
            {"id": "welcome", "context": "party", "subgoal_level": 1},
            {"id": "cook", "context": "party"},
            {"id": "serve", "context": "party", "is_end_goal": True},
            {"id": "orphan", "context": "party"},
        ],
        "dependencies": [["welcome", "cook"], ["welcome", "serve"], ["cook", "serve"]],
    }
)


def _event(task: str, kind: str, robot: str = "", t: float = 0.0, **detail: object) -> dict[str, object]:
    return {"task": task, "kind": kind, "robot": robot, "t": t, "detail": detail}


def _events(mode: str = "skill") -> list[dict[str, object]]:
    return [
        _event(
            "",
            "allocated",
            goal="g.json",
            team="t.yaml",
            mode=mode,
            top_k=2,
            assignments={
                "welcome": {"member": "r0", "score": 0.9, "fallbacks": ["r1"]},
                "cook": {"member": "r1", "score": 0.5, "fallbacks": ["r0"]},
                "serve": {"member": "r0", "score": 0.8, "fallbacks": []},
            },
            ranking={"welcome": [["r0", 0.9], ["r1", 0.7]], "cook": [["r0", 0.8], ["r1", 0.5]], "serve": [["r0", 0.8]]},
            workload={"r0": 2, "r1": 1, "r2": 0},
        ),
        _event("welcome", "assigned", "r0", score=0.9, fallbacks=["r1"]),
        _event("welcome", "started", "r0", 1.0, phases=1),
        _event("welcome", "met", "r0", 11.0),
        _event("cook", "started", "r1", 12.0, phases=2),
        _event("cook", "failed", "r1", 20.0),
        _event("cook", "reassigned", "r0", 20.0, **{"from": "r1"}),
        _event("cook", "started", "r0", 21.0, phases=2),
        _event("cook", "met", "r0", 30.0),
        _event("serve", "started", "r0", 31.0, phases=1),
    ]


def _read(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def test_files_carry_the_experiment_loggers_headers(tmp_path: pathlib.Path) -> None:
    export.export(_events(), GOAL, "run_1", tmp_path, timestamp=99.0)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(export.FILES)
    allocation = "run_id,task_id,mode,is_end_goal,context,initial_agent,initial_score,final_agent,final_score,was_rebalanced,final_status,top_candidates,best_possible_score,skill_fit_gap,num_predecessors,num_successors,predecessors,successors,is_bottleneck,is_leaf,is_source"
    assert _read(tmp_path / "task_allocation.csv")[0] == allocation.split(",")
    assert _read(tmp_path / "agent_workload.csv")[0] == "run_id,agent_id,mode,initial_task_count,final_task_count,task_count_delta,is_idle".split(",")
    execution = "run_id,agent_id,task_id,execution_order,num_predecessors,num_successors,predecessors,successors,is_bottleneck,is_source,is_leaf,successor_statuses,successors_unblocked_by_this,timestamp,started_at,finished_at"
    assert _read(tmp_path / "task_execution.csv")[0] == execution.split(",")
    summary = "run_id,mode,top_k,num_tasks,num_agents,num_unassigned_tasks,num_rebalanced_tasks,pct_rebalanced,num_idle_agents,pct_idle_agents,max_tasks_per_agent,min_tasks_per_agent,workload_spread,workload_stdev,avg_tasks_per_agent,avg_skill_fit_gap,max_skill_fit_gap,num_bottleneck_tasks,num_source_tasks,num_leaf_tasks,timestamp"
    assert _read(tmp_path / "allocation_summary.csv")[0] == summary.split(",")


def test_task_allocation_rows(tmp_path: pathlib.Path) -> None:
    export.export(_events(), GOAL, "run_1", tmp_path, timestamp=99.0)
    rows = {row["task_id"]: row for row in _read(tmp_path / "task_allocation.csv")[1]}
    assert list(rows) == ["welcome", "cook", "serve", "orphan"]
    assert rows["welcome"] == {
        "run_id": "run_1",
        "task_id": "welcome",
        "mode": "skill",
        "is_end_goal": "False",
        "context": "party",
        "initial_agent": "r0",
        "initial_score": "0.9",
        "final_agent": "r0",
        "final_score": "0.9",
        "was_rebalanced": "False",
        "final_status": "done",
        "top_candidates": '["r1"]',
        "best_possible_score": "0.9",
        "skill_fit_gap": "0.0",
        "num_predecessors": "0",
        "num_successors": "2",
        "predecessors": "[]",
        "successors": '["cook", "serve"]',
        "is_bottleneck": "True",
        "is_leaf": "False",
        "is_source": "True",
    }
    assert float(rows["cook"]["skill_fit_gap"]) == pytest.approx(0.3)
    assert rows["cook"]["final_status"] == "done"
    assert rows["serve"]["final_status"] == "running"
    assert rows["serve"]["is_end_goal"] == "True"
    assert rows["serve"]["predecessors"] == '["welcome", "cook"]'
    assert rows["orphan"]["final_agent"] == ""
    assert rows["orphan"]["final_status"] == "not assigned"
    assert rows["orphan"]["is_leaf"] == "True"
    assert rows["orphan"]["is_source"] == "True"


def test_workload_rows_list_idle_agents(tmp_path: pathlib.Path) -> None:
    export.export(_events(), GOAL, "run_1", tmp_path, timestamp=99.0)
    assert _read(tmp_path / "agent_workload.csv")[1] == [
        {"run_id": "run_1", "agent_id": "r0", "mode": "skill", "initial_task_count": "2", "final_task_count": "2", "task_count_delta": "0", "is_idle": "False"},
        {"run_id": "run_1", "agent_id": "r1", "mode": "skill", "initial_task_count": "1", "final_task_count": "1", "task_count_delta": "0", "is_idle": "False"},
        {"run_id": "run_1", "agent_id": "r2", "mode": "skill", "initial_task_count": "0", "final_task_count": "0", "task_count_delta": "0", "is_idle": "True"},
    ]


def test_summary_row(tmp_path: pathlib.Path) -> None:
    export.export(_events(), GOAL, "run_1", tmp_path, timestamp=99.0)
    (row,) = _read(tmp_path / "allocation_summary.csv")[1]
    assert row["num_tasks"] == "4"
    assert row["num_agents"] == "3"
    assert row["num_unassigned_tasks"] == "1"
    assert row["num_rebalanced_tasks"] == "0"
    assert row["num_idle_agents"] == "1"
    assert float(row["pct_idle_agents"]) == pytest.approx(100 / 3)
    assert (row["max_tasks_per_agent"], row["min_tasks_per_agent"], row["workload_spread"]) == ("2", "0", "2")
    assert float(row["workload_stdev"]) == pytest.approx(0.8164965809277)
    assert float(row["avg_tasks_per_agent"]) == pytest.approx(1.0)
    assert float(row["avg_skill_fit_gap"]) == pytest.approx(0.1)
    assert float(row["max_skill_fit_gap"]) == pytest.approx(0.3)
    assert (row["num_bottleneck_tasks"], row["num_source_tasks"], row["num_leaf_tasks"]) == ("1", "2", "2")
    assert row["timestamp"] == "99.0"


def test_task_execution_rows_follow_completion_per_agent(tmp_path: pathlib.Path) -> None:
    export.export(_events(), GOAL, "run_1", tmp_path, timestamp=99.0)
    rows = _read(tmp_path / "task_execution.csv")[1]
    assert [(row["agent_id"], row["task_id"], row["execution_order"]) for row in rows] == [("r0", "welcome", "1"), ("r0", "cook", "2")]
    welcome, cook = rows
    assert (welcome["started_at"], welcome["finished_at"], welcome["timestamp"]) == ("1.0", "11.0", "11.0")
    assert (cook["started_at"], cook["finished_at"]) == ("21.0", "30.0")
    assert json.loads(welcome["successor_statuses"]) == {"cook": "ready", "serve": "pending"}
    assert welcome["successors_unblocked_by_this"] == "1"
    assert json.loads(cook["successor_statuses"]) == {"serve": "ready"}
    assert cook["successors_unblocked_by_this"] == "1"
    assert (welcome["is_source"], welcome["is_bottleneck"], welcome["is_leaf"]) == ("True", "True", "False")
    assert cook["predecessors"] == '["welcome"]'


def test_balance_workload_leaves_the_pre_rebalance_columns_empty(tmp_path: pathlib.Path) -> None:
    export.export(_events("balance_workload"), GOAL, "run_1", tmp_path, timestamp=99.0)
    task = _read(tmp_path / "task_allocation.csv")[1][0]
    assert (task["initial_agent"], task["initial_score"], task["was_rebalanced"], task["final_agent"]) == ("", "", "", "r0")
    agent = _read(tmp_path / "agent_workload.csv")[1][0]
    assert (agent["initial_task_count"], agent["task_count_delta"], agent["final_task_count"]) == ("", "", "2")
    summary = _read(tmp_path / "allocation_summary.csv")[1][0]
    assert (summary["num_rebalanced_tasks"], summary["pct_rebalanced"]) == ("", "")


def test_balance_workload_fills_the_pre_rebalance_columns_from_initial(tmp_path: pathlib.Path) -> None:
    events = _events("balance_workload")
    events[0]["detail"]["initial"] = {"welcome": {"member": "r0", "score": 0.9}, "cook": {"member": "r0", "score": 0.8}, "serve": {"member": "r0", "score": 0.8}}
    export.export(events, GOAL, "run_1", tmp_path, timestamp=99.0)
    tasks = {row["task_id"]: row for row in _read(tmp_path / "task_allocation.csv")[1]}
    assert (tasks["cook"]["initial_agent"], tasks["cook"]["initial_score"], tasks["cook"]["final_agent"], tasks["cook"]["was_rebalanced"]) == ("r0", "0.8", "r1", "True")
    assert tasks["welcome"]["was_rebalanced"] == "False"
    agents = {row["agent_id"]: row for row in _read(tmp_path / "agent_workload.csv")[1]}
    assert (agents["r0"]["initial_task_count"], agents["r0"]["final_task_count"], agents["r0"]["task_count_delta"]) == ("3", "2", "-1")
    assert (agents["r1"]["initial_task_count"], agents["r1"]["task_count_delta"]) == ("0", "1")
    summary = _read(tmp_path / "allocation_summary.csv")[1][0]
    assert summary["num_rebalanced_tasks"] == "1"
    assert float(summary["pct_rebalanced"]) == pytest.approx(100.0 / int(summary["num_tasks"]))


def test_runs_append_with_one_header(tmp_path: pathlib.Path) -> None:
    export.export(_events(), GOAL, "run_1", tmp_path, timestamp=1.0)
    export.export(_events(), GOAL, "run_2", tmp_path, timestamp=2.0)
    rows = _read(tmp_path / "allocation_summary.csv")[1]
    assert [row["run_id"] for row in rows] == ["run_1", "run_2"]
    assert (tmp_path / "allocation_summary.csv").read_text().count("run_id,mode") == 1


def test_episodes_accepts_events_records_and_json_strings() -> None:
    events = _events()
    assert export.episodes(events) == [events]
    assert export.episodes({"tasks": json.dumps(events)}) == [events]
    assert export.episodes([{"tasks": events}, {"tasks": json.dumps(events)}]) == [events, events]
    with pytest.raises(ValueError, match="neither a task event"):
        export.episodes([{"other": 1}])


def test_cli_exports_episode_records(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    goal = tmp_path / "goal.json"
    goal.write_text(json.dumps({"tasks": [{"id": "cook", "context": "party"}], "dependencies": []}))
    allocated = _event("", "allocated", mode="skill", top_k=1, assignments={"cook": {"member": "r0", "score": 1.0, "fallbacks": []}}, ranking={"cook": [["r0", 1.0]]}, workload={"r0": 1})
    events = tmp_path / "episodes.json"
    events.write_text(json.dumps([{"tasks": json.dumps([allocated])}, {"tasks": [allocated]}]))
    out = tmp_path / "logs"
    assert export.main([str(events), str(goal), "--run-id", "exp", "--out", str(out)]) == 0
    assert [row["run_id"] for row in _read(out / "task_allocation.csv")[1]] == ["exp_0", "exp_1"]
    assert "appended 2 run(s)" in capsys.readouterr().out


def test_cli_explains_itself(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert export.main([]) == 2
    out = capsys.readouterr().out
    assert "task_allocation.csv" in out
    assert "--run-id" in out
    with pytest.raises(SystemExit) as raised:
        export.main([str(tmp_path / "missing.json"), str(tmp_path / "goal.json"), "--run-id", "x"])
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "does not exist" in err
    assert "--help" in err
