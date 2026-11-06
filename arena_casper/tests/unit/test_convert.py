from __future__ import annotations

import json
import pathlib

import pytest
import yaml
from casper.goal import load_goal

from arena_casper import convert
from arena_casper.team import parse_team

GOAL = {
    "tasks": [
        {"id": "Grill", "context": "Party", "required_skills": {"cook": 0.5}, "required_constraints": {"can_move": True, "workspace": ["Grill_Table", "Prep_Table"]}, "duration": 10, "priority": 1},
        {"id": "Serve", "context": "Party", "required_skills": {"transport": 0.5}, "required_constraints": {"workspace": "Drinks_Table"}, "duration": 5, "priority": 2},
    ],
    "dependencies": [["Grill", "Serve"]],
}
ARM = {
    "debug": False,
    "use_sim": True,
    "robot_model": "FrankaResearch3",
    "robot_id": "fr3_arm_1",
    "x_pos": 5,
    "y_pos": 5.2,
    "yaw": -1.57,
    "spawn_delay": 15.0,
    "constraints": {"can_move": False, "can_manipulate": True, "max_payload_kg": 3.0},
    "workspace": ["Grill_Table", "Prep_Table"],
    "skill_weights": {"Party": {"cook": [1.0, 1]}},
}
HOST = {"robot_model": "Human", "robot_id": "host", "workspace": ["Drinks_Table", "Prep_Table"], "skill_weights": {"Party": {"transport": [0.9, 1]}}}
EXP = {
    "task": "configs/goals/party.json",
    "party_duration": 1680,
    "optimizeMode": "preference",
    "world_name": "backyard",
    "use_sim": True,
    "team_size": 2,
    "top_k": 1,
    "agents": ["fr3_arm_1", "human_host_1"],
    "supervisor": "human_host_1",
    "supervisor_team": False,
    "guests": 5,
}
ROBOTS = {"fr3_arm_1": "arm", "human_host_1": "person"}


def _dump(path: pathlib.Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content) if path.suffix == ".yaml" else json.dumps(content))


@pytest.fixture
def config(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "configs"
    _dump(root / "robots" / "fr3_arm_1.yaml", ARM)
    _dump(root / "robots" / "human_host_1.yaml", HOST)
    _dump(root / "exps" / "party.yaml", EXP)
    _dump(root / "goals" / "party.json", GOAL)
    return root


def test_team_is_keyed_by_arena_robot_names_without_derived_constraints(config: pathlib.Path) -> None:
    conversion = convert.convert(config, "party", ROBOTS)
    assert conversion.team == {
        "supervisor": "person",
        "supervisor_team": False,
        "members": {
            "arm": {"workspace": ["Grill_Table", "Prep_Table"], "constraints": {"max_payload_kg": 3.0}, "skills": {"Party": {"cook": [1.0, 1]}}},
            "person": {"workspace": ["Drinks_Table", "Prep_Table"], "skills": {"Party": {"transport": [0.9, 1]}}},
        },
    }
    assert conversion.launch == {"task.casper.mode": "preference", "task.casper.top_k": "1"}
    assert "fr3_arm_1: constraints.can_move" in conversion.dropped
    assert "fr3_arm_1: constraints.can_manipulate" in conversion.dropped
    assert "fr3_arm_1: x_pos" in conversion.dropped
    assert "human_host_1: robot_model" in conversion.dropped
    assert "party.yaml: party_duration" in conversion.dropped
    assert "party.yaml: guests" in conversion.dropped
    parse_team(conversion.team, {"arm": {"can_move": True, "can_manipulate": True}, "person": {"can_move": True, "can_manipulate": True}})


def test_zone_renames_apply_to_members_and_goal(config: pathlib.Path) -> None:
    conversion = convert.convert(config, "party", ROBOTS, {"Grill_Table": "grill", "Drinks_Table": "bar"})
    assert conversion.team["members"]["arm"]["workspace"] == ["grill", "Prep_Table"]
    assert conversion.team["members"]["person"]["workspace"] == ["bar", "Prep_Table"]
    assert conversion.goal["tasks"][0]["required_constraints"]["workspace"] == ["grill", "Prep_Table"]
    assert conversion.goal["tasks"][1]["required_constraints"]["workspace"] == "bar"
    assert GOAL["tasks"][0]["required_constraints"]["workspace"] == ["Grill_Table", "Prep_Table"]


def test_world_folder_robot_files_win(config: pathlib.Path) -> None:
    _dump(config / "robots" / "backyard" / "fr3_arm_1.yaml", {**ARM, "workspace": ["Elsewhere"]})
    assert convert.convert(config, "party", ROBOTS).team["members"]["arm"]["workspace"] == ["Elsewhere"]


def test_write_produces_files_the_layer_loads_and_the_launch_line(config: pathlib.Path, tmp_path: pathlib.Path) -> None:
    conversion = convert.convert(config, "party", ROBOTS, {"Grill_Table": "grill"})
    team, goal, launch = convert.write(conversion, tmp_path / "out")
    assert load_goal(goal).tasks["Grill"].workspaces == ("grill", "Prep_Table")
    assert yaml.safe_load(team.read_text()) == conversion.team
    assert launch == f"task.casper.goal:={goal} task.casper.team:={team} task.casper.mode:=preference task.casper.top_k:=1"


def test_unmapped_agent_is_named(config: pathlib.Path) -> None:
    with pytest.raises(ValueError, match=r"agents \['human_host_1'\] have no --robot mapping"):
        convert.convert(config, "party", {"fr3_arm_1": "arm"})


def test_robot_mapping_for_a_stranger_is_named(config: pathlib.Path) -> None:
    with pytest.raises(ValueError, match=r"--robot names \['ghost'\] are no agents"):
        convert.convert(config, "party", {**ROBOTS, "ghost": "g"})


def test_two_agents_on_one_robot_are_refused(config: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="two agents to one Arena robot"):
        convert.convert(config, "party", {"fr3_arm_1": "arm", "human_host_1": "arm"})


def test_unknown_robot_key_is_named(config: pathlib.Path) -> None:
    _dump(config / "robots" / "fr3_arm_1.yaml", {**ARM, "payload_style": 1})
    with pytest.raises(ValueError, match=r"unknown keys \['payload_style'\]"):
        convert.convert(config, "party", ROBOTS)


def test_unknown_experiment_key_is_named(config: pathlib.Path) -> None:
    _dump(config / "exps" / "party.yaml", {**EXP, "mystery": 1})
    with pytest.raises(ValueError, match=r"unknown keys \['mystery'\]"):
        convert.convert(config, "party", ROBOTS)


def test_unused_zone_rename_is_named(config: pathlib.Path) -> None:
    with pytest.raises(ValueError, match=r"--zone names \['Nowhere'\] occur in no member workspace"):
        convert.convert(config, "party", ROBOTS, {"Nowhere": "x"})


def test_bad_mode_and_missing_robot_file_are_named(config: pathlib.Path) -> None:
    _dump(config / "exps" / "party.yaml", {**EXP, "optimizeMode": "fastest"})
    with pytest.raises(ValueError, match=r"optimizeMode 'fastest' is none of \['skill'"):
        convert.convert(config, "party", ROBOTS)
    _dump(config / "exps" / "party.yaml", EXP)
    (config / "robots" / "human_host_1.yaml").unlink()
    with pytest.raises(ValueError, match="agent 'human_host_1' has no robot file"):
        convert.convert(config, "party", ROBOTS)


def test_goal_override_replaces_the_exps_path(config: pathlib.Path, tmp_path: pathlib.Path) -> None:
    other = tmp_path / "other.json"
    _dump(other, {"tasks": [GOAL["tasks"][1]], "dependencies": []})
    assert [task["id"] for task in convert.convert(config, "party", ROBOTS, goal=other).goal["tasks"]] == ["Serve"]


def test_cli_converts_and_prints_the_dropped_and_launch_arguments(config: pathlib.Path, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    argv = [str(config), "party", "--robot", "fr3_arm_1=arm", "--robot", "human_host_1=person", "--zone", "Grill_Table=grill", "--out", str(out)]
    assert convert.main(argv) == 0
    printed = capsys.readouterr().out
    assert (out / "team.yaml").is_file()
    assert (out / "goal.json").is_file()
    assert "fr3_arm_1: constraints.can_move" in printed
    assert "task.casper.mode:=preference" in printed


def test_cli_explains_itself(config: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert convert.main([]) == 2
    out = capsys.readouterr().out
    assert "--robot" in out
    assert "--zone" in out
    assert "team.yaml" in out
    with pytest.raises(SystemExit) as raised:
        convert.main([str(config), "party", "--robot", "fr3_arm_1"])
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert "<name in CASPER_MAS>=<name in Arena>" in err
    assert "--help" in err
    with pytest.raises(SystemExit):
        convert.main([str(config), "party", "--robot", "fr3_arm_1=arm"])
    assert "have no --robot mapping" in capsys.readouterr().err
