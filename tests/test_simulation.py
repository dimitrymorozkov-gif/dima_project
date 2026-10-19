"""Replay synthetic CSV fixtures against the real LangGraph and SQLite service."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from game_agent.simulation import read_rows, replay_csv
from game_agent.storage import SQLiteStore


ROOT = Path(__file__).resolve().parents[1]
HAPPY = ROOT / "data" / "simulated_answers.csv"
EDGES = ROOT / "data" / "simulation_edge_cases.csv"


def edited_csv(tmp_path, **changes):
    with HAPPY.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        rows = list(reader)
    rows[0].update(changes)
    path = tmp_path / "input.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_game(output, run):
    return json.loads((output / run["game"]).read_text(encoding="utf-8"))


def test_happy_replay_runs_real_graph_and_persists_outputs(tmp_path):
    output, database = tmp_path / "results", tmp_path / "game.sqlite3"
    summary = replay_csv(HAPPY, output, database)
    assert summary["synthetic"] is True
    assert summary["mode"] == "mock"
    run = summary["runs"][0]
    assert run["submission_count"] == 7
    assert run["team_outcomes"] == {"team_1": "success", "team_2": "success"}
    game = load_game(output, run)
    assert game["phase"] == "finished"
    assert game["version"] == 7
    assert game["participants"] == [f"Игрок {index}" for index in range(1, 7)]
    assert [team["event"]["id"] for team in game["teams"]] == ["accident", "unhappy"]
    assert {record["outcome"] for record in game["history"]} == {"continue"}
    assert {record["elapsed_seconds"] for record in game["history"]} == {10}
    assert [record["provenance"] for record in game["history"]] == ["authored_branch"] * 4 + ["mock_heuristic"] * 3
    assert SQLiteStore(database).get(run["game_id"])["history"] == game["history"]
    transcript_path = output / run["transcript"]
    assert transcript_path.read_bytes().startswith(b"\xef\xbb\xbf")
    with transcript_path.open(encoding="utf-8-sig", newline="") as transcript:
        records = list(csv.DictReader(transcript))
    assert len(records) == 7
    assert records[-1]["phase"] == "final"
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == summary


def test_edge_replay_clarification_failure_and_timeout_recover(tmp_path):
    output = tmp_path / "edges"
    summary = replay_csv(EDGES, output, tmp_path / "edge.sqlite3")
    runs = {run["run_id"]: run for run in summary["runs"]}
    assert len(runs) == 3
    clarify = load_game(output, runs["synthetic_clarify"])
    assert [record["outcome"] for record in clarify["history"]].count("clarify") == 1
    assert clarify["teams"][0]["outcome"] == "success"
    clarified_records = [record for record in clarify["history"] if record["team_id"] == "team_1" and record["phase"] == "depth"]
    assert [record["elapsed_seconds"] for record in clarified_records] == [10, 20]
    for run_id, expected_provenance in [("synthetic_recovery", "mock_heuristic"), ("synthetic_timeout", "timer")]:
        game = load_game(output, runs[run_id])
        failures = [record for record in game["history"] if record["outcome"] == "fail"]
        assert len(failures) == 1
        assert failures[0]["phase"] == "depth"
        assert failures[0]["provenance"] == expected_provenance
        assert game["teams"][0]["outcome"] == "recovered"
        assert game["teams"][0]["failures"] == 1
        assert game["teams"][1]["outcome"] == "success"
        assert [record["phase"] for record in game["teams"][0]["history"]] == ["intro", "decision", "depth", "education"]
        assert game["final_result"]["outcome"] == "continue"
    timeout = load_game(output, runs["synthetic_timeout"])
    assert next(record for record in timeout["history"] if record["provenance"] == "timer")["elapsed_seconds"] == 91
    assert timeout["teams"][0]["history"][-1]["elapsed_seconds"] == 10


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf", "-0.1", "not-a-number"])
def test_rejects_invalid_elapsed_seconds(tmp_path, value):
    with pytest.raises(ValueError, match="elapsed_seconds"):
        read_rows(edited_csv(tmp_path, elapsed_seconds=value))


def test_expected_outcome_mismatch_publishes_no_results(tmp_path):
    source = edited_csv(tmp_path, expected_outcome="fail")
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="expected_outcome=fail"):
        replay_csv(source, output, tmp_path / "game.sqlite3")
    assert not output.exists()


def test_phase_mismatch_is_rejected_before_submission(tmp_path):
    source = edited_csv(tmp_path, phase="decision")
    with pytest.raises(ValueError, match="ожидался этап decision"):
        replay_csv(source, tmp_path / "results", tmp_path / "game.sqlite3")


@pytest.mark.parametrize("run_id", ["../victim", "a/b", "/absolute", ".", "a\\b", "x" * 65])
def test_rejects_unsafe_run_id_before_writing(tmp_path, run_id):
    source = edited_csv(tmp_path, run_id=run_id)
    sentinel = tmp_path / "victim"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="run_id"):
        replay_csv(source, tmp_path / "results", tmp_path / "game.sqlite3")
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (tmp_path / "game.sqlite3").exists()


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    existing = output / "transcript_synthetic_happy.csv"
    existing.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="уже существуют"):
        replay_csv(HAPPY, output, tmp_path / "game.sqlite3")
    assert existing.read_text(encoding="utf-8") == "preserve"
    assert not (tmp_path / "game.sqlite3").exists()


def test_expected_outcome_column_is_optional(tmp_path):
    with HAPPY.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = [field for field in reader.fieldnames if field != "expected_outcome"]
        records = [{key: value for key, value in record.items() if key != "expected_outcome"} for record in reader]
    path = tmp_path / "optional.csv"
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fields)
        writer.writeheader()
        writer.writerows(records)
    summary = replay_csv(path, tmp_path / "optional", tmp_path / "game.sqlite3")
    assert summary["runs"][0]["final_outcome"] == "continue"


def test_cli_replays_csv(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "game_agent.simulation", "--input", str(HAPPY),
         "--output", str(tmp_path / "results"), "--db", str(tmp_path / "game.sqlite3")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["runs"][0]["phase"] == "finished"
