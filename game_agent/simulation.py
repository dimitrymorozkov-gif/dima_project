"""Replay explicitly synthetic player answers through the real game service.

CSV fixtures are demonstrations, not observations of real people. This module
always uses the offline mock evaluator and never sends API requests.
``elapsed_seconds`` is time since the current stage started, not a sleep or a
per-row delay. All teams share one monotonic virtual clock; later submissions
can therefore exceed their stage deadline even when their requested time is
shorter. ``expected_outcome`` checks the submitted assessment's
``continue``/``clarify``/``fail`` result. It is optional.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable

from .game import GameService, transcript_csv


REQUIRED_COLUMNS = {
    "run_id", "seed", "participant_count", "scenario_ids", "team_id", "phase",
    "choice_id", "answer", "elapsed_seconds",
}
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", re.ASCII)
TEAM_ID = re.compile(r"team_[1-9][0-9]*", re.ASCII)
PHASES = {"intro", "decision", "depth", "education", "final"}
OUTCOMES = {"continue", "clarify", "fail"}


@dataclass(frozen=True)
class SimulationRow:
    run_id: str
    seed: int
    participant_count: int
    scenario_ids: tuple[str, ...]
    team_id: str
    phase: str
    choice_id: str | None
    answer: str
    elapsed_seconds: float
    expected_outcome: str | None = None
    line_number: int = 0


@dataclass
class VirtualClock:
    """One virtual clock per replay, without wall-clock waiting."""

    now: float = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance_to_elapsed(self, started_at: float, elapsed_seconds: float) -> None:
        self.now = max(self.now, started_at + elapsed_seconds)


def read_rows(input_path: str | Path) -> list[SimulationRow]:
    """Parse UTF-8/BOM CSV and reject unsafe identifiers or invalid timings."""
    rows = []
    with Path(input_path).open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fields = reader.fieldnames or []
        missing = REQUIRED_COLUMNS - set(fields)
        if missing:
            raise ValueError(f"CSV: отсутствуют столбцы: {', '.join(sorted(missing))}")
        if len(fields) != len(set(fields)):
            raise ValueError("CSV: названия столбцов должны быть уникальными")
        for raw in reader:
            line = reader.line_num
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"CSV, строка {line}: неверное число полей")
            values = {key: value.strip() for key, value in raw.items()}
            run_id = values["run_id"]
            if not RUN_ID.fullmatch(run_id):
                raise ValueError(f"CSV, строка {line}: run_id должен содержать только латинские буквы, цифры, '_' или '-' (1–64 символа)")
            try:
                seed = int(values["seed"])
                participant_count = int(values["participant_count"])
                elapsed = float(values["elapsed_seconds"])
            except ValueError as error:
                raise ValueError(f"CSV, строка {line}: seed и participant_count — целые числа, elapsed_seconds — число секунд") from error
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError(f"CSV, строка {line}: elapsed_seconds должен быть конечным неотрицательным числом")
            if not 6 <= participant_count <= 10:
                raise ValueError(f"CSV, строка {line}: нужно 6–10 участников")
            scenario_ids = tuple(part.strip() for part in values["scenario_ids"].split("|"))
            if any(scenario_id not in {"pet", "family"} for scenario_id in scenario_ids):
                raise ValueError(f"CSV, строка {line}: scenario_ids содержит неизвестный сценарий")
            team_id, phase = values["team_id"], values["phase"]
            if phase not in PHASES:
                raise ValueError(f"CSV, строка {line}: неизвестный phase {phase!r}")
            if (phase == "final" and team_id != "all") or (phase != "final" and not TEAM_ID.fullmatch(team_id)):
                raise ValueError(f"CSV, строка {line}: для финала укажите all, для этапов команды — team_N")
            choice_id = values["choice_id"] or None
            if phase in {"depth", "education", "final"} and choice_id is not None:
                raise ValueError(f"CSV, строка {line}: этап {phase} принимает свободный ответ без choice_id")
            answer = values["answer"]
            if not 1 <= len(answer) <= 12000:
                raise ValueError(f"CSV, строка {line}: answer должен содержать 1–12000 символов")
            expected = values.get("expected_outcome", "") or None
            if expected is not None and expected not in OUTCOMES:
                raise ValueError(f"CSV, строка {line}: неизвестный expected_outcome {expected!r}")
            rows.append(SimulationRow(
                run_id=run_id, seed=seed, participant_count=participant_count,
                scenario_ids=scenario_ids, team_id=team_id, phase=phase,
                choice_id=choice_id, answer=answer, elapsed_seconds=elapsed,
                expected_outcome=expected, line_number=line,
            ))
    if not rows:
        raise ValueError("CSV: нет ответов для воспроизведения")
    return rows


def _group_runs(rows: Iterable[SimulationRow]) -> dict[str, list[SimulationRow]]:
    runs: dict[str, list[SimulationRow]] = {}
    for row in rows:
        if not RUN_ID.fullmatch(row.run_id):
            raise ValueError("Некорректный run_id")
        run = runs.setdefault(row.run_id, [])
        if run and (row.seed, row.participant_count, row.scenario_ids) != (
            run[0].seed, run[0].participant_count, run[0].scenario_ids
        ):
            raise ValueError(f"run_id {row.run_id}: seed, participant_count и scenario_ids должны совпадать во всех строках")
        run.append(row)
    return runs


def _replay_run(rows: list[SimulationRow], db_path: str | Path) -> dict:
    first = rows[0]
    clock = VirtualClock()
    service = GameService(db_path=db_path, mode="mock", clock=clock)
    game = service.create_game(
        participants=[f"Игрок {index}" for index in range(1, first.participant_count + 1)],
        seed=first.seed, scenario_ids=list(first.scenario_ids), policy="flexible", timed=True,
    )
    for row in rows:
        if row.phase == "final":
            if game["phase"] != "final":
                raise ValueError(f"{row.run_id}, строка {row.line_number}: ожидался final, текущая фаза игры {game['phase']}")
            started_at = game["final_started_at"]
        else:
            team = next((team for team in game["teams"] if team["id"] == row.team_id), None)
            if team is None:
                raise ValueError(f"{row.run_id}, строка {row.line_number}: неизвестная команда {row.team_id}")
            if game["phase"] != "playing" or team["phase"] != row.phase:
                raise ValueError(f"{row.run_id}, строка {row.line_number}: ожидался этап {row.phase}, текущие фазы {game['phase']}/{team['phase']}")
            started_at = team["started_at"]
        clock.advance_to_elapsed(started_at, row.elapsed_seconds)
        if row.phase == "final":
            game = service.submit_final(game["id"], row.answer, expected_version=game["version"])
        else:
            game = service.submit(game["id"], row.team_id, row.answer,
                                  choice_id=row.choice_id, expected_version=game["version"])
        actual = game["history"][-1]["outcome"]
        if row.expected_outcome is not None and actual != row.expected_outcome:
            raise ValueError(f"{row.run_id}, строка {row.line_number}: expected_outcome={row.expected_outcome}, фактический outcome={actual}")
    if game["phase"] != "finished":
        raise ValueError(f"{first.run_id}: CSV закончился до завершения игры (phase={game['phase']})")
    return game


def replay_csv(input_path: str | Path, output_dir: str | Path, db_path: str | Path) -> dict:
    """Replay each synthetic run, then save JSON states and BOM CSV transcripts.

    Existing result files are never overwritten. Choose a fresh output folder
    for a subsequent replay. No artifacts are written for outcome mismatches.
    """
    runs = _group_runs(read_rows(input_path))
    output = Path(output_dir)
    target_paths = [output / "summary.json"]
    for run_id in runs:
        target_paths.extend([output / f"transcript_{run_id}.csv", output / f"game_{run_id}.json"])
    if any(path.exists() or path.is_symlink() for path in target_paths):
        raise ValueError("Файлы результатов уже существуют; укажите новую папку --output")
    # Complete the service/graph replays before publishing any result files.
    games = {run_id: _replay_run(rows, db_path) for run_id, rows in runs.items()}
    summary = {
        "synthetic": True,
        "mode": "mock",
        "source_csv": str(Path(input_path)),
        "runs": [
            {
                "run_id": run_id, "game_id": game["id"], "seed": game["seed"],
                "participant_count": len(game["participants"]), "phase": game["phase"],
                "submission_count": len(game["history"]),
                "final_outcome": game["final_result"]["outcome"],
                "team_outcomes": {team["id"]: team["outcome"] for team in game["teams"]},
                "assessment_outcomes": [record["outcome"] for record in game["history"]],
                "transcript": f"transcript_{run_id}.csv", "game": f"game_{run_id}.json",
            }
            for run_id, game in games.items()
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    for run_id, game in games.items():
        with (output / f"transcript_{run_id}.csv").open("x", encoding="utf-8-sig", newline="") as transcript:
            transcript.write(transcript_csv(game))
        with (output / f"game_{run_id}.json").open("x", encoding="utf-8") as state_file:
            json.dump({"synthetic": True, **game}, state_file, ensure_ascii=False, indent=2, allow_nan=False)
            state_file.write("\n")
    with (output / "summary.json").open("x", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2, allow_nan=False)
        summary_file.write("\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Воспроизведение синтетических ответов игроков через LangGraph в режиме mock")
    parser.add_argument("--input", type=Path, required=True, help="CSV с синтетическими ответами")
    parser.add_argument("--output", type=Path, required=True, help="Новая папка для transcript_*.csv, game_*.json и summary.json")
    parser.add_argument("--db", type=Path, default=Path("runtime/simulation.sqlite3"), help="SQLite для состояния игр")
    args = parser.parse_args(argv)
    try:
        summary = replay_csv(args.input, args.output, args.db)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Ошибка симуляции: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
