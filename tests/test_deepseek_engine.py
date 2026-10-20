"""Provider integration checks using real LangGraph and SQLite, without API calls."""

from copy import deepcopy
import csv
import io
import json

import pytest
from pydantic import ValidationError

from game_agent.evaluation import Assessment, SCORE_AXES
from game_agent.game import GameService, transcript_csv


PLAYERS = [f"Участник {number}" for number in range(1, 7)]
ANSWER = "Распределим роли, согласуем помощь специалиста, проверим риски и запасной план."
TEST_API_KEY = "synthetic-deepseek-secret-never-persist-this"


class FakeEvaluator:
    def __init__(self, response_mode="valid"):
        self.response_mode = response_mode
        self.contexts = []

    def evaluate(self, context):
        self.contexts.append(deepcopy(context))
        if self.response_mode != "valid":
            # A provider failure must not persist even an evaluator's changes.
            context["history"].clear()
            if self.response_mode == "missing":
                return None
            return Assessment.model_validate({"outcome": "invalid-provider-value"})
        return Assessment(
            outcome="continue",
            feedback="План содержит конкретные действия и учитывает последствия.",
            scores={axis: 1 for axis in SCORE_AXES},
            reasons=["Команда распределила роли и предусмотрела запасной план."],
        )


def create_service(path, mode="deepseek", evaluator=None, api_key=None):
    return GameService(
        path,
        mode=mode,
        api_key=api_key,
        evaluator=evaluator if evaluator is not None else FakeEvaluator(),
        clock=lambda: 1000.0,
    )


def create_game(service):
    return service.create_game(PLAYERS, timed=False, scenario_ids=["pet", "family"])


def reach_depth(service, game, team):
    choices = ("vet", "drive") if team["scenario_id"] == "pet" else ("gift", "cheap")
    for choice in choices:
        game = service.submit(
            game["id"], team["id"], ANSWER, choice,
            expected_version=game["version"],
        )
    return game


def test_deepseek_session_keeps_authored_branches_and_provider_in_state_and_csv(tmp_path):
    path = tmp_path / "sessions.sqlite3"
    evaluator = FakeEvaluator()
    service = create_service(path, evaluator=evaluator, api_key=TEST_API_KEY)
    game = create_game(service)
    assert game["mode"] == "deepseek"
    assert service.load_game(game["id"]) == game

    for team in game["teams"]:
        game = reach_depth(service, game, team)
        assert [record["provenance"] for record in game["history"][-2:]] == ["authored_branch"] * 2
        game = service.submit(game["id"], team["id"], ANSWER, expected_version=game["version"])
        assert game["history"][-1]["provenance"] == "deepseek"
    assert game["phase"] == "final"
    game = service.submit_final(game["id"], ANSWER, expected_version=game["version"])
    assert game["phase"] == "finished"
    assert game["final_result"]["provenance"] == "deepseek"
    assert game["history"][-1]["phase"] == "final"
    assert len(evaluator.contexts) == 3
    assert evaluator.contexts[-1]["phase"] == "final"

    exported = transcript_csv(game)
    rows = list(csv.DictReader(io.StringIO(exported)))
    assert len(rows) == len(game["history"]) == 7
    assert {row["mode"] for row in rows} == {"deepseek"}
    assert {row["provenance"] for row in rows if row["phase"] in {"intro", "decision"}} == {"authored_branch"}
    assert {row["provenance"] for row in rows if row["phase"] in {"depth", "final"}} == {"deepseek"}
    assert service.load_game(game["id"]) == game
    assert TEST_API_KEY not in json.dumps(game, ensure_ascii=False)
    assert TEST_API_KEY not in json.dumps(evaluator.contexts, ensure_ascii=False)
    assert TEST_API_KEY not in exported
    assert TEST_API_KEY.encode() not in path.read_bytes()


def test_reopened_deepseek_service_continues_existing_session_with_new_credentials(tmp_path):
    path = tmp_path / "sessions.sqlite3"
    original = create_service(path, api_key=TEST_API_KEY)
    game = create_game(original)
    team = game["teams"][0]
    game = original.submit(game["id"], team["id"], ANSWER, "vet", expected_version=0)

    resumed_evaluator = FakeEvaluator()
    resumed = create_service(path, evaluator=resumed_evaluator, api_key="replacement-synthetic-key")
    assert resumed.load_game(game["id"]) == game
    game = resumed.submit(game["id"], team["id"], ANSWER, "drive", expected_version=1)
    game = resumed.submit(game["id"], team["id"], ANSWER, expected_version=2)
    assert game["version"] == 3
    assert game["mode"] == "deepseek"
    assert game["teams"][0]["phase"] == "ready"
    assert game["history"][-1]["provenance"] == "deepseek"
    assert len(resumed_evaluator.contexts) == 1
    assert original.load_game(game["id"]) == game


def test_submissions_cannot_change_provider_for_any_persisted_session(tmp_path):
    path = tmp_path / "sessions.sqlite3"
    evaluators = {mode: FakeEvaluator() for mode in ("deepseek", "openai", "mock")}
    services = {mode: create_service(path, mode=mode, evaluator=evaluator) for mode, evaluator in evaluators.items()}
    for source_mode, source_service in services.items():
        game = create_game(source_service)
        team = game["teams"][0]
        for destination_mode, other_service in services.items():
            if source_mode == destination_mode:
                continue
            with pytest.raises(ValueError, match="режим оценки"):
                other_service.submit(game["id"], team["id"], ANSWER, "alternative", expected_version=0)
            with pytest.raises(ValueError, match="режим оценки"):
                other_service.submit_final(game["id"], ANSWER, expected_version=0)
            assert source_service.load_game(game["id"]) == game
    assert all(not evaluator.contexts for evaluator in evaluators.values())


@pytest.mark.parametrize("response_mode, expected_error", [("missing", AttributeError), ("invalid", ValidationError)])
def test_deepseek_missing_or_invalid_model_response_rolls_back_and_can_retry(tmp_path, response_mode, expected_error):
    evaluator = FakeEvaluator(response_mode)
    service = create_service(tmp_path / "sessions.sqlite3", evaluator=evaluator)
    game = create_game(service)
    team = game["teams"][0]
    game = reach_depth(service, game, team)
    before = deepcopy(game)

    with pytest.raises(expected_error):
        service.submit(game["id"], team["id"], ANSWER, expected_version=game["version"])
    assert len(evaluator.contexts) == 1
    assert service.load_game(game["id"]) == before
    assert game == before

    evaluator.response_mode = "valid"
    retried = service.submit(game["id"], team["id"], ANSWER, expected_version=game["version"])
    assert retried["version"] == before["version"] + 1
    assert len(retried["history"]) == len(before["history"]) + 1
    assert retried["history"][-1]["provenance"] == "deepseek"
