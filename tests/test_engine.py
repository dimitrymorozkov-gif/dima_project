"""Behavioral checks against the real LangGraph game and persistent service."""

from collections import deque
from copy import deepcopy

import pytest
from pydantic import ValidationError

from game_agent.catalog import load_catalog
from game_agent.evaluation import Assessment, MockEvaluator, SCORE_AXES
from game_agent.game import GameService, LIMITS, current_prompt
from game_agent.storage import ConcurrentUpdate


PLAYERS = [f"Участник {index}" for index in range(1, 7)]
PLAN = (
    "Распределим роли: один позвонит специалисту, второй спокойно поддержит семью. "
    "Вместе обсудим риски и семейный бюджет, чтобы помочь людям и животным. "
    "Если помощь задержится, проверим запасной план и затем обсудим выводы."
)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class ScriptedEvaluator:
    def __init__(self, *outcomes):
        self.outcomes = deque(outcomes)
        self.contexts = []

    def evaluate(self, context):
        self.contexts.append(deepcopy(context))
        outcome = self.outcomes.popleft() if self.outcomes else "continue"
        return Assessment(
            outcome=outcome,
            feedback="У плана есть конкретные действия и обсуждение последствий.",
            follow_up="Кто отвечает за связь со специалистом?" if outcome == "clarify" else "",
            scores={axis: 1 for axis in SCORE_AXES},
            reasons=["Команда описала совместные действия."],
            educational="Согласуйте роли и запасной план.",
        )


class LatentEvaluator(ScriptedEvaluator):
    """Simulate one 45-second API request without wall-clock waiting."""

    def __init__(self, *outcomes):
        super().__init__(*outcomes)
        self.clock = None

    def evaluate(self, context):
        result = super().evaluate(context)
        self.clock.now += 45
        return result


def new_game(tmp_path, *, evaluator=None, **settings):
    clock = Clock()
    evaluator = evaluator if evaluator is not None else ScriptedEvaluator()
    service = GameService(tmp_path / "sessions.sqlite3", clock=clock, evaluator=evaluator)
    game = service.create_game(
        PLAYERS, scenario_ids=settings.pop("scenario_ids", ["pet"]),
        timed=settings.pop("timed", False), **settings,
    )
    return service, clock, evaluator, game


def team_of(game, team_id):
    return next(team for team in game["teams"] if team["id"] == team_id)


def choose(service, game, team_id, choice_id):
    return service.submit(game["id"], team_id, PLAN, choice_id, expected_version=game["version"])


def reach_depth(service, game, team_id):
    team = team_of(game, team_id)
    intro, decision = ("vet", "drive") if team["scenario_id"] == "pet" else ("gift", "expensive")
    game = choose(service, game, team_id, intro)
    return choose(service, game, team_id, decision)


def finish_team(service, game, team_id):
    game = reach_depth(service, game, team_id)
    return service.submit(game["id"], team_id, PLAN, expected_version=game["version"])


def reach_final(service, game):
    for team_id in [team["id"] for team in game["teams"]]:
        game = finish_team(service, game, team_id)
    assert game["phase"] == "final"
    return game


@pytest.mark.parametrize("count", range(6, 11))
def test_groups_have_two_or_three_people_without_losses_and_seed_is_repeatable(tmp_path, count):
    service = GameService(tmp_path / "sessions.sqlite3", clock=Clock())
    players = [f"Игрок {index}" for index in range(count)]
    first = service.create_game(players, seed=123)
    second = service.create_game(players, seed=123)
    groups = [team["members"] for team in first["teams"]]
    assert all(len(group) in {2, 3} for group in groups)
    assert sorted(name for group in groups for name in group) == sorted(players)
    assert len({name for group in groups for name in group}) == count
    assert groups == [team["members"] for team in second["teams"]]


@pytest.mark.parametrize("participants", [
    PLAYERS[:5], PLAYERS + ["7", "8", "9", "10", "11"],
    PLAYERS[:-1] + [" участник 1 "], PLAYERS[:-1] + [" "],
    PLAYERS[:-1] + ["x" * 81], PLAYERS[:-1] + [1], "Участники",
])
def test_invalid_participants_are_rejected(tmp_path, participants):
    service = GameService(tmp_path / "sessions.sqlite3", clock=Clock())
    with pytest.raises(ValueError):
        service.create_game(participants)


@pytest.mark.parametrize("settings", [
    {"policy": "unknown"}, {"scenario_ids": ["unknown"]}, {"scenario_ids": []},
    {"limits": {"intro": 0}}, {"limits": {"depth": True}},
    {"limits": {"decision": float("inf")}}, {"limits": {"unknown": 1}},
])
def test_invalid_game_configuration_is_rejected(tmp_path, settings):
    service = GameService(tmp_path / "sessions.sqlite3", clock=Clock())
    with pytest.raises(ValueError):
        service.create_game(PLAYERS, **settings)


def test_graph_progresses_in_order_and_draws_event_only_after_decision(tmp_path):
    service, _, evaluator, game = new_game(tmp_path)
    assert {"validate", "evaluate", "consequence", "timeout", "advance", "barrier"} <= set(service.graph.get_graph().nodes)
    team_id = game["teams"][0]["id"]
    assert team_of(game, team_id)["event"] is None
    game = choose(service, game, team_id, "vet")
    assert team_of(game, team_id)["phase"] == "decision"
    assert team_of(game, team_id)["event"] is None
    game = choose(service, game, team_id, "drive")
    team = team_of(game, team_id)
    assert team["phase"] == "depth"
    assert team["event"] in load_catalog()["pet"]["events"]
    assert team["event"]["question"] in current_prompt(game, team_id)["prompt"]
    assert team["choices"] == {"intro": "vet", "decision": "drive"}
    game = service.submit(game["id"], team_id, PLAN)
    assert team_of(game, team_id)["phase"] == "ready"
    assert [entry["phase"] for entry in team_of(game, team_id)["history"]] == ["intro", "decision", "depth"]
    assert evaluator.contexts[0]["event"] == team["event"]
    assert evaluator.contexts[0]["team_size"] in {2, 3}
    assert game["phase"] == "playing"


def test_identical_seed_and_choices_select_identical_events(tmp_path):
    service, _, _, first = new_game(tmp_path, seed="fixed-seed")
    second = service.create_game(PLAYERS, seed="fixed-seed", timed=False, scenario_ids=["pet"])
    first = reach_depth(service, first, first["teams"][0]["id"])
    second = reach_depth(service, second, second["teams"][0]["id"])
    assert first["teams"][0]["event"] == second["teams"][0]["event"]


def test_psychologist_path_keeps_family_out_of_the_store(tmp_path):
    service, _, evaluator, game = new_game(tmp_path, scenario_ids=["family"])
    team_id = game["teams"][0]["id"]
    game = choose(service, game, team_id, "psychologist")
    prompt = current_prompt(game, team_id)
    assert "без обязательной покупки" in prompt["prompt"]
    assert [option["id"] for option in prompt["options"]] == ["alternative"]
    game = choose(service, game, team_id, "alternative")
    team = team_of(game, team_id)
    assert team["phase"] == "depth"
    assert team["event"]["id"] == "appointment"
    assert team["event"]["source"] == "mvp_extension"
    assert "магазин" not in current_prompt(game, team_id)["prompt"].lower()
    assert evaluator.contexts[0]["scenario_id"] == "family"


def test_free_text_family_alternative_does_not_teleport_to_a_store(tmp_path):
    service, _, _, game = new_game(tmp_path, scenario_ids=["family"])
    team_id = game["teams"][0]["id"]
    answer = (
        "Покупку отложим. Спокойно обсудим чувства ребенка и семейные границы, "
        "родители согласуют правила, затем договоримся о консультации психолога."
    )
    game = service.submit(game["id"], team_id, answer, "alternative")
    assert "без обязательной покупки" in current_prompt(game, team_id)["prompt"]
    game = service.submit(game["id"], team_id, answer, "alternative")
    event = team_of(game, team_id)["event"]
    assert event not in load_catalog()["family"]["events"]
    assert event["source"] == "mvp_extension"
    assert "магазин" not in current_prompt(game, team_id)["prompt"].lower()


@pytest.mark.parametrize("policy, expected_phase, failures", [
    ("flexible", "depth", 0), ("author", "education", 1),
])
def test_authored_chat_consequence_depends_on_selected_policy(tmp_path, policy, expected_phase, failures):
    service, _, evaluator, game = new_game(tmp_path, policy=policy)
    team_id = game["teams"][0]["id"]
    game = choose(service, game, team_id, "vet")
    game = choose(service, game, team_id, "chat")
    team = team_of(game, team_id)
    assert team["phase"] == expected_phase
    assert team["failures"] == failures
    assert team["history"][-1]["provenance"] == "authored_branch"
    assert not evaluator.contexts


def test_free_text_clarification_keeps_phase_event_and_original_deadline(tmp_path):
    evaluator = ScriptedEvaluator("clarify", "continue")
    service, clock, _, game = new_game(tmp_path, evaluator=evaluator, timed=True)
    team_id = game["teams"][0]["id"]
    game = reach_depth(service, game, team_id)
    original = deepcopy(team_of(game, team_id))
    clock.now += 10
    game = service.submit(game["id"], team_id, "Уточним план")
    team = team_of(game, team_id)
    assert team["phase"] == "depth"
    assert team["started_at"] == original["started_at"]
    assert team["event"] == original["event"]
    assert current_prompt(game, team_id)["follow_up"]
    assert current_prompt(game, team_id)["time_limit"] == 90
    clock.now += 5
    game = service.submit(game["id"], team_id, PLAN)
    assert team_of(game, team_id)["phase"] == "ready"
    assert team_of(game, team_id)["history"][-1]["elapsed_seconds"] == 15


def test_failure_has_educational_retry_and_can_recover(tmp_path):
    service, clock, _, game = new_game(tmp_path, evaluator=ScriptedEvaluator("fail", "continue"))
    team_id = game["teams"][0]["id"]
    game = reach_depth(service, game, team_id)
    clock.now += 5
    game = service.submit(game["id"], team_id, "Неудачный план")
    assert team_of(game, team_id)["phase"] == "education"
    assert team_of(game, team_id)["started_at"] == clock.now
    assert "Что измените" in current_prompt(game, team_id)["prompt"]
    game = service.submit(game["id"], team_id, PLAN)
    team = team_of(game, team_id)
    assert team["phase"] == "ready"
    assert team["outcome"] == "recovered"
    assert team["failures"] == 1
    assert game["phase"] == "playing"


@pytest.mark.parametrize("phase, limit", [("intro", 180), ("decision", 120), ("depth", 90), ("education", 180), ("final", 180)])
@pytest.mark.parametrize("overdue", [False, True])
def test_deadlines_accept_exact_boundary_and_reject_overdue_submission(tmp_path, phase, limit, overdue):
    evaluator = ScriptedEvaluator("fail") if phase == "education" else ScriptedEvaluator()
    service, clock, _, game = new_game(tmp_path, timed=True, evaluator=evaluator)
    assert LIMITS[phase] == limit
    team_id = game["teams"][0]["id"]
    choice_id = "vet"
    if phase == "decision":
        game = choose(service, game, team_id, "vet")
        choice_id = "drive"
    elif phase in {"depth", "education"}:
        game = reach_depth(service, game, team_id)
        choice_id = None
        if phase == "education":
            game = service.submit(game["id"], team_id, PLAN)
    elif phase == "final":
        game = reach_final(service, game)
    prompt = current_prompt(game, None if phase == "final" else team_id)
    assert prompt["time_limit"] == limit
    clock.now = prompt["started_at"] + limit + (0.001 if overdue else 0)
    if phase == "final":
        game = service.submit_final(game["id"], PLAN)
    else:
        game = service.submit(game["id"], team_id, PLAN, choice_id)
    record = game["history"][-1]
    assert record["phase"] == phase
    assert record["elapsed_seconds"] == limit + (0.001 if overdue else 0)
    assert (record["provenance"] == "timer") is overdue
    assert record["outcome"] == ("fail" if overdue else "continue")


def test_final_is_a_barrier_then_finishes_with_awards_and_rejects_more_answers(tmp_path):
    service, clock, evaluator, game = new_game(tmp_path)
    team_ids = [team["id"] for team in game["teams"]]
    game = finish_team(service, game, team_ids[0])
    before = deepcopy(game)
    with pytest.raises(ValueError, match="всех команд"):
        service.submit_final(game["id"], PLAN)
    assert service.load_game(game["id"]) == before
    clock.now += 12
    game = finish_team(service, game, team_ids[1])
    assert game["phase"] == "final"
    assert all(team["phase"] == "ready" for team in game["teams"])
    assert game["final_started_at"] == clock.now
    with pytest.raises(ValueError):
        service.submit(game["id"], team_ids[0], PLAN)
    game = service.submit_final(game["id"], PLAN)
    assert evaluator.contexts[-1]["scenario_id"] == "common"
    assert evaluator.contexts[-1]["team_size"] == len(PLAYERS)
    assert game["phase"] == "finished"
    assert game["completed_at"] == clock.now
    assert {award["team_id"] for award in game["awards"]} == set(team_ids)
    assert all(award["score"] == 5 and award["prize"] for award in game["awards"])
    assert all(award["title"] == "Общая цель достигнута" for award in game["awards"])
    finished = deepcopy(game)
    with pytest.raises(ValueError, match="завершена"):
        service.submit_final(game["id"], PLAN)
    with pytest.raises(ValueError, match="завершена"):
        service.submit(game["id"], team_ids[0], PLAN)
    assert service.load_game(game["id"]) == finished


def test_final_clarification_preserves_shared_deadline_and_defers_awards(tmp_path):
    service, clock, evaluator, game = new_game(tmp_path, timed=True)
    game = reach_final(service, game)
    started = game["final_started_at"]
    evaluator.outcomes.append("clarify")
    clock.now += 20
    game = service.submit_final(game["id"], "Уточним общий план")
    assert game["phase"] == "final"
    assert game["final_started_at"] == started
    assert current_prompt(game)["follow_up"]
    assert game["awards"] == []
    game = service.submit_final(game["id"], PLAN)
    assert game["phase"] == "finished"


def test_model_latency_starts_ready_and_common_final_after_response(tmp_path):
    evaluator = LatentEvaluator()
    service, clock, _, game = new_game(tmp_path, timed=True, evaluator=evaluator)
    evaluator.clock = clock
    first_id, last_id = [team["id"] for team in game["teams"]]
    game = finish_team(service, game, first_id)
    game = reach_depth(service, game, last_id)
    clock.now += 20
    submitted_at = clock.now
    game = service.submit(game["id"], last_id, PLAN)
    assert clock.now == submitted_at + 45
    assert team_of(game, last_id)["phase"] == "ready"
    assert team_of(game, last_id)["started_at"] == clock.now
    assert game["phase"] == "final"
    assert game["final_started_at"] == clock.now
    assert game["history"][-1]["elapsed_seconds"] == 20
    final_prompt = current_prompt(game)
    assert final_prompt["time_limit"] - (clock.now - final_prompt["started_at"]) == 180


def test_model_latency_during_team_and_final_clarifications_keeps_player_time(tmp_path):
    evaluator = LatentEvaluator("clarify", "continue")
    service, clock, _, game = new_game(tmp_path, timed=True, evaluator=evaluator)
    evaluator.clock = clock
    first_id, last_id = [team["id"] for team in game["teams"]]
    game = reach_depth(service, game, first_id)
    original_started = team_of(game, first_id)["started_at"]
    clock.now += 10
    game = service.submit(game["id"], first_id, "Уточним план")
    team = team_of(game, first_id)
    assert team["phase"] == "depth"
    assert team["started_at"] == original_started + 45
    assert clock.now - team["started_at"] == 10
    assert team["history"][-1]["elapsed_seconds"] == 10
    clock.now += 5
    game = service.submit(game["id"], first_id, PLAN)
    assert team_of(game, first_id)["phase"] == "ready"
    assert team_of(game, first_id)["history"][-1]["elapsed_seconds"] == 15

    game = finish_team(service, game, last_id)
    final_started = game["final_started_at"]
    evaluator.outcomes.append("clarify")
    clock.now += 20
    game = service.submit_final(game["id"], "Уточним общий план")
    assert game["phase"] == "final"
    assert game["final_started_at"] == final_started + 45
    assert clock.now - game["final_started_at"] == 20
    assert game["history"][-1]["elapsed_seconds"] == 20
    clock.now += 5
    game = service.submit_final(game["id"], PLAN)
    assert game["phase"] == "finished"
    assert game["history"][-1]["elapsed_seconds"] == 25


def test_session_restores_after_service_restart_and_can_continue(tmp_path):
    service, clock, _, game = new_game(tmp_path)
    team_id = game["teams"][0]["id"]
    game = choose(service, game, team_id, "vet")
    reloaded = GameService(tmp_path / "sessions.sqlite3", clock=clock, evaluator=ScriptedEvaluator())
    assert reloaded.load_game(game["id"]) == game
    resumed = choose(reloaded, game, team_id, "drive")
    assert resumed["version"] == game["version"] + 1
    assert team_of(resumed, team_id)["phase"] == "depth"
    assert reloaded.load_game(game["id"]) == resumed


def test_stale_version_rejects_submission_before_running_evaluator(tmp_path):
    service, _, evaluator, game = new_game(tmp_path)
    team_id = game["teams"][0]["id"]
    stale = game["version"]
    game = choose(service, game, team_id, "vet")
    with pytest.raises(ConcurrentUpdate):
        service.submit(game["id"], team_id, PLAN, "alternative", expected_version=stale)
    assert not evaluator.contexts
    assert service.load_game(game["id"]) == game


@pytest.mark.parametrize("answer, choice_id", [(" ", "vet"), ("x" * 12001, "vet"), (PLAN, "unknown"), (PLAN, None)])
def test_invalid_submissions_leave_persisted_state_unchanged(tmp_path, answer, choice_id):
    service, _, evaluator, game = new_game(tmp_path)
    with pytest.raises(ValueError):
        service.submit(game["id"], game["teams"][0]["id"], answer, choice_id)
    assert service.load_game(game["id"]) == game
    assert not evaluator.contexts


def test_model_validation_error_cannot_partially_persist_graph_mutations(tmp_path):
    class InvalidEvaluator:
        def evaluate(self, context):
            context["history"].clear()
            return Assessment.model_validate({"outcome": "invented"})

    service, _, _, game = new_game(tmp_path, evaluator=InvalidEvaluator())
    team_id = game["teams"][0]["id"]
    game = reach_depth(service, game, team_id)
    before = deepcopy(game)
    with pytest.raises(ValidationError):
        service.submit(game["id"], team_id, PLAN)
    assert service.load_game(game["id"]) == before
    assert game == before


def test_pet_alternative_at_home_does_not_force_a_car_journey(tmp_path):
    service, _, evaluator, game = new_game(tmp_path)
    team_id = game['teams'][0]['id']
    game = choose(service, game, team_id, 'vet')
    game = service.submit(game['id'], team_id,
                          'Остаемся дома с кошкой. Один вызовет ветеринара, второй спокойно подготовит информацию и запасной план.',
                          'alternative')
    event = team_of(game, team_id)['event']
    assert event['source'] == 'mvp_extension'
    assert event['id'] == 'alternative_coordination'
    assert 'машин' not in event['text'].lower()
    assert 'авари' not in event['text'].lower()


def test_real_offline_evaluator_completes_both_scenarios_without_api(tmp_path):
    service, _, _, game = new_game(tmp_path, evaluator=MockEvaluator(), scenario_ids=["pet", "family"])
    game = reach_final(service, game)
    game = service.submit_final(game["id"], PLAN)
    assert game["phase"] == "finished"
    assert game["final_result"]["outcome"] == "continue"
    assert game["final_result"]["provenance"] == "mock_heuristic"
    assert game["awards"]
