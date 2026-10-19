"""Exercise the persistent game flow through the actual Streamlit controls."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from game_agent.game import GameService, current_prompt


ROOT = Path(__file__).resolve().parents[1]
PARTICIPANTS = "Анна\nБорис\nВера\nГлеб\nДарья\nЕгор"


def click(app: AppTest, label: str) -> AppTest:
    return next(button for button in app.button if button.label == label).click().run()


def test_persistent_game_create_submit_and_restore(tmp_path, monkeypatch):
    database = tmp_path / "game.sqlite3"
    monkeypatch.setenv("GAME_DB_PATH", str(database))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = AppTest.from_file(str(ROOT / "play.py")).run()
    assert not app.exception
    next(area for area in app.text_area if area.label.startswith("Участники")).set_value(PARTICIPANTS)
    click(app, "Создать игру")
    assert not app.exception
    game_id = app.session_state["game_id"]
    service = GameService(db_path=str(database), mode="mock")
    game = service.load_game(game_id)
    assert sum(len(team["members"]) for team in game["teams"]) == 6
    assert all(2 <= len(team["members"]) <= 3 for team in game["teams"])
    assert not game["history"]

    # Selecting an option alone cannot bypass the mandatory explanation.
    assert app.radio
    app.radio[0].set_value(current_prompt(game, game["teams"][0]["id"])["options"][0]["id"])
    click(app, "Отправить решение")
    assert not app.exception
    assert app.warning
    assert not service.load_game(game_id)["history"]

    next(area for area in app.text_area if area.label == "Решение команды и объяснение").set_value(
        "Обсудим риски, назначим ответственного и обратимся к специалисту. "
        "Другой участник организует помощь, а при задержке используем запасной план."
    )
    click(app, "Отправить решение")
    assert not app.exception
    assert len(service.load_game(game_id)["history"]) == 1

    # A fresh browser session can recover the SQLite game by its public ID.
    restored = AppTest.from_file(str(ROOT / "play.py")).run()
    next(field for field in restored.text_input if field.label == "ID игры").set_value(game_id)
    click(restored, "Восстановить игру")
    assert not restored.exception
    assert restored.session_state["game_id"] == game_id
    assert any("Последняя обратная связь" in element.value for element in restored.markdown)


def test_invalid_participant_count_keeps_creation_form(tmp_path, monkeypatch):
    monkeypatch.setenv("GAME_DB_PATH", str(tmp_path / "game.sqlite3"))
    app = AppTest.from_file(str(ROOT / "play.py")).run()
    next(area for area in app.text_area if area.label.startswith("Участники")).set_value("Анна\nБорис")
    click(app, "Создать игру")
    assert not app.exception
    assert app.warning
    assert "game_id" not in app.session_state


def test_common_final_awards_downloads_and_finished_persistence(tmp_path, monkeypatch):
    database = tmp_path / "game.sqlite3"
    monkeypatch.setenv("GAME_DB_PATH", str(database))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = GameService(db_path=str(database), mode="mock")
    game = service.create_game(PARTICIPANTS.splitlines(), seed=42, timed=True)
    team_plan = (
        "Сначала обсудим риски и безопасные действия, потому что важна помощь семье и животным. "
        "Один участник позвонит специалисту, второй организует помощь, вместе согласуем роли. "
        "Если специалист занят, проверим запасной план. Обсудим опыт и выводы команды."
    )
    # Set up the actual persisted state through every normal team stage, without
    # replacing the graph or evaluator with UI-specific test doubles.
    for team in game["teams"]:
        team_id = team["id"]
        for expected_phase in ("intro", "decision", "depth"):
            current_team = next(item for item in game["teams"] if item["id"] == team_id)
            assert current_team["phase"] == expected_phase
            prompt = current_prompt(game, team_id)
            choice_id = prompt["options"][0]["id"] if prompt["options"] else None
            game = service.submit(
                game["id"], team_id, team_plan, choice_id=choice_id,
                expected_version=game["version"],
            )
    assert game["phase"] == "final"
    assert all(team["phase"] == "ready" for team in game["teams"])
    assert len(game["history"]) == 6

    app = AppTest.from_file(str(ROOT / "play.py")).run()
    next(field for field in app.text_input if field.label == "ID игры").set_value(game["id"])
    click(app, "Восстановить игру")
    assert not app.exception
    assert any(element.value == "Общее финальное испытание" for element in app.subheader)
    assert any("Все команды готовы" in element.value for element in app.info)
    assert not any(button.label == "Отправить решение" for button in app.button)

    final_plan = (
        "Обсудим опыт всех команд, реальные потребности семьи, людей и животных и определим бюджет. "
        "Один участник позвонит в благотворительную организацию и уточнит необходимую помощь, "
        "второй организует сбор вещей, третий согласует посильное пожертвование с командами. "
        "Вместе поможем людям и животным, сохраним резерв для потребностей семьи и откажемся "
        "от лишних покупок, потому что цель — осознанное потребление и забота. Если возникнет "
        "задержка, используем запасной план. Проверим безопасность и обсудим выводы."
    )
    next(area for area in app.text_area if area.label == "Общий план и объяснение").set_value(final_plan)
    click(app, "Отправить общий план")
    assert not app.exception
    assert any("Игра завершена" in element.value for element in app.success)
    assert len([element for element in app.subheader if element.value.startswith("🏅")]) == 2
    downloads = {element.proto.label: element.proto for element in app.get("download_button")}
    assert downloads["Результаты CSV"].url
    assert downloads["Результаты JSON"].url
    assert any("История игры: 7 ответов" in element.label for element in app.expander)

    # A newly constructed service and a UI refresh see the final saved result;
    # refreshing cannot submit the final form for a second time.
    finished = GameService(db_path=str(database), mode="mock").load_game(game["id"])
    assert finished["phase"] == "finished"
    assert finished["final_result"]["outcome"] == "continue"
    assert len(finished["awards"]) == len(finished["teams"])
    assert len(finished["history"]) == 7
    assert finished["history"][-1]["phase"] == "final"
    assert finished["history"][-1]["answer"] == final_plan
    click(app, "Обновить время и состояние")
    assert not app.exception
    assert not any(button.label == "Отправить общий план" for button in app.button)
    assert len(service.load_game(game["id"])["history"]) == 7
