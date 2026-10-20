"""Provider selection and credential boundaries, with no external API calls."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from game_agent import game as game_module
from game_agent.evaluation import MockEvaluator


ROOT = Path(__file__).resolve().parents[1]
PARTICIPANTS = "Анна\nБорис\nВера\nГлеб\nДарья\nЕгор"


def text_input(app, label):
    return next(field for field in app.text_input if field.label == label)


def select_provider(app, provider):
    return next(field for field in app.selectbox if field.label == "API-провайдер").set_value(provider).run()


def click(app, label):
    return next(button for button in app.button if button.label == label).click().run()


@pytest.fixture
def offline_ui(tmp_path, monkeypatch):
    database = tmp_path / "provider_games.sqlite3"
    monkeypatch.setenv("GAME_DB_PATH", str(database))
    for variable in ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(variable, raising=False)
    calls = []

    def fake_factory(mode="mock", model=None, api_key=None):
        calls.append({"mode": mode, "model": model, "api_key": api_key})
        return MockEvaluator()

    # Keep real GameService, SQLite, and graph behavior. Only evaluation is
    # replaced, so provider wiring is observable without sending any data.
    monkeypatch.setattr(game_module, "create_evaluator", fake_factory)
    return database, calls


def test_deepseek_missing_key_does_not_use_openai_key_or_mock(offline_ui, monkeypatch):
    _, calls = offline_ui
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-env-only")
    app = AppTest.from_file(str(ROOT / "play.py")).run()
    assert not app.exception
    assert next(field for field in app.selectbox if field.label == "API-провайдер").value == "deepseek"
    assert calls[-1]["mode"] == "mock"
    assert calls[-1]["api_key"] is None
    calls.clear()

    app.toggle[0].set_value(False).run()
    assert not app.exception
    assert any("DEEPSEEK_API_KEY" in item.value for item in app.error)
    assert not calls  # No API service or substitute mock is constructed.
    assert not any(button.label in {"Создать игру", "Отправить решение"} for button in app.button)
    assert "game_id" not in app.session_state


def test_selected_provider_uses_own_credentials_and_model(offline_ui, monkeypatch):
    _, calls = offline_ui
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-env")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-env")
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-deepseek-env-model")
    monkeypatch.setenv("OPENAI_MODEL", "test-openai-env-model")
    app = AppTest.from_file(str(ROOT / "play.py")).run()
    assert not app.exception
    assert text_input(app, "Модель").value == "test-deepseek-env-model"
    assert text_input(app, "DeepSeek API key").key == "api_key_deepseek"
    assert text_input(app, "Модель").key == "api_model_deepseek"
    text_input(app, "DeepSeek API key").set_value("test-deepseek-user")
    text_input(app, "Модель").set_value("test-deepseek-user-model")
    app.toggle[0].set_value(False).run()
    assert not app.exception
    assert calls[-1] == {"mode": "deepseek", "model": "test-deepseek-user-model", "api_key": "test-deepseek-user"}
    assert any("отправляются в DeepSeek" in item.value for item in app.caption)

    select_provider(app, "openai")
    assert not app.exception
    assert text_input(app, "OpenAI API key").value == ""
    assert text_input(app, "OpenAI API key").key == "api_key_openai"
    assert text_input(app, "Модель").key == "api_model_openai"
    assert calls[-1] == {"mode": "openai", "model": "test-openai-env-model", "api_key": "test-openai-env"}
    text_input(app, "OpenAI API key").set_value("test-openai-user")
    text_input(app, "Модель").set_value("test-openai-user-model").run()
    assert calls[-1] == {"mode": "openai", "model": "test-openai-user-model", "api_key": "test-openai-user"}
    assert any("отправляются в OpenAI" in item.value for item in app.caption)

    select_provider(app, "deepseek")
    assert not app.exception
    assert calls[-1]["mode"] == "deepseek"
    assert calls[-1]["api_key"] in {"test-deepseek-env", "test-deepseek-user"}
    assert calls[-1]["model"] in {"test-deepseek-env-model", "test-deepseek-user-model"}
    app.toggle[0].set_value(True).run()
    assert calls[-1]["mode"] == "mock"
    assert calls[-1]["api_key"] is None


def test_deepseek_game_persists_and_requires_original_provider(offline_ui):
    database, calls = offline_ui
    app = AppTest.from_file(str(ROOT / "play.py")).run()
    text_input(app, "DeepSeek API key").set_value("test-deepseek-private-key")
    app.toggle[0].set_value(False).run()
    assert not app.exception
    assert calls[-1] == {"mode": "deepseek", "model": "deepseek-flash", "api_key": "test-deepseek-private-key"}
    next(field for field in app.text_area if field.label.startswith("Участники")).set_value(PARTICIPANTS)
    click(app, "Создать игру")
    assert not app.exception
    game_id = app.session_state["game_id"]
    reader = game_module.GameService(db_path=database, mode="mock", evaluator=MockEvaluator())
    assert reader.load_game(game_id)["mode"] == "deepseek"
    app.radio[0].set_value("alternative")
    next(field for field in app.text_area if field.label == "Решение команды и объяснение").set_value(
        "Сначала обсудим риски и безопасность. Один участник позвонит специалисту, "
        "второй организует помощь. Вместе согласуем роли и запасной план, потому что "
        "важна забота о семье и животных. Обсудим опыт команды."
    )
    click(app, "Отправить решение")
    assert not app.exception
    saved = reader.load_game(game_id)
    assert len(saved["history"]) == 1
    assert saved["history"][0]["mode"] == "deepseek"
    assert saved["history"][0]["provenance"] == "deepseek"
    assert "test-deepseek-private-key" not in json.dumps(saved, ensure_ascii=False)

    # The DeepSeek password cannot satisfy OpenAI, even within the same UI.
    calls.clear()
    select_provider(app, "openai")
    assert not app.exception
    assert any("OPENAI_API_KEY" in item.value for item in app.error)
    assert not calls
    assert not any(button.label == "Отправить решение" for button in app.button)

    # A fresh session can inspect a saved API game in demo, but it must select
    # its original provider and supply that provider's key to continue.
    restored = AppTest.from_file(str(ROOT / "play.py")).run()
    text_input(restored, "ID игры").set_value(game_id)
    click(restored, "Восстановить игру")
    assert not restored.exception
    assert any("Эта игра создана с DeepSeek" in item.value for item in restored.warning)
    assert next(button for button in restored.button if button.label == "Отправить решение").disabled
    text_input(restored, "DeepSeek API key").set_value("test-deepseek-restored-key")
    restored.toggle[0].set_value(False).run()
    assert not restored.exception
    assert not next(button for button in restored.button if button.label == "Отправить решение").disabled
    assert calls[-1]["mode"] == "deepseek"
    assert calls[-1]["api_key"] == "test-deepseek-restored-key"
    assert len(reader.load_game(game_id)["history"]) == 1
