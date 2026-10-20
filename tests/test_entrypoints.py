"""Entry points reject bare Python launches and stop safely for missing games."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entrypoint", ["app.py", "play.py"])
def test_bare_python_launch_explains_streamlit_command(entrypoint, tmp_path):
    database = tmp_path / "should_not_be_created.sqlite3"
    child_environment = os.environ.copy()
    child_environment["GAME_DB_PATH"] = str(database)
    result = subprocess.run(
        [sys.executable, str(ROOT / entrypoint)],
        cwd=tmp_path,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "python -m streamlit run app.py" in output
    assert "Traceback" not in output
    assert "NoneType" not in output
    assert "AttributeError" not in output
    assert not database.exists()


def test_missing_saved_game_stops_before_rendering_answer_form(tmp_path, monkeypatch):
    monkeypatch.setenv("GAME_DB_PATH", str(tmp_path / "games.sqlite3"))
    app = AppTest.from_file(str(ROOT / "app.py"))
    app.session_state["game_id"] = "missing-game"
    app.run()

    assert not app.exception
    assert any("Игра или команда не найдена" in item.value for item in app.error)
    assert not any(button.label in {"Отправить решение", "Отправить общий план"} for button in app.button)
    assert not any(area.label in {"Решение команды и объяснение", "Общий план и объяснение"} for area in app.text_area)
