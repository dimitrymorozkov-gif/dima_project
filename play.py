"""Persistent multiplayer game interface: ``streamlit run play.py``."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import streamlit as st

from game_agent.game import ConcurrentUpdate, GameService, current_prompt, transcript_csv


ROOT = Path(__file__).resolve().parent
PROVIDERS = {
    "deepseek": {"name": "DeepSeek", "key_env": "DEEPSEEK_API_KEY", "model_env": "DEEPSEEK_MODEL", "default_model": "deepseek-flash"},
    "openai": {"name": "OpenAI", "key_env": "OPENAI_API_KEY", "model_env": "OPENAI_MODEL", "default_model": "gpt-4.1-mini"},
}
PHASE_NAMES = {
    "intro": "Вводный выбор",
    "decision": "Решение команды",
    "depth": "Неожиданное событие",
    "education": "Разбор опыта",
    "ready": "Готова к общему финалу",
    "final": "Общий финал",
    "finished": "Игра завершена",
}


def perform(action: Callable[[], Any], validation_message: str) -> Any | None:
    """Keep credentials and provider error bodies out of the rendered page."""
    try:
        with st.spinner("Ведущий рассматривает решение…"):
            return action()
    except ConcurrentUpdate:
        st.warning(
            "Игра уже изменилась в другой вкладке. Обновите состояние, "
            "прочитайте текущий вопрос и отправьте ответ снова."
        )
    except KeyError:
        st.error("Игра или команда не найдена. Проверьте ID игры и обновите страницу.")
    except ValueError:
        st.warning(validation_message)
    except Exception:
        st.error(
            "Не удалось выполнить запрос. Проверьте настройки режима, "
            "доступ к модели и соединение, затем повторите попытку."
        )
    return None


def show_feedback(item: dict[str, Any]) -> None:
    if item.get("feedback"):
        st.write(item["feedback"])
    if item.get("outcome"):
        labels = {
            "continue": "Решение принято",
            "clarify": "Нужно уточнение",
            "fail": "Разберите последствия решения",
            "balanced": "Взвешенное решение",
            "risky": "Решение с рисками",
            "needs_clarification": "Нужно уточнение",
            "success": "Цель достигнута",
            "failure": "Разберите последствия решения",
            "timeout": "Лимит времени превышен",
        }
        st.caption(labels.get(item["outcome"], item["outcome"]))
    for key, label in (("strengths", "Сильные стороны"), ("risks", "Риски")):
        values = item.get(key)
        if values:
            st.write(f"{label}: " + ("; ".join(map(str, values)) if isinstance(values, list) else str(values)))
    if item.get("scores"):
        with st.expander("Оценка решения"):
            axes = {
                "action": "Конкретные действия", "consequences": "Последствия и запасной план",
                "teamwork": "Роли и координация", "care": "Забота и безопасность",
                "reflection": "Обоснование и выводы",
            }
            for axis, score in item["scores"].items():
                st.write(f"{axes.get(axis, axis)}: {score}/2")
            if item.get("reasons"):
                for reason in item["reasons"]:
                    st.caption(reason)
    if item.get("educational"):
        st.write(item["educational"])
    if item.get("follow_up"):
        st.info(item["follow_up"])


def show_clock(prompt: dict[str, Any]) -> None:
    limit = prompt.get("time_limit")
    if not limit:
        st.caption("Ограничение времени отключено.")
        return
    started_at = prompt.get("started_at")
    try:
        started = float(started_at)
    except (TypeError, ValueError):
        try:
            started = datetime.fromisoformat(str(started_at).replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
            st.caption(f"Лимит этапа: {limit} с.")
            return
    elapsed = max(0, int(time.time() - started))
    remaining = max(0, int(limit) - elapsed)
    st.caption(f"Лимит: {limit} с · Прошло: {elapsed} с · Осталось: {remaining} с")
    if elapsed >= limit:
        st.warning("Лимит этапа истек. Отправьте решение, чтобы ведущий учел время и продолжил игру.")
    st.caption("Время обновляется при действиях на странице; отсчет продолжается при закрытой вкладке.")


def answer_form(
    service: GameService,
    game: dict[str, Any],
    prompt: dict[str, Any],
    team_id: str | None = None,
    enabled: bool = True,
) -> None:
    st.subheader(prompt.get("title") or "Общий финал")
    if prompt.get("goal"):
        st.caption("Цель: " + str(prompt["goal"]))
    st.write(prompt.get("prompt", ""))
    if prompt.get("follow_up"):
        st.info(prompt["follow_up"])
    show_clock(prompt)
    options = prompt.get("options") or []
    form_id = f"answer_{game['id']}_{team_id or 'final'}_{game['version']}"
    with st.form(form_id, clear_on_submit=False):
        choice_id = None
        if options:
            option_by_id = {option["id"]: option["label"] for option in options}
            choice_id = st.radio(
                "Вариант решения",
                list(option_by_id),
                index=None,
                format_func=option_by_id.__getitem__,
                help="Для своего варианта выберите «Другой обоснованный план команды» и объясните его в тексте.",
            )
        answer = st.text_area(
            "Общий план и объяснение" if team_id is None else "Решение команды и объяснение",
            max_chars=12000,
            help="Объясните причины выбора, действия участников, риски и запасной план.",
        )
        submitted = st.form_submit_button(
            "Отправить общий план" if team_id is None else "Отправить решение",
            type="primary",
            disabled=not enabled,
        )
    if not submitted:
        return
    if not answer.strip():
        st.warning("Добавьте объяснение решения. Один выбор варианта недостаточен.")
        return
    if options and choice_id is None:
        st.warning("Выберите вариант решения или «Другой обоснованный план команды».")
        return
    if team_id is None:
        action = lambda: service.submit_final(
            game["id"], answer.strip(), choice_id=choice_id, expected_version=game["version"]
        )
    else:
        action = lambda: service.submit(
            game["id"], team_id, answer.strip(), choice_id=choice_id, expected_version=game["version"]
        )
    updated = perform(action, "Ответ не принят. Обновите состояние и проверьте текущий этап и выбранный вариант.")
    if updated is not None:
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Игра решений — команды", page_icon="🎲", layout="wide")
    st.title("Игра решений")
    st.caption("6–10 участников, команды по 2–3 человека, самостоятельные решения и общий финал.")
    with st.sidebar:
        st.header("Ведущий")
        demo = st.toggle("Демо без API", value=True)
        provider = st.selectbox(
            "API-провайдер", list(PROVIDERS),
            format_func=lambda value: PROVIDERS[value]["name"],
        )
        provider_settings = PROVIDERS[provider]
        api_key = st.text_input(
            f"{provider_settings['name']} API key",
            type="password",
            key=f"api_key_{provider}",
            help="Используется только в текущем сеансе. Ключ не записывается в игру или результаты.",
        )
        model = st.text_input(
            "Модель", key=f"api_model_{provider}",
            value=os.getenv(provider_settings["model_env"], provider_settings["default_model"]),
        )
        effective_key = (api_key.strip() or os.getenv(provider_settings["key_env"], "")).strip()
        mode = "mock" if demo else provider
        if mode == "mock":
            st.caption("Демонстрационный ведущий использует словарную эвристику. Это имитация оценки для проверки игры.")
        else:
            st.caption(
                f"Вопросы, ответы и история игры отправляются в {provider_settings['name']} "
                "для оценки. Игра сохраняется локально."
            )
        examples = ROOT / "data" / "simulated_answers.csv"
        if examples.is_file():
            st.download_button(
                "Скачать примеры ответов CSV", examples.read_bytes(),
                file_name=examples.name, mime="text/csv; charset=utf-8",
            )

    if mode != "mock" and not effective_key:
        st.error(
            f"Для {provider_settings['name']} нужен собственный API-ключ. Укажите его в поле "
            f"«{provider_settings['name']} API key» или задайте {provider_settings['key_env']}. "
            "Создание и продолжение игры недоступны, пока ключ не указан."
        )
        st.stop()

    service = perform(
        lambda: GameService(
            db_path=os.getenv("GAME_DB_PATH", str(ROOT / "runtime" / "game.sqlite3")),
            mode=mode, model=model.strip() or None, api_key=effective_key if mode != "mock" else None,
        ),
        "Проверьте настройки ведущего и путь к хранилищу.",
    )
    if service is None:
        st.stop()

    with st.sidebar:
        st.header("Сохраненная игра")
        with st.form("restore_game"):
            restore_id = st.text_input("ID игры", placeholder="Вставьте сохраненный ID")
            restore = st.form_submit_button("Восстановить игру")
        if restore:
            if not restore_id.strip():
                st.warning("Введите ID сохраненной игры.")
            else:
                restored = perform(
                    lambda: service.load_game(restore_id.strip()),
                    "Проверьте ID сохраненной игры.",
                )
                if restored is not None:
                    st.session_state["game_id"] = restored["id"]
                    st.rerun()

    with st.expander("Новая игра", expanded="game_id" not in st.session_state):
        with st.form("create_game"):
            participants = st.text_area(
                "Участники — по одному имени в строке",
                placeholder="Анна\nБорис\nВера\nГлеб\nДарья\nЕгор",
                help="От 6 до 10 разных имен. Команды сформируются автоматически.",
                height=180,
            )
            seed = st.number_input("Seed распределения команд и событий", min_value=0, max_value=2147483647, value=42, step=1)
            policy = st.selectbox(
                "Правила оценки", ["flexible", "author"],
                format_func=lambda value: "Гибкая оценка" if value == "flexible" else "Авторские критерии",
                help="Гибкая оценка принимает обоснованные альтернативы. Авторские критерии воспроизводят игровую рубрику концепции.",
            )
            timed = st.checkbox("Ограничивать время решений", value=True)
            st.caption("Лимиты: вводный выбор — 180 с, решение — 120 с, углубление — 90 с, общий финал — 180 с.")
            create = st.form_submit_button("Создать игру", type="primary")
        if create:
            names = [line.strip() for line in participants.splitlines() if line.strip()]
            if not 6 <= len(names) <= 10 or len({name.casefold() for name in names}) != len(names):
                st.warning("Введите от 6 до 10 уникальных имен — каждое на отдельной строке.")
            else:
                created = perform(
                    lambda: service.create_game(names, seed=int(seed), policy=policy, timed=timed),
                    "Не удалось создать игру. Проверьте список из 6–10 уникальных участников.",
                )
                if created is not None:
                    st.session_state["game_id"] = created["id"]
                    st.rerun()

    if "game_id" not in st.session_state:
        st.info("Создайте игру или восстановите сохраненную по ID.")
        st.stop()
    game = perform(
        lambda: service.load_game(st.session_state["game_id"]),
        "Не удалось открыть игру. Проверьте ее ID.",
    )
    if game is None:
        st.stop()
    can_submit = game.get("mode", mode) == mode
    if not can_submit and game["phase"] != "finished":
        if game["mode"] == "mock":
            st.warning("Эта игра создана в демо. Включите «Демо без API» для продолжения.")
        else:
            saved_provider = PROVIDERS[game["mode"]]
            st.warning(
                f"Эта игра создана с {saved_provider['name']}. Отключите «Демо без API», "
                f"выберите {saved_provider['name']} в поле «API-провайдер» и укажите его ключ для продолжения."
            )
    st.caption("ID игры — сохраните его для восстановления в этом хранилище:")
    st.code(game["id"], language=None)
    if st.button("Обновить время и состояние"):
        st.rerun()

    teams = game["teams"]
    ready = sum(team["phase"] == "ready" for team in teams)
    st.progress(ready / len(teams), text=f"Готовность к общему финалу: {ready} из {len(teams)} команд")
    for team in teams:
        st.write(f"**{team['name']}** · {', '.join(team['members'])} · {PHASE_NAMES.get(team['phase'], team['phase'])}")

    if game["phase"] == "playing":
        team_by_id = {team["id"]: team for team in teams}
        selected_id = st.selectbox("Текущая команда", list(team_by_id), format_func=lambda value: team_by_id[value]["name"])
        team = team_by_id[selected_id]
        st.caption("Участники: " + ", ".join(team["members"]))
        if team.get("history"):
            st.markdown("**Последняя обратная связь ведущего**")
            show_feedback(team["history"][-1])
        if team["phase"] == "ready":
            st.success("Команда прошла свою историю. Дождитесь остальных команд, затем обсудите общий план.")
        else:
            prompt = perform(lambda: current_prompt(game, selected_id), "Обновите состояние игры.")
            if prompt is not None:
                answer_form(service, game, prompt, selected_id, enabled=can_submit)
    elif game["phase"] == "final":
        st.info("Все команды готовы. Обсудите ситуацию вместе и отправьте один общий план.")
        prompt = perform(lambda: current_prompt(game), "Обновите состояние игры.")
        if prompt is not None:
            answer_form(service, game, prompt, enabled=can_submit)
    elif game["phase"] == "finished":
        st.success("Игра завершена. Обсудите опыт и достижения команд.")
        if game.get("final_result"):
            show_feedback(game["final_result"])
        for award in game.get("awards", []):
            name = award.get("title") or award.get("name") or "Награда команды"
            recipient = award.get("team_name") or next(
                (team["name"] for team in teams if team["id"] == award.get("team_id")), ""
            )
            st.subheader("🏅 " + str(name))
            if recipient:
                st.write(recipient)
            if award.get("reason") or award.get("description"):
                st.write(award.get("reason") or award["description"])
            if award.get("prize"):
                st.write(award["prize"])

    history = game.get("history", [])
    if history:
        with st.expander(f"История игры: {len(history)} ответов"):
            for number, item in enumerate(history, start=1):
                team_name = next((team["name"] for team in teams if team["id"] == item.get("team_id")), "Все команды")
                st.markdown(f"**{number}. {team_name} · {PHASE_NAMES.get(item.get('phase'), item.get('phase', ''))}**")
                st.write(item.get("answer", ""))
                show_feedback(item)
                st.divider()
    with st.expander("Скачать результаты"):
        transcript = perform(lambda: transcript_csv(game), "Не удалось подготовить CSV результатов.")
        if transcript is not None:
            st.download_button(
                "Результаты CSV", transcript.encode("utf-8-sig"),
                file_name=f"game_{game['id']}.csv", mime="text/csv; charset=utf-8",
            )
        st.download_button(
            "Результаты JSON", json.dumps(game, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"game_{game['id']}.json", mime="application/json",
        )


if __name__ == "__main__":
    main()
