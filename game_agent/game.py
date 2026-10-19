"""Persistent group game orchestrated by a LangGraph state machine."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import math
import random
import time
import uuid
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .catalog import load_catalog
from .evaluation import create_evaluator
from .storage import ConcurrentUpdate, SQLiteStore

LIMITS = {"intro": 180, "decision": 120, "depth": 90, "education": 180, "final": 180}
AXES = ("action", "consequences", "teamwork", "care", "reflection")


class GraphState(TypedDict, total=False):
    session: dict
    command: dict
    route: str
    assessment: dict
    record: dict


def _team(game, team_id):
    for team in game["teams"]:
        if team["id"] == team_id:
            return team
    raise ValueError("Команда не найдена в этой игре.")


def _nonpurchase(team):
    return team["scenario_id"] == "family" and team["choices"].get("intro") in {
        "psychologist", "ignore", "shout", "alternative"
    }


def current_prompt(game: dict, team_id: str | None = None) -> dict:
    catalog = load_catalog()
    if game["phase"] == "final" and team_id is None:
        ids = list(dict.fromkeys(t["scenario_id"] for t in game["teams"]))
        parts = []
        for sid in ids:
            final = catalog[sid]["final"]
            variants = "\n".join(o["label"] for o in final["options"])
            parts.append(f"{catalog[sid]['title']}\n{final['prompt']}\n{variants}")
        return {
            "title": "Общее финальное испытание", "goal": "Помощь другим и осознанное отношение к ресурсам.",
            "prompt": "\n\n".join(parts) + "\n\nОбъедините опыт всех команд. Предложите общий план: действия, роли, ресурсы, риски и выводы. Допустимы обоснованные альтернативы.",
            "options": [], "time_limit": game["limits"]["final"] if game["timed"] else None,
            "started_at": game["final_started_at"], "follow_up": game.get("final_follow_up", ""),
        }
    if team_id is None:
        raise ValueError("Для командного этапа укажите команду.")
    team = _team(game, team_id)
    sc = catalog[team["scenario_id"]]
    phase = team["phase"]
    options = []
    if phase in {"intro", "decision"}:
        prompt = sc[phase]["prompt"]
        options = copy.deepcopy(sc[phase]["options"])
        if phase == "decision" and _nonpurchase(team):
            prompt = "Вы выбрали путь без обязательной покупки. Как согласуете действия взрослых, спокойно обсудите желания ребенка и организуете следующий шаг? Объясните последствия и запасной план."
            options = []
        options.append({"id": "alternative", "label": "Другой обоснованный план команды"})
    elif phase == "depth":
        event = team["event"]
        prompt = event["text"] + "\n\n" + event["question"]
    elif phase == "education":
        prompt = sc["education"] + "\n\nЧто измените в своем решении? Предложите конкретные действия, распределите роли и объясните, как избежать прежней ошибки."
    else:
        prompt = "Команда завершила свою историю. Дождитесь остальных команд для общего финала."
    return {
        "title": f"{team['name']} · {sc['title']} · {phase}", "goal": sc["goal"], "prompt": prompt,
        "options": options, "time_limit": game["limits"].get(phase) if game["timed"] else None,
        "started_at": team["started_at"], "follow_up": team.get("follow_up", ""),
    }


def _record(state, assessment):
    game, command = state["session"], state["command"]
    phase = "final" if command["kind"] == "final" else _team(game, command["team_id"])["phase"]
    started = game["final_started_at"] if phase == "final" else _team(game, command["team_id"])["started_at"]
    team = None if phase == "final" else _team(game, command["team_id"])
    return {
        "team_id": command.get("team_id", "all"), "phase": phase,
        "scenario_id": "common" if team is None else team["scenario_id"],
        "event_id": (team.get("event") or {}).get("id", "") if team else "",
        "prompt": current_prompt(game, None if team is None else team["id"])["prompt"],
        "answer": command["answer"], "choice_id": command.get("choice_id"),
        "elapsed_seconds": round(max(0, command["now"] - started), 3),
        "mode": game["mode"], "at": command["now"], **assessment,
    }


def build_game_graph(evaluator, clock=time.time):
    """One invocation is one submitted decision; the whole cohort is the state."""
    def validate(state):
        game, cmd = state["session"], state["command"]
        if game["phase"] == "finished":
            raise ValueError("Игра уже завершена.")
        answer = cmd.get("answer", "").strip()
        if not answer or len(answer) > 12000:
            raise ValueError("Введите решение длиной от 1 до 12000 символов.")
        cmd["answer"] = answer
        if cmd["kind"] == "final":
            if game["phase"] != "final":
                raise ValueError("Общий финал откроется после завершения историй всех команд.")
            prompt = current_prompt(game)
            route = "evaluate"
        else:
            if game["phase"] != "playing":
                raise ValueError("Командные этапы уже завершены.")
            team = _team(game, cmd["team_id"])
            if team["phase"] == "ready":
                raise ValueError("Команда уже готова к финалу.")
            prompt = current_prompt(game, team["id"])
            if prompt["options"] and cmd.get("choice_id") not in {o["id"] for o in prompt["options"]}:
                raise ValueError("Выберите вариант или другой обоснованный план.")
            if not prompt["options"] and cmd.get("choice_id") is not None:
                raise ValueError("На этом этапе требуется свободный ответ без номера варианта.")
            route = "consequence" if team["phase"] in {"intro", "decision"} and cmd.get("choice_id") != "alternative" else "evaluate"
        elapsed = max(0, cmd["now"] - prompt["started_at"])
        if prompt["time_limit"] is not None and elapsed > prompt["time_limit"]:
            route = "timeout"
        return {"route": route}

    def consequence(state):
        game, cmd = state["session"], state["command"]
        team = _team(game, cmd["team_id"])
        option = next(o for o in load_catalog()[team["scenario_id"]][team["phase"]]["options"] if o["id"] == cmd["choice_id"])
        failed = option["author_fail"] and game["policy"] == "author"
        assessment = {
            "outcome": "fail" if failed else "continue", "feedback": option["consequence"],
            "follow_up": "", "scores": {}, "reasons": ["Заданное сюжетное последствие выбранного варианта; свободное объяснение на этом шаге не оценивается моделью."],
            "educational": "", "provenance": "authored_branch",
        }
        return {"assessment": assessment, "record": _record(state, assessment)}

    def evaluate(state):
        game, cmd = state["session"], state["command"]
        final = cmd["kind"] == "final"
        team = None if final else _team(game, cmd["team_id"])
        prompt = current_prompt(game, None if final else team["id"])
        catalog = load_catalog()
        context = {
            "phase": "final" if final else team["phase"], "scenario_id": "common" if final else team["scenario_id"],
            "goal": prompt["goal"], "prompt": prompt["prompt"], "answer": cmd["answer"],
            "event": None if final else team.get("event"), "choice_id": cmd.get("choice_id"),
            "history": game["history"] if final else team["history"], "team_size": len(game["participants"]) if final else len(team["members"]),
            "policy": game["policy"], "teams": game["teams"] if final else [],
            "scenario_goals": [catalog[sid]["goal"] for sid in dict.fromkeys(t["scenario_id"] for t in game["teams"])],
        }
        assessment = evaluator.evaluate(context).model_dump()
        cmd["evaluation_seconds"] = max(0, clock() - cmd["now"])
        assessment["provenance"] = "mock_heuristic" if game["mode"] == "mock" else "openai"
        return {"assessment": assessment, "record": _record(state, assessment)}

    def timeout(state):
        assessment = {
            "outcome": "fail", "feedback": "Время на решение истекло. Разберите, как сократить задержку и распределить действия.",
            "follow_up": "", "scores": {}, "reasons": ["Превышен заданный игровой лимит времени."],
            "educational": "Таймер — правило игры; он не оценивает личные качества участников.", "provenance": "timer",
        }
        return {"assessment": assessment, "record": _record(state, assessment)}

    def advance(state):
        game, cmd, assessment = state["session"], state["command"], state["assessment"]
        record = state["record"]
        game["history"].append(record)
        if cmd["kind"] == "final":
            game["final_result"] = assessment
            game["final_follow_up"] = assessment["follow_up"]
            if assessment["outcome"] != "clarify":
                game["phase"] = "finished"
                game["completed_at"] = cmd["now"]
                won = assessment["outcome"] == "continue"
                game["awards"] = [
                    {"team_id": t["id"], "team_name": t["name"], "title": "Общая цель достигнута" if won else "Участие и разбор опыта", "score": sum(t["scores"].values()), "prize": "Символический диплом"}
                    for t in game["teams"]
                ]
            else:
                game["final_started_at"] += cmd.get("evaluation_seconds", 0)
            return {"session": game}
        team = _team(game, cmd["team_id"])
        team["history"].append(record)
        team["follow_up"] = assessment["follow_up"]
        phase, outcome = team["phase"], assessment["outcome"]
        if outcome == "clarify":
            team["started_at"] += cmd.get("evaluation_seconds", 0)
            return {"session": game}  # Keep the original deadline during clarification.
        if assessment["scores"]:
            team["scores"] = assessment["scores"]
        if outcome == "fail":
            team["failures"] += 1
            team["outcome"] = "fail"
            team["phase"] = "education"
        elif phase == "intro":
            team["choices"][phase] = cmd["choice_id"]
            team["phase"] = "decision"
        elif phase == "decision":
            team["choices"][phase] = cmd["choice_id"]
            sc = load_catalog()[team["scenario_id"]]
            key = team["choices"].get("intro") if _nonpurchase(team) else cmd["choice_id"]
            events = sc.get("extension_events", {}).get(key, sc["events"])
            digest = hashlib.sha256(f"{game['seed']}:{team['id']}:{key}".encode()).digest()
            team["event"] = copy.deepcopy(random.Random(int.from_bytes(digest, "big")).choice(events))
            team["phase"] = "depth"
        else:
            team["phase"] = "ready"
            team["outcome"] = "recovered" if team["failures"] else "success"
        team["started_at"] = max(cmd["now"], clock())
        return {"session": game}

    def barrier(state):
        game = state["session"]
        if game["phase"] == "playing" and all(t["phase"] == "ready" for t in game["teams"]):
            game["phase"] = "final"
            game["final_started_at"] = max(state["command"]["now"], clock())
        return {"session": game}

    graph = StateGraph(GraphState)
    for name, fn in {"validate": validate, "consequence": consequence, "evaluate": evaluate, "timeout": timeout, "advance": advance, "barrier": barrier}.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "validate")
    graph.add_conditional_edges("validate", lambda state: state["route"], {n: n for n in ("consequence", "evaluate", "timeout")})
    for node in ("consequence", "evaluate", "timeout"):
        graph.add_edge(node, "advance")
    graph.add_edge("advance", "barrier")
    graph.add_edge("barrier", END)
    return graph.compile()


class GameService:
    def __init__(self, db_path: str | Path = "runtime/game.sqlite3", mode="mock", model=None, api_key=None, clock=time.time, evaluator=None):
        if mode not in {"mock", "openai"}:
            raise ValueError("Неизвестный режим оценки.")
        self.store = SQLiteStore(db_path)
        self.mode = mode
        self.clock = clock
        self.graph = build_game_graph(evaluator or create_evaluator(mode=mode, model=model, api_key=api_key), clock=clock)

    def create_game(self, participants, seed=42, policy="flexible", timed=True, scenario_ids=None, limits=None):
        if not isinstance(participants, list) or any(not isinstance(n, str) for n in participants):
            raise ValueError("Укажите список имен участников.")
        names = [n.strip() for n in participants]
        if not 6 <= len(names) <= 10 or any(not n or len(n) > 80 for n in names) or len(set(n.casefold() for n in names)) != len(names):
            raise ValueError("Нужны 6–10 участников с разными непустыми именами до 80 символов.")
        if policy not in {"flexible", "author"}:
            raise ValueError("Неизвестная политика оценки.")
        available = load_catalog()
        ids = list(available) if scenario_ids is None else scenario_ids
        if not ids or any(sid not in available for sid in ids):
            raise ValueError("Неизвестный сценарий.")
        configured = {**LIMITS, **(limits or {})}
        if set(configured) != set(LIMITS) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in configured.values()):
            raise ValueError("Лимиты должны быть положительными числами секунд.")
        count = math.ceil(len(names) / 3)
        sizes = [len(names) // count + (i < len(names) % count) for i in range(count)]
        shuffled = names.copy()
        random.Random(seed).shuffle(shuffled)
        now, offset, teams = self.clock(), 0, []
        for i, size in enumerate(sizes):
            teams.append({"id": f"team_{i+1}", "name": f"Команда {i+1}", "members": shuffled[offset:offset+size],
                          "scenario_id": ids[i % len(ids)], "phase": "intro", "started_at": now,
                          "history": [], "choices": {}, "event": None, "scores": {}, "follow_up": "", "outcome": "playing", "failures": 0})
            offset += size
        state = {"id": uuid.uuid4().hex, "version": 0, "participants": names, "seed": seed, "policy": policy,
                 "timed": bool(timed), "limits": configured, "mode": self.mode, "phase": "playing", "teams": teams,
                 "history": [], "created_at": now, "final_started_at": None, "final_result": None, "awards": []}
        return self.store.create(state)

    def load_game(self, game_id):
        return self.store.get(game_id)

    def _submit(self, game_id, command, expected_version):
        game = self.load_game(game_id)
        if game["mode"] != self.mode:
            raise ValueError("Выберите режим оценки, с которым была создана игра.")
        version = game["version"]
        if expected_version is not None and expected_version != version:
            raise ConcurrentUpdate("Состояние изменилось. Обновите игру.")
        command["now"] = self.clock()
        result = self.graph.invoke({"session": copy.deepcopy(game), "command": command})["session"]
        return self.store.save(result, expected_version=version)

    def submit(self, game_id, team_id, answer, choice_id=None, expected_version=None):
        return self._submit(game_id, {"kind": "team", "team_id": team_id, "answer": answer, "choice_id": choice_id}, expected_version)

    def submit_final(self, game_id, answer, choice_id=None, expected_version=None):
        if choice_id is not None:
            raise ValueError("Общий финал принимает свободный план всех команд.")
        return self._submit(game_id, {"kind": "final", "answer": answer}, expected_version)


def transcript_csv(game: dict) -> str:
    fields = ["game_id", "version", "team_id", "scenario_id", "phase", "prompt", "event_id", "choice_id", "answer", "elapsed_seconds", "outcome", "feedback", "follow_up", "mode", "provenance", "score"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fields)
    writer.writeheader()
    teams = {t["id"]: t for t in game["teams"]}
    for record in game["history"]:
        team = teams.get(record["team_id"], {})
        row = {k: record.get(k, "") for k in fields}
        row.update(game_id=game["id"], version=game["version"], scenario_id=record.get("scenario_id", team.get("scenario_id", "common")),
                   event_id=record.get("event_id", ""),
                   score=sum(record.get("scores", {}).values()) if record.get("scores") else "")
        # Keep downloaded team text from becoming an Excel formula.
        writer.writerow({k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r")) else v for k, v in row.items()})
    return buffer.getvalue()
