from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import sqlite3
from threading import Barrier

import pytest

from game_agent.storage import ConcurrentUpdate, SQLiteStore


def test_persistence_after_reopening_and_unicode(tmp_path):
    path = tmp_path / "nested" / "sessions.sqlite3"
    store = SQLiteStore(str(path))
    created = store.create({"id": "игра-1", "answers": ["Распределим роли"]})
    assert created["version"] == 0
    created["answers"].append("Обсудим риски")
    saved = store.save(created, expected_version=0)

    assert SQLiteStore(path).get("игра-1") == saved
    assert saved["version"] == 1
    with sqlite3.connect(path) as connection:
        serialized = connection.execute(
            "SELECT state_json FROM game_sessions WHERE id = ?", ("игра-1",)
        ).fetchone()[0]
    assert "Распределим роли" in serialized


def test_snapshots_never_mutate_or_share_caller_objects(tmp_path):
    store = SQLiteStore(tmp_path / "sessions.sqlite3")
    original = {"id": "session", "history": [{"answer": ["first"]}]}
    original_before = deepcopy(original)
    created = store.create(original)
    assert original == original_before
    created["history"][0]["answer"].append("second")
    assert original == original_before
    assert store.get("session")["history"][0]["answer"] == ["first"]

    before_save = deepcopy(created)
    saved = store.save(created, expected_version=0)
    assert created == before_save
    assert saved["version"] == 1
    saved["history"][0]["answer"].append("third")
    assert created == before_save
    loaded = store.get("session")
    assert loaded["history"][0]["answer"] == ["first", "second"]
    loaded["history"].clear()
    assert store.get("session")["history"]


def test_concurrent_updates_do_not_overwrite_each_other(tmp_path):
    path = tmp_path / "sessions.sqlite3"
    store = SQLiteStore(path)
    store.create({"id": "session", "answer": "initial"})
    # Both writers have read the same persisted version before either may save.
    stores = [SQLiteStore(path), SQLiteStore(path)]
    ready = Barrier(2)

    def write(index):
        snapshot = stores[index].get("session")
        snapshot["answer"] = f"answer-{index}"
        ready.wait(timeout=5)
        try:
            return stores[index].save(snapshot, expected_version=snapshot["version"])
        except ConcurrentUpdate as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, range(2)))
    successes = [result for result in results if isinstance(result, dict)]
    conflicts = [result for result in results if isinstance(result, ConcurrentUpdate)]
    assert len(successes) == len(conflicts) == 1
    assert store.get("session") == successes[0]
    assert successes[0]["version"] == 1


def test_missing_session_and_stale_snapshot_are_distinct(tmp_path):
    store = SQLiteStore(tmp_path / "sessions.sqlite3")
    with pytest.raises(KeyError, match="missing"):
        store.get("missing")
    with pytest.raises(KeyError, match="missing"):
        store.save({"id": "missing"}, expected_version=0)

    state = store.create({"id": "session", "version": 0})
    store.save(state, expected_version=0)
    with pytest.raises(ConcurrentUpdate):
        store.save(state, expected_version=0)
    assert state["version"] == 0
    assert store.get("session")["version"] == 1


def test_ids_are_bound_as_parameters(tmp_path):
    store = SQLiteStore(tmp_path / "sessions.sqlite3")
    game_id = "'); DROP TABLE game_sessions; --"
    state = store.create({"id": game_id})
    assert store.get(game_id) == state
    assert store.save(state, expected_version=0)["version"] == 1
    assert store.create({"id": "another"})["version"] == 0


@pytest.mark.parametrize("version", [-1, 1, True, "0"])
def test_create_requires_initial_version_zero(tmp_path, version):
    store = SQLiteStore(tmp_path / "sessions.sqlite3")
    with pytest.raises(ValueError, match="version 0"):
        store.create({"id": "session", "version": version})
