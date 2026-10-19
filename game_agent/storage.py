"""Persist complete game states with optimistic concurrency control."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
from typing import Any


class ConcurrentUpdate(ValueError):
    """The session changed after the caller loaded its snapshot."""


class SQLiteStore:
    """Store JSON session snapshots using a fresh connection for each operation.

    ``save`` uses the loaded version as a compare-and-swap token. Callers must
    retain the returned snapshot, whose version advances after a successful save.
    Returned values never share nested mutable objects with the input state.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS game_sessions ("
                "id TEXT PRIMARY KEY, "
                "version INTEGER NOT NULL, "
                "state_json TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @staticmethod
    def _serialize(state: dict[str, Any], version: int) -> tuple[str, str]:
        if not isinstance(state, dict):
            raise TypeError("Session state must be a dictionary")
        game_id = state.get("id")
        if not isinstance(game_id, str):
            raise ValueError("Session id must be a string")
        # Only the shallow copy is changed; JSON serialization copies the rest.
        snapshot = {**state, "version": version}
        return game_id, json.dumps(snapshot, ensure_ascii=False, allow_nan=False)

    def create(self, state: dict[str, Any]) -> dict[str, Any]:
        """Create a version-zero session; duplicate ids raise IntegrityError."""
        version = state.get("version", 0)
        if type(version) is not int or version != 0:
            raise ValueError("New sessions must have version 0")
        game_id, serialized = self._serialize(state, version=0)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO game_sessions (id, version, state_json) VALUES (?, ?, ?)",
                (game_id, 0, serialized),
            )
        return json.loads(serialized)

    def get(self, game_id: str) -> dict[str, Any]:
        """Load an independent snapshot, raising KeyError for an unknown id."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT state_json FROM game_sessions WHERE id = ?", (game_id,)
            ).fetchone()
        if row is None:
            raise KeyError(game_id)
        return json.loads(row[0])

    def save(self, state: dict[str, Any], expected_version: int) -> dict[str, Any]:
        """Atomically replace a session only when its version still matches."""
        if type(expected_version) is not int or expected_version < 0:
            raise ValueError("Expected version must be a nonnegative integer")
        next_version = expected_version + 1
        game_id, serialized = self._serialize(state, version=next_version)
        with closing(self._connect()) as connection, connection:
            result = connection.execute(
                "UPDATE game_sessions SET state_json = ?, version = ? "
                "WHERE id = ? AND version = ?",
                (serialized, next_version, game_id, expected_version),
            )
            if result.rowcount != 1:
                exists = connection.execute(
                    "SELECT 1 FROM game_sessions WHERE id = ?", (game_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(game_id)
                raise ConcurrentUpdate(
                    f"Session {game_id!r} changed from version {expected_version}"
                )
        return json.loads(serialized)
