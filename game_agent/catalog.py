"""Load the game catalog adapted from the attached author concept.

Author choice labels and final prompts are retained in ``game_catalog.json``.
Events added to complete otherwise unspecified branches carry
``source="mvp_extension"`` in the catalog.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "game_catalog.json"


def load_catalog() -> dict[str, Any]:
    """Return a fresh catalog so callers cannot mutate another session's data."""
    with CATALOG_PATH.open(encoding="utf-8") as catalog_file:
        return json.load(catalog_file)


def scenario(scenario_id: str) -> dict[str, Any]:
    """Return one scenario, raising ``KeyError`` for an unknown identifier."""
    return load_catalog()[scenario_id]
