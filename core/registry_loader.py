"""Utility functions for loading custom action definitions from disk.

This module is intentionally schema-agnostic: it only returns raw Python
structures. Validation against ``ActionDefinition`` is performed in
``core.action_registry``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

try:
    import yaml  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    yaml = None


DEFAULT_CUSTOM_ACTIONS_PATH = os.path.expanduser("~/.alara/custom_actions.yaml")


def _resolve_path(path: Optional[Union[str, Path]]) -> Path:
    if path is not None:
        return Path(path).expanduser()
    env_path = os.getenv("ALARA_CUSTOM_ACTIONS_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return Path(DEFAULT_CUSTOM_ACTIONS_PATH).expanduser()


def load_custom_actions(path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Load raw custom action definitions from YAML or JSON.

    Returns a list of mapping objects suitable for validation by
    ``ActionDefinition``. Malformed files result in warnings but do not
    raise.
    """
    file_path = _resolve_path(path)
    if not file_path.exists():
        logger.debug("No custom actions file found at {}", file_path)
        return []

    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - filesystem errors
        logger.warning("Failed to read custom actions file {}: {}", file_path, exc)
        return []

    suffix = file_path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                logger.warning(
                    "PyYAML is not installed; cannot load YAML custom actions from {}",
                    file_path,
                )
                return []
            data = yaml.safe_load(text)  # type: ignore[arg-type]
        else:
            # Treat everything else as JSON.
            data = json.loads(text)
    except Exception as exc:
        logger.warning("Failed to parse custom actions file {}: {}", file_path, exc)
        return []

    if data is None:
        return []

    # Support either a list of action dicts or a mapping name -> dict.
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        return [v for v in data.values() if isinstance(v, dict)]

    logger.warning(
        "Unexpected custom actions data structure in {} (type={}): ignoring",
        file_path,
        type(data),
    )
    return []

