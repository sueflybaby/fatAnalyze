"""Default config loader."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "targets.yaml"


@lru_cache(maxsize=1)
def load_default_config() -> Dict[str, Any]:
    """Load the bundled targets.yaml as a dict.

    Cached for the process lifetime. Use ``load_config_from(path)`` for
    custom configs.
    """
    return load_config_from(_DEFAULT_CONFIG_PATH)


def load_config_from(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config from an arbitrary path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
