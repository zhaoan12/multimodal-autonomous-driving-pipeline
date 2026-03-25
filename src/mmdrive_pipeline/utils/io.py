"""Configuration and JSON/YAML IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            try:
                import yaml
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    f"PyYAML is required to read {path}. Use JSON configs or install PyYAML."
                ) from exc
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}.")
    return data


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
