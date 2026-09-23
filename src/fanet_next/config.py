"""TOML experiment configs with ``extends`` inheritance and dotted overrides."""

from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path
from typing import Any

from .registry import ConfigError


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and "type" not in value:
            out[key] = deep_merge(out[key], value)
        elif isinstance(value, dict) and isinstance(out.get(key), dict) and value.get("type") == out[key].get("type"):
            out[key] = deep_merge(out[key], value)
        else:
            # A different ``type`` replaces the whole section so that stale
            # parameters of the previous implementation do not leak through.
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    parent = raw.pop("extends", None)
    if parent is not None:
        base = load_config(path.parent / parent)
        raw = deep_merge(base, raw)
    for item in overrides or []:
        apply_override(raw, item)
    return raw


def parse_value(text: str) -> Any:
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def apply_override(cfg: dict, item: str) -> None:
    """Apply ``a.b.c=value`` (value parsed as TOML, falling back to a string)."""
    if "=" not in item:
        raise ConfigError(f"override must look like key.path=value, got {item!r}")
    key, text = item.split("=", 1)
    parts = key.strip().split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {item!r}: {part!r} is not a table")
    value = parse_value(text.strip())
    if parts[-1] == "type" and node.get("type") != value:
        node.clear()
    node[parts[-1]] = value


def dump_json(cfg: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(cfg, indent=2, ensure_ascii=False, sort_keys=True))
