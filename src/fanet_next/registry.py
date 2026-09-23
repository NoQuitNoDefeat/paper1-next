"""Component slots and config-driven construction.

Every replaceable responsibility is a *slot* holding a small name -> factory table.
A config section selects an implementation with ``type`` and passes the remaining
keys as constructor keyword arguments.  There is no dynamic discovery: an
implementation becomes available by being imported and decorated with
``SLOT.register("name")``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


class ConfigError(ValueError):
    """A config section cannot be turned into a component."""


@dataclass
class Entry:
    name: str
    factory: Callable[..., Any]
    role: str  # "primary", "baseline", "control" or "variant"
    doc: str


@dataclass
class Slot:
    """A replaceable responsibility, e.g. ``feasibility`` or ``comm_encoder``."""

    name: str
    doc: str
    entries: dict[str, Entry] = field(default_factory=dict)

    def register(self, name: str, *, role: str = "variant") -> Callable:
        if role not in {"primary", "baseline", "control", "variant"}:
            raise ValueError(f"unknown role {role!r}")

        def deco(factory: Callable) -> Callable:
            if name in self.entries:
                raise ValueError(f"{self.name}: duplicate implementation {name!r}")
            doc = inspect.getdoc(factory) or ""
            self.entries[name] = Entry(name, factory, role, doc.split("\n", 1)[0])
            factory.registered_name = name  # type: ignore[attr-defined]
            factory.registered_role = role  # type: ignore[attr-defined]
            return factory

        return deco

    def names(self) -> list[str]:
        return sorted(self.entries)

    def entry(self, name: str) -> Entry:
        try:
            return self.entries[name]
        except KeyError:
            raise ConfigError(
                f"{self.name}: unknown implementation {name!r}; available: {self.names()}"
            ) from None

    def build(self, spec: dict[str, Any] | str, **deps: Any) -> Any:
        """Build from ``{"type": name, **params}``; ``deps`` are injected objects."""
        if isinstance(spec, str):
            spec = {"type": spec}
        if "type" not in spec:
            raise ConfigError(f"{self.name}: config section needs a 'type' key, got {spec}")
        params = {k: v for k, v in spec.items() if k != "type"}
        entry = self.entry(spec["type"])
        sig = inspect.signature(entry.factory)
        accepts_any = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
        unknown = [k for k in params if k not in sig.parameters and not accepts_any]
        if unknown:
            raise ConfigError(
                f"{self.name}.{entry.name}: unknown parameters {unknown}; "
                f"accepted: {[p for p in sig.parameters if p not in deps]}"
            )
        clash = set(params) & set(deps)
        if clash:
            raise ConfigError(f"{self.name}.{entry.name}: {sorted(clash)} are injected, not configurable")
        wanted_deps = {k: v for k, v in deps.items() if k in sig.parameters or accepts_any}
        return entry.factory(**params, **wanted_deps)


SLOTS: dict[str, Slot] = {}


def slot(name: str, doc: str) -> Slot:
    if name in SLOTS:
        return SLOTS[name]
    SLOTS[name] = Slot(name, doc)
    return SLOTS[name]


def describe() -> str:
    """Human-readable table of every slot and its implementations."""
    lines = []
    for s in SLOTS.values():
        lines.append(f"[{s.name}] {s.doc}")
        for e in sorted(s.entries.values(), key=lambda e: e.name):
            lines.append(f"    {e.name:<24} {e.role:<9} {e.doc}")
    return "\n".join(lines)
