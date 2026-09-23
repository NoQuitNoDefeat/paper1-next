"""Checkpoints for resuming training within this version.

Saved: model, optimizer and PPO minibatch RNG, progress counters, config,
reward scaler, training seed stream, torch / numpy / python / sampling RNG
states and — when the backend supports it — the full environment objects, so
running episodes continue exactly.  Resume is exact at iteration boundaries
on CPU with the same code and config.  Checkpoints from other versions or with
changed model / feature configs are not supported.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

FORMAT = 1


def rng_state(policy_generator: torch.Generator) -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(), "policy": policy_generator.get_state()}


def set_rng_state(state: dict, policy_generator: torch.Generator) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    policy_generator.set_state(state["policy"])


def save_checkpoint(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format": FORMAT, **payload}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("format") != FORMAT:
        raise ValueError(f"unsupported checkpoint format {data.get('format')} (expected {FORMAT})")
    return data
