"""Mixture of scenario distributions: per episode one component, picked from the seed.

Used to widen a training distribution without dropping the original one (E14):
every component is the shared parameters overridden by its own entries, e.g.

    [scenario]
    type = "mixture"
    base_type = "random"
    weights = [0.5, 0.5]
    components = [{}, {speed_mps = [0.0, 5.0], area_m = [150.0, 300.0]}]
    num_nodes = 16        # shared parameters (radio, horizon, ...) as for base_type
    ...

The component is drawn from a stream of its own (seed, tag), and the component then
builds the episode from the same seed, so an episode of component i is exactly the
episode that component i alone would give for that seed (trace splits included).
Evaluation-only changes (drain: ``horizon``, ``traffic_stop_cycle``) arrive as shared
parameters and reach every component.
"""

from __future__ import annotations

import numpy as np

from ..config import deep_merge
from .base import SCENARIO, Scenario

PICK_TAG = 0x6D6978  # "mix"


@SCENARIO.register("mixture", role="variant")
class MixtureScenarioSource:
    """Per episode one of several scenario distributions (fixed weights), e.g. a widened training set."""

    def __init__(self, base_type: str, components: list[dict], weights: list[float] | None = None,
                 **shared):
        if not components:
            raise ValueError("mixture: no components")
        w = np.ones(len(components)) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (len(components),) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("mixture: one non-negative weight per component, not all zero")
        self.cum = np.cumsum(w / w.sum())
        self.specs = [deep_merge({"type": base_type, **shared}, dict(c)) for c in components]
        self.components = [SCENARIO.build(s) for s in self.specs]
        if len({c.radio for c in self.components}) != 1:
            raise ValueError("mixture: all components must share one radio")
        self.radio = self.components[0].radio

    def component_of(self, seed: int) -> int:
        u = np.random.default_rng([int(seed), PICK_TAG]).random()
        return min(int(np.searchsorted(self.cum, u, side="right")), len(self.components) - 1)

    def make(self, seed: int) -> Scenario:
        return self.components[self.component_of(seed)].make(seed)
