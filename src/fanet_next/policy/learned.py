"""Learned micro-action policy (dual-graph model driven by the shared runner)."""

from __future__ import annotations

import numpy as np
import torch

from ..model.dual_graph import SchedulingModel
from ..model.runner import initial_values, sample
from .base import POLICY, DecisionOutput, Policy


@POLICY.register("ppo", role="primary")
class LearnedPolicy(Policy):
    """Masked sequential sampling (training) or greedy argmax (evaluation), no STOP action."""

    needs_graph = True
    learnable = True

    def __init__(self, model: SchedulingModel, device: str = "cpu", seed: int = 0):
        self.model = model
        self.device = device
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def seed(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    def act(self, inputs, *, mode="sample"):
        was_training = self.model.training
        self.model.eval()  # no dropout/batch-norm in the default model; keeps modes consistent
        recs = sample(self.model, inputs, mode=mode, generator=self.generator, device=self.device)
        self.model.train(was_training)
        return [DecisionOutput(actions=r.actions.tolist(), record={"micro": r}) for r in recs]

    def values_of(self, inputs) -> np.ndarray:
        return initial_values(self.model, inputs, device=self.device)
