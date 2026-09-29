"""Shared pieces of the learned baselines: environment streams, GAE, graph batching.

A learned baseline is a registered policy that owns its network (``net``) plus a
``Learner`` that trains it by the method of its original paper.  Checkpoints
store the policy spec and ``net`` weights, so evaluation and checkpoint
selection rebuild it like any other policy (``experiment.assemble``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from ..contracts import EndType
from ..policy.base import Policy
from ..training.collector import SeedStream


class LearnedBaseline(Policy):
    """A baseline policy with its own network; ``learnable`` so evaluation uses greedy mode."""

    learnable = True
    builds_own_model = True
    net: torch.nn.Module

    def state_dict(self) -> dict:
        return self.net.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.net.load_state_dict(state)


class Learner(ABC):
    """Trains a :class:`LearnedBaseline` for one iteration of ``rollout_cycles`` per env."""

    def setup(self, run) -> dict:
        """Before the first iteration (e.g. an imitation warm start); returns a log entry."""
        return {}

    @abstractmethod
    def iteration(self, it: int, total: int) -> dict:
        """Collect experience, update the network; return stats with 'cycles' and 'episodes'."""

    def state_dict(self) -> dict:
        return {}


class EnvStream:
    """Environments stepped continuously; finished episodes restart from a seed stream."""

    def __init__(self, envs, seeds: SeedStream):
        self.envs, self.seeds = envs, seeds
        self.counter = 0
        for e in range(len(envs)):
            self._start(e)

    def _start(self, e: int) -> None:
        self.envs[e].reset(self.seeds.next(), episode=self.counter)
        self.counter += 1

    @property
    def inputs(self):
        return [env.current for env in self.envs]

    def step(self, e: int, actions: list[int]):
        """Step env ``e``; returns (transition, episode summary or None)."""
        tr = self.envs[e].step(actions)
        summary = None
        if tr.end is not EndType.CONTINUE:
            summary = self.envs[e].metrics.summary()
            self._start(e)
        return tr, summary


def gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, ends: list[str],
        gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """GAE over one stream of single-transition cycles.  ``ends``: 'continue' | 'truncated'
    (bootstrap next_values, stop the recursion) | 'terminated' (no bootstrap) | 'cut'
    (slice end: bootstrap, stop)."""
    adv = np.zeros(len(rewards))
    last = 0.0
    for t in reversed(range(len(rewards))):
        if ends[t] == "terminated":
            delta, last = rewards[t] - values[t], 0.0
        elif ends[t] in ("truncated", "cut"):
            delta, last = rewards[t] + gamma * next_values[t] - values[t], 0.0
        else:
            delta = rewards[t] + gamma * next_values[t] - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def pad_stack(mats: list[np.ndarray], size: int, fill=0.0) -> np.ndarray:
    """Stack (c_i, ...) arrays into (B, size, ...) with padding."""
    shape = (len(mats), size) + mats[0].shape[1:]
    out = np.full(shape, fill, dtype=np.float32)
    for b, m in enumerate(mats):
        out[b, : len(m)] = m
    return out
