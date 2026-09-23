"""Reward scaling by the running std of the discounted return (learning signal only)."""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    def __init__(self) -> None:
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        b_mean, b_var, b_n = x.mean(), x.var(), x.size
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, s: dict) -> None:
        self.mean, self.var, self.count = s["mean"], s["var"], s["count"]


class ReturnScaler:
    """r / std(discounted return), tracked per environment stream."""

    def __init__(self, num_envs: int, gamma: float, enabled: bool = True, eps: float = 1e-8,
                 clip: float = 10.0):
        self.enabled = enabled
        self.gamma, self.eps, self.clip = gamma, eps, clip
        self.ret = np.zeros(num_envs)
        self.rms = RunningMeanStd()

    def __call__(self, env: int, reward: float, episode_end: bool) -> float:
        if not self.enabled:
            return reward
        self.ret[env] = self.ret[env] * self.gamma + reward
        self.rms.update(np.array([self.ret[env]]))
        if episode_end:
            self.ret[env] = 0.0
        return float(np.clip(reward / np.sqrt(self.rms.var + self.eps), -self.clip, self.clip))

    @property
    def scale(self) -> float:
        return float(np.sqrt(self.rms.var + self.eps)) if self.enabled else 1.0

    def state_dict(self) -> dict:
        return {"ret": self.ret.copy(), "rms": self.rms.state_dict()}

    def load_state_dict(self, s: dict) -> None:
        self.ret = np.array(s["ret"])
        self.rms.load_state_dict(s["rms"])
