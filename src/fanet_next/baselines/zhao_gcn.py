"""Zhao et al., "Link Scheduling Using Graph Neural Networks", IEEE TWC 2023 (GCN-LGS).

Faithful parts: an L-layer GCN on the links' conflict graph,
``X_l = sigma(X_{l-1} Theta0 + L X_{l-1} Theta1)`` (L: normalized Laplacian, leaky ReLU,
linear scalar output), input feature = per-link utility, topology-aware utility
``w = z * u`` fed to a greedy MWIS solver; trained by their graph-based deterministic
policy gradient, grad J = gamma * grad Psi * v_hat with gamma = u(v_GCN) / u(v_greedy),
epsilon exploration with random z ~ N(1, 0.2) (their Random-LGS) decaying 1 -> 0.002,
Adam lr 1e-5 (their training script), batches of 200.
Adapted: utility = next-hop queue length (their q*r with a single rate); conflict edges =
shared endpoint or pairwise SINR failure (``pairwise_conflicts``); the greedy runs through
this project's full-SINR controller, so plans are feasible; L = 3 layers of width 32
(their script trains 1, 3 and 20 layers).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..policy.base import POLICY, DecisionOutput
from ..policy.classical import queue_weights
from ..scheduling.maxweight import greedy_complete, pairwise_conflicts
from .common import EnvStream, Learner, LearnedBaseline, pad_stack


def laplacian(conflict: np.ndarray) -> np.ndarray:
    """Normalized Laplacian I - D^-1/2 A D^-1/2 (isolated vertices: identity row)."""
    a = conflict.astype(np.float32)
    deg = a.sum(axis=1)
    inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0).astype(np.float32)
    return np.eye(len(a), dtype=np.float32) - inv[:, None] * a * inv[None, :]


class GCN(nn.Module):
    def __init__(self, layers: int = 3, hidden: int = 32):
        super().__init__()
        dims = [1] + [hidden] * (layers - 1) + [1]
        self.theta0 = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims, dims[1:]))
        self.theta1 = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims, dims[1:]))

    def forward(self, lap: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """lap (B, C, C), x (B, C, 1) -> z (B, C)."""
        for i, (t0, t1) in enumerate(zip(self.theta0, self.theta1)):
            x = t0(x) + t1(lap @ x)
            if i < len(self.theta0) - 1:
                x = nn.functional.leaky_relu(x)
        return x.squeeze(-1)


@POLICY.register("zhao_gcn", role="baseline")
class ZhaoGcnPolicy(LearnedBaseline):
    """GCN-guided greedy max-weight (Zhao et al., TWC 2023), adapted to full SINR."""

    def __init__(self, layers: int = 3, hidden: int = 32, seed: int = 0):
        self.net = GCN(layers, hidden)
        self.rng = np.random.default_rng(seed)
        self.epsilon = 0.0  # exploration probability, set by the learner while training

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def inputs_of(inp):
        u = queue_weights(inp)
        lap = laplacian(pairwise_conflicts(inp.problem))
        return u, lap

    @torch.no_grad()
    def scores(self, inputs) -> list[np.ndarray]:
        data = [self.inputs_of(inp) for inp in inputs]
        c_max = max((len(u) for u, _ in data), default=0)
        if c_max == 0:
            return [np.zeros(0) for _ in inputs]
        lap = np.zeros((len(data), c_max, c_max), dtype=np.float32)
        x = np.zeros((len(data), c_max, 1), dtype=np.float32)
        for b, (u, l) in enumerate(data):
            lap[b, : len(u), : len(u)] = l
            x[b, : len(u), 0] = u / max(u.max(), 1e-9) if len(u) else 0.0
        z = self.net(torch.as_tensor(lap), torch.as_tensor(x)).numpy()
        return [z[b, : len(u)] for b, (u, _) in enumerate(data)]

    def act(self, inputs, *, mode="sample"):
        zs = self.scores(inputs)
        outs = []
        for inp, z in zip(inputs, zs):
            explore = mode == "sample" and self.rng.random() < self.epsilon
            if explore:
                z = self.rng.normal(1.0, 0.2, size=len(z))
            w = z * queue_weights(inp)
            ctl = inp.controller
            while not ctl.done:
                idx = np.nonzero(ctl.mask)[0]
                ctl.step(int(idx[np.argmax(w[idx])]))
            outs.append(DecisionOutput(actions=list(ctl.selected), record={"explore": explore}))
        return outs


class ZhaoGcnLearner(Learner):
    def __init__(self, policy: ZhaoGcnPolicy, stream: EnvStream, *, rollout_cycles: int = 128,
                 lr: float = 1e-5, batch: int = 200, eps_start: float = 1.0, eps_end: float = 0.002,
                 seed: int = 0):
        self.policy, self.stream = policy, stream
        self.rollout_cycles, self.batch = int(rollout_cycles), int(batch)
        self.eps = (float(eps_start), float(eps_end))
        self.opt = torch.optim.Adam(policy.net.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)

    def iteration(self, it: int, total: int) -> dict:
        frac = it / max(total - 1, 1)
        self.policy.epsilon = self.eps[0] + frac * (self.eps[1] - self.eps[0])
        samples, episodes = [], []
        for _ in range(self.rollout_cycles):
            inputs = self.stream.inputs
            outs = self.policy.act(inputs, mode="sample")
            for e, (inp, out) in enumerate(zip(inputs, outs)):
                if inp.problem.num_candidates:
                    u, lap = ZhaoGcnPolicy.inputs_of(inp)
                    base = float(u[greedy_complete(inp.problem, u)].sum())
                    if base > 0:
                        sel = np.zeros(len(u), dtype=np.float32)
                        sel[out.actions] = 1.0
                        samples.append((lap, u / max(u.max(), 1e-9), sel,
                                        float(u[out.actions].sum()) / base))
                _, summary = self.stream.step(e, out.actions)
                if summary:
                    episodes.append(summary)
        losses = []
        order = self.rng.permutation(len(samples))
        for start in range(0, len(samples), self.batch):
            mb = [samples[i] for i in order[start:start + self.batch]]
            c_max = max(len(s[1]) for s in mb)
            lap = np.zeros((len(mb), c_max, c_max), dtype=np.float32)
            for b, s in enumerate(mb):
                lap[b, : len(s[1]), : len(s[1])] = s[0]
            x = torch.as_tensor(pad_stack([s[1][:, None] for s in mb], c_max))
            sel = torch.as_tensor(pad_stack([s[2] for s in mb], c_max))
            gamma = torch.as_tensor([s[3] for s in mb], dtype=torch.float32)
            z = self.policy.net(torch.as_tensor(lap), x)
            loss = -(gamma * (z * sel).sum(1)).mean()  # ascent on gamma * <z, v_hat>
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
        g = np.array([s[3] for s in samples]) if samples else np.zeros(1)
        return {"cycles": self.rollout_cycles * len(self.stream.envs), "episodes": episodes,
                "loss": float(np.mean(losses)) if losses else 0.0, "gamma_mean": float(g.mean()),
                "epsilon": self.policy.epsilon, "samples": len(samples)}

    def state_dict(self) -> dict:
        return {"opt": self.opt.state_dict()}
