"""GRLinQ-style iterative graph RL (Shan et al., arXiv 2408.09394 / ISIT 2024), adapted.

Faithful parts: an interference graph with K = 10 strongest incoming interferers per
link; message-passing GNN policy and value networks (edge-update MLP, message =
edge feature (.) sender feature, sum aggregation, node MLP on [own, aggregate]; sum
readout for the value); an MDP inside each scheduling slot: every link starts
"pending" and at each of at most T = 32 iterations each pending link chooses active /
inactive / pending (pending links left at T become inactive); reward = change of the
objective minus 1/T per iteration; trained by PPO.  Node features follow the paper's
model-driven design: SNR margin, strongest / total interference received and caused
(FlashLinQ / ITLinQ-style quantities), the MDP state (one-hot) and t / T.
Adapted: channel gains instead of distances (the paper's CSI variant); objective =
queue-weighted successful links (weights = next-hop queue lengths, the paper's
weighted sum rate with a single rate: a link succeeds iff it meets its SINR threshold
under all active links and shares no endpoint with another active link), normalised
by the total weight; edges also join links sharing an endpoint (half duplex, absent
in D2D); the final active set is replayed through the full-SINR controller
(infeasible links dropped, no fill).  Evaluation follows the paper's use of the RL
policy's randomness ("the same test can be conducted in parallel"): ``eval_samples``
seeded rollouts per slot, the one with the best objective is executed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..policy.base import POLICY, DecisionOutput
from ..policy.classical import queue_weights
from ..scheduling.maxweight import node_conflicts
from .common import EnvStream, Learner, LearnedBaseline

ACTIVE, INACTIVE, PENDING = 0, 1, 2
NODE_STATIC, NODE_DIM, EDGE_DIM = 6, 10, 2


def slot_graph(inp, k: int = 10):
    """Static node features (C, 6), edges (E, 2) as (sender j, receiver i), edge features (E, 2),
    weights w (C,), and the pieces needed to score an active set."""
    p = inp.problem
    c = p.num_candidates
    w = queue_weights(inp)
    inr = p.cross / p.noise  # [j, i]: j's transmitter at i's receiver
    off = inr.copy()
    np.fill_diagonal(off, 0.0)
    share = node_conflicts(p)
    feats = np.zeros((c, NODE_STATIC), dtype=np.float32)
    if c:
        feats[:, 0] = np.log10(np.maximum(p.signal / (p.threshold * p.noise), 1e-12)) / 2
        feats[:, 1] = np.log10(1 + off.max(axis=0)) / 4  # strongest received
        feats[:, 2] = np.log10(1 + off.max(axis=1)) / 4  # strongest caused
        feats[:, 3] = np.log10(1 + off.sum(axis=0)) / 4  # total received
        feats[:, 4] = w / max(w.max(), 1e-9)
        feats[:, 5] = np.log1p(w) / 4
    edges = set()
    for i in range(c):
        others = [j for j in np.argsort(-off[:, i]) if j != i][:k]
        edges.update((int(j), i) for j in others if off[j, i] > 0)
        edges.update((int(j), i) for j in np.nonzero(share[:, i])[0])
    e = np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)
    ef = np.zeros((len(e), EDGE_DIM), dtype=np.float32)
    if len(e):
        ef[:, 0] = np.log10(1 + off[e[:, 0], e[:, 1]]) / 4
        ef[:, 1] = share[e[:, 0], e[:, 1]]
    return {"x": feats, "edges": e, "ef": ef, "w": w}


def objective(inp, w: np.ndarray, active: np.ndarray) -> float:
    """Queue-weighted share of successful active links (half duplex + cumulative SINR)."""
    p, idx = inp.problem, np.nonzero(active)[0]
    total = w.sum()
    if len(idx) == 0 or total <= 0:
        return 0.0
    nodes = p.links[idx].ravel()
    counts = np.bincount(nodes, minlength=p.num_nodes)
    hd_ok = (counts[p.links[idx, 0]] == 1) & (counts[p.links[idx, 1]] == 1)
    interf = p.cross[np.ix_(idx, idx)].sum(axis=0) - p.signal[idx]
    sinr_ok = p.signal[idx] >= p.threshold * (p.noise + interf)
    return float(w[idx][hd_ok & sinr_ok].sum() / total)


class MPGNN(nn.Module):
    def __init__(self, out_dim: int, hidden: int = 64, layers: int = 3, readout: bool = False):
        super().__init__()
        self.inp = nn.Linear(NODE_DIM, hidden)
        self.edge = nn.ModuleList(nn.Sequential(nn.Linear(EDGE_DIM, hidden), nn.ReLU(),
                                                nn.Linear(hidden, hidden)) for _ in range(layers))
        self.node = nn.ModuleList(nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                                nn.Linear(hidden, hidden), nn.ReLU())
                                  for _ in range(layers))
        self.out = nn.Linear(hidden, out_dim)
        self.readout = readout

    def forward(self, x, edges, ef, graph_of=None, num_graphs: int = 0):
        h = torch.relu(self.inp(x))
        for fe, fn in zip(self.edge, self.node):
            m = fe(ef) * h[edges[:, 0]]
            agg = torch.zeros_like(h).index_add_(0, edges[:, 1], m)
            h = fn(torch.cat([h, agg], dim=-1))
        if not self.readout:
            return self.out(h)
        pooled = torch.zeros(num_graphs, h.shape[1]).index_add_(0, graph_of, h)
        return self.out(pooled).squeeze(-1)


def _batch(graphs, states, ts, horizon):
    """Concatenate per-slot graphs with the dynamic features of their MDP states."""
    xs, es, efs, gof, off = [], [], [], [], 0
    for b, (g, s, t) in enumerate(zip(graphs, states, ts)):
        c = len(g["x"])
        dyn = np.zeros((c, 4), dtype=np.float32)
        dyn[np.arange(c), s] = 1.0
        dyn[:, 3] = t / horizon
        xs.append(np.concatenate([g["x"], dyn], axis=1))
        es.append(g["edges"] + off)
        efs.append(g["ef"])
        gof.append(np.full(c, b))
        off += c
    return (torch.as_tensor(np.concatenate(xs)), torch.as_tensor(np.concatenate(es)).long(),
            torch.as_tensor(np.concatenate(efs)), torch.as_tensor(np.concatenate(gof)).long())


class GRLinQNet(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.pi = MPGNN(3, hidden, layers)
        self.v = MPGNN(1, hidden, layers, readout=True)


@POLICY.register("grlinq", role="baseline")
class GRLinQPolicy(LearnedBaseline):
    """Iterative active / inactive / pending decisions per link by a GNN (GRLinQ-style)."""

    maximal_plans = False
    stochastic_eval = True

    def __init__(self, hidden: int = 64, layers: int = 3, k_nearest: int = 10, horizon: int = 32,
                 eval_samples: int = 8, seed: int = 0):
        self.net = GRLinQNet(hidden, layers)
        self.k, self.horizon = int(k_nearest), int(horizon)
        self.eval_samples = int(eval_samples)
        self.generator = torch.Generator().manual_seed(seed)

    def seed(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    @torch.no_grad()
    def rollout(self, inputs, mode: str):
        """Run the slot MDP for every input; returns final states and per-step transitions."""
        graphs = [slot_graph(inp, self.k) for inp in inputs]
        states = [np.full(len(g["x"]), PENDING, dtype=np.int64) for g in graphs]
        objs = [0.0] * len(inputs)
        when = [np.full(len(g["x"]), self.horizon, dtype=np.int64) for g in graphs]
        steps = [[] for _ in inputs]
        live = [b for b, s in enumerate(states) if len(s)]
        t = 0
        while live and t < self.horizon:
            x, e, ef, gof = _batch([graphs[b] for b in live], [states[b] for b in live],
                                   [t] * len(live), self.horizon)
            logits = self.net.pi(x, e, ef)
            value = self.net.v(x, e, ef, gof, len(live))
            if mode == "greedy":
                a = logits.argmax(-1)
            else:
                a = torch.multinomial(torch.softmax(logits, -1), 1, generator=self.generator).squeeze(1)
            logp_all = torch.log_softmax(logits, -1)
            lp = logp_all.gather(1, a[:, None]).squeeze(1)
            pos = 0
            nxt = []
            for i, b in enumerate(live):
                c = len(states[b])
                ab, lpb = a[pos:pos + c].numpy(), lp[pos:pos + c].numpy()
                pos += c
                pend = states[b] == PENDING
                new = states[b].copy()
                new[pend] = ab[pend]
                if t == self.horizon - 1:
                    new[new == PENDING] = INACTIVE
                when[b][(states[b] == PENDING) & (new == ACTIVE)] = t
                obj = objective(inputs[b], graphs[b]["w"], new == ACTIVE)
                done = not (new == PENDING).any()
                steps[b].append({"state": states[b].copy(), "t": t, "action": np.where(pend, ab, -1),
                                 "logp": float(lpb[pend].sum()), "value": float(value[i]),
                                 "reward": obj - objs[b] - 1.0 / self.horizon, "done": done})
                states[b], objs[b] = new, obj
                if not done:
                    nxt.append(b)
            live, t = nxt, t + 1
        return graphs, states, when, steps

    def act(self, inputs, *, mode="sample"):
        graphs, states, when, steps = self.rollout(inputs, "sample")
        if mode == "greedy":  # best of several sampled rollouts (evaluation)
            best = [objective(inp, g["w"], s == ACTIVE) for inp, g, s in zip(inputs, graphs, states)]
            for _ in range(self.eval_samples - 1):
                _, st2, wh2, sp2 = self.rollout(inputs, "sample")
                for b, inp in enumerate(inputs):
                    obj = objective(inp, graphs[b]["w"], st2[b] == ACTIVE)
                    if obj > best[b]:
                        best[b], states[b], when[b], steps[b] = obj, st2[b], wh2[b], sp2[b]
        outs = []
        for b, inp in enumerate(inputs):
            ctl = inp.controller
            w = graphs[b]["w"]
            for i in sorted(np.nonzero(states[b] == ACTIVE)[0], key=lambda i: (when[b][i], -w[i])):
                if ctl.mask[i]:
                    ctl.step(int(i))
            outs.append(DecisionOutput(actions=list(ctl.selected),
                                       record={"graph": graphs[b], "steps": steps[b]}))
        return outs


class GRLinQLearner(Learner):
    """PPO over the slot-MDP transitions (each slot is one finite episode, gamma = 1)."""

    def __init__(self, policy: GRLinQPolicy, stream: EnvStream, *, rollout_cycles: int = 128,
                 gamma: float = 1.0, gae_lambda: float = 0.95, lr: float = 3e-4, clip: float = 0.2,
                 epochs: int = 4, minibatch: int = 512, ent_coef: float = 0.01, vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5, seed: int = 0):
        self.policy, self.stream = policy, stream
        self.net = policy.net
        self.rollout_cycles = int(rollout_cycles)
        self.gamma, self.lam = gamma, gae_lambda
        self.clip, self.epochs, self.mb = clip, int(epochs), int(minibatch)
        self.ent_coef, self.vf_coef, self.max_grad = ent_coef, vf_coef, max_grad_norm
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.rng = np.random.default_rng(seed)

    def iteration(self, it: int, total: int) -> dict:
        trans, episodes, finals = [], [], []
        for _ in range(self.rollout_cycles):
            inputs = self.stream.inputs
            outs = self.policy.act(inputs, mode="sample")
            for e, out in enumerate(outs):
                steps = out.record["steps"]
                if steps:  # finite slot episode: V(terminal) = 0
                    nxt_v = [s1["value"] for s1 in steps[1:]] + [0.0]
                    last = 0.0
                    for s, v1 in zip(reversed(steps), reversed(nxt_v)):
                        v1 = 0.0 if s["done"] else v1
                        delta = s["reward"] + self.gamma * v1 - s["value"]
                        last = delta + (0.0 if s["done"] else self.gamma * self.lam * last)
                        s["adv"], s["ret"], s["graph"] = last, last + s["value"], out.record["graph"]
                    trans += steps
                    finals.append(sum(s["reward"] for s in steps) + len(steps) / self.policy.horizon)
                _, summary = self.stream.step(e, out.actions)
                if summary:
                    episodes.append(summary)
        stats = self._update(trans) if trans else {}
        stats.update(cycles=self.rollout_cycles * len(self.stream.envs), episodes=episodes,
                     transitions=len(trans), objective_mean=float(np.mean(finals)) if finals else 0.0,
                     mdp_steps_mean=len(trans) / max(len(finals), 1))
        return stats

    def _update(self, trans) -> dict:
        adv_all = np.array([s["adv"] for s in trans])
        mean, std = adv_all.mean(), adv_all.std() + 1e-8
        out = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "n": 0}
        h = self.policy.horizon
        for _ in range(self.epochs):
            order = self.rng.permutation(len(trans))
            for start in range(0, len(trans), self.mb):
                mb = [trans[i] for i in order[start:start + self.mb]]
                x, e, ef, gof = _batch([s["graph"] for s in mb], [s["state"] for s in mb],
                                       [s["t"] for s in mb], h)
                logits = self.net.pi(x, e, ef)
                value = self.net.v(x, e, ef, gof, len(mb))
                act = torch.as_tensor(np.concatenate([s["action"] for s in mb]))
                pend = act >= 0
                logp_all = torch.log_softmax(logits, -1)
                lp_node = logp_all.gather(1, act.clamp(min=0)[:, None]).squeeze(1) * pend
                logp = torch.zeros(len(mb)).index_add_(0, gof, lp_node)
                ent_node = -(logp_all.exp() * logp_all).sum(-1) * pend
                n_pend = torch.zeros(len(mb)).index_add_(0, gof, pend.float()).clamp(min=1)
                ent = (torch.zeros(len(mb)).index_add_(0, gof, ent_node) / n_pend).mean()
                old = torch.as_tensor([s["logp"] for s in mb], dtype=torch.float32)
                adv = torch.as_tensor([(s["adv"] - mean) / std for s in mb], dtype=torch.float32)
                ret = torch.as_tensor([s["ret"] for s in mb], dtype=torch.float32)
                ratio = torch.exp(logp - old)
                pl = -torch.min(ratio * adv, ratio.clamp(1 - self.clip, 1 + self.clip) * adv).mean()
                vl = 0.5 * ((value - ret) ** 2).mean()
                loss = pl + self.vf_coef * vl - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad)
                self.opt.step()
                out["policy_loss"] += pl.item()
                out["value_loss"] += vl.item()
                out["entropy"] += ent.item()
                out["approx_kl"] += (old - logp).mean().item()
                out["n"] += 1
        n = max(out.pop("n"), 1)
        return {k: v / n for k, v in out.items()}

    def state_dict(self) -> dict:
        return {"opt": self.opt.state_dict()}
