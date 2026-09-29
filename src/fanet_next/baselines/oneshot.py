"""The common DRL formulation of link activation: one independent decision per link.

Same dual-graph model, observation, reward, imitation warm start and PPO settings as
the primary method; only the action formulation differs.  Per cycle the model is
evaluated once at the empty-set state: every individually feasible candidate gets a
Bernoulli(sigmoid(logit)) transmit decision; the decisions are
repaired by the controller in descending logit order (a link that has become
infeasible is dropped).  No work-conserving fill: links decided "off" stay off.
PPO treats the decision vector as one action per cycle (log-prob = sum of the
Bernoulli log-probs; entropy bonus = mean Bernoulli entropy over the candidates).
Evaluation samples the trained stochastic policy (seeded): its per-link marginals
are mostly below 1/2, so the per-link mode would usually schedule nothing
(``threshold_eval=True`` gives that variant).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from ..model.dual_graph import SchedulingModel
from ..observation.graph import batch_graphs
from ..policy.base import POLICY, DecisionOutput, Policy
from ..scheduling.controller import CANDIDATE_FEATURE_DIM, MICRO_FEATURE_DIM
from ..training.collector import SeedStream
from ..training.normalize import ReturnScaler
from .common import EnvStream, Learner, gae


def _state0(inputs, c_max: int):
    b_n = len(inputs)
    mask = np.zeros((b_n, c_max), dtype=bool)
    micro = np.zeros((b_n, MICRO_FEATURE_DIM), dtype=np.float32)
    dyn = np.zeros((b_n, c_max, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    for b, inp in enumerate(inputs):
        m = inp.controller.mask
        mask[b, : len(m)] = m
        micro[b] = inp.controller.micro_features()
        dyn[b, : len(m)] = inp.controller.candidate_features()
    return mask, micro, dyn


def _forward(model: SchedulingModel, graphs, mask, micro, dyn):
    enc = model.encode(batch_graphs(graphs, "cpu"))
    state = model.init_state(enc)
    mask_t, dyn_t = torch.as_tensor(mask), torch.as_tensor(dyn)
    logits = model.logits(enc, state, dyn_t)
    value = model.value(enc, state, mask_t, torch.as_tensor(micro), dyn_t)
    return logits, value, mask_t


def bernoulli_logp(logits, actions, mask):
    lp = actions * F.logsigmoid(logits) + (1 - actions) * F.logsigmoid(-logits)
    return (lp * mask).sum(-1)


def bernoulli_entropy(logits, mask):
    p = torch.sigmoid(logits)
    ent = -(p * F.logsigmoid(logits) + (1 - p) * F.logsigmoid(-logits))
    return (ent * mask).sum(-1) / mask.sum(-1).clamp(min=1)


@POLICY.register("oneshot_ppo", role="baseline")
class OneShotPolicy(Policy):
    """Independent per-link transmit decisions (Bernoulli) + feasibility repair; dual-graph model."""

    needs_graph = True
    learnable = True
    maximal_plans = False
    stochastic_eval = True

    def __init__(self, model: SchedulingModel, device: str = "cpu", seed: int = 0,
                 threshold_eval: bool = False):
        self.model = model
        self.generator = torch.Generator().manual_seed(seed)
        self.threshold_eval = bool(threshold_eval)
        self.stochastic_eval = not self.threshold_eval

    def seed(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    @property
    def net(self):
        return self.model

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state)

    @torch.no_grad()
    def act(self, inputs, *, mode="sample"):
        c = [inp.problem.num_candidates for inp in inputs]
        c_max = max(max(c, default=0), 1)
        mask, micro, dyn = _state0(inputs, c_max)
        was = self.model.training
        self.model.eval()
        logits, value, mask_t = _forward(self.model, [inp.graph for inp in inputs], mask, micro, dyn)
        self.model.train(was)
        if mode == "greedy" and self.threshold_eval:
            a = (logits > 0).float()
        else:
            a = torch.bernoulli(torch.sigmoid(logits), generator=self.generator)
        a = a * mask_t
        logp = bernoulli_logp(logits, a, mask_t.float())
        outs = []
        for b, inp in enumerate(inputs):
            ctl = inp.controller
            lg = logits[b, : c[b]].numpy()
            for i in sorted(np.nonzero(a[b, : c[b]].numpy() > 0)[0], key=lambda i: -lg[i]):
                if ctl.mask[i]:
                    ctl.step(int(i))
            outs.append(DecisionOutput(actions=list(ctl.selected), record={
                "graph": inp.graph, "mask": mask[b, : c[b]], "micro": micro[b],
                "dyn": dyn[b, : c[b]], "a": a[b, : c[b]].numpy(), "logp": float(logp[b]),
                "value": float(value[b])}))
        return outs

    @torch.no_grad()
    def values_of(self, inputs) -> np.ndarray:
        c_max = max(max((inp.problem.num_candidates for inp in inputs), default=0), 1)
        mask, micro, dyn = _state0(inputs, c_max)
        return _forward(self.model, [inp.graph for inp in inputs], mask, micro, dyn)[1].numpy()


def _batch(records):
    c_max = max(max(len(r["mask"]) for r in records), 1)
    b_n = len(records)
    mask = np.zeros((b_n, c_max), dtype=bool)
    dyn = np.zeros((b_n, c_max, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    act = np.zeros((b_n, c_max), dtype=np.float32)
    for b, r in enumerate(records):
        c = len(r["mask"])
        mask[b, :c], dyn[b, :c], act[b, :c] = r["mask"], r["dyn"], r["a"]
    micro = np.stack([r["micro"] for r in records]).astype(np.float32)
    return [r["graph"] for r in records], mask, micro, dyn, torch.as_tensor(act)


class OneShotLearner(Learner):
    """Imitation warm start (Bernoulli likelihood of the teacher's set + critic regression),
    then clipped PPO over one decision vector per cycle, as the primary method is trained."""

    def __init__(self, policy: OneShotPolicy, stream: EnvStream, *, rollout_cycles: int = 128,
                 gamma: float = 0.99, gae_lambda: float = 0.95, lr: float = 1e-4, clip: float = 0.2,
                 epochs: int = 4, minibatch_cycles: int = 256, ent_coef: float = 0.003,
                 vf_coef: float = 0.5, max_grad_norm: float = 0.5, reward_norm: bool = True,
                 imitation: dict | None = None, seed: int = 0):
        self.policy, self.stream = policy, stream
        self.model = policy.model
        self.rollout_cycles = int(rollout_cycles)
        self.gamma, self.lam = gamma, gae_lambda
        self.clip, self.epochs, self.mb = clip, int(epochs), int(minibatch_cycles)
        self.ent_coef, self.vf_coef, self.max_grad = ent_coef, vf_coef, max_grad_norm
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.scaler = ReturnScaler(len(stream.envs), gamma, enabled=reward_norm)
        self.imitation = imitation
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    # ---------------------------------------------------------------- imitation
    def setup(self, run) -> dict:
        if not self.imitation:
            return {}
        from ..experiment.assemble import build_env, build_policy
        icfg = {"teacher": "longest_queue", "cycles": 500, "epochs": 8, "lr": 1e-3,
                "minibatch": 256, **self.imitation}
        teacher = build_policy(run.cfg, icfg["teacher"], seed=self.seed)
        envs = [build_env(run.cfg, run_id=f"imitate-env{e}") for e in range(len(self.stream.envs))]
        stream = EnvStream(envs, SeedStream(run.seed_base + 50_000))
        recs, rewards, ends = [[] for _ in envs], [[] for _ in envs], [[] for _ in envs]
        for _ in range(int(icfg["cycles"])):
            inputs = stream.inputs
            c_max = max(max(inp.problem.num_candidates for inp in inputs), 1)
            mask, micro, dyn = _state0(inputs, c_max)
            outs = teacher.act(inputs, mode="greedy")
            for e, (inp, out) in enumerate(zip(inputs, outs)):
                c = inp.problem.num_candidates
                a = np.zeros(c, dtype=np.float32)
                a[out.actions] = 1.0
                recs[e].append({"graph": inp.graph, "mask": mask[e, :c], "micro": micro[e],
                                "dyn": dyn[e, :c], "a": a})
                tr, _ = stream.step(e, out.actions)
                rewards[e].append(tr.reward.total)
                ends[e].append(tr.end.value)
        records, returns = [], []
        for e in range(len(envs)):
            ends[e][-1] = "cut"
            z = np.zeros(len(rewards[e]))
            _, ret = gae(np.array(rewards[e]), z, z, ends[e], self.gamma, 1.0)
            records += recs[e]
            returns += list(ret)
        returns = np.array(returns)
        scale = float(returns.std()) if self.scaler.enabled and returns.std() > 0 else 1.0
        opt = torch.optim.Adam(self.model.parameters(), lr=float(icfg["lr"]), eps=1e-5)
        nll = 0.0
        for _ in range(int(icfg["epochs"])):
            order = self.rng.permutation(len(records))
            nll, n = 0.0, 0
            for start in range(0, len(records), int(icfg["minibatch"])):
                idx = order[start:start + int(icfg["minibatch"])]
                graphs, mask, micro, dyn, act = _batch([records[i] for i in idx])
                logits, value, mask_t = _forward(self.model, graphs, mask, micro, dyn)
                m = mask_t.float()
                loss_pi = -(bernoulli_logp(logits, act, m) / m.sum(-1).clamp(min=1)).mean()
                ret = torch.as_tensor(returns[idx] / scale, dtype=torch.float32)
                loss = loss_pi + 0.5 * self.vf_coef * ((value - ret) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                nll += loss_pi.item() * len(idx)
                n += len(idx)
            nll /= max(n, 1)
        if self.scaler.enabled:  # PPO starts with the return scale the critic was fitted to
            self.scaler.rms.var, self.scaler.rms.count = scale ** 2, 1e3
        return {"event": "imitation", "teacher": icfg["teacher"], "records": len(records),
                "nll_per_link": nll, "return_scale": scale}

    # --------------------------------------------------------------------- PPO
    def iteration(self, it: int, total: int) -> dict:
        n_env = len(self.stream.envs)
        streams = [[] for _ in range(n_env)]
        episodes = []
        for _ in range(self.rollout_cycles):
            inputs = self.stream.inputs
            outs = self.policy.act(inputs, mode="sample")
            for e, out in enumerate(outs):
                tr, summary = self.stream.step(e, out.actions)
                rec = dict(out.record)
                end = tr.end.value
                rec["reward"] = self.scaler(e, tr.reward.total, end != "continue")
                rec["raw"] = tr.reward.total
                rec["end"] = end
                if end == "truncated":
                    rec["next_value"] = float(self.policy.values_of([tr.next_input])[0])
                streams[e].append(rec)
                if summary:
                    episodes.append(summary)
        last = self.policy.values_of(self.stream.inputs)
        records = []
        for e, s in enumerate(streams):
            s[-1]["end"] = s[-1]["end"] if s[-1]["end"] != "continue" else "cut"
            values = np.array([r["value"] for r in s])
            nxt = np.array([r.get("next_value", values[t + 1] if t + 1 < len(s) else last[e])
                            for t, r in enumerate(s)])
            if s[-1]["end"] == "cut":
                nxt[-1] = last[e]
            adv, ret = gae(np.array([r["reward"] for r in s]), values, nxt,
                           [r["end"] for r in s], self.gamma, self.lam)
            for r, a, g in zip(s, adv, ret):
                r["adv"], r["ret"] = a, g
            records += s
        stats = self._update(records)
        stats.update(cycles=len(records), episodes=episodes,
                     reward_raw_mean=float(np.mean([r["raw"] for r in records])))
        return stats

    def _update(self, records) -> dict:
        adv_all = np.array([r["adv"] for r in records])
        mean, std = adv_all.mean(), adv_all.std() + 1e-8
        out = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "n": 0}
        for _ in range(self.epochs):
            order = self.rng.permutation(len(records))
            for start in range(0, len(records), self.mb):
                mb = [records[i] for i in order[start:start + self.mb]]
                graphs, mask, micro, dyn, act = _batch(mb)
                logits, value, mask_t = _forward(self.model, graphs, mask, micro, dyn)
                m = mask_t.float()
                logp = bernoulli_logp(logits, act, m)
                old = torch.as_tensor([r["logp"] for r in mb], dtype=torch.float32)
                adv = torch.as_tensor([(r["adv"] - mean) / std for r in mb], dtype=torch.float32)
                ret = torch.as_tensor([r["ret"] for r in mb], dtype=torch.float32)
                ratio = torch.exp(logp - old)
                pl = -torch.min(ratio * adv, ratio.clamp(1 - self.clip, 1 + self.clip) * adv).mean()
                vl = 0.5 * ((value - ret) ** 2).mean()
                ent = bernoulli_entropy(logits, m).mean()
                loss = pl + self.vf_coef * vl - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad)
                self.opt.step()
                out["policy_loss"] += pl.item()
                out["value_loss"] += vl.item()
                out["entropy"] += ent.item()
                out["approx_kl"] += (old - logp).mean().item()
                out["n"] += 1
        n = max(out.pop("n"), 1)
        return {k: v / n for k, v in out.items()}

    def state_dict(self) -> dict:
        return {"opt": self.opt.state_dict(), "scaler": self.scaler.state_dict()}
