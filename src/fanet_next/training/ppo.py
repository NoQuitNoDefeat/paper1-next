"""Micro-step PPO update.

The actor loss uses only records with an action (micro transitions); the
critic regresses every micro state including the action-free boundary state
of each cycle.  Probabilities and values are recomputed from stored graph
inputs and action prefixes with the current parameters (``model.runner.replay``).
"""

from __future__ import annotations

import numpy as np
import torch

from ..model.dual_graph import SchedulingModel
from ..model.runner import replay
from ..registry import slot
from .records import CycleRecord

TRAINER = slot("trainer", "policy optimisation algorithm")


def _pad(arrs: list[np.ndarray], width: int) -> np.ndarray:
    out = np.zeros((len(arrs), max(width, 1)), dtype=np.float32)
    for i, a in enumerate(arrs):
        out[i, : len(a)] = a
    return out[:, :width] if width else out[:, :0]


@TRAINER.register("ppo", role="primary")
class PPO:
    """Clipped PPO over micro transitions with time-varying discount GAE."""

    def __init__(self, model: SchedulingModel, *, lr: float = 3e-4, clip: float = 0.2,
                 epochs: int = 4, minibatch_cycles: int = 256, ent_coef: float = 0.01,
                 vf_coef: float = 0.5, max_grad_norm: float = 0.5, value_clip: float | None = None,
                 target_kl: float | None = None, adv_norm: bool = True, ratio: str = "step",
                 seed: int = 0, device: str = "cpu"):
        if ratio not in ("step", "sequence"):
            raise ValueError(f"unknown ratio {ratio!r}")
        # "step": one probability ratio and clip per micro action (the method's default);
        # "sequence": one ratio per cycle, the product over the ordered selections (the
        # probability of choosing these links in this order), clipped as a whole and paired
        # with the cycle's first advantage - use with cycle-level credit (E17 arm A2)
        self.ratio = ratio
        self.model = model
        self.clip, self.epochs, self.mb = clip, epochs, minibatch_cycles
        self.ent_coef, self.vf_coef, self.max_grad_norm = ent_coef, vf_coef, max_grad_norm
        self.value_clip, self.target_kl, self.adv_norm = value_clip, target_kl, adv_norm
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.rng = np.random.default_rng(seed)

    def set_lr(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def update(self, records: list[CycleRecord]) -> dict[str, float]:
        acts = np.concatenate([r.advantages[: r.num_actions] for r in records])
        mean, std = (acts.mean(), acts.std() + 1e-8) if len(acts) and self.adv_norm else (0.0, 1.0)
        stats: dict[str, list[float]] = {k: [] for k in
                                         ("policy_loss", "value_loss", "entropy", "approx_kl",
                                          "clip_frac", "grad_norm")}
        self.model.train()
        stop = False
        for _ in range(self.epochs):
            order = self.rng.permutation(len(records))
            for start in range(0, len(records), self.mb):
                mb = [records[i] for i in order[start:start + self.mb]]
                st = self._step(mb, mean, std)
                for k, v in st.items():
                    stats[k].append(v)
                if self.target_kl is not None and st["approx_kl"] > 1.5 * self.target_kl:
                    stop = True
                    break
            if stop:
                break
        out = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
        out.update(self._explained_variance(records))
        out["actions"] = int(len(acts))
        out["cycles"] = len(records)
        return out

    def _step(self, mb: list[CycleRecord], adv_mean: float, adv_std: float) -> dict[str, float]:
        dev = self.device
        rp = replay(self.model, [r.micro for r in mb], device=dev)
        k_max = rp.logp.shape[1]
        adv = torch.as_tensor(_pad([(r.advantages[: r.num_actions] - adv_mean) / adv_std for r in mb], k_max), device=dev)
        old_logp = torch.as_tensor(_pad([r.micro.logp for r in mb], k_max), device=dev)
        ret = torch.as_tensor(_pad([r.returns for r in mb], k_max + 1), device=dev)
        old_v = torch.as_tensor(_pad([r.micro.values for r in mb], k_max + 1), device=dev)
        am, sm = rp.act_mask, rp.state_mask
        n_act = am.sum().clamp(min=1)

        log_ratio = torch.where(am, rp.logp - old_logp, torch.zeros_like(old_logp))
        if self.ratio == "sequence":
            has = am.any(1)
            log_ratio = log_ratio.sum(1, keepdim=True)  # (B, 1): ordered-sequence log ratio
            ratio = log_ratio.exp()
            a = adv[:, :1]
            surr = torch.min(ratio * a, ratio.clamp(1 - self.clip, 1 + self.clip) * a)
            # divided by the number of actions, as in step mode: at ratio 1 the gradient equals
            # the step-mode gradient with a shared advantage, so the two differ only in clipping
            # (not in the actor's update strength relative to the critic's)
            policy_loss = -(surr * has.unsqueeze(1)).sum() / n_act
            am = has.unsqueeze(1)  # statistics below: one ratio per cycle
        else:
            ratio = log_ratio.exp()
            surr = torch.min(ratio * adv, ratio.clamp(1 - self.clip, 1 + self.clip) * adv)
            policy_loss = -(surr * am).sum() / n_act
        entropy = (rp.entropy * rp.act_mask).sum() / rp.act_mask.sum().clamp(min=1)

        v = rp.values
        v_err = (v - ret) ** 2
        if self.value_clip is not None:
            v_clip = old_v + (v - old_v).clamp(-self.value_clip, self.value_clip)
            v_err = torch.max(v_err, (v_clip - ret) ** 2)
        value_loss = 0.5 * (v_err * sm).sum() / sm.sum().clamp(min=1)

        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        with torch.no_grad():
            approx_kl = (((ratio - 1) - log_ratio) * am).sum() / n_act
            clip_frac = (((ratio - 1).abs() > self.clip) & am).sum() / n_act
        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                "entropy": entropy.item(), "approx_kl": approx_kl.item(),
                "clip_frac": clip_frac.item(), "grad_norm": float(grad_norm)}

    @staticmethod
    def _explained_variance(records: list[CycleRecord]) -> dict[str, float]:
        v = np.concatenate([r.micro.values for r in records])
        ret = np.concatenate([r.returns for r in records])
        var = ret.var()
        # boundary states only (where the reward enters)
        vb = np.array([r.micro.values[-1] for r in records])
        rb = np.array([r.returns[-1] for r in records])
        return {"explained_var": float(1 - (ret - v).var() / var) if var > 0 else 0.0,
                "explained_var_boundary": float(1 - (rb - vb).var() / rb.var()) if rb.var() > 0 else 0.0,
                "value_mean": float(v.mean()), "return_mean": float(ret.mean())}

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict(), "rng": self.rng.bit_generator.state}

    def load_state_dict(self, s: dict) -> None:
        self.optimizer.load_state_dict(s["optimizer"])
        self.rng.bit_generator.state = s["rng"]
