"""Build components from a config and check that they fit together.

Config sections map one-to-one to slots; see ``configs/base.toml``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# importing the packages registers every implementation
from .. import backend as _backend  # noqa: F401
from .. import baselines as _baselines  # noqa: F401
from .. import model as _model  # noqa: F401
from .. import observation as _observation  # noqa: F401
from .. import policy as _policy  # noqa: F401
from .. import reward as _reward  # noqa: F401
from .. import scenario as _scenario  # noqa: F401
from .. import scheduling as _scheduling  # noqa: F401
from ..backend import BACKEND
from ..loop import SchedulingEnv
from ..model import MODEL
from ..observation import OBSERVATION
from ..observation.graph import FeatureSchema
from ..policy import POLICY, Policy
from ..registry import SLOTS, ConfigError
from ..reward.standard import REWARD
from ..scenario import SCENARIO
from ..scheduling import CANDIDATES, ConstraintSet
from ..training import ppo as _ppo  # noqa: F401  (register trainer)

TRAIN_SEED_BASE = 1_000_000
DEV_SEED_BASE = 10_000_000
TEST_SEED_BASE = 20_000_000


def _spec(cfg: dict, key: str, default):
    value = cfg.get(key, default)
    return {"type": value} if isinstance(value, str) else value


def switch_backend(cfg: dict, backend_type: str) -> dict:
    """Same environment, other executor: change the backend ``type`` and keep its
    parameters (routing, channel, queue policy).  A plain ``--set backend.type=...``
    clears the section instead, which would silently fall back to default routing.
    Parameters the target backend does not accept fail at build time."""
    out = dict(cfg)
    out["backend"] = {**_spec(cfg, "backend", "lightweight"), "type": backend_type}
    return out


def build_env(cfg: dict, *, run_id: str = "run", build_graph: bool = True,
              scenario_override: dict | None = None) -> SchedulingEnv:
    scenario = scenario_override if scenario_override is not None else cfg["scenario"]
    return SchedulingEnv(
        scenario_source=SCENARIO.build(scenario),
        backend=BACKEND.build(_spec(cfg, "backend", "lightweight")),
        candidates=CANDIDATES.build(_spec(cfg, "candidates", "standard")),
        constraints=ConstraintSet.from_config(cfg.get("constraints", {})),
        observation=OBSERVATION.build(_spec(cfg, "observation", "standard")),
        reward=REWARD.build(_spec(cfg, "reward", "standard")),
        build_graph=build_graph, run_id=run_id)


def feature_schema(cfg: dict) -> FeatureSchema:
    return OBSERVATION.build(_spec(cfg, "observation", "standard")).schema


def build_model(cfg: dict, schema: FeatureSchema | None = None):
    return MODEL.build(_spec(cfg, "model", "dual_graph"), schema=schema or feature_schema(cfg))


def build_policy(cfg: dict, spec: dict | str | None = None, model=None, seed: int = 0) -> Policy:
    spec = _spec({"p": spec}, "p", None) if spec is not None else _spec(cfg, "policy", "ppo")
    entry = POLICY.entry(spec["type"])
    if getattr(entry.factory, "learnable", False) and not getattr(entry.factory, "builds_own_model", False):
        return POLICY.build(spec, model=model if model is not None else build_model(cfg), seed=seed)
    params = dict(spec)
    if "seed" in __import__("inspect").signature(entry.factory).parameters:
        params.setdefault("seed", seed)
    return POLICY.build(params)


def policy_from_checkpoint(cfg: dict, ckpt: dict) -> Policy:
    """The learned policy stored in a checkpoint: the primary dual-graph PPO policy, or a
    learned baseline (``policy_spec`` + its network weights)."""
    spec = ckpt.get("policy_spec")
    if spec is None:
        model = build_model(cfg, feature_schema(cfg))
        model.load_state_dict(ckpt["model"])
        return build_policy(cfg, "ppo", model=model)
    policy = build_policy(cfg, spec)
    policy.load_state_dict(ckpt["model"])
    return policy


def check_compatibility(env: SchedulingEnv, policy: Policy) -> dict:
    """Fail early on mismatched components; return the component manifest."""
    if policy.needs_graph and not env.build_graph:
        raise ConfigError("policy needs the dual graph but the environment does not build it")
    model = getattr(policy, "model", None)
    if model is not None:
        schema = getattr(model, "schema", None)
        if schema is not None and schema.dims() != env.observation.schema.dims():
            raise ConfigError(f"model expects features {schema.dims()} but observation provides "
                              f"{env.observation.schema.dims()}")
    return manifest(env, policy)


def _role(obj) -> str:
    return f"{getattr(type(obj), 'registered_name', type(obj).__name__)}({getattr(type(obj), 'registered_role', '?')})"


def manifest(env: SchedulingEnv, policy: Policy) -> dict:
    """Which implementation (and role) fills each slot in this run."""
    cs = env.constraints
    m = {
        "scenario": _role(env.scenario_source), "backend": _role(env.backend),
        "channel": _role(env.backend.channel) if hasattr(env.backend, "channel") else "-",
        "routing": _role(env.backend.routing) if hasattr(env.backend, "routing") else "-",
        "candidates": _role(env.candidates), "resource": _role(cs.resource),
        "interference": _role(cs.interference), "observation": _role(env.observation),
        "reward": _role(env.reward), "policy": _role(policy),
        "constraints_primary": cs.is_primary,
    }
    model = getattr(policy, "model", None)
    if model is not None:
        m["model"] = _role(model)
        for attr, key in (("comm", "comm_encoder"), ("lift", "lift"), ("inter", "interaction_encoder")):
            m[key] = _role(getattr(model.actor_trunk, attr))
        m["set_summary"] = _role(model.summary_a)
        m["actor_head"], m["critic_head"] = _role(model.actor), _role(model.critic)
        m["model_parameters"] = int(sum(p.numel() for p in model.parameters()))
    return m


def source_hash(root: Path | None = None) -> str:
    """sha256 over every source file and config (identifies code without a commit)."""
    import hashlib

    root = root or Path(__file__).resolve().parents[3]
    h = hashlib.sha256()
    files = sorted(list((root / "src").rglob("*.py")) + list((root / "configs").rglob("*.toml")))
    for f in files:
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def code_version(root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parents[3]
    src = source_hash(root)
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "configs"],
                                    cwd=root, capture_output=True, text=True).stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        head, dirty = "unknown", True
    return {"commit": head or "none", "dirty": dirty, "source_sha256_16": src}


def available() -> dict[str, list[str]]:
    return {name: s.names() for name, s in SLOTS.items()}
