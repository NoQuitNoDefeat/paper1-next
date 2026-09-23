"""训练: rollout collection, time-varying-discount GAE, PPO, checkpoints."""

from .advantages import compute_stream_advantages, gae
from .records import CycleRecord

__all__ = ["CycleRecord", "compute_stream_advantages", "gae"]
