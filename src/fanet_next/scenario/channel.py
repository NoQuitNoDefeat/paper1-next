"""Channel models: decision-time estimate vs. execution-time truth.

The scheduler only ever sees ``estimate``; the backend decides success with
``execution``.  ``ideal`` makes them identical, so a plan that is feasible
under the full-SINR check always succeeds.  ``lognormal`` adds estimation
error, small-scale fading at execution and intra-cycle mobility, so that
"feasible" and "succeeded" can differ.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..physics import db_to_lin, path_gain, RadioParams
from ..registry import slot

CHANNEL = slot("channel", "decision-time channel estimate and execution-time channel")


class ChannelModel(ABC):
    @abstractmethod
    def estimate(self, positions: np.ndarray, radio: RadioParams, rng: np.random.Generator) -> np.ndarray:
        """Ghat at the sample time from reported positions."""

    @abstractmethod
    def execution(self, positions: np.ndarray, velocities: np.ndarray, radio: RadioParams,
                  rng: np.random.Generator) -> np.ndarray:
        """True gain used to decide success of the executed set."""

    def true_gain(self, positions: np.ndarray, radio: RadioParams) -> np.ndarray:
        """Noise-free gain used for routing / physical availability."""
        return path_gain(positions, radio)


@CHANNEL.register("ideal", role="primary")
class IdealChannel(ChannelModel):
    """Exact, frozen CSI: execution uses the same gains as the decision."""

    def estimate(self, positions, radio, rng):
        return path_gain(positions, radio)

    def execution(self, positions, velocities, radio, rng):
        return path_gain(positions, radio)


@CHANNEL.register("lognormal")
class LognormalChannel(ChannelModel):
    """Estimation error + execution fading (both log-normal, dB std) + intra-cycle motion."""

    def __init__(self, estimate_error_db: float = 1.0, fading_db: float = 2.0,
                 move_during_window: bool = True):
        self.estimate_error_db = float(estimate_error_db)
        self.fading_db = float(fading_db)
        self.move_during_window = bool(move_during_window)

    def _perturb(self, gain: np.ndarray, std_db: float, rng: np.random.Generator) -> np.ndarray:
        if std_db <= 0:
            return gain
        out = gain * db_to_lin(rng.normal(0.0, std_db, size=gain.shape))
        np.fill_diagonal(out, 0.0)
        return out

    def estimate(self, positions, radio, rng):
        return self._perturb(path_gain(positions, radio), self.estimate_error_db, rng)

    def execution(self, positions, velocities, radio, rng):
        if self.move_during_window:
            positions = positions + velocities * (radio.service_window_s / 2.0)
        return self._perturb(path_gain(positions, radio), self.fading_db, rng)
