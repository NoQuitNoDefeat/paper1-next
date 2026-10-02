"""Channel models: decision-time estimate vs. execution-time truth.

The scheduler only ever sees ``estimate``; the backend decides success with
``execution``.  ``ideal`` makes them identical, so a plan that is feasible
under the full-SINR check always succeeds.  ``lognormal`` adds estimation
error, small-scale fading at execution and intra-cycle mobility, so that
"feasible" and "succeeded" can differ.  ``rician`` is the air-to-air channel
with measured parameters (decisions.md §10): the scheduler knows the mean
gain, execution adds Rician block fading, and the scheduler plans with a fade
margin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import ncx2

from ..physics import db_to_lin, path_gain, RadioParams
from ..registry import slot

CHANNEL = slot("channel", "decision-time channel estimate and execution-time channel")


class ChannelModel(ABC):
    # dB added to the decoding threshold for planning (Report.threshold)
    fade_margin_db: float = 0.0
    # True when ``estimate`` draws no randomness: a backend that precomputes the whole
    # episode (ns-3) can then reproduce the observed gains exactly
    deterministic_estimate: bool = True

    def reset(self, num_nodes: int, rng: np.random.Generator) -> None:
        """Start of an episode, before any other call (draw per-episode state here)."""

    @abstractmethod
    def estimate(self, positions: np.ndarray, radio: RadioParams, rng: np.random.Generator) -> np.ndarray:
        """Ghat at the sample time from reported positions."""

    @abstractmethod
    def execution(self, positions: np.ndarray, velocities: np.ndarray, radio: RadioParams,
                  rng: np.random.Generator) -> np.ndarray:
        """True gain used to decide success of the executed set."""

    def execution_at(self, positions: np.ndarray, radio: RadioParams,
                     rng: np.random.Generator) -> np.ndarray:
        """Execution gain with the nodes at ``positions``, for a backend that moves the
        nodes itself (ns-3); draws the same randomness, in the same order, as ``execution``."""
        return self.execution(positions, np.zeros_like(positions), radio, rng)

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

    deterministic_estimate = False

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


def rician_power(k_factor: float, num_nodes: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-mean Rician power gains |h|^2, one draw per unordered node pair (reciprocal)."""
    iu = np.triu_indices(num_nodes, 1)
    m = len(iu[0])
    w = (rng.standard_normal(m) + 1j * rng.standard_normal(m)) / np.sqrt(2.0)
    h = np.sqrt(k_factor / (k_factor + 1.0)) + np.sqrt(1.0 / (k_factor + 1.0)) * w
    out = np.zeros((num_nodes, num_nodes))
    out[iu] = np.abs(h) ** 2
    return out + out.T


def rician_margin_db(k_factor_db: float, outage: float) -> float:
    """Fade margin (dB) at which an isolated link planned exactly at the margin fails with
    probability ``outage``: P(|h|^2 < 10^(-margin/10)) = outage.

    2(K+1)|h|^2 is noncentral chi-square with 2 degrees of freedom and noncentrality 2K.
    """
    k = float(db_to_lin(k_factor_db))
    x = ncx2.ppf(outage, 2, 2.0 * k) / (2.0 * (k + 1.0))
    return float(-10.0 * np.log10(x))


@CHANNEL.register("rician", role="variant")
class RicianChannel(ChannelModel):
    """Air-to-air, measured parameters: the scheduler sees the mean (path-loss) gain;
    execution multiplies it by reciprocal Rician block fading (one draw per node pair and
    cycle, unit mean) at mid-window positions; the scheduler plans against threshold x
    fade margin (margin from ``outage`` unless given).  Optional log-normal shadowing is
    fixed per node pair and episode and known to the scheduler (decisions.md §10); with
    ``shadowing_unit_power`` its linear mean is 1 (same mean received power, spread only)."""

    def __init__(self, k_factor_db: float = 10.0, outage: float = 0.1,
                 fade_margin_db: float | None = None, shadowing_db: float = 0.0,
                 shadowing_unit_power: bool = False, move_during_window: bool = True):
        if not 0.0 < outage < 1.0:
            raise ValueError("outage must lie in (0, 1)")
        self.k_factor_db = float(k_factor_db)
        self.k_factor = float(db_to_lin(k_factor_db))
        self.outage = float(outage)
        self.fade_margin_db = (rician_margin_db(self.k_factor_db, self.outage)
                               if fade_margin_db is None else float(fade_margin_db))
        self.shadowing_db = float(shadowing_db)
        # False: zero mean in dB (the usual fit convention; raises the linear mean gain by
        # sigma^2 / (2 xi) dB, xi = 10 / ln 10).  True: dB mean -sigma^2 / (2 xi), so the linear
        # mean gain stays 1 and only the spread is added (E15b)
        self.shadowing_unit_power = bool(shadowing_unit_power)
        self.move_during_window = bool(move_during_window)
        self._shadow: np.ndarray | None = None

    def reset(self, num_nodes, rng):
        self._shadow = None
        if self.shadowing_db > 0:
            # drawn from a spawned child stream: the episode's fading draws are the same
            # with and without shadowing (common random numbers for paired comparisons)
            child = rng.spawn(1)[0]
            iu = np.triu_indices(num_nodes, 1)
            s = np.ones((num_nodes, num_nodes))
            mean_db = -self.shadowing_db ** 2 * np.log(10.0) / 20.0 if self.shadowing_unit_power else 0.0
            s[iu] = db_to_lin(mean_db + child.normal(0.0, self.shadowing_db, size=len(iu[0])))
            self._shadow = np.triu(s, 1) + np.triu(s, 1).T

    def _mean(self, positions, radio):
        gain = path_gain(positions, radio)
        if self._shadow is not None:
            if self._shadow.shape != gain.shape:
                raise RuntimeError("rician channel: reset() was not called for this episode")
            gain = gain * self._shadow
        return gain

    def true_gain(self, positions, radio):
        return self._mean(positions, radio)

    def estimate(self, positions, radio, rng):
        return self._mean(positions, radio)

    def execution_at(self, positions, radio, rng):
        return self._mean(positions, radio) * rician_power(self.k_factor, len(positions), rng)

    def execution(self, positions, velocities, radio, rng):
        if self.move_during_window:
            positions = positions + velocities * (radio.service_window_s / 2.0)
        return self.execution_at(positions, radio, rng)
