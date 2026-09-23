"""Shared radio physics: path gain, SINR of a transmission set, unit helpers.

Used by the lightweight backend (with true gains) and by the scheduler's
feasibility checks (with decision-time estimates).  Gains are linear and
directed: ``gain[s, r]`` is the power gain from transmitter ``s`` to receiver ``r``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def db_to_lin(db):
    return 10.0 ** (np.asarray(db, dtype=float) / 10.0)


def lin_to_db(x, floor: float = 1e-30):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), floor))


def dbm_to_w(dbm):
    return db_to_lin(dbm) * 1e-3


@dataclass(frozen=True)
class RadioParams:
    """Single-channel, fixed-power, single-rate radio (起点假设)."""

    tx_power_dbm: float = 20.0  # 0.1 W
    noise_dbm: float = -80.0  # 1e-11 W
    pathloss_ref_db: float = 40.05  # loss at 1 m (~ free space at 2.4 GHz)
    pathloss_exponent: float = 2.5
    min_distance_m: float = 1.0
    sinr_threshold_db: float = 4.771212547196624  # linear 3
    rate_bps: float = 2e6
    service_window_s: float = 0.02  # data window per cycle (<= cycle length)

    @property
    def tx_power_w(self) -> float:
        return float(dbm_to_w(self.tx_power_dbm))

    @property
    def noise_w(self) -> float:
        return float(dbm_to_w(self.noise_dbm))

    @property
    def threshold(self) -> float:
        return float(db_to_lin(self.sinr_threshold_db))

    @property
    def service_bytes(self) -> int:
        """Bytes one scheduled link can carry in one service window."""
        return int(self.rate_bps * self.service_window_s // 8)

    def range_m(self) -> float:
        """Distance at which the interference-free SNR equals the threshold."""
        budget = self.tx_power_dbm - self.noise_dbm - self.sinr_threshold_db - self.pathloss_ref_db
        return float(10.0 ** (budget / (10.0 * self.pathloss_exponent)))


def path_gain(positions: np.ndarray, params: RadioParams) -> np.ndarray:
    """Deterministic log-distance gain matrix, zero on the diagonal."""
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.maximum(np.linalg.norm(diff, axis=-1), params.min_distance_m)
    loss_db = params.pathloss_ref_db + 10.0 * params.pathloss_exponent * np.log10(dist)
    gain = db_to_lin(-loss_db)
    np.fill_diagonal(gain, 0.0)
    return gain


def set_sinr(links: np.ndarray, gain: np.ndarray, power: np.ndarray, noise: float) -> np.ndarray:
    """SINR of every link in a concurrent set (cumulative interference).

    ``SINR_l = P_s G[s_l, r_l] / (N0 + sum_{m != l} P_{s_m} G[s_m, r_l])``.
    """
    links = np.asarray(links, dtype=int).reshape(-1, 2)
    if len(links) == 0:
        return np.zeros(0)
    tx, rx = links[:, 0], links[:, 1]
    # cross[m, l] = power from transmitter of m received at receiver of l
    cross = power[tx][:, None] * gain[tx][:, rx]
    signal = np.diag(cross).copy()
    interference = cross.sum(axis=0) - signal
    return signal / (noise + interference)
