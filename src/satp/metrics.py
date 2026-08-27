"""Performance metrics used in SAT-P calibration and evaluation."""

from __future__ import annotations

import numpy as np


def _paired(observed, simulated) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observed, dtype=float)
    sim = np.asarray(simulated, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    return obs[mask], sim[mask]


def squared_correlation(observed, simulated) -> float:
    """Return squared Pearson correlation, the R² definition used here."""

    obs, sim = _paired(observed, simulated)
    if len(obs) < 2 or np.std(obs) < 1e-12 or np.std(sim) < 1e-12:
        return float("nan")
    return float(np.corrcoef(obs, sim)[0, 1] ** 2)


def nash_sutcliffe_efficiency(observed, simulated) -> float:
    """Return the Nash–Sutcliffe model efficiency coefficient."""

    obs, sim = _paired(observed, simulated)
    if len(obs) < 2:
        return float("nan")
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    if denominator < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((obs - sim) ** 2) / denominator)

