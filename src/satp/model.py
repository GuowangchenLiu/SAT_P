"""Core SAT-P and discharge-only rating-curve equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SATPParameters:
    """Eight fitted parameters used by the SAT-P equation."""

    a1: float
    a2: float
    a3: float
    b1: float
    b2: float
    b3: float
    b4: float
    b5: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "SATPParameters":
        return cls(**{name: float(values[name]) for name in cls.__dataclass_fields__})

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


def _as_float_array(values: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def exhaustion_weight(exhaustion_index: np.ndarray | list[float]) -> np.ndarray:
    """Return the negative logistic exhaustion weight used by SAT-P."""

    index = np.clip(_as_float_array(exhaustion_index), -700.0, 700.0)
    return -1.0 / (1.0 + np.exp(-index))


def predict_satp(
    discharge: np.ndarray | list[float],
    normalized_temperature: np.ndarray | list[float],
    normalized_discharge_increase: np.ndarray | list[float],
    exhaustion_index: np.ndarray | list[float],
    parameters: SATPParameters,
    minimum_tpc: float | None = 0.005,
) -> np.ndarray:
    """Predict daily total phosphorus concentration in mg/L.

    All input arrays must be broadcast-compatible. Discharge must be positive.
    The temperature and discharge-increase predictors are normally scaled to
    the interval [0, 1] by :func:`satp.preprocessing.prepare_daily_inputs`.
    """

    q = _as_float_array(discharge)
    t = _as_float_array(normalized_temperature)
    qi = _as_float_array(normalized_discharge_increase)
    gei = _as_float_array(exhaustion_index)

    if np.any(q <= 0):
        raise ValueError("discharge must be strictly positive")

    p = parameters
    a = exhaustion_weight(gei)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        tpc = (
            a * p.a1 * np.power(q, p.b1 * t + p.b2 + 1.0)
            + a * (p.a2 * qi + p.b4) * np.power(q, p.b3 + 1.0)
            + p.a3 * np.power(q, p.b5 + 1.0)
        )

    tpc = np.asarray(tpc, dtype=float)
    tpc[~np.isfinite(tpc)] = np.nan
    if minimum_tpc is not None:
        tpc = np.maximum(tpc, float(minimum_tpc))
    return tpc


def predict_raw(
    discharge: np.ndarray | list[float],
    a1: float,
    b1: float,
    minimum_tpc: float | None = None,
) -> np.ndarray:
    """Predict TPC with the discharge-only rating curve ``a1 * Q**b1``."""

    q = _as_float_array(discharge)
    if np.any(q <= 0):
        raise ValueError("discharge must be strictly positive")
    prediction = float(a1) * np.power(q, float(b1))
    if minimum_tpc is not None:
        prediction = np.maximum(prediction, float(minimum_tpc))
    return prediction

