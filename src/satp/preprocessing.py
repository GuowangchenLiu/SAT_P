"""Preprocessing utilities for daily SAT-P inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration matching the predictor definitions in the manuscript."""

    temperature_window_days: int = 8
    discharge_increase_days: int = 2
    temperature_unit: str = "K"


def _minmax(series: pd.Series) -> pd.Series:
    lower = series.min(skipna=True)
    upper = series.max(skipna=True)
    span = upper - lower
    if not np.isfinite(span) or span == 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - lower) / span


def prepare_daily_inputs(
    data: pd.DataFrame,
    config: PreprocessingConfig | None = None,
) -> pd.DataFrame:
    """Create the thermal, flushing, and exhaustion predictors.

    Required columns are ``date``, ``temperature``, ``precipitation``, and
    ``discharge``. An optional ``TP`` column is preserved. The public workflow
    uses the cumulative fraction of annual precipitation as the current
    exhaustion proxy, consistent with the supplied research implementation.
    """

    config = config or PreprocessingConfig()
    required = {"date", "temperature", "precipitation", "discharge"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    if config.temperature_window_days < 1 or config.discharge_increase_days < 1:
        raise ValueError("window lengths must be positive integers")

    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date").reset_index(drop=True)

    unit = config.temperature_unit.upper()
    if unit == "K":
        frame["temperature_c"] = frame["temperature"].astype(float) - 273.15
    elif unit in {"C", "°C"}:
        frame["temperature_c"] = frame["temperature"].astype(float)
    else:
        raise ValueError("temperature_unit must be 'K' or 'C'")

    window = config.temperature_window_days
    frame["temperature_8d_mean"] = frame["temperature_c"].rolling(
        window=window,
        min_periods=1,
    ).mean()
    frame["temperature_8d_norm"] = _minmax(frame["temperature_8d_mean"])

    lag = config.discharge_increase_days
    discharge_change = frame["discharge"].astype(float).diff(periods=lag)
    frame["discharge_increase_2d"] = discharge_change.clip(lower=0).fillna(0.0)
    frame["discharge_increase_2d_norm"] = _minmax(frame["discharge_increase_2d"])

    frame["precipitation"] = frame["precipitation"].astype(float).bfill().fillna(0.0)
    frame["year"] = frame["date"].dt.year
    frame["cumulative_precipitation"] = frame.groupby("year")["precipitation"].cumsum()
    annual_total = frame.groupby("year")["precipitation"].transform("sum")
    frame["annual_precipitation"] = annual_total
    frame["exhaustion_index"] = np.where(
        annual_total > 0,
        frame["cumulative_precipitation"] / annual_total,
        0.0,
    )
    return frame

