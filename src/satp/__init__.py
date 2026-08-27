"""Public API for the SAT-P model."""

from .metrics import nash_sutcliffe_efficiency, squared_correlation
from .model import SATPParameters, predict_raw, predict_satp
from .preprocessing import PreprocessingConfig, prepare_daily_inputs

__all__ = [
    "SATPParameters",
    "PreprocessingConfig",
    "nash_sutcliffe_efficiency",
    "predict_raw",
    "predict_satp",
    "prepare_daily_inputs",
    "squared_correlation",
]

__version__ = "0.1.0"

