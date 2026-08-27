#!/usr/bin/env python3
"""Calibrate SAT-P with NSGA-III using R² and NSE as objectives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satp import (  # noqa: E402
    PreprocessingConfig,
    nash_sutcliffe_efficiency,
    prepare_daily_inputs,
    squared_correlation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "demo" / "daily_inputs_2022_2024.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "calibration")
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--population", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--minimum-tpc", type=float, default=0.005)
    return parser.parse_args()


def predict_population(q, temperature, q_increase, gei, candidates):
    a1, a2, a3, b1, b2, b3, b4, b5 = [candidates[:, i, None] for i in range(8)]
    q = q[None, :]
    temperature = temperature[None, :]
    q_increase = q_increase[None, :]
    gei = gei[None, :]
    exhaustion = -1.0 / (1.0 + np.exp(-gei))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        predictions = (
            exhaustion * a1 * q ** (b1 * temperature + b2 + 1.0)
            + exhaustion * (a2 * q_increase + b4) * q ** (b3 + 1.0)
            + a3 * q ** (b5 + 1.0)
        )
    return predictions


def main() -> None:
    args = parse_args()
    try:
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize
        from pymoo.util.ref_dirs import get_reference_directions
    except ImportError as exc:
        raise SystemExit('Install calibration support with: python -m pip install -e ".[calibration]"') from exc

    prepared = prepare_daily_inputs(pd.read_csv(args.input), PreprocessingConfig())
    mask = (
        prepared["discharge"].notna()
        & (prepared["discharge"] > 0)
        & prepared["TP"].notna()
        & (prepared["TP"] > args.minimum_tpc)
    )
    calibration = prepared.loc[mask]
    q = calibration["discharge"].to_numpy(float)
    temperature = calibration["temperature_8d_norm"].to_numpy(float)
    q_increase = calibration["discharge_increase_2d_norm"].to_numpy(float)
    gei = calibration["exhaustion_index"].to_numpy(float)
    observed = calibration["TP"].to_numpy(float)

    class CalibrationProblem(Problem):
        def __init__(self):
            super().__init__(n_var=8, n_obj=2, n_constr=0, xl=[-50.0] * 8, xu=[50.0] * 8)

        def _evaluate(self, candidates, out, *unused_args, **unused_kwargs):
            predictions = predict_population(q, temperature, q_increase, gei, candidates)
            predictions = np.maximum(predictions, args.minimum_tpc)
            r2 = np.array([squared_correlation(observed, row) for row in predictions])
            nse = np.array([nash_sutcliffe_efficiency(observed, row) for row in predictions])
            r2 = np.nan_to_num(r2, nan=0.0, posinf=0.0, neginf=0.0)
            nse = np.nan_to_num(nse, nan=-100.0, posinf=-100.0, neginf=-100.0)
            out["F"] = -np.column_stack([r2, nse])

    reference_directions = get_reference_directions("das-dennis", 2, n_partitions=12)
    algorithm = NSGA3(
        pop_size=args.population,
        ref_dirs=reference_directions,
        eliminate_duplicates=True,
    )
    result = minimize(
        CalibrationProblem(),
        algorithm,
        ("n_gen", args.generations),
        seed=args.seed,
        verbose=True,
    )

    scores = -result.F
    compromise_index = int(np.nanargmax(scores[:, 0] + scores[:, 1]))
    parameter_names = ["a1", "a2", "a3", "b1", "b2", "b3", "b4", "b5"]
    best = {name: float(value) for name, value in zip(parameter_names, result.X[compromise_index])}
    summary = {
        "input": args.input.name,
        "n_observations": int(len(observed)),
        "generations": args.generations,
        "population": args.population,
        "seed": args.seed,
        "r2_squared_correlation": float(scores[compromise_index, 0]),
        "nse": float(scores[compromise_index, 1]),
        "parameters": best,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibrated_parameters.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    pareto = pd.DataFrame(result.X, columns=parameter_names)
    pareto["r2_squared_correlation"] = scores[:, 0]
    pareto["nse"] = scores[:, 1]
    pareto.to_csv(args.output_dir / "pareto_front.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
