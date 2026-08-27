#!/usr/bin/env python3
"""Run SAT-P and the discharge-only model on the partial demo dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satp import (  # noqa: E402
    PreprocessingConfig,
    SATPParameters,
    nash_sutcliffe_efficiency,
    predict_raw,
    predict_satp,
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
    parser.add_argument(
        "--parameters",
        type=Path,
        default=ROOT / "config" / "parameters.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--temperature-unit", choices=["K", "C"], default="K")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.input)
    settings = json.loads(args.parameters.read_text(encoding="utf-8"))
    minimum_tpc = float(settings["minimum_tpc_mg_l"])

    prepared = prepare_daily_inputs(
        raw,
        PreprocessingConfig(temperature_unit=args.temperature_unit),
    )
    satp_parameters = SATPParameters.from_mapping(settings["satp"])

    valid_q = prepared["discharge"].notna() & (prepared["discharge"] > 0)
    prepared["satp_tpc"] = np.nan
    prepared["raw_tpc"] = np.nan
    prepared.loc[valid_q, "satp_tpc"] = predict_satp(
        prepared.loc[valid_q, "discharge"].to_numpy(),
        prepared.loc[valid_q, "temperature_8d_norm"].to_numpy(),
        prepared.loc[valid_q, "discharge_increase_2d_norm"].to_numpy(),
        prepared.loc[valid_q, "exhaustion_index"].to_numpy(),
        satp_parameters,
        minimum_tpc=minimum_tpc,
    )
    prepared.loc[valid_q, "raw_tpc"] = predict_raw(
        prepared.loc[valid_q, "discharge"].to_numpy(),
        settings["raw"]["a1"],
        settings["raw"]["b1"],
        minimum_tpc=minimum_tpc,
    )

    observed = prepared["TP"].notna() & (prepared["TP"] > minimum_tpc) & valid_q
    metrics = {
        "dataset": args.input.name,
        "n_daily_records": int(len(prepared)),
        "n_observations_used": int(observed.sum()),
        "note": "Demo metrics use the partial public dataset and supplied example parameters.",
        "satp": {
            "r2_squared_correlation": squared_correlation(
                prepared.loc[observed, "TP"], prepared.loc[observed, "satp_tpc"]
            ),
            "nse": nash_sutcliffe_efficiency(
                prepared.loc[observed, "TP"], prepared.loc[observed, "satp_tpc"]
            ),
        },
        "raw": {
            "r2_squared_correlation": squared_correlation(
                prepared.loc[observed, "TP"], prepared.loc[observed, "raw_tpc"]
            ),
            "nse": nash_sutcliffe_efficiency(
                prepared.loc[observed, "TP"], prepared.loc[observed, "raw_tpc"]
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "date",
        "temperature_c",
        "temperature_8d_norm",
        "precipitation",
        "discharge",
        "discharge_increase_2d_norm",
        "exhaustion_index",
        "TP",
        "satp_tpc",
        "raw_tpc",
    ]
    prepared[output_columns].to_csv(args.output_dir / "demo_predictions.csv", index=False)
    (args.output_dir / "demo_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.scatter(
        prepared.loc[observed, "date"],
        prepared.loc[observed, "TP"],
        marker="*",
        s=30,
        color="#B9B9B9",
        label="Observed TPC",
        zorder=4,
    )
    ax.plot(prepared["date"], prepared["satp_tpc"], color="#BF812D", lw=1.8, label="SAT-P")
    ax.plot(prepared["date"], prepared["raw_tpc"], color="#35978F", lw=1.6, label="Raw model")
    ax.set_ylabel("TPC (mg/L)")
    ax.set_xlabel("Date")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(args.output_dir / "demo_timeseries.png", dpi=300)
    plt.close(fig)

    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
