# Results

This folder separates two types of results:

1. `published_metrics.csv` records the model-performance values stated in the manuscript.
2. Files beginning with `demo_` are generated from the partial public dataset by `scripts/run_demo.py`.

Demo results are included to verify that the code runs end to end. They are not expected to reproduce the complete calibration and long-term analysis because the public input file contains only a subset of dates and observations.

The current demonstration uses 221 observations after applying the TPC and discharge filters. It gives $R^2=0.545$ and NSE = 0.533 for SAT-P, compared with $R^2=0.197$ and NSE = 0.178 for the raw model.

To regenerate the demo files from the repository root:

```bash
python scripts/run_demo.py
```
