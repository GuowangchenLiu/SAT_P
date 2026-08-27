# Results

Files beginning with `demo_` are generated from the partial public dataset by `scripts/run_demo.py`. They are included to verify that the code runs end to end.

The current demonstration uses 221 observations after applying the TPC and discharge filters. It gives $R^2=0.545$ and NSE = 0.533 for SAT-P, compared with $R^2=0.197$ and NSE = 0.178 for the raw model.

To regenerate the demo files from the repository root:

```bash
python scripts/run_demo.py
```
