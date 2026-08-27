# Demonstration data dictionary

| Column | Unit | Description |
|---|---:|---|
| `date` | YYYY/MM/DD | Daily timestamp |
| `temperature` | K | Daily air temperature; converted to °C by the preprocessing code |
| `precipitation` | m day⁻¹ | Daily ERA5-Land precipitation in the supplied file |
| `discharge` | m³ s⁻¹ | Daily river discharge; may be missing |
| `TP` | mg L⁻¹ | Observed total phosphorus concentration; may be missing |

Derived variables are written to `results/demo_predictions.csv` by `scripts/run_demo.py`.

| Derived column | Description |
|---|---|
| `temperature_c` | Air temperature converted to °C |
| `temperature_8d_norm` | Min–max-normalized 8-day mean temperature |
| `discharge_increase_2d_norm` | Min–max-normalized positive 2-day discharge increase |
| `exhaustion_index` | Cumulative fraction of annual precipitation used as the current exhaustion proxy |
| `satp_tpc` | SAT-P simulated TPC |
| `raw_tpc` | Discharge-only rating-curve TPC |

