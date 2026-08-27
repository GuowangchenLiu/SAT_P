# Data included in this repository

`demo/daily_inputs_2022_2024.csv` is a partial daily dataset supplied to demonstrate the SAT-P workflow. It contains 819 records from 3 April 2022 to 29 June 2024 and 349 non-missing TPC observations before quality filtering.

The file is intentionally limited in temporal coverage. It does not contain the complete 2000–2023 analysis archive used for the long-term results in the manuscript.

## Sources

- Air temperature and precipitation originate from ERA5-Land in the research workflow.
- Daily discharge and TPC are observational inputs used in the SAT-P case study.

Users are responsible for citing the original ERA5-Land product when reusing the climate variables:

> Muñoz-Sabater et al. (2021), ERA5-Land. https://doi.org/10.24381/cds.68d2bb30

## License

The demonstration file is distributed under CC BY 4.0. See [../DATA_LICENSE.md](../DATA_LICENSE.md).

## Missing values

Missing discharge or TPC values are represented by empty CSV fields. Do not replace missing TPC values with zeros. The preprocessing retains complete climate records to calculate rolling and cumulative predictors.

