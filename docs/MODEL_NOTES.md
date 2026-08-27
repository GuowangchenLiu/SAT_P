# Model notes

## Process interpretation

SAT-P is a parsimonious daily model rather than a fully distributed biogeochemical model. Its terms are intended to represent:

- temperature-sensitive phosphorus availability, including weathering, freeze–thaw effects, permafrost thaw, and microbial mineralization;
- rapid phosphorus flushing during snowmelt, rainfall, and rising discharge; and
- progressive exhaustion of readily mobilized phosphorus during the hydrological year.

The model therefore combines process information with an empirical concentration–discharge formulation.

## Current public implementation

The public preprocessing uses the variable definitions in the manuscript:

- 8-day rolling mean air temperature;
- positive 2-day discharge increase; and
- a dimensionless within-year exhaustion proxy.

The supplied research scripts contain older experimental window choices. Those versions are not used by the public command-line workflow.

## Calibration

Calibration uses NSGA-III to maximize squared Pearson correlation and Nash–Sutcliffe efficiency simultaneously. The manuscript used a population of 400 and 1,000 generations. Because evolutionary optimization is stochastic, a random seed is exposed by the calibration script.

## Limitations

- Parameters should be recalibrated when the preprocessing, basin, sampling regime, or observation units change.
- The current formulation does not explicitly separate dissolved and particulate phosphorus.
- Adsorption–desorption, thermokarst events, and vegetation feedbacks are not represented as independent state variables.
- The partial demonstration data are intended for code verification, not full manuscript reproduction.

