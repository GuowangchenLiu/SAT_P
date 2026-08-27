# Model notes

## Process interpretation

SAT-P is a parsimonious daily model rather than a fully distributed biogeochemical model. Its terms are intended to represent:

- temperature-sensitive phosphorus availability, including weathering, freeze–thaw effects, permafrost thaw, and microbial mineralization;
- rapid phosphorus flushing during snowmelt, rainfall, and rising discharge; and
- progressive exhaustion of readily mobilized phosphorus during the hydrological year.

The model therefore combines process information with an empirical concentration–discharge formulation.

## Current public implementation

The public preprocessing uses the following variable definitions:

- 8-day rolling mean air temperature;
- positive 2-day discharge increase; and
- a dimensionless within-year exhaustion proxy.

The supplied research scripts contain older experimental window choices. Those versions are not used by the public command-line workflow.

## Calibration

Calibration uses NSGA-III to maximize squared Pearson correlation and Nash–Sutcliffe efficiency simultaneously. Because evolutionary optimization is stochastic, the calibration script exposes the population size, number of generations, and random seed.

## Limitations

- Parameters should be recalibrated when the preprocessing, basin, sampling regime, or observation units change.
- The current formulation does not explicitly separate dissolved and particulate phosphorus.
- Adsorption–desorption, thermokarst events, and vegetation feedbacks are not represented as independent state variables.
- The partial demonstration data are intended for code verification rather than full long-term reconstruction.
