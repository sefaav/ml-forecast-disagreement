# When Do Machine-Learning Equity Signals Deserve Trust?

**Cross-Model Disagreement and Out-of-Sample Forecast Reliability**

This project asks when machine-learning forecasts of cross-sectional stock
returns can be trusted. The planned design trains several model families, each
over multiple bootstrap replicas, in a walk-forward setting, and will use
between-family disagreement among their forecasts, together with within-family
estimation uncertainty across replicas, to study whether disagreement signals
lower out-of-sample forecast reliability
beyond signal strength and observable risk, and whether that information has
economic value.

> **Status:** the data pipeline, the walk-forward forecasts, the signals,
> the disagreement measures and the H2 controls are implemented and tested on
> synthetic data; the pre-analysis protocol is frozen
> ([docs/pre_analysis_plan.md](docs/pre_analysis_plan.md)). The estimation
> engine is not yet implemented and no empirical result has been examined.
> The repository contains no licensed data, models or empirical results.
