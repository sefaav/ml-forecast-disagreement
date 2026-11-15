# When Do Machine-Learning Equity Signals Deserve Trust?

**Cross-Model Disagreement and Out-of-Sample Forecast Reliability**

Machine-learning forecasts of cross-sectional stock returns report one score
per stock and do not show whether the models behind it agree. This project
asks whether disagreement across structurally different model families tells
us *when* such a signal is more or less reliable, beyond signal strength,
volatility, size and illiquidity, and how that information relates to the
estimation uncertainty within each family
([research question](docs/research_question.md)).

- **Primary (H1–H2).** At a comparable signal level, does higher conditional
  between-family disagreement weaken the relation between the ensemble signal
  and next-month returns (H1), and does that effect remain after size,
  volatility and illiquidity controls (H2)?
- **Secondary (H3).** Do between-family disagreement and within-family
  estimation uncertainty carry distinct information about reliability?
- **Secondary economic application (H4).** A pre-specified comparison of
  disagreement-based and other selection rules.

Disagreement is measured within 20 monthly bins of the ensemble signal level,
because bounded family ranks make raw disagreement mechanically small when the
signal is extreme. Hypotheses, expected directions and verdict categories:
[docs/hypotheses.md](docs/hypotheses.md).

## Status

> The descriptive out-of-sample forecasting diagnostics (monthly rank IC of
> the four family ranks and the ensemble) have been executed on the real
> outcomes. No hypothesis has been tested: the confirmatory H1–H3 inference
> has not been executed, and the other pre-specified robustness analyses and
> H4 are pending. The repository contains no licensed data, models or
> empirical results.

**Implemented and tested on synthetic data**

- *Data:* CRSP CIZ monthly security data and the pinned October 2025 Open
  Source Asset Pricing release; a formation-time universe of NYSE, AMEX and
  NASDAQ common stocks above the NYSE 20th size percentile; 212 candidate
  predictors; the next-month return and its monthly rank `target_rank`
  ([data pipeline](docs/data_pipeline.md)).
- *Walk-forward forecasts:* annual retraining with an expanding training
  window and a five-year rolling validation window, out-of-sample formation
  months 1985-01 to 2024-12 (480 months). Four model families (Elastic Net,
  Random Forest, LightGBM and a small MLP), each estimated on five
  month-cluster bootstrap replicas.
- *Signals:* family ranks; the ensemble signal, the mean of the four family
  ranks; conditional between-family disagreement, the standard deviation of
  the family ranks ranked within the 20 signal bins; conditional within-family
  uncertainty from the replica ranks; the H2 control ranks.
- *Core inference for H1–H3:* the frozen monthly regressions (specification C
  with bin-local quadratic signal controls, specification B, H2, C on the H2
  sample, H3), Fama–MacBeth inference with a Bartlett HAC standard error
  (lag 6 decisional, lags 0 and 12 as sensitivity), the frozen decision rules,
  and a guarded command that records the SHA-256 of every input and publishes
  the results atomically ([data/README.md](data/README.md)).
- *Forecasting diagnostics:* the monthly rank IC of the frozen model-selection
  metric for the four family ranks and the ensemble, with equal month weights
  (descriptive; executed).

The forecast-only design evidence behind the H1 specification was regenerated
from the saved signals by a committed diagnostic, without reading any outcome
(pre-analysis plan, "Reproducibility of the evidence").

**Pre-specified but not yet implemented**

- H1 robustness analyses: excluding the twelve 2021 formation months; raw,
  range and MAD disagreement measures; ten signal bins; leave-one-family-out
  ensembles; the two subperiods 1985–2004 and 2005–2024.
- H4 and its separate secondary backtesting extension.

The implemented robustness checks are specification B, the raw `ret_next`
outcome and the HAC lags 0 and 12.

## Protocol

The research protocol was frozen on 2026-11-09 and amended once on
2026-11-10, before any outcome was examined:

- [docs/pre_analysis_plan.md](docs/pre_analysis_plan.md): data, forecasts,
  estimation protocol, decision rules, robustness analyses, H3 and H4;
- [docs/methodology_changes.md](docs/methodology_changes.md): every change
  to the design, dated, with its reason;
- [docs/data_governance.md](docs/data_governance.md): handling of licensed
  data, credentials and derived outputs.

## Software and findings

The test suite (`python -m pytest`) checks that the code implements the
pre-specified computations, using synthetic inputs only. Passing tests are
evidence about the software, not about any hypothesis. The only empirical
results so far are the descriptive out-of-sample forecasting diagnostics.
Formal H1–H3 hypothesis-test results will exist only after the frozen
inference is run on the real outcomes, and they will be reported under the
pre-specified rules whatever their direction. H4 and the remaining robustness
analyses are pending.

## Repository

| Path | Content |
|---|---|
| `src/disagreement/` | data pipeline, forecasts, signals, controls, diagnostics, inference |
| `tests/` | synthetic tests |
| `configs/baseline.yaml` | frozen baseline configuration |
| `docs/` | research question, hypotheses, protocol, data pipeline, governance |
| `data/` | local licensed data, ignored by Git; commands in [data/README.md](data/README.md) |
