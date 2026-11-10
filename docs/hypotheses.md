# Hypotheses

These hypotheses are specified before any empirical results are produced. No
return-based results exist yet. Amendments made before any return is examined
are recorded in [methodology_changes.md](methodology_changes.md); the
estimation protocol was frozen on 2026-11-09
([pre_analysis_plan.md](pre_analysis_plan.md), "Estimation protocol").

H1 and H2 are the primary contribution. H3 is a secondary
mechanism/decomposition analysis, and H4 is a secondary descriptive
comparison of selection rules.

## Definitions and fixed design choices

- **Model families:** Elastic Net, Random Forest, LightGBM and a small MLP,
  each estimated on K = 5 month-cluster bootstrap replicas.
- **Forecasting schedule:** monthly forecasts, annual retraining, expanding
  historical training window.
- **Universe U_t:** eligible NYSE/AMEX/NASDAQ common stocks, excluding stocks
  below the monthly 20th percentile of NYSE market capitalization measured at
  formation time t.
- **Sample:** formation months 1970-01 through 2024-12, inclusive.
- **Training target (primary):** the cross-sectional rank of next-month return
  within U_t, mapped to [-1, 1].
- **Features:** the 212 candidate predictors, filtered at each training
  cutoff using historical months only. A predictor is eligible if it has a
  finite observed value for at least 60% of U_t in at least 95% of the
  historical months available at that cutoff. All four model families use
  the same eligible predictors within a training episode.
- **Family consensus rank:** the average of a family's replica prediction
  ranks, re-ranked cross-sectionally on [0, 1].
- **Ensemble signal level $`S`$:** the equal-weighted average of the family
  ranks, $`S \in [0, 1]`$.
- **Centered ensemble signal $`S^c`$:** $`S^c = 2(S - 0.5) \in [-1, 1]`$.
- **Signal strength (signal extremity):** $`|S^c|`$. $`S`$ and $`S^c`$ are
  signal levels; only $`|S^c|`$ is referred to as signal strength.
- **Conditioning bins:** 20 monthly rank-based bins of the ensemble signal
  level $`S`$ (equivalently of $`S^c`$, since centering is monotone), not of
  $`|S^c|`$; exact formula in [pre_analysis_plan.md](pre_analysis_plan.md),
  "Conditional disagreement".
- **Raw between-family disagreement:** the population standard deviation
  (ddof = 0) of the four family ranks.
- **Conditional between-family disagreement (primary measure):** raw
  between-family disagreement ranked within the 20 monthly conditioning bins
  of $`S`$. This addresses the mechanical constraint on disagreement at high
  signal strength $`|S^c|`$, which arises because family ranks are bounded.
- **Within-family estimation uncertainty:** replica-rank dispersion computed
  separately for each family, ranked cross-sectionally within each family,
  averaged across families, and then conditioned on the ensemble signal level
  $`S`$.
- **Forecast reliability:** the strength of the relationship between the
  ensemble signal and the next-month outcome, measured by the signal slope
  of the H1 specification.
- **Evaluation outcome:** `target_rank`, the cross-sectional rank of the
  next-month return `ret_next` within $`U_t`$ mapped to $`[-1, 1]`$ (primary);
  raw `ret_next` (robustness). Estimation, inference and reporting rules:
  [pre_analysis_plan.md](pre_analysis_plan.md), "Estimation protocol".
- **H2 control characteristics:** size, realized volatility and illiquidity,
  defined in [pre_analysis_plan.md](pre_analysis_plan.md), "H2 control
  characteristics", each transformed separately into a monthly
  cross-sectional rank within $`U_t`$ scaled to $`[-1, 1]`$.

## Primary hypotheses

### H1: Forecast reliability

At comparable ensemble signal level $`S`$, and after accounting for the relation
between subsequent returns and the signal within each signal bin, higher
conditional between-family disagreement is expected to weaken the relationship
between the ensemble signal and subsequent returns.

The primary H1 reliability specification is estimated separately in each OOS
formation month $`t`$, in a Fama–MacBeth framework:

```math
y_{i,t+1} = \alpha_{t,b(i,t)} + \gamma_{t,b(i,t)}\,S^c_{i,t}
+ \eta_{t,b(i,t)}\,(S^c_{i,t})^2 + \beta_{2,t}\,CD^B_{i,t}
+ \beta_{3,t}\,(S^c_{i,t} \times CD^B_{i,t}) + \varepsilon_{i,t+1}
```

where $`y_{i,t+1}`$ is the evaluation outcome, $`S^c = 2(S - 0.5)`$ uses the
original, non-re-ranked ensemble signal, $`b(i,t)`$ is the stock's existing
monthly signal bin among the 20 conditioning bins, and $`CD^B`$ is the
existing conditional between-family disagreement. The intercept and the
linear and quadratic signal terms are specific to each (month, signal bin);
the disagreement level and the signal × disagreement interaction are common
to the month's cross-section. The primary coefficient of interest,
$`\beta_{\text{interaction}}`$, is the Fama–MacBeth average of the monthly
$`\beta_{3,t}`$ (estimation and inference: pre-analysis plan, "Estimation
protocol").

The mechanical-bias diagnostics ([pre_analysis_plan.md](pre_analysis_plan.md),
"Signal controls in the H1 specification") found that, inside the extreme
signal bins, conditional disagreement still depends strongly on the exact
signal location, with within-bin correlations of about +0.65 in bin 1 and
−0.66 in bin 20. Because H1 rests on a signal × disagreement interaction,
signal controls that are not flexible enough can let a nonlinear
signal–return relation appear as a disagreement effect. The bin-specific
quadratic controls absorb any relation between returns and the signal that is
quadratic within a bin, without redefining the signal or the disagreement
measure. They do not eliminate every conceivable nonlinear relation, and they
do not establish causality. They replace the global controls $`|S^c|`$ and
$`S^c \times |S^c|`$ of the original specification, which is preserved in
[methodology_changes.md](methodology_changes.md).

- *Null:* $`\beta_{\text{interaction}} = 0`$; at comparable signal level, the
  relationship between the ensemble signal and subsequent returns does not vary
  with conditional between-family disagreement.
- *Alternative (two-sided):* $`\beta_{\text{interaction}} \neq 0`$.
- *Pre-specified expected direction:* $`\beta_{\text{interaction}} < 0`$; the
  relationship is weaker where conditional between-family disagreement is
  higher.

The economic hypothesis is directional, but statistical inference is two-sided.
A statistically significant positive interaction will be reported as evidence
in the opposite direction, not discarded.

**Verdict.** Two-sided test at the 5% level on the Fama–MacBeth average of
$`\beta_{3,t}`$, with the HAC inference of the pre-analysis plan. H1 is
"supported" if the estimate is negative and $`p < 0.05`$, "evidence in the
opposite direction" if it is positive and $`p < 0.05`$, and "inconclusive"
otherwise; the 95% interval is reported in every case. A negative
interaction is a statistical statement about the signal slope. The economic
reading that a positively informative signal becomes less reliable where
disagreement is higher is claimed only if, in addition, the average signal
slope at low disagreement ($`CD^B = 0.1`$) is significantly positive
(pre-analysis plan, "Reporting rules for H1 and H2").

H1 concerns the signal–return relationship, not the level of returns.

**Pre-specified H1 robustness**

- *Bin-specific linear controls:* the same equation without the bin-specific
  quadratic term $`\eta_{t,b(i,t)}\,(S^c_{i,t})^2`$. This is the primary
  specification robustness.
- *Excluding 2021:* the primary test repeated without the twelve 2021 OOS
  formation months (2021-01 to 2021-12), during which the Elastic Net family
  produced constant cross-sectional forecasts (see the pre-analysis plan). The
  baseline sample, the four-family ensemble and the construction of $`CD^B`$ and
  $`CW`$ are unchanged; the exclusion is a robustness analysis only.
- *Other pre-specified robustness analyses* (raw outcome, raw, range and MAD
  disagreement measures, 10 bins, leave-one-family-out, HAC lags, two
  subperiods): pre-analysis plan, "Robustness and subperiods".

### H2: Incremental information

The H1 effect remains when the primary H1 specification, with its bin-specific
quadratic signal controls, is extended with three stock characteristics,
all measured with information available at the end of formation month $`t`$:

- *size:* the natural logarithm of the formation-month security-level market
  capitalisation (CRSP CIZ `MthCap`, the universe's `mktcap`);
- *realized volatility:* the OSAP predictor `RealizedVol` with its sign
  reversed: the standard deviation (one degree of freedom lost) of the
  stock's daily excess returns in month $`t`$, requiring at least 15 daily
  observations;
- *illiquidity:* the OSAP predictor `Illiquidity`, sign preserved: the
  average of the twelve most recent available monthly means of Amihud's
  daily ratio $`\lvert r_d\rvert / (\lvert p_d\rvert \times \text{vol}_d)`$,
  the latest being month $`t`$.

Each characteristic is converted separately into a monthly cross-sectional rank
among its finite values within $`U_t`$, scaled to $`[-1, 1]`$ (average ranks for
ties), and enters both as a level term and interacted with the original centered
signal $`S^c`$. Exact construction, sources, rank and missing-value conventions
and known limitations: [pre_analysis_plan.md](pre_analysis_plan.md), "H2 control
characteristics". H2 asks whether disagreement carries incremental information
about *signal reliability*, not merely about return levels. The interactions are
therefore needed so that disagreement is not simply standing in for these
characteristics' own effect on the signal–return relationship. The three
characteristics are also model predictors, by design: H2 tests incremental
information conditional on them, not causal identification or complete risk
adjustment.

- *Null:* once these controls are included, conditional between-family
  disagreement no longer modifies the relationship between the ensemble signal
  and subsequent returns.
- *Alternative (two-sided):* the interaction coefficient remains different
  from zero after the controls, with the same pre-specified expected direction
  ($`\beta_{\text{interaction}} < 0`$) as in H1.

Within-family estimation uncertainty is not part of the H2 control set. Its
relation to between-family disagreement is examined in H3.

H2 is estimated on the complete-case subsample without imputation; the
sample rules, the re-estimation of specification C on the exact H2 sample
and the inference are fixed in the pre-analysis plan, "Estimation
protocol".

**Verdict.** H1 and H2 form a fixed-sequence pair: H2 is confirmatory only
if H1 is supported in the negative direction, which keeps the familywise
error of the pair at 5% without correction. H2 "survives" if its
interaction estimate is negative with $`p < 0.05`$ and "does not survive"
otherwise. H2 is estimated and fully reported whatever the H1 verdict, and
labelled descriptive when the gate is not passed. The paired differences
between specification C on the H2 sample, H2 and H1 are reported with
intervals and are not interpreted causally (pre-analysis plan, "Reporting
rules for H1 and H2").

## Secondary analysis

### H3: Between-family disagreement vs within-family estimation uncertainty

Between-family disagreement and within-family estimation uncertainty may
contain distinct information. This is a secondary mechanism/decomposition
analysis.

- Neither measure is assumed or required to dominate the other.
- The decomposition is related to, but not identical to, that of Sun (2026),
  which decomposes total machine forecast dispersion into disagreement across
  investors and model uncertainty within investors. Here, between-family
  disagreement compares structurally different model families, and
  within-family estimation uncertainty is measured across month-cluster
  bootstrap replicas of a fixed family. This distinction is used to examine
  whether structural model disagreement and estimation uncertainty contain
  distinct information about forecast reliability.
- The intended analysis is a joint specification: conditional between-family
  disagreement $`CD^B`$ and conditional within-family uncertainty $`CW`$ both
  enter as level terms and as separate interactions with the original centered
  signal $`S^c`$, with the same bin-specific quadratic signal controls as the
  primary H1 specification and in the same Fama–MacBeth framework. The
  question is whether each measure contributes information about forecast
  reliability conditional on the other. A direct comparison of the two
  interaction coefficients does not by itself establish which uncertainty
  concept is more informative: the measures are built from different numbers
  of components and different ranking operations.
- Both measures contain finite-model noise (four families for $`CD^B`$,
  K = 5 replicas per family for $`CW`$) and both are bounded ranks, so their
  errors are neither additive nor independent of the level: classical
  attenuation results do not apply, no ordering of their measurement
  precision is asserted, and no measurement-error correction or
  replica-split reliability diagnostic is performed. An analysis-of-variance
  correction treating within-family dispersion as independent sampling noise
  is not appropriate either: for a given replica the four families share the
  same bootstrap month draws, and $`CD^B`$ and $`CW`$ use different ranking
  operations.
- *Test:* each interaction is tested two-sided at 5%; "distinct information"
  requires both to reject (intersection–union test, level 5% without
  correction), and the joint Wald test of the two interactions is reported
  as a complement. H3 is outside the H1 → H2 sequence.

## Secondary economic application

### H4: Disagreement-based filter

A pre-specified secondary comparison of selection rules on gross one-month
returns: holdings are fixed at formation and the statistic averages the
observed returns of the selected holdings, so it is a conditional
observed-outcome comparison, not an implementable fixed-weight portfolio
backtest, and no claim of economic value is made. In each OOS formation
month, a long leg and a short leg are formed from the extreme signal deciles
of $`U_t`$ with equal weights fixed at formation, and a fraction
$`\varphi = 1/3`$ of each leg is removed by one of three filters applied at
equal intensity:

- disagreement filter: the stocks with the highest conditional
  between-family disagreement $`CD^B`$;
- volatility filter: the stocks with the highest realized volatility;
- signal-extremity filter: the stocks with the lowest signal extremity
  $`|S^c|`$.

The unfiltered portfolio is the benchmark. Each filter is compared with no
filter on identical months through paired monthly return differences,
two-sided at 5%. No expected ranking of the alternatives is specified and no
superiority claim will be made. Construction, missing-return policy,
coverage disclosure and statistics: pre-analysis plan, "H4". A separate
secondary backtesting extension with assumed trading costs is specified
there; it does not enter the H4 comparison or any verdict.
