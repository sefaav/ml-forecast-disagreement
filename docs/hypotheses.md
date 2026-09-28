# Hypotheses

These hypotheses are specified before any empirical results are produced. No
results exist yet.

H1 and H2 are the primary contribution. H3 is a secondary
mechanism/decomposition analysis, and H4 is a secondary economic application.

## Definitions and fixed design choices

- **Model families:** Elastic Net, Random Forest, LightGBM and a small MLP,
  each estimated on K = 5 month-cluster bootstrap replicas.
- **Forecasting schedule:** monthly forecasts, annual retraining, expanding
  historical training window.
- **Universe U_t:** eligible NYSE/AMEX/NASDAQ common stocks, excluding stocks
  below the monthly 20th percentile of NYSE market capitalization measured at
  formation time t.
  [TODO: exact sample start and end dates.]
- **Training target (primary):** the cross-sectional rank of next-month return
  within U_t, mapped to [-1, 1].
- **Features:** [TODO: exact feature-coverage threshold and resulting feature
  count.]
- **Family consensus rank:** the average of a family's replica prediction
  ranks, re-ranked cross-sectionally on [0, 1].
- **Ensemble signal level $S$:** the equal-weighted average of the family
  ranks, $S \in [0, 1]$.
- **Centered ensemble signal $S^c$:** $S^c = 2(S - 0.5) \in [-1, 1]$.
- **Signal strength (signal extremity):** $|S^c|$. $S$ and $S^c$ are signal
  levels; only $|S^c|$ is referred to as signal strength.
- **Conditioning bins:** 20 monthly quantile bins of the ensemble signal level
  $S$ (equivalently of $S^c$, since centering is monotone), not of $|S^c|$.
- **Raw between-family disagreement:** the standard deviation across family
  ranks.
- **Conditional between-family disagreement (primary measure):** raw
  between-family disagreement ranked within the 20 monthly conditioning bins
  of $S$. This addresses the mechanical constraint on disagreement at high
  signal strength $|S^c|$, which arises because family ranks are bounded.
- **Within-family estimation uncertainty:** replica-rank dispersion computed
  separately for each family, ranked cross-sectionally within each family,
  averaged across families, and then conditioned on the ensemble signal level
  $S$.
- **Forecast reliability:** the strength of the relationship between the
  ensemble signal and subsequent returns.
  [TODO: the statistic that measures the strength of this relationship,
  including the return variable used in evaluation; exact HAC/Newey-West lag.]

## Primary hypotheses

### H1: Forecast reliability

At comparable ensemble signal level $S$, and after accounting for nonlinear
signal strength, higher conditional between-family disagreement is expected to
weaken the relationship between the ensemble signal and subsequent returns.

Conceptually, the primary H1 reliability specification is

$$
r_{i,t+1} = \alpha_t + \beta_1 S^c + \beta_2 CD^B + \beta_3 (S^c \times CD^B)
+ \beta_4 |S^c| + \beta_5 (S^c \times |S^c|) + \varepsilon_{i,t+1}
$$

where $CD^B$ is conditional between-family disagreement and
$\beta_3 \equiv \beta_{\text{interaction}}$ is the primary interaction
coefficient (estimation details: see the forecast-reliability TODO above). The
nonlinear signal-strength controls $|S^c|$ and $S^c \times |S^c|$ allow
subsequent returns to depend nonlinearly on signal extremity even after
disagreement is conditioned within signal-level bins; the bins are not assumed
to eliminate the mechanical relation between disagreement and signal strength
perfectly.

- *Null:* $\beta_{\text{interaction}} = 0$; at comparable signal level, the
  relationship between the ensemble signal and subsequent returns does not vary
  with conditional between-family disagreement.
- *Alternative (two-sided):* $\beta_{\text{interaction}} \neq 0$.
- *Pre-specified expected direction:* $\beta_{\text{interaction}} < 0$; the
  relationship is weaker where conditional between-family disagreement is
  higher.

The economic hypothesis is directional, but statistical inference is two-sided.
A statistically significant positive interaction will be reported as evidence
in the opposite direction, not discarded.

H1 concerns the signal–return relationship, not the level of returns.

### H2: Incremental information

The H1 effect remains when the H1 specification, which already includes the
nonlinear signal-strength controls, is extended with controls for:

- volatility;
- size;
- illiquidity.

Volatility, size and illiquidity enter both as level controls and interacted
with $S^c$. H2 asks whether disagreement carries incremental
information about *signal reliability*, not merely about return levels. The
interactions are therefore needed so that disagreement is not simply
standing in for these characteristics' own effect on the signal–return
relationship.

- *Null:* once these controls are included, conditional between-family
  disagreement no longer modifies the relationship between the ensemble signal
  and subsequent returns.
- *Alternative (two-sided):* the interaction coefficient remains different
  from zero after the controls, with the same pre-specified expected direction
  ($\beta_{\text{interaction}} < 0$) as in H1.

Within-family estimation uncertainty is not part of the H2 control set. Its
relation to between-family disagreement is examined in H3.

[TODO: exact trailing-volatility definition; exact illiquidity measure; exact
size measure.]

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
- H3 remains secondary because both measures contain finite-model measurement
  noise: four families for between-family disagreement and K = 5 replicas per
  family for within-family estimation uncertainty. The two measures may be
  estimated with different precision, so any comparison between them must take
  this noise into account.
  [TODO: how measurement noise will be addressed; literature reference needed.]

## Secondary economic application

### H4: Disagreement-based filter

A disagreement-based filter will later be compared with:

- no filter;
- a volatility filter;
- a signal-extremity filter.

No expected ranking of these alternatives is specified. This analysis is not
performed at this stage.
[TODO: filter construction, portfolio formation and evaluation criteria.]
