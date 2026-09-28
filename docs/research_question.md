# Research Question

## Question

> Does disagreement across structurally different machine-learning model
> families predict the reliability of cross-sectional equity-return signals
> beyond signal strength, volatility, size, and illiquidity, and how does this
> information relate to within-family estimation uncertainty?

*Reliability* refers to the strength of the relationship between the ensemble
signal and subsequent returns. The question is whether disagreement tells us
*when* a given machine-learning signal is more or less informative. It does not
ask whether disagreement predicts the level of returns.

*Signal strength* refers to the extremity $|S^c|$ of the centered ensemble
signal $S^c = 2(S - 0.5)$, where $S \in [0, 1]$ is the ensemble signal level.
Signal level and signal strength are distinct; see
[hypotheses.md](hypotheses.md) for the definitions.

- **Primary contribution (H1–H2):** whether between-family disagreement
  predicts signal reliability beyond signal strength, volatility, size and
  illiquidity.
- **Secondary mechanism/decomposition analysis (H3):** how between-family
  disagreement relates to within-family estimation uncertainty, and whether
  the two capture distinct information about forecast reliability.

## Motivation

- Machine-learning prediction of cross-sectional stock returns has a
  foundational reference in Gu, Kelly, and Xiu (2020). Chen, Hanauer, and
  Kalsbach (2026) document substantial sensitivity of return-prediction results
  to modeling and design choices.
- Structurally different model families can rank the same stock differently.
  An ensemble signal reports one score per stock and does not show whether the
  families behind it agree.
- Family prediction ranks are bounded. When the ensemble signal is extreme, the
  families must largely agree, so raw disagreement is mechanically related to
  signal strength. A naive analysis could confuse disagreement with signal
  strength.
- Disagreement may also partly reflect simple stock characteristics such as
  volatility, size or illiquidity [TODO: literature reference needed].

## Proposed contribution

Primary (H1–H2):

1. Test whether disagreement identifies variation in forecast reliability,
   rather than directly predicting return levels.
2. Explicitly address the mechanical relation between bounded family ranks,
   signal strength $|S^c|$ and disagreement. The primary measure ranks
   disagreement within 20 monthly quantile bins of the ensemble signal level
   $S$ (not of $|S^c|$).
3. Test whether disagreement contains information about signal reliability
   beyond simple observable stock-risk proxies.

Secondary (H3):

4. Relate between-family disagreement, measured across structurally different
   model families, to within-family estimation uncertainty, measured across
   month-cluster bootstrap replicas of a fixed family, in the
   signal-reliability setting.

## Distinction from the closest work

- **Bali et al. (2026), "Machine Forecast Disagreement".** They represent
  heterogeneous investors using different machine-learning model
  specifications, measure disagreement as the dispersion in return forecasts,
  and study its relation to future stock returns. This project does not
  replicate that return-level test. It asks whether disagreement modifies the
  reliability of a common machine-learning signal at comparable signal levels.
- **Sun (2026), "What Does Machine Forecast Disagreement Measure?".** It allows
  investors to consider multiple forecasting specifications and decomposes
  total machine forecast dispersion into disagreement across investors and
  model uncertainty within investors. This is related to, but not identical to,
  this project's decomposition: here, between-family disagreement compares
  structurally different model families (Elastic Net, Random Forest, LightGBM
  and a small MLP), while within-family estimation uncertainty is measured
  across month-cluster bootstrap replicas of a fixed family. This decomposition
  is used only in the secondary analysis (H3) to examine whether structural
  model disagreement and estimation uncertainty contain distinct information
  about forecast reliability. The primary question is whether cross-model
  disagreement identifies variation in the reliability of a common 
  cross-sectional signal, conditional on the signal level, signal strength,
  and observable risk proxies.
- **Allena (2026), "Confident Risk Premiums and Investments Using Machine
  Learning Uncertainties".** It constructs ex-ante confidence intervals around
  stock risk-premium forecasts from linear and machine-learning models, and
  shows that forecast uncertainty can be economically relevant. This project
  uses cross-model disagreement rather than model-specific confidence
  intervals, and it studies reliability conditional on the ensemble signal
  level.

## What the paper does not claim

- That disagreement is primarily a predictor of the level of future returns;
  the central question concerns forecast reliability conditional on the
  underlying signal.
- That between-family disagreement is inherently more informative than
  within-family estimation uncertainty, or vice versa.
- That its decomposition into between-family disagreement and within-family
  estimation uncertainty is itself novel. It differs from the decomposition in
  Sun (2026), and is used here to examine whether structural model disagreement
  and estimation uncertainty contain distinct information about forecast
  reliability.
- That the paper introduces new machine-learning models or seeks to maximize
  predictive performance.
- That the proposed hypotheses are true. All empirical claims remain to be
  tested, and no results are assumed in advance.
- That the relationship between disagreement and subsequent returns is causal.
