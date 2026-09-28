# Working Title

When Do Machine-Learning Equity Signals Deserve Trust? Cross-Model Disagreement
and Out-of-Sample Forecast Reliability

# Motivation

Gu, Kelly, and Xiu (2020) is a foundational reference for machine-learning
prediction of cross-sectional stock returns, and Chen, Hanauer, and Kalsbach
(2026) document substantial sensitivity of such results to modeling and design
choices. Structurally different model families can disagree about the same
stock, yet an ensemble signal reports a single score. Forecast disagreement and
machine-learning forecast uncertainty already exist in the literature (Bali et
al., 2026; Sun, 2026; Allena, 2026). This project asks whether cross-family
disagreement indicates *when* a given signal is more or less reliable.

# Research Question

Does disagreement across structurally different machine-learning model families
predict the reliability of cross-sectional equity-return signals beyond signal
strength, volatility, size, and illiquidity, and how does this information
relate to within-family estimation uncertainty? Reliability means the strength
of the relationship between the ensemble signal and subsequent returns. The
first part (H1–H2) is the primary contribution. The relation to within-family
estimation uncertainty is a secondary analysis (H3), examining whether the two
measures capture distinct information about forecast reliability.

# Related Literature and Research Gap

Bali et al. (2026) construct machine forecast disagreement from dispersion
across heterogeneous machine-learning return forecasts and study its relation
to subsequent stock returns. Sun (2026) allows investors to consider multiple
forecasting specifications and decomposes total machine forecast dispersion
into disagreement across investors and model uncertainty within investors.
Allena (2026) studies forecast precision using ex-ante confidence intervals
around linear and machine-learning risk-premium forecasts. This project builds
on all three but asks a different question: whether cross-model disagreement
identifies variation in the reliability of a common cross-sectional return
signal, conditional on the signal's level and strength and on observable risk
proxies. Relative to Bali et al. (2026), the object of interest is the
signal–return relationship rather than the level of future returns. Relative
to Sun (2026), the project's decomposition is related but not identical:
between-family disagreement compares structurally different model families,
and within-family estimation uncertainty is measured across month-cluster
bootstrap replicas of a fixed family. It is used in a secondary analysis (H3)
to examine whether structural model disagreement and within-family estimation
uncertainty contain distinct information about forecast reliability. Relative
to Allena (2026), uncertainty is measured by disagreement
across model families rather than by model-specific confidence intervals. The
empirical design explicitly addresses the mechanical relationship between
bounded prediction ranks, signal extremity, and measured disagreement.

# Hypotheses

- **H1 (primary):** at comparable signal level $S$, and controlling for
  nonlinear signal strength ($|S^c|$ and $S^c \times |S^c|$), higher
  conditional between-family disagreement is expected to weaken the
  signal–return relationship. The expected interaction sign is negative, but
  inference is two-sided: a significant positive interaction is reported as
  evidence in the opposite direction.
- **H2 (primary):** the effect remains after additionally controlling for
  volatility, size and illiquidity, as levels and interacted with $S^c$.
- **H3 (secondary):** examines whether between-family disagreement and
  within-family estimation uncertainty capture distinct information about
  forecast reliability; both are measured with noise.
- **H4 (secondary):** a disagreement filter will later be compared with no
  filter, a volatility filter and a signal-extremity filter.

# Proposed Empirical Design

Elastic Net, Random Forest, LightGBM and a small MLP, each with K = 5
month-cluster bootstrap replicas, forecast monthly with annual retraining on an
expanding window. The target is the cross-sectional rank of next-month return
within the formation-time universe $U_t$ (eligible NYSE/AMEX/NASDAQ common
stocks, excluding those below the monthly 20th percentile of NYSE market
capitalization), mapped to [-1, 1]. Family consensus ranks (averaged and
re-ranked replica ranks on [0, 1]) are equally weighted into the ensemble
signal level $S \in [0, 1]$. The centered signal is
$S^c = 2(S - 0.5) \in [-1, 1]$, and signal strength (extremity) is $|S^c|$.
Raw disagreement is the standard deviation of the family ranks. Because ranks
are bounded, raw disagreement is mechanically constrained at high signal
strength, so the primary measure ranks it within 20 monthly quantile bins of
the signal level $S$ (not of $|S^c|$). Within-family estimation uncertainty is
constructed from replica-rank dispersion, computed separately within each model
family. Given the design sensitivity documented by Chen et al. (2026),
remaining implementation choices will be pre-specified before any empirical
results are examined.

# Expected Contribution

Primarily, the project tests disagreement as an indicator of signal reliability
rather than a return predictor, with explicit treatment of the mechanical link
to signal strength and controls for observable risk proxies. Secondarily, it
examines how between-family disagreement relates to within-family estimation
uncertainty in this reliability setting, using a decomposition that is related
to, but differs from, that of Sun (2026). The exact empirical claims remain to
be tested.

# Main Risks and Limitations

- Signal binning may not fully remove the mechanical link to signal strength.
- With four families and five replicas each, both uncertainty measures are
  noisy, lowering test power and complicating the H3 comparison.
- Disagreement may proxy for characteristics the controls do not capture.
- Findings depend on the pre-specified design and may not generalize.
- Licensed WRDS/CRSP data cannot be redistributed; replication requires access.
