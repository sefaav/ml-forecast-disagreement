# Methodology Changes

Dated record of amendments to the pre-analysis plan. Each entry states the
evidence it rests on. None of the decisions below was taken after examining
realized future returns or any H1–H4 result.

## 2026-11-09 — H1 signal controls; Elastic Net in 2021

**Status.** Pre-results amendment. The plan is not frozen: the return
variable, the Fama–MacBeth inference implementation and HAC/Newey–West lag,
the exact H2 control definitions and the remaining robustness details are
still open.

**Evidence used.** (i) The mechanical-bias diagnostics on the saved OOS
signals over the 480 baseline months: the 20-bin conditioning removes the
relation between $CD^B$ and the signal in the full monthly cross-section, but
inside the extreme bins $CD^B$ still depends on the exact signal location
(mean within-bin correlation with $S$ about +0.65 in bin 1 and −0.66 in bin
20, same sign in every month except the twelve 2021 months, where it
reverses; within-bin $E[CD^B \mid S]$ close to linear,
slopes about +9 and −8). (ii) The same diagnostic for $CW$ (about +0.72 and
−0.69) and for 30 and 40 bins (extreme-bin correlations unchanged at about
±0.66). (iii) An audit of the Elastic Net 2021 episode from the saved
forecasts and model-selection metadata. (iv) Mathematical analysis of the
interaction regressor and experiments with fully synthetic outcomes, using
the saved $S$, $CD^B$ and bins as fixed regressors. Inputs were therefore
predictor outputs, model metadata, mathematics and synthetic outcomes only;
no realized return, target or H1–H3 result was inspected.

**Decision 1 — H1 specification.** The global signal-strength controls
$|S^c|$ and $S^c \times |S^c|$ are replaced by (month, signal bin) intercepts
and bin-specific linear and quadratic terms in $S^c$ (specification C in
[pre_analysis_plan.md](pre_analysis_plan.md), "Signal controls in the H1
specification"), estimated month by month in a Fama–MacBeth framework. The
interaction $S^c \times CD^B$ on the original, non-re-ranked signal, the
expected negative sign and the two-sided inference are unchanged. Reason: the
interaction regressor contains a term $S^c\,E[CD^B \mid S]$ that is a
function of the signal alone, quadratic within each bin; under a nonlinear
signal–return relation it loads on the interaction coefficient unless the
signal controls span it. The sign of the induced bias depends on the
signal–return relation and is not determined by the geometry alone; it was
negative, in the expected H1 direction, under the synthetic scenarios used
here. In synthetic-outcome experiments with no true disagreement
effect, a boundary-nonlinearity scenario produced spurious rejections at the
5% level in about 18.3% of replications for the original specification A,
5.7% for B (linear bin-specific controls), 5.3% for C and 20.7% for D (spline
on the re-ranked signal). C keeps about 98.15% of B's identifying variance, a
standard-error cost of about 0.9%, and absorbs exactly a within-bin quadratic
nuisance component under which B retained a bias approaching one standard
error in a targeted stress scenario. B is retained as the pre-specified
specification robustness. The spline/re-ranked-signal approach is rejected:
on the saved signals it removes none of the bin-dependent structure of
$CD^B$, and re-ranking changes the estimand. A two-step "detrend $CD^B$
within bin, then interact" estimator is algebraically identical to C when it
uses C's controls (Frisch–Waugh–Lovell, verified numerically) and distinct
when it uses only B's controls; an earlier unconditional statement of this
identity was corrected during the review. The synthetic experiments assume
independent homoskedastic noise and compare specifications; they are not a
guarantee of nominal inference under actual return dependence.

**Decision 2 — Elastic Net in 2021.** In the 2021 episode all Elastic Net
candidates had a negative validation rank IC except the two most penalised
ones, which fit the intercept only, score exactly 0 and were selected under
the frozen rule, giving constant forecasts in all twelve OOS formation months
and a family rank of 0.5 for every stock. This is a legitimate outcome of the
protocol, not an implementation or data error, and no other family-year is
affected. With one family fixed at 0.5 the feasible geometry of four-family
disagreement changes and the usual within-bin dependence of $CD^B$ on $S$
reverses at the extremes (2021: about −0.947 in bin 1 and +0.788 in bin 20,
against about +0.674 and −0.629 for a three-family measure built from the
active families). Treatment: the fixed four-family design, the existing
$CD^B$/$CW$ construction and the saved 2021 outputs stay primary; no
inactive-family rule is introduced; the hyperparameter selection rule is
unchanged; the primary H1 test is repeated without the twelve 2021 formation
months as a robustness analysis, because the bin-specific controls absorb the
systematic dependence on signal location but do not make the four- and
three-family rankings equivalent. No claim about the effect of 2021 on future
returns or on the sign of any H1 estimate is made.

**Decision 3 — H2 and H3 consistency.** H2 extends specification C with
volatility, size and illiquidity, as level controls and interacted with the
original centered signal; their definitions remain to be fixed. H3 is a joint
specification in which $CD^B$ and $CW$ enter as levels and as separate
interactions with $S^c$, with the same bin-specific quadratic controls, to
test whether each contributes information conditional on the other; a direct
comparison of the two coefficients is not treated as establishing which
concept is more informative. An analysis-of-variance noise correction and a
common-scale variance decomposition are not required analyses, because the
four families share the same bootstrap month draws for a given replica and
the two measures use different ranking operations; the measurement-noise
question stays open in [hypotheses.md](hypotheses.md).

**Documents synchronised.** [research_proposal.md](research_proposal.md)
was updated on the same date to state H1 with the bin-specific signal
controls, to mention them in the empirical design and to qualify the binning
risk; research question, scope and all other content are unchanged.

**Reproducibility.** The within-bin correlations of $CD^B$ with the signal
are reproducible with `disagreement.diagnostics.mechanical_bias`. The other
figures cited above (within-bin slopes, $CW$, 30/40 bins, specification
diagnostics, synthetic experiments, Frisch–Waugh–Lovell check, Elastic Net
2021 audit) were produced by temporary scripts deleted after use and are
recorded, not reproducible, results; their definitions are in
[pre_analysis_plan.md](pre_analysis_plan.md), "Reproducibility of the
evidence", pending a committed diagnostic module.

**Original H1 specification (superseded, kept for the record).** Before this
amendment, [hypotheses.md](hypotheses.md) specified

$$
r_{i,t+1} = \alpha_t + \beta_1 S^c + \beta_2 CD^B + \beta_3 (S^c \times CD^B)
+ \beta_4 |S^c| + \beta_5 (S^c \times |S^c|) + \varepsilon_{i,t+1},
$$

with the global controls $|S^c|$ and $S^c \times |S^c|$, the same expected
sign and the same two-sided inference.

## 2026-11-09 — H2 control definitions

**Status.** Pre-results amendment, taken after the H1 amendment of the same
date. It closes the H2 control definitions that the previous entry listed as
open. The return variable, the Fama–MacBeth inference implementation, the
HAC/Newey–West lag, the policy for observations without an outcome and the
H1/H2 decision rules remain open; the pre-analysis plan is not frozen.

**Evidence used.** The repository's data layer and its coverage diagnostics
(the CRSP extract holds fifteen monthly fields and no volume or daily data;
the panel holds the OSAP predictors `RealizedVol`, `Illiquidity` and `Size`
and the universe holds `mktcap`; monthly coverage of these columns among
$U_t$ over the 480 OOS months), the OSAP code at tag `v2.0.0` of
`OpenSourceAP/CrossSection` (commit `b4e911e6…`), CRSP and WRDS
documentation on NASDAQ volume, and the literature on Amihud's measure and
realized volatility. No return, target or H1–H4 result was inspected.

**Decision.** The H2 controls are: size = $\ln(\texttt{mktcap})$ from the
formation-month universe; realized volatility = the sign-reversed OSAP
`RealizedVol` (standard deviation of daily excess returns in month $t$, at
least 15 observations); illiquidity = the OSAP `Illiquidity` (mean of the
twelve most recent available monthly Amihud means through month $t$). Each
is ranked separately within $U_t$ among its finite values, scaled to
$[-1, 1]$ with average ranks for ties, and enters specification C as a level
term and interacted with $S^c$. Complete-case estimation without imputation;
$U_t$, $S$, $CD^B$, $CW$ and the bins unchanged; specification C re-estimated
on the H2 sample for comparison. Full specification:
[pre_analysis_plan.md](pre_analysis_plan.md), "H2 control characteristics".

**Why existing predictors rather than a daily CRSP reconstruction.** The
alternative was to extract the CRSP daily file (`stkdlysecuritydata`), and to
build month-$t$ volatility and an Amihud measure with chosen minimum-day
rules and the Gao–Ritter NASDAQ volume adjustment. It was not chosen because
it requires a new licensed extraction and a second licensed-data pipeline,
whereas the existing predictors are already in the panel, are dated at month
$t$ without lag, are the characteristics the four model families actually
received, and match the intended definitions closely (volatility of daily
returns in month $t$ with a 15-observation floor; a twelve-month Amihud
average). The simpler design was preferred.

**Limitations accepted.** Unadjusted NASDAQ volume before 2004 understates
the Amihud ratio of NASDAQ stocks relative to other stocks in the same
month, and monthly ranking does not necessarily remove this; the twelve
`Illiquidity` observations are available months, not necessarily consecutive
calendar months; missing `Illiquidity` values exclude mostly recently listed
stocks (coverage among $U_t$ averages 0.954 over 1985–2024, with a minimum
of 0.863); `RealizedVol` keeps a stock-month with at least 15 excess-return
observations even if fewer residuals are valid; the OSAP signal
documentation describes `RealizedVol` as a CAPM-residual volatility while
the reference code computes the total excess-return volatility, and the code
is taken as the reference; the controls overlap with the model predictors by
design. The panel's `Size` predictor (OSAP `mve_c`, PERMNO-level from the
legacy monthly file, master-table step not verified) is not treated as
identical to the CIZ `mktcap` and is not used.

## 2026-11-09 — Estimation protocol, decision rules, H3 and H4; plan frozen

**Status.** Pre-results amendment, taken after the two entries above; it
closes every item they listed as open. The pre-analysis plan is frozen for
H1–H4. What remains is implementation: the diagnostic module of
"Reproducibility of the evidence" and the estimation engine, both built and
tested on synthetic data before the first regression on real outcomes, and a
supervision checkpoint to be recorded here.

**Evidence used.** Outcome-free design diagnostics on the saved regressors
over the 480 OOS months (cell counts, condition numbers of the raw and
bin-local bases, variance inflation and identifying variance of
$S^c \times CD^B$ with and without the H2 controls, monthly correlation of
$CD^B$ and $CW$), a synthetic-outcome check that both bases give the same
$\beta_3$, the saved H2-control coverage and the econometric literature. No
return, target or H1–H4 result was inspected. The diagnostics were produced
by temporary scripts and are recorded figures, to be reproduced by the
diagnostic module.

**Decisions** (full statement: [pre_analysis_plan.md](pre_analysis_plan.md),
"Estimation protocol").

- Primary evaluation outcome `target_rank`, raw `ret_next` as robustness:
  the question concerns the reliability of the ordering, the scale is the
  same every month and extreme returns do not dominate monthly coefficients;
  the raw return keeps the economic magnitude.
- Equal-weighted monthly OLS and Fama–MacBeth averages; bin-local
  polynomial basis with the column space of specification C (condition
  number about 4 against about $4 \times 10^3$).
- HAC: Bartlett kernel, 6 calendar months, normalisation $1/T$, no
  finite-sample correction, standard normal inference; lags 0 and 12 as
  sensitivity. A fixed lag was preferred to a data-dependent bandwidth so
  that the standard error does not depend on the realised coefficient
  series; fixed-$b$ inference and Student critical values were not adopted.
- Month estimability (30 observations and 3 distinct $S^c$ per cell, scaled
  $\sigma_{\min} / \sigma_{\max} > 10^{-8}$), calendar-aware HAC with
  missing months, paired comparisons on common months.
- Missing outcomes excluded month by month; $U_t$, signals, bins, controls
  and `target_rank` never recomputed; H2 sample = H1 sample ∩ complete
  case; specification C always re-estimated on the H2 sample.
- Fixed-sequence H1 → H2 at 5%, two-sided, with verdict categories; paired
  differences with intervals and no causal wording; marginal slopes at
  $CD^B \in \{0.1, 0.5, 0.9\}$; statistical and economic statements kept
  apart.
- Robustness set and subperiods 1985–2004 / 2005–2024 fixed.
- H3: intersection–union test of the two interactions, joint Wald test as a
  complement, no measurement-error correction or replica-split diagnostic
  (closes the H3 TODO).
- H4 frozen: extreme-decile long–short, equal weights fixed at formation,
  one-month holding, $\varphi = 1/3$ (0.2 and 0.5 as sensitivity), three
  filters at equal intensity with a PERMNO tie-break, missing filter
  characteristic not eligible for removal, leg statistic = mean `ret_next`
  over holdings with an observed return (no imputation, no zero return, no
  implementable-backtest claim), common months across variants, coverage
  disclosed, gross returns, paired HAC differences (closes the H4 TODO). A
  complete-case calendar-month rule was rejected because it would discard
  most months and retain a selected set of them.

**Reproducibility.** The pre-analysis plan now separates deterministic
forecast-only diagnostics, expected to reproduce the recorded figures, from
the synthetic-outcome experiments, whose code was lost and which can only be
rerun under a newly documented implementation and reported as a dated rerun.

**Documents synchronised.** [hypotheses.md](hypotheses.md) (evaluation
outcome, verdicts, H3 and H4, TODOs removed), the pre-analysis plan
("Estimation protocol" added, "Estimation details still to be fixed"
removed, reproducibility section revised) and
[research_proposal.md](research_proposal.md) (H4 and design wording).

**Final clarifications before freezing (same date, no change of
substance).** The pre-analysis plan's model-selection steps now describe
the implemented forecasting (OOS forecasts from the five bootstrap
replicas, no un-resampled refit); the bin-local basis constants are fixed
from the full $U_t$ of each cell; H2 marginal slopes are evaluated at
$X^c = 0$ (primary) and at the complete-case means (descriptive); H4 is
labelled a conditional observed-outcome comparison with its missingness
caveat; robustness analyses apply to H1 only and the H4 comparisons are
unadjusted; the reproducibility checkpoint states what blocks the outcome
analysis.
