# Pre-Analysis Plan

## Baseline sample

Formation months 1970-01 to 2024-12, inclusive (`configs/baseline.yaml`,
`data.sample`). The formation-time panel itself covers all CRSP months; the
sample restriction is applied when training and evaluation data are drawn
from it. The end point reflects the usable OSAP coverage of the pinned
October 2025 release; the baseline does not extend into 2025.

## Prediction target

The prediction horizon is one month. For each stock in $U_t$ (formation
month $t$), the realized return `ret_next` is the same PERMNO's CRSP CIZ
`MthRet` in calendar month $t+1$; the formation month 2024-12 uses 2025-01
returns. `MthRet` already includes the delisting return when CRSP provides
one, and no separate delisting return is compounded. Nothing else from $t+1$
is used: a stock that delists, changes exchange or leaves the universe during
$t+1$ keeps its formation observation and its $t+1$ return. Without a finite
`MthRet` for the same PERMNO in exactly $t+1$, `ret_next` is missing.

The primary training target `target_rank` ranks the finite `ret_next` values
within each formation month's $U_t$, with average ranks for ties, and maps
them to $[-1, 1]$:

$$Y_{i,t+1} = 2\,\frac{\operatorname{rank}_{i,t} - 1}{n_t - 1} - 1,$$

where $n_t$ is the number of stocks in $U_t$ with a finite `ret_next`
($Y = 0$ when $n_t = 1$). Stocks with a missing `ret_next` have a missing
target. Implementation: `disagreement.data.target`
(`python -m disagreement.data target`, output under
`data/processed/target/`).

## Predictor availability

The candidate predictors are the 212 predictors of the formation-time panel:
the 209 OSAP predictors, Price, Size and STreversal
([data_pipeline.md](data_pipeline.md)). Identifier and universe columns
(`permno`, `yyyymm`, `exchange`, `mktcap`) are never candidates.

A candidate is eligible in a training episode if both conditions hold
(`data.predictors`):

1. in a historical formation month, it is observed (non-missing, finite) for
   at least 60% of the stocks in $U_t$;
2. this monthly condition holds in at least 95% of the historical training
   months available to the episode.

Eligibility uses only formation months at or before the episode's training
cutoff; the episode's validation and OOS blocks are not inspected when
determining predictor eligibility. It depends
on predictor coverage alone: no return, target value, model prediction,
forecasting performance or H1–H4 result enters the rule. All four model
families use the same eligible predictors within a training episode.

Implementation: `disagreement.data.availability.eligible_predictors`, applied
to the history supplied for an episode.

### Rationale for the predictor-availability rule

The availability thresholds were fixed using predictor missingness only,
before examining return targets, model predictions, or empirical results. A
sensitivity analysis compared monthly coverage thresholds of 50%, 60%, 70%,
and 80% and historical-month requirements of 90%, 95%, and 100%. The 60% / 95%
rule provided a relatively stable compromise between retaining a broad
predictor set and requiring sustained cross-sectional availability.

The availability history was also compared with trailing 60-, 120-, and
180-month windows. Trailing windows admitted more later-introduced predictors
but produced substantially greater turnover in the eligible feature set across
annual training cutoffs. The expanding-history rule was therefore retained
because it provides a more stable and conservative predictor set over time, at
the acknowledged cost of favoring characteristics with longer data histories.

These choices were made solely from data-availability diagnostics and were not
selected using forecasting performance or subsequent-return results.

## Predictor preprocessing

Predictors are transformed month by month within $U_t$, each predictor
separately. In formation month $t$, the finite values of predictor $j$ are
average-ranked across stocks and mapped to $[-1, 1]$:

$$X^{*}_{i,j,t} = 2\,\frac{\operatorname{rank}_{i,j,t} - 1}{n_{j,t} - 1} - 1,$$

where $n_{j,t}$ is the number of finite values ($X^{*} = 0$ when
$n_{j,t} = 1$; a constant predictor maps to 0). Missing and non-finite values
take no part in the ranking and are set to 0 afterwards. No information from
other months, no outcome and no model result enters the transformation, so
the same rule applies to training, validation and out-of-sample months
without any fitted parameter. All four model families receive the same
transformed predictors. The predictors to transform are those eligible under
the availability rule at the episode's training cutoff. Implementation:
`disagreement.preprocessing.ranks.rank_predictors`.

### Rationale for the preprocessing rule

The characteristics have heterogeneous units and distributions. Cross-sectional
ranks place them on a common scale and reduce sensitivity to scale and extreme
magnitudes. Zero, the center of rank space, provides a common neutral value for
missing observations. The rule was fixed before any model was estimated or any
forecasting result was examined.

## Walk-forward design

The baseline out-of-sample (OOS) formation-month period runs from January
1985 through December 2024 in 40 annual episodes (`configs/baseline.yaml`,
`walk_forward`). For OOS year $Y$, in formation months:

- training: 1970-01 through $(Y-6)$-12, an expanding window;
- validation: $(Y-5)$-01 through $(Y-1)$-12, a rolling five-year window;
- OOS test: $Y$-01 through $Y$-12, twelve formation months, each used to
  forecast the following month's return.

The first episode trains on 1970-01–1979-12, validates on 1980-01–1984-12 and
tests on formation months 1985-01–1985-12. The final episode trains on
1970-01–2018-12, validates on 2019-01–2023-12 and tests on formation months
2024-01–2024-12. Within each annual episode, the training, validation, and OOS
formation-month blocks are chronological, contiguous, and mutually disjoint.
Across episodes, training and validation windows overlap by design, while the
OOS test blocks are mutually disjoint and together cover every formation month
from 1985-01 through 2024-12 exactly once. Because the target for formation
month $t$ is the return in $t+1$, these 480 OOS formation months correspond to
forecasted return months from 1985-02 through 2025-01.

Predictor eligibility is determined using the training block only, with
cutoff $(Y-6)$-12, and the resulting predictor set is fixed for the episode.

The validation block ends in December $Y-1$. Its final target is the
January-$Y$ return, which is known at the January-$Y$ formation date. The
annual model-selection/refit procedure can therefore use labeled formation
observations through December $Y-1$ before forecasting the February-$Y$ return
from January-$Y$ characteristics. No outcome associated with the current
episode's OOS formation months enters that episode's training, validation,
predictor eligibility or model selection. Implementation:
`disagreement.walkforward.calendar`.

### Rationale for the walk-forward design

Chronological blocks mimic genuine forecasting. The expanding training window
retains earlier information, while the rolling five-year validation window
reflects more recent conditions. Annual retraining limits computational cost
and keeps the specification fixed across the twelve OOS formation months of
each episode.

## Model selection

Hyperparameters are selected separately for each model family and annual
episode (`configs/baseline.yaml`, `model_selection`):

1. the eligible predictors are determined from the training block only
   (cutoff `train_end`) and stay fixed for the episode;
2. they are preprocessed month by month with the rank transformation above;
3. each candidate configuration is fitted on the training block only, with
   tuning seed 0 for stochastic fits, and predicts the five-year validation
   block;
4. the candidate with the highest validation score is selected; an exact tie
   keeps the first candidate in the configured order;
5. the selected hyperparameters are frozen for the episode;
6. the forecasts of the twelve OOS formation months are produced by the five
   month-cluster bootstrap replicas of "Out-of-sample replica forecasts",
   each fitted with those hyperparameters and the same predictor set on a
   resample of training and validation; no un-resampled refit is made and
   nothing is refitted within the year.

The validation score (`monthly_rank_ic`) is the equal-weighted mean, over
validation formation months, of the monthly rank information coefficient. In
each month, the predictions of the observations with a finite target are
average-ranked cross-sectionally, and $IC_t$ is the Pearson correlation
between these ranks and `target_rank`, that is, the Spearman correlation
between predictions and next-month returns:

$$IC_{\mathrm{val}} = \frac{1}{T_{\mathrm{val}}}\sum_t IC_t,
\qquad
IC_t = \operatorname{Corr}\bigl(\operatorname{rank}(\hat y_{i,t}), Y_{i,t+1}\bigr).$$

Higher is better. Constant predictions carry no ranking information and give
$IC_t = 0$. A validation month without any finite target, or whose target
does not vary, is an error. No raw-return, portfolio, H1–H4 or OOS result
enters model selection, and no outcome of the current episode's OOS formation
months is used. Later bootstrap replicas of a family share the
hyperparameters selected for that family and episode. Implementation: `disagreement.models.selection`.

Frozen candidate spaces (Cartesian products, in this order):

| Family | Fixed | Tuned | Candidates |
|---|---|---|---:|
| Elastic Net | — | `alpha` ∈ {1e-4, 1e-3, 1e-2, 1e-1}; `l1_ratio` ∈ {0.1, 0.5, 0.9} | 12 |
| Random Forest | `n_estimators` = 300 | `max_depth` ∈ {4, 8, 12}; `min_samples_leaf` ∈ {20, 100}; `max_features` ∈ {sqrt, 0.5} | 12 |
| LightGBM | `min_child_samples` = 100 | `n_estimators` ∈ {200, 500}; `learning_rate` ∈ {0.03, 0.10}; `num_leaves` ∈ {15, 31} | 8 |
| MLP | Adam; `batch_size` = 4096; `max_epochs` = 30; no early stopping | `hidden_layers` ∈ {(64), (64, 32)}; `learning_rate` ∈ {3e-4, 1e-3}; `weight_decay` ∈ {1e-5, 1e-3} | 8 |

All four families are regression models trained to predict `target_rank`
with squared-error loss: least squares with an elastic-net penalty (Elastic
Net), `criterion = "squared_error"` (Random Forest), the L2 `regression`
objective (LightGBM) and mean squared error (MLP). These training objectives
are distinct from the validation score above, which measures the monthly rank
correlation between predictions and `target_rank`. The search spaces are not expanded or
redefined after validation results are seen.

### Rationale for the model-selection rule

The validation criterion directly measures the model's ability to rank stocks
by subsequent returns, which is the downstream use of the forecasts. Computing
rank IC separately by month and weighting months equally prevents
cross-sectional sample size from driving model selection.

### Elastic Net

The Elastic Net family uses `sklearn.linear_model.ElasticNet` to predict
`target_rank` from the preprocessed predictors by minimizing
$\frac{1}{2n}\lVert y - Xw\rVert_2^2 + \alpha\,\rho\,\lVert w\rVert_1 + \frac{\alpha(1-\rho)}{2}\lVert w\rVert_2^2$,
with $\rho$ = `l1_ratio`. Only `alpha` and `l1_ratio` are tuned, over the grid
above. No additional predictor scaling is applied, since the predictors are
already monthly cross-sectional ranks. An unpenalized intercept is estimated,
and cyclic coordinate descent (`selection = "cyclic"`) keeps the fit
deterministic, with `max_iter = 10000` and `tol = 1e-6`. A convergence warning
is treated as a failure; no setting is changed to obtain convergence. The
validation criterion remains the monthly rank IC. Implementation:
`disagreement.models.elastic_net`.

### Random Forest

The Random Forest family uses `sklearn.ensemble.RandomForestRegressor` to
predict `target_rank` from the preprocessed predictors, with
`criterion = "squared_error"` and bootstrap sampling of the training
observations. `n_estimators = 300` is fixed; `max_depth`, `min_samples_leaf`
and `max_features` are tuned over the grid above, and the other parameters keep
their scikit-learn defaults. The tuning seed is fixed at 0
(`random_state = 0`) for every candidate, so the bootstrap samples and feature
draws are reproducible. Fitting uses `n_jobs = -1` for computational
efficiency, while prediction runs serially (`n_jobs = 1`) so that the tree
predictions are always aggregated in the same order and are reproducible bit
for bit; this computational choice does not alter the fitted forest or the
statistical specification. No additional preprocessing is applied, and the
trees' native missing-value handling is not used since missing predictors are
already set to 0. The validation criterion remains the equal-weighted mean
monthly rank IC. Implementation: `disagreement.models.random_forest`.

### LightGBM

The LightGBM family uses `lightgbm.LGBMRegressor` to predict `target_rank`
from the preprocessed predictors, with the L2 regression objective
(`objective = "regression"`) and standard GBDT.
`min_child_samples = 100` is fixed; `n_estimators`, `learning_rate` and
`num_leaves` are tuned over the grid above, and the other parameters keep their
LightGBM defaults. No evaluation set, early-stopping callback, or other
early-stopping mechanism is used; the candidate's `n_estimators` is passed
directly to LightGBM. The realized number of fitted boosting iterations may
nevertheless be smaller if LightGBM cannot construct additional trees. CPU
deterministic mode is enabled (`device_type = "cpu"`, `deterministic = True`,
`force_col_wise = True`) with the tuning seed 0 (`random_state = 0`), so
repeated fits are reproducible within a fixed software environment. Fitting is
parallelized (`n_jobs = -1`), while predictions are made with a single thread.
No additional preprocessing is applied, and LightGBM's native missing-value
handling is not used since missing predictors are already set to 0. Model
selection remains based on the equal-weighted mean monthly rank IC.
Implementation: `disagreement.models.lightgbm`.

### MLP

The MLP family is a fully connected feed-forward network implemented in
PyTorch that predicts `target_rank` from the preprocessed predictors. Each
hidden layer is a linear layer followed by a ReLU activation, and the output is
a single linear unit without activation; the hidden layers are `[64]` or
`[64, 32]`. Weights use PyTorch's default initialization, and there is no
dropout or batch normalization. Training minimizes the mean squared error
with Adam (`betas = (0.9, 0.999)`, `eps = 1e-8`) and standard coupled weight
decay, not AdamW; the learning rate and the weight decay are tuned over the
grid above. Each candidate trains for exactly 30 epochs with
`batch_size = 4096`, without early stopping, learning-rate schedule or
gradient clipping; every epoch visits each observation once, in a new
permutation, and keeps the last, smaller mini-batch. The tuning seed 0 fixes
the initialization and the sequence of epoch permutations, and training and
prediction run on the CPU in float32 with PyTorch's deterministic algorithms,
so repeated runs are reproducible within a fixed software and hardware
environment. No additional preprocessing is applied. Model selection uses the
equal-weighted mean monthly rank IC, not the training loss. Implementation:
`disagreement.models.mlp`.

## Month-cluster bootstrap

The bootstrap resamples formation months, not individual stock-month
observations. If the historical estimation sample contains $M$ distinct
formation months, $M$ months are drawn with replacement from them. Each
selected month contributes its complete stock cross-section; a month drawn
several times contributes its complete cross-section that many times, and a
month that is not drawn contributes no rows. The resampling operates on
already preprocessed, model-ready observations: predictor ranks and
`target_rank` are not recomputed within a bootstrap sample. The draws are
deterministic given the supplied seed. Implementation:
`disagreement.bootstrap.months.month_cluster_bootstrap`.

## Out-of-sample replica forecasts

For each annual episode and model family, hyperparameters are selected once,
on training → validation, as described above: the eligible predictors come
from the training block only, every candidate is fitted on the original
training block (observations with a finite `target_rank`) and scored on the
validation block, and the selected hyperparameters are then frozen. Five
replicas (`configs/baseline.yaml`, `bootstrap`) are each trained on a
month-cluster bootstrap of the whole pre-OOS history, training and validation,
drawn from all of its formation months; resampled observations without a
finite `target_rank` are left out of the fit only. The bootstrap seed of a
replica is derived from the base seed 0, the OOS year and the replica number
(`numpy.random.SeedSequence`), so within a given year and replica the four
families resample the same months. Model-internal randomness stays fixed at
the tuning seed 0, so the variation across replicas comes from resampling
historical months. Each replica predicts every stock of the twelve OOS
formation months without refitting within the year. The raw predictions are
stored, one file per family and OOS year, and are not ranked at this stage.
Targets are read only through December $Y-1$, so no outcome of the current
OOS year enters estimation or selection. Implementation:
`disagreement.walkforward.forecast`
(`python -m disagreement.walkforward.forecast`, output under
`data/processed/oos_predictions/`).

## Family signals

The raw OOS predictions are turned into ranks month by month, separately for
each model family. Within each OOS formation month, the predictions of each
replica are ranked across stocks with average ranks for ties and mapped to
$[0, 1]$,

$$q^{(r)}_{i,f,t} = \frac{\operatorname{rank}^{(r)}_{i,f,t} - 1}{n_t - 1},$$

with $q = 0.5$ when $n_t = 1$; a constant cross-section maps to 0.5. Each
stock's mean replica rank $\bar q_{i,f,t}$ averages its five replica ranks,
and the family rank re-ranks $\bar q_{i,f,t}$ across the month's stocks with
the same convention. Raw predictions are never averaged across replicas,
ranks never cross months, replicas or families, and no target or return is
used. The output keeps one row per replica, with the replica rank, the mean
replica rank and the family rank. Implementation:
`disagreement.signals.family` (`python -m disagreement.signals.family`,
output under `data/processed/family_signals/`).

## Ensemble signal

For each stock $i$ and OOS formation month $t$, the ensemble signal is the
equal-weight average of the four family ranks,

$$S_{i,t} = \frac{1}{4}\left(R^{EN}_{i,t} + R^{RF}_{i,t} + R^{LGBM}_{i,t} + R^{MLP}_{i,t}\right),$$

where $R^{f}_{i,t}$ is the family rank of family $f$, on $[0, 1]$. $S_{i,t}$
is not re-ranked, centred or rescaled, so it lies in $[0, 1]$ but is not
itself a cross-sectional rank. Raw predictions, replica ranks and mean replica
ranks are not averaged across families, and no target or return is used. The
ensemble is defined only where all four families are present: the four
families must cover exactly the same stock-months, spanning all twelve months
of the OOS year, and a stock-month missing from any family is an error rather
than dropped. The output has one row per
stock-month, with the four family ranks and $S_{i,t}$. Implementation:
`disagreement.signals.ensemble` (`python -m disagreement.signals.ensemble`,
output under `data/processed/ensemble_signals/`).

## Between-family disagreement

For each stock $i$ and OOS formation month $t$, the raw between-family
disagreement is the dispersion of the four family ranks around the ensemble
signal,

$$D^{B,\mathrm{raw}}_{i,t} = \sqrt{\frac{1}{4}\sum_f \left(R^{f}_{i,t} - S_{i,t}\right)^2},$$

the population standard deviation (`ddof = 0`): the four model families are
the complete fixed set of the design, not a sample of families. Since every
family rank lies in $[0, 1]$, $D^{B,\mathrm{raw}}_{i,t}$ lies in $[0, 0.5]$;
it is 0 when the four families agree and 0.5 when two rank the stock at 0 and
two at 1. It is not re-ranked and not yet conditioned on the ensemble signal,
and no target or return is used. The ensemble input is checked first: twelve
OOS months, ranks in $[0, 1]$, and a saved $S_{i,t}$ equal to the mean of its
four family ranks. The output has one row per stock-month, with $S_{i,t}$ and
$D^{B,\mathrm{raw}}_{i,t}$. Implementation: `disagreement.signals.between`
(`python -m disagreement.signals.between`, output under
`data/processed/between_family_disagreement/`).

## Within-family estimation uncertainty

For each model family $f$, stock $i$ and OOS formation month $t$, the
estimation uncertainty is the dispersion of the stock's $K$ replica ranks
$q^{(r)}_{i,f,t}$ ($K = 5$ bootstrap replicas in the baseline),

$$D^{W,f}_{i,t} = \sqrt{\frac{1}{K-1}\sum_{r=1}^{K}\left(q^{(r)}_{i,f,t} - \bar q_{i,f,t}\right)^2},$$

the sample standard deviation (`ddof = 1`): unlike the four fixed families
of the between-family disagreement, the replicas are bootstrap draws that
approximate estimation uncertainty. It is computed from the saved replica
ranks, not from raw predictions, mean replica ranks or family ranks. Each
family's $D^{W,f}_{i,t}$ is then ranked across the month's stocks, separately
for each family and month, with average ranks for ties and the convention of
the family signals,

$$R^{W,f}_{i,t} = \frac{\operatorname{rank}(D^{W,f}_{i,t}) - 1}{n_t - 1} \in [0, 1],$$

with 0.5 when $n_t = 1$ or when the month's dispersions are all equal. The
raw within-family uncertainty is the equal-weight mean of the four family
uncertainty ranks,

$$W^{\mathrm{raw}}_{i,t} = \frac{1}{4}\left(R^{W,EN}_{i,t} + R^{W,RF}_{i,t} + R^{W,LGBM}_{i,t} + R^{W,MLP}_{i,t}\right) \in [0, 1].$$

It is not re-ranked and not yet conditioned on the ensemble signal, and no
target or return is used. The four families must cover exactly the same
stock-months over the twelve OOS months. The output has one row per
stock-month, with the four $D^{W,f}_{i,t}$, the four $R^{W,f}_{i,t}$ and
$W^{\mathrm{raw}}_{i,t}$. Implementation: `disagreement.signals.within`
(`python -m disagreement.signals.within`, output under
`data/processed/within_family_uncertainty/`).

## Conditional disagreement

Disagreement and uncertainty are compared only among stocks with similar
ensemble signals. Within each OOS formation month separately, stocks are
sorted into $B = 20$ signal bins by the ensemble signal alone. With
$r_{i,t}$ the average rank of $S_{i,t}$ among the month's $n_t$ stocks,

$$\mathit{bin}_{i,t} = 1 + \left\lfloor 20\,\frac{r_{i,t} - 1}{n_t} \right\rfloor \in \{1, \dots, 20\},$$

from 1 for the lowest signals to 20 for the highest. Stocks with exactly the
same ensemble signal share their average rank and therefore their bin: ties
are never split by row order, identifier or randomisation, so the bins are
approximately rather than exactly equal-frequency when signals tie, and as
even as possible when they do not. Bins never pool months or use the
individual family ranks.

Inside each (month, signal bin) group, the raw between-family disagreement
$D^{B,\mathrm{raw}}_{i,t}$ is ranked across stocks with average ranks for
ties and mapped to $[0, 1]$,

$$CD^{B}_{i,t} = \frac{\operatorname{rank}(D^{B,\mathrm{raw}}_{i,t}) - 1}{n_{b,t} - 1},$$

with 0.5 for a single stock or a constant group. The raw within-family
uncertainty $W^{\mathrm{raw}}_{i,t}$ is ranked separately in exactly the same
groups and in the same way, giving $CW_{i,t}$. Both conditional variables
lie in $[0, 1]$; they are the primary disagreement and uncertainty measures
of the empirical specifications. No target or return is used, and no
regression or diagnostic is run at this stage. The output has one row per
stock-month, with $S_{i,t}$, the signal bin, both raw measures and both
conditional measures. Implementation: `disagreement.signals.conditional`
(`python -m disagreement.signals.conditional`, output under
`data/processed/conditional_disagreement/`).

## Mechanical-bias diagnostics

Because the four family ranks lie in $[0, 1]$ and $S_{i,t}$ is their mean,
their dispersion is mechanically limited when $S_{i,t}$ is close to 0 or 1.
A descriptive diagnostic checks how far the signal-bin conditioning removes
this relation. For diagnosis only, the ensemble signal is centred,
$S^c_{i,t} = 2(S_{i,t} - 0.5)$, and its extremity is $|S^c_{i,t}|$; neither
is stored with the signals.

In every OOS formation month separately, four Pearson correlations are
computed over the month's stocks: $D^{B,\mathrm{raw}}$ with $S$,
$D^{B,\mathrm{raw}}$ with $|S^c|$, $CD^B$ with $S$, and $CD^B$ with $|S^c|$.
Each monthly series is summarised with equal month weights (number of finite
months, mean, median, mean absolute value, 5th and 95th percentiles). The
same four correlations are computed inside every (month, signal bin), and
the bins are profiled with equal month weights: the mean over months of the
monthly bin means of $S$, $|S^c|$, $D^{B,\mathrm{raw}}$ and $CD^B$, and the
mean, median and mean absolute value of each within-bin correlation. Bins 1
and 20, where the mechanical bound binds most, are the ones to inspect, but
every bin is computed in the same way. A correlation with fewer than two
stocks or a constant variable is undefined and left missing, never set to
0; summaries use the finite correlations only.

Only the conditional-disagreement files are read, and their signal bins are
checked against the ensemble signal; no target or return is used. There is
no significance test, threshold or pass/fail rule, and the diagnostic does
not change the design: any methodological decision it prompts is taken
separately and explicitly, as in the next section. Implementation:
`disagreement.diagnostics.mechanical_bias`
(`python -m disagreement.diagnostics.mechanical_bias`, over the complete
baseline OOS sample, output under `data/diagnostics/mechanical_bias/`).

## Signal controls in the H1 specification

This section records a pre-results amendment of 2026-11-09 to the H1
specification of [hypotheses.md](hypotheses.md); the change history is in
[methodology_changes.md](methodology_changes.md). All evidence below comes from
the saved OOS signals and model metadata, from mathematical analysis and from
experiments with fully synthetic outcomes. No realized future return and no
H1–H4 result was examined.

### Diagnostic evidence (saved signals only)

Over the 480 baseline OOS months, the signal-bin conditioning removes the
relation between disagreement and the signal in the full monthly cross-section
(mean monthly correlation of $CD^B$ with $|S^c|$: −0.017) but not inside the
extreme bins: the equal-weight mean within-bin correlation of $CD^B$ with $S$
is about +0.65 in bin 1 and −0.66 in bin 20, with the same sign in every
month except the twelve 2021 months, where it reverses (see "Elastic Net in
2021" below), and about +0.21 and −0.21 in bins 2 and 19. Within a bin,
$E[CD^B \mid S]$ is close to linear in $S$, with slopes of roughly +9 in bin 1
and −8 in bin 20 and near 0 in the central bins. The conditional within-family
uncertainty $CW$ shows the same pattern (about +0.72 and −0.69). Finer
conditioning does not remove it: with 30 or 40 bins the extreme-bin
correlations stay at about ±0.66 and the dependence moves into the narrower
boundary bins. The pattern is a boundary property of bounded ranks, not a
consequence of the bin width.

### Consequence for H1

Write $CD^B = m(S) + u$, with $m(S) = E[CD^B \mid S]$ inside the bin. The
interaction regressor is $S^c \times CD^B = S^c\,m(S) + S^c\,u$. The first
term is a function of the signal alone, quadratic within each bin when $m$ is
linear there. Under a nonlinear signal–return relation and no true
disagreement effect, signal controls that do not span this term let it load
on the interaction coefficient. The sign of this bias is not fixed by the
geometry of $m$ alone: it depends on how the unspanned part of the
signal–return relation covaries with $S^c\,m(S)$, so it can take either sign.
Under the synthetic signal–return relations used in the experiments below,
with $m$ increasing in the low bins and decreasing in the high bins, the bias
was negative, in the pre-specified H1 direction; this is a property of those
scenarios, not a general result. Controls that are bin-specific in $S$ absorb
the term whatever the sign. Global smooth controls, in $|S^c|$ or in splines
of $S$ or of its rank, do not, because the term has a bin-dependent,
non-smooth structure.

### Specification comparison

Candidates, each estimated month by month:

- A: the original specification, with the global controls $|S^c|$ and
  $S^c \times |S^c|$;
- B: (month, signal bin) intercepts and bin-specific linear terms in $S^c$;
- C: B plus bin-specific quadratic terms in $S^c$;
- D: a natural cubic spline of the re-ranked signal with seven fixed knots,
  $CD^B$ residualised on the spline, and the interaction of that residual
  with the re-ranked signal;
- E: a two-step version in which $CD^B$ is first detrended linearly on $S$
  within each (month, signal bin). With the controls of C this estimator is
  algebraically identical to C (Frisch–Waugh–Lovell, verified numerically;
  exact conditions in "Reproducibility of the evidence"); with the controls
  of B only it is a distinct specification, close to B and C but identical
  to neither.

Diagnostics on the saved signals: after partialling out each candidate's other
regressors, the interaction regressor of A and D keeps a within-bin
correlation of about −0.65 with $S$ in bins 1 and 20, and about half of its
remaining variance in those bins is a function of the signal alone; for B
this share is about 5–6% against a noise floor of about 2%, and for C it is
zero (verified on 2026-11-09, "Reproducibility of the evidence"). The spline
of D removes none of the bin-dependent structure of $CD^B$. C retains about
98.15% of the identifying interaction variance of B, an estimated
standard-error cost of about 0.9%.

Experiments with fully synthetic outcomes, using the saved $S$, $CD^B$ and
bins as fixed regressors and independent homoskedastic noise: under a
boundary-nonlinearity scenario with no true disagreement interaction,
two-sided 5% tests rejected in about 18.3% of replications for A, 5.7% for
B, 5.3% for C and 20.7% for D, and the spurious rejections of A and D were
almost all in the pre-specified direction. In a targeted stress scenario with
strong within-bin curvature of opposite sign on the two sides of $S = 0.5$, B
retained a bias approaching one standard error, while C absorbed the
quadratic nuisance component exactly. These experiments compare
specifications; they are not a guarantee of nominal inference under the
dependence of actual equity returns.

Decision: C is the primary H1 specification and B is the pre-specified
specification robustness. The spline/re-ranked-signal approach is rejected,
and the ensemble signal is not re-ranked: re-ranking changes the estimand
without removing the bin-level dependence. The definitions of $S$, $CD^B$,
$CW$ and the 20 signal bins are unchanged.

### Elastic Net in 2021

In the 2021 episode the Elastic Net family produced constant forecasts in all
twelve OOS formation months: every candidate of the frozen grid had a negative
validation rank IC except the two most penalised ones, which fit the intercept
only, score exactly 0 and were selected under the frozen rule. This is a
legitimate outcome of the protocol, verified from the saved metadata and
forecasts; no other family-year is affected. The ranking convention assigns a
family rank of 0.5 to a constant cross-section. With one family fixed at 0.5,
$S = (0.5 + 3\bar S_3)/4$ and
$(D^{B,\mathrm{raw}})^2 = \tfrac{3}{4}\,D_3^2 + \tfrac{3}{16}\,(\bar S_3 - 0.5)^2$,
where $\bar S_3$ and $D_3$ are the mean and the population dispersion of the
three active family ranks: the feasible geometry of four-family disagreement
changes, and the usual within-bin dependence of $CD^B$ on $S$ reverses at the
extremes. In 2021 the mean within-bin correlation of $CD^B$ with $S$ was about
−0.947 in bin 1 and +0.788 in bin 20, against about +0.674 and −0.629 for a
three-family measure built from the active families, and about +0.67 and
−0.67 in 2020 and 2022. The signal ordering, the bins and $CW$ are the same
under both constructions up to ties.

Decision: the fixed four-family design and the saved 2021 outputs are kept as
primary; no inactive-family rule is introduced, and the hyperparameter
selection rule is unchanged. The bin-specific signal controls of C absorb a
systematic dependence on signal location whatever its sign, but they do not
make the four-family and three-family disagreement rankings equivalent, so the
primary H1 test is repeated without the twelve 2021 formation months as a
robustness analysis. No claim about the effect of 2021 on future returns or on
the sign of any H1 estimate is made.

### Reproducibility of the evidence

The evidence above is regenerated by one committed module, a sibling of
`disagreement.diagnostics.mechanical_bias`:
`python -m disagreement.diagnostics.design_evidence` (about 90 seconds;
outputs under `data/diagnostics/design_evidence/`, not versioned, with
`summary.json` written last as the completion marker). It reads the saved
conditional disagreement, ensemble signals, H2 controls, model-selection
metadata and, for a family-year with constant ranks, the raw OOS
predictions; it writes aggregates only, records the SHA-256 of every file
read and of the configuration, and its outputs are removed whenever a
conditional-disagreement or H2-control year is rebuilt. It is tested on
synthetic signals (`tests/test_design_evidence.py`) and keeps two categories
apart.

*Deterministic forecast-only diagnostics* depend only on the saved signals
and are expected to reproduce the recorded figures within rounding
(correlations ±0.005, shares ±0.05 percentage point). Run of 2026-11-09:
reproduced within tolerance are the within-bin correlations of $CD^B$ with
$S$ (+0.653, +0.209, −0.213, −0.661 in bins 1, 2, 19, 20), their slopes
(+9.02, −8.33), the $CW$ correlations (+0.721, −0.689), the 30- and 40-bin
extremes (±0.66), the partialled interaction correlations of A and D in
bin 1 (−0.650, −0.646) and of D in bin 20 (−0.655), the identifying
variance of C relative to B (98.159% pooled, 98.152% as the mean of monthly
shares) and its standard-error cost (0.93%), the Elastic Net 2021 figures
(constant forecasts in all 60 month-replica cross-sections, the two
intercept-only candidates scoring exactly 0 and the first selected, $CD^B$
correlations −0.947 and +0.788, three-family +0.674 and −0.629, the twelve
2021 months being the only ones with reversed extreme-bin signs, the
identities on $S$ and $(D^{B,\mathrm{raw}})^2$ exact to $10^{-16}$), the
2020 bin-1 value (+0.670), the minimum cell sizes (86; 52 complete cases),
the scaled condition numbers (C in the original basis about
$3.7 \times 10^3$, in the local basis 4.1 and at most 5.2; the
complete-case H2 design at most 9.0) and the basis invariance of $\beta_2$
and $\beta_3$ ($1.5 \times 10^{-14}$). Two diagnostics cited in
[methodology_changes.md](methodology_changes.md) without a recorded value
are now computed: the mean monthly correlation of $CD^B$ with $CW$ is
−0.017 (mean absolute value 0.046; +0.34 for the raw measures), and on the
complete-case sample the H2 controls leave 96.5% of C's identifying
interaction variance (monthly minimum 87.7%; variance inflation 1.04, at
most 1.14), the complete-case restriction itself keeping 93.8% of the
full-sample identifying variance. Outside tolerance, listed by the module
as resolved discrepancies (no unresolved failure, checkpoint passed) because
they are coarse summaries that bore on no decision: the recorded
"about −0.65" for the partialled correlations of A and D in bins 1 and 20
is one rounded
value for four cells (−0.650, −0.659, −0.646, −0.655; A in bin 20 is 0.009
away), and "about +0.67 and −0.67 in 2020 and 2022" is a rounded two-year
summary of +0.670/−0.694 (2020) and +0.682/−0.644 (2022), +0.676/−0.669
pooled. Two statements are made precise rather than reproduced. The
two-step specification E coincides with C only when $CD^B$ is detrended on
a basis $V$ such that $V$ and $S^c V$ lie in the span of C's controls
(demeaning or linear detrending within the cell, with the detrended level
term kept); detrending on C's own quadratic controls is not identical to C
($S^c$ times the fitted quadratic has a cubic part outside the span;
difference about 0.7% of $\beta_3$ on the check outcome), nor is linear
detrending followed by B's controls (distinct from B and C). The within-bin
signal-variance share of B's partialled interaction in bins 1 and 20 is
5–6%, above the 2% noise floor rather than at it; C's is exactly 0. The
recorded figures are kept as recorded; these clarifications are logged in
[methodology_changes.md](methodology_changes.md).

*Synthetic-outcome experiments* cannot be replicated exactly: their code
is lost, and the recorded seed does not fix the order of the draws in a
new implementation. The module reruns them under the definitions below
and reports the results as a dated rerun, never as a replication of the
18.3%, 5.7%, 5.3% and 20.7% figures. Rerun of 2026-11-09 (300
replications, `default_rng(20261109)`, one noise vector per replication
shared by every scenario and specification, the recorded Fama–MacBeth
convention): boundary scenario A 17.3%, B 4.0%, C 4.3%, D 18.7%
(Monte-Carlo standard errors 1.1–2.2 points; 19.0%, 5.3%, 5.7% and 19.7%
with the frozen HAC $L = 6$); cubic $g$: A 9.0%, B 4.3%, C 4.3%, D 17.7%;
tanh $g$: A 7.7%, B 4.0%, C 4.3%, D 4.3%; linear $g$: all between 4.0% and
6.0%; the spurious rejections of A and D were almost all negative (all of
them in the boundary scenario); power 100% for every specification at
$d = \pm 0.01$. Stress scenario (noise-free, biases in units of the
reference standard error 0.00076): B's bias −1.00, C's −0.0003 (that is
$-2 \times 10^{-7}$ in coefficient units). The new rates agree with the
recorded ones within Monte-Carlo error and meet the qualitative
expectation (A and D well above 5%, B and C near 5%).
Limitations of these scenarios: the boundary kink sits at the monthly 5th
and 95th percentiles of $S$, which are the edges of bins 1 and 20, so $g$ is
linear inside every bin and both B and C span it exactly; the scenarios
isolate the mechanism (a signal-only term outside the span of A and D) and
do not probe curvature inside a bin. The stress scenario uses C's own
quadratic basis function, so it shows what B leaves unabsorbed and what C
absorbs by construction, not a general robustness ranking. The noise scale
(0.10) and the slopes were not calibrated to the outcome, so only the
bias-to-standard-error ratios and the relative rejection rates are
informative; the 100% power at $d = \pm 0.01$ says nothing about real power.

Definitions implemented by the module:

- Regressors: the saved $S$, $CD^B$ and 20 signal bins over the 480 baseline
  months, used as fixed. Bin-specific terms use $S^c$ demeaned within
  (month, bin), which spans the same space as raw $S^c$.
- Specification D: natural cubic spline basis with seven knots at the 0.05,
  0.20, 0.35, 0.50, 0.65, 0.80 and 0.95 quantiles of the within-month unit
  rank of $S$; $CD^B$ residualised on the basis; interaction of that residual
  with the centred rank. Variant D_S: the same basis on $S^c$ with monthly
  quantile knots.
- Synthetic outcomes: $y = g(S) + d\,S^c\,CD^B + \varepsilon$ with
  $\varepsilon \sim N(0, 0.10^2)$ i.i.d., `numpy.random.default_rng(20261109)`,
  300 replications, common random numbers across scenarios and
  specifications. Scenarios with $d = 0$: $g = 0.01\,S^c$;
  $g = 0.01\,S^c + 0.015\,(S^c)^3$; $g = 0.012\tanh(2.5\,S^c)$; boundary
  nonlinearity
  $g = 0.01\,S^c + 0.05\max(S - q_{95}, 0) - 0.05\max(q_{05} - S, 0)$ with
  monthly 5th and 95th percentiles of $S$. Scenarios with $d = -0.01$ (linear
  and cubic $g$) and $d = +0.01$ (linear $g$).
- Inference in the experiments: monthly OLS, Fama–MacBeth mean of the 480
  interaction coefficients, standard error equal to their standard deviation
  divided by $\sqrt{480}$, rejection when $|t| > 1.96$. The module computes
  each monthly coefficient as the Frisch–Waugh–Lovell linear functional of
  the outcome, so an estimate is its noise-free part plus the common noise
  projection; the frozen HAC $L = 6$ rates are reported alongside.
- Stress scenario (noise-free exact bias):
  $g = 0.01\,S^c + 0.01\,\operatorname{sign}(S^c)\,(\tilde S / h_b)^2$, where
  $\tilde S$ is $S^c$ demeaned within (month, bin) and $h_b$ is half the bin's
  range of $S^c$; the Fama–MacBeth standard-error reference was about 0.00078.
- Elastic Net 2021: validation scores and selected candidate index read from
  the saved model-selection metadata; the three-family measure is the
  population standard deviation of the Random Forest, LightGBM and MLP ranks,
  ranked within the existing (month, bin) cells.

Decision status. A failed deterministic check (a forecast-only figure
outside its tolerance, an integrity or provenance failure) or a material
mathematical inconsistency blocks the outcome analysis until it is resolved
and recorded in [methodology_changes.md](methodology_changes.md). The module
keeps the recorded values and tolerances and fails closed: a fixed list of
expected checks must all be computed; an entry outside its tolerance counts
as resolved only if it matches, within 0.0005, the reconstructed value
recorded for it on 2026-11-09; a missing, undefined or materially changed
value is an unresolved failure, and the checkpoint is reported as passed or
blocked accordingly. The rerun
synthetic-outcome results are disclosed as new simulations: neither a
reproduction of the recorded figures nor an automatic pass/fail gate. If
they seriously challenge the validity of the frozen specification (for
example C rejecting far above 5% with no true effect), the cause is
investigated and any amendment is documented with its evidence before any
real outcome is inspected; the protocol is never changed silently on the
basis of simulations.

## H2 control characteristics

H2 ([hypotheses.md](hypotheses.md)) extends the primary H1 specification C
with three stock characteristics, each entering as a level term and
interacted with the original centered signal $S^c$. This section fixes their
definitions (amendment of 2026-11-09, recorded in
[methodology_changes.md](methodology_changes.md)). The characteristics are
taken from data the repository already holds: the universe's market
capitalisation and two OSAP predictors of the formation-time panel. No new
WRDS extraction is required, and no daily CRSP data are used directly.

### Sources and upstream reference

- Market capitalisation: the universe column `mktcap`, the CRSP CIZ monthly
  field `MthCap` of formation month $t$ (security level, thousands of
  dollars), finite and strictly positive for every member of $U_t$ by
  construction ([data_pipeline.md](data_pipeline.md)).
- `RealizedVol` and `Illiquidity`: columns of the formation-time panel,
  from the OSAP October 2025 release, version 2.0.0 (SHA-256 pinned in
  `configs/baseline.yaml`). The methodological reference for their
  construction is the OSAP code at the GitHub tag `v2.0.0` of
  `OpenSourceAP/CrossSection` (commit
  `b4e911e69678a7424f318617a61d813f54183123`, release published
  2025-10-22): `Signals/pyCode/Predictors/ZZ0_RealizedVol_IdioVol3F_ReturnSkew3F.py`,
  `Signals/pyCode/Predictors/Illiquidity.py`,
  `Signals/pyCode/utils/save_standardized.py` and
  `Signals/pyCode/DataDownloads/CRSPMonthly.py`. That release builds the
  signals in Python from the legacy CRSP daily and monthly files; the later
  code on the default branch differs for `Illiquidity.py` and is not the
  reference. Both predictors are dated by OSAP at the calendar month of the
  data they use, with no lag in the construction or saving code, and the
  panel joins them to formation month $t$ without an additional lag.
- The panel's own `Size` predictor is not used. In the reference code it is
  $\ln(\texttt{mve\_c})$ with `mve_c` = shares outstanding × |price| from
  the legacy monthly file (millions of dollars) at the PERMNO level; the
  firm-level aggregate `mve_permco` is a separate column that `Size.py` does
  not read, but the intermediate master table feeding `Size.py` was not
  verified. The universe's `mktcap` is the CIZ security-level
  capitalisation. The two are conceptually close but are not treated as
  identical; the control is defined from `mktcap` only.

### Definitions

Economic values, before transformation. The OSAP columns are stored signed so
that higher values predict higher returns (sign −1 for `RealizedVol`, +1 for
`Illiquidity`); the economic value reverses the first sign and keeps the
second.

1. **Size:** $\mathrm{SIZE}_{i,t} = \ln(\texttt{mktcap}_{i,t})$. Units do
   not matter: a constant unit conversion shifts $\ln(\texttt{mktcap})$ by a
   constant within the month and leaves the monthly rank unchanged.
2. **Realized volatility:**
   $\mathrm{RVOL}_{i,t} = -\,\texttt{RealizedVol}_{i,t}$. In the reference code,
   daily CRSP returns are inner-joined to the daily
   Fama–French factor file (days without factor data are dropped), the
   excess return $r_d - r_{f,d}$ is formed, days are grouped by calendar
   month, a stock-month is kept only if it has at least 15 excess-return
   observations, an OLS regression of the excess return on the three
   Fama–French factors with intercept is run within the stock-month and days
   whose residual is missing are removed; `RealizedVol` is then the sample
   standard deviation (one degree of freedom lost) of the remaining daily
   *excess returns*, not of the residuals (the residual standard deviation
   is the separate predictor `IdioVol3F`, which is not used). No
   annualisation or scaling is applied. The 15-observation requirement is
   checked before the residual filter, so a stock-month keeps its value even
   if fewer than 15 residuals are valid. The OSAP signal documentation
   describes `RealizedVol` as a CAPM-residual volatility; the code is the
   reference and it computes the total excess-return volatility.
3. **Illiquidity:** $\mathrm{ILLIQ}_{i,t} = +\,\texttt{Illiquidity}_{i,t}$.
   In the reference code the daily ratio
   $\lvert r_d\rvert / (\lvert p_d\rvert \times \text{vol}_d)$ is
   computed from the legacy daily file, infinite values (zero volume) are
   set to missing, the ratio is averaged over the available days of each
   calendar month (no minimum number of days), and the monthly series is
   lagged 1 to 11 times by plain row shifts within the PERMNO over the
   months that exist in the daily data. `Illiquidity` in month $t$ is the
   mean of the current and the eleven lagged monthly means when all twelve
   are non-missing, and missing otherwise. The twelve observations are
   therefore the twelve most recent *available* monthly observations, not
   necessarily twelve consecutive calendar months. No price or exchange
   filter, scaling or NASDAQ volume adjustment is applied. Volume and price
   units are those of the legacy CRSP daily file; they are constant across
   stocks within a month and do not affect the rank.

### Timing

All three characteristics use information available no later than the end
of formation month $t$: `mktcap` is the month-$t$ record, `RealizedVol` uses
the trading days of month $t$, and `Illiquidity` uses months up to and
including $t$. No month-$t+1$ information enters.

### Transformation

Each economic characteristic $X \in \{\mathrm{SIZE}, \mathrm{RVOL},
\mathrm{ILLIQ}\}$ is transformed separately, in each formation month $t$,
among the members of $U_t$ with a finite value of $X$:

$$
X^c_{i,t} = 2\,\frac{\operatorname{rank}_{i,t}(X) - 1}{n^X_t - 1} - 1
\in [-1, 1],
$$

where $\operatorname{rank}$ is the average rank for ties and $n^X_t$ is the
number of finite values of $X$ in $U_t$ that month. A constant $X$ gives
every stock the middle rank and $X^c = 0$; a month with $n^X_t = 1$ gives
$X^c = 0$. This is the convention of `unit_rank` in
`disagreement.signals.family`, centred to $[-1, 1]$ like $S^c$. Non-finite
values are not ranked and are not imputed; they stay missing. The ranking
population is all finite observations in $U_t$, before any restriction to
the H2 estimation sample, so the ranks of the complete-case sample are not
recomputed. No winsorisation or logarithm is needed after ranking; the
logarithm in the size definition is kept for interpretability of the
economic variable only.

### H2 regressors

The H2 specification is specification C with, in addition, the three level
terms $X^c_{i,t}$ and the three interactions $S^c_{i,t} \times X^c_{i,t}$,
estimated month by month in the same Fama–MacBeth framework. $CD^B$, the
interaction $S^c \times CD^B$ and the bin-specific signal controls are
unchanged, and $\beta_{\text{interaction}}$ remains the coefficient of
interest. Within-family uncertainty $CW$ is not a control.

### Estimation sample and missing values

- No imputation. The H2 sample, the treatment of missing outcomes and the
  re-estimation of specification C on the exact H2 sample are fixed in
  "Estimation protocol"; $U_t$, $S$, $CD^B$, $CW$ and the bins keep their
  full-cross-section values.
- Coverage of the characteristics among $U_t$ over the 480 OOS formation
  months 1985-01 to 2024-12 (share of members with a finite value, from the
  panel's coverage diagnostics): `mktcap` 1.000 in every month;
  `RealizedVol` mean 0.997, minimum 0.985; `Illiquidity` mean 0.954,
  minimum 0.863, below 0.90 in 18 months (1996-05/06, 1999-09 to 2000-10,
  2021-07/08).

### Known limitations

- NASDAQ volume before 2004 is reported differently from NYSE/AMEX volume
  (dealer-market double counting), which understates the Amihud ratio of
  NASDAQ stocks relative to other stocks in the same month; the reference
  code applies no adjustment. Monthly ranking is invariant to a common scale
  but not to an exchange-specific one, so it does not necessarily remove
  this distortion.
- The twelve OSAP observations behind `Illiquidity` are twelve available
  monthly observations, not necessarily twelve consecutive calendar months.
- Missing `Illiquidity` values fall mostly on recently listed stocks, which
  lack twelve monthly observations; the complete-case H2 sample therefore
  under-represents young firms, most visibly in 1999–2000 and 2021.
- The three characteristics overlap with the model predictors by design (the
  panel's `Size`, `RealizedVol` and `Illiquidity` are candidate predictors
  and $S$ depends on them). H2 tests incremental information conditional on
  these characteristics, not causal identification or complete risk
  adjustment.
- The controls are existing predictors derived from licensed data already
  held locally; they require no new WRDS extraction and are not
  redistributed.

### Implementation references

Inputs: `data/processed/universe/` (`mktcap`), `data/processed/panel/`
(`RealizedVol`, `Illiquidity`), `data/processed/conditional_disagreement/`
($S$, bins, $CD^B$, $CW$). Rank convention: `unit_rank` in
`disagreement.signals.family`. Upstream reference: OSAP tag `v2.0.0` as
above.

Module: `disagreement.controls.h2`, run with
`python -m disagreement.controls.h2 [--year Y]` (one OOS year, repeatable;
default: all baseline OOS years). It reads the universe, the panel and the
conditional disagreement of each OOS year, checks that the three cover
exactly the same stock-months with the same `mktcap`, and writes
`data/processed/h2_controls/<year>.parquet` with one row per member of
$U_t$, sorted by (`yyyymm`, `permno`), and the columns

| Column | Content |
|---|---|
| `permno`, `yyyymm` | the universe's stock-month keys |
| `size`, `realized_volatility`, `illiquidity` | SIZE, RVOL, ILLIQ as defined above; missing source: missing |
| `size_rank`, `realized_volatility_rank`, `illiquidity_rank` | $X^c$ in $[-1, 1]$; missing value: missing |
| `complete_case` | all three ranks present |
| `centered_signal` | $S^c = 2(S - 0.5)$ from the saved ensemble signal |
| `signal_x_<control>_rank`, three columns | $S^c \times X^c$; missing rank: missing |

The ranks are computed on the full universe of the month, before the
complete-case flag is formed, so that an H2 estimation restricted to
`complete_case` keeps the full-universe ranks. $S$, the signal bins, $CD^B$
and $CW$ are read for the signal only and never recomputed. A JSON sidecar
per year, written last as the completion marker, records aggregate
coverage, the frozen transformation definition and its SHA-256, and the
SHA-256 of the three input files and of the output file itself;
`load_h2_controls` in the same module, the entry point for any downstream
use, rejects a year whose inputs, output file or definition have changed
since it was built, with no option to skip these checks.
Rerunning the conditional disagreement of a year, a new target build, a new
`build` or a new CRSP extract removes the controls as well. Tests on
synthetic data: `tests/test_h2_controls.py`.

## Estimation protocol

Frozen on 2026-11-09 before any outcome was examined
([methodology_changes.md](methodology_changes.md)). It fixes how H1–H4 are
estimated and reported. The estimation engine and the diagnostic module of
"Reproducibility of the evidence" are implemented separately and tested on
synthetic data before the first regression on real outcomes. Hypotheses,
expected directions and verdict categories: [hypotheses.md](hypotheses.md).

### Evaluation outcome

Primary: `target_rank` ("Prediction target"), so that $\beta_3$ measures how
the slope of the outcome rank on $S^c$ varies with $CD^B$ on a scale that is
identical in every month and not dominated by extreme returns. Robustness:
raw `ret_next`, neither winsorised nor standardised, with the same
regressors. Both are always reported; the primary outcome decides.

### Monthly regressions

Each OOS formation month $t$ is estimated separately by equal-weighted OLS.
Specification C is estimated in the bin-local basis

$$
y_i = \sum_{b=1}^{20} \mathbb{1}[b(i) = b]\,
\big(\alpha_b + \tilde\gamma_b\, z_i + \tilde\eta_b\, q_i\big)
+ \beta_2\, CD^B_i + \beta_3\, S^c_i\, CD^B_i + \varepsilon_i,
$$

with $z_i = (S^c_i - c_b)/h_b$, where $c_b$ and $h_b$ are the mean and the
half-range of $S^c$ in cell $(t, b)$, and $q_i = z_i^2 - \overline{z^2_b}$.
The constants $c_b$, $h_b$ and $\overline{z^2_b}$ are computed from all
members of $U_t$ in the cell, before the outcome and complete-case
restrictions, so every specification and sample of a month uses the same
basis. The 20 cell intercepts replace a global intercept. The basis spans
exactly the bin-specific terms in $S^c$ and $(S^c)^2$ of hypotheses.md, so
$\beta_2$
and $\beta_3$ are unchanged (checked with synthetic outcomes on the saved
regressors to $3 \times 10^{-13}$) while the scaled condition number falls
from about $4 \times 10^3$ to about 4. $S^c \times CD^B$ uses the raw $S^c$
and the saved $CD^B$. Specification B drops the $q_i$ terms. H2 adds
$X^c_k$ and $S^c X^c_k$ for the three controls on the complete-case sample;
H3 adds $CW$ and $S^c \times CW$ on the H1 sample.

Samples: the H1 sample is the members of $U_t$ with a finite outcome; the
H2 sample is the H1 sample with `complete_case` true. $U_t$, $S$, the bins,
$CD^B$, $CW$, $X^c$ and `target_rank` are never recomputed on a subsample.
Specification C is always re-estimated on the H2 sample and reported next to
H2. The monthly share of missing outcomes and the sample sizes are reported,
together with a descriptive coverage diagnostic: the share of $U_t$ members
with a finite outcome by signal bin and by tercile of $CD^B$ inside each
(month, bin) cell (same rank convention as the bins, equal month weights,
mean and minimum monthly share; `disagreement.inference.coverage`, tested on
synthetic data). It describes where outcomes are missing and changes neither
the sample, nor the estimation, nor the inference.

### Estimable months

A month is estimable for a sample and a design if every cell $(t, b)$ holds
at least 30 observations of that sample with at least 3 distinct values of
$S^c$, and if the design matrix with columns scaled to unit Euclidean norm
has $\sigma_{\min} / \sigma_{\max} > 10^{-8}$ (singular value
decomposition). On the saved regressors every cell holds at least 86 stocks
(52 on the complete-case sample) and the scaled condition number is below
10, so no month is expected to fail. A failing month is dropped from the
corresponding coefficient series and listed. Paired comparisons (C on the H2
sample against H2 and against H1; H3 against H1) use the months estimable
for every series involved; if that set differs from the primary H1 set, H1
is also reported on it.

### Fama–MacBeth inference

For a monthly series $\beta_\tau$ (a coefficient, a slope or a paired
difference) on the calendar grid $\tau = 1, \dots, N$ of the 480 OOS months,
let $a_\tau = 1$ for estimable months, $T = \sum_\tau a_\tau$,
$\bar\beta = T^{-1} \sum_\tau a_\tau \beta_\tau$ and
$x_\tau = a_\tau (\beta_\tau - \bar\beta)$, zero elsewhere. With $L = 6$
calendar months,

$$
\hat\Gamma_l = \frac{1}{T} \sum_{\tau = l+1}^{N} x_\tau x_{\tau - l}, \qquad
\hat\Omega = \hat\Gamma_0
+ 2 \sum_{l=1}^{L} \Big(1 - \frac{l}{L + 1}\Big) \hat\Gamma_l, \qquad
\mathrm{SE} = \sqrt{\hat\Omega / T}.
$$

Bartlett kernel, normalisation $1/T$, no finite-sample correction. Without
missing months this is Newey and West (1987); $L = 0$ gives the classical
Fama–MacBeth standard error with the population variance. Inference uses
the standard normal distribution throughout: $t = \bar\beta / \mathrm{SE}$,
$p = 2\,(1 - \Phi(|t|))$ and the 95% interval $\bar\beta \pm 1.96\,\mathrm{SE}$.
$L = 0$ and $L = 12$ are reported as sensitivity analyses without
decisional value. For a vector series the same weights give $\hat\Omega$ as
a matrix and the Wald statistic $T\, \bar\beta' \hat\Omega^{-1} \bar\beta$,
referred to $\chi^2$ with the vector's dimension. Implementation checks: a
complete series must match `statsmodels`
`cov_hac_simple(nlags=6, use_correction=False)` of a regression on a
constant; a series with gaps must match the explicit sum over observed
pairs.

### Reporting rules for H1 and H2

- *Fixed sequence.* H1 is tested first, two-sided at 5%; H2 is confirmatory
  only if H1 is supported in the negative direction (verdicts:
  hypotheses.md). Every coefficient, standard error, p-value and interval
  of every specification is reported whatever the gate outcome.
- *Paired differences*, with HAC intervals and no causal reading. Write
  $\beta_3^{H1}$, $\beta_3^{C|H2}$ and $\beta_3^{H2}$ for C on the H1
  sample, C on the H2 sample and H2.
  $\Delta_\tau = \hat\beta^{H2}_{3,\tau} - \hat\beta^{C|H2}_{3,\tau}$ is the
  change due to the controls on a fixed sample and
  $\Delta^S_\tau = \hat\beta^{C|H2}_{3,\tau} - \hat\beta^{H1}_{3,\tau}$ the
  change due to sample composition. When
  $\bar\beta_3^{C|H2} < 0$, an interval for $\bar\Delta$ excluding 0 is
  reported as attenuation towards zero if $\bar\Delta > 0$ and as
  amplification if $\bar\Delta < 0$; an interval containing 0 is reported as
  "no detected change" together with the interval, never as absence of
  attenuation. Controls absorb part of the covariation; they do not
  "explain" it.
- *Marginal signal slopes.* In cell $(t, b)$, $\partial E[y] / \partial S^c =
  (\tilde\gamma_b + 2 \tilde\eta_b z) / h_b + \beta_3\, CD^B +
  \sum_k \delta_k X^c_k$, the last term in H2 only, with $h_b$ the fixed
  constant of "Monthly regressions". The reported monthly slope is the
  equal-weight average over the 20 bins at the bin centre ($z = 0$),
  $\bar g_\tau(c) = \tfrac{1}{20} \sum_b \tilde\gamma_{b,\tau} / h_{b,\tau} +
  c\, \hat\beta_{3,\tau} + \sum_k \hat\delta_{k,\tau}\, x_k$ for
  $c \in \{0.1, 0.5, 0.9\}$, where $x_k = 0$ in H1 (no control term) and in
  the primary H2 evaluation ($X^c = 0$, the median stock of $U_t$). As an
  optional descriptive comparison, H2 also reports the slope at
  $x_k = \bar X^c_{k,\tau}$, the monthly mean of $X^c_k$ among the H2
  complete cases. Each $\bar g_\tau(c)$ is a monthly series with the HAC
  interval of "Fama–MacBeth inference"; the $\hat\delta_k$ are reported with
  intervals. These are local conditional slopes of the outcome with respect
  to $S^c$ inside a bin, averaged over bins; they are not a global measure
  of rank-prediction accuracy and are not compared with any reference value.
- *Two statements.* "Negative $S^c \times CD^B$ interaction" is the
  statistical verdict. "A positively informative signal becomes less
  reliable where disagreement is higher" is claimed only if, in addition,
  $\bar g(0.1) > 0$ with $p < 0.05$ in the primary evaluation ($x_k = 0$).
  If $\bar g(0.9)$ is significantly negative, the result is described as a
  sign reversal. $\bar g$ is a local within-bin slope estimated with far
  less precision than across-bin contrasts, so a condition not met is
  reported as "informativeness not established at this precision", never as
  evidence that the signal is uninformative; the across-bin relation itself
  is carried by the cell intercepts and is not part of the rule.

### Robustness and subperiods

Each analysis repeats the primary H1 test with one change; H2, H3 and H4
are estimated in their primary form only. Results are descriptive (sign and
significance are compared with the primary result) and do not alter the
primary verdict.

| Analysis | Change |
|---|---|
| Specification B | no bin-specific quadratic term (hypotheses.md) |
| Excluding 2021 | without the twelve 2021 formation months (hypotheses.md) |
| Raw outcome | `ret_next` in place of `target_rank` |
| Raw disagreement | $D^{B,\mathrm{raw}}$ in place of $CD^B$, same controls |
| Range measure | max − min of the four family ranks, ranked within the same (month, bin) cells, in place of $CD^B$ |
| 10 bins | $CD^B$, $CW$ and the bin-specific controls recomputed with 10 monthly bins of $S$ |
| Leave one family out | four variants: $S$ and $D^{B,\mathrm{raw}}$ from the three remaining families, bins and $CD^B$ recomputed on that $S$ |
| HAC lag | $L = 0$ and $L = 12$ |
| Subperiods | 1985-01 to 2004-12 and 2005-01 to 2024-12, 240 months each, chosen for equal length and not for any event |

### H3

Specification C plus $CW$ and $S^c \times CW$ on the H1 sample. The two
interaction coefficients are tested separately, two-sided at 5%; "distinct
information" requires both to reject (intersection–union test, level 5%
without correction), and the joint Wald test of the two interactions is
reported as a complement. No magnitude comparison, no measurement-error
correction and no replica-split reliability diagnostic (reasons:
hypotheses.md, H3). H3 is outside the H1 → H2 gate.

### H4

Secondary, pre-specified, conditional observed-outcome comparison of
selection rules on gross one-month returns. It is not an implementable
fixed-weight portfolio backtest.

*Formation.* In each OOS formation month $t$ the long leg is signal decile
10 and the short leg signal decile 1 of $U_t$, decile $d$ being the union of
signal bins $2d - 1$ and $2d$ ("Conditional disagreement"). Holdings and
weights are fixed at $t$: equal weights $1 / n_{\mathrm{leg}}$, one-month
holding.

*Filters.* Each filter removes $k = \lfloor \varphi\, n_{\mathrm{leg}} \rfloor$
stocks from each leg before weighting, with $\varphi = 1/3$ (primary) and
$\varphi \in \{0.2, 0.5\}$ (sensitivity). The disagreement filter removes
the $k$ highest $CD^B$; the volatility filter the $k$ highest economic
realized volatility (`realized_volatility`, that is $-$`RealizedVol`, higher
meaning more volatile); the signal-extremity filter the $k$ lowest $|S^c|$,
the least extreme signals. Ties are broken by ascending PERMNO. A stock whose
filter characteristic is missing at $t$ (only `RealizedVol`, about 0.3% of
$U_t$) is not eligible for removal: the $k$ removals are taken among the
stocks with a finite value, and their count is reported. Every filter
variable is known at formation.

*Returns and missing outcomes.* Holdings, filters and the equal weights are
fixed at $t$, before any $t + 1$ return is observed. The intended weights
are $1 / n_{\mathrm{leg}}$ on every selected holding; the leg statistic is
the equal-weight mean of `ret_next` over the selected holdings whose
`ret_next` is finite, that is weights $1 / n_{\mathrm{obs}}$ renormalised on
the observed holdings, which equals the intended portfolio return only when
every holding has a return. The long–short statistic is the difference of
the two leg means. This is a conditional observed-outcome comparison of
selection rules, not the return of an implementable fixed-weight portfolio
and not a demonstration of economic value: a holding
without a finite `ret_next` contributes no return, the leg's coverage (share
of selected holdings with a finite `ret_next`) is recorded, and the
statistic estimates the rule's mean next-month return without bias only if
returns are missing at random within the leg. Non-random missingness, such
as a delisting without a CRSP delisting return, can bias the comparison
even though the procedure is pre-specified; the coverage differences below
are the disclosed measure of that risk. Missing returns are never imputed
or set to zero. A complete-case calendar-month rule, keeping only months in
which every holding has a return, is not used: with 170 to 320 holdings per
leg it would discard most months and retain a selected set of them.

*Months and disclosure.* A month enters the H4 series if every leg of every
variant and intensity has at least 20 holdings with a finite `ret_next`; all
variants are evaluated on this common set of months, whose size is reported.
Coverage is disclosed per month, leg and variant (mean, minimum, months below
0.95) together with the coverage difference between each filter and no
filter; a coverage difference is a stated limitation of the comparison.

*Statistics.* For each variant, the mean long–short return with the HAC
standard error of "Fama–MacBeth inference" ($L = 6$, normal interval) and
the annualised Sharpe ratio (mean over population standard deviation times
$\sqrt{12}$, descriptive). For each filter, the paired monthly difference
against no filter, with HAC mean and interval, two-sided at 5%. These
comparisons are descriptive and unadjusted for their number (three filters
at three intensities); no expected ranking, no superiority claim and no test
on Sharpe differences.
