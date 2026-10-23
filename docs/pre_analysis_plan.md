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
5. the selected specification is refitted on training and validation, with
   the same predictor set;
6. that single refitted model produces the forecasts for all twelve OOS
   formation months of the episode, without refitting within the year.

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
