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
cutoff; validation and out-of-sample months are never inspected. It depends
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
