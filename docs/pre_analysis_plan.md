# Pre-Analysis Plan

## Baseline sample

Formation months 1970-01 to 2024-12, inclusive (`configs/baseline.yaml`,
`data.sample`). The formation-time panel itself covers all CRSP months; the
sample restriction is applied when training and evaluation data are drawn
from it. The end point reflects the usable OSAP coverage of the pinned
October 2025 release; the baseline does not extend into 2025.

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
