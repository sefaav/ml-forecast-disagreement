# Data

Everything in this directory except this README is ignored by Git. Never
commit, share or paste rows of these files into any external service
([data governance](../docs/data_governance.md)). Methodology:
[docs/data_pipeline.md](../docs/data_pipeline.md).

## Sources

| Source | Access |
|---|---|
| CRSP monthly security data, CIZ format (`crsp.stkmthsecuritydata`) | WRDS account; downloaded by `crsp-extract` |
| OSAP firm-level characteristics, October 2025 release (`signed_predictors_dl_wide.zip`) | manual download from <https://www.openassetpricing.com/data/> |

Save the OSAP archive, **without unzipping it**, as
`data/raw/osap/signed_predictors_dl_wide.zip`. About 5 GB of free disk space
is needed.

## Commands

From the repository root (Windows PowerShell):

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e . --no-deps

.venv\Scripts\python -m disagreement.data crsp-extract   # WRDS login; full fresh extract
.venv\Scripts\python -m disagreement.data build          # universe, panel, validation
.venv\Scripts\python -m disagreement.data target         # next-month return and ranked target
.venv\Scripts\python -m disagreement.walkforward.forecast [--family F] [--year Y]   # OOS replica forecasts
.venv\Scripts\python -m disagreement.signals.family [--family F] [--year Y]        # replica and family ranks
.venv\Scripts\python -m disagreement.signals.ensemble [--year Y]                   # mean of the four family ranks
.venv\Scripts\python -m disagreement.signals.between [--year Y]                    # SD of the four family ranks
.venv\Scripts\python -m disagreement.signals.within [--year Y]                     # replica-rank SD, ranked by family
.venv\Scripts\python -m disagreement.signals.conditional [--year Y]                # both ranked within 20 signal bins
.venv\Scripts\python -m disagreement.controls.h2 [--year Y]                        # H2 controls as monthly centred ranks
.venv\Scripts\python -m disagreement.diagnostics.mechanical_bias                   # descriptive, all OOS years
.venv\Scripts\python -m disagreement.diagnostics.design_evidence                   # H1 design evidence, all OOS years
.venv\Scripts\python -m disagreement.diagnostics.forecast_performance              # dry run only (see Forecasting diagnostics)
.venv\Scripts\python -m disagreement.inference.run                                 # dry run only (see Inference)
.venv\Scripts\python -m disagreement.inference.robustness                          # dry run only (see H1 robustness)
.venv\Scripts\python -m pytest                           # synthetic tests
```

`crsp-extract` reads the WRDS username from `--wrds-username` or
`WRDS_USERNAME`, otherwise `wrds` prompts. It downloads the whole table into
a temporary directory, which replaces the previous extract only after the
extraction succeeds; the universe, panel, target, OOS forecasts, family and
ensemble signals, between-family disagreement, within-family uncertainty,
conditional disagreement and their diagnostics are then removed, so `build`
must be run again. `build` checks the extract against `_manifest.json` and
the OSAP archive against the pinned release before touching earlier outputs.
`target` reads the universe of the last successful build and the CRSP
extract; `build` removes it together with the universe, panel, OOS forecasts,
signals, disagreement and uncertainty, and a successful `target` run removes
the OOS forecasts, signals, disagreement and uncertainty. Rerunning any stage
for an OOS year first removes everything downstream of it for that year: a
forecast or family signal removes the ensemble, the between-family
disagreement, the within-family uncertainty and the conditional disagreement
of that year; an ensemble removes the between-family and conditional
disagreement, since the within-family uncertainty is computed from the family
signals; the between-family disagreement and the within-family uncertainty
each remove the conditional disagreement. Removing or rerunning any
conditional-disagreement year removes that year's H2 controls and the whole
`diagnostics/mechanical_bias/` and `diagnostics/design_evidence/`
directories, which describe all OOS years at once; rerunning any H2-controls
year removes `diagnostics/design_evidence/`; `build`, `crsp-extract` and a
successful `target` run remove all three.
The H2 controls also record the SHA-256 of their three input files and of
their own output file; their loader rejects a year whose inputs or output
changed after it was built. The ensemble is the plain mean of
the four family ranks, not re-ranked; the between-family disagreement is
their population standard deviation, not re-ranked. All commands print
aggregate information only.

## Local files

```
data/
  raw/
    crsp/                         CRSP extract, one file per year, and _manifest.json
    osap/                         OSAP archive
  interim/                        temporary files, removed by build
  processed/
    universe/                     U_t members, one file per year
    panel/                        U_t x 212 predictors, one file per year
    target/                       ret_next and target_rank, one file per year
    oos_predictions/              raw replica forecasts, one file per family and OOS year
    family_signals/               replica and family ranks, one file per family and OOS year
    ensemble_signals/             four family ranks and their mean, one file per OOS year
    between_family_disagreement/  ensemble signal and raw disagreement, one file per OOS year
    within_family_uncertainty/    replica-rank SDs, their ranks and mean, one file per OOS year
    conditional_disagreement/     signal bins and conditional ranks, one file per OOS year
    h2_controls/                  size, volatility and illiquidity ranks, one file per OOS year
  diagnostics/                    summary.json, universe_by_month.csv
    mechanical_bias/              monthly and signal-bin correlations, summary.json
    design_evidence/              H1 design diagnostics and synthetic reruns, summary.json
    forecast_performance/         monthly rank IC of the five signals, summary.json
    inference/                    H1-H3 inference results, manifest.json, summary.json
    h1_robustness/                remaining H1 robustness analyses, manifest.json, summary.json
```

## Successful run

`build` prints the summary saved in `data/diagnostics/summary.json`, with
`validation: PASS`. If a check fails, it exits with status 2, names the check
and leaves no `summary.json` or `universe_by_month.csv`. An OSAP file other
than the release pinned in `configs/baseline.yaml` (SHA-256) is rejected.

## Forecasting diagnostics

Descriptive out-of-sample evaluation of the saved forecasts, separate from the
H1–H3 inference; it tests no hypothesis. It reads the saved outcomes and is
guarded like the inference:

```
.venv\Scripts\python -m disagreement.diagnostics.forecast_performance            # dry run
.venv\Scripts\python -m disagreement.diagnostics.forecast_performance --execute  # evaluate and publish
```

The five signals are the four family ranks and the ensemble signal of the
saved ensemble files. The metric is the frozen model-selection score. In
every OOS formation month, the stocks with a finite saved `target_rank` are
kept and each signal is average-ranked among them, ties sharing their rank.
The ranks are correlated with `target_rank` (Pearson), and a constant signal
scores 0. The saved signals are ranked again for the correlation only and
never modified. Every month weighs the same. The ensemble and target files of
each OOS year are validated and must cover exactly the same stock-months.

Results go to `data/diagnostics/forecast_performance/`, which is never
replaced; remove it to rerun:

- `monthly_ic.csv`: one row per month and signal, with the rank IC, whether
  the scored signal was constant, and the numbers of $`U_t`$ members and of
  scored stocks;
- `summary.json`, written last: per signal the mean, median and sample SD of
  the monthly IC, the shares of positive, zero and negative months and the
  constant months; the outcome coverage; the SHA-256 of the configuration and
  of every input file.

Like everything here, these files stay local and are never committed.

## Inference

The frozen H1–H3 inference of the pre-analysis plan ("Estimation protocol")
reads the saved outcomes. It is guarded:

```
.venv\Scripts\python -m disagreement.inference.run                         # dry run
.venv\Scripts\python -m disagreement.inference.run --execute               # load, estimate, publish
.venv\Scripts\python -m disagreement.inference.run --execute --replace     # supersede a valid result set
```

- **Without `--execute`** the command reads only the configuration. It prints
  the calendar (480 months, 40 OOS years) and the destination, and loads,
  estimates and writes nothing.
- **With `--execute`** it loads each OOS year's conditional disagreement, H2
  controls and target. It estimates the frozen specifications on
  `target_rank`, plus specification C on the raw `ret_next`, and computes the
  Fama–MacBeth inference at lags 6, 0 and 12 and the verdicts. It then
  publishes the result set. No lag, window or specification can be chosen.
- **Options:** `--config` (default `configs/baseline.yaml`), `--data-dir`
  (default `data`) and `--output-dir` (default `data/diagnostics/inference/`).
- **An existing result set** is kept unless `--replace` is given. Only a
  directory that verifies as a published inference result set, made of
  regular files, is ever replaced.
- **Unsafe destinations are refused** before any outcome is loaded:
  - a symbolic link or junction;
  - the data root, `diagnostics/`, the repository or the configuration file,
    or an ancestor of any of them;
  - a pipeline data directory, its ancestors or anything inside it;
  - a directory inside another result set;
  - a directory inside the repository but not below its Git-ignored output
    areas `data/` and `experiments/results/`, even if it does not exist yet.
    Destinations outside the repository remain possible.
- **Output:** the command prints administrative lines only, with no
  coefficient, return or verdict. On any failure it exits with status 2 and
  publishes nothing.

The result files are monthly and aggregate tables derived from licensed data;
no stock-level row is written. Like everything here they are never
committed or pasted into an external service.

| File | Content |
|---|---|
| `report.json` | the five decisions (H1, H2 behind the H1 → H2 gate, the paired Δ reading, the economic H1 reading, H3) with their conditions, notes and lag-6 statistics; the H3 joint Wald test at every lag; non-estimable months; inference failures |
| `inference.csv` | one row per series and lag: mean, HAC standard error, t, p, 95% interval, months used, or the failure reason |
| `series.csv` | every monthly series on the 480-month grid, with its estimability mask |
| `monthly_estimates.csv` | one row per month and specification: estimability, reasons, sample size, conditioning diagnostics, coefficients |
| `monthly_bins.csv` | one row per month, specification and signal bin: cell size, bin-local coefficients, basis constants |
| `monthly_summary.csv`, `coverage_monthly.csv`, `coverage_profile.csv`, `h2_control_means.csv` | sample sizes, outcome coverage, H2 control means (descriptive) |
| `manifest.json` | SHA-256 of every input file read and of every file above |
| `summary.json` | completion marker, written last |

- **`manifest.json`** is the integrity and provenance record. It holds the
  configuration and, for each OOS year, the conditional-disagreement and
  H2-control files with their markers, the target file, and the universe and
  panel files behind the H2 controls. Each input is hashed before and after
  its year is loaded, and again just before publication; a change at any
  point blocks the publication.
- **`summary.json`** is a technical completion marker. It means that the core
  inference run completed and that the manifest, whose SHA-256 it records,
  matches the files.
- **Scientific completeness** is a different matter. The marker lists the
  implemented and the outstanding analyses and states
  `scientific_protocol_complete: false`. The robustness analyses of the
  pre-analysis plan other than specification B, the raw outcome and the HAC
  lags (excluding 2021, raw, range and MAD disagreement, ten bins,
  leave-one-family-out, subperiods) are not part of this result set; they
  have their own command (see H1 robustness). H4 and its backtesting
  extension are not implemented.

**Verifying a result set.** Verification uses the Python API; the command
has no verification option. Pass the data directory and the configuration
the result was produced with to check the inputs as well:

```
.venv\Scripts\python -c "from pathlib import Path; from disagreement.data.config import DEFAULT_CONFIG, DataPaths; from disagreement.inference.output import verify_published; verify_published(Path('data/diagnostics/inference'), DataPaths(Path('data')), DEFAULT_CONFIG); print('verified')"
```

`verify_published` checks the exact file inventory, the marker, the manifest,
the SHA-256 of every result file and the consecutive monthly calendar. It
also checks the marker's scientific scope (completion meaning, primary
outcome and lag, implemented and outstanding analyses, roles) against the
code's constants and the H1 → H2 gate in the hashed `report.json`. With
the data directory and configuration it also checks every recorded input. It
raises an error on the first difference. No pipeline command removes
`diagnostics/inference/`. After an upstream stage is rerun, an earlier result
set stays in place even though its inputs have changed, so verify it with the
inputs before relying on it.

**Replacement.** The result set is written to a staging directory
`.inference.staging-<id>` beside the destination and published by renaming
it. With `--replace` the existing result set is first renamed to
`.inference.previous-<id>`. If the final rename fails, it is renamed back; if
that also fails, it is kept at the `.previous-` path and never deleted.
Between the two renames the destination is briefly absent. A crash of the
process at that moment, or before the staging directory is cleaned up, is
not recovered automatically: restore the `.previous-` directory by hand and
delete a leftover `.staging-` directory.

## H1 robustness

The pre-specified H1 robustness analyses that the core inference does not
include ("Robustness and subperiods"). Each repeats the primary H1 test
(specification C, `target_rank`, lag 6) with one change; all are descriptive
and none alters the H1 verdict.

```
.venv\Scripts\python -m disagreement.inference.robustness            # dry run
.venv\Scripts\python -m disagreement.inference.robustness --execute  # load, estimate, publish
```

- **Without `--execute`** only the configuration is read.
- **With `--execute`** the command first verifies `diagnostics/inference/`
  with its inputs. It then takes:
  - the primary monthly interaction series from that result set, for the
    analysis excluding 2021 and for the two subperiods;
  - each OOS year's conditional disagreement, ensemble signals and target,
    for the analyses that are re-estimated: raw, range and MAD disagreement,
    ten bins, and the four leave-one-family-out ensembles.
- **Checks before estimation:**
  - the saved raw disagreement must equal the SD of the saved family ranks;
  - the saved CD^B must equal that SD ranked within the saved cells;
  - re-estimating C must reproduce the published series.
- **Nothing upstream is rerun or modified.**
- **Destination:** `--output-dir`, default `data/diagnostics/h1_robustness/`.
  It is checked like the inference destination and is never replaced.

| File | Content |
|---|---|
| `inference.csv` | one row per analysis (and the published primary for comparison) at lag 6: mean, HAC standard error, t, p, 95% interval, months, or the failure reason |
| `monthly.csv` | one row per re-estimated analysis and month: bins, estimability, reasons, sample size, β₂, β₃ |
| `manifest.json` | SHA-256 of the configuration, of every input file (the published inference files included) and of the two tables |
| `summary.json` | completion marker, written last |

Verify a result set with
`disagreement.inference.robustness.verify_robustness(path, DataPaths(Path('data')), DEFAULT_CONFIG)`.
