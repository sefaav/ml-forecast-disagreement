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
.venv\Scripts\python -m disagreement.diagnostics.mechanical_bias                   # descriptive, all OOS years
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
conditional-disagreement year removes the whole `diagnostics/mechanical_bias/`
directory, which describes all OOS years at once; `build`, `crsp-extract` and
a successful `target` run remove it too. The ensemble is the plain mean of
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
  diagnostics/                    summary.json, universe_by_month.csv
    mechanical_bias/              monthly and signal-bin correlations, summary.json
```

## Successful run

`build` prints the summary saved in `data/diagnostics/summary.json`, with
`validation: PASS`. If a check fails, it exits with status 2, names the check
and leaves no `summary.json` or `universe_by_month.csv`. An OSAP file other
than the release pinned in `configs/baseline.yaml` (SHA-256) is rejected.
