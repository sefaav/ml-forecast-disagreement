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
.venv\Scripts\python -m pytest                           # synthetic tests
```

`crsp-extract` reads the WRDS username from `--wrds-username` or
`WRDS_USERNAME`, otherwise `wrds` prompts. It downloads the whole table into
a temporary directory, which replaces the previous extract only after the
extraction succeeds. `build` checks the extract against `_manifest.json` and
the OSAP archive against the pinned release before touching earlier outputs.
Both commands print aggregate information only.

## Local files

```
data/
  raw/crsp/      CRSP extract, one file per year, and _manifest.json
  raw/osap/      OSAP archive
  interim/       temporary files, removed by build
  processed/     universe/ (U_t members) and panel/ (U_t x 212 predictors), one file per year
  diagnostics/   summary.json, universe_by_month.csv
```

## Successful run

`build` prints the summary saved in `data/diagnostics/summary.json`, with
`validation: PASS`. If a check fails, it exits with status 2, names the check
and leaves no `summary.json` or `universe_by_month.csv`. An OSAP file other
than the release pinned in `configs/baseline.yaml` (SHA-256) is rejected.
