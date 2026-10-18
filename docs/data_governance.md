# Data Governance

These are the project's own conservative rules; they do not restate or
interpret the WRDS licence terms.

## Licensed data

- Licensed WRDS/CRSP row-level data must never be committed to this repository
  or redistributed.
- Raw and intermediate licensed data remain local only, under `data/`, which is
  excluded from version control except for `data/README.md`.

## Credentials

- WRDS credentials, passwords and connection secrets must never appear in
  source files, notebooks or committed configuration files.
- Local credential files such as `.env` and `.pgpass` are excluded by
  `.gitignore`.

## Derived outputs

- Stock-level or date-level derived outputs that could reproduce licensed data
  are not committed by default. For this reason, `experiments/results/` is
  excluded from version control; a file there is committed only after explicit
  review.
- Public repository outputs are limited to code, methodology, aggregate tables
  and figures that are permitted to be shared, and other non-restricted
  research outputs.

## Notebooks

- Notebook outputs must be inspected before each commit and stripped if they
  contain any licensed row-level data.

## AI tools and external services

- Licensed row-level data must never be printed into an AI tool conversation,
  pasted or uploaded to any external service, or shown in terminal output
  that such a tool can read.
- Pipeline commands print only schema names, counts, date ranges, quantiles
  and validation status; problems are reproduced with synthetic data, not
  with licensed rows.

## Tests

- Tests will use clearly synthetic fixtures, never licensed data.

## Data sources and pipeline

- Sources, access procedures and the local data layout are documented in
  [data/README.md](../data/README.md) and
  [data_pipeline.md](data_pipeline.md).
