# Data Pipeline

The data pipeline builds a monthly, formation-time data layer: the investable
universe $U_t$ from CRSP and a panel of candidate predictors for its members.
It contains no future return and no prediction target.

Code: `src/disagreement/data/`. Commands and local files:
[data/README.md](../data/README.md). Data rules:
[data_governance.md](data_governance.md).

## Data sources

**CRSP (WRDS, CIZ format).** Monthly security data from
`crsp.stkmthsecuritydata` (CRSP Stock Version 2). The return field `MthRet`
includes the delisting return when CRSP has one; no separate delisting return
is ever compounded into it.

**Open Source Asset Pricing (OSAP).** Chen and Zimmermann firm-level
characteristics, October 2025 release, version 2.0.0:
`signed_predictors_dl_wide.zip` from the official data page
(<https://www.openassetpricing.com/data/>, Google Drive file
`1avFIMjz_7LoF3p3nO26eqLW5KdRTOdhW`), retrieved 2026-10-12, SHA-256
`aae101c1fd4749c5a68d83734b6ce5e90a0f2f18b456f498e6f3b803081f6ee3`.
It holds 209 signed characteristics (higher values predict higher mean
returns). Some characteristics extend beyond 2024 because OSAP lags
accounting data six months and carries them for twelve months. `build`
checks the file's SHA-256 and its 209 predictor columns before any
processing.

Both sources are keyed by CRSP PERMNO and the month `yyyymm`.

## Universe construction

For each month $t$, using only the month-$t$ CRSP record, a security is
eligible if it is:

1. a U.S. common stock: `ShareType = NS`, `SecurityType = EQTY`,
   `SecuritySubType = COM`, `USIncFlg = Y`, `IssuerType` in `ACOR`, `CORP`
   (the CIZ equivalent of legacy share codes 10 and 11);
2. listed on NYSE, AMEX or NASDAQ (`PrimaryExch` in `N`, `A`, `Q`), traded
   regular way (`ConditionalType = RW`) and active (`TradingStatusFlg = A`),
   the CIZ equivalent of legacy exchange codes 1, 2 and 3;
3. still alive at the end of month $t$: excluded if its month-$t$ record has
   `MthRetFlg = DE` or a delisting flag `MthDelFlg` of `A`, `P`, `V`, `G` or
   `M`, whether or not the delisting return is included in `MthRet`;
4. priced with a finite, positive market capitalisation `MthCap`.

$U_t$ keeps the eligible NYSE, AMEX and NASDAQ stocks whose `MthCap` is at or
above the 20th percentile of `MthCap` among eligible NYSE stocks in the same
month (security-level market capitalisation, linear interpolation; a stock
exactly at the breakpoint is kept). A month without eligible NYSE stocks has
no breakpoint and fails the build.

## Predictor panel

The panel is a left join from $U_t$ on PERMNO-month: every member keeps one
row, with or without an OSAP record. It holds 212 candidate predictors
before any coverage filtering: the 209 OSAP characteristics and the three
predictors that OSAP leaves to be built from CRSP. These follow OSAP's own
code (CrossSection v2) and its sign convention (sign -1 for all three), using
the month-$t$ CIZ record:

| Predictor | OSAP definition | Stored value |
|---|---|---|
| Price | $\ln\lvert prc\rvert$ | $-\ln\lvert \text{MthPrc}\rvert$ |
| Size | $\ln(\lvert prc\rvert \times shrout)$, per PERMNO | $-\ln(\text{MthCap})$ |
| STreversal | month return in percent, missing set to 0 | $-100 \times \text{MthRet}$, missing set to 0 |

CIZ `MthCap` is in thousands of dollars, like $\lvert prc\rvert \times shrout$
in the legacy files.

Positive and negative infinity is stored as missing. No other transformation
(winsorising, imputation, ranking) is applied.

## Timing

- Universe membership and predictors use only information available at the
  end of formation month $t$. OSAP data "are always end-of-month": a
  characteristic dated `yyyymm` is joined to month $t$ = `yyyymm` without an
  additional lag.
- No future return, survival, delisting or availability enters $U_t$.
- A security in $U_t$ that delists during $t+1$ stays in $U_t$.

## Reproducibility

From the repository root, with the environment pinned in
`requirements-lock.txt` (`wrds` 3.5 requires `pandas < 2.3`):

```
.venv\Scripts\python -m disagreement.data crsp-extract   # CRSP fields from WRDS
.venv\Scripts\python -m disagreement.data build          # universe, panel, validation
.venv\Scripts\python -m pytest                           # synthetic tests
```

Licensed CRSP data, the OSAP archive and all derived files stay under
`data/`, which Git ignores. Commands print aggregate information only. The
OSAP release, its SHA-256 and its number of predictors are pinned in
`configs/baseline.yaml`.

## Validation

`build` stops with an error unless the OSAP file matches the pinned release
(SHA-256, 209 distinct predictor names, none colliding with a panel column);
the CRSP extract matches its manifest (table, fields, rows per year, first
and last month of the live table), each yearly file holds only its own year,
and CRSP has records in every month between its first and last month; every
PERMNO-month key is unique in CRSP, OSAP, $U_t$ and the panel; every month
has a valid NYSE breakpoint; and the panel has exactly one row per $U_t$
member. The inputs are checked before earlier outputs are removed, and
`data/diagnostics/summary.json` and `universe_by_month.csv` are written only
when every check passes. Tests on synthetic data check the eligibility rules,
the breakpoint, the absence of month-$t+1$ information, the join and the
predictor formulas.
