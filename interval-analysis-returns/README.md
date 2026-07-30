# Classical and Interval Analysis of Stock Returns

A study of the DIA ETF against 15 of its constituent stocks over
2021–2025, run twice: once with classical point-valued daily returns,
and once with **interval-valued** returns, to see what the interval
representation adds.

## The idea

A classical daily return collapses a trading day to one number,
close-to-close: `r = C_t / C_{t-1} − 1`. That throws away the day's
range. Interval analysis instead treats the day as the interval
`[L_t, H_t]`, normalised by the previous close, and works with two
derived series:

```
mid_t = (L_t + H_t) / (2·C_{t-1}) − 1     centre of the interval
rad_t = (H_t − L_t) / (2·C_{t-1})         half-width
```

The midpoint is the analogue of the classical return; the radius is a
direct, per-day measure of intraday volatility that the classical series
cannot express. Every classical technique in the report is then re-run on
the midpoint and radius series, and the two accounts are compared.

## Data

`DIA` (Dow Jones ETF) as the dependent variable, against 15 constituents
grouped by sector: AAPL, MSFT, NVDA, GOOGL, META, AMZN, NFLX, TSLA, JPM,
BAC, V, JNJ, UNH, LLY, HD.

Pulled from Yahoo Finance via `quantmod` and **cached to `data/` as
CSV** — 1,254 trading days, 2021-01-04 to 2025-12-30. The report reads
the cache when it exists and only downloads on a miss, so knitting is
deterministic, works offline, and the data travels with the repository.
The RNG seed is the student registration number.

## What the analysis covers

Part A, on the classical returns, and Part B, repeating it on midpoint
and radius:

- Descriptive statistics, normality and moment analysis (`moments`,
  `psych`)
- Correlation structure (`corrplot`)
- Multiple regression of the ETF on its constituents, with full
  diagnostics: VIF for multicollinearity (`car`), `gvlma` for the linear
  model assumptions, and robust/HAC standard errors (`sandwich`,
  `lmtest`) since financial residuals are heteroskedastic and
  autocorrelated
- Time-series checks (`tseries`)
- Exploratory factor analysis (`psych`, `GPArotation`)
- Structural equation modelling (`lavaan`)
- A regression tree as a nonlinear comparison (`rpart`, `rpart.plot`)
- Constrained regression via non-negative least squares (`nnls`) —
  appropriate here because an index is a non-negative combination of its
  constituents

## Building it

Open `project_report.Rmd` in RStudio and knit, or:

```r
rmarkdown::render("project_report.Rmd")
```

The setup chunk installs any missing packages and TinyTeX once, then
no-ops, so it builds on a clean machine. Output is PDF via **xelatex**,
which is required — the report is in Greek and needs proper Unicode font
handling. Figures use `cairo_pdf` for the same reason, and the palette is
Okabe–Ito for colour-blind safety.

## Files

- `project_report.Rmd` — the full analysis, ~2,170 lines of literate R
- `report.pdf` — the knitted report, as submitted
- `refs.bib` — bibliography
- `data/` — 16 cached price series, one CSV per ticker
