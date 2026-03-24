# Polymarket BTC 5m / 15m Signal Study

## Objective

This project tests a narrow question:

- do recurring Polymarket `Bitcoin Up or Down` markets on 5-minute and 15-minute horizons contain a short-term signal that is useful for BTC trading?

The project does not trade on Polymarket itself. Polymarket is treated as a read-only information source, while BTC is the tradable reference asset.

## Research question

The useful question is not:

- "Was Polymarket right by market close?"

The useful question is:

- "After observing Polymarket at a time `t` after market open, is there still a tradable BTC move left between `t` and market close?"

That distinction matters because a market can be directionally correct by expiry while still offering no remaining trading edge once the BTC move has already happened.

## Method

### 1. Discover the real recurring Polymarket markets

The pipeline targets recurring Gamma event slugs of the form:

- `btc-updown-5m-*`
- `btc-updown-15m-*`

This was necessary because simple text search was not sufficient to recover the live recurring markets shown on the Polymarket website.

### 2. Track three BTC prices per market

For every market, the study tracks three BTC prices:

- `S0`: BTC price at market open
- `Se`: BTC price at the chosen entry time after market open
- `ST`: BTC price at market close

These three prices produce three returns:

- `open_to_entry = Se / S0 - 1`
- `entry_to_close = ST / Se - 1`
- `open_to_close = ST / S0 - 1`

This separation is critical because it distinguishes:

- contract correctness: did the Polymarket contract resolve in the right direction between open and close?
- trade correctness: would a BTC trade entered at `Se` and held to `ST` have been in the right direction?

### 3. Test entry offsets after market open

The strategy does not force entry at open.

Instead, it tests multiple post-open offsets and asks which offset is most useful. For example:

- in `5m` markets, the candidate entry offsets are short and constrained by the horizon
- in `15m` markets, more offsets are available

The trading rule is intentionally simple:

- go long if the Polymarket implied `Up` probability is above the bullish threshold
- go short if the Polymarket implied `Up` probability is below the bearish threshold
- otherwise do not trade

### 4. Evaluate contract quality and trade quality separately

The pipeline produces two distinct labels:

- `contract_outcome`: whether the market resolved correctly from `S0` to `ST`
- `trade_outcome`: whether a BTC trade from `Se` to `ST` was in the correct direction

It then runs:

- an entry-offset study
- basic probability diagnostics
- a simple logistic benchmark
- a paper BTC backtest with costs

## Reference run

The reference run documented in this repository uses the following backfill window:

- `2026-03-22T15:00:00Z` to `2026-03-23T15:30:00Z`

Summary of the resulting dataset:

- total registry size: `717` markets
- Polymarket markets: `399`
- detected Polymarket horizons:
  - `302` markets in `5m`
  - `97` markets in `15m`
- BTC candles: `1471`
- normalized market observations: `5787`
- final feature rows: `1871`

## What the outputs mean

Selected outputs are copied into `docs/results/` so the study can be read on GitHub without rerunning the pipeline.

### Entry-offset charts

Files:

- `docs/results/entry-offset-5m.png`
- `docs/results/entry-offset-15m.png`

These charts show the trade-off between:

- `trade_threshold_accuracy`: accuracy of the threshold-based rule on the remaining BTC move from `Se` to `ST`
- `trade_coverage`: fraction of rows that actually trigger a trade

Interpretation:

- high accuracy with very low coverage usually means the rule fires only rarely
- this is useful as a signal-quality diagnostic, but not enough by itself to prove a scalable strategy
- these charts are exploratory diagnostics across candidate entry offsets, not the final backtest by themselves
- the final baseline and backtest retain one offset per horizon; when several offsets tie under the conservative selector, the earliest offset is kept

### Calibration charts

Files:

- `docs/results/calibration-5m.png`
- `docs/results/calibration-15m.png`

These charts compare predicted probabilities from the benchmark model with realized frequencies.

Interpretation:

- points near the diagonal indicate good probability calibration
- large deviations indicate that the model's estimated probabilities should not be trusted as literal probabilities

### ROC charts

Files:

- `docs/results/roc-5m.png`
- `docs/results/roc-15m.png`

These charts evaluate ranking quality across all classification thresholds.

Interpretation:

- an AUC near `0.5` means little to no ranking power
- an AUC above `0.5` suggests some predictive power
- an AUC below `0.5` means the model is effectively worse than random on that sample

### Equity chart

File:

- `docs/results/equity-5m.png`

This chart shows the cumulative BTC paper-trading equity curve of the threshold-based rule, after costs.

Interpretation:

- the `5m` reference run only generated one trade, so the curve contains a single upward step
- there is no `15m` equity chart in the reference run because the current thresholds did not trigger any `15m` trades

### Metric files

Files:

- `docs/results/baseline-metrics.json`
- `docs/results/backtest-metrics.json`
- `docs/results/backtest-trades.csv`
- `docs/results/entry-offset-5m.csv`
- `docs/results/entry-offset-15m.csv`

These files contain the raw numbers behind the study and are useful if the charts need to be audited or reproduced.

## Current results

### 5-minute horizon

Key metrics from the reference run:

- evaluated rows: `89`
- `trade_coverage = 0.0112`
- `benchmark_auc = 0.5532`
- `benchmark_brier = 0.2498`
- paper backtest trades: `1`
- paper backtest cumulative return: `0.0007755`

Interpretation:

- the `5m` horizon shows a possible weak signal
- however, it is sparse and not yet robust enough to claim a reliable edge

### 15-minute horizon

Key metrics from the reference run:

- evaluated rows: `30`
- `trade_coverage = 0.0`
- `benchmark_auc = 0.3654`
- `benchmark_brier = 0.2484`
- paper backtest trades: `0`

Interpretation:

- the `15m` horizon is not empty
- some candidate entry offsets show isolated in-sample threshold hits in the exploratory offset chart
- but the final conservative selector retains offset `1` in this sample, and under that final rule the `15m` horizon does not produce an actionable trade

## Honest conclusion

The current study supports a restrained conclusion:

- yes, Polymarket BTC short-horizon markets can be collected and analyzed cleanly
- yes, the `5m` horizon may contain a weak signal
- no, the current evidence does not support a strong or production-ready trading edge

The `5m` horizon is the only one that shows early signs of predictive value in this reference sample, and even there the signal is rare. The `15m` horizon has data, but no threshold-triggered edge in the current run.

## What would improve the study

- backfill additional days or weeks
- tune thresholds by horizon instead of using a single static rule
- test less sparse signal rules
- explicitly compare Polymarket probabilities with the BTC move that has already happened before entry
- validate stability on out-of-sample windows

## Bottom line

This repository now answers the original question in a pragmatic way:

- it finds the real Polymarket markets of interest
- it measures the information value of those markets using a clean `S0 / Se / ST` framework
- it shows a possible `5m` signal, but not yet a robust edge
