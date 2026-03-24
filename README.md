# Polymarket BTC Short-Horizon Signal Study

Research repo to test one question: can Polymarket BTC `Up/Down` markets on 5-minute and 15-minute horizons provide a usable short-term signal for Bitcoin?

The repo is focused on research, not live trading. Polymarket is treated as a read-only signal source, and BTC is the traded reference asset.

## Study summary

The full write-up of the idea, method, charts, and conclusions is in [docs/polymarket-btc-study.md](docs/polymarket-btc-study.md).

A GitHub-friendly copy of the main generated charts and metrics is stored in `docs/results/`.

## What the code does

- discovers recurring Polymarket BTC `Up/Down` 5m and 15m events
- backfills BTC candles and Polymarket market data
- stores normalized research datasets locally
- builds features around three BTC prices:
  - market open
  - chosen entry time after open
  - market close
- tests entry offsets after market open
- evaluates contract correctness and trade correctness separately
- runs a simple baseline study and a paper backtest

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

btcxpoly registry-refresh
btcxpoly backfill --start 2026-03-22T15:00:00Z --end 2026-03-23T15:30:00Z
btcxpoly build-features
btcxpoly run-baseline
btcxpoly run-backtest
```

## CLI

- `btcxpoly registry-refresh`
- `btcxpoly backfill --start <ISO8601> --end <ISO8601>`
- `btcxpoly collect-live --minutes <int>`
- `btcxpoly build-features`
- `btcxpoly run-baseline`
- `btcxpoly run-backtest`

## Repo layout

- `src/btcxpoly/`: pipeline code
- `tests/`: unit and pipeline tests
- `notebooks/`: optional exploratory notebooks
- `docs/`: human-readable study notes and selected results for publication
- `data/` and `outputs/`: local generated artifacts, intentionally not committed

## Notes

- Research only. No live execution.
- Public read-only APIs only.
- The active strategy layer is Polymarket-only.
- The backfill step discovers recurring Polymarket `btc-updown-5m` and `btc-updown-15m` event slugs inside the requested time window.
