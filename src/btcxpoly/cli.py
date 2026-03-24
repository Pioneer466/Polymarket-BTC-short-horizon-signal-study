from __future__ import annotations

import time
from collections import Counter

import pandas as pd
import typer

from btcxpoly.clients import BTCClient, KalshiClient, PolymarketClient
from btcxpoly.config import AppConfig, load_config
from btcxpoly.features import build_features
from btcxpoly.models import BTCCandle, NormalizedObservation
from btcxpoly.storage import LocalStore
from btcxpoly.utils import parse_datetime

app = typer.Typer(add_completion=False, help="BTC x prediction markets research pipeline")


def _config(config_path: str) -> AppConfig:
    return load_config(config_path)


def _btc_price_map(candles: list[BTCCandle]) -> dict[pd.Timestamp, float]:
    return {
        pd.Timestamp(parse_datetime(candle.ts_utc)): candle.close
        for candle in candles
    }


def _select_kalshi_entries(
    entries: list,
    reference_price: float,
    nearest_per_event: int,
) -> list:
    if not entries or nearest_per_event <= 0:
        return entries
    grouped: dict[str, list] = {}
    passthrough: list = []
    for entry in entries:
        if entry.strike is None:
            passthrough.append(entry)
            continue
        key = entry.event_ticker or entry.series_ticker or entry.market_id
        grouped.setdefault(key, []).append(entry)
    selected = list(passthrough)
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: abs((item.strike or reference_price) - reference_price))
        selected.extend(ordered[:nearest_per_event])
    return selected


def _write_market_observations(store: LocalStore, observations: list[NormalizedObservation]) -> None:
    if not observations:
        return
    frame = pd.DataFrame([item.to_record() for item in observations])
    store.write_dataframe("market_observations", frame)


def _write_btc_candles(store: LocalStore, candles: list[BTCCandle]) -> None:
    if not candles:
        return
    frame = pd.DataFrame([candle.to_record() for candle in candles])
    store.write_dataframe("btc_candles", frame)


@app.command("registry-refresh")
def registry_refresh(config_path: str = typer.Option("config.yaml", help="Path to config file")) -> None:
    config = _config(config_path)
    store = LocalStore(config)
    polymarket = PolymarketClient(config)
    kalshi = KalshiClient(config)
    try:
        poly_entries = polymarket.discover_markets()
        kalshi_entries = kalshi.discover_markets()
        registry = poly_entries + kalshi_entries
        poly_type_counts = Counter(entry.market_type for entry in poly_entries)
        poly_closed_count = sum(bool(entry.metadata.get("closed")) for entry in poly_entries)
        poly_active_count = sum(bool(entry.metadata.get("active")) for entry in poly_entries)
        store.save_registry(registry)
        store.write_output_json(
            "registry_summary",
            {
                "total_markets": len(registry),
                "polymarket_markets": len(poly_entries),
                "kalshi_markets": len(kalshi_entries),
                "polymarket_active_markets": poly_active_count,
                "polymarket_closed_markets": poly_closed_count,
                "polymarket_market_types": dict(poly_type_counts),
            },
        )
        typer.echo(f"Saved {len(registry)} markets to {store.registry_path()}")
    finally:
        polymarket.close()
        kalshi.close()


@app.command("backfill")
def backfill(
    start: str = typer.Option(..., help="Start datetime in ISO8601 UTC"),
    end: str = typer.Option(..., help="End datetime in ISO8601 UTC"),
    config_path: str = typer.Option("config.yaml", help="Path to config file"),
) -> None:
    config = _config(config_path)
    store = LocalStore(config)
    registry = store.load_registry()
    if not registry:
        raise typer.BadParameter("Registry is empty. Run registry-refresh first.")

    start_dt = parse_datetime(start)
    end_dt = parse_datetime(end)
    if start_dt is None or end_dt is None or start_dt >= end_dt:
        raise typer.BadParameter("Start and end must be valid ISO8601 datetimes with start < end.")

    btc_client = BTCClient(config)
    polymarket = PolymarketClient(config)
    kalshi = KalshiClient(config)
    try:
        discovered_poly_entries = polymarket.discover_markets_between(start_dt, end_dt)
        if discovered_poly_entries:
            registry_map = {entry.market_id: entry for entry in registry}
            for entry in discovered_poly_entries:
                registry_map[entry.market_id] = entry
            registry = list(registry_map.values())
            store.save_registry(registry)

        btc_candles = btc_client.backfill(start_dt, end_dt)
        _write_btc_candles(store, btc_candles)
        btc_prices = _btc_price_map(btc_candles)

        poly_entries = [entry for entry in registry if entry.venue == "polymarket"]
        kalshi_entries = [entry for entry in registry if entry.venue == "kalshi"]
        if btc_candles:
            reference_price = btc_candles[-1].close
            kalshi_entries = _select_kalshi_entries(
                kalshi_entries,
                reference_price=reference_price,
                nearest_per_event=config.kalshi.nearest_strikes_per_event,
            )

        poly_obs = polymarket.backfill_entries(poly_entries, start_dt, end_dt, btc_prices)
        kalshi_obs = kalshi.backfill_entries(kalshi_entries, start_dt, end_dt, btc_prices)

        _write_market_observations(store, poly_obs + kalshi_obs)
        typer.echo(
            f"Backfill completed. BTC candles={len(btc_candles)}, "
            f"polymarket_registry_added={len(discovered_poly_entries)}, "
            f"observations={len(poly_obs) + len(kalshi_obs)}"
        )
    finally:
        btc_client.close()
        polymarket.close()
        kalshi.close()


@app.command("collect-live")
def collect_live(
    minutes: int = typer.Option(2, min=1, help="Number of minutes to collect"),
    config_path: str = typer.Option("config.yaml", help="Path to config file"),
) -> None:
    config = _config(config_path)
    store = LocalStore(config)
    registry = store.load_registry()
    if not registry:
        raise typer.BadParameter("Registry is empty. Run registry-refresh first.")

    btc_client = BTCClient(config)
    polymarket = PolymarketClient(config)
    kalshi = KalshiClient(config)
    end_monotonic = time.monotonic() + minutes * 60
    ws_done = False
    try:
        poly_entries = [entry for entry in registry if entry.venue == "polymarket"]
        kalshi_entries = [entry for entry in registry if entry.venue == "kalshi"]
        while time.monotonic() < end_monotonic:
            candle = btc_client.latest_candle()
            _write_btc_candles(store, [candle])
            btc_price = candle.close
            kalshi_subset = _select_kalshi_entries(
                kalshi_entries,
                reference_price=btc_price,
                nearest_per_event=config.kalshi.nearest_strikes_per_event,
            )

            rest_observations: list[NormalizedObservation] = []
            rest_observations.extend(polymarket.snapshot_entries(poly_entries, btc_price))
            rest_observations.extend(kalshi.snapshot_entries(kalshi_subset, btc_price))
            _write_market_observations(store, rest_observations)

            if poly_entries and not ws_done:
                ws_observations = polymarket.stream_best_bid_ask(
                    poly_entries,
                    btc_price_ref=btc_price,
                    seconds=config.polling.websocket_batch_seconds,
                )
                _write_market_observations(store, ws_observations)
                ws_done = True

            time.sleep(min(config.polling.kalshi_seconds, max(int(end_monotonic - time.monotonic()), 1)))
        typer.echo("Live collection completed.")
    finally:
        btc_client.close()
        polymarket.close()
        kalshi.close()


@app.command("build-features")
def build_features_command(config_path: str = typer.Option("config.yaml", help="Path to config file")) -> None:
    config = _config(config_path)
    store = LocalStore(config)
    frame = build_features(store, config)
    typer.echo(f"Built feature dataset with {len(frame)} rows.")


@app.command("run-baseline")
def run_baseline_command(config_path: str = typer.Option("config.yaml", help="Path to config file")) -> None:
    from btcxpoly.research import run_baseline

    config = _config(config_path)
    store = LocalStore(config)
    metrics = run_baseline(store, config)
    typer.echo(f"Baseline metrics written to {store.outputs_dir / 'baseline-metrics.json'}")
    typer.echo(str(metrics))


@app.command("run-backtest")
def run_backtest_command(config_path: str = typer.Option("config.yaml", help="Path to config file")) -> None:
    from btcxpoly.backtest import run_backtest

    config = _config(config_path)
    store = LocalStore(config)
    metrics = run_backtest(store, config)
    typer.echo(f"Backtest metrics written to {store.outputs_dir / 'backtest-metrics.json'}")
    typer.echo(str(metrics))
