from pathlib import Path

import pandas as pd

from btcxpoly.backtest import run_backtest
from btcxpoly.config import (
    AppConfig,
    BTCConfig,
    FiltersConfig,
    KalshiConfig,
    PathsConfig,
    PolymarketConfig,
    ResearchConfig,
    StrategyConfig,
    ThresholdsConfig,
    UniverseConfig,
)
from btcxpoly.features import build_features
from btcxpoly.models import MarketRegistryEntry
from btcxpoly.research import run_baseline
from btcxpoly.storage import LocalStore


def _test_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(
            data_dir=str(tmp_path / "data"),
            raw_dir=str(tmp_path / "data" / "raw"),
            normalized_dir=str(tmp_path / "data" / "normalized"),
            registry_dir=str(tmp_path / "data" / "registry"),
            outputs_dir=str(tmp_path / "outputs"),
        ),
        universe=UniverseConfig(horizons_minutes=[5, 15]),
        filters=FiltersConfig(min_liquidity=500.0, max_spread=0.10),
        thresholds=ThresholdsConfig(upper_probability=0.55, lower_probability=0.45),
        research=ResearchConfig(benchmark_test_fraction=0.3, fee_bps=1.0, slippage_bps=1.0, lead_lag_window_minutes=10),
        strategy=StrategyConfig(primary_venue="polymarket", min_entry_offset_minutes=1, max_entry_fraction=0.7, min_rows_per_offset=3),
        polymarket=PolymarketConfig(),
        kalshi=KalshiConfig(),
        btc=BTCConfig(),
    )


def test_polymarket_delayed_entry_pipeline(tmp_path: Path) -> None:
    config = _test_config(tmp_path)
    store = LocalStore(config)

    timestamps = pd.date_range("2026-03-23T00:00:00Z", periods=220, freq="1min")
    close_prices = []
    price = 100.0
    move_map = {}
    for market_idx in range(12):
        start_idx = market_idx * 10
        move_map[start_idx] = 1 if market_idx % 2 == 0 else -1
    for idx, _ in enumerate(timestamps):
        if idx in move_map:
            direction = move_map[idx]
            price += direction * 1.0
        elif idx - 1 in move_map or idx - 2 in move_map or idx - 3 in move_map or idx - 4 in move_map:
            start_candidates = [candidate for candidate in move_map if 0 < idx - candidate <= 4]
            if start_candidates:
                price += move_map[start_candidates[-1]] * 1.0
        close_prices.append(price)

    btc = pd.DataFrame(
        {
            "ts_utc": timestamps,
            "open": close_prices,
            "high": [value + 0.5 for value in close_prices],
            "low": [value - 0.5 for value in close_prices],
            "close": close_prices,
            "volume": [10.0] * len(timestamps),
            "source": ["binance"] * len(timestamps),
        }
    )
    store.write_dataframe("btc_candles", btc)

    registry_entries: list[MarketRegistryEntry] = []
    observations = []
    for market_idx in range(12):
        start_ts = timestamps[market_idx * 10]
        end_ts = start_ts + pd.Timedelta(minutes=5)
        market_id = f"poly-{market_idx}"
        registry_entries.append(
            MarketRegistryEntry(
                venue="polymarket",
                market_id=market_id,
                question=f"BTC Up or Down {market_idx}",
                horizon_minutes=5,
                market_type="binary_directional",
                open_time=start_ts.isoformat(),
                close_time=end_ts.isoformat(),
                event_ticker=f"evt-{market_idx}",
                yes_token_id=f"yes-{market_idx}",
                no_token_id=f"no-{market_idx}",
                rules_hash=f"rules-{market_idx}",
                metadata={"liquidity": 2000},
            )
        )
        up_market = market_idx % 2 == 0
        probability_path = [0.54, 0.60, 0.68] if up_market else [0.46, 0.40, 0.32]
        for offset, probability in enumerate(probability_path, start=1):
            ts = start_ts + pd.Timedelta(minutes=offset)
            observations.append(
                {
                    "ts_utc": ts,
                    "venue": "polymarket",
                    "market_id": market_id,
                    "question": f"BTC Up or Down {market_idx}",
                    "horizon_minutes": 5,
                    "side_up_probability_mid": probability,
                    "yes_bid": max(probability - 0.01, 0.01),
                    "yes_ask": min(probability + 0.01, 0.99),
                    "spread": 0.02,
                    "volume": 1000,
                    "open_interest": 500,
                    "liquidity": 2000,
                    "btc_price_ref": None,
                    "rules_hash": f"rules-{market_idx}",
                    "event_ticker": f"evt-{market_idx}",
                    "series_ticker": None,
                    "source": "test",
                    "metadata_json": "{}",
                }
            )

    store.save_registry(registry_entries)
    store.write_dataframe("market_observations", pd.DataFrame(observations))

    features = build_features(store, config)
    assert not features.empty
    assert set(features["entry_offset_minutes"].unique()) == {1, 2, 3}
    assert features["signal_polymarket"].abs().sum() > 0
    assert {"btc_open_at_market_open", "btc_entry_price", "btc_close_at_market_end"}.issubset(features.columns)
    assert {"return_open_to_entry", "return_entry_to_close", "return_open_to_close"}.issubset(features.columns)
    assert {"contract_outcome", "trade_outcome"}.issubset(features.columns)
    assert features["btc_open_at_market_open"].notna().all()
    assert features["btc_entry_price"].notna().all()
    assert features["btc_close_at_market_end"].notna().all()
    assert (features["contract_outcome"].isin([0.0, 1.0])).all()
    assert (features["trade_outcome"].isin([0.0, 1.0])).all()

    baseline = run_baseline(store, config)
    assert baseline["5"]["best_entry_offset_minutes"] in {1, 2, 3}
    assert baseline["5"]["contract_probability_brier"] is not None
    assert baseline["5"]["trade_threshold_accuracy"] is not None

    backtest = run_backtest(store, config)
    assert backtest["5"]["trades"] > 0
