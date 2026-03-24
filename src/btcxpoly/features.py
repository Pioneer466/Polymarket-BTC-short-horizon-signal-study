from __future__ import annotations

import math

import numpy as np
import pandas as pd

from btcxpoly.config import AppConfig
from btcxpoly.storage import LocalStore
from btcxpoly.utils import implied_drift


def _candidate_offsets(horizon_minutes: int, config: AppConfig) -> list[int]:
    max_offset = int(math.floor(horizon_minutes * config.strategy.max_entry_fraction))
    max_offset = min(max_offset, max(horizon_minutes - 1, config.strategy.min_entry_offset_minutes))
    max_offset = max(max_offset, config.strategy.min_entry_offset_minutes)
    if max_offset < config.strategy.min_entry_offset_minutes:
        return []
    return list(range(config.strategy.min_entry_offset_minutes, max_offset + 1))


def _prepare_btc_frame(store: LocalStore) -> pd.DataFrame:
    btc = store.load_dataset("btc_candles")
    if btc.empty:
        raise ValueError("No BTC candles found. Run backfill or collect-live first.")
    btc = btc.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last")
    btc["ts_utc"] = pd.to_datetime(btc["ts_utc"], utc=True)
    btc["btc_return_1m"] = btc["close"].pct_change()
    btc["btc_vol_15m"] = btc["btc_return_1m"].rolling(15).std()
    btc["btc_vol_30m"] = btc["btc_return_1m"].rolling(30).std()
    btc["btc_boundary_price"] = btc["open"]
    return btc


def _prepare_registry_frame(store: LocalStore, venue: str) -> pd.DataFrame:
    entries = [entry.to_record() for entry in store.load_registry() if entry.venue == venue]
    if not entries:
        return pd.DataFrame()
    registry = pd.DataFrame(entries)
    registry["open_time"] = pd.to_datetime(registry["open_time"], utc=True)
    registry["close_time"] = pd.to_datetime(registry["close_time"], utc=True)
    return registry[
        [
            "market_id",
            "question",
            "market_type",
            "horizon_minutes",
            "open_time",
            "close_time",
            "event_ticker",
            "rules_hash",
        ]
    ].rename(
        columns={
            "question": "registry_question",
            "market_type": "registry_market_type",
            "horizon_minutes": "registry_horizon_minutes",
            "event_ticker": "registry_event_ticker",
            "rules_hash": "registry_rules_hash",
        }
    )


def _prepare_market_observations(store: LocalStore, config: AppConfig, btc: pd.DataFrame) -> pd.DataFrame:
    venue = config.strategy.primary_venue
    market_obs = store.load_dataset("market_observations")
    if market_obs.empty:
        raise ValueError("No market observations found. Run backfill or collect-live first.")

    market_obs["ts_utc"] = pd.to_datetime(market_obs["ts_utc"], utc=True)
    market_obs = market_obs[market_obs["venue"] == venue].copy()
    if market_obs.empty:
        return market_obs

    registry = _prepare_registry_frame(store, venue)
    if registry.empty:
        return pd.DataFrame()

    market_obs = market_obs.merge(registry, on="market_id", how="inner")
    market_obs["market_type"] = market_obs["registry_market_type"]
    market_obs["horizon_minutes"] = market_obs["horizon_minutes"].fillna(market_obs["registry_horizon_minutes"])
    market_obs["question"] = market_obs["question"].fillna(market_obs["registry_question"])
    market_obs["event_ticker"] = market_obs["event_ticker"].fillna(market_obs["registry_event_ticker"])
    market_obs["rules_hash"] = market_obs["rules_hash"].fillna(market_obs["registry_rules_hash"])

    market_obs["market_total_minutes"] = (
        (market_obs["close_time"] - market_obs["open_time"]).dt.total_seconds() / 60.0
    )
    market_obs["elapsed_minutes"] = np.floor(
        (market_obs["ts_utc"] - market_obs["open_time"]).dt.total_seconds() / 60.0
    )
    market_obs["remaining_minutes"] = np.floor(
        (market_obs["close_time"] - market_obs["ts_utc"]).dt.total_seconds() / 60.0
    )
    market_obs = market_obs[
        (market_obs["market_total_minutes"] > 0)
        & (market_obs["elapsed_minutes"] >= 0)
        & (market_obs["remaining_minutes"] > 0)
        & (market_obs["horizon_minutes"].isin(config.universe.horizons_minutes))
        & (market_obs["market_type"] == "binary_directional")
    ].copy()

    btc_entry = btc[
        ["ts_utc", "btc_boundary_price", "btc_return_1m", "btc_vol_15m", "btc_vol_30m"]
    ].rename(columns={"btc_boundary_price": "btc_entry_price"})
    btc_open = btc[["ts_utc", "btc_boundary_price"]].rename(
        columns={"ts_utc": "market_open_floor", "btc_boundary_price": "btc_open_at_market_open"}
    )
    btc_exit = btc[["ts_utc", "btc_boundary_price"]].rename(
        columns={"ts_utc": "market_close_floor", "btc_boundary_price": "btc_close_at_market_end"}
    )

    market_obs["market_open_floor"] = market_obs["open_time"].dt.floor("min")
    market_obs["market_close_floor"] = market_obs["close_time"].dt.floor("min")
    market_obs = market_obs.merge(btc_entry, on="ts_utc", how="left")
    market_obs = market_obs.merge(btc_open, on="market_open_floor", how="left")
    market_obs = market_obs.merge(btc_exit, on="market_close_floor", how="left")
    market_obs = market_obs.sort_values(["market_id", "ts_utc"]).copy()
    market_obs["probability_change"] = market_obs.groupby("market_id")["side_up_probability_mid"].diff()
    market_obs["confidence"] = (market_obs["side_up_probability_mid"] - 0.5).abs()
    market_obs["return_open_to_entry"] = (
        market_obs["btc_entry_price"] / market_obs["btc_open_at_market_open"] - 1.0
    )
    market_obs["return_entry_to_close"] = (
        market_obs["btc_close_at_market_end"] / market_obs["btc_entry_price"] - 1.0
    )
    market_obs["return_open_to_close"] = (
        market_obs["btc_close_at_market_end"] / market_obs["btc_open_at_market_open"] - 1.0
    )
    market_obs["log_return_open_to_entry"] = np.log(
        market_obs["btc_entry_price"] / market_obs["btc_open_at_market_open"]
    )
    market_obs["log_return_entry_to_close"] = np.log(
        market_obs["btc_close_at_market_end"] / market_obs["btc_entry_price"]
    )
    market_obs["log_return_open_to_close"] = np.log(
        market_obs["btc_close_at_market_end"] / market_obs["btc_open_at_market_open"]
    )
    market_obs["contract_outcome"] = (market_obs["btc_close_at_market_end"] >= market_obs["btc_open_at_market_open"]).astype(float)
    market_obs["trade_outcome"] = (market_obs["btc_close_at_market_end"] >= market_obs["btc_entry_price"]).astype(float)
    market_obs["spread"] = market_obs["spread"].astype(float)
    return market_obs


def _select_entry_candidates(observations: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for horizon in config.universe.horizons_minutes:
        horizon_obs = observations[observations["horizon_minutes"] == horizon].copy()
        if horizon_obs.empty:
            continue
        offsets = _candidate_offsets(horizon, config)
        for _, group in horizon_obs.groupby("market_id", sort=False):
            group = group.sort_values("ts_utc")
            for offset in offsets:
                eligible = group[group["elapsed_minutes"] >= offset]
                if eligible.empty:
                    continue
                row = eligible.iloc[0].copy()
                if row["remaining_minutes"] <= 0:
                    continue
                row["entry_offset_minutes"] = offset
                row["entry_delay_fraction"] = offset / max(row["market_total_minutes"], 1.0)
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_features(store: LocalStore, config: AppConfig) -> pd.DataFrame:
    btc = _prepare_btc_frame(store)
    observations = _prepare_market_observations(store, config, btc)
    if observations.empty:
        frame = pd.DataFrame()
        store.replace_dataframe("features", frame)
        return frame

    frame = _select_entry_candidates(observations, config)
    if frame.empty:
        store.replace_dataframe("features", frame)
        return frame

    spread_ok = frame["spread"].isna() | (frame["spread"] <= config.filters.max_spread)
    liquidity_ok = frame["liquidity"].fillna(0.0) >= config.filters.min_liquidity
    active_filters = spread_ok & liquidity_ok

    frame["p_poly"] = frame["side_up_probability_mid"]
    frame["p_kalshi"] = np.nan
    frame["p_consensus"] = frame["p_poly"]
    frame["p_consensus_weighted"] = frame["p_poly"]
    frame["cross_venue_spread"] = np.nan
    frame["liquidity_consensus"] = frame["liquidity"].fillna(0.0)
    frame["signal_polymarket"] = np.select(
        [
            active_filters & (frame["p_poly"] >= config.thresholds.upper_probability),
            active_filters & (frame["p_poly"] <= config.thresholds.lower_probability),
        ],
        [1, -1],
        default=0,
    )
    frame["consensus_signal"] = frame["signal_polymarket"]
    frame["implied_drift_poly"] = [
        implied_drift(probability, sigma, max(int(remaining), 1))
        for probability, sigma, remaining in zip(
            frame["p_poly"],
            frame["btc_vol_15m"],
            frame["remaining_minutes"].fillna(1),
        )
    ]
    frame["implied_drift_consensus"] = frame["implied_drift_poly"]
    frame["target_up"] = frame["trade_outcome"]
    frame["future_close"] = frame["btc_close_at_market_end"]

    selected_columns = [
        "market_id",
        "question",
        "event_ticker",
        "rules_hash",
        "ts_utc",
        "open_time",
        "close_time",
        "market_total_minutes",
        "elapsed_minutes",
        "remaining_minutes",
        "entry_offset_minutes",
        "entry_delay_fraction",
        "horizon_minutes",
        "p_poly",
        "p_kalshi",
        "p_consensus",
        "p_consensus_weighted",
        "confidence",
        "probability_change",
        "spread",
        "liquidity",
        "liquidity_consensus",
        "btc_open_at_market_open",
        "btc_entry_price",
        "btc_close_at_market_end",
        "btc_return_1m",
        "btc_vol_15m",
        "btc_vol_30m",
        "return_open_to_entry",
        "return_entry_to_close",
        "return_open_to_close",
        "log_return_open_to_entry",
        "log_return_entry_to_close",
        "log_return_open_to_close",
        "future_close",
        "target_up",
        "contract_outcome",
        "trade_outcome",
        "signal_polymarket",
        "consensus_signal",
        "implied_drift_poly",
        "implied_drift_consensus",
    ]
    frame = frame[selected_columns].sort_values(["horizon_minutes", "market_id", "entry_offset_minutes"])
    store.replace_dataframe("features", frame)
    return frame
