from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from btcxpoly.config import AppConfig
from btcxpoly.storage import LocalStore


def _load_best_offsets(store: LocalStore) -> dict[str, int]:
    path = store.outputs_dir / "baseline-metrics.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    best_offsets: dict[str, int] = {}
    for horizon, metrics in payload.items():
        if isinstance(metrics, dict) and metrics.get("best_entry_offset_minutes") is not None:
            best_offsets[horizon] = int(metrics["best_entry_offset_minutes"])
    return best_offsets


def _backtest_horizon(
    frame: pd.DataFrame,
    horizon: int,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = frame.sort_values("ts_utc").reset_index(drop=True)
    cost = 2.0 * (fee_bps + slippage_bps) / 10_000.0
    trades: list[dict[str, Any]] = []
    equity = 1.0
    last_exit_time = None

    for _, row in working.iterrows():
        signal = int(row.get("signal_polymarket", 0))
        future_return = row.get("return_entry_to_close")
        close_time = row.get("close_time")
        if signal == 0 or pd.isna(future_return) or pd.isna(close_time):
            continue
        if last_exit_time is not None and row["ts_utc"] < last_exit_time:
            continue
        gross = float(signal) * float(future_return)
        net = gross - cost
        equity *= 1.0 + net
        last_exit_time = close_time
        trades.append(
            {
                "ts_utc": row["ts_utc"],
                "close_time": close_time,
                "market_id": row["market_id"],
                "horizon_minutes": horizon,
                "entry_offset_minutes": row["entry_offset_minutes"],
                "signal": signal,
                "btc_open_at_market_open": row["btc_open_at_market_open"],
                "entry_price": row["btc_entry_price"],
                "exit_price": row["btc_close_at_market_end"],
                "return_open_to_entry": row["return_open_to_entry"],
                "return_entry_to_close": row["return_entry_to_close"],
                "return_open_to_close": row["return_open_to_close"],
                "gross_return": gross,
                "net_return": net,
                "equity": equity,
            }
        )

    trades_frame = pd.DataFrame(trades)
    if trades_frame.empty:
        return trades_frame, {
            "trades": 0,
            "win_rate": None,
            "cumulative_return": None,
            "average_trade_return": None,
            "best_entry_offset_minutes": None,
        }

    metrics = {
        "trades": int(len(trades_frame)),
        "win_rate": float((trades_frame["net_return"] > 0).mean()),
        "cumulative_return": float(trades_frame["equity"].iloc[-1] - 1.0),
        "average_trade_return": float(trades_frame["net_return"].mean()),
    }
    return trades_frame, metrics


def _plot_equity(frame: pd.DataFrame, horizon: int, output_dir: Path) -> Path | None:
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frame["ts_utc"], frame["equity"], color="#0b6e4f")
    ax.set_title(f"Equity curve ({horizon}m horizon)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    path = output_dir / f"equity-{horizon}m.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_backtest(store: LocalStore, config: AppConfig) -> dict[str, Any]:
    features = store.load_dataset("features")
    if features.empty:
        metrics = {str(horizon): {"status": "no_data"} for horizon in config.universe.horizons_minutes}
        store.write_output_json("backtest_metrics", metrics)
        return metrics

    best_offsets = _load_best_offsets(store)
    metrics: dict[str, Any] = {}
    all_trades: list[pd.DataFrame] = []

    for horizon in config.universe.horizons_minutes:
        frame = features[features["horizon_minutes"] == horizon].copy()
        if frame.empty:
            metrics[str(horizon)] = {"status": "no_data"}
            continue
        best_offset = best_offsets.get(str(horizon))
        if best_offset is None:
            available = sorted(frame["entry_offset_minutes"].dropna().astype(int).unique())
            best_offset = int(available[0]) if available else None
        if best_offset is None:
            metrics[str(horizon)] = {"status": "no_offsets"}
            continue
        best_offset = int(best_offset)

        selected = frame[frame["entry_offset_minutes"] == best_offset].copy()
        trades, horizon_metrics = _backtest_horizon(
            selected,
            horizon=horizon,
            fee_bps=config.research.fee_bps,
            slippage_bps=config.research.slippage_bps,
        )
        horizon_metrics["best_entry_offset_minutes"] = best_offset
        if not trades.empty:
            _plot_equity(trades, horizon, store.outputs_dir)
            all_trades.append(trades)
        metrics[str(horizon)] = horizon_metrics

    if all_trades:
        trades_frame = pd.concat(all_trades, ignore_index=True)
        store.replace_dataframe("backtest_trades", trades_frame)
        store.write_output_table("backtest_trades", trades_frame)
    store.write_output_json("backtest_metrics", metrics)
    return metrics
