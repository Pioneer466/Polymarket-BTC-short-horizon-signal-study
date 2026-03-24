from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, roc_curve

from btcxpoly.config import AppConfig
from btcxpoly.storage import LocalStore


def _evaluate_offset(frame: pd.DataFrame) -> dict[str, Any]:
    actionable = frame[frame["signal_polymarket"] != 0].copy()
    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "trade_coverage": float((frame["signal_polymarket"] != 0).mean()) if not frame.empty else 0.0,
        "avg_confidence": float(frame["confidence"].mean()) if not frame.empty else None,
        "contract_threshold_accuracy": None,
        "trade_threshold_accuracy": None,
        "contract_probability_brier": None,
        "avg_trade_return": None,
        "avg_open_to_entry_return": None,
        "avg_open_to_close_return": None,
    }
    if not frame.empty:
        result["contract_probability_brier"] = float(
            brier_score_loss(frame["contract_outcome"], frame["p_poly"].fillna(0.5))
        )
        result["avg_open_to_entry_return"] = float(frame["return_open_to_entry"].mean())
        result["avg_open_to_close_return"] = float(frame["return_open_to_close"].mean())
    if actionable.empty:
        return result
    predictions = (actionable["signal_polymarket"] > 0).astype(int)
    result["contract_threshold_accuracy"] = float(accuracy_score(actionable["contract_outcome"], predictions))
    result["trade_threshold_accuracy"] = float(accuracy_score(actionable["trade_outcome"], predictions))
    result["avg_trade_return"] = float((actionable["signal_polymarket"] * actionable["return_entry_to_close"]).mean())
    return result


def _offset_score(result: dict[str, Any]) -> float:
    accuracy = result.get("trade_threshold_accuracy")
    coverage = result.get("trade_coverage") or 0.0
    confidence = result.get("avg_confidence") or 0.0
    if accuracy is None:
        return 0.0
    return max(accuracy - 0.5, 0.0) * coverage * confidence


def _plot_offset_summary(summary: pd.DataFrame, horizon: int, output_dir: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(
        summary["entry_offset_minutes"],
        summary["trade_threshold_accuracy"],
        marker="o",
        color="#134074",
        label="Trade Accuracy",
    )
    ax1.set_xlabel("Entry offset after market open (minutes)")
    ax1.set_ylabel("Accuracy", color="#134074")
    ax1.tick_params(axis="y", labelcolor="#134074")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(summary["entry_offset_minutes"], summary["trade_coverage"], marker="s", color="#0b6e4f", label="Coverage")
    ax2.set_ylabel("Coverage", color="#0b6e4f")
    ax2.tick_params(axis="y", labelcolor="#0b6e4f")

    ax1.set_title(f"Polymarket entry offset study ({horizon}m horizon)")
    path = output_dir / f"entry-offset-{horizon}m.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_calibration(truth: pd.Series, probability: pd.Series, horizon: int, output_dir: Path) -> Path:
    prob_true, prob_pred = calibration_curve(truth, probability, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7d8597", label="Perfect")
    ax.set_title(f"Calibration ({horizon}m horizon)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    path = output_dir / f"calibration-{horizon}m.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_roc(truth: pd.Series, probability: pd.Series, horizon: int, output_dir: Path) -> Path:
    fpr, tpr, _ = roc_curve(truth, probability)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label="Benchmark")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7d8597")
    ax.set_title(f"ROC ({horizon}m horizon)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    ax.grid(alpha=0.3)
    path = output_dir / f"roc-{horizon}m.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_baseline(store: LocalStore, config: AppConfig) -> dict[str, Any]:
    features = store.load_dataset("features")
    if features.empty:
        metrics = {str(horizon): {"status": "no_data"} for horizon in config.universe.horizons_minutes}
        store.write_output_json("baseline_metrics", metrics)
        return metrics
    output_dir = store.outputs_dir
    metrics: dict[str, Any] = {}

    for horizon in config.universe.horizons_minutes:
        frame = features[features["horizon_minutes"] == horizon].copy()
        frame = frame.dropna(
            subset=[
                "p_poly",
                "contract_outcome",
                "trade_outcome",
                "return_entry_to_close",
                "entry_offset_minutes",
                "return_open_to_entry",
            ]
        )
        frame = frame.sort_values(["open_time", "market_id", "entry_offset_minutes"])
        if frame.empty:
            metrics[str(horizon)] = {"status": "no_data"}
            continue

        markets = (
            frame[["market_id", "open_time"]]
            .drop_duplicates()
            .sort_values("open_time")
            .reset_index(drop=True)
        )
        split_idx = int(len(markets) * (1.0 - config.research.benchmark_test_fraction))
        split_idx = min(max(split_idx, 1), len(markets) - 1) if len(markets) > 1 else 1
        train_ids = set(markets.iloc[:split_idx]["market_id"])
        test_ids = set(markets.iloc[split_idx:]["market_id"]) if len(markets) > 1 else set(markets["market_id"])

        train = frame[frame["market_id"].isin(train_ids)].copy()
        test = frame[frame["market_id"].isin(test_ids)].copy()

        summary_rows: list[dict[str, Any]] = []
        for offset in sorted(frame["entry_offset_minutes"].dropna().astype(int).unique()):
            train_offset = train[train["entry_offset_minutes"] == offset].copy()
            test_offset = test[test["entry_offset_minutes"] == offset].copy()
            if len(train_offset) < config.strategy.min_rows_per_offset:
                continue
            train_eval = _evaluate_offset(train_offset)
            test_eval = _evaluate_offset(test_offset) if not test_offset.empty else _evaluate_offset(train_offset)
            summary_rows.append(
                {
                    "entry_offset_minutes": offset,
                    "train_rows": train_eval["rows"],
                    "trade_coverage": test_eval["trade_coverage"],
                    "contract_threshold_accuracy": test_eval["contract_threshold_accuracy"],
                    "trade_threshold_accuracy": test_eval["trade_threshold_accuracy"],
                    "avg_confidence": test_eval["avg_confidence"],
                    "contract_probability_brier": test_eval["contract_probability_brier"],
                    "avg_trade_return": test_eval["avg_trade_return"],
                    "avg_open_to_entry_return": test_eval["avg_open_to_entry_return"],
                    "avg_open_to_close_return": test_eval["avg_open_to_close_return"],
                    "selection_score": _offset_score(train_eval),
                }
            )

        summary = pd.DataFrame(summary_rows)
        if summary.empty:
            metrics[str(horizon)] = {"status": "no_candidate_offsets"}
            continue

        summary = summary.sort_values(["selection_score", "entry_offset_minutes"], ascending=[False, True])
        best_offset = int(summary.iloc[0]["entry_offset_minutes"])
        store.write_output_table(f"entry_offset_{horizon}m", summary)
        _plot_offset_summary(summary.sort_values("entry_offset_minutes"), horizon, output_dir)

        selected = test[test["entry_offset_minutes"] == best_offset].copy()
        if selected.empty:
            selected = frame[frame["entry_offset_minutes"] == best_offset].copy()

        result = _evaluate_offset(selected)
        result["best_entry_offset_minutes"] = best_offset
        result["benchmark_auc"] = None
        result["benchmark_brier"] = None

        model_features = [
            "p_poly",
            "confidence",
            "probability_change",
            "btc_return_1m",
            "btc_vol_15m",
            "remaining_minutes",
            "entry_delay_fraction",
            "return_open_to_entry",
            "log_return_open_to_entry",
        ]
        active_features = [column for column in model_features if selected[column].notna().any()]
        if len(active_features) >= 2:
            benchmark_frame = frame[frame["entry_offset_minutes"] == best_offset].dropna(
                subset=active_features + ["trade_outcome"]
            )
            benchmark_markets = (
                benchmark_frame[["market_id", "open_time"]]
                .drop_duplicates()
                .sort_values("open_time")
                .reset_index(drop=True)
            )
            if len(benchmark_markets) >= 10 and benchmark_frame["trade_outcome"].nunique() > 1:
                split_idx = int(len(benchmark_markets) * (1.0 - config.research.benchmark_test_fraction))
                split_idx = min(max(split_idx, 1), len(benchmark_markets) - 1)
                train_ids = set(benchmark_markets.iloc[:split_idx]["market_id"])
                test_ids = set(benchmark_markets.iloc[split_idx:]["market_id"])
                train_model = benchmark_frame[benchmark_frame["market_id"].isin(train_ids)]
                test_model = benchmark_frame[benchmark_frame["market_id"].isin(test_ids)]
                if (
                    not train_model.empty
                    and not test_model.empty
                    and train_model["trade_outcome"].nunique() > 1
                    and test_model["trade_outcome"].nunique() > 1
                ):
                    model = LogisticRegression(max_iter=200)
                    model.fit(train_model[active_features], train_model["trade_outcome"])
                    probabilities = model.predict_proba(test_model[active_features])[:, 1]
                    result["benchmark_auc"] = float(roc_auc_score(test_model["trade_outcome"], probabilities))
                    result["benchmark_brier"] = float(
                        brier_score_loss(test_model["trade_outcome"], probabilities)
                    )
                    _plot_calibration(test_model["trade_outcome"], pd.Series(probabilities), horizon, output_dir)
                    _plot_roc(test_model["trade_outcome"], pd.Series(probabilities), horizon, output_dir)

        metrics[str(horizon)] = result

    store.write_output_json("baseline_metrics", metrics)
    return metrics
