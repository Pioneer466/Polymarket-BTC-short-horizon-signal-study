from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PathsConfig:
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    normalized_dir: str = "data/normalized"
    registry_dir: str = "data/registry"
    outputs_dir: str = "outputs"


@dataclass(slots=True)
class UniverseConfig:
    horizons_minutes: list[int] = field(default_factory=lambda: [5, 15])


@dataclass(slots=True)
class FiltersConfig:
    min_liquidity: float = 1000.0
    max_spread: float = 0.10


@dataclass(slots=True)
class ThresholdsConfig:
    upper_probability: float = 0.55
    lower_probability: float = 0.45


@dataclass(slots=True)
class PollingConfig:
    kalshi_seconds: int = 60
    btc_seconds: int = 60
    websocket_batch_seconds: int = 90


@dataclass(slots=True)
class ResearchConfig:
    benchmark_test_fraction: float = 0.30
    fee_bps: float = 1.0
    slippage_bps: float = 2.0
    lead_lag_window_minutes: int = 60


@dataclass(slots=True)
class StrategyConfig:
    primary_venue: str = "polymarket"
    min_entry_offset_minutes: int = 1
    max_entry_fraction: float = 0.70
    min_rows_per_offset: int = 10


@dataclass(slots=True)
class PolymarketConfig:
    gamma_url: str = "https://gamma-api.polymarket.com"
    clob_url: str = "https://clob.polymarket.com"
    data_url: str = "https://data-api.polymarket.com"
    websocket_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    include_active: bool = True
    include_closed: bool = True
    recurring_slug_lookback_hours: int = 24
    seed_event_slugs: list[str] = field(default_factory=list)
    search_terms: list[str] = field(
        default_factory=lambda: [
            "bitcoin",
            "btc",
            "bitcoin up or down",
            "btc up or down",
            "bitcoin 5 minutes",
            "bitcoin 15 minutes",
            "btc 5 minutes",
            "btc 15 minutes",
            "bitcoin above",
            "bitcoin below",
            "btc above",
            "btc below",
        ]
    )
    max_results_per_search: int = 200
    discovery_pages: int = 10


@dataclass(slots=True)
class KalshiConfig:
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    series_tickers: list[str] = field(default_factory=lambda: ["KXBTCD"])
    page_limit: int = 200
    nearest_strikes_per_event: int = 3


@dataclass(slots=True)
class BTCConfig:
    binance_url: str = "https://api.binance.com"
    coinbase_url: str = "https://api.exchange.coinbase.com"
    symbol: str = "BTCUSDT"
    product_id: str = "BTC-USD"


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    polling: PollingConfig = field(default_factory=PollingConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    btc: BTCConfig = field(default_factory=BTCConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping")
    return data


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    section = data.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    return section


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    raw = _load_yaml(Path(path))
    return AppConfig(
        paths=PathsConfig(**_section(raw, "paths")),
        universe=UniverseConfig(**_section(raw, "universe")),
        filters=FiltersConfig(**_section(raw, "filters")),
        thresholds=ThresholdsConfig(**_section(raw, "thresholds")),
        polling=PollingConfig(**_section(raw, "polling")),
        research=ResearchConfig(**_section(raw, "research")),
        strategy=StrategyConfig(**_section(raw, "strategy")),
        polymarket=PolymarketConfig(**_section(raw, "polymarket")),
        kalshi=KalshiConfig(**_section(raw, "kalshi")),
        btc=BTCConfig(**_section(raw, "btc")),
    )
