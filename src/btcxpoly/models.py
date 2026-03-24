from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from btcxpoly.utils import json_dumps


@dataclass(slots=True)
class MarketRegistryEntry:
    venue: str
    market_id: str
    question: str
    horizon_minutes: int | None
    market_type: str
    open_time: str | None = None
    close_time: str | None = None
    expected_expiration_time: str | None = None
    event_id: str | None = None
    event_ticker: str | None = None
    series_ticker: str | None = None
    yes_token_id: str | None = None
    no_token_id: str | None = None
    strike: float | None = None
    rules_primary: str | None = None
    rules_secondary: str | None = None
    rules_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "venue": self.venue,
            "market_id": self.market_id,
            "question": self.question,
            "horizon_minutes": self.horizon_minutes,
            "market_type": self.market_type,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "expected_expiration_time": self.expected_expiration_time,
            "event_id": self.event_id,
            "event_ticker": self.event_ticker,
            "series_ticker": self.series_ticker,
            "yes_token_id": self.yes_token_id,
            "no_token_id": self.no_token_id,
            "strike": self.strike,
            "rules_primary": self.rules_primary,
            "rules_secondary": self.rules_secondary,
            "rules_hash": self.rules_hash,
            "metadata_json": json_dumps(self.metadata),
        }


@dataclass(slots=True)
class NormalizedObservation:
    ts_utc: str
    venue: str
    market_id: str
    question: str
    horizon_minutes: int | None
    side_up_probability_mid: float | None
    yes_bid: float | None
    yes_ask: float | None
    no_bid: float | None = None
    no_ask: float | None = None
    spread: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    liquidity: float | None = None
    btc_price_ref: float | None = None
    strike: float | None = None
    rules_hash: str | None = None
    event_ticker: str | None = None
    series_ticker: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "ts_utc": self.ts_utc,
            "venue": self.venue,
            "market_id": self.market_id,
            "question": self.question,
            "horizon_minutes": self.horizon_minutes,
            "side_up_probability_mid": self.side_up_probability_mid,
            "yes_bid": self.yes_bid,
            "yes_ask": self.yes_ask,
            "no_bid": self.no_bid,
            "no_ask": self.no_ask,
            "spread": self.spread,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "liquidity": self.liquidity,
            "btc_price_ref": self.btc_price_ref,
            "strike": self.strike,
            "rules_hash": self.rules_hash,
            "event_ticker": self.event_ticker,
            "series_ticker": self.series_ticker,
            "source": self.source,
            "metadata_json": json_dumps(self.metadata),
        }


@dataclass(slots=True)
class BTCCandle:
    ts_utc: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str

    def to_record(self) -> dict[str, Any]:
        return {
            "ts_utc": self.ts_utc,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "source": self.source,
        }
