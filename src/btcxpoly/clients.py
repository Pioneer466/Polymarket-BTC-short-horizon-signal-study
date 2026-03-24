from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pandas as pd
import websockets

from btcxpoly.config import AppConfig
from btcxpoly.models import BTCCandle, MarketRegistryEntry, NormalizedObservation
from btcxpoly.utils import (
    coalesce,
    ensure_utc,
    floor_to_minute,
    infer_horizon_minutes,
    json_dumps,
    midpoint,
    parse_datetime,
    parse_decimal,
    safe_json_loads,
    sha256_text,
    spread,
)


class BaseHTTPClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"User-Agent": "btcxpoly/0.1.0"},
        )

    def request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.request(method, path, **kwargs)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(1.0 + attempt)
        if last_error is None:
            raise RuntimeError("HTTP request failed with no captured exception")
        raise last_error

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self.request_json("GET", path, params=params)

    def post_json(
        self,
        path: str,
        payload: Any,
        params: dict[str, Any] | None = None,
    ) -> Any:
        return self.request_json("POST", path, json=payload, params=params)

    def close(self) -> None:
        self.client.close()


class PolymarketClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self.gamma = BaseHTTPClient(config.polymarket.gamma_url)
        self.clob = BaseHTTPClient(config.polymarket.clob_url)

    def close(self) -> None:
        self.gamma.close()
        self.clob.close()

    def _discovery_status_filters(self) -> list[dict[str, str]]:
        filters: list[dict[str, str]] = []
        if self.config.polymarket.include_active:
            filters.append({"active": "true", "closed": "false"})
        if self.config.polymarket.include_closed:
            filters.append({"closed": "true"})
        if not filters:
            filters.append({})
        return filters

    @staticmethod
    def _market_text(market: dict[str, Any]) -> str:
        event = (market.get("events") or [{}])[0]
        return " ".join(
            str(chunk or "")
            for chunk in [
                market.get("question"),
                market.get("description"),
                market.get("slug"),
                market.get("resolutionSource"),
                event.get("description"),
            ]
        ).lower()

    @staticmethod
    def _classify_market_type(text: str) -> str | None:
        directional_terms = (
            "up or down",
            " up ",
            " down ",
            "higher",
            "lower",
            "rise",
            "fall",
            "increase",
            "decrease",
            "gain",
            "lose",
            "bullish",
            "bearish",
        )
        threshold_terms = (
            "above",
            "below",
            "over",
            "under",
            "at or above",
            "at or below",
            "higher than",
            "lower than",
            "greater than",
            "less than",
            "reach",
            "hit",
            "$",
            "price",
        )
        padded = f" {text} "
        if any(term in padded for term in directional_terms):
            return "binary_directional"
        if any(term in text for term in threshold_terms):
            return "binary_threshold"
        return None

    @staticmethod
    def _event_text(event: dict[str, Any], market: dict[str, Any]) -> str:
        series = (event.get("series") or [{}])[0]
        tags = event.get("tags") or []
        tag_bits = []
        for tag in tags:
            if isinstance(tag, dict):
                tag_bits.extend([str(tag.get("label") or ""), str(tag.get("slug") or "")])
            else:
                tag_bits.append(str(tag))
        return " ".join(
            bit
            for bit in [
                str(event.get("title") or ""),
                str(event.get("description") or ""),
                str(event.get("slug") or ""),
                str(event.get("ticker") or ""),
                str(event.get("resolutionSource") or ""),
                str(market.get("question") or ""),
                str(market.get("description") or ""),
                str(market.get("slug") or ""),
                str(series.get("title") or ""),
                str(series.get("slug") or ""),
                str(series.get("ticker") or ""),
                str(series.get("recurrence") or ""),
                *tag_bits,
            ]
            if bit
        ).lower()

    @staticmethod
    def _outcome_token_mapping(outcomes: list[Any], token_ids: list[Any]) -> tuple[str | None, str | None]:
        if len(outcomes) < 2 or len(token_ids) < 2:
            return None, None
        pairs = list(zip([str(item).strip().lower() for item in outcomes], [str(item) for item in token_ids]))
        positive_labels = {"yes", "up", "higher", "above", "over", "bullish", "long"}
        negative_labels = {"no", "down", "lower", "below", "under", "bearish", "short"}
        positive_token = next((token for label, token in pairs if label in positive_labels), None)
        negative_token = next((token for label, token in pairs if label in negative_labels), None)
        if positive_token and negative_token:
            return positive_token, negative_token
        return str(token_ids[0]), str(token_ids[1])

    def _event_to_registry_entries(self, event: dict[str, Any]) -> list[MarketRegistryEntry]:
        entries: list[MarketRegistryEntry] = []
        for market in event.get("markets") or []:
            entry = self._market_to_registry(market, event=event)
            if entry is not None:
                entries.append(entry)
        return entries

    def _recurring_event_slug_candidates(self) -> list[str]:
        candidates = {slug for slug in self.config.polymarket.seed_event_slugs if slug}
        lookback_hours = max(int(self.config.polymarket.recurring_slug_lookback_hours), 0)
        if lookback_hours <= 0:
            return sorted(candidates)

        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start = now - timedelta(hours=lookback_hours)
        candidates.update(self._recurring_event_slug_candidates_between(start, now))
        return sorted(candidates)

    def _recurring_event_slug_candidates_between(
        self,
        start: datetime,
        end: datetime,
    ) -> list[str]:
        start = ensure_utc(start).replace(second=0, microsecond=0)
        end = ensure_utc(end).replace(second=0, microsecond=0)
        if end < start:
            return []
        candidates: set[str] = set()
        horizon_prefix = {
            5: "btc-updown-5m",
            15: "btc-updown-15m",
        }
        for horizon in self.config.universe.horizons_minutes:
            prefix = horizon_prefix.get(horizon)
            if prefix is None:
                continue
            floored_minute = (start.minute // horizon) * horizon
            current = start.replace(minute=floored_minute, second=0, microsecond=0)
            while current <= end:
                candidates.add(f"{prefix}-{int(current.timestamp())}")
                current += timedelta(minutes=horizon)
        return sorted(candidates)

    def get_event_by_slug(self, slug: str) -> dict[str, Any] | None:
        try:
            payload = self.gamma.get_json(f"/events/slug/{slug}")
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise
        if not isinstance(payload, dict):
            return None
        return payload

    def discover_markets_between(
        self,
        start: datetime,
        end: datetime,
    ) -> list[MarketRegistryEntry]:
        seen: dict[str, MarketRegistryEntry] = {}
        for slug in self._recurring_event_slug_candidates_between(start, end):
            event = self.get_event_by_slug(slug)
            if event is None:
                continue
            for entry in self._event_to_registry_entries(event):
                seen[entry.market_id] = entry
        return sorted(seen.values(), key=lambda item: (item.horizon_minutes or 0, item.open_time or "", item.market_id))

    @staticmethod
    def _entry_overlap_window(
        entry: MarketRegistryEntry,
        start: datetime,
        end: datetime,
    ) -> tuple[datetime, datetime] | None:
        open_time = parse_datetime(entry.open_time)
        close_time = parse_datetime(entry.close_time)
        if open_time is None or close_time is None:
            return None
        query_start = max(ensure_utc(start), open_time)
        query_end = min(ensure_utc(end), close_time)
        if query_end < query_start:
            return None
        return query_start, query_end

    def discover_markets(self) -> list[MarketRegistryEntry]:
        seen: dict[str, MarketRegistryEntry] = {}
        for slug in self._recurring_event_slug_candidates():
            event = self.get_event_by_slug(slug)
            if event is None:
                continue
            for entry in self._event_to_registry_entries(event):
                seen[entry.market_id] = entry

        for status_filter in self._discovery_status_filters():
            for page in range(self.config.polymarket.discovery_pages):
                payload = self.gamma.get_json(
                    "/events",
                    params={
                        "limit": self.config.polymarket.max_results_per_search,
                        "offset": page * self.config.polymarket.max_results_per_search,
                        **status_filter,
                    },
                )
                if not payload:
                    break
                for event in payload or []:
                    for entry in self._event_to_registry_entries(event):
                        seen[entry.market_id] = entry
                if len(payload) < self.config.polymarket.max_results_per_search:
                    break
        return sorted(seen.values(), key=lambda item: (item.horizon_minutes or 0, item.market_id))

    def _market_to_registry(
        self,
        market: dict[str, Any],
        event: dict[str, Any] | None = None,
    ) -> MarketRegistryEntry | None:
        event = event or (market.get("events") or [{}])[0]
        question = str(market.get("question") or event.get("title") or "")
        description = str(market.get("description") or event.get("description") or "")
        slug = str(market.get("slug") or event.get("slug") or "")
        text = self._event_text(event, market) if event else self._market_text(market)
        if "bitcoin" not in text and "btc" not in text:
            return None
        market_type = self._classify_market_type(text)
        if market_type is None:
            return None

        outcomes = safe_json_loads(market.get("outcomes"), default=[]) or []
        token_ids = safe_json_loads(market.get("clobTokenIds"), default=[]) or []
        positive_token_id, negative_token_id = self._outcome_token_mapping(outcomes, token_ids)
        if positive_token_id is None or negative_token_id is None:
            return None

        open_time = coalesce(
            parse_datetime(market.get("eventStartTime")),
            parse_datetime(event.get("startTime")) if event else None,
            parse_datetime(market.get("startDate")),
            parse_datetime(event.get("startDate")) if event else None,
        )
        close_time = coalesce(
            parse_datetime(market.get("endDate")),
            parse_datetime(event.get("endDate")) if event else None,
        )
        series = (event.get("series") or [{}])[0] if event else {}
        series_text = " ".join(
            value
            for value in [
                str(series.get("title") or ""),
                str(series.get("slug") or ""),
                str(series.get("recurrence") or ""),
            ]
            if value
        )
        horizon_minutes = infer_horizon_minutes(f"{question} {series_text}".strip(), description, open_time, close_time)
        if horizon_minutes not in self.config.universe.horizons_minutes:
            return None
        rules_blob = "\n".join(
            chunk
            for chunk in [
                description,
                str(market.get("resolutionSource") or event.get("resolutionSource") or ""),
                str(event.get("description") or "") if event else "",
            ]
            if chunk
        )
        return MarketRegistryEntry(
            venue="polymarket",
            market_id=str(market.get("conditionId") or market.get("id")),
            question=question,
            horizon_minutes=horizon_minutes,
            market_type=market_type,
            open_time=open_time.isoformat() if open_time else None,
            close_time=close_time.isoformat() if close_time else None,
            event_id=str(event.get("id") or market.get("id") or "") if event else str(market.get("id") or ""),
            event_ticker=str(event.get("ticker") or "") if event else "",
            series_ticker=str(series.get("ticker") or "") if isinstance(series, dict) else None,
            yes_token_id=positive_token_id,
            no_token_id=negative_token_id,
            rules_primary=description or None,
            rules_secondary=str(market.get("resolutionSource") or event.get("resolutionSource") or "") or None,
            rules_hash=sha256_text(rules_blob or question),
            metadata={
                "slug": slug,
                "outcomes": outcomes,
                "positive_outcome_label": next(
                    (str(label) for label in outcomes if str(label).strip().lower() in {"yes", "up", "higher", "above", "over", "bullish", "long"}),
                    str(outcomes[0]) if outcomes else None,
                ),
                "negative_outcome_label": next(
                    (str(label) for label in outcomes if str(label).strip().lower() in {"no", "down", "lower", "below", "under", "bearish", "short"}),
                    str(outcomes[1]) if len(outcomes) > 1 else None,
                ),
                "liquidity": market.get("liquidityNum") or market.get("liquidity"),
                "volume": market.get("volumeNum") or market.get("volume"),
                "open_interest": market.get("openInterest") or event.get("openInterest"),
                "best_bid": market.get("bestBid"),
                "best_ask": market.get("bestAsk"),
                "last_trade_price": market.get("lastTradePrice"),
                "active": market.get("active"),
                "closed": market.get("closed"),
                "event_slug": event.get("slug") if event else None,
                "series_slug": series.get("slug") if isinstance(series, dict) else None,
                "series_recurrence": series.get("recurrence") if isinstance(series, dict) else None,
                "tags": [tag.get("slug") if isinstance(tag, dict) else str(tag) for tag in (event.get("tags") or [])] if event else [],
                "event_start_time": market.get("eventStartTime") or (event.get("startTime") if event else None),
            },
        )

    def get_books(self, token_ids: list[str]) -> list[dict[str, Any]]:
        if not token_ids:
            return []
        payload = [{"token_id": token_id} for token_id in token_ids]
        return self.clob.post_json("/books", payload=payload)

    def get_price_history(
        self,
        token_id: str,
        start: datetime,
        end: datetime,
        fidelity: int = 1,
    ) -> list[dict[str, Any]]:
        payload = self.clob.get_json(
            "/prices-history",
            params={
                "market": token_id,
                "startTs": int(ensure_utc(start).timestamp()),
                "endTs": int(ensure_utc(end).timestamp()),
                "fidelity": fidelity,
            },
        )
        return payload.get("history", [])

    def snapshot_entries(
        self,
        entries: list[MarketRegistryEntry],
        btc_price_ref: float | None,
    ) -> list[NormalizedObservation]:
        yes_token_ids = [entry.yes_token_id for entry in entries if entry.yes_token_id]
        books = self.get_books([token_id for token_id in yes_token_ids if token_id is not None])
        book_map = {str(book.get("asset_id")): book for book in books}
        observations: list[NormalizedObservation] = []
        for entry in entries:
            if not entry.yes_token_id:
                continue
            book = book_map.get(entry.yes_token_id)
            if not book:
                continue
            bids = book.get("bids") or []
            asks = book.get("asks") or []
            yes_bid = parse_decimal(bids[0].get("price")) if bids else None
            yes_ask = parse_decimal(asks[0].get("price")) if asks else None
            last_trade = parse_decimal(book.get("last_trade_price"))
            event_ts = parse_datetime(book.get("timestamp")) or datetime.now(timezone.utc)
            observations.append(
                NormalizedObservation(
                    ts_utc=floor_to_minute(event_ts).isoformat(),
                    venue="polymarket",
                    market_id=entry.market_id,
                    question=entry.question,
                    horizon_minutes=entry.horizon_minutes,
                    side_up_probability_mid=midpoint(yes_bid, yes_ask, last_trade),
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    spread=spread(yes_bid, yes_ask),
                    volume=parse_decimal(entry.metadata.get("volume")),
                    open_interest=parse_decimal(entry.metadata.get("open_interest")),
                    liquidity=parse_decimal(entry.metadata.get("liquidity")),
                    btc_price_ref=btc_price_ref,
                    rules_hash=entry.rules_hash,
                    event_ticker=entry.event_ticker,
                    source="polymarket_rest_book",
                    metadata={"asset_id": entry.yes_token_id, "book_hash": book.get("hash")},
                )
            )
        return observations

    def backfill_entries(
        self,
        entries: list[MarketRegistryEntry],
        start: datetime,
        end: datetime,
        btc_prices: dict[pd.Timestamp, float],
    ) -> list[NormalizedObservation]:
        observations: list[NormalizedObservation] = []
        for entry in entries:
            if not entry.yes_token_id:
                continue
            overlap = self._entry_overlap_window(entry, start, end)
            if overlap is None:
                continue
            query_start, query_end = overlap
            history = self.get_price_history(entry.yes_token_id, query_start, query_end, fidelity=1)
            for point in history:
                timestamp = floor_to_minute(
                    datetime.fromtimestamp(int(point["t"]), tz=timezone.utc)
                )
                observations.append(
                    NormalizedObservation(
                        ts_utc=timestamp.isoformat(),
                        venue="polymarket",
                        market_id=entry.market_id,
                        question=entry.question,
                        horizon_minutes=entry.horizon_minutes,
                        side_up_probability_mid=parse_decimal(point.get("p")),
                        yes_bid=None,
                        yes_ask=None,
                        spread=None,
                        volume=parse_decimal(entry.metadata.get("volume")),
                        open_interest=parse_decimal(entry.metadata.get("open_interest")),
                        liquidity=parse_decimal(entry.metadata.get("liquidity")),
                        btc_price_ref=btc_prices.get(pd.Timestamp(timestamp)),
                        rules_hash=entry.rules_hash,
                        event_ticker=entry.event_ticker,
                        source="polymarket_prices_history",
                        metadata={"asset_id": entry.yes_token_id},
                    )
                )
        return observations

    async def _stream_market_messages(
        self,
        asset_ids: list[str],
        seconds: int,
    ) -> list[dict[str, Any]]:
        if not asset_ids or seconds <= 0:
            return []
        messages: list[dict[str, Any]] = []
        async with websockets.connect(self.config.polymarket.websocket_url, ping_interval=20) as ws:
            await ws.send(json_dumps({"type": "market", "assets_ids": asset_ids}))
            start = time.monotonic()
            while time.monotonic() - start < seconds:
                timeout = max(seconds - (time.monotonic() - start), 0.1)
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                try:
                    messages.append(json.loads(raw))
                except ValueError:
                    continue
        return messages

    def stream_best_bid_ask(
        self,
        entries: list[MarketRegistryEntry],
        btc_price_ref: float | None,
        seconds: int,
    ) -> list[NormalizedObservation]:
        lookup = {entry.yes_token_id: entry for entry in entries if entry.yes_token_id}
        asset_ids = list(lookup)
        if not asset_ids:
            return []
        raw_messages = asyncio.run(self._stream_market_messages(asset_ids, seconds=seconds))
        observations: list[NormalizedObservation] = []
        for message in raw_messages:
            asset_id = str(message.get("asset_id") or "")
            entry = lookup.get(asset_id)
            if entry is None:
                continue
            event_type = message.get("event_type")
            if event_type not in {"best_bid_ask", "book"}:
                continue
            yes_bid = parse_decimal(message.get("best_bid"))
            yes_ask = parse_decimal(message.get("best_ask"))
            if event_type == "book":
                bids = message.get("bids") or []
                asks = message.get("asks") or []
                yes_bid = yes_bid if yes_bid is not None else parse_decimal(bids[0].get("price")) if bids else None
                yes_ask = yes_ask if yes_ask is not None else parse_decimal(asks[0].get("price")) if asks else None
            event_ts = parse_datetime(message.get("timestamp")) or datetime.now(timezone.utc)
            observations.append(
                NormalizedObservation(
                    ts_utc=floor_to_minute(event_ts).isoformat(),
                    venue="polymarket",
                    market_id=entry.market_id,
                    question=entry.question,
                    horizon_minutes=entry.horizon_minutes,
                    side_up_probability_mid=midpoint(yes_bid, yes_ask),
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    spread=spread(yes_bid, yes_ask),
                    volume=parse_decimal(entry.metadata.get("volume")),
                    open_interest=parse_decimal(entry.metadata.get("open_interest")),
                    liquidity=parse_decimal(entry.metadata.get("liquidity")),
                    btc_price_ref=btc_price_ref,
                    rules_hash=entry.rules_hash,
                    event_ticker=entry.event_ticker,
                    source=f"polymarket_ws_{event_type}",
                    metadata={"asset_id": asset_id},
                )
            )
        return observations


class KalshiClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self.api = BaseHTTPClient(config.kalshi.base_url)

    def close(self) -> None:
        self.api.close()

    @staticmethod
    def _entry_overlap_window(
        entry: MarketRegistryEntry,
        start: datetime,
        end: datetime,
    ) -> tuple[datetime, datetime] | None:
        open_time = parse_datetime(entry.open_time)
        close_time = parse_datetime(entry.close_time) or parse_datetime(entry.expected_expiration_time)
        if open_time is None or close_time is None:
            return None
        query_start = max(ensure_utc(start), open_time)
        query_end = min(ensure_utc(end), close_time)
        if query_end < query_start:
            return None
        return query_start, query_end

    def discover_markets(self) -> list[MarketRegistryEntry]:
        seen: dict[str, MarketRegistryEntry] = {}
        for series_ticker in self.config.kalshi.series_tickers:
            cursor: str | None = None
            while True:
                params: dict[str, Any] = {
                    "series_ticker": series_ticker,
                    "limit": self.config.kalshi.page_limit,
                    "status": "open",
                }
                if cursor:
                    params["cursor"] = cursor
                payload = self.api.get_json("/markets", params=params)
                markets = payload.get("markets", [])
                for market in markets:
                    entry = self._market_to_registry(series_ticker, market)
                    if entry is None:
                        continue
                    seen[entry.market_id] = entry
                cursor = payload.get("cursor") or None
                if not cursor:
                    break
        return sorted(seen.values(), key=lambda item: (item.horizon_minutes or 0, item.market_id))

    def _market_to_registry(
        self,
        series_ticker: str,
        market: dict[str, Any],
    ) -> MarketRegistryEntry | None:
        title = str(market.get("title") or "")
        subtitle = str(market.get("subtitle") or "")
        event_ticker = str(market.get("event_ticker") or "")
        text = " ".join([series_ticker, event_ticker, title, subtitle]).lower()
        if "bitcoin" not in text and "kxbtc" not in text:
            return None
        open_time = parse_datetime(market.get("open_time"))
        close_time = parse_datetime(market.get("close_time"))
        expected_expiration = parse_datetime(market.get("expected_expiration_time"))
        rules_primary = str(market.get("rules_primary") or "")
        rules_secondary = str(market.get("rules_secondary") or "")
        rules_blob = "\n".join(chunk for chunk in [rules_primary, rules_secondary, title, subtitle] if chunk)
        return MarketRegistryEntry(
            venue="kalshi",
            market_id=str(market.get("ticker")),
            question=f"{title} {subtitle}".strip(),
            horizon_minutes=infer_horizon_minutes(title, subtitle, open_time, coalesce(close_time, expected_expiration)),
            market_type="binary_strike",
            open_time=open_time.isoformat() if open_time else None,
            close_time=close_time.isoformat() if close_time else None,
            expected_expiration_time=expected_expiration.isoformat() if expected_expiration else None,
            event_ticker=event_ticker or None,
            series_ticker=series_ticker,
            strike=parse_decimal(coalesce(market.get("floor_strike"), market.get("cap_strike"))),
            rules_primary=rules_primary or None,
            rules_secondary=rules_secondary or None,
            rules_hash=sha256_text(rules_blob or title),
            metadata={
                "liquidity": market.get("liquidity_dollars"),
                "volume": market.get("volume_fp"),
                "open_interest": market.get("open_interest_fp"),
                "yes_bid": market.get("yes_bid_dollars"),
                "yes_ask": market.get("yes_ask_dollars"),
                "price_ranges": market.get("price_ranges"),
                "strike_type": market.get("strike_type"),
            },
        )

    def get_market(self, ticker: str) -> dict[str, Any]:
        return self.api.get_json(f"/markets/{ticker}")

    def get_historical_cutoff(self) -> dict[str, Any]:
        return self.api.get_json("/historical/cutoff")

    def get_batch_candlesticks(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        payload = self.api.get_json(
            "/markets/candlesticks",
            params={
                "market_tickers": ",".join(tickers),
                "start_ts": int(ensure_utc(start).timestamp()),
                "end_ts": int(ensure_utc(end).timestamp()),
                "period_interval": 1,
            },
        )
        return payload.get("markets", [])

    def get_market_candlesticks(
        self,
        entry: MarketRegistryEntry,
        start: datetime,
        end: datetime,
        historical: bool = False,
    ) -> list[dict[str, Any]]:
        if historical:
            path = f"/historical/markets/{entry.market_id}/candlesticks"
        else:
            path = f"/series/{entry.series_ticker}/markets/{entry.market_id}/candlesticks"
        payload = self.api.get_json(
            path,
            params={
                "start_ts": int(ensure_utc(start).timestamp()),
                "end_ts": int(ensure_utc(end).timestamp()),
                "period_interval": 1,
            },
        )
        return payload.get("candlesticks", [])

    def snapshot_entries(
        self,
        entries: list[MarketRegistryEntry],
        btc_price_ref: float | None,
    ) -> list[NormalizedObservation]:
        observations: list[NormalizedObservation] = []
        for entry in entries:
            market = self.get_market(entry.market_id)
            yes_bid = parse_decimal(market.get("yes_bid_dollars"))
            yes_ask = parse_decimal(market.get("yes_ask_dollars"))
            last_price = parse_decimal(market.get("last_price_dollars"))
            event_ts = parse_datetime(market.get("updated_time")) or datetime.now(timezone.utc)
            observations.append(
                NormalizedObservation(
                    ts_utc=floor_to_minute(event_ts).isoformat(),
                    venue="kalshi",
                    market_id=entry.market_id,
                    question=entry.question,
                    horizon_minutes=entry.horizon_minutes,
                    side_up_probability_mid=midpoint(yes_bid, yes_ask, last_price),
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    spread=spread(yes_bid, yes_ask),
                    volume=parse_decimal(market.get("volume_fp")),
                    open_interest=parse_decimal(market.get("open_interest_fp")),
                    liquidity=parse_decimal(market.get("liquidity_dollars")),
                    btc_price_ref=btc_price_ref,
                    strike=entry.strike,
                    rules_hash=entry.rules_hash,
                    event_ticker=entry.event_ticker,
                    series_ticker=entry.series_ticker,
                    source="kalshi_market_detail",
                    metadata={"strike_type": market.get("strike_type")},
                )
            )
        return observations

    def backfill_entries(
        self,
        entries: list[MarketRegistryEntry],
        start: datetime,
        end: datetime,
        btc_prices: dict[pd.Timestamp, float],
    ) -> list[NormalizedObservation]:
        observations: list[NormalizedObservation] = []
        cutoff_payload = self.get_historical_cutoff()
        cutoff = parse_datetime(cutoff_payload.get("market_settled_ts"))
        live_entries: list[MarketRegistryEntry] = []
        historical_entries: list[MarketRegistryEntry] = []
        for entry in entries:
            overlap = self._entry_overlap_window(entry, start, end)
            if overlap is None:
                continue
            close_time = parse_datetime(entry.close_time)
            if cutoff and close_time and close_time < cutoff:
                historical_entries.append(entry)
            else:
                live_entries.append(entry)

        for group in [live_entries]:
            tickers = [entry.market_id for entry in group]
            for batch in [tickers[idx : idx + 100] for idx in range(0, len(tickers), 100)]:
                if not batch:
                    continue
                try:
                    payload = self.get_batch_candlesticks(batch, start, end)
                    mapping = {item["market_ticker"]: item.get("candlesticks", []) for item in payload}
                except httpx.HTTPStatusError:
                    mapping = {
                        entry.market_id: self.get_market_candlesticks(entry, start, end, historical=False)
                        for entry in group
                        if entry.market_id in batch
                    }
                for entry in group:
                    for candle in mapping.get(entry.market_id, []):
                        timestamp = floor_to_minute(
                            datetime.fromtimestamp(int(candle["end_period_ts"]), tz=timezone.utc)
                        )
                        yes_bid = parse_decimal(candle.get("yes_bid", {}).get("close_dollars"))
                        yes_ask = parse_decimal(candle.get("yes_ask", {}).get("close_dollars"))
                        last_price = parse_decimal(candle.get("price", {}).get("close_dollars"))
                        observations.append(
                            NormalizedObservation(
                                ts_utc=timestamp.isoformat(),
                                venue="kalshi",
                                market_id=entry.market_id,
                                question=entry.question,
                                horizon_minutes=entry.horizon_minutes,
                                side_up_probability_mid=midpoint(yes_bid, yes_ask, last_price),
                                yes_bid=yes_bid,
                                yes_ask=yes_ask,
                                spread=spread(yes_bid, yes_ask),
                                volume=parse_decimal(candle.get("volume_fp")),
                                open_interest=parse_decimal(candle.get("open_interest_fp")),
                                liquidity=parse_decimal(entry.metadata.get("liquidity")),
                                btc_price_ref=btc_prices.get(pd.Timestamp(timestamp)),
                                strike=entry.strike,
                                rules_hash=entry.rules_hash,
                                event_ticker=entry.event_ticker,
                                series_ticker=entry.series_ticker,
                                source="kalshi_candlesticks_live",
                                metadata={"strike_type": entry.metadata.get("strike_type")},
                            )
                        )

        for entry in historical_entries:
            overlap = self._entry_overlap_window(entry, start, end)
            if overlap is None:
                continue
            query_start, query_end = overlap
            for candle in self.get_market_candlesticks(entry, query_start, query_end, historical=True):
                timestamp = floor_to_minute(
                    datetime.fromtimestamp(int(candle["end_period_ts"]), tz=timezone.utc)
                )
                yes_bid = parse_decimal(candle.get("yes_bid", {}).get("close_dollars"))
                yes_ask = parse_decimal(candle.get("yes_ask", {}).get("close_dollars"))
                last_price = parse_decimal(candle.get("price", {}).get("close_dollars"))
                observations.append(
                    NormalizedObservation(
                        ts_utc=timestamp.isoformat(),
                        venue="kalshi",
                        market_id=entry.market_id,
                        question=entry.question,
                        horizon_minutes=entry.horizon_minutes,
                        side_up_probability_mid=midpoint(yes_bid, yes_ask, last_price),
                        yes_bid=yes_bid,
                        yes_ask=yes_ask,
                        spread=spread(yes_bid, yes_ask),
                        volume=parse_decimal(candle.get("volume_fp")),
                        open_interest=parse_decimal(candle.get("open_interest_fp")),
                        liquidity=parse_decimal(entry.metadata.get("liquidity")),
                        btc_price_ref=btc_prices.get(pd.Timestamp(timestamp)),
                        strike=entry.strike,
                        rules_hash=entry.rules_hash,
                        event_ticker=entry.event_ticker,
                        series_ticker=entry.series_ticker,
                        source="kalshi_candlesticks_historical",
                        metadata={"strike_type": entry.metadata.get("strike_type")},
                    )
                )
        return observations


class BTCClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self.binance = BaseHTTPClient(config.btc.binance_url)
        self.coinbase = BaseHTTPClient(config.btc.coinbase_url)

    def close(self) -> None:
        self.binance.close()
        self.coinbase.close()

    def _binance_klines(
        self,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> list[list[Any]]:
        rows: list[list[Any]] = []
        current_ms = int(ensure_utc(start).timestamp() * 1000)
        end_ms = int(ensure_utc(end).timestamp() * 1000)
        while current_ms <= end_ms:
            payload = self.binance.get_json(
                "/api/v3/klines",
                params={
                    "symbol": self.config.btc.symbol,
                    "interval": "1m",
                    "limit": limit,
                    "startTime": current_ms,
                    "endTime": min(end_ms, current_ms + (limit - 1) * 60_000),
                },
            )
            if not payload:
                break
            rows.extend(payload)
            next_ms = int(payload[-1][0]) + 60_000
            if next_ms <= current_ms:
                break
            current_ms = next_ms
        return rows

    def backfill(
        self,
        start: datetime,
        end: datetime,
    ) -> list[BTCCandle]:
        candles: list[BTCCandle] = []
        try:
            payload = self._binance_klines(start, end)
            for row in payload:
                timestamp = floor_to_minute(
                    datetime.fromtimestamp(int(row[0]) / 1000.0, tz=timezone.utc)
                )
                candles.append(
                    BTCCandle(
                        ts_utc=timestamp.isoformat(),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                        source="binance",
                    )
                )
            return candles
        except Exception:
            return self.backfill_coinbase(start, end)

    def backfill_coinbase(self, start: datetime, end: datetime) -> list[BTCCandle]:
        payload = self.coinbase.get_json(
            f"/products/{self.config.btc.product_id}/candles",
            params={
                "start": ensure_utc(start).isoformat(),
                "end": ensure_utc(end).isoformat(),
                "granularity": 60,
            },
        )
        candles: list[BTCCandle] = []
        for row in payload:
            timestamp = floor_to_minute(datetime.fromtimestamp(int(row[0]), tz=timezone.utc))
            candles.append(
                BTCCandle(
                    ts_utc=timestamp.isoformat(),
                    low=float(row[1]),
                    high=float(row[2]),
                    open=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    source="coinbase",
                )
            )
        return sorted(candles, key=lambda candle: candle.ts_utc)

    def latest_candle(self) -> BTCCandle:
        payload = self.binance.get_json(
            "/api/v3/klines",
            params={"symbol": self.config.btc.symbol, "interval": "1m", "limit": 2},
        )
        row = payload[-1]
        timestamp = floor_to_minute(datetime.fromtimestamp(int(row[0]) / 1000.0, tz=timezone.utc))
        return BTCCandle(
            ts_utc=timestamp.isoformat(),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            source="binance",
        )
