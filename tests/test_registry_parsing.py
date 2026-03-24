from datetime import datetime, timezone

from btcxpoly.clients import KalshiClient, PolymarketClient
from btcxpoly.config import load_config


def test_polymarket_registry_conversion() -> None:
    client = PolymarketClient(load_config())
    market = {
        "id": "123",
        "conditionId": "cond-1",
        "question": "Bitcoin Up or Down - 5 Minutes",
        "description": "This market resolves to Yes if Bitcoin is up in 5 minutes.",
        "slug": "bitcoin-up-or-down-5-minutes",
        "outcomes": "[\"Yes\", \"No\"]",
        "clobTokenIds": "[\"yes-token\", \"no-token\"]",
        "startDate": "2026-03-23T11:00:00Z",
        "endDate": "2026-03-23T11:05:00Z",
        "events": [{"id": "evt-1", "ticker": "btc-5m", "openInterest": 321}],
    }
    entry = client._market_to_registry(market)
    assert entry is not None
    assert entry.market_id == "cond-1"
    assert entry.horizon_minutes == 5
    assert entry.yes_token_id == "yes-token"
    client.close()


def test_polymarket_closed_threshold_registry_conversion() -> None:
    client = PolymarketClient(load_config())
    market = {
        "id": "124",
        "conditionId": "cond-2",
        "question": "Will Bitcoin be above $70,000 in 15 minutes?",
        "description": "This market resolves to Yes if BTC is above $70,000 after 15 minutes.",
        "slug": "bitcoin-above-70000-15-minutes",
        "outcomes": "[\"Yes\", \"No\"]",
        "clobTokenIds": "[\"yes-token-2\", \"no-token-2\"]",
        "startDate": "2026-03-23T11:00:00Z",
        "endDate": "2026-03-23T11:15:00Z",
        "active": False,
        "closed": True,
        "events": [{"id": "evt-2", "ticker": "btc-15m", "openInterest": 123}],
    }
    entry = client._market_to_registry(market)
    assert entry is not None
    assert entry.market_id == "cond-2"
    assert entry.horizon_minutes == 15
    assert entry.market_type == "binary_threshold"
    assert entry.metadata["closed"] is True
    client.close()


def test_polymarket_event_updown_15m_conversion() -> None:
    client = PolymarketClient(load_config())
    event = {
        "id": "296743",
        "ticker": "btc-updown-15m-1774278900",
        "slug": "btc-updown-15m-1774278900",
        "title": "Bitcoin Up or Down - March 23, 11:15AM-11:30AM ET",
        "description": 'This market will resolve to "Up" if the Bitcoin price at the end of the time range specified in the title is greater than or equal to the price at the beginning of that range. Otherwise, it will resolve to "Down".',
        "resolutionSource": "https://data.chain.link/streams/btc-usd",
        "startDate": "2026-03-22T15:24:37.333636Z",
        "startTime": "2026-03-23T15:15:00Z",
        "endDate": "2026-03-23T15:30:00Z",
        "active": True,
        "closed": False,
        "series": [
            {
                "ticker": "btc-up-or-down-15m",
                "slug": "btc-up-or-down-15m",
                "title": "BTC Up or Down 15m",
                "recurrence": "15m",
            }
        ],
        "tags": [
            {"label": "Up or Down", "slug": "up-or-down"},
            {"label": "Bitcoin", "slug": "bitcoin"},
            {"label": "15M", "slug": "15M"},
        ],
        "markets": [
            {
                "id": "1683066",
                "question": "Bitcoin Up or Down - March 23, 11:15AM-11:30AM ET",
                "conditionId": "0xf1331d219ad116a46a98cbea0410d76bada4a671b43a83a73be5d23c67c4e0bf",
                "slug": "btc-updown-15m-1774278900",
                "endDate": "2026-03-23T15:30:00Z",
                "startDate": "2026-03-22T15:23:25.54094Z",
                "eventStartTime": "2026-03-23T15:15:00Z",
                "description": 'This market will resolve to "Up" if the Bitcoin price at the end of the time range specified in the title is greater than or equal to the price at the beginning of that range. Otherwise, it will resolve to "Down".',
                "outcomes": "[\"Up\", \"Down\"]",
                "clobTokenIds": "[\"up-token\", \"down-token\"]",
                "active": True,
                "closed": False,
                "liquidityNum": 30836.5151,
                "volumeNum": 22279.869645999977,
                "bestBid": 0.27,
                "bestAsk": 0.28,
                "lastTradePrice": 0.33,
            }
        ],
    }
    entries = client._event_to_registry_entries(event)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.market_id == "0xf1331d219ad116a46a98cbea0410d76bada4a671b43a83a73be5d23c67c4e0bf"
    assert entry.horizon_minutes == 15
    assert entry.market_type == "binary_directional"
    assert entry.open_time == "2026-03-23T15:15:00+00:00"
    assert entry.close_time == "2026-03-23T15:30:00+00:00"
    assert entry.yes_token_id == "up-token"
    assert entry.no_token_id == "down-token"
    client.close()


def test_polymarket_recurring_slug_candidates_between() -> None:
    client = PolymarketClient(load_config())
    start = datetime(2026, 3, 23, 15, 15, tzinfo=timezone.utc)
    end = datetime(2026, 3, 23, 15, 30, tzinfo=timezone.utc)
    candidates = client._recurring_event_slug_candidates_between(start, end)
    assert "btc-updown-15m-1774278900" in candidates
    assert "btc-updown-5m-1774278900" in candidates
    assert "btc-updown-5m-1774279200" in candidates
    client.close()


def test_kalshi_registry_conversion() -> None:
    client = KalshiClient(load_config())
    market = {
        "ticker": "KXBTCD-TEST-T70000",
        "event_ticker": "KXBTCD-TEST",
        "title": "Bitcoin price on Mar 23, 2026?",
        "subtitle": "$70,000 or above",
        "open_time": "2026-03-23T11:00:00Z",
        "close_time": "2026-03-23T11:15:00Z",
        "expected_expiration_time": "2026-03-23T11:16:00Z",
        "floor_strike": 70000,
        "rules_primary": "If the Bitcoin index is above 70000 at expiration, the market resolves to Yes.",
        "rules_secondary": "Uses BRTI.",
        "liquidity_dollars": "1500.00",
        "volume_fp": "100.0",
        "open_interest_fp": "80.0",
        "yes_bid_dollars": "0.52",
        "yes_ask_dollars": "0.54",
        "strike_type": "greater",
    }
    entry = client._market_to_registry("KXBTCD", market)
    assert entry is not None
    assert entry.market_id == "KXBTCD-TEST-T70000"
    assert entry.horizon_minutes == 15
    assert entry.strike == 70000
    client.close()
