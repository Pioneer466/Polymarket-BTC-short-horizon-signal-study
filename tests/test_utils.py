from btcxpoly.utils import implied_drift, infer_horizon_minutes, parse_datetime


def test_infer_horizon_from_text() -> None:
    assert infer_horizon_minutes("Bitcoin Up or Down - 5 Minutes") == 5
    assert infer_horizon_minutes("BTC 15m direction") == 15


def test_implied_drift_returns_value() -> None:
    value = implied_drift(0.6, 0.02, 15)
    assert value is not None


def test_parse_datetime_accepts_fractional_timezone_timestamp() -> None:
    parsed = parse_datetime("2021-12-02T20:28:07.73+00:00")
    assert parsed is not None
    assert parsed.year == 2021
