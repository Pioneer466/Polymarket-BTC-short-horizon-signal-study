from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

from dateutil.parser import isoparse

ISO_8601_Z_RE = re.compile(r"Z$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if ISO_8601_Z_RE.search(text):
        text = ISO_8601_Z_RE.sub("+00:00", text)
    try:
        return ensure_utc(datetime.fromisoformat(text))
    except ValueError:
        return ensure_utc(isoparse(text))


def parse_decimal(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def safe_json_loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def floor_to_minute(value: datetime) -> datetime:
    value = ensure_utc(value)
    return value.replace(second=0, microsecond=0)


def horizon_from_text(*chunks: str) -> int | None:
    text = " ".join(chunk for chunk in chunks if chunk).lower()
    patterns = (
        r"\b(5|15)\s*minutes?\b",
        r"\b(5|15)\s*mins?\b",
        r"\b(5|15)m\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def horizon_from_times(start: datetime | None, end: datetime | None) -> int | None:
    if not start or not end:
        return None
    minutes = int(round((end - start).total_seconds() / 60.0))
    if minutes <= 0:
        return None
    return minutes


def infer_horizon_minutes(
    text: str,
    description: str = "",
    start: datetime | None = None,
    end: datetime | None = None,
) -> int | None:
    return coalesce(
        horizon_from_text(text, description),
        horizon_from_times(start, end),
    )


def midpoint(bid: float | None, ask: float | None, fallback: float | None = None) -> float | None:
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    return fallback


def spread(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    return max(ask - bid, 0.0)


def normal_ppf(probability: float | None) -> float | None:
    if probability is None:
        return None
    clipped = min(max(probability, 1e-6), 1 - 1e-6)
    return NormalDist().inv_cdf(clipped)


def realized_volatility(returns: list[float]) -> float | None:
    clean = [value for value in returns if value is not None and not math.isnan(value)]
    if len(clean) < 2:
        return None
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)
    return math.sqrt(variance)


def implied_drift(probability: float | None, sigma: float | None, horizon_minutes: int) -> float | None:
    if probability is None or sigma is None or sigma <= 0 or horizon_minutes <= 0:
        return None
    z_score = normal_ppf(probability)
    if z_score is None:
        return None
    dt = horizon_minutes / (60.0 * 24.0 * 365.0)
    if dt <= 0:
        return None
    return sigma * z_score / math.sqrt(dt)


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("size must be positive")
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def datetime_range(start: datetime, end: datetime, step: timedelta) -> list[datetime]:
    values: list[datetime] = []
    current = ensure_utc(start)
    end = ensure_utc(end)
    while current <= end:
        values.append(current)
        current += step
    return values
