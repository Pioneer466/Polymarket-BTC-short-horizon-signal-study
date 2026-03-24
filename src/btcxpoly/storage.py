from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from btcxpoly.config import AppConfig
from btcxpoly.models import MarketRegistryEntry
from btcxpoly.utils import ensure_directory, safe_json_loads, slugify, utc_now


class LocalStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.raw_dir = ensure_directory(Path(config.paths.raw_dir))
        self.normalized_dir = ensure_directory(Path(config.paths.normalized_dir))
        self.registry_dir = ensure_directory(Path(config.paths.registry_dir))
        self.outputs_dir = ensure_directory(Path(config.paths.outputs_dir))

    def registry_path(self) -> Path:
        return self.registry_dir / "markets.parquet"

    def write_raw_json(self, dataset: str, payload: Any) -> Path:
        timestamp = utc_now()
        date_key = timestamp.strftime("%Y-%m-%d")
        target_dir = ensure_directory(self.raw_dir / slugify(dataset) / f"date={date_key}")
        path = target_dir / f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)
        return path

    def write_dataframe(self, dataset: str, frame: pd.DataFrame, ts_column: str = "ts_utc") -> list[Path]:
        if frame.empty:
            return []
        working = frame.copy()
        working[ts_column] = pd.to_datetime(working[ts_column], utc=True)
        paths: list[Path] = []
        for date_key, part in working.groupby(working[ts_column].dt.strftime("%Y-%m-%d")):
            target_dir = ensure_directory(self.normalized_dir / slugify(dataset) / f"date={date_key}")
            path = target_dir / f"batch_{utc_now().strftime('%Y%m%dT%H%M%S%fZ')}.parquet"
            part.to_parquet(path, index=False)
            paths.append(path)
        return paths

    def clear_dataset(self, dataset: str) -> None:
        base = self.normalized_dir / slugify(dataset)
        if not base.exists():
            return
        for path in sorted(base.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()

    def replace_dataframe(self, dataset: str, frame: pd.DataFrame, ts_column: str = "ts_utc") -> list[Path]:
        self.clear_dataset(dataset)
        return self.write_dataframe(dataset, frame, ts_column=ts_column)

    def load_dataset(self, dataset: str) -> pd.DataFrame:
        base = self.normalized_dir / slugify(dataset)
        files = sorted(base.rglob("*.parquet"))
        if not files:
            return pd.DataFrame()
        con = duckdb.connect(database=":memory:")
        try:
            frame = con.read_parquet([str(path) for path in files]).df()
        finally:
            con.close()
        if "ts_utc" in frame.columns:
            frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True)
        return frame

    def save_registry(self, entries: list[MarketRegistryEntry]) -> Path:
        records = [entry.to_record() for entry in entries]
        frame = pd.DataFrame(records)
        path = self.registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return path

    def load_registry(self) -> list[MarketRegistryEntry]:
        path = self.registry_path()
        if not path.exists():
            return []
        frame = pd.read_parquet(path)
        entries: list[MarketRegistryEntry] = []
        for record in frame.to_dict(orient="records"):
            metadata = safe_json_loads(record.pop("metadata_json", "{}"), default={}) or {}
            entries.append(MarketRegistryEntry(metadata=metadata, **record))
        return entries

    def write_output_table(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.outputs_dir / f"{slugify(name)}.csv"
        frame.to_csv(path, index=False)
        return path

    def write_output_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.outputs_dir / f"{slugify(name)}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)
        return path
