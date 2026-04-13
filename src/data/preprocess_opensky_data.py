from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

STATE_COLUMNS = [
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "longitude",
    "latitude",
    "baro_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "sensors",
    "geo_altitude",
    "squawk",
    "spi",
    "position_source",
]


def _latest_snapshot(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("states_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No snapshots found in {raw_dir}")
    return candidates[-1]


def _flatten_states(payload: dict) -> pd.DataFrame:
    snapshot_time = payload.get("time")
    states = payload.get("states") or []

    rows: list[dict] = []
    for state in states:
        padded = (state + [None] * len(STATE_COLUMNS))[: len(STATE_COLUMNS)]
        row = dict(zip(STATE_COLUMNS, padded))
        row["callsign"] = row["callsign"].strip() if isinstance(row["callsign"], str) else None
        row["snapshot_time"] = snapshot_time
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = [
        "time_position",
        "last_contact",
        "longitude",
        "latitude",
        "baro_altitude",
        "velocity",
        "true_track",
        "vertical_rate",
        "geo_altitude",
        "position_source",
        "snapshot_time",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["captured_at_utc"] = datetime.now(timezone.utc).isoformat()
    df["event_time_utc"] = pd.to_datetime(df["last_contact"], unit="s", utc=True, errors="coerce")

    return df


def preprocess_opensky_data(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    output_format: str = "csv",
) -> int:
    try:
        raw_path = _latest_snapshot(Path(raw_dir))
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        df = _flatten_states(payload)

        if df.empty:
            print("Preprocess finished, but no records were found.")
            return 1

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        if output_format == "parquet":
            out_path = out_dir / f"states_processed_{ts}.parquet"
            df.to_parquet(out_path, index=False)
            history_path = out_dir / "states_history.parquet"
        else:
            out_path = out_dir / f"states_processed_{ts}.csv"
            df.to_csv(out_path, index=False)
            history_path = out_dir / "states_history.csv"

        # Keep a cumulative history dataset that is updated on each preprocess run.
        if history_path.exists():
            if history_path.suffix.lower() == ".parquet":
                existing = pd.read_parquet(history_path)
            else:
                existing = pd.read_csv(history_path)
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df.copy()

        dedupe_keys = ["icao24", "last_contact", "time_position"]
        available_keys = [col for col in dedupe_keys if col in combined.columns]
        if available_keys:
            combined = combined.drop_duplicates(subset=available_keys, keep="last")
        else:
            combined = combined.drop_duplicates(keep="last")

        if "last_contact" in combined.columns:
            combined = combined.sort_values(by="last_contact", kind="stable")

        if history_path.suffix.lower() == ".parquet":
            combined.to_parquet(history_path, index=False)
        else:
            combined.to_csv(history_path, index=False)

        print(f"Loaded raw snapshot: {raw_path}")
        print(f"Saved processed dataset: {out_path}")
        print(f"Updated cumulative dataset: {history_path}")
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"Cumulative rows: {len(combined)}")
        return 0
    except Exception as exc:
        print(f"Preprocess failed: {exc}")
        return 1