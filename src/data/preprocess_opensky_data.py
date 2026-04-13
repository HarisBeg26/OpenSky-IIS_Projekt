from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_FORMAT = "csv"

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


def _load_preprocess_params(params_path: str = "params.yaml") -> dict:
    defaults = {
        "raw_dir": DEFAULT_RAW_DIR,
        "output_dir": DEFAULT_PROCESSED_DIR,
        "format": DEFAULT_FORMAT,
    }
    params_file = Path(params_path)
    if not params_file.exists():
        return defaults

    loaded = yaml.safe_load(params_file.read_text(encoding="utf-8")) or {}
    preprocess_params = loaded.get("preprocess", {}) if isinstance(loaded, dict) else {}
    return {
        "raw_dir": preprocess_params.get("raw_dir", defaults["raw_dir"]),
        "output_dir": preprocess_params.get("output_dir", defaults["output_dir"]),
        "format": preprocess_params.get("format", defaults["format"]),
    }


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


def _resolve_preprocess_config(raw_dir: str, output_dir: str, output_format: str) -> tuple[str, str, str]:
    params = _load_preprocess_params()
    effective_raw_dir = raw_dir if raw_dir != DEFAULT_RAW_DIR else params["raw_dir"]
    effective_output_dir = output_dir if output_dir != DEFAULT_PROCESSED_DIR else params["output_dir"]
    effective_output_format = output_format if output_format != DEFAULT_FORMAT else params["format"]
    return effective_raw_dir, effective_output_dir, effective_output_format


def _save_snapshot_and_get_history_paths(df: pd.DataFrame, out_dir: Path, output_format: str, ts: str) -> tuple[Path, Path]:
    if output_format == "parquet":
        out_path = out_dir / f"states_processed_{ts}.parquet"
        df.to_parquet(out_path, index=False)
        return out_path, out_dir / "states_history.parquet"

    out_path = out_dir / f"states_processed_{ts}.csv"
    df.to_csv(out_path, index=False)
    return out_path, out_dir / "states_history.csv"


def _update_history(df: pd.DataFrame, history_path: Path) -> pd.DataFrame:
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

    return combined


def preprocess_opensky_data(
    raw_dir: str = DEFAULT_RAW_DIR,
    output_dir: str = DEFAULT_PROCESSED_DIR,
    output_format: str = DEFAULT_FORMAT,
) -> int:
    try:
        effective_raw_dir, effective_output_dir, effective_output_format = _resolve_preprocess_config(
            raw_dir,
            output_dir,
            output_format,
        )

        raw_path = _latest_snapshot(Path(effective_raw_dir))
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        df = _flatten_states(payload)

        if df.empty:
            print("Preprocess finished, but no records were found.")
            return 1

        out_dir = Path(effective_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        out_path, history_path = _save_snapshot_and_get_history_paths(
            df,
            out_dir,
            effective_output_format,
            ts,
        )
        combined = _update_history(df, history_path)

        print(f"Loaded raw snapshot: {raw_path}")
        print(f"Saved processed dataset: {out_path}")
        print(f"Updated cumulative dataset: {history_path}")
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"Cumulative rows: {len(combined)}")
        return 0
    except Exception as exc:
        print(f"Preprocess failed: {exc}")
        return 1