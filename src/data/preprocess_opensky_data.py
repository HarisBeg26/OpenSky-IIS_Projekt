from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_FORMAT = "csv"
DEFAULT_HISTORY_FILENAME = "states_history.csv"

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
        "history_filename": DEFAULT_HISTORY_FILENAME,
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
        "history_filename": preprocess_params.get("history_filename", defaults["history_filename"]),
    }


def _snapshot_files(raw_dir: Path) -> list[Path]:
    candidates = sorted(raw_dir.glob("states_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No snapshots found in {raw_dir}")
    return candidates


def _snapshot_id(raw_path: Path) -> str:
    match = re.fullmatch(r"states_(.+)\.json", raw_path.name)
    if not match:
        raise ValueError(f"Unexpected snapshot file name: {raw_path.name}")
    return match.group(1)


def _flatten_states(payload: dict, raw_path: Path) -> pd.DataFrame:
    snapshot_time = payload.get("time")
    states = payload.get("states") or []
    source_snapshot = raw_path.name

    rows: list[dict] = []
    for state in states:
        padded = (state + [None] * len(STATE_COLUMNS))[: len(STATE_COLUMNS)]
        row = dict(zip(STATE_COLUMNS, padded))
        row["callsign"] = row["callsign"].strip() if isinstance(row["callsign"], str) else None
        row["snapshot_time"] = snapshot_time
        row["snapshot_time_utc"] = pd.to_datetime(snapshot_time, unit="s", utc=True, errors="coerce")
        row["source_snapshot"] = source_snapshot
        rows.append(row)

    base_columns = STATE_COLUMNS + ["snapshot_time", "snapshot_time_utc", "source_snapshot", "event_time_utc"]
    df = pd.DataFrame(rows, columns=base_columns)
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

    df["event_time_utc"] = pd.to_datetime(df["last_contact"], unit="s", utc=True, errors="coerce")

    return df


def _resolve_preprocess_config(raw_dir: str, output_dir: str, output_format: str) -> tuple[str, str, str, str]:
    params = _load_preprocess_params()
    effective_raw_dir = raw_dir if raw_dir != DEFAULT_RAW_DIR else params["raw_dir"]
    effective_output_dir = output_dir if output_dir != DEFAULT_PROCESSED_DIR else params["output_dir"]
    effective_output_format = output_format if output_format != DEFAULT_FORMAT else params["format"]
    effective_history_filename = params["history_filename"]
    return effective_raw_dir, effective_output_dir, effective_output_format, effective_history_filename


def _prepare_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("states_processed_*.csv", "states_processed_*.parquet", "states_history.csv", "states_history.parquet"):
        for path in out_dir.glob(pattern):
            path.unlink()


def _save_processed_snapshot(df: pd.DataFrame, out_dir: Path, output_format: str, snapshot_name: str) -> Path:
    if output_format == "parquet":
        out_path = out_dir / f"states_processed_{snapshot_name}.parquet"
        df.to_parquet(out_path, index=False)
        return out_path

    out_path = out_dir / f"states_processed_{snapshot_name}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _resolve_history_path(out_dir: Path, history_filename: str, output_format: str) -> Path:
    history_path = out_dir / history_filename
    expected_suffix = ".parquet" if output_format == "parquet" else ".csv"
    if history_path.suffix.lower() != expected_suffix:
        history_path = history_path.with_suffix(expected_suffix)
    return history_path


def _build_history(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    dedupe_keys = ["icao24", "last_contact", "time_position"]
    available_keys = [col for col in dedupe_keys if col in combined.columns]
    if available_keys:
        combined = combined.drop_duplicates(subset=available_keys, keep="last")
    else:
        combined = combined.drop_duplicates(keep="last")

    if "last_contact" in combined.columns:
        combined = combined.sort_values(by="last_contact", kind="stable")

    return combined


def _save_history(df: pd.DataFrame, history_path: Path) -> None:
    if history_path.suffix.lower() == ".parquet":
        df.to_parquet(history_path, index=False)
    else:
        df.to_csv(history_path, index=False)


def preprocess_opensky_data(
    raw_dir: str = DEFAULT_RAW_DIR,
    output_dir: str = DEFAULT_PROCESSED_DIR,
    output_format: str = DEFAULT_FORMAT,
) -> int:
    try:
        (
            effective_raw_dir,
            effective_output_dir,
            effective_output_format,
            effective_history_filename,
        ) = _resolve_preprocess_config(
            raw_dir,
            output_dir,
            output_format,
        )

        raw_paths = _snapshot_files(Path(effective_raw_dir))
        out_dir = Path(effective_output_dir)
        _prepare_output_dir(out_dir)

        processed_files: list[Path] = []
        frames: list[pd.DataFrame] = []

        for raw_path in raw_paths:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            df = _flatten_states(payload, raw_path)
            processed_files.append(
                _save_processed_snapshot(
                    df,
                    out_dir,
                    effective_output_format,
                    _snapshot_id(raw_path),
                )
            )
            frames.append(df)

        combined = _build_history(frames)
        history_path = _resolve_history_path(out_dir, effective_history_filename, effective_output_format)
        _save_history(combined, history_path)

        print(f"Processed raw snapshots: {len(raw_paths)}")
        print(f"Saved processed datasets: {len(processed_files)}")
        print(f"Updated cumulative dataset: {history_path}")
        print(f"Cumulative rows: {len(combined)}")
        return 0
    except Exception as exc:
        print(f"Preprocess failed: {exc}")
        return 1
