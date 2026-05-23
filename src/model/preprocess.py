from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_history(path: str | Path) -> pd.DataFrame:
    history_path = Path(path)
    if not history_path.exists():
        raise FileNotFoundError(f"Training data not found: {history_path}")

    if history_path.suffix.lower() == ".parquet":
        return pd.read_parquet(history_path)
    return pd.read_csv(history_path)


def normalize_opensky_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = [
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
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "on_ground" in df.columns:
        df["on_ground"] = df["on_ground"].map(_to_bool).astype("float")

    if "spi" in df.columns:
        df["spi"] = df["spi"].map(_to_bool).astype("float")

    if "event_time_utc" in df.columns:
        df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True, errors="coerce")

    return df


def build_sequence_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    max_sequences: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    required = ["icao24", "last_contact", "latitude", "longitude", "on_ground"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required sequence columns: {missing}")

    candidate_columns = list(dict.fromkeys(["icao24", "event_time_utc", "last_contact", *feature_columns, *required]))
    available_columns = [column for column in candidate_columns if column in df.columns]
    dataset = df[available_columns].copy()
    dataset = dataset.dropna(subset=["icao24", "last_contact", "latitude", "longitude", "on_ground"])
    dataset = dataset.sort_values(["icao24", "last_contact"], kind="stable")

    sequences: list[np.ndarray] = []
    position_targets: list[np.ndarray] = []
    ground_targets: list[float] = []
    target_times: list[float] = []

    for _, group in dataset.groupby("icao24", sort=False):
        group = group.sort_values("last_contact", kind="stable").reset_index(drop=True)
        if len(group) <= window_size:
            continue

        feature_values = group[feature_columns].to_numpy(dtype=float)
        position_values = group[["latitude", "longitude"]].to_numpy(dtype=float)
        ground_values = group["on_ground"].to_numpy(dtype=float)
        time_values = group["last_contact"].to_numpy(dtype=float)

        for index in range(len(group) - window_size):
            target_index = index + window_size
            sequences.append(feature_values[index:target_index])
            position_targets.append(position_values[target_index])
            ground_targets.append(ground_values[target_index])
            target_times.append(time_values[target_index])

    if not sequences:
        counts = dataset["icao24"].value_counts()
        max_observations = int(counts.max()) if not counts.empty else 0
        raise ValueError(
            "No valid aircraft sequences were created for model training. "
            f"The largest aircraft track has {max_observations} observations, "
            f"but window_size={window_size} requires at least {window_size + 1} observations per aircraft."
        )

    X = np.asarray(sequences, dtype=float)
    y_position = np.asarray(position_targets, dtype=float)
    y_ground = np.asarray(ground_targets, dtype=float)
    times = np.asarray(target_times, dtype=float)

    order = np.argsort(times, kind="stable")
    X = X[order]
    y_position = y_position[order]
    y_ground = y_ground[order]
    times = times[order]

    if max_sequences and len(X) > max_sequences:
        X = X[-max_sequences:]
        y_position = y_position[-max_sequences:]
        y_ground = y_ground[-max_sequences:]
        times = times[-max_sequences:]

    return X, y_position, y_ground, times
def _to_bool(value: object) -> bool | float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)
