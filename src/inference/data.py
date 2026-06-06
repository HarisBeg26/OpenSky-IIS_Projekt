from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.model.preprocess import normalize_opensky_history, read_history

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PredictionInputError(Exception):
    """Raised when an aircraft does not have enough usable history."""


def models_dir() -> Path:
    return _configured_path("MODELS_DIR", "models/opensky")


def history_file() -> Path:
    return _configured_path("HISTORY_FILE", "data/processed/states_history.csv")


@lru_cache(maxsize=4)
def load_preprocessors(path: str | None = None) -> dict[str, Any]:
    preprocessors_path = Path(path) if path else models_dir() / "preprocessors.pkl"
    if not preprocessors_path.exists():
        raise FileNotFoundError(f"Missing model preprocessor artifact: {preprocessors_path}")
    with preprocessors_path.open("rb") as file:
        return pickle.load(file)


def load_history(path: str | Path | None = None) -> pd.DataFrame:
    source = Path(path) if path else history_file()
    if not source.exists():
        raise FileNotFoundError(f"Processed history not found: {source}")
    return normalize_opensky_history(read_history(source))


def prepare_aircraft_request(
    icao24: str,
    history: pd.DataFrame | None = None,
    preprocessors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    history = history if history is not None else load_history()
    preprocessors = preprocessors or load_preprocessors()
    feature_columns = list(preprocessors["feature_columns"])
    window_size = int(preprocessors["window_size"])
    aircraft = aircraft_history(history, icao24)
    if len(aircraft) < window_size:
        raise PredictionInputError(
            f"Aircraft {icao24} has {len(aircraft)} observations, but model requires {window_size}."
        )

    latest_window = aircraft.tail(window_size)
    features = latest_window[feature_columns].astype(object).where(
        pd.notna(latest_window[feature_columns]),
        None,
    )
    latest = latest_window.iloc[-1].replace({np.nan: None}).to_dict()
    return {
        "icao24": icao24.lower().strip(),
        "callsign": clean_text(latest.get("callsign")) or icao24.lower().strip(),
        "features": features.values.tolist(),
        "latest_state": json_safe_state(latest),
        "feature_columns": feature_columns,
        "window_size": window_size,
    }


def aircraft_history(history: pd.DataFrame, icao24: str) -> pd.DataFrame:
    if "icao24" not in history.columns:
        raise PredictionInputError("History dataset does not contain icao24.")
    normalized_id = icao24.lower().strip()
    aircraft = history[history["icao24"].astype(str).str.lower().str.strip() == normalized_id]
    aircraft = aircraft.dropna(subset=["last_contact"])
    aircraft = aircraft.sort_values("last_contact", kind="stable")
    if aircraft.empty:
        raise PredictionInputError(f"Aircraft {icao24} was not found in processed history.")
    return aircraft


def json_safe_state(state: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, pd.Timestamp):
            safe[key] = value.isoformat()
        elif isinstance(value, np.generic):
            safe[key] = value.item()
        elif value is None or pd.isna(value):
            safe[key] = None
        else:
            safe[key] = value
    return safe


def clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _configured_path(env_name: str, default: str) -> Path:
    configured = Path(os.getenv(env_name, default))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured
