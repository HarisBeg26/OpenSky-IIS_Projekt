from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.model.preprocess import normalize_opensky_history, read_history

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "opensky"
HISTORY_FILE = PROJECT_ROOT / "data" / "processed" / "states_history.csv"


class PredictionError(Exception):
    """Raised when prediction cannot be produced for a requested aircraft."""


@lru_cache(maxsize=1)
def _load_artifacts():
    trajectory_path = MODELS_DIR / "trajectory_lstm.keras"
    ground_path = MODELS_DIR / "on_ground_lstm.keras"
    preprocessors_path = MODELS_DIR / "preprocessors.pkl"
    missing = [path for path in [trajectory_path, ground_path, preprocessors_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {[str(path) for path in missing]}")

    import tensorflow as tf

    trajectory_model = tf.keras.models.load_model(trajectory_path)
    ground_model = tf.keras.models.load_model(ground_path)
    with preprocessors_path.open("rb") as file:
        preprocessors = pickle.load(file)
    return trajectory_model, ground_model, preprocessors


def predict_aircraft_state(icao24: str) -> dict[str, Any]:
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(f"Processed history not found: {HISTORY_FILE}")

    trajectory_model, ground_model, preprocessors = _load_artifacts()
    feature_columns = preprocessors["feature_columns"]
    window_size = int(preprocessors["window_size"])

    history = normalize_opensky_history(read_history(HISTORY_FILE))
    aircraft = _aircraft_history(history, icao24)
    if len(aircraft) < window_size:
        raise PredictionError(
            f"Aircraft {icao24} has {len(aircraft)} observations, but model requires {window_size}."
        )

    latest_window = aircraft.tail(window_size)
    features = latest_window[feature_columns].to_numpy(dtype=float).reshape(1, window_size, len(feature_columns))
    scaled_features = _scale_features(features, preprocessors)

    position_scaled = trajectory_model.predict(scaled_features, verbose=0)
    position = preprocessors["target_scaler"].inverse_transform(position_scaled)[0]
    on_ground_probability = float(ground_model.predict(scaled_features, verbose=0).reshape(-1)[0])
    latest = latest_window.iloc[-1].replace({np.nan: None}).to_dict()

    return {
        "icao24": icao24.lower().strip(),
        "callsign": _clean_text(latest.get("callsign")) or icao24.lower().strip(),
        "latest_state": _json_safe_state(latest),
        "prediction": {
            "next_latitude": float(position[0]),
            "next_longitude": float(position[1]),
            "next_on_ground_probability": on_ground_probability,
            "next_on_ground": on_ground_probability >= 0.5,
        },
        "model": {
            "trajectory_model": "trajectory_lstm.keras",
            "on_ground_model": "on_ground_lstm.keras",
            "window_size": window_size,
            "feature_columns": feature_columns,
        },
    }


def _aircraft_history(history: pd.DataFrame, icao24: str) -> pd.DataFrame:
    if "icao24" not in history.columns:
        raise PredictionError("History dataset does not contain icao24.")
    normalized_id = icao24.lower().strip()
    aircraft = history[history["icao24"].astype(str).str.lower().str.strip() == normalized_id]
    aircraft = aircraft.dropna(subset=["last_contact"])
    aircraft = aircraft.sort_values("last_contact", kind="stable")
    if aircraft.empty:
        raise PredictionError(f"Aircraft {icao24} was not found in processed history.")
    return aircraft


def _scale_features(features: np.ndarray, preprocessors: dict[str, Any]) -> np.ndarray:
    n_samples, window_size, n_features = features.shape
    flat = features.reshape(-1, n_features)
    imputed = preprocessors["feature_imputer"].transform(flat)
    scaled = preprocessors["feature_scaler"].transform(imputed)
    return scaled.reshape(n_samples, window_size, n_features)


def _json_safe_state(state: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, pd.Timestamp):
            safe[key] = value.isoformat()
        elif isinstance(value, np.generic):
            safe[key] = value.item()
        elif pd.isna(value):
            safe[key] = None
        else:
            safe[key] = value
    return safe


def _clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None
