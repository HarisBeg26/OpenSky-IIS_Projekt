from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.inference.data import load_preprocessors, models_dir


@lru_cache(maxsize=1)
def _load_sessions():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("ONNX Runtime is required for production inference. Run `uv sync`.") from exc

    trajectory_path = models_dir() / "trajectory_lstm.onnx"
    ground_path = models_dir() / "on_ground_lstm.onnx"
    missing = [path for path in (trajectory_path, ground_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ONNX model artifacts: {[str(path) for path in missing]}")

    providers = ["CPUExecutionProvider"]
    return (
        ort.InferenceSession(str(trajectory_path), providers=providers),
        ort.InferenceSession(str(ground_path), providers=providers),
    )


def predict_with_onnx(request: dict[str, Any]) -> dict[str, Any]:
    preprocessors = load_preprocessors()
    features = np.asarray(request["features"], dtype=np.float32)
    expected_shape = (int(request["window_size"]), len(request["feature_columns"]))
    if features.shape != expected_shape:
        raise ValueError(f"Expected feature shape {expected_shape}, received {features.shape}.")

    scaled_features = _scale_features(features.reshape(1, *expected_shape), preprocessors)
    trajectory_session, ground_session = _load_sessions()
    position_scaled = _run_session(trajectory_session, scaled_features)
    ground_output = _run_session(ground_session, scaled_features)
    position_raw = preprocessors["target_scaler"].inverse_transform(position_scaled)[0]
    latest = request["latest_state"]
    position = _position_from_model_output(position_raw, latest, preprocessors)
    probability = float(np.asarray(ground_output).reshape(-1)[0])
    threshold = float(preprocessors.get("on_ground_threshold", 0.5))

    return {
        "icao24": request["icao24"],
        "callsign": request["callsign"],
        "latest_state": latest,
        "prediction": {
            "next_latitude": float(position[0]),
            "next_longitude": float(position[1]),
            "next_on_ground_probability": probability,
            "next_on_ground_threshold": threshold,
            "next_on_ground": probability >= threshold,
        },
        "model": {
            "trajectory_model": "trajectory_lstm.onnx",
            "on_ground_model": "on_ground_lstm.onnx",
            "runtime": "onnxruntime",
            "window_size": request["window_size"],
            "feature_columns": request["feature_columns"],
        },
    }


def model_artifacts_available() -> bool:
    required = [
        models_dir() / "trajectory_lstm.onnx",
        models_dir() / "on_ground_lstm.onnx",
        models_dir() / "preprocessors.pkl",
    ]
    return all(path.exists() for path in required)


def _run_session(session, features: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    return np.asarray(session.run(None, {input_name: features.astype(np.float32)})[0])


def _scale_features(features: np.ndarray, preprocessors: dict[str, Any]) -> np.ndarray:
    n_samples, window_size, n_features = features.shape
    flat = features.reshape(-1, n_features)
    imputed = preprocessors["feature_imputer"].transform(flat)
    scaled = preprocessors["feature_scaler"].transform(imputed)
    return scaled.reshape(n_samples, window_size, n_features).astype(np.float32)


def _position_from_model_output(
    model_output: np.ndarray,
    latest_state: dict[str, Any],
    preprocessors: dict[str, Any],
) -> np.ndarray:
    target_mode = preprocessors.get("trajectory_target_mode", "absolute")
    if target_mode == "absolute":
        return model_output
    if target_mode != "delta":
        raise ValueError(f"Unsupported trajectory target mode: {target_mode}")

    latitude = latest_state.get("latitude")
    longitude = latest_state.get("longitude")
    if latitude is None or longitude is None:
        raise ValueError("Delta trajectory prediction requires latest latitude and longitude.")
    return np.array([float(latitude), float(longitude)]) + model_output


def clear_runtime_cache() -> None:
    _load_sessions.cache_clear()
    load_preprocessors.cache_clear()
