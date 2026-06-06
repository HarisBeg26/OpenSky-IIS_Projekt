from __future__ import annotations

import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for LSTM training. Run `uv sync` after the pyproject.toml update."
        ) from exc
    return tf


def load_mlflow():
    try:
        import mlflow
        import mlflow.keras  # noqa: F401
    except ImportError as exc:
        raise ImportError("MLflow is required for experiment tracking. Run `uv sync`.") from exc
    return mlflow


def load_tf2onnx():
    try:
        import tf2onnx
    except ImportError as exc:
        raise ImportError("tf2onnx is required for ONNX model export. Run `uv sync`.") from exc
    return tf2onnx


def set_reproducibility(random_state: int, tf_module) -> None:
    os.environ["PYTHONHASHSEED"] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    tf_module.random.set_seed(random_state)


def temporal_split_arrays(
    X: np.ndarray,
    y_position: np.ndarray,
    y_ground: np.ndarray,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_idx = int(len(X) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("Not enough sequences for the requested train/test split.")

    return (
        X[:split_idx],
        X[split_idx:],
        y_position[:split_idx],
        y_position[split_idx:],
        y_ground[:split_idx],
        y_ground[split_idx:],
    )


def scale_sequences(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    n_train, window_size, n_features = X_train.shape
    n_test = X_test.shape[0]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_flat = X_train.reshape(-1, n_features)
    test_flat = X_test.reshape(-1, n_features)

    train_scaled = scaler.fit_transform(imputer.fit_transform(train_flat))
    test_scaled = scaler.transform(imputer.transform(test_flat))

    return (
        train_scaled.reshape(n_train, window_size, n_features),
        test_scaled.reshape(n_test, window_size, n_features),
        imputer,
        scaler,
    )


def build_trajectory_lstm(input_shape: tuple[int, int], params: dict, tf_module):
    layers = tf_module.keras.layers
    model = tf_module.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(params["lstm_units"], return_sequences=True),
            layers.Dropout(params["dropout"]),
            layers.LSTM(max(16, params["lstm_units"] // 2)),
            layers.Dropout(params["dropout"]),
            layers.Dense(params["dense_units"], activation="relu"),
            layers.Dense(2),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_on_ground_lstm(input_shape: tuple[int, int], params: dict, tf_module):
    layers = tf_module.keras.layers
    model = tf_module.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(params["lstm_units"]),
            layers.Dropout(params["dropout"]),
            layers.Dense(params["dense_units"], activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def clean_models_dir(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for path in models_dir.glob("*"):
        if path.is_file():
            path.unlink()


def write_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(value, file)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_keras_model_to_onnx(
    model,
    output_path: Path,
    input_shape: tuple[int, int],
    tf_module,
    tf2onnx_module,
    opset: int,
    output_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Some tf2onnx conversion paths still expect this Keras 2-style attribute.
    if not hasattr(model, "output_names"):
        model.output_names = [output_name]

    input_signature = (
        tf_module.TensorSpec((None, input_shape[0], input_shape[1]), tf_module.float32, name="aircraft_sequence"),
    )
    tf2onnx_module.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=str(output_path),
        large_model=False,
    )
