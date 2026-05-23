from __future__ import annotations

import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    from .preprocess import build_sequence_dataset, normalize_opensky_history, read_history
except ImportError:
    from preprocess import build_sequence_dataset, normalize_opensky_history, read_history

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HISTORY_FILE = "data/processed/states_history.csv"
DEFAULT_MODELS_DIR = "models/opensky"
DEFAULT_METRICS_PATH = "reports/model_training/opensky_metrics.json"
DEFAULT_MLFLOW_TRACKING_URI = "https://dagshub.com/HarisBeg26/OpenSky-IIS_Projekt.mlflow"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "OpenSky-IIS_Projekt_train"
DEFAULT_FEATURE_COLUMNS = [
    "longitude",
    "latitude",
    "baro_altitude",
    "velocity",
    "true_track",
    "vertical_rate",
    "geo_altitude",
    "on_ground",
    "spi",
    "position_source",
]


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_train_params(params_path: str = "params.yaml") -> dict:
    defaults = {
        "history_file": DEFAULT_HISTORY_FILE,
        "models_dir": DEFAULT_MODELS_DIR,
        "metrics_path": DEFAULT_METRICS_PATH,
        "mlflow_tracking_uri": DEFAULT_MLFLOW_TRACKING_URI,
        "mlflow_experiment_name": DEFAULT_MLFLOW_EXPERIMENT_NAME,
        "feature_columns": DEFAULT_FEATURE_COLUMNS,
        "test_size": 0.2,
        "max_sequences": 12000,
        "window_size": 2,
        "random_state": 42,
        "lstm_units": 64,
        "dense_units": 32,
        "dropout": 0.2,
        "epochs": 20,
        "batch_size": 64,
        "validation_split": 0.2,
        "patience": 4,
        "onnx_opset": 13,
    }

    params_file = _project_path(params_path)
    if not params_file.exists():
        return defaults

    loaded = yaml.safe_load(params_file.read_text(encoding="utf-8")) or {}
    train_params = loaded.get("train", {}) if isinstance(loaded, dict) else {}
    return {
        "history_file": train_params.get("history_file", defaults["history_file"]),
        "models_dir": train_params.get("models_dir", defaults["models_dir"]),
        "metrics_path": train_params.get("metrics_path", defaults["metrics_path"]),
        "mlflow_tracking_uri": train_params.get("mlflow_tracking_uri", defaults["mlflow_tracking_uri"]),
        "mlflow_experiment_name": train_params.get("mlflow_experiment_name", defaults["mlflow_experiment_name"]),
        "feature_columns": train_params.get("feature_columns", defaults["feature_columns"]),
        "test_size": float(train_params.get("test_size", defaults["test_size"])),
        "max_sequences": int(train_params.get("max_sequences", defaults["max_sequences"])),
        "window_size": int(train_params.get("window_size", defaults["window_size"])),
        "random_state": int(train_params.get("random_state", defaults["random_state"])),
        "lstm_units": int(train_params.get("lstm_units", defaults["lstm_units"])),
        "dense_units": int(train_params.get("dense_units", defaults["dense_units"])),
        "dropout": float(train_params.get("dropout", defaults["dropout"])),
        "epochs": int(train_params.get("epochs", defaults["epochs"])),
        "batch_size": int(train_params.get("batch_size", defaults["batch_size"])),
        "validation_split": float(train_params.get("validation_split", defaults["validation_split"])),
        "patience": int(train_params.get("patience", defaults["patience"])),
        "onnx_opset": int(train_params.get("onnx_opset", defaults["onnx_opset"])),
    }


def _load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for LSTM training. Run `uv sync` after the pyproject.toml update."
        ) from exc
    return tf


def _load_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("MLflow is required for experiment tracking. Run `uv sync`.") from exc
    return mlflow


def _load_tf2onnx():
    try:
        import tf2onnx
    except ImportError as exc:
        raise ImportError("tf2onnx is required for ONNX model export. Run `uv sync`.") from exc
    return tf2onnx


def _validate_mlflow_credentials(tracking_uri: str) -> None:
    if "dagshub.com" not in tracking_uri:
        return
    missing = [
        name
        for name in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
        if not os.getenv(name)
    ]
    if missing:
        raise EnvironmentError(
            "DagsHub MLflow tracking requires environment variables: "
            + ", ".join(missing)
        )


def _set_reproducibility(random_state: int, tf_module) -> None:
    os.environ["PYTHONHASHSEED"] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    tf_module.random.set_seed(random_state)


def _temporal_split_arrays(
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


def _scale_sequences(
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


def _build_trajectory_lstm(input_shape: tuple[int, int], params: dict, tf_module):
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


def _build_on_ground_lstm(input_shape: tuple[int, int], params: dict, tf_module):
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


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _clean_models_dir(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for path in models_dir.glob("*"):
        if path.is_file():
            path.unlink()


def _write_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(value, file)


def _export_keras_model_to_onnx(
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


def train_opensky_models(params_path: str = "params.yaml") -> int:
    try:
        params = _load_train_params(params_path)
        tf = _load_tensorflow()
        mlflow = _load_mlflow()
        tf2onnx = _load_tf2onnx()
        _set_reproducibility(params["random_state"], tf)
        _validate_mlflow_credentials(params["mlflow_tracking_uri"])
        mlflow.set_tracking_uri(params["mlflow_tracking_uri"])
        mlflow.set_experiment(params["mlflow_experiment_name"])

        history = read_history(_project_path(params["history_file"]))
        history = normalize_opensky_history(history)

        feature_columns = [column for column in params["feature_columns"] if column in history.columns]
        if not feature_columns:
            raise ValueError("No configured training feature columns are present in the history dataset.")

        X, y_position, y_ground, _ = build_sequence_dataset(
            history,
            feature_columns=feature_columns,
            window_size=params["window_size"],
            max_sequences=params["max_sequences"],
        )
        if len(X) < 100:
            raise ValueError(f"Not enough sequences for LSTM training: {len(X)}")

        X_train, X_test, y_position_train, y_position_test, y_ground_train, y_ground_test = _temporal_split_arrays(
            X,
            y_position,
            y_ground,
            params["test_size"],
        )

        X_train, X_test, feature_imputer, feature_scaler = _scale_sequences(X_train, X_test)
        target_scaler = StandardScaler()
        y_position_train_scaled = target_scaler.fit_transform(y_position_train)

        input_shape = (X_train.shape[1], X_train.shape[2])
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=params["patience"],
            restore_best_weights=True,
        )

        with mlflow.start_run(run_name="train_opensky_lstm"):
            _log_mlflow_params(mlflow, params, feature_columns, len(X_train), len(X_test))

            trajectory_model = _build_trajectory_lstm(input_shape, params, tf)
            ground_model = _build_on_ground_lstm(input_shape, params, tf)

            trajectory_history = trajectory_model.fit(
                X_train,
                y_position_train_scaled,
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                validation_split=params["validation_split"],
                callbacks=[early_stopping],
                verbose=1,
            )
            ground_history = ground_model.fit(
                X_train,
                y_ground_train,
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                validation_split=params["validation_split"],
                callbacks=[early_stopping],
                verbose=1,
            )

            position_pred_scaled = trajectory_model.predict(X_test, verbose=0)
            position_pred = target_scaler.inverse_transform(position_pred_scaled)
            ground_probability = ground_model.predict(X_test, verbose=0).reshape(-1)
            ground_pred = (ground_probability >= 0.5).astype(int)

            position_mae = mean_absolute_error(y_position_test, position_pred, multioutput="raw_values")
            position_rmse = [
                _rmse(y_position_test[:, index], position_pred[:, index])
                for index in range(position_pred.shape[1])
            ]

            metrics = {
                "training_sequences": int(len(X_train)),
                "test_sequences": int(len(X_test)),
                "window_size": params["window_size"],
                "feature_columns": feature_columns,
                "mlflow": {
                    "tracking_uri": params["mlflow_tracking_uri"],
                    "experiment_name": params["mlflow_experiment_name"],
                },
                "models": {
                    "trajectory_lstm": {
                        "task": "Predict next latitude and longitude from a sequence of aircraft states.",
                        "type": "TensorFlow Keras LSTM",
                        "target_columns": ["target_next_latitude", "target_next_longitude"],
                        "epochs_trained": len(trajectory_history.history.get("loss", [])),
                        "mae_latitude": float(position_mae[0]),
                        "mae_longitude": float(position_mae[1]),
                        "rmse_latitude": float(position_rmse[0]),
                        "rmse_longitude": float(position_rmse[1]),
                        "final_validation_loss": float(trajectory_history.history.get("val_loss", [np.nan])[-1]),
                    },
                    "on_ground_lstm": {
                        "task": "Predict whether the aircraft will be on the ground in the next state.",
                        "type": "TensorFlow Keras LSTM",
                        "target_column": "target_next_on_ground",
                        "epochs_trained": len(ground_history.history.get("loss", [])),
                        "accuracy": float(accuracy_score(y_ground_test.astype(int), ground_pred)),
                        "f1": float(f1_score(y_ground_test.astype(int), ground_pred, zero_division=0)),
                        "final_validation_loss": float(ground_history.history.get("val_loss", [np.nan])[-1]),
                    },
                },
                "params": {
                    "test_size": params["test_size"],
                    "max_sequences": params["max_sequences"],
                    "window_size": params["window_size"],
                    "random_state": params["random_state"],
                    "lstm_units": params["lstm_units"],
                    "dense_units": params["dense_units"],
                    "dropout": params["dropout"],
                    "epochs": params["epochs"],
                    "batch_size": params["batch_size"],
                    "validation_split": params["validation_split"],
                    "patience": params["patience"],
                    "onnx_opset": params["onnx_opset"],
                },
            }

            models_dir = _project_path(params["models_dir"])
            _clean_models_dir(models_dir)
            trajectory_path = models_dir / "trajectory_lstm.keras"
            ground_path = models_dir / "on_ground_lstm.keras"
            trajectory_onnx_path = models_dir / "trajectory_lstm.onnx"
            ground_onnx_path = models_dir / "on_ground_lstm.onnx"
            preprocessors_path = models_dir / "preprocessors.pkl"
            metrics["onnx"] = {
                "trajectory_lstm": str(trajectory_onnx_path.relative_to(PROJECT_ROOT)),
                "on_ground_lstm": str(ground_onnx_path.relative_to(PROJECT_ROOT)),
                "opset": params["onnx_opset"],
            }
            trajectory_model.save(trajectory_path)
            ground_model.save(ground_path)
            _export_keras_model_to_onnx(
                trajectory_model,
                trajectory_onnx_path,
                input_shape,
                tf,
                tf2onnx,
                params["onnx_opset"],
                "trajectory_output",
            )
            _export_keras_model_to_onnx(
                ground_model,
                ground_onnx_path,
                input_shape,
                tf,
                tf2onnx,
                params["onnx_opset"],
                "on_ground_output",
            )
            _write_pickle(
                preprocessors_path,
                {
                    "feature_imputer": feature_imputer,
                    "feature_scaler": feature_scaler,
                    "target_scaler": target_scaler,
                    "feature_columns": feature_columns,
                    "window_size": params["window_size"],
                },
            )

            metrics_path = _project_path(params["metrics_path"])
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            _log_mlflow_metrics_and_artifacts(
                mlflow,
                metrics,
                metrics_path,
                [trajectory_path, ground_path, trajectory_onnx_path, ground_onnx_path, preprocessors_path],
            )

        print(f"Training sequences: {metrics['training_sequences']}, test sequences: {metrics['test_sequences']}")
        print(f"Window size: {params['window_size']}, features: {feature_columns}")
        print(f"Trajectory MAE latitude: {metrics['models']['trajectory_lstm']['mae_latitude']:.6f}")
        print(f"Trajectory MAE longitude: {metrics['models']['trajectory_lstm']['mae_longitude']:.6f}")
        print(f"On-ground accuracy: {metrics['models']['on_ground_lstm']['accuracy']:.6f}")
        print(f"Saved ONNX models with opset: {params['onnx_opset']}")
        print(f"Saved models to: {models_dir}")
        print(f"Saved metrics to: {metrics_path}")
        return 0
    except Exception as exc:
        print(f"Model training failed: {exc}")
        return 1


def _log_mlflow_params(mlflow, params: dict, feature_columns: list[str], training_sequences: int, test_sequences: int) -> None:
    mlflow.log_params(
        {
            "test_size": params["test_size"],
            "max_sequences": params["max_sequences"],
            "window_size": params["window_size"],
            "random_state": params["random_state"],
            "lstm_units": params["lstm_units"],
            "dense_units": params["dense_units"],
            "dropout": params["dropout"],
            "epochs": params["epochs"],
            "batch_size": params["batch_size"],
            "validation_split": params["validation_split"],
            "patience": params["patience"],
            "onnx_opset": params["onnx_opset"],
            "feature_columns": ",".join(feature_columns),
            "training_sequences": training_sequences,
            "test_sequences": test_sequences,
        }
    )


def _log_mlflow_metrics_and_artifacts(mlflow, metrics: dict, metrics_path: Path, artifact_paths: list[Path]) -> None:
    trajectory = metrics["models"]["trajectory_lstm"]
    on_ground = metrics["models"]["on_ground_lstm"]
    mlflow.log_metrics(
        {
            "trajectory_mae_latitude": trajectory["mae_latitude"],
            "trajectory_mae_longitude": trajectory["mae_longitude"],
            "trajectory_rmse_latitude": trajectory["rmse_latitude"],
            "trajectory_rmse_longitude": trajectory["rmse_longitude"],
            "trajectory_final_validation_loss": trajectory["final_validation_loss"],
            "on_ground_accuracy": on_ground["accuracy"],
            "on_ground_f1": on_ground["f1"],
            "on_ground_final_validation_loss": on_ground["final_validation_loss"],
        }
    )
    mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
    for path in artifact_paths:
        mlflow.log_artifact(str(path), artifact_path="models")


if __name__ == "__main__":
    raise SystemExit(train_opensky_models())
