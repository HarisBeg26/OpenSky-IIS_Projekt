from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HISTORY_FILE = "data/processed/states_history.csv"
DEFAULT_MODELS_DIR = "models/opensky"
DEFAULT_METRICS_PATH = "reports/model_training/opensky_metrics.json"
DEFAULT_COMPRESSION_REPORT_PATH = "reports/model_compression/opensky_compression.json"
DEFAULT_EXPLAINABILITY_REPORT_PATH = "reports/model_explainability/opensky_explainability.json"
DEFAULT_MLFLOW_TRACKING_URI = "https://dagshub.com/HarisBeg26/OpenSky-IIS_Projekt.mlflow"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "OpenSky-IIS_Projekt_train"
DEFAULT_MLFLOW_MODE = "auto"
DEFAULT_REGISTER_MODELS = True
DEFAULT_LOCAL_MLFLOW_FALLBACK = True
DEFAULT_REQUIRE_REMOTE_MLFLOW = False
DEFAULT_TRAJECTORY_REGISTERED_MODEL_NAME = "OpenSkyTrajectoryLSTM"
DEFAULT_ON_GROUND_REGISTERED_MODEL_NAME = "OpenSkyOnGroundLSTM"
DEFAULT_AWAIT_MODEL_REGISTRATION_SECONDS = 120
DEFAULT_COMPRESSION_ENABLED = True
DEFAULT_COMPRESSION_METHOD = "float16_weight_quantization"
DEFAULT_QUANTIZATION_DTYPE = "float16"
DEFAULT_EXPLAINABILITY_ENABLED = True
DEFAULT_EXPLAINABILITY_SAMPLE_SIZE = 1000
DEFAULT_EXPLAINABILITY_REPEATS = 1
DEFAULT_EXPLAINABILITY_TOP_N = 10
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


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_train_params(params_path: str = "params.yaml") -> dict:
    defaults = {
        "history_file": DEFAULT_HISTORY_FILE,
        "models_dir": DEFAULT_MODELS_DIR,
        "metrics_path": DEFAULT_METRICS_PATH,
        "compression_report_path": DEFAULT_COMPRESSION_REPORT_PATH,
        "explainability_report_path": DEFAULT_EXPLAINABILITY_REPORT_PATH,
        "mlflow_tracking_uri": DEFAULT_MLFLOW_TRACKING_URI,
        "mlflow_experiment_name": DEFAULT_MLFLOW_EXPERIMENT_NAME,
        "mlflow_mode": DEFAULT_MLFLOW_MODE,
        "register_models": DEFAULT_REGISTER_MODELS,
        "local_mlflow_fallback": DEFAULT_LOCAL_MLFLOW_FALLBACK,
        "require_remote_mlflow": DEFAULT_REQUIRE_REMOTE_MLFLOW,
        "trajectory_registered_model_name": DEFAULT_TRAJECTORY_REGISTERED_MODEL_NAME,
        "on_ground_registered_model_name": DEFAULT_ON_GROUND_REGISTERED_MODEL_NAME,
        "await_model_registration_seconds": DEFAULT_AWAIT_MODEL_REGISTRATION_SECONDS,
        "compression_enabled": DEFAULT_COMPRESSION_ENABLED,
        "compression_method": DEFAULT_COMPRESSION_METHOD,
        "quantization_dtype": DEFAULT_QUANTIZATION_DTYPE,
        "explainability_enabled": DEFAULT_EXPLAINABILITY_ENABLED,
        "explainability_sample_size": DEFAULT_EXPLAINABILITY_SAMPLE_SIZE,
        "explainability_repeats": DEFAULT_EXPLAINABILITY_REPEATS,
        "explainability_top_n": DEFAULT_EXPLAINABILITY_TOP_N,
        "feature_columns": DEFAULT_FEATURE_COLUMNS,
        "test_size": 0.2,
        "max_sequences": 12000,
        "window_size": 2,
        "max_sequence_gap_seconds": 1800,
        "trajectory_target_mode": "delta",
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

    params_file = project_path(params_path)
    if not params_file.exists():
        return defaults

    loaded = yaml.safe_load(params_file.read_text(encoding="utf-8")) or {}
    train_params = loaded.get("train", {}) if isinstance(loaded, dict) else {}
    return {
        "history_file": train_params.get("history_file", defaults["history_file"]),
        "models_dir": train_params.get("models_dir", defaults["models_dir"]),
        "metrics_path": train_params.get("metrics_path", defaults["metrics_path"]),
        "compression_report_path": train_params.get("compression_report_path", defaults["compression_report_path"]),
        "explainability_report_path": train_params.get(
            "explainability_report_path",
            defaults["explainability_report_path"],
        ),
        "mlflow_tracking_uri": train_params.get("mlflow_tracking_uri", defaults["mlflow_tracking_uri"]),
        "mlflow_experiment_name": train_params.get("mlflow_experiment_name", defaults["mlflow_experiment_name"]),
        "mlflow_mode": train_params.get("mlflow_mode", defaults["mlflow_mode"]),
        "register_models": as_bool(train_params.get("register_models", defaults["register_models"])),
        "local_mlflow_fallback": as_bool(
            train_params.get("local_mlflow_fallback", defaults["local_mlflow_fallback"])
        ),
        "require_remote_mlflow": as_bool(
            train_params.get("require_remote_mlflow", defaults["require_remote_mlflow"])
        ),
        "trajectory_registered_model_name": train_params.get(
            "trajectory_registered_model_name",
            defaults["trajectory_registered_model_name"],
        ),
        "on_ground_registered_model_name": train_params.get(
            "on_ground_registered_model_name",
            defaults["on_ground_registered_model_name"],
        ),
        "await_model_registration_seconds": int(
            train_params.get("await_model_registration_seconds", defaults["await_model_registration_seconds"])
        ),
        "compression_enabled": as_bool(
            train_params.get("compression_enabled", defaults["compression_enabled"])
        ),
        "compression_method": train_params.get("compression_method", defaults["compression_method"]),
        "quantization_dtype": train_params.get("quantization_dtype", defaults["quantization_dtype"]),
        "explainability_enabled": as_bool(
            train_params.get("explainability_enabled", defaults["explainability_enabled"])
        ),
        "explainability_sample_size": int(
            train_params.get("explainability_sample_size", defaults["explainability_sample_size"])
        ),
        "explainability_repeats": int(
            train_params.get("explainability_repeats", defaults["explainability_repeats"])
        ),
        "explainability_top_n": int(
            train_params.get("explainability_top_n", defaults["explainability_top_n"])
        ),
        "feature_columns": train_params.get("feature_columns", defaults["feature_columns"]),
        "test_size": float(train_params.get("test_size", defaults["test_size"])),
        "max_sequences": int(train_params.get("max_sequences", defaults["max_sequences"])),
        "window_size": int(train_params.get("window_size", defaults["window_size"])),
        "max_sequence_gap_seconds": float(
            train_params.get("max_sequence_gap_seconds", defaults["max_sequence_gap_seconds"])
        ),
        "trajectory_target_mode": train_params.get("trajectory_target_mode", defaults["trajectory_target_mode"]),
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


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def configure_mlflow_tracking(params: dict) -> dict:
    params = params.copy()
    mode = os.getenv("MLFLOW_TRACKING_MODE", params.get("mlflow_mode", DEFAULT_MLFLOW_MODE))
    mode = str(mode).strip().lower()
    if params["require_remote_mlflow"]:
        mode = "remote"
    if mode not in {"auto", "local", "remote"}:
        raise ValueError("train.mlflow_mode must be one of: auto, local, remote.")

    tracking_uri = params["mlflow_tracking_uri"]
    if mode == "local":
        return _configure_local_mlflow(params, "Local MLflow mode was requested.")

    if "dagshub.com" not in tracking_uri:
        params["mlflow_tracking_mode"] = "configured"
        return params

    missing = [
        name
        for name in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
        if not os.getenv(name)
    ]
    if not missing:
        params["mlflow_tracking_mode"] = "remote"
        return params

    if mode == "remote" or not params["local_mlflow_fallback"]:
        raise EnvironmentError(
            "DagsHub MLflow tracking requires environment variables: " + ", ".join(missing)
        )

    return _configure_local_mlflow(
        params,
        "Missing DagsHub MLflow credentials: " + ", ".join(missing),
    )


def _configure_local_mlflow(params: dict, reason: str) -> dict:
    local_tracking_uri = (PROJECT_ROOT / "mlruns").resolve().as_uri()
    params["mlflow_tracking_uri"] = local_tracking_uri
    params["register_models"] = False
    params["mlflow_tracking_mode"] = "local"
    params["mlflow_fallback_reason"] = reason
    print(
        f"Using local MLflow tracking at {local_tracking_uri}. "
        "DagsHub model registration is disabled for this run."
    )
    return params


def validate_compression_params(params: dict) -> None:
    if not params["compression_enabled"]:
        return
    if params["compression_method"] != DEFAULT_COMPRESSION_METHOD:
        raise ValueError("Only float16_weight_quantization is currently implemented for NN compression.")
    if str(params["quantization_dtype"]).lower() != DEFAULT_QUANTIZATION_DTYPE:
        raise ValueError("Only float16 quantization is currently implemented.")
