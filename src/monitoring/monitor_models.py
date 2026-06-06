from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = "reports/model_monitoring/production_model_monitoring.json"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_MODELS_DIR = "models/opensky"
DEFAULT_TRAINING_METRICS = "reports/model_training/opensky_metrics.json"
DEFAULT_COMPRESSION_REPORT = "reports/model_compression/opensky_compression.json"
DEFAULT_EXPLAINABILITY_REPORT = "reports/model_explainability/opensky_explainability.json"
DEFAULT_DEPLOYMENT_REPORT = "reports/deployment/model_deployment_patterns.json"
DEFAULT_BATCH_PREDICTIONS = "data/predictions/latest_predictions.json"


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_monitoring_params(params_path: str = "params.yaml") -> dict[str, Any]:
    defaults = {
        "report_path": DEFAULT_REPORT_PATH,
        "processed_dir": DEFAULT_PROCESSED_DIR,
        "models_dir": DEFAULT_MODELS_DIR,
        "training_metrics": DEFAULT_TRAINING_METRICS,
        "compression_report": DEFAULT_COMPRESSION_REPORT,
        "explainability_report": DEFAULT_EXPLAINABILITY_REPORT,
        "deployment_report_path": DEFAULT_DEPLOYMENT_REPORT,
        "batch_predictions": DEFAULT_BATCH_PREDICTIONS,
        "max_trajectory_mae": 5.0,
        "max_trajectory_mean_distance_km": 50.0,
        "min_trajectory_skill_score": 0.0,
        "min_on_ground_accuracy": 0.85,
        "min_on_ground_f1": 0.10,
        "max_data_age_hours": 72,
    }
    params_file = _project_path(params_path)
    if not params_file.exists():
        return defaults
    loaded = yaml.safe_load(params_file.read_text(encoding="utf-8")) or {}
    monitoring = loaded.get("monitoring", {}) if isinstance(loaded, dict) else {}
    return {key: monitoring.get(key, value) for key, value in defaults.items()}


def monitor_models(params_path: str = "params.yaml") -> int:
    params = _load_monitoring_params(params_path)
    report = _build_report(params)
    report_path = _project_path(params["report_path"])
    _write_json(report_path, report)

    deployment_report = _build_deployment_report(params, report)
    deployment_report_path = _project_path(params["deployment_report_path"])
    _write_json(deployment_report_path, deployment_report)

    print(f"Model monitoring status: {report['status']}")
    print(f"Model monitoring report: {report_path}")
    print(f"Deployment patterns report: {deployment_report_path}")
    return 0


def _build_report(params: dict[str, Any]) -> dict[str, Any]:
    training = _read_json(params["training_metrics"])
    compression = _read_json(params["compression_report"])
    explainability = _read_json(params["explainability_report"])
    batch_predictions = _read_json(params["batch_predictions"])
    latest_snapshot = _latest_processed_snapshot(_project_path(params["processed_dir"]))
    models = _model_inventory(_project_path(params["models_dir"]))

    checks = [
        _check_models(models),
        _check_training_metrics(training, params),
        _check_compression_report(compression),
        _check_explainability_report(explainability),
        _check_batch_predictions(batch_predictions, params),
        _check_latest_snapshot(latest_snapshot, params),
    ]

    status = "ok"
    if any(check["status"] == "fail" for check in checks):
        status = "fail"
    elif any(check["status"] == "warn" for check in checks):
        status = "warn"

    return {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "models": models,
        "training_metrics": training,
        "compression": compression,
        "explainability": explainability,
        "batch_predictions": _batch_prediction_summary(batch_predictions),
        "latest_snapshot": _snapshot_summary(latest_snapshot),
    }


def _check_models(models: list[dict[str, Any]]) -> dict[str, Any]:
    names = {model["name"] for model in models}
    expected = {
        "trajectory_lstm.keras",
        "on_ground_lstm.keras",
        "trajectory_lstm.onnx",
        "on_ground_lstm.onnx",
        "trajectory_lstm.float16_weights.npz",
        "on_ground_lstm.float16_weights.npz",
        "preprocessors.pkl",
    }
    missing = sorted(expected - names)
    return {
        "name": "model_artifacts",
        "status": "ok" if not missing else "fail",
        "message": "All expected model artifacts are present." if not missing else f"Missing artifacts: {missing}",
    }


def _check_training_metrics(training: dict[str, Any] | None, params: dict[str, Any]) -> dict[str, Any]:
    if not training:
        return {"name": "training_metrics", "status": "fail", "message": "Training metrics are missing."}

    models = training.get("models", {})
    trajectory = models.get("trajectory_lstm", {})
    on_ground = models.get("on_ground_lstm", {})
    max_mae = float(params["max_trajectory_mae"])
    max_distance_km = float(params["max_trajectory_mean_distance_km"])
    min_skill_score = float(params["min_trajectory_skill_score"])
    min_accuracy = float(params["min_on_ground_accuracy"])
    min_f1 = float(params["min_on_ground_f1"])

    mae_values = [
        float(trajectory.get("mae_latitude", 999)),
        float(trajectory.get("mae_longitude", 999)),
    ]
    mean_distance_m = trajectory.get("mae_distance_m")
    mean_distance_km = float(mean_distance_m) / 1000 if mean_distance_m is not None else None
    skill_score = trajectory.get("skill_score_vs_baseline")
    skill_score = float(skill_score) if skill_score is not None else None
    accuracy = float(on_ground.get("accuracy", 0))
    f1 = float(on_ground.get("f1", 0))
    failures = []
    if mean_distance_km is not None:
        if mean_distance_km > max_distance_km:
            failures.append(f"trajectory mean distance error above {max_distance_km} km")
    elif max(mae_values) > max_mae:
        failures.append(f"trajectory MAE above legacy threshold {max_mae}")
    if skill_score is not None and skill_score < min_skill_score:
        failures.append(f"trajectory skill score below {min_skill_score}")
    if accuracy < min_accuracy:
        failures.append(f"on_ground accuracy below threshold {min_accuracy}")
    if f1 < min_f1:
        failures.append(f"on_ground F1 below threshold {min_f1}")

    return {
        "name": "model_quality",
        "status": "ok" if not failures else "warn",
        "message": "Model metrics are within thresholds." if not failures else "; ".join(failures),
        "values": {
            "mae_latitude": mae_values[0],
            "mae_longitude": mae_values[1],
            "mean_distance_km": mean_distance_km,
            "skill_score_vs_baseline": skill_score,
            "on_ground_accuracy": accuracy,
            "on_ground_f1": f1,
        },
    }


def _check_compression_report(compression: dict[str, Any] | None) -> dict[str, Any]:
    if not compression:
        return {
            "name": "model_compression",
            "status": "warn",
            "message": "Model compression report is missing.",
        }

    summary = compression.get("summary", {}) if isinstance(compression, dict) else {}
    model_count = int(summary.get("model_count", 0) or 0)
    best_reduction = summary.get("best_storage_reduction_vs_keras")
    return {
        "name": "model_compression",
        "status": "ok" if model_count else "warn",
        "message": (
            f"Compression profile available for {model_count} models."
            if model_count
            else "Compression profile does not contain model artifacts."
        ),
        "values": {
            "model_count": model_count,
            "best_storage_reduction_vs_keras": best_reduction,
            "runtime_candidate": summary.get("runtime_candidate"),
            "storage_candidate": summary.get("storage_candidate"),
        },
    }


def _check_explainability_report(explainability: dict[str, Any] | None) -> dict[str, Any]:
    if not explainability:
        return {
            "name": "model_explainability",
            "status": "warn",
            "message": "Model explainability report is missing.",
        }

    feature_count = len(explainability.get("feature_importance", []) or [])
    status = explainability.get("status", "unknown")
    return {
        "name": "model_explainability",
        "status": "ok" if status == "available" and feature_count else "warn",
        "message": (
            f"Permutation importance available for {feature_count} features."
            if feature_count
            else f"Explainability report status is {status}."
        ),
        "values": {
            "method": explainability.get("method"),
            "sample_size": explainability.get("sample_size"),
            "feature_count": feature_count,
        },
    }


def _check_batch_predictions(
    batch_predictions: dict[str, Any] | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    if not batch_predictions:
        return {
            "name": "batch_predictions",
            "status": "warn",
            "message": "Batch prediction artifact is missing.",
        }

    count = int(batch_predictions.get("count", 0) or 0)
    age_hours = _timestamp_age_hours(batch_predictions.get("generated_at_utc"))
    max_age = float(params["max_data_age_hours"])
    status = "ok"
    message = f"{count} batch predictions are available."
    if count == 0:
        status = "warn"
        message = "Batch prediction artifact contains no predictions."
    elif age_hours is None:
        status = "warn"
        message = "Batch predictions are available, but their generation time is invalid."
    elif age_hours > max_age:
        status = "warn"
        message = f"Batch predictions are {age_hours:.2f} hours old."

    return {
        "name": "batch_predictions",
        "status": status,
        "message": message,
        "values": {
            "count": count,
            "age_hours": age_hours,
            "max_age_hours": max_age,
        },
    }


def _check_latest_snapshot(snapshot: Path | None, params: dict[str, Any]) -> dict[str, Any]:
    if snapshot is None:
        return {"name": "production_data", "status": "fail", "message": "No processed snapshot found."}
    age_hours = (datetime.now(timezone.utc).timestamp() - snapshot.stat().st_mtime) / 3600
    max_age = float(params["max_data_age_hours"])
    return {
        "name": "production_data",
        "status": "ok" if age_hours <= max_age else "warn",
        "message": f"Latest processed snapshot age is {age_hours:.2f} hours.",
        "values": {"age_hours": age_hours, "max_age_hours": max_age},
    }


def _read_json(path: str | Path) -> dict[str, Any] | None:
    file_path = _project_path(path)
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_deployment_report(params: dict[str, Any], monitoring_report: dict[str, Any]) -> dict[str, Any]:
    models = {model["name"]: model for model in monitoring_report.get("models", [])}
    training = monitoring_report.get("training_metrics") or {}
    registry = (training.get("mlflow", {}) or {}).get("model_registry", {}) if isinstance(training, dict) else {}
    compression = monitoring_report.get("compression") or {}
    batch_predictions = monitoring_report.get("batch_predictions") or {}
    onnx_ready = {
        "trajectory_lstm.onnx",
        "on_ground_lstm.onnx",
        "preprocessors.pkl",
    }.issubset(models)
    batch_ready = int(batch_predictions.get("count", 0) or 0) > 0

    return {
        "status": "available",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_pattern": "hybrid_embedded_online_and_batch_inference",
        "patterns": [
            {
                "key": "online_model_as_dependency",
                "name": "Embedded ONNX online inference",
                "status": "active" if onnx_ready else "missing",
                "description": "The free Render web service loads ONNX artifacts locally and serves online predictions without a second paid service.",
                "evidence": ["src/app/prediction.py", "src/inference/onnx.py", "render.yaml"],
            },
            {
                "key": "batch_offline_prediction",
                "name": "Batch/offline prediction",
                "status": "active" if batch_ready else "pending",
                "description": "The DVC pipeline generates predictions for the latest aircraft snapshot and exposes them as a resilient fallback.",
                "evidence": ["src/inference/batch.py", "data/predictions/latest_predictions.json"],
            },
            {
                "key": "online_model_as_a_service",
                "name": "Model as a Service reference deployment",
                "status": "local-ready" if onnx_ready else "missing",
                "description": "A separate ONNX model service remains available for local Docker Compose demonstrations and future paid deployment.",
                "evidence": ["src/model_service/main.py", "docker/model-service.Dockerfile", "docker-compose.yml"],
            },
            {
                "key": "registry_governed_promotion",
                "name": "Registry-governed promotion",
                "status": "active" if registry.get("enabled") else "local-governance",
                "description": "MLflow model registry and local lifecycle state control Candidate, Staging, Production and Archived stages.",
                "evidence": ["reports/model_registry/model_registry_state.json", "MLflow registered model names"],
            },
            {
                "key": "compressed_portable_artifacts",
                "name": "Quantized model artifacts",
                "status": "ready" if compression.get("status") == "available" else "pending",
                "description": "Float16 post-training quantized weight archives reduce NN artifact size; ONNX remains the portable runtime candidate.",
                "evidence": ["models/opensky/*.float16_weights.npz", "reports/model_compression/opensky_compression.json"],
            },
        ],
        "serving_contract": {
            "prediction_endpoint": "/api/predictions/{icao24}",
            "batch_prediction_endpoint": "/api/batch-predictions/{icao24}",
            "optional_model_service_endpoint": "/v1/predict",
            "admin_endpoint": "/api/admin/advanced",
            "health_endpoint": "/api/health",
            "models_dir": params["models_dir"],
            "production_target": "render_free_web_service",
        },
        "testing_strategy": {
            "key": "shadow_testing",
            "description": "Live LSTM predictions are compared with a kinematic baseline; this is a testing strategy, not a deployment pattern.",
        },
    }


def _latest_processed_snapshot(processed_dir: Path) -> Path | None:
    candidates = sorted(
        list(processed_dir.glob("states_processed_*.csv"))
        + list(processed_dir.glob("states_processed_*.parquet"))
    )
    return candidates[-1] if candidates else None


def _model_inventory(models_dir: Path) -> list[dict[str, Any]]:
    if not models_dir.exists():
        return []
    return [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "updated_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        }
        for path in sorted(models_dir.glob("*"))
        if path.is_file()
    ]


def _snapshot_summary(snapshot: Path | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    if snapshot.suffix.lower() == ".parquet":
        df = pd.read_parquet(snapshot)
    else:
        df = pd.read_csv(snapshot)
    return {
        "file": snapshot.name,
        "rows": int(len(df)),
        "aircraft": int(df["icao24"].nunique()) if "icao24" in df.columns else None,
        "countries": int(df["origin_country"].nunique()) if "origin_country" in df.columns else None,
    }


def _batch_prediction_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "status": payload.get("status"),
        "pattern": payload.get("pattern"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "source_snapshot": payload.get("source_snapshot"),
        "count": int(payload.get("count", 0) or 0),
        "skipped_count": int(payload.get("skipped_count", 0) or 0),
    }


def _timestamp_age_hours(value: Any) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 3600
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(monitor_models())
