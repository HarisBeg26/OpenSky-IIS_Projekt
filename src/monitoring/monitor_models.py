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


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_monitoring_params(params_path: str = "params.yaml") -> dict[str, Any]:
    defaults = {
        "report_path": DEFAULT_REPORT_PATH,
        "processed_dir": DEFAULT_PROCESSED_DIR,
        "models_dir": DEFAULT_MODELS_DIR,
        "training_metrics": DEFAULT_TRAINING_METRICS,
        "max_trajectory_mae": 5.0,
        "min_on_ground_accuracy": 0.85,
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
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Model monitoring status: {report['status']}")
    print(f"Model monitoring report: {report_path}")
    return 0


def _build_report(params: dict[str, Any]) -> dict[str, Any]:
    training = _read_json(params["training_metrics"])
    latest_snapshot = _latest_processed_snapshot(_project_path(params["processed_dir"]))
    models = _model_inventory(_project_path(params["models_dir"]))

    checks = [
        _check_models(models),
        _check_training_metrics(training, params),
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
        "latest_snapshot": _snapshot_summary(latest_snapshot),
    }


def _check_models(models: list[dict[str, Any]]) -> dict[str, Any]:
    names = {model["name"] for model in models}
    expected = {
        "trajectory_lstm.keras",
        "on_ground_lstm.keras",
        "trajectory_lstm.onnx",
        "on_ground_lstm.onnx",
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
    min_accuracy = float(params["min_on_ground_accuracy"])

    mae_values = [
        float(trajectory.get("mae_latitude", 999)),
        float(trajectory.get("mae_longitude", 999)),
    ]
    accuracy = float(on_ground.get("accuracy", 0))
    failures = []
    if max(mae_values) > max_mae:
        failures.append(f"trajectory MAE above threshold {max_mae}")
    if accuracy < min_accuracy:
        failures.append(f"on_ground accuracy below threshold {min_accuracy}")

    return {
        "name": "model_quality",
        "status": "ok" if not failures else "warn",
        "message": "Model metrics are within thresholds." if not failures else "; ".join(failures),
        "values": {
            "mae_latitude": mae_values[0],
            "mae_longitude": mae_values[1],
            "on_ground_accuracy": accuracy,
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


if __name__ == "__main__":
    raise SystemExit(monitor_models())
