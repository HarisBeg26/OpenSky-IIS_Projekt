from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from src.app.prediction import (
    PredictionError,
    batch_prediction_summary,
    load_batch_prediction,
    predict_aircraft_state,
)
from src.app.pretrained_risk import (
    build_flight_risk_text,
    classify_with_pretrained_model,
    pretrained_model_status,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
LEGACY_STATIC_DIR = PROJECT_ROOT / "src" / "app" / "static"
GX_DOCS_DIR = PROJECT_ROOT / "gx" / "uncommitted" / "data_docs" / "local_site"
REPORTS_DIR = PROJECT_ROOT / "reports"
EVIDENTLY_REPORTS_DIR = PROJECT_ROOT / "reports" / "evidently"
MODEL_REGISTRY_STATE_PATH = PROJECT_ROOT / "reports" / "model_registry" / "model_registry_state.json"
ALLOWED_MODEL_STAGES = {"Candidate", "Staging", "Production", "Archived"}


class ExperienceSettings(BaseModel):
    low_altitude_m: float = 300
    descent_rate_ms: float = -6
    unstable_vertical_rate_ms: float = 12
    high_velocity_ms: float = 170
    attention_threshold: int = 45
    critical_threshold: int = 75


class ModelStageUpdate(BaseModel):
    stage: str
    note: str | None = None

app = FastAPI(title="SkyWatch OpenSky Intelligence API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if LEGACY_STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=LEGACY_STATIC_DIR), name="legacy-static")

if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")

if GX_DOCS_DIR.exists():
    app.mount("/reports/gx-site", StaticFiles(directory=GX_DOCS_DIR, html=True), name="gx-docs")

if EVIDENTLY_REPORTS_DIR.exists():
    app.mount("/reports/evidently", StaticFiles(directory=EVIDENTLY_REPORTS_DIR), name="evidently-reports")

if REPORTS_DIR.exists():
    app.mount("/reports/files", StaticFiles(directory=REPORTS_DIR), name="report-files")


@app.get("/", include_in_schema=False)
def index() -> Response:
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(
        """
        <!doctype html>
        <title>SkyWatch API</title>
        <main style="font-family: sans-serif; max-width: 760px; margin: 4rem auto;">
          <h1>SkyWatch API is running</h1>
          <p>React frontend has not been built yet. Run <code>cd frontend && npm install && npm run dev</code>
          for local UI development or <code>npm run build</code> for production assets.</p>
          <p>Try <a href="/api/intelligence/briefing">/api/intelligence/briefing</a>.</p>
        </main>
        """,
    )


@app.get("/admin", include_in_schema=False)
def admin() -> Response:
    return index()


@app.get("/api/health")
def health() -> dict[str, Any]:
    latest_snapshot = _latest_processed_snapshot()
    return {
        "status": "ok",
        "service": "skywatch",
        "online_serving": "model_as_a_service" if _model_service_configured() else "model_as_dependency",
        "latest_processed_snapshot": str(latest_snapshot.relative_to(PROJECT_ROOT)) if latest_snapshot else None,
        "models_available": _models_available(),
        "batch_predictions": batch_prediction_summary(),
        "frontend_built": (FRONTEND_DIST / "index.html").exists(),
    }


def _settings_from_query(
    low_altitude_m: Annotated[float, Query(ge=50, le=3000)] = 300,
    descent_rate_ms: Annotated[float, Query(ge=-50, le=-0.1)] = -6,
    unstable_vertical_rate_ms: Annotated[float, Query(ge=1, le=60)] = 12,
    high_velocity_ms: Annotated[float, Query(ge=20, le=400)] = 170,
    attention_threshold: Annotated[int, Query(ge=1, le=100)] = 45,
    critical_threshold: Annotated[int, Query(ge=1, le=100)] = 75,
) -> ExperienceSettings:
    return ExperienceSettings(
        low_altitude_m=low_altitude_m,
        descent_rate_ms=descent_rate_ms,
        unstable_vertical_rate_ms=unstable_vertical_rate_ms,
        high_velocity_ms=high_velocity_ms,
        attention_threshold=attention_threshold,
        critical_threshold=max(critical_threshold, attention_threshold),
    )


@app.get("/api/flights")
def flights(
    settings: Annotated[ExperienceSettings, Depends(_settings_from_query)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 150,
) -> dict[str, Any]:
    snapshot, records = _flight_records(limit, settings)
    return {
        "snapshot": snapshot.name if snapshot else None,
        "count": len(records),
        "experience_settings": settings.model_dump(),
        "flights": records,
    }


@app.get("/api/intelligence/briefing")
def intelligence_briefing(
    settings: Annotated[ExperienceSettings, Depends(_settings_from_query)],
    limit: Annotated[int, Query(ge=10, le=1000)] = 120,
) -> dict[str, Any]:
    snapshot, records = _flight_records(limit, settings)
    attention = sorted(
        [record for record in records if record["status"] not in {"normal", "ground"}],
        key=lambda item: item["attention_score"],
        reverse=True,
    )
    training = _read_json("reports/model_training/opensky_metrics.json") or {}
    monitoring = _read_json("reports/model_monitoring/production_model_monitoring.json") or {}
    critical_count = sum(1 for item in attention if item["status"] == "critical")

    return {
        "snapshot": snapshot.name if snapshot else None,
        "summary": {
            "aircraft_count": len(records),
            "attention_count": len(attention),
            "critical_count": critical_count,
            "priority": _priority_label(len(records), len(attention), critical_count, monitoring),
            "briefing": _briefing_text(len(records), len(attention), critical_count, monitoring),
        },
        "model_readiness": _model_confidence(training),
        "monitoring": monitoring,
        "experience_settings": settings.model_dump(),
        "attention_queue": attention[:10],
        "traffic": sorted(records, key=lambda item: item["attention_score"], reverse=True)[:80],
    }


@app.get("/api/predictions/{icao24}")
def aircraft_prediction(icao24: str) -> dict[str, Any]:
    try:
        prediction = predict_aircraft_state(icao24)
    except PredictionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    latest = prediction.get("latest_state", {})
    insight = _flight_insight(latest) if latest else {}
    pretrained = classify_with_pretrained_model(build_flight_risk_text(latest, insight)) if latest else {}
    shadow = _shadow_compare_prediction(latest, prediction.get("prediction", {})) if latest else {}
    return {
        **prediction,
        "pretrained_risk_model": pretrained,
        "shadow_evaluation": shadow,
        "decision_support": {
            "status": insight.get("status"),
            "attention_score": insight.get("attention_score"),
            "reason": insight.get("reason"),
            "recommended_action": insight.get("recommended_action"),
        },
    }


@app.get("/api/admin/summary")
def admin_summary() -> dict[str, Any]:
    return {
        "validation": _read_json("reports/validation/opensky_validation.json"),
        "drift": _read_json("reports/evidently/opensky_data_drift_summary.json"),
        "training": _read_json("reports/model_training/opensky_metrics.json"),
        "monitoring": _read_json("reports/model_monitoring/production_model_monitoring.json"),
        "compression": _read_json("reports/model_compression/opensky_compression.json"),
        "explainability": _read_json("reports/model_explainability/opensky_explainability.json"),
        "deployment": _read_json("reports/deployment/model_deployment_patterns.json"),
        "models": _model_inventory(),
        "latest_snapshot": _latest_snapshot_summary(),
    }


@app.get("/api/admin/advanced")
def advanced_admin_summary() -> dict[str, Any]:
    validation = _read_json("reports/validation/opensky_validation.json")
    drift = _read_json("reports/evidently/opensky_data_drift_summary.json")
    training = _read_json("reports/model_training/opensky_metrics.json")
    monitoring = _read_json("reports/model_monitoring/production_model_monitoring.json")
    compression = _read_json("reports/model_compression/opensky_compression.json")
    explainability = _read_json("reports/model_explainability/opensky_explainability.json")
    deployment = _read_json("reports/deployment/model_deployment_patterns.json")
    return {
        "quality_gates": [
            _validation_gate(validation),
            _drift_gate(drift),
            _training_gate(training),
            _compression_gate(compression),
            _explainability_gate(explainability),
            _monitoring_gate(monitoring),
        ],
        "experiment_tracking": _experiment_tracking_summary(training),
        "model_registry": _model_registry_summary(training),
        "compression": _compression_summary(compression, training),
        "explainability": _explainability_summary(explainability),
        "deployment_patterns": _deployment_patterns_summary(deployment),
        "report_links": _admin_report_links(),
        "lifecycle_stages": sorted(ALLOWED_MODEL_STAGES),
        "shadow_testing": _shadow_testing_summary(training),
        "pretrained_model": pretrained_model_status(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/admin/model-registry/{model_key}/stage")
def update_model_stage(model_key: str, update: ModelStageUpdate) -> dict[str, Any]:
    if update.stage not in ALLOWED_MODEL_STAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid stage '{update.stage}'. Allowed stages: {sorted(ALLOWED_MODEL_STAGES)}",
        )

    model_key = model_key.strip()
    registry_state = _read_model_registry_state()
    registry_state.setdefault("models", {})
    registry_state["models"][model_key] = {
        "stage": update.stage,
        "note": update.note or "",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    MODEL_REGISTRY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_REGISTRY_STATE_PATH.write_text(json.dumps(registry_state, indent=2), encoding="utf-8")
    return {"model_key": model_key, **registry_state["models"][model_key]}


@app.get("/api/admin/model-monitoring")
def model_monitoring() -> dict[str, Any]:
    return _read_json("reports/model_monitoring/production_model_monitoring.json") or {
        "status": "missing",
        "message": "Model monitoring report has not been generated yet.",
    }


def _flight_records(
    limit: int = 150,
    settings: ExperienceSettings | None = None,
) -> tuple[Path | None, list[dict[str, Any]]]:
    settings = settings or ExperienceSettings()
    snapshot = _latest_processed_snapshot()
    if snapshot is None:
        return None, []

    df = _read_table(snapshot).tail(limit)
    records = []
    for row in df.to_dict(orient="records"):
        insight = _flight_insight(row, settings)
        records.append(
            {
                "icao24": row.get("icao24"),
                "callsign": _clean_text(row.get("callsign")) or row.get("icao24"),
                "origin_country": row.get("origin_country"),
                "longitude": _number(row.get("longitude")),
                "latitude": _number(row.get("latitude")),
                "baro_altitude": _number(row.get("baro_altitude")),
                "velocity": _number(row.get("velocity")),
                "vertical_rate": _number(row.get("vertical_rate")),
                "true_track": _number(row.get("true_track")),
                "on_ground": _bool(row.get("on_ground")),
                "status": insight["status"],
                "attention_score": insight["attention_score"],
                "reason": insight["reason"],
                "recommended_action": insight["recommended_action"],
                "forecast": insight["forecast"],
                "matched_settings": insight["matched_settings"],
            }
        )

    return snapshot, records


def _priority_label(
    aircraft_count: int,
    attention_count: int,
    critical_count: int,
    monitoring: dict[str, Any],
) -> str:
    if aircraft_count == 0:
        return "missing-data"
    if critical_count:
        return "critical"
    if attention_count:
        return "elevated"
    if monitoring.get("status") == "warn":
        return "stale-data"
    return "nominal"


def _briefing_text(
    aircraft_count: int,
    attention_count: int,
    critical_count: int,
    monitoring: dict[str, Any],
) -> str:
    if aircraft_count == 0:
        return "Ni svezega OpenSky snapshota. Sistem trenutno ne more pripraviti operativnega briefinga."
    if critical_count:
        return f"Sistem spremlja {aircraft_count} zrakoplovov; {critical_count} zahteva takojsnjo pozornost."
    if attention_count:
        return f"Sistem spremlja {aircraft_count} zrakoplovov; {attention_count} jih ima povisan operativni signal."
    if monitoring.get("status") == "warn":
        return f"Promet je stabilen, vendar monitoring opozarja na svezino podatkov pri {aircraft_count} zapisih."
    return f"Promet je stabilen. Sistem spremlja {aircraft_count} zrakoplovov brez kriticnih signalov."


def _model_confidence(training: dict[str, Any]) -> dict[str, Any]:
    models = training.get("models", {}) if isinstance(training, dict) else {}
    trajectory = models.get("trajectory_lstm", {})
    ground = models.get("on_ground_lstm", {})
    longitude_mae = float(trajectory.get("mae_longitude", 0) or 0)
    latitude_mae = float(trajectory.get("mae_latitude", 0) or 0)
    accuracy = float(ground.get("accuracy", 0) or 0)
    label = "limited"
    if accuracy >= 0.9 and max(latitude_mae, longitude_mae) <= 5:
        label = "usable"
    if accuracy >= 0.95 and max(latitude_mae, longitude_mae) <= 2:
        label = "strong"
    return {
        "label": label,
        "trajectory_mae": max(latitude_mae, longitude_mae),
        "on_ground_accuracy": accuracy,
    }


def _flight_insight(
    row: dict[str, Any],
    settings: ExperienceSettings | None = None,
) -> dict[str, Any]:
    settings = settings or ExperienceSettings()
    altitude = _number(row.get("baro_altitude")) or _number(row.get("geo_altitude")) or 0
    velocity = _number(row.get("velocity")) or 0
    vertical_rate = _number(row.get("vertical_rate")) or 0
    on_ground = _bool(row.get("on_ground"))
    forecast = _forecast_next_state(row)

    if on_ground:
        return {
            "status": "ground",
            "attention_score": 5,
            "reason": "Zrakoplov je oznacen kot na tleh.",
            "recommended_action": "Ni potrebnega ukrepa; ohrani v zgodovini prometa.",
            "forecast": forecast,
            "matched_settings": [],
        }

    score = 15
    reasons: list[str] = []
    matched_settings: list[str] = []
    if altitude < settings.low_altitude_m and velocity > 30:
        score += 35
        reasons.append("nizka visina")
        matched_settings.append("low_altitude_m")
    if vertical_rate < settings.descent_rate_ms and altitude < 1500:
        score += 30
        reasons.append("hitro spuscanje")
        matched_settings.append("descent_rate_ms")
    if abs(vertical_rate) > settings.unstable_vertical_rate_ms:
        score += 20
        reasons.append("nestabilna vertikalna hitrost")
        matched_settings.append("unstable_vertical_rate_ms")
    if velocity > settings.high_velocity_ms:
        score += 15
        reasons.append("visoka hitrost")
        matched_settings.append("high_velocity_ms")
    if forecast and forecast.get("altitude_m") is not None and forecast["altitude_m"] < 150:
        score += 20
        reasons.append("napovedana zelo nizka visina")
        matched_settings.append("forecast_low_altitude")

    status = "normal"
    if score >= settings.critical_threshold:
        status = "critical"
    elif score >= settings.attention_threshold:
        status = "watch"

    reason = ", ".join(reasons) if reasons else "parametri leta so znotraj pricakovanega obmocja"
    return {
        "status": status,
        "attention_score": min(score, 100),
        "reason": reason,
        "recommended_action": _recommended_action(status, altitude, vertical_rate),
        "forecast": forecast,
        "matched_settings": matched_settings,
    }


def _recommended_action(status: str, altitude: float, vertical_rate: float) -> str:
    if status == "critical":
        if altitude < 300:
            return "Prioritetno preveri trajektorijo in najblizje obmocje pristanka."
        if vertical_rate < -6:
            return "Spremljaj hitro spremembo visine v naslednjem intervalu."
        return "Rocno preveri zrakoplov v prioritetni vrsti."
    if status == "watch":
        return "Dodaj v opazovanje in preveri trend ob naslednjem snapshotu."
    return "Samo pasivno spremljanje."


def _forecast_next_state(row: dict[str, Any], seconds: int = 90) -> dict[str, float | None] | None:
    latitude = _number(row.get("latitude"))
    longitude = _number(row.get("longitude"))
    velocity = _number(row.get("velocity"))
    true_track = _number(row.get("true_track"))
    altitude = _number(row.get("baro_altitude")) or _number(row.get("geo_altitude"))
    vertical_rate = _number(row.get("vertical_rate")) or 0
    if latitude is None or longitude is None or velocity is None or true_track is None:
        return None

    distance_m = velocity * seconds
    track_rad = math.radians(true_track)
    lat_delta = math.cos(track_rad) * distance_m / 111_320
    lon_scale = max(math.cos(math.radians(latitude)), 0.05)
    lon_delta = math.sin(track_rad) * distance_m / (111_320 * lon_scale)
    return {
        "latitude": latitude + lat_delta,
        "longitude": longitude + lon_delta,
        "altitude_m": (altitude + vertical_rate * seconds) if altitude is not None else None,
        }


def _shadow_compare_prediction(latest: dict[str, Any], model_prediction: dict[str, Any]) -> dict[str, Any]:
    heuristic = _forecast_next_state(latest)
    model_lat = _number(model_prediction.get("next_latitude"))
    model_lon = _number(model_prediction.get("next_longitude"))
    if not heuristic or model_lat is None or model_lon is None:
        return {
            "status": "unavailable",
            "message": "Shadow comparison needs both heuristic forecast and model prediction.",
        }

    heuristic_lat = heuristic.get("latitude")
    heuristic_lon = heuristic.get("longitude")
    distance_m = _haversine_m(heuristic_lat, heuristic_lon, model_lat, model_lon)
    status = "aligned" if distance_m < 5_000 else "review"
    return {
        "status": status,
        "baseline": "kinematic_90s_forecast",
        "candidate": "trajectory_lstm",
        "distance_m": distance_m,
        "message": f"LSTM prediction is {distance_m:.0f}m from the heuristic shadow baseline.",
    }


def _haversine_m(lat1: float | None, lon1: float | None, lat2: float | None, lon2: float | None) -> float:
    if None in {lat1, lon1, lat2, lon2}:
        return float("nan")
    radius_m = 6_371_000
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    delta_phi = math.radians(float(lat2) - float(lat1))
    delta_lambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return float(radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def _validation_gate(validation: dict[str, Any] | None) -> dict[str, Any]:
    if not validation:
        return _gate("Great Expectations", "missing", "Validation report is not available yet.")
    success = validation.get("success")
    statistics = validation.get("statistics") or {}
    evaluated = statistics.get("evaluated_expectations")
    successful = statistics.get("successful_expectations")
    status = "pass" if success is True else "fail" if success is False else validation.get("status", "unknown")
    return _gate(
        "Great Expectations",
        status,
        f"{successful or 0}/{evaluated or 0} expectations passed." if evaluated else "Validation report loaded.",
        {"evaluated_expectations": evaluated, "successful_expectations": successful},
    )


def _drift_gate(drift: dict[str, Any] | None) -> dict[str, Any]:
    if not drift:
        return _gate("Evidently drift", "missing", "Data drift summary is not available yet.")
    status = drift.get("status") or drift.get("result") or "available"
    failed = drift.get("failed_tests") or drift.get("failed_test_count")
    total = drift.get("total_tests") or drift.get("tests_count")
    message = f"{failed or 0}/{total or 0} drift tests failed." if total else "Drift report loaded."
    return _gate("Evidently drift", status, message, {"failed_tests": failed, "total_tests": total})


def _training_gate(training: dict[str, Any] | None) -> dict[str, Any]:
    if not training:
        return _gate(
            "Model training",
            "missing",
            "Training has not generated metrics yet. Run: uv run python main.py train",
        )
    readiness = _model_confidence(training)
    status = "pass" if readiness["label"] in {"usable", "strong"} else "warn"
    return _gate(
        "Model training",
        status,
        f"Readiness is {readiness['label']} with trajectory MAE {readiness['trajectory_mae']:.2f}.",
        readiness,
    )


def _compression_gate(compression: dict[str, Any] | None) -> dict[str, Any]:
    if not compression:
        return _gate(
            "Model compression",
            "missing",
            "The float16 quantization report is generated by the model training command.",
        )
    summary = compression.get("summary", {}) if isinstance(compression, dict) else {}
    model_count = summary.get("model_count") or 0
    reduction = summary.get("best_storage_reduction_vs_keras")
    message = f"Compression profile available for {model_count} models."
    if reduction is not None:
        message += f" Best storage reduction is {float(reduction) * 100:.1f}%."
    return _gate(
        "Model compression",
        "pass" if model_count else "warn",
        message,
        summary,
    )


def _explainability_gate(explainability: dict[str, Any] | None) -> dict[str, Any]:
    if not explainability:
        return _gate(
            "Model explainability",
            "missing",
            "Feature importance is calculated after a successful model training run.",
        )
    feature_count = len(explainability.get("feature_importance", []) or [])
    status = explainability.get("status", "unknown")
    return _gate(
        "Model explainability",
        "pass" if status == "available" and feature_count else "warn",
        f"{feature_count} features explained with {explainability.get('method', 'unknown method')}.",
        {
            "method": explainability.get("method"),
            "sample_size": explainability.get("sample_size"),
            "feature_count": feature_count,
        },
    )


def _shadow_testing_summary(training: dict[str, Any] | None) -> dict[str, Any]:
    if not training:
        return {
            "status": "missing",
            "strategy": "trajectory_lstm_vs_kinematic_baseline",
            "message": "Training metrics are missing; live shadow comparisons still run per prediction when possible.",
        }

    models = training.get("models", {}) if isinstance(training, dict) else {}
    trajectory = models.get("trajectory_lstm", {})
    mae_longitude = float(trajectory.get("mae_longitude", 0) or 0)
    mae_latitude = float(trajectory.get("mae_latitude", 0) or 0)
    status = "watch" if max(mae_longitude, mae_latitude) > 5 else "ready"
    return {
        "status": status,
        "strategy": "trajectory_lstm_vs_kinematic_baseline",
        "baseline": "90-second kinematic projection from current speed and track",
        "candidate": "trained trajectory_lstm neural network",
        "message": "Each live prediction returns distance between the trained model and the heuristic shadow baseline.",
        "training_mae_latitude": mae_latitude,
        "training_mae_longitude": mae_longitude,
    }


@app.get("/api/batch-predictions")
def batch_predictions() -> dict[str, Any]:
    return batch_prediction_summary()


@app.get("/api/batch-predictions/{icao24}")
def aircraft_batch_prediction(icao24: str) -> dict[str, Any]:
    prediction = load_batch_prediction(icao24)
    if prediction is None:
        raise HTTPException(status_code=404, detail=f"No batch prediction found for aircraft {icao24}.")
    return prediction


def _compression_summary(
    compression: dict[str, Any] | None,
    training: dict[str, Any] | None,
) -> dict[str, Any]:
    if not compression:
        return {
            "status": "missing",
            "message": "Compression report has not been generated yet.",
            "models": [],
        }

    training_onnx = (training or {}).get("onnx", {}) if isinstance(training, dict) else {}
    return {
        "status": compression.get("status", "available"),
        "purpose": compression.get("purpose"),
        "summary": compression.get("summary", {}),
        "onnx": training_onnx,
        "models": compression.get("models", []),
    }


def _explainability_summary(explainability: dict[str, Any] | None) -> dict[str, Any]:
    if not explainability:
        return {
            "status": "missing",
            "method": "permutation_feature_importance",
            "message": "Explainability report has not been generated yet.",
            "top_trajectory_features": [],
            "top_on_ground_features": [],
        }
    baseline = explainability.get("baseline", {})
    return {
        "status": explainability.get("status"),
        "method": explainability.get("method"),
        "method_note": (
            "Global permutation feature importance, not SHAP: each feature is shuffled across "
            "test examples and the resulting degradation in performance is measured."
        ),
        "sample_size": explainability.get("sample_size"),
        "repeats": explainability.get("repeats"),
        "baseline": baseline,
        "interpretation": explainability.get("interpretation") or _legacy_explainability_interpretation(
            explainability,
            baseline,
        ),
        "top_trajectory_features": explainability.get("top_trajectory_features", []),
        "top_on_ground_features": explainability.get("top_on_ground_features", []),
    }


def _legacy_explainability_interpretation(
    explainability: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    base_distance = float(baseline.get("trajectory_mae_distance_m", 0) or 0)
    base_ground_f1 = float(baseline.get("on_ground_f1", 0) or 0)
    repeats = int(explainability.get("repeats", 0) or 0)
    coordinate_multiple = max(
        (
            float(row.get("trajectory_distance_increase_m", 0) or 0) / max(base_distance, 1)
            for row in explainability.get("feature_importance", [])
            if row.get("feature") in {"latitude", "longitude"}
        ),
        default=0,
    )
    warnings = []
    if coordinate_multiple >= 5:
        warnings.append(
            "Very strong coordinate dependence can indicate geographic memorization; validate on unseen aircraft or regions."
        )
    if base_ground_f1 < 0.5:
        warnings.append(
            "Baseline on-ground F1 is below 0.50, so this ranking is fragile."
        )
    if repeats < 3:
        warnings.append(
            "Fewer than three permutation repeats were used; values may be unstable."
        )
    return {
        "quality": "caution" if warnings else "stable",
        "headline": "This chart measures model reliance, not whether a feature is good or bad.",
        "scope": "global test-set explanation",
        "higher_means": "Shuffling the feature damaged test performance more, so the model relied on it more.",
        "near_zero_means": "The feature had little measurable influence on this test sample.",
        "negative_means": "Shuffling improved performance, which can indicate noise or sampling variation.",
        "not_causality": "This is permutation importance, not SHAP or a causal explanation.",
        "bar_scale": "Bars show relative ranking within each task, not an accuracy score.",
        "baseline_trajectory_mae_distance_m": base_distance,
        "baseline_on_ground_f1": base_ground_f1,
        "coordinate_reliance_multiple": coordinate_multiple,
        "warnings": warnings,
    }


def _deployment_patterns_summary(deployment: dict[str, Any] | None) -> dict[str, Any]:
    if not deployment:
        return {
            "status": "fallback",
            "active_pattern": "hybrid_embedded_online_and_batch_inference",
            "patterns": [
                {
                    "key": "online_model_as_dependency",
                    "name": "Embedded ONNX online inference",
                    "status": "active" if _models_available() else "missing",
                    "description": "The free Render web service executes ONNX models locally.",
                },
                {
                    "key": "batch_offline_prediction",
                    "name": "Batch/offline prediction",
                    "status": batch_prediction_summary().get("status", "missing"),
                    "description": "DVC prepares predictions that remain available if online inference is unavailable.",
                },
                {
                    "key": "online_model_as_a_service",
                    "name": "Model as a Service reference deployment",
                    "status": "local-ready" if _models_available() else "missing",
                    "description": "The separate ONNX service remains available through local Docker Compose.",
                },
            ],
        }
    return {
        "status": deployment.get("status"),
        "active_pattern": deployment.get("active_pattern"),
        "patterns": deployment.get("patterns", []),
        "serving_contract": deployment.get("serving_contract", {}),
    }


def _monitoring_gate(monitoring: dict[str, Any] | None) -> dict[str, Any]:
    if not monitoring:
        return _gate("Production monitoring", "missing", "Production monitoring report is not available yet.")
    checks = monitoring.get("checks") if isinstance(monitoring.get("checks"), list) else []
    failing = [check for check in checks if check.get("status") == "fail"]
    warning = [check for check in checks if check.get("status") == "warn"]
    status = "fail" if failing else "warn" if warning else monitoring.get("status", "pass")
    return _gate(
        "Production monitoring",
        status,
        f"{len(failing)} failing and {len(warning)} warning production checks.",
        {"checks": len(checks), "failing": len(failing), "warning": len(warning)},
    )


def _gate(name: str, status: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = str(status).lower()
    if normalized in {"ok", "success", "passed", "true"}:
        normalized = "pass"
    return {
        "name": name,
        "status": normalized,
        "message": message,
        "details": details or {},
    }


def _experiment_tracking_summary(training: dict[str, Any] | None) -> dict[str, Any]:
    mlflow = training.get("mlflow", {}) if isinstance(training, dict) else {}
    return {
        "tracking_uri": mlflow.get("tracking_uri"),
        "experiment_name": mlflow.get("experiment_name"),
        "registry_enabled": bool((mlflow.get("model_registry") or {}).get("enabled")),
        "metrics_logged": sorted(_flatten_metric_names(training.get("models", {}))) if training else [],
    }


def _flatten_metric_names(models: dict[str, Any]) -> set[str]:
    metric_names: set[str] = set()
    for model_key, model_metrics in models.items():
        if isinstance(model_metrics, dict):
            for key, value in model_metrics.items():
                if isinstance(value, (int, float)):
                    metric_names.add(f"{model_key}.{key}")
    return metric_names


def _model_registry_summary(training: dict[str, Any] | None) -> list[dict[str, Any]]:
    registry = {}
    if isinstance(training, dict):
        registry = (training.get("mlflow", {}) or {}).get("model_registry", {}) or {}

    registered_names = registry.get("registered_model_names") or {
        "trajectory_lstm": registry.get("trajectory_registered_model_name") or "OpenSkyTrajectoryLSTM",
        "on_ground_lstm": registry.get("on_ground_registered_model_name") or "OpenSkyOnGroundLSTM",
    }
    state = _read_model_registry_state().get("models", {})
    model_tasks = {
        "trajectory_lstm": "Predict next latitude and longitude from aircraft state sequences.",
        "on_ground_lstm": "Predict whether the aircraft will be on the ground in the next state.",
    }
    rows = []
    for key, task in model_tasks.items():
        model_info = registry.get(key, {}) if isinstance(registry.get(key, {}), dict) else {}
        lifecycle = state.get(key, {})
        rows.append(
            {
                "key": key,
                "registered_name": registered_names.get(key),
                "task": task,
                "stage": lifecycle.get("stage", "Candidate"),
                "note": lifecycle.get("note", ""),
                "updated_at": lifecycle.get("updated_at"),
                "model_uri": model_info.get("model_uri"),
                "run_id": model_info.get("run_id"),
                "registered_model_version": model_info.get("registered_model_version"),
            }
        )
    return rows


def _read_model_registry_state() -> dict[str, Any]:
    if not MODEL_REGISTRY_STATE_PATH.exists():
        return {"models": {}}
    try:
        return json.loads(MODEL_REGISTRY_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"models": {}}


def _admin_report_links() -> list[dict[str, Any]]:
    return [
        _report_link(
            "Great Expectations validation report",
            GX_DOCS_DIR / "index.html",
            "/reports/gx-site/index.html",
        ),
        _report_link(
            "Evidently data drift report",
            PROJECT_ROOT / "reports" / "evidently" / "opensky_data_drift_report.html",
            "/reports/evidently/opensky_data_drift_report.html",
        ),
        _report_link(
            "Validation JSON",
            PROJECT_ROOT / "reports" / "validation" / "opensky_validation.json",
            "/reports/files/validation/opensky_validation.json",
        ),
        _report_link(
            "Model monitoring JSON",
            PROJECT_ROOT / "reports" / "model_monitoring" / "production_model_monitoring.json",
            "/reports/files/model_monitoring/production_model_monitoring.json",
        ),
        _report_link(
            "Model compression JSON",
            PROJECT_ROOT / "reports" / "model_compression" / "opensky_compression.json",
            "/reports/files/model_compression/opensky_compression.json",
        ),
        _report_link(
            "Model explainability JSON",
            PROJECT_ROOT / "reports" / "model_explainability" / "opensky_explainability.json",
            "/reports/files/model_explainability/opensky_explainability.json",
        ),
        _report_link(
            "Deployment patterns JSON",
            PROJECT_ROOT / "reports" / "deployment" / "model_deployment_patterns.json",
            "/reports/files/deployment/model_deployment_patterns.json",
        ),
    ]


def _report_link(name: str, path: Path, url: str | None) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path.relative_to(PROJECT_ROOT)) if path.exists() else str(path),
        "url": url,
        "available": path.exists(),
    }


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _latest_processed_snapshot() -> Path | None:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    candidates = sorted(
        list(processed_dir.glob("states_processed_*.csv"))
        + list(processed_dir.glob("states_processed_*.parquet"))
    )
    return candidates[-1] if candidates else None


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _read_json(path: str | Path) -> dict[str, Any] | None:
    file_path = _project_path(path)
    if not file_path.exists():
        return None
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_json", "path": str(file_path)}


def _latest_snapshot_summary() -> dict[str, Any] | None:
    snapshot = _latest_processed_snapshot()
    if snapshot is None:
        return None
    df = _read_table(snapshot)
    return {
        "file": snapshot.name,
        "rows": int(len(df)),
        "aircraft": int(df["icao24"].nunique()) if "icao24" in df.columns else None,
        "countries": int(df["origin_country"].nunique()) if "origin_country" in df.columns else None,
    }


def _model_inventory() -> list[dict[str, Any]]:
    models_dir = PROJECT_ROOT / "models" / "opensky"
    if not models_dir.exists():
        return []
    return [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "type": path.suffix.lstrip("."),
        }
        for path in sorted(models_dir.glob("*"))
        if path.is_file()
    ]


def _models_available() -> bool:
    expected = [
        PROJECT_ROOT / "models" / "opensky" / "trajectory_lstm.keras",
        PROJECT_ROOT / "models" / "opensky" / "on_ground_lstm.keras",
        PROJECT_ROOT / "models" / "opensky" / "trajectory_lstm.onnx",
        PROJECT_ROOT / "models" / "opensky" / "on_ground_lstm.onnx",
        PROJECT_ROOT / "models" / "opensky" / "preprocessors.pkl",
    ]
    return all(path.exists() for path in expected)


def _model_service_configured() -> bool:
    return bool(
        os.getenv("MODEL_SERVICE_URL")
        or (os.getenv("MODEL_SERVICE_HOST") and os.getenv("MODEL_SERVICE_PORT"))
    )


def _clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _number(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}
