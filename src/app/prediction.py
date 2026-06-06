from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from src.inference.data import PredictionInputError, prepare_aircraft_request
from src.inference.onnx import predict_with_onnx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BATCH_PREDICTIONS_FILE = PROJECT_ROOT / "data" / "predictions" / "latest_predictions.json"


class PredictionError(Exception):
    """Raised when prediction cannot be produced for a requested aircraft."""


def predict_aircraft_state(icao24: str) -> dict[str, Any]:
    try:
        request = prepare_aircraft_request(icao24)
    except PredictionInputError as exc:
        raise PredictionError(str(exc)) from exc

    model_service_url = _model_service_url()
    if model_service_url:
        try:
            return _remote_prediction(model_service_url, request)
        except requests.RequestException as exc:
            batch = load_batch_prediction(icao24)
            if batch:
                return {
                    **batch,
                    "serving": {
                        "pattern": "batch_offline_fallback",
                        "service": "skywatch-api",
                        "fallback_reason": str(exc),
                    },
                }
            if not _env_bool("ALLOW_LOCAL_MODEL_FALLBACK", default=True):
                raise FileNotFoundError(f"Private model service is unavailable: {exc}") from exc

    result = predict_with_onnx(request)
    return {
        **result,
        "serving": {
            "pattern": "online_model_as_dependency",
            "service": "skywatch-api",
            "runtime": "onnxruntime",
        },
    }


def load_batch_prediction(icao24: str) -> dict[str, Any] | None:
    if not BATCH_PREDICTIONS_FILE.exists():
        return None
    payload = json.loads(BATCH_PREDICTIONS_FILE.read_text(encoding="utf-8"))
    return (payload.get("predictions") or {}).get(icao24.lower().strip())


def batch_prediction_summary() -> dict[str, Any]:
    if not BATCH_PREDICTIONS_FILE.exists():
        return {"status": "missing", "count": 0, "file": str(BATCH_PREDICTIONS_FILE)}
    payload = json.loads(BATCH_PREDICTIONS_FILE.read_text(encoding="utf-8"))
    return {
        "status": "available",
        "count": payload.get("count", 0),
        "generated_at_utc": payload.get("generated_at_utc"),
        "source_snapshot": payload.get("source_snapshot"),
        "file": str(BATCH_PREDICTIONS_FILE.relative_to(PROJECT_ROOT)),
    }


def _remote_prediction(url: str, request: dict[str, Any]) -> dict[str, Any]:
    headers = {}
    token = os.getenv("MODEL_SERVICE_TOKEN")
    if token:
        headers["X-Model-Service-Token"] = token
    response = requests.post(
        f"{url.rstrip('/')}/v1/predict",
        json=request,
        headers=headers,
        timeout=float(os.getenv("MODEL_SERVICE_TIMEOUT_SECONDS", "20")),
    )
    response.raise_for_status()
    return response.json()


def _model_service_url() -> str | None:
    explicit = os.getenv("MODEL_SERVICE_URL")
    if explicit:
        return explicit
    host = os.getenv("MODEL_SERVICE_HOST")
    port = os.getenv("MODEL_SERVICE_PORT")
    if host and port:
        return f"http://{host}:{port}"
    return None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
