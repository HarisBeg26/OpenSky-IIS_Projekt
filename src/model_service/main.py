from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.inference.onnx import model_artifacts_available, predict_with_onnx


class PredictionRequest(BaseModel):
    icao24: str
    callsign: str
    features: list[list[float | None]]
    latest_state: dict[str, Any]
    feature_columns: list[str]
    window_size: int


app = FastAPI(title="SkyWatch Private ONNX Model Service", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if model_artifacts_available() else "missing_artifacts",
        "service": "skywatch-model-service",
        "runtime": "onnxruntime",
        "models_available": model_artifacts_available(),
    }


@app.post("/v1/predict")
def predict(
    request: PredictionRequest,
    x_model_service_token: str | None = Header(default=None),
) -> dict[str, Any]:
    expected_token = os.getenv("MODEL_SERVICE_TOKEN")
    if expected_token and x_model_service_token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid model service token.")
    try:
        result = predict_with_onnx(request.model_dump())
    except (FileNotFoundError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        **result,
        "serving": {
            "pattern": "online_model_as_a_service",
            "service": "skywatch-model-service",
            "runtime": "onnxruntime",
        },
    }
