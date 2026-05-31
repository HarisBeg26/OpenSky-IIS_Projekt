from __future__ import annotations

import os
from typing import Any

import requests

PRETRAINED_MODEL_ID = "typeform/distilbert-base-uncased-mnli"
PRETRAINED_MODEL_URL = f"https://huggingface.co/{PRETRAINED_MODEL_ID}"
HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{PRETRAINED_MODEL_ID}"
DEFAULT_LABELS = ["nominal flight", "watch flight", "critical landing risk", "ground operation"]


def build_flight_risk_text(state: dict[str, Any], insight: dict[str, Any]) -> str:
    return (
        f"Aircraft {state.get('icao24', 'unknown')} from {state.get('origin_country', 'unknown country')} "
        f"is at latitude {state.get('latitude')} and longitude {state.get('longitude')}. "
        f"Altitude is {state.get('baro_altitude') or state.get('geo_altitude')} meters, "
        f"velocity is {state.get('velocity')} meters per second, vertical rate is {state.get('vertical_rate')}. "
        f"The operational heuristic status is {insight.get('status')} because {insight.get('reason')}."
    )


def classify_with_pretrained_model(text: str, labels: list[str] | None = None) -> dict[str, Any]:
    labels = labels or DEFAULT_LABELS
    if os.getenv("HF_INFERENCE_ENABLED", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return {
            "status": "disabled",
            "model_id": PRETRAINED_MODEL_ID,
            "model_url": PRETRAINED_MODEL_URL,
            "task": "zero-shot-classification",
            "message": "Set HF_INFERENCE_ENABLED=true to call the pretrained HuggingFace model.",
            "candidate_labels": labels,
        }

    headers = {}
    token = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.post(
            HF_INFERENCE_URL,
            headers=headers,
            json={"inputs": text, "parameters": {"candidate_labels": labels}},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {
            "status": "unavailable",
            "model_id": PRETRAINED_MODEL_ID,
            "model_url": PRETRAINED_MODEL_URL,
            "task": "zero-shot-classification",
            "message": str(exc),
            "candidate_labels": labels,
        }

    return {
        "status": "ready",
        "model_id": PRETRAINED_MODEL_ID,
        "model_url": PRETRAINED_MODEL_URL,
        "task": "zero-shot-classification",
        "labels": payload.get("labels", []),
        "scores": payload.get("scores", []),
        "top_label": payload.get("labels", [None])[0] if payload.get("labels") else None,
        "top_score": payload.get("scores", [None])[0] if payload.get("scores") else None,
    }
