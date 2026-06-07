from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_PRETRAINED_MODEL_ID = "facebook/bart-large-mnli"
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
    model_id = os.getenv("HF_MODEL_ID", DEFAULT_PRETRAINED_MODEL_ID)
    model_url = f"https://huggingface.co/{model_id}"
    inference_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    if not _inference_enabled():
        return {
            "status": "disabled",
            "model_id": model_id,
            "model_url": model_url,
            "task": "zero-shot-classification",
            "message": "External model was explicitly disabled with HF_INFERENCE_ENABLED=false.",
            "candidate_labels": labels,
        }

    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HF_API_TOKEN")
        or os.getenv("HUGGINGFACE_API_TOKEN")
    )
    if not token:
        return {
            "status": "configuration_required",
            "model_id": model_id,
            "model_url": model_url,
            "task": "zero-shot-classification",
            "message": "HF_INFERENCE_ENABLED is true, but HF_TOKEN is not configured.",
            "candidate_labels": labels,
        }

    try:
        response = requests.post(
            inference_url,
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": text, "parameters": {"candidate_labels": labels}},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {
            "status": "unavailable",
            "model_id": model_id,
            "model_url": model_url,
            "task": "zero-shot-classification",
            "message": f"HuggingFace inference is currently unavailable: {exc}",
            "candidate_labels": labels,
        }

    predictions = _normalize_predictions(payload)
    return {
        "status": "ready",
        "model_id": model_id,
        "model_url": model_url,
        "task": "zero-shot-classification",
        "labels": [item["label"] for item in predictions],
        "scores": [item["score"] for item in predictions],
        "top_label": predictions[0]["label"] if predictions else None,
        "top_score": predictions[0]["score"] if predictions else None,
    }


def pretrained_model_status() -> dict[str, Any]:
    model_id = os.getenv("HF_MODEL_ID", DEFAULT_PRETRAINED_MODEL_ID)
    enabled = _inference_enabled()
    token_configured = bool(
        os.getenv("HF_TOKEN")
        or os.getenv("HF_API_TOKEN")
        or os.getenv("HUGGINGFACE_API_TOKEN")
    )
    return {
        "model_id": model_id,
        "model_url": f"https://huggingface.co/{model_id}",
        "source": "HuggingFace Inference Providers",
        "task": "zero-shot-classification",
        "enabled": enabled,
        "token_configured": token_configured,
        "status": "ready" if enabled and token_configured else "disabled" if not enabled else "configuration_required",
    }


def _inference_enabled() -> bool:
    return os.getenv("HF_INFERENCE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_predictions(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("labels"), list):
        predictions = [
            {"label": label, "score": float(score)}
            for label, score in zip(payload.get("labels", []), payload.get("scores", []))
        ]
    elif isinstance(payload, list):
        predictions = [
            {"label": str(item["label"]), "score": float(item["score"])}
            for item in payload
            if isinstance(item, dict) and "label" in item and "score" in item
        ]
    else:
        predictions = []
    return sorted(predictions, key=lambda item: item["score"], reverse=True)
