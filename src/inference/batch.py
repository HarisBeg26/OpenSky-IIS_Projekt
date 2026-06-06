from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.inference.data import (
    PROJECT_ROOT,
    PredictionInputError,
    load_history,
    load_preprocessors,
    prepare_aircraft_request,
)
from src.inference.onnx import predict_with_onnx

DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_OUTPUT_FILE = "data/predictions/latest_predictions.json"
DEFAULT_MAX_AIRCRAFT = 250


def generate_batch_predictions(params_path: str = "params.yaml") -> int:
    try:
        params = _load_params(params_path)
        processed_dir = _project_path(params["processed_dir"])
        source_snapshot = _latest_snapshot(processed_dir)
        if source_snapshot is None:
            raise FileNotFoundError(f"No processed OpenSky snapshot found in {processed_dir}")

        snapshot = _read_table(source_snapshot)
        aircraft_ids = _active_aircraft_ids(snapshot, params["max_aircraft"])
        history = load_history()
        preprocessors = load_preprocessors()
        predictions: dict[str, Any] = {}
        skipped: list[dict[str, str]] = []

        for icao24 in aircraft_ids:
            try:
                request = prepare_aircraft_request(icao24, history, preprocessors)
                prediction = predict_with_onnx(request)
                predictions[icao24] = {
                    **prediction,
                    "serving": {
                        "pattern": "batch_offline_prediction",
                        "service": "dvc-pipeline",
                        "runtime": "onnxruntime",
                    },
                }
            except (PredictionInputError, ValueError) as exc:
                skipped.append({"icao24": icao24, "reason": str(exc)})

        output_path = _project_path(params["output_file"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "available",
            "pattern": "batch_offline_prediction",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_snapshot": str(source_snapshot.relative_to(PROJECT_ROOT)),
            "count": len(predictions),
            "skipped_count": len(skipped),
            "skipped": skipped,
            "predictions": predictions,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Generated batch predictions: {len(predictions)}")
        print(f"Skipped aircraft: {len(skipped)}")
        print(f"Batch prediction file: {output_path}")
        return 0
    except Exception as exc:
        print(f"Batch prediction failed: {exc}")
        return 1


def _load_params(params_path: str) -> dict[str, Any]:
    params_file = _project_path(params_path)
    loaded = yaml.safe_load(params_file.read_text(encoding="utf-8")) if params_file.exists() else {}
    configured = loaded.get("batch_prediction", {}) if isinstance(loaded, dict) else {}
    return {
        "processed_dir": configured.get("processed_dir", DEFAULT_PROCESSED_DIR),
        "output_file": configured.get("output_file", DEFAULT_OUTPUT_FILE),
        "max_aircraft": int(configured.get("max_aircraft", DEFAULT_MAX_AIRCRAFT)),
    }


def _latest_snapshot(processed_dir: Path) -> Path | None:
    candidates = sorted(
        list(processed_dir.glob("states_processed_*.csv"))
        + list(processed_dir.glob("states_processed_*.parquet"))
    )
    return candidates[-1] if candidates else None


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _active_aircraft_ids(snapshot: pd.DataFrame, max_aircraft: int) -> list[str]:
    if "icao24" not in snapshot.columns:
        raise ValueError("Processed snapshot does not contain icao24.")
    values = snapshot["icao24"].dropna().astype(str).str.lower().str.strip()
    return [value for value in values.drop_duplicates().tolist() if value][:max_aircraft]


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(generate_batch_predictions())
