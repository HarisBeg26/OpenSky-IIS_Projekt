from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "src" / "app" / "static"

app = FastAPI(title="SkyWatch OpenSky Intelligence", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin", include_in_schema=False)
def admin() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    latest_snapshot = _latest_processed_snapshot()
    return {
        "status": "ok",
        "service": "skywatch",
        "latest_processed_snapshot": str(latest_snapshot.relative_to(PROJECT_ROOT)) if latest_snapshot else None,
        "models_available": _models_available(),
    }


@app.get("/api/flights")
def flights(limit: int = Query(default=150, ge=1, le=1000)) -> dict[str, Any]:
    snapshot = _latest_processed_snapshot()
    if snapshot is None:
        return {"snapshot": None, "count": 0, "flights": []}

    df = _read_table(snapshot).tail(limit)
    records = []
    for row in df.to_dict(orient="records"):
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
                "status": _flight_status(row),
            }
        )

    return {
        "snapshot": snapshot.name,
        "count": len(records),
        "flights": records,
    }


@app.get("/api/admin/summary")
def admin_summary() -> dict[str, Any]:
    return {
        "validation": _read_json("reports/validation/opensky_validation.json"),
        "drift": _read_json("reports/evidently/opensky_data_drift_summary.json"),
        "training": _read_json("reports/model_training/opensky_metrics.json"),
        "monitoring": _read_json("reports/model_monitoring/production_model_monitoring.json"),
        "models": _model_inventory(),
        "latest_snapshot": _latest_snapshot_summary(),
    }


@app.get("/api/admin/model-monitoring")
def model_monitoring() -> dict[str, Any]:
    return _read_json("reports/model_monitoring/production_model_monitoring.json") or {
        "status": "missing",
        "message": "Model monitoring report has not been generated yet.",
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


def _flight_status(row: dict[str, Any]) -> str:
    vertical_rate = abs(_number(row.get("vertical_rate")) or 0)
    velocity = _number(row.get("velocity")) or 0
    altitude = _number(row.get("baro_altitude")) or _number(row.get("geo_altitude")) or 0
    if _bool(row.get("on_ground")):
        return "ground"
    if vertical_rate > 12 or velocity > 180:
        return "watch"
    if altitude < 150 and velocity > 35:
        return "low-altitude"
    return "normal"
