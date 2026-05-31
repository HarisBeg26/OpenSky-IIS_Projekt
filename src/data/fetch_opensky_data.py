from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

OPEN_SKY_URL = "https://opensky-network.org/api/states/all"
DEFAULT_RAW_DIR = "data/raw"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_RETRIES = 5
DEFAULT_RETRY_BACKOFF_SECONDS = 20
DEFAULT_ALLOW_CACHED_ON_FAILURE = True
DEFAULT_MAX_CACHED_AGE_HOURS = 12


def _load_fetch_params(params_path: str = "params.yaml") -> dict:
    defaults = {
        "url": OPEN_SKY_URL,
        "output_dir": DEFAULT_RAW_DIR,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "retries": DEFAULT_RETRIES,
        "retry_backoff_seconds": DEFAULT_RETRY_BACKOFF_SECONDS,
        "allow_cached_on_failure": DEFAULT_ALLOW_CACHED_ON_FAILURE,
        "max_cached_age_hours": DEFAULT_MAX_CACHED_AGE_HOURS,
        "bbox": None,
    }
    params_file = Path(params_path)
    if not params_file.exists():
        return defaults

    loaded = yaml.safe_load(params_file.read_text(encoding="utf-8")) or {}
    fetch_params = loaded.get("fetch", {}) if isinstance(loaded, dict) else {}
    bbox = fetch_params.get("bbox")
    return {
        "url": fetch_params.get("url", defaults["url"]),
        "output_dir": fetch_params.get("output_dir", defaults["output_dir"]),
        "timeout_seconds": fetch_params.get("timeout_seconds", defaults["timeout_seconds"]),
        "retries": fetch_params.get("retries", defaults["retries"]),
        "retry_backoff_seconds": fetch_params.get(
            "retry_backoff_seconds", defaults["retry_backoff_seconds"]
        ),
        "allow_cached_on_failure": fetch_params.get(
            "allow_cached_on_failure", defaults["allow_cached_on_failure"]
        ),
        "max_cached_age_hours": fetch_params.get(
            "max_cached_age_hours", defaults["max_cached_age_hours"]
        ),
        "bbox": bbox if isinstance(bbox, dict) else defaults["bbox"],
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _build_bbox_query(bbox: dict | None) -> dict[str, float]:
    if not bbox:
        return {}

    required_keys = ("lamin", "lomin", "lamax", "lomax")
    missing = [key for key in required_keys if key not in bbox]
    if missing:
        raise ValueError(f"Bounding box is missing keys: {missing}")

    return {key: float(bbox[key]) for key in required_keys}


def _latest_cached_snapshot(output_dir: str | Path) -> Path | None:
    out_dir = Path(output_dir)
    if not out_dir.exists():
        return None

    snapshots = sorted(out_dir.glob("states_*.json"))
    return snapshots[-1] if snapshots else None


def _cached_snapshot_age_hours(snapshot_path: Path) -> float:
    modified_at = datetime.fromtimestamp(snapshot_path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - modified_at).total_seconds() / 3600


def _request_opensky_payload(
    url: str,
    auth: tuple[str, str] | None,
    query_params: dict[str, float],
    timeout_seconds: int,
) -> dict:
    response = requests.get(
        url,
        auth=auth,
        params=query_params or None,
        headers={"User-Agent": "SkyWatch/0.1"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def fetch_opensky_data(output_dir: str = DEFAULT_RAW_DIR) -> int:
    try:
        params = _load_fetch_params()
        username = os.getenv("OPENSKY_USERNAME")
        password = os.getenv("OPENSKY_PASSWORD")
        auth = (username, password) if username and password else None
        configured_url = params["url"]
        url = os.getenv("OPENSKY_URL", configured_url)
        timeout_seconds = int(params["timeout_seconds"])
        retries = max(1, int(params["retries"]))
        retry_backoff_seconds = max(0, int(params["retry_backoff_seconds"]))
        allow_cached_on_failure = _as_bool(params["allow_cached_on_failure"])
        max_cached_age_hours = float(params["max_cached_age_hours"])
        query_params = _build_bbox_query(params["bbox"])

        configured_output_dir = params["output_dir"]
        effective_output_dir = output_dir if output_dir != DEFAULT_RAW_DIR else configured_output_dir

        last_error: requests.RequestException | None = None
        for attempt in range(1, retries + 1):
            try:
                payload = _request_opensky_payload(url, auth, query_params, timeout_seconds)
                break
            except requests.RequestException as exc:
                last_error = exc
                print(f"Fetch attempt {attempt}/{retries} failed: {exc}")
                if attempt < retries and retry_backoff_seconds:
                    time.sleep(retry_backoff_seconds * attempt)
        else:
            cached_snapshot = _latest_cached_snapshot(effective_output_dir)
            if allow_cached_on_failure and cached_snapshot:
                cached_age_hours = _cached_snapshot_age_hours(cached_snapshot)
                if cached_age_hours > max_cached_age_hours:
                    print(
                        "Cached raw snapshot is too old for fallback: "
                        f"{cached_snapshot} is {cached_age_hours:.2f}h old "
                        f"(max {max_cached_age_hours:.2f}h)."
                    )
                    print(f"Fetch failed: {last_error}")
                    return 1

                print(f"OpenSky fetch unavailable, using cached raw snapshot: {cached_snapshot}")
                print(f"Cached snapshot age: {cached_age_hours:.2f}h (max {max_cached_age_hours:.2f}h).")
                print("DVC pipeline will continue with existing raw data.")
                return 0

            print(f"Fetch failed: {last_error}")
            return 1

        out_dir = Path(effective_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_file = out_dir / f"states_{ts}.json"
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Saved OpenSky snapshot to: {out_file}")
        print(f"Snapshot time: {payload.get('time')}, records: {len(payload.get('states') or [])}")
        if query_params:
            print(f"Bounding box: {query_params}")
        return 0
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return 1
