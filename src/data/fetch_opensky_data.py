from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

OPEN_SKY_URL = "https://opensky-network.org/api/states/all"
DEFAULT_RAW_DIR = "data/raw"


def _load_fetch_params(params_path: str = "params.yaml") -> dict:
    defaults = {
        "url": OPEN_SKY_URL,
        "output_dir": DEFAULT_RAW_DIR,
    }
    params_file = Path(params_path)
    if not params_file.exists():
        return defaults

    loaded = yaml.safe_load(params_file.read_text(encoding="utf-8")) or {}
    fetch_params = loaded.get("fetch", {}) if isinstance(loaded, dict) else {}
    return {
        "url": fetch_params.get("url", defaults["url"]),
        "output_dir": fetch_params.get("output_dir", defaults["output_dir"]),
    }


def fetch_opensky_data(output_dir: str = DEFAULT_RAW_DIR) -> int:
    try:
        params = _load_fetch_params()
        username = os.getenv("OPENSKY_USERNAME")
        password = os.getenv("OPENSKY_PASSWORD")
        auth = (username, password) if username and password else None
        configured_url = params["url"]
        url = os.getenv("OPENSKY_URL", configured_url)

        configured_output_dir = params["output_dir"]
        effective_output_dir = output_dir if output_dir != DEFAULT_RAW_DIR else configured_output_dir

        response = requests.get(
            url,
            auth=auth,
            headers={"User-Agent": "SkyWatch/0.1"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        out_dir = Path(effective_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_file = out_dir / f"states_{ts}.json"
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Saved OpenSky snapshot to: {out_file}")
        print(f"Snapshot time: {payload.get('time')}, records: {len(payload.get('states') or [])}")
        return 0
    except requests.RequestException as exc:
        print(f"Fetch failed: {exc}")
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return 1