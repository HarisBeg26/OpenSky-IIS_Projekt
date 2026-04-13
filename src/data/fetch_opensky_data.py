from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

OPEN_SKY_URL = "https://opensky-network.org/api/states/all"


def fetch_opensky_data(output_dir: str = "data/raw") -> int:
    try:
        username = os.getenv("OPENSKY_USERNAME")
        password = os.getenv("OPENSKY_PASSWORD")
        auth = (username, password) if username and password else None
        url = os.getenv("OPENSKY_URL", OPEN_SKY_URL)

        response = requests.get(
            url,
            auth=auth,
            headers={"User-Agent": "SkyWatch/0.1"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        out_dir = Path(output_dir)
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