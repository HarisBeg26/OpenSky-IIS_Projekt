from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "data/processed/states_history.csv",
    "data/predictions/latest_predictions.json",
    "models/opensky/trajectory_lstm.keras",
    "models/opensky/trajectory_lstm.onnx",
    "models/opensky/on_ground_lstm.keras",
    "models/opensky/on_ground_lstm.onnx",
    "models/opensky/preprocessors.pkl",
    "reports/validation/opensky_validation.json",
    "reports/evidently/opensky_data_drift_report.html",
    "reports/evidently/opensky_data_drift_summary.json",
    "reports/model_training/opensky_metrics.json",
    "reports/model_monitoring/production_model_monitoring.json",
    "reports/model_compression/opensky_compression.json",
    "reports/model_explainability/opensky_explainability.json",
    "reports/deployment/model_deployment_patterns.json",
    "gx/uncommitted/data_docs/local_site/index.html",
]


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify files required for a local SkyWatch run.")
    parser.add_argument(
        "--require-frontend",
        action="store_true",
        help="Also require the production React build in frontend/dist.",
    )
    args = parser.parse_args()

    required = list(REQUIRED_FILES)
    if args.require_frontend:
        required.append("frontend/dist/index.html")

    missing = [path for path in required if not (PROJECT_ROOT / path).is_file()]
    dotenv = _load_dotenv(PROJECT_ROOT / ".env")

    print("SkyWatch local setup verification")
    print(f"[{'PASS' if not missing else 'FAIL'}] Required artifacts: {len(required) - len(missing)}/{len(required)}")
    for path in missing:
        print(f"  - missing: {path}")

    python_ok = sys.version_info[:2] == (3, 11)
    print(f"[{'PASS' if python_ok else 'FAIL'}] Python: {sys.version.split()[0]} (required: 3.11.x)")

    token = os.getenv("HF_TOKEN") or dotenv.get("HF_TOKEN", "")
    if token and token.startswith("hf_"):
        print("[PASS] HuggingFace token is configured.")
    else:
        print("[WARN] HF_TOKEN is not configured; the external pretrained model will be unavailable.")

    dvc_access = bool(
        (os.getenv("DAGSHUB_ACCESS_KEY_ID") or dotenv.get("DAGSHUB_ACCESS_KEY_ID"))
        and (os.getenv("DAGSHUB_SECRET_ACCESS_KEY") or dotenv.get("DAGSHUB_SECRET_ACCESS_KEY"))
    )
    local_dvc_config = PROJECT_ROOT / ".dvc" / "config.local"
    if local_dvc_config.exists():
        local_config_text = local_dvc_config.read_text(encoding="utf-8")
        dvc_access = dvc_access or (
            "access_key_id" in local_config_text and "secret_access_key" in local_config_text
        )
    print(
        f"[{'PASS' if dvc_access else 'WARN'}] "
        "DagsHub credentials are configured."
        if dvc_access
        else "[WARN] DagsHub credentials are absent; future dvc pull/push commands need them."
    )

    if missing:
        print("\nRun the setup script after configuring DagsHub credentials in .env.")
        return 1
    if not python_ok:
        return 1

    print("\nLocal runtime prerequisites are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
