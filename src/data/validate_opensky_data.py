from __future__ import annotations

from pathlib import Path

import yaml

from gx.run_checkpoint import run_checkpoint

DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_REPORT_PATH = "reports/validation/opensky_validation.json"


def _load_validate_params(params_path: str = "params.yaml") -> dict:
    defaults = {
        "processed_dir": DEFAULT_PROCESSED_DIR,
        "history_file": None,
        "report_path": DEFAULT_REPORT_PATH,
    }
    params_file = Path(params_path)
    if not params_file.exists():
        return defaults

    loaded = yaml.safe_load(params_file.read_text(encoding="utf-8")) or {}
    validate_params = loaded.get("validate", {}) if isinstance(loaded, dict) else {}
    return {
        "processed_dir": validate_params.get("processed_dir", defaults["processed_dir"]),
        "history_file": validate_params.get("history_file", defaults["history_file"]),
        "report_path": validate_params.get("report_path", defaults["report_path"]),
    }


def validate_opensky_data(
    input_file: str | None = None,
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    report_path: str | None = None,
) -> int:
    params = _load_validate_params()
    effective_processed_dir = processed_dir if processed_dir != DEFAULT_PROCESSED_DIR else params["processed_dir"]
    effective_history_file = input_file if input_file else params["history_file"]
    effective_report_path = report_path if report_path else params["report_path"]

    return run_checkpoint(
        history_file=effective_history_file,
        processed_dir=effective_processed_dir,
        report_path=effective_report_path,
    )
