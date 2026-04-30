from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_REPORT_PATH = "reports/validation/opensky_validation.json"


def _latest_processed(processed_dir: Path) -> Path:
    history_candidates = [
        processed_dir / "states_history.parquet",
        processed_dir / "states_history.csv",
    ]
    for history_file in history_candidates:
        if history_file.exists():
            return history_file

    candidates = sorted(
        list(processed_dir.glob("states_processed_*.parquet"))
        + list(processed_dir.glob("states_processed_*.csv"))
    )
    if not candidates:
        raise FileNotFoundError(f"No processed files found in {processed_dir}")
    return candidates[-1]


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


def _load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _append_required_column_issues(df: pd.DataFrame, issues: list[str]) -> None:
    required = {"icao24", "last_contact", "longitude", "latitude", "velocity", "true_track"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")


def _append_null_issue(df: pd.DataFrame, column: str, issues: list[str]) -> None:
    if column not in df.columns:
        return
    null_count = int(df[column].isna().sum())
    if null_count > 0:
        issues.append(f"{column} null count = {null_count}")


def _append_range_issue(
    df: pd.DataFrame,
    column: str,
    low: float,
    high: float,
    issues: list[str],
) -> None:
    if column not in df.columns:
        return
    mask = (~df[column].between(low, high)) & df[column].notna()
    count = int(mask.sum())
    if count > 0:
        issues.append(f"invalid {column} count = {count}")


def _append_negative_issue(df: pd.DataFrame, column: str, issues: list[str]) -> None:
    if column not in df.columns:
        return
    mask = (df[column] < 0) & df[column].notna()
    count = int(mask.sum())
    if count > 0:
        issues.append(f"negative {column} count = {count}")


def _write_report(report_path: Path, validation_file: Path, row_count: int, issues: list[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "passed" if not issues else "failed",
        "validation_file": str(validation_file),
        "row_count": row_count,
        "issue_count": len(issues),
        "issues": issues,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_opensky_data(
    input_file: str | None = None,
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    report_path: str | None = None,
) -> int:
    try:
        params = _load_validate_params()
        effective_processed_dir = processed_dir if processed_dir != DEFAULT_PROCESSED_DIR else params["processed_dir"]
        effective_report_path = Path(report_path) if report_path else Path(params["report_path"])

        configured_history_file = params["history_file"]
        if input_file:
            path = Path(input_file)
        elif configured_history_file:
            path = Path(configured_history_file)
        else:
            path = _latest_processed(Path(effective_processed_dir))

        df = _load_dataframe(path)

        issues: list[str] = []

        _append_required_column_issues(df, issues)
        _append_null_issue(df, "icao24", issues)
        _append_range_issue(df, "latitude", -90, 90, issues)
        _append_range_issue(df, "longitude", -180, 180, issues)
        _append_negative_issue(df, "velocity", issues)
        _append_range_issue(df, "true_track", 0, 360, issues)

        print(f"Validation file: {path}")
        print(f"Rows: {len(df)}")
        _write_report(effective_report_path, path, len(df), issues)
        print(f"Validation report: {effective_report_path}")

        if issues:
            print("VALIDATION FAILED")
            for issue in issues:
                print(f"- {issue}")
            return 1

        print("VALIDATION PASSED")
        return 0
    except Exception as exc:
        print(f"Validation failed: {exc}")
        return 1
