from __future__ import annotations

from pathlib import Path

import pandas as pd


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


def validate_opensky_data(
    input_file: str | None = None,
    processed_dir: str = "data/processed",
) -> int:
    try:
        path = Path(input_file) if input_file else _latest_processed(Path(processed_dir))

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