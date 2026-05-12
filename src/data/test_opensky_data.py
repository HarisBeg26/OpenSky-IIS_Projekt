from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from evidently import Report
from evidently.metrics import DriftedColumnsCount
from evidently.presets import DataDriftPreset, DataSummaryPreset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_REFERENCE_DIR = "data/reference/opensky"
DEFAULT_REFERENCE_FILE = "states_reference.csv"
DEFAULT_REPORT_HTML = "reports/evidently/opensky_data_drift_report.html"
DEFAULT_REPORT_JSON = "reports/evidently/opensky_data_drift_summary.json"
DEFAULT_MIN_ROWS = 30
DEFAULT_DRIFT_SHARE = 0.7
DEFAULT_DROP_COLUMNS = [
    "icao24",
    "callsign",
    "time_position",
    "last_contact",
    "sensors",
    "squawk",
    "snapshot_time",
    "snapshot_time_utc",
    "source_snapshot",
    "event_time_utc",
    "captured_at_utc",
]


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_test_params(params_path: str = "params.yaml") -> dict:
    defaults = {
        "processed_dir": DEFAULT_PROCESSED_DIR,
        "reference_dir": DEFAULT_REFERENCE_DIR,
        "reference_file": DEFAULT_REFERENCE_FILE,
        "report_html": DEFAULT_REPORT_HTML,
        "report_json": DEFAULT_REPORT_JSON,
        "min_rows": DEFAULT_MIN_ROWS,
        "drift_share": DEFAULT_DRIFT_SHARE,
        "drop_columns": DEFAULT_DROP_COLUMNS,
    }
    params_file = _project_path(params_path)
    if not params_file.exists():
        return defaults

    loaded = yaml.safe_load(params_file.read_text(encoding="utf-8")) or {}
    test_params = loaded.get("test_data", {}) if isinstance(loaded, dict) else {}
    return {
        "processed_dir": test_params.get("processed_dir", defaults["processed_dir"]),
        "reference_dir": test_params.get("reference_dir", defaults["reference_dir"]),
        "reference_file": test_params.get("reference_file", defaults["reference_file"]),
        "report_html": test_params.get("report_html", defaults["report_html"]),
        "report_json": test_params.get("report_json", defaults["report_json"]),
        "min_rows": int(test_params.get("min_rows", defaults["min_rows"])),
        "drift_share": float(test_params.get("drift_share", defaults["drift_share"])),
        "drop_columns": test_params.get("drop_columns", defaults["drop_columns"]),
    }


def _read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _latest_processed_snapshot(processed_dir: Path) -> Path:
    candidates = sorted(
        list(processed_dir.glob("states_processed_*.csv"))
        + list(processed_dir.glob("states_processed_*.parquet"))
    )
    if not candidates:
        raise FileNotFoundError(f"No processed snapshot files found in {processed_dir}")
    return candidates[-1]


def _resolve_paths(
    current_file: str | None,
    processed_dir: str,
    reference_dir: str,
    reference_file: str,
    report_html: str,
    report_json: str,
) -> tuple[Path, Path, Path, Path]:
    current_path = _project_path(current_file) if current_file else _latest_processed_snapshot(_project_path(processed_dir))
    reference_path = _project_path(reference_dir) / reference_file
    report_html_path = _project_path(report_html)
    report_json_path = _project_path(report_json)
    return current_path, reference_path, report_html_path, report_json_path


def _align_for_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    drop_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    shared_columns = [column for column in current.columns if column in reference.columns and column not in set(drop_columns)]
    if not shared_columns:
        raise ValueError("No comparable columns remain after applying drop_columns.")

    reference_eval = reference[shared_columns].copy()
    current_eval = current[shared_columns].copy()

    removed_empty: list[str] = []
    for column in list(shared_columns):
        if reference_eval[column].dropna().empty or current_eval[column].dropna().empty:
            del reference_eval[column]
            del current_eval[column]
            removed_empty.append(column)

    if reference_eval.empty or current_eval.empty:
        raise ValueError("No non-empty comparable columns remain for Evidently drift testing.")

    used_columns = list(current_eval.columns)
    return reference_eval, current_eval, used_columns, removed_empty


def _collect_failed_tests(report_dict: dict) -> list[dict]:
    tests = report_dict.get("tests", [])
    return [test for test in tests if test.get("status") != "SUCCESS"]


def _evaluate_gate(
    current_rows: int,
    reference_rows: int,
    total_tests: int,
    failed_tests: list[dict],
    min_rows: int,
    drift_share: float,
) -> tuple[bool, str, dict]:
    failed_count = len(failed_tests)

    gate_details = {
        "current_rows": current_rows,
        "reference_rows": reference_rows,
        "min_rows": min_rows,
        "dataset_drift_share_threshold": drift_share,
        "total_tests": total_tests,
        "failed_test_count": failed_count,
    }

    if current_rows < min_rows or reference_rows < min_rows:
        reason = (
            "Insufficient rows for strict drift gating: "
            f"current={current_rows}, reference={reference_rows}, required>={min_rows}."
        )
        gate_details["decision"] = "insufficient_rows"
        return True, reason, gate_details

    if failed_count > 0:
        reason = (
            "Dataset-level Evidently drift detected. "
            f"Threshold share={drift_share:.0%}."
        )
        gate_details["decision"] = "dataset_drift_detected"
        return False, reason, gate_details

    reason = (
        "Dataset-level Evidently drift not detected. "
        f"Threshold share={drift_share:.0%}."
    )
    gate_details["decision"] = "passed"
    return True, reason, gate_details


def _write_summary(
    report_json_path: Path,
    current_path: Path,
    reference_path: Path,
    report_html_path: Path,
    result_dict: dict,
    used_columns: list[str],
    removed_empty: list[str],
    failed_tests: list[dict],
    gate_passed: bool,
    gate_reason: str,
    gate_details: dict,
) -> None:
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "passed" if gate_passed else "failed",
        "current_file": str(current_path),
        "reference_file": str(reference_path),
        "report_html": str(report_html_path),
        "used_columns": used_columns,
        "removed_empty_columns": removed_empty,
        "test_count": len(result_dict.get("tests", [])),
        "failed_test_count": len(failed_tests),
        "gate_reason": gate_reason,
        "gate_details": gate_details,
        "failed_tests": [
            {
                "name": test.get("name"),
                "group": test.get("group"),
                "status": test.get("status"),
                "description": test.get("description"),
            }
            for test in failed_tests
        ],
    }
    report_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def test_opensky_data(
    current_file: str | None = None,
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    reference_dir: str | None = None,
    report_html: str | None = None,
) -> int:
    try:
        params = _load_test_params()
        effective_processed_dir = processed_dir if processed_dir != DEFAULT_PROCESSED_DIR else params["processed_dir"]
        effective_reference_dir = reference_dir if reference_dir else params["reference_dir"]
        effective_report_html = report_html if report_html else params["report_html"]

        current_path, reference_path, report_html_path, report_json_path = _resolve_paths(
            current_file=current_file,
            processed_dir=effective_processed_dir,
            reference_dir=effective_reference_dir,
            reference_file=params["reference_file"],
            report_html=effective_report_html,
            report_json=params["report_json"],
        )

        current = _read_dataframe(current_path)
        if not reference_path.exists():
            print(f"Reference file not found. Copying current snapshot to {reference_path}")
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            current.to_csv(reference_path, index=False)

        reference = _read_dataframe(reference_path)
        reference_eval, current_eval, used_columns, removed_empty = _align_for_drift(
            reference=reference,
            current=current,
            drop_columns=list(params["drop_columns"]),
        )

        visual_report = Report(
            [
                DataSummaryPreset(),
                DataDriftPreset(drift_share=float(params["drift_share"])),
            ],
            include_tests=False,
        )
        visual_result = visual_report.run(current_data=current_eval, reference_data=reference_eval)

        gate_report = Report(
            [
                DriftedColumnsCount(drift_share=float(params["drift_share"])),
            ],
            include_tests=True,
        )
        gate_result = gate_report.run(current_data=current_eval, reference_data=reference_eval)

        report_html_path.parent.mkdir(parents=True, exist_ok=True)
        visual_result.save_html(str(report_html_path))

        result_dict = gate_result.dict()
        failed_tests = _collect_failed_tests(result_dict)
        gate_passed, gate_reason, gate_details = _evaluate_gate(
            current_rows=len(current_eval),
            reference_rows=len(reference_eval),
            total_tests=len(result_dict.get("tests", [])),
            failed_tests=failed_tests,
            min_rows=int(params["min_rows"]),
            drift_share=float(params["drift_share"]),
        )

        gate_metric = {
            "drift_share_threshold": float(params["drift_share"]),
            "tested_columns": len(used_columns),
        }
        metrics = result_dict.get("metrics", [])
        if metrics:
            gate_metric["raw_metric"] = metrics[0]

        gate_details["metric"] = gate_metric

        _write_summary(
            report_json_path=report_json_path,
            current_path=current_path,
            reference_path=reference_path,
            report_html_path=report_html_path,
            result_dict=result_dict,
            used_columns=used_columns,
            removed_empty=removed_empty,
            failed_tests=failed_tests,
            gate_passed=gate_passed,
            gate_reason=gate_reason,
            gate_details=gate_details,
        )

        print(f"Current snapshot: {current_path}")
        print(f"Reference snapshot: {reference_path}")
        print(f"Compared columns: {used_columns}")
        print(f"Evidently report: {report_html_path}")
        print(f"Evidently summary: {report_json_path}")
        print(gate_reason)

        if not gate_passed:
            print("Evidently data drift gate failed.")
            return 1

        print("Evidently data drift gate passed.")
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        current.to_csv(reference_path, index=False)
        if current_path.resolve() != reference_path.resolve():
            print("Updated reference snapshot with the current processed snapshot.")
        return 0
    except Exception as exc:
        print(f"Evidently testing failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(test_opensky_data())
