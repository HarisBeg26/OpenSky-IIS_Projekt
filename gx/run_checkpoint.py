from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import great_expectations as ge
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_HISTORY_FILE = DEFAULT_PROCESSED_DIR / "states_history.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "validation" / "opensky_validation.json"
DEFAULT_GX_DIR = PROJECT_ROOT / "gx"

EXPECTED_COLUMNS = [
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "longitude",
    "latitude",
    "baro_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "sensors",
    "geo_altitude",
    "squawk",
    "spi",
    "position_source",
    "snapshot_time",
    "snapshot_time_utc",
    "source_snapshot",
    "event_time_utc",
]


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_validate_params(params_path: Path = PROJECT_ROOT / "params.yaml") -> dict[str, str]:
    defaults = {
        "processed_dir": str(DEFAULT_PROCESSED_DIR),
        "history_file": str(DEFAULT_HISTORY_FILE),
        "report_path": str(DEFAULT_REPORT_PATH),
        "gx_dir": str(DEFAULT_GX_DIR),
        "datasource_name": "opensky_processed",
        "data_asset_name": "opensky_history",
        "expectation_suite_name": "opensky_history_suite",
        "checkpoint_name": "opensky_history_checkpoint",
        "docs_site_name": "local_site",
    }
    if not params_path.exists():
        return defaults

    loaded = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
    validate_params = loaded.get("validate", {}) if isinstance(loaded, dict) else {}
    return {
        "processed_dir": validate_params.get("processed_dir", defaults["processed_dir"]),
        "history_file": validate_params.get("history_file", defaults["history_file"]),
        "report_path": validate_params.get("report_path", defaults["report_path"]),
        "gx_dir": validate_params.get("gx_dir", defaults["gx_dir"]),
        "datasource_name": validate_params.get("datasource_name", defaults["datasource_name"]),
        "data_asset_name": validate_params.get("data_asset_name", defaults["data_asset_name"]),
        "expectation_suite_name": validate_params.get(
            "expectation_suite_name",
            defaults["expectation_suite_name"],
        ),
        "checkpoint_name": validate_params.get("checkpoint_name", defaults["checkpoint_name"]),
        "docs_site_name": validate_params.get("docs_site_name", defaults["docs_site_name"]),
    }


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


def _resolve_history_file(history_file: str | None, processed_dir: str | None, configured_history_file: str) -> Path:
    if history_file:
        return _project_path(history_file)
    if processed_dir:
        processed_path = _project_path(processed_dir)
        return _latest_processed(processed_path)

    configured_path = _project_path(configured_history_file)
    if configured_path.exists():
        return configured_path
    return _latest_processed(_project_path("data/processed"))


def _get_context(gx_dir: Path):
    gx_dir.mkdir(parents=True, exist_ok=True)
    return ge.get_context(context_root_dir=str(gx_dir))


def _ensure_datasource_and_asset(context, datasource_name: str, data_asset_name: str, history_path: Path):
    base_directory = str(history_path.parent)
    batching_regex = re.escape(history_path.name)

    try:
        datasource = context.get_datasource(datasource_name)
    except Exception:
        datasource = context.sources.add_pandas_filesystem(
            name=datasource_name,
            base_directory=base_directory,
        )

    try:
        return datasource.get_asset(data_asset_name)
    except Exception:
        if history_path.suffix.lower() == ".parquet":
            return datasource.add_parquet_asset(
                name=data_asset_name,
                batching_regex=batching_regex,
            )
        return datasource.add_csv_asset(
            name=data_asset_name,
            batching_regex=batching_regex,
        )


def _configure_expectation_suite(context, asset, expectation_suite_name: str):
    context.add_or_update_expectation_suite(expectation_suite_name=expectation_suite_name)
    batch_request = asset.build_batch_request()
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=expectation_suite_name,
    )

    validator.expectation_suite.expectations = []

    validator.expect_table_row_count_to_be_between(min_value=1)
    validator.expect_table_columns_to_match_ordered_list(column_list=EXPECTED_COLUMNS)

    validator.expect_column_values_to_not_be_null(column="icao24")
    validator.expect_column_values_to_match_regex(column="icao24", regex=r"^[0-9a-f]{6}$")
    validator.expect_column_values_to_not_be_null(column="last_contact")
    validator.expect_column_values_to_not_be_null(column="snapshot_time")
    validator.expect_column_values_to_not_be_null(column="source_snapshot")
    validator.expect_column_values_to_match_regex(
        column="source_snapshot",
        regex=r"^states_\d{8}T\d{6}Z\.json$",
    )
    validator.expect_column_values_to_not_be_null(column="origin_country", mostly=0.99)
    validator.expect_column_values_to_be_between(column="latitude", min_value=-90, max_value=90, mostly=0.99)
    validator.expect_column_values_to_be_between(column="longitude", min_value=-180, max_value=180, mostly=0.99)
    validator.expect_column_values_to_be_between(column="velocity", min_value=0, max_value=400, mostly=0.99)
    validator.expect_column_values_to_be_between(column="true_track", min_value=0, max_value=360, mostly=0.99)
    validator.expect_column_values_to_be_in_set(
        column="on_ground",
        value_set=[True, False, "True", "False"],
        mostly=1.0,
    )
    validator.expect_column_values_to_be_in_set(column="position_source", value_set=[0, 1, 2], mostly=0.99)
    validator.expect_column_values_to_match_regex(
        column="snapshot_time_utc",
        regex=r"^\d{4}-\d{2}-\d{2}.*\+\d{2}:\d{2}$",
        mostly=0.99,
    )
    validator.expect_column_values_to_match_regex(
        column="event_time_utc",
        regex=r"^\d{4}-\d{2}-\d{2}.*\+\d{2}:\d{2}$",
        mostly=0.99,
    )

    validator.save_expectation_suite(discard_failed_expectations=False)
    return batch_request


def _configure_checkpoint(context, checkpoint_name: str, batch_request, expectation_suite_name: str):
    return context.add_or_update_checkpoint(
        name=checkpoint_name,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": expectation_suite_name,
            }
        ],
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_report(
    report_path: Path,
    history_path: Path,
    checkpoint_name: str,
    expectation_suite_name: str,
    checkpoint_result,
    data_docs_sites: dict[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "passed" if checkpoint_result["success"] else "failed",
        "success": bool(checkpoint_result["success"]),
        "checkpoint_name": checkpoint_name,
        "expectation_suite_name": expectation_suite_name,
        "validation_file": str(history_path),
        "data_docs": _json_safe(data_docs_sites),
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_checkpoint(
    history_file: str | None = None,
    processed_dir: str | None = None,
    report_path: str | None = None,
    gx_dir: str | None = None,
) -> int:
    try:
        params = _load_validate_params()
        history_path = _resolve_history_file(history_file, processed_dir, params["history_file"])
        report_output = _project_path(report_path or params["report_path"])
        gx_root = _project_path(gx_dir or params["gx_dir"])

        context = _get_context(gx_root)
        asset = _ensure_datasource_and_asset(
            context,
            params["datasource_name"],
            params["data_asset_name"],
            history_path,
        )
        batch_request = _configure_expectation_suite(
            context,
            asset,
            params["expectation_suite_name"],
        )
        checkpoint = _configure_checkpoint(
            context,
            params["checkpoint_name"],
            batch_request,
            params["expectation_suite_name"],
        )

        checkpoint_result = checkpoint.run(run_id="opensky_validation_run")
        data_docs_sites = context.build_data_docs()
        _write_report(
            report_output,
            history_path,
            params["checkpoint_name"],
            params["expectation_suite_name"],
            checkpoint_result,
            data_docs_sites,
        )

        print(f"Validation file: {history_path}")
        print(f"Validation report: {report_output}")
        print(f"Data docs: {data_docs_sites}")

        if checkpoint_result["success"]:
            print("Great Expectations validation passed.")
            return 0

        print("Great Expectations validation failed.")
        return 1
    except Exception as exc:
        print(f"Great Expectations validation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(run_checkpoint())
