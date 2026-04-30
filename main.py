from __future__ import annotations

import argparse

from src.data.fetch_opensky_data import fetch_opensky_data
from src.data.preprocess_opensky_data import preprocess_opensky_data
from src.data.validate_opensky_data import validate_opensky_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SkyWatch MVP CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Fetch one OpenSky snapshot")
    ingest.add_argument("--output-dir", default="data/raw", help="Raw snapshot directory")

    preprocess = subparsers.add_parser("preprocess", help="Flatten all raw snapshots and rebuild history")
    preprocess.add_argument("--raw-dir", default="data/raw", help="Directory with raw snapshots")
    preprocess.add_argument("--output-dir", default="data/processed", help="Processed directory")
    preprocess.add_argument("--format", choices=["csv", "parquet"], default="csv")

    validate = subparsers.add_parser("validate", help="Run the Great Expectations checkpoint for processed flight data")
    validate.add_argument("--input-file", default=None, help="Optional explicit processed file")
    validate.add_argument("--processed-dir", default="data/processed", help="Directory with processed files")
    validate.add_argument("--report-path", default=None, help="Optional validation report output path")

    test_data = subparsers.add_parser("test-data", help="Run Evidently drift checks for the latest processed snapshot")
    test_data.add_argument("--current-file", default=None, help="Optional explicit processed snapshot file")
    test_data.add_argument("--processed-dir", default="data/processed", help="Directory with processed snapshot files")
    test_data.add_argument("--reference-dir", default=None, help="Optional reference dataset directory")
    test_data.add_argument("--report-html", default=None, help="Optional Evidently HTML report path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        raise SystemExit(fetch_opensky_data(output_dir=args.output_dir))
    if args.command == "preprocess":
        raise SystemExit(
            preprocess_opensky_data(
                raw_dir=args.raw_dir,
                output_dir=args.output_dir,
                output_format=args.format,
            )
        )
    if args.command == "validate":
        raise SystemExit(
            validate_opensky_data(
                input_file=args.input_file,
                processed_dir=args.processed_dir,
                report_path=args.report_path,
            )
        )
    if args.command == "test-data":
        from src.data.test_opensky_data import test_opensky_data

        raise SystemExit(
            test_opensky_data(
                current_file=args.current_file,
                processed_dir=args.processed_dir,
                reference_dir=args.reference_dir,
                report_html=args.report_html,
            )
        )


if __name__ == "__main__":
    main()
