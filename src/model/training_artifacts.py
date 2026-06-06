from __future__ import annotations

from pathlib import Path

from .compression import build_compression_report, write_float16_quantized_weights
from .explainability import build_explainability_report
from .training_config import PROJECT_ROOT, project_path
from .training_models import (
    clean_models_dir,
    export_keras_model_to_onnx,
    write_json,
    write_pickle,
)


def save_artifacts_and_reports(
    models: dict,
    data: dict,
    evaluation: dict,
    metrics: dict,
    params: dict,
    tf,
    tf2onnx,
) -> list[Path]:
    paths = _artifact_paths(project_path(params["models_dir"]))
    clean_models_dir(paths["models_dir"])

    models["trajectory_model"].save(paths["trajectory_keras"])
    models["ground_model"].save(paths["ground_keras"])
    _export_onnx_models(models, paths, params, tf, tf2onnx)
    metrics["onnx"] = {
        "trajectory_lstm": _relative(paths["trajectory_onnx"]),
        "on_ground_lstm": _relative(paths["ground_onnx"]),
        "opset": params["onnx_opset"],
    }

    compression_report_path, compressed_artifacts = _write_compression_report(models, paths, metrics, params)
    explainability_report_path = _write_explainability_report(models, data, evaluation, metrics, params)
    write_pickle(paths["preprocessors"], _preprocessor_payload(data, params, evaluation))

    return [
        paths["trajectory_keras"],
        paths["ground_keras"],
        paths["trajectory_onnx"],
        paths["ground_onnx"],
        paths["trajectory_quantized"],
        paths["ground_quantized"],
        paths["preprocessors"],
        compression_report_path,
        explainability_report_path,
        *compressed_artifacts,
    ]


def _write_compression_report(
    models: dict,
    paths: dict[str, Path],
    metrics: dict,
    params: dict,
) -> tuple[Path, list[Path]]:
    quantization_report = {
        "method": params["compression_method"],
        "dtype": params["quantization_dtype"],
        "trajectory_lstm": write_float16_quantized_weights(
            models["trajectory_model"],
            paths["trajectory_quantized"],
        ),
        "on_ground_lstm": write_float16_quantized_weights(
            models["ground_model"],
            paths["ground_quantized"],
        ),
    }
    compression_report, compressed_artifacts = build_compression_report(
        {
            "trajectory_lstm": {
                "keras": paths["trajectory_keras"],
                "onnx": paths["trajectory_onnx"],
                "quantized_weights": paths["trajectory_quantized"],
            },
            "on_ground_lstm": {
                "keras": paths["ground_keras"],
                "onnx": paths["ground_onnx"],
                "quantized_weights": paths["ground_quantized"],
            },
        },
        params,
    )
    compression_report["quantization"] = quantization_report
    compression_report_path = project_path(params["compression_report_path"])
    write_json(compression_report_path, compression_report)
    metrics["compression"] = {
        "report_path": _relative(compression_report_path),
        "primary_method": compression_report["primary_method"],
        "quantization_dtype": compression_report["quantization_dtype"],
        "summary": compression_report["summary"],
    }
    return compression_report_path, compressed_artifacts


def _write_explainability_report(
    models: dict,
    data: dict,
    evaluation: dict,
    metrics: dict,
    params: dict,
) -> Path:
    explainability_report_path = project_path(params["explainability_report_path"])
    explainability_report = build_explainability_report(
        models["trajectory_model"],
        models["ground_model"],
        data["X_test"],
        data["X_test_raw"],
        data["y_position_test"],
        data["y_ground_test"],
        data["target_scaler"],
        data["feature_columns"],
        params,
        evaluation["ground_threshold"],
    )
    write_json(explainability_report_path, explainability_report)
    metrics["explainability"] = {
        "report_path": _relative(explainability_report_path),
        "status": explainability_report.get("status"),
        "method": explainability_report.get("method"),
        "top_trajectory_features": [
            item["feature"] for item in explainability_report.get("top_trajectory_features", [])[:5]
        ],
        "top_on_ground_features": [
            item["feature"] for item in explainability_report.get("top_on_ground_features", [])[:5]
        ],
    }
    return explainability_report_path


def _artifact_paths(models_dir: Path) -> dict[str, Path]:
    return {
        "models_dir": models_dir,
        "trajectory_keras": models_dir / "trajectory_lstm.keras",
        "ground_keras": models_dir / "on_ground_lstm.keras",
        "trajectory_onnx": models_dir / "trajectory_lstm.onnx",
        "ground_onnx": models_dir / "on_ground_lstm.onnx",
        "trajectory_quantized": models_dir / "trajectory_lstm.float16_weights.npz",
        "ground_quantized": models_dir / "on_ground_lstm.float16_weights.npz",
        "preprocessors": models_dir / "preprocessors.pkl",
    }


def _export_onnx_models(models: dict, paths: dict[str, Path], params: dict, tf, tf2onnx) -> None:
    export_keras_model_to_onnx(
        models["trajectory_model"],
        paths["trajectory_onnx"],
        models["input_shape"],
        tf,
        tf2onnx,
        params["onnx_opset"],
        "trajectory_output",
    )
    export_keras_model_to_onnx(
        models["ground_model"],
        paths["ground_onnx"],
        models["input_shape"],
        tf,
        tf2onnx,
        params["onnx_opset"],
        "on_ground_output",
    )


def _preprocessor_payload(data: dict, params: dict, evaluation: dict) -> dict:
    return {
        "feature_imputer": data["feature_imputer"],
        "feature_scaler": data["feature_scaler"],
        "target_scaler": data["target_scaler"],
        "trajectory_target_mode": params["trajectory_target_mode"],
        "on_ground_threshold": evaluation["ground_threshold"],
        "feature_columns": data["feature_columns"],
        "window_size": params["window_size"],
    }


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))
