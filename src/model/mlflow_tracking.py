from __future__ import annotations

from pathlib import Path


def log_and_register_mlflow_models(
    mlflow,
    trajectory_model,
    ground_model,
    params: dict,
) -> dict[str, object]:
    registered_model_names = {
        "trajectory_lstm": params["trajectory_registered_model_name"],
        "on_ground_lstm": params["on_ground_registered_model_name"],
    }
    registration_enabled = params["register_models"]

    if params.get("mlflow_tracking_mode") == "local" and not registration_enabled:
        return {
            "enabled": False,
            "tracking": "artifact_only",
            "registered_model_names": registered_model_names,
            "trajectory_lstm": None,
            "on_ground_lstm": None,
        }

    trajectory_info = mlflow.keras.log_model(
        trajectory_model,
        artifact_path="trajectory_lstm_mlflow_model",
        registered_model_name=registered_model_names["trajectory_lstm"] if registration_enabled else None,
        await_registration_for=params["await_model_registration_seconds"],
    )
    ground_info = mlflow.keras.log_model(
        ground_model,
        artifact_path="on_ground_lstm_mlflow_model",
        registered_model_name=registered_model_names["on_ground_lstm"] if registration_enabled else None,
        await_registration_for=params["await_model_registration_seconds"],
    )

    return {
        "enabled": registration_enabled,
        "registered_model_names": registered_model_names,
        "trajectory_lstm": _model_info_to_dict(trajectory_info),
        "on_ground_lstm": _model_info_to_dict(ground_info),
    }


def log_mlflow_params(
    mlflow,
    params: dict,
    feature_columns: list[str],
    training_sequences: int,
    test_sequences: int,
) -> None:
    mlflow.log_params(
        {
            "test_size": params["test_size"],
            "max_sequences": params["max_sequences"],
            "window_size": params["window_size"],
            "max_sequence_gap_seconds": params["max_sequence_gap_seconds"],
            "trajectory_target_mode": params["trajectory_target_mode"],
            "random_state": params["random_state"],
            "lstm_units": params["lstm_units"],
            "dense_units": params["dense_units"],
            "dropout": params["dropout"],
            "epochs": params["epochs"],
            "batch_size": params["batch_size"],
            "validation_split": params["validation_split"],
            "patience": params["patience"],
            "onnx_opset": params["onnx_opset"],
            "register_models": params["register_models"],
            "mlflow_tracking_mode": params.get("mlflow_tracking_mode", "configured"),
            "local_mlflow_fallback": params["local_mlflow_fallback"],
            "require_remote_mlflow": params["require_remote_mlflow"],
            "compression_enabled": params["compression_enabled"],
            "compression_method": params["compression_method"],
            "quantization_dtype": params["quantization_dtype"],
            "explainability_enabled": params["explainability_enabled"],
            "explainability_sample_size": params["explainability_sample_size"],
            "explainability_repeats": params["explainability_repeats"],
            "explainability_top_n": params["explainability_top_n"],
            "trajectory_registered_model_name": params["trajectory_registered_model_name"],
            "on_ground_registered_model_name": params["on_ground_registered_model_name"],
            "feature_columns": ",".join(feature_columns),
            "training_sequences": training_sequences,
            "test_sequences": test_sequences,
        }
    )


def log_mlflow_metrics_and_artifacts(
    mlflow,
    metrics: dict,
    metrics_path: Path,
    artifact_paths: list[Path],
) -> None:
    trajectory = metrics["models"]["trajectory_lstm"]
    on_ground = metrics["models"]["on_ground_lstm"]
    baseline = trajectory.get("baseline", {})
    compression_summary = metrics.get("compression", {}).get("summary", {})
    logged_metrics = {
        "trajectory_mae_latitude": trajectory["mae_latitude"],
        "trajectory_mae_longitude": trajectory["mae_longitude"],
        "trajectory_rmse_latitude": trajectory["rmse_latitude"],
        "trajectory_rmse_longitude": trajectory["rmse_longitude"],
        "trajectory_mae_distance_m": trajectory["mae_distance_m"],
        "trajectory_median_distance_m": trajectory["median_distance_m"],
        "trajectory_p95_distance_m": trajectory["p95_distance_m"],
        "trajectory_rmse_distance_m": trajectory["rmse_distance_m"],
        "trajectory_baseline_mae_distance_m": baseline.get("mae_distance_m"),
        "trajectory_skill_score_vs_baseline": trajectory.get("skill_score_vs_baseline"),
        "trajectory_final_validation_loss": trajectory["final_validation_loss"],
        "on_ground_accuracy": on_ground["accuracy"],
        "on_ground_balanced_accuracy": on_ground["balanced_accuracy"],
        "on_ground_precision": on_ground["precision"],
        "on_ground_recall": on_ground["recall"],
        "on_ground_f1": on_ground["f1"],
        "on_ground_positive_rate_true": on_ground["positive_rate_true"],
        "on_ground_positive_rate_predicted": on_ground["positive_rate_predicted"],
        "on_ground_final_validation_loss": on_ground["final_validation_loss"],
        "compression_best_storage_reduction_vs_keras": compression_summary.get(
            "best_storage_reduction_vs_keras"
        ),
    }
    mlflow.log_metrics({key: value for key, value in logged_metrics.items() if value is not None})
    mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
    for path in artifact_paths:
        mlflow.log_artifact(str(path), artifact_path="models")


def _model_info_to_dict(model_info) -> dict[str, object]:
    return {
        "artifact_path": getattr(model_info, "artifact_path", None),
        "model_uri": getattr(model_info, "model_uri", None),
        "run_id": getattr(model_info, "run_id", None),
        "registered_model_version": getattr(model_info, "registered_model_version", None),
    }
